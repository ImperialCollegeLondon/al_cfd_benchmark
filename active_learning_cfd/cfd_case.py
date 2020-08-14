from PyFoam.Execution.ConvergenceRunner import ConvergenceRunner
from PyFoam.Execution.UtilityRunner import UtilityRunner
from PyFoam.Execution.BasicRunner import BasicRunner
from PyFoam.LogAnalysis.BoundingLogAnalyzer import BoundingLogAnalyzer
from PyFoam.LogAnalysis.LogAnalyzerApplication import LogAnalyzerApplication
from PyFoam.LogAnalysis.SimpleLineAnalyzer import SimpleLineAnalyzer
from PyFoam.LogAnalysis.UtilityAnalyzer import UtilityAnalyzer
from PyFoam.LogAnalysis.RegExpLineAnalyzer import RegExpLineAnalyzer
from PyFoam.RunDictionary.SolutionFile import SolutionFile
from PyFoam.RunDictionary.SolutionDirectory import SolutionDirectory
from PyFoam.RunDictionary.ParsedParameterFile import ParsedParameterFile
from PyFoam.RunDictionary.TimeDirectory import TimeDirectory
from uuid import uuid4

import os


class CFDCaseAnalyzer(UtilityAnalyzer):
    def __init__(self, output_list, progress=False):
        UtilityAnalyzer.__init__(self, progress=progress)
        for output_name, output_regex in output_list:
            self.addExpression(output_name, output_regex)

    def addExpression(self, name, expr, idNr=None):
        self.addAnalyzer(
            name, RegExpLineAnalyzer(name, expr, idNr, doTimelines=True, doFiles=False)
        )


class CFDCase:
    mesher = "blockMesh"
    solver = "icoFoam"
    template = "case"
    clone_files = ("*.orig", "parameters", "*.py", "initial")
    output_list = ()

    def __init__(self, parallel=None, np=1, hpc=False):
        self.np = np
        self.hpc = hpc

    def create(self, caseName, input_parameters):
        direTemplate = SolutionDirectory(self.template)
        for f in self.clone_files:
            direTemplate.addToClone(f)

        dire = direTemplate.cloneCase(caseName)

        parameters = ParsedParameterFile(os.path.join(dire.name, "parameters"))
        for key, value in input_parameters.items():
            parameters[key] = value
        parameters.writeFile()

        if self.np > 1:
            decomposeParDict = ParsedParameterFile(
                os.path.join(dire.systemDir(), "decomposeParDict")
            )
            decomposeParDict["numberOfSubdomains"] = self.np
            decomposeParDict.writeFile()

        self.dire = dire

    def generate_name(self, input_parameters):
        base_name = ""
        for param in self.parameter_names:
            base_name += "{}_{}_".format(param, input_parameters[param])
        base_name += str(uuid4())[:8]
        return base_name

    def prepare(self):
        if isinstance(self.mesher, str):
            mesher = UtilityRunner(
                argv=[self.mesher, "-case", self.dire.name],
                silent=True,
                logname=self.mesher,
            )
            mesher.start()
            if not mesher.data["OK"]:
                raise RuntimeError("Failed running mesher")
        else:
            for mesher_step in self.mesher:
                mesher = UtilityRunner(
                    argv=[mesher_step, "-case", self.dire.name],
                    silent=True,
                    logname=mesher_step,
                )
                mesher.start()
            if not mesher.data["OK"]:
                raise RuntimeError(f"Failed running mesher {mesher_step}")
        if self.np != 1:
            decomposer = UtilityRunner(
                argv=["decomposePar", "-case", self.dire.name],
                silent=True,
                logname="decomposePar",
            )
            decomposer.start()
            if not decomposer.data["OK"]:
                raise RuntimeError("Failed decomposing case")

    def run_solver(self):
        run_command = self.solver
        if self.np != 1:
            np_substring = "" if self.hpc else f"-np {self.np} "
            run_command = f"mpiexec {np_substring}{self.solver} -parallel"
        runner = ConvergenceRunner(
            BoundingLogAnalyzer(),
            argv=[run_command, "-case", self.dire.name],
            silent=True,
            logname=self.solver,
        )
        runner.start()
        if not runner.data["OK"]:
            raise RuntimeError("Failed running solver")

    def postprocess(self):
        analyzer = CFDCaseAnalyzer(self.output_list)
        fh = open(os.path.join(self.dire.name, self.solver + ".logfile"), "r")
        analyzer.analyze(fh)
        self.results = {}
        for output_name, output_regex in self.output_list:
            data = analyzer.getAnalyzer(output_name).lines.getLatestData()
            if "value 0" not in data:
                raise RuntimeError(f"Failed retrieving results for {output_name}")
            self.results[output_name] = data["value 0"]

    def solve(self, input_parameters, workDir="tmp"):
        if not os.path.exists(workDir):
            os.makedirs(workDir)

        case_name = self.generate_name(input_parameters)
        try:
            # Setup
            self.create(os.path.join(workDir, case_name), input_parameters)
            self.prepare()
            # Run
            self.run_solver()
            # Postprocess
            self.postprocess()
        except RuntimeError as e:
            raise RuntimeError(
                f"Error running case with parameters {input_parameters}"
            ) from e

