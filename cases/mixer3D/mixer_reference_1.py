import numpy as np

from active_learning_cfd.cfd_regressor import create_query_grid
from active_learning_cfd.custom_initialization import random_initialization
from active_learning_cfd.error_measures import generate_reference_solution

from mixer_runner import Mixer

from argparse import ArgumentParser

parser = ArgumentParser(description="Process some integers.")
parser.add_argument(
    "np",
    metavar="N",
    type=int,
    nargs="?",
    default="1",
    help="number of processors (-1 to use original decomposeParDict)",
)
parser.add_argument("--hpc", dest="hpc", action="store_true")
parser.add_argument(
    "-s",
    dest="s",
    type=int,
    nargs="?",
    default="-1",
    help="strategy to run (-1 to run all)",
)
parser.set_defaults(hpc=False)

args = parser.parse_args()

function = Mixer(np=args.np, hpc=args.hpc)

reference_filename = "reference_solution_1.csv"

features_ranges = np.array(
    [np.log([0.1, 10]), [0.05, 0.4], [0.5, 1.5], [2, 4], [0.5, 7.0], [0.01, 0.05]]
)  # List of maximum and minimum values for each parameter
query_minimum_spacing = np.array(
    [0.1, 0.01, 0.05, 2, 0.1, 0.01]
)  # List of minimum spacing for each parameter

full_grid, query_grid = create_query_grid(features_ranges, query_minimum_spacing)
np.random.seed(1)
X_true, y_true = generate_reference_solution(
    function, random_initialization, query_grid, 33
)

full_matrix = np.hstack((X_true, np.array(y_true).reshape(-1, 1)))
np.savetxt(reference_filename, full_matrix, delimiter=",")
