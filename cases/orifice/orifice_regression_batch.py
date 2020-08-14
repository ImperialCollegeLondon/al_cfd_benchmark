import numpy as np
from active_learning_cfd.cfd_regressor import active_learner_regressor
from active_learning_cfd.custom_sampling import (
    probabilistic_std_sampling,
    random_sampling,
    greedy_sampling_input,
    greedy_sampling_output,
    greedy_sampling_input_output,
)
from active_learning_cfd.basic_regressors import regressor_list

from orifice_runner import Orifice

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

function = Orifice(np=args.np, hpc=args.hpc)

features_ranges = np.array(
    [[0.3, 0.7], [0.2, 3.0]]
)  # List of maximum and minimum values for each parameter
query_minimum_spacing = np.array(
    [0.01, 0.01]
)  # List of minimum spacing for each parameter
n_initial = 2 * len(features_ranges)  # Number of initial samples
n_queries = 60  # Number of queries
plot_regression = True
plot_brute_force = False

strategy_list = (
    ("nn_greedyio", "neural_network", greedy_sampling_input_output, 10),
)

if args.s >= 0:
    if args.s < len(strategy_list):
        strategy_list = [strategy_list[args.s]]
    else:
        raise ValueError("Invalid strategy index.")

for name, regressor_name, query_strategy, repetitions in strategy_list:
    for n in range(repetitions):
        np.random.seed(n)
        regressor_constructor = regressor_list[regressor_name]
        regressor = regressor_constructor()

        active_learner_regressor(
            function,
            features_ranges,
            regressor,
            query_minimum_spacing,
            n_queries,
            query_strategy,
            n_initial,
            plot_regression=plot_regression,
            plot_brute_force=plot_brute_force,
            save_path="figs_" + name,
            regression_history_path="regression_" + name,
            save_name="orifice" + "_n{0:03d}".format(n),
        )
