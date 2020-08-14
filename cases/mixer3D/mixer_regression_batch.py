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

features_ranges = np.array(
    [np.log([0.1, 10]), [0.05, 0.4], [0.5, 1.5], [2, 4], [0.5, 7.0], [0.01, 0.05]]
)  # List of maximum and minimum values for each parameter
query_minimum_spacing = np.array(
    [0.1, 0.01, 0.05, 2, 0.1, 0.01]
)  # List of minimum spacing for each parameter
n_initial = 6  # Number of initial samples
plot_regression = False
plot_brute_force = False

strategy_list = (
    ("gp_rbf_std", "gaussian_process_rbf", probabilistic_std_sampling, 1),
    ("gp_52_std", "gaussian_process_matern52", probabilistic_std_sampling, 1),
    ("gp_rbf_greedyio", "gaussian_process_rbf", greedy_sampling_input_output, 1),
    ("gp_cubic_greedyio", "gaussian_process_cubic", greedy_sampling_input_output, 1),
    ("lin_greedyio", "linear_regression", greedy_sampling_input_output, 1),
    ("svr_greedyio", "svr", greedy_sampling_input_output, 1),
    ("gp_52_greedyi", "gaussian_process_matern52", greedy_sampling_input, 1),
    ("gp_52_greedyo", "gaussian_process_matern52", greedy_sampling_output, 1),
    ("gp_52_greedyio", "gaussian_process_matern52", greedy_sampling_input_output, 1),
    ("rfr_greedyio", "random_forest_regression", greedy_sampling_input_output, 1),
    ("nn_greedyio", "neural_network", greedy_sampling_input_output, 1),
    ("gp_52_rdm", "gaussian_process_matern52", random_sampling, 1),
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
            save_name="mixer" + "_n{0:03d}".format(n),
        )
