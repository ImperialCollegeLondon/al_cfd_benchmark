import numpy as np

from active_learning_cfd.cfd_regressor import active_learner_regressor
from active_learning_cfd.custom_sampling import probabilistic_std_sampling
from active_learning_cfd.basic_regressors import regressor_list

from pitz_daily_runner import PitzDaily

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
parser.set_defaults(hpc=False)

args = parser.parse_args()

function = PitzDaily(np=args.np, hpc=args.hpc)

features_ranges = np.array(
    [[4, 6], [20.0, 30.0]]
)  # List of maximum and minimum values for each parameter
query_minimum_spacing = np.array(
    [0.01, 0.01]
)  # List of minimum spacing for each parameter
n_initial = 9  # Number of initial samples
n_queries = 30  # Number of queries
plot_regression = True

regressor_constructor = regressor_list["gaussian_process_rbf"]
regressor = regressor_constructor()

query_strategy = probabilistic_std_sampling

active_learner_regressor(
    function,
    features_ranges,
    regressor,
    query_minimum_spacing,
    n_queries,
    query_strategy,
    n_initial,
    plot_regression=plot_regression,
    save_path="figs",
    save_name="pitzdaily",
)
