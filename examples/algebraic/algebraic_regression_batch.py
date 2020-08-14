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

from algebraic_runner import TwoParameterFunction

function = TwoParameterFunction()

features_ranges = np.array(
    [[0, 2], [0, 1]]
)  # List of maximum and minimum values for each parameter
query_minimum_spacing = np.array(
    [0.01, 0.01]
)  # List of minimum spacing for each parameter
n_initial = 9  # Number of initial samples
n_queries = 30  # Number of queries
plot_regression = True
plot_brute_force = True

strategy_list = (
    ("gp_rbf_std", "gaussian_process_rbf", probabilistic_std_sampling),
    ("gp_52_std", "gaussian_process_matern52", probabilistic_std_sampling),
    ("lin_greedyio", "linear_regression", greedy_sampling_input_output),
    ("rfr_greedyio", "random_forest_regression", greedy_sampling_input_output),
    ("svr_greedyio", "svr", greedy_sampling_input_output),
    ("nn_greedyio", "neural_network", greedy_sampling_input_output),
)

for name, regressor_name, query_strategy in strategy_list:
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
        save_name="algebraic",
    )
