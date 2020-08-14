import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import pandas as pd

from active_learning_cfd.cfd_regressor import load_regression_history_repetitions
from active_learning_cfd.error_measures import calculate_error, mean_relative_error

case_name = "orifice"

reference_filename = "reference_solution.csv"

reference_solution = np.genfromtxt(reference_filename, delimiter=",")
X_true = reference_solution[:, 0:-1]
y_true = reference_solution[:, -1]

table_extract = [5, 10, 20]

strategy_list = (
    ["gp_52_greedyio", "GP (Matern 5/2)"],
    ["gp_rbf_greedyio", "GP (RBF)"],
    ["gp_cubic_greedyio", "GP (Cubic)"],
    ["lin_greedyio", "Linear"],
    ["rfr_greedyio", "Random forest"],
    ["svr_greedyio", "Support vector"],
    ["nn_greedyio", "Multilayer perceptron"],
)

columns = ["Method"] + table_extract

regression_results = []
for name, label in strategy_list:
    regression_history_all = load_regression_history_repetitions(
        "regression_" + name, case_name
    )

    n_samples_list = regression_history_all[0][0].keys()
    error_samples = sorted(n_samples_list)
    error_avgs = np.zeros(len(n_samples_list))
    error_min = np.zeros(len(n_samples_list))
    error_max = np.zeros(len(n_samples_list))
    error_repetitions = np.zeros(len(n_samples_list))
    for i, n_samples in enumerate(error_samples):
        error_list = []
        for regression_history, features_range in regression_history_all:
            error = calculate_error(
                X_true,
                y_true,
                regression_history[n_samples],
                features_range,
                error_measure=mean_relative_error,
            )
            error_list.append(error)
        error_avgs[i] = np.average(error_list)
        error_min[i], error_max[i] = sp.stats.t.interval(
            0.95,
            len(error_list) - 1,
            loc=np.mean(error_list),
            scale=sp.stats.sem(error_list),
        )
        error_repetitions[i] = len(error_list)

    aux = [label] + [f"{error_avgs[i] * 100:#.2g}" for i in table_extract]
    regression_results.append(aux)

df = pd.DataFrame(regression_results, columns=columns)
df.to_csv("regression_results.csv", index=False)

sampling_list = (
    ["gp_52_std", "Variational"],
    ["gp_52_greedyi", "Greedy I"],
    ["gp_52_greedyo", "Greedy O"],
    ["gp_52_greedyio", "Greedy I/O"],
    ["gp_52_rdm", "Random"],
)

sampling_results = []
for name, label in sampling_list:
    regression_history_all = load_regression_history_repetitions(
        "regression_" + name, case_name
    )

    n_samples_list = regression_history_all[0][0].keys()
    error_samples = sorted(n_samples_list)
    error_avgs = np.zeros(len(n_samples_list))
    error_min = np.zeros(len(n_samples_list))
    error_max = np.zeros(len(n_samples_list))
    error_repetitions = np.zeros(len(n_samples_list))
    for i, n_samples in enumerate(error_samples):
        error_list = []
        for regression_history, features_range in regression_history_all:
            error = calculate_error(
                X_true,
                y_true,
                regression_history[n_samples],
                features_range,
                error_measure=mean_relative_error,
            )
            error_list.append(error)
        error_avgs[i] = np.average(error_list)
        error_min[i], error_max[i] = sp.stats.t.interval(
            0.95,
            len(error_list) - 1,
            loc=np.mean(error_list),
            scale=sp.stats.sem(error_list),
        )
        error_repetitions[i] = len(error_list)

    aux = [label] + [f"{error_avgs[i] * 100:#.2g}" for i in table_extract]
    sampling_results.append(aux)
    aux = [label + " - Range"] + [
        f"[{error_min[i] * 100:#.2g}-{error_max[i] * 100:#.2g}]" for i in table_extract
    ]
    sampling_results.append(aux)

df = pd.DataFrame(sampling_results, columns=columns)
df.to_csv("sampling_results.csv", index=False)
