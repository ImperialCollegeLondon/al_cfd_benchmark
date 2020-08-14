import numpy as np
import matplotlib.pyplot as plt
import scipy as sp

from active_learning_cfd.cfd_regressor import load_regression_history_repetitions
from active_learning_cfd.error_measures import calculate_error, mean_relative_error

from matplotlib import rc
from cycler import cycler

rc("text", usetex=True)
plt.style.use("tableau-colorblind10")
colors = plt.rcParams["axes.prop_cycle"]
linestyle_cycler = colors + cycler(
    "linestyle", ["--", "-", ":", "-.", "-", "-", "-", "-", "-", "-"]
)
rc("axes", prop_cycle=linestyle_cycler)

case_name = "static_mixer"

reference_filename = "reference_solution.csv"
figsize = [6.4, 2.5]

reference_solution = np.genfromtxt(reference_filename, delimiter=",")
X_true = reference_solution[:, 0:-1]
y_true = reference_solution[:, -1]

strategy_list = (
    ["gp_52_greedyio", "GP (Matern 5/2)"],
    ["gp_rbf_greedyio", "GP (RBF)"],
    ["gp_cubic_greedyio", "GP (Cubic)"],
    ["lin_greedyio", "Linear"],
    ["rfr_greedyio", "Random forest"],
    ["svr_greedyio", "Support vector"],
    ["nn_greedyio", "Multilayer perceptron"],
)

plt.figure(figsize=figsize)
plt.xlabel("Samples")
plt.ylabel("$\epsilon$ [\%]")

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

    plt.plot(error_samples, error_avgs * 100, label=label)
    plt.fill_between(error_samples, error_min * 100, error_max * 100, alpha=0.5)

plt.xlim([0, 60])
plt.text(
    0.9,
    0.9,
    "Case 1",
    horizontalalignment="left",
    verticalalignment="top",
    transform=plt.gca().transAxes,
)
plt.tight_layout()
plt.savefig(case_name + "_regressions.png", dpi=400)

handles, labels = plt.gca().get_legend_handles_labels()
fig_legend = plt.figure(figsize=[5.0, 0.8])
axi = fig_legend.add_subplot(111)
fig_legend.legend(handles, labels, loc="center", ncol=3)
axi.xaxis.set_visible(False)
axi.yaxis.set_visible(False)
axi.axis("off")

fig_legend.tight_layout()
fig_legend.savefig("regressions_legend.png", dpi=400)

sampling_list = (
    ["gp_52_std", "Variational"],
    ["gp_52_greedyi", "Greedy I"],
    ["gp_52_greedyo", "Greedy O"],
    ["gp_52_greedyio", "Greedy I/O"],
    ["gp_52_rdm", "Random"],
)

plt.figure(figsize=figsize)
plt.xlabel("Samples")
plt.ylabel("$\epsilon$ [\%]")

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

    plt.plot(error_samples, error_avgs * 100, label=label)
    plt.fill_between(error_samples, error_min * 100, error_max * 100, alpha=0.5)

plt.xlim([0, 60])
plt.text(
    0.9,
    0.9,
    "Case 1",
    horizontalalignment="left",
    verticalalignment="top",
    transform=plt.gca().transAxes,
)
plt.tight_layout()
plt.savefig(case_name + "_sampling.png", dpi=400)

handles, labels = plt.gca().get_legend_handles_labels()
fig_legend = plt.figure(figsize=[5.0, 0.8])
axi = fig_legend.add_subplot(111)
fig_legend.legend(handles, labels, loc="center", ncol=3)
axi.xaxis.set_visible(False)
axi.yaxis.set_visible(False)
axi.axis("off")

fig_legend.tight_layout()
fig_legend.savefig("sampling_legend.png", dpi=400)

plt.show()
