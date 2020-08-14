import numpy as np
import matplotlib.pyplot as plt
import scipy as sp

from active_learning_cfd.cfd_regressor import (
    load_regression_history,
    create_query_grid,
    raw_to_normalized_features,
    normalized_to_raw_features,
)
from active_learning_cfd.error_measures import calculate_error, mean_relative_error

plt.style.use("tableau-colorblind10")
colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

strategy_list = (
    ["gp_52_greedyio", "GP (Matern 5/2) + Greedy I/O"],
    ["gp_rbf_greedyio", "GP (RBF) + Greedy I/O"],
    ["gp_cubic_greedyio", "GP (Cubic) + Greedy I/O"],
    ["lin_greedyio", "Linear + Greedy I/O"],
    ["rfr_greedyio", "Random forest + Greedy I/O"],
    ["svr_greedyio", "Support vector + Greedy I/O"],
    ["nn_greedyio", "Multilayer perceptron + Greedy I/O"],
    ["gp_52_std", "GP (Matern 5/2) + Variational"],
    ["gp_52_greedyi", "GP (Matern 5/2) + Greedy I"],
    ["gp_52_greedyo", "GP (Matern 5/2) + Greedy O"],
    ["gp_52_greedyio", "GP (Matern 5/2) + Greedy I/O"],
    ["gp_52_rdm", "GP (Matern 5/2) + Random"],
)

plot_sample_order = True
xlabel = "$L/D$"
ylabel = "$\\theta$"

for name, label in strategy_list:
    plt.figure(figsize=(3.5, 3))
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    regressions, features_ranges = load_regression_history(
        "regression_" + name, "static_mixer_n000"
    )

    regressor = regressions[20]

    query_minimum_spacing = (features_ranges[:, 1] - features_ranges[:, 0]) / 100.0
    full_grid, query_grid = create_query_grid(features_ranges, query_minimum_spacing)

    x_plot, y_plot = full_grid[0:2, :]
    predicted_responses = np.array(
        [
            regressor.predict(raw_to_normalized_features(row, features_ranges))
            for row in np.swapaxes(full_grid, 0, 2)
        ]
    ).T.reshape(x_plot.shape)
    mpb = plt.pcolormesh(x_plot, y_plot, predicted_responses, vmin=-11, vmax=-3)
    X_norm = normalized_to_raw_features(regressor.X_training, features_ranges)
    plt.plot(X_norm[:, 0], X_norm[:, 1], ".", color=colors[1], markersize=8)

    if plot_sample_order:
        n_initial = 8
        for i, row in enumerate(X_norm):
            if i >= n_initial:
                x, y = row
                ha = "left"
                va = "bottom"
                x_range = features_ranges[0, 1] - features_ranges[0, 0]
                y_range = features_ranges[1, 1] - features_ranges[1, 0]
                dx = x_range / 100
                dy = y_range / 100
                x += 1 * dx
                if x >= features_ranges[0, 0] + 0.9 * x_range:
                    x -= 2 * dx
                    ha = "right"
                if y >= features_ranges[1, 0] + 0.9 * y_range:
                    y -= 2 * dy
                    va = "top"
                plt.text(
                    x,
                    y,
                    i - n_initial + 1,
                    fontsize=8,
                    horizontalalignment=ha,
                    verticalalignment=va,
                )

    plt.tight_layout()
    plt.savefig("static_mixer_example_" + name + "_numbered.png")
    plt.close()

fig = plt.figure(figsize=(3.5, 1.0))
cbar = plt.colorbar(mpb, orientation="horizontal")
cbar.set_label("$\log{c_v}$")
cbar.ax.xaxis.set_ticks_position("top")
cbar.ax.xaxis.set_label_position("top")
plt.gca().remove()
plt.tight_layout()
plt.savefig("static_mixer_colourbar.png")
plt.close()
