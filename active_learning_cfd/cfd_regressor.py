import matplotlib.pyplot as plt
import numpy as np
from modAL.models import ActiveLearner
from joblib import dump
from active_learning_cfd.custom_initialization import greedy_input_initialization

import os
import re
from joblib import load


def load_regression_history(directory, case_name):
    patt = case_name + "_query([0-9]*).joblib"
    res = [re.search(patt, f) for f in os.listdir(directory)]
    n_query_available = [int(f.group(1)) + 1 for f in res if f]
    regressions_available = [
        load(os.path.join(directory, f.group(0))) for f in res if f
    ]
    regression_initial = load(os.path.join(directory, case_name + "_initial.joblib"))
    n_query_available.append(0)
    regressions_available.append(regression_initial)
    regressions = dict(zip(n_query_available, regressions_available))
    features_range = load(os.path.join(directory, case_name + "_ranges.joblib"))
    return (regressions, features_range)


def load_regression_history_repetitions(directory, case_name):
    patt = case_name + "_n([0-9]*)"
    res = [re.search(patt, f) for f in os.listdir(directory)]
    repetitions_available = np.unique([f.group(0) for f in res if f])
    regressions = [
        load_regression_history(directory, case_repetition_name)
        for case_repetition_name in repetitions_available
    ]
    return regressions


def create_query_grid(features_ranges, query_minimum_spacing):
    n_features = len(features_ranges)
    query_grid_axis = [
        np.linspace(
            f_range[0],
            f_range[1],
            int(np.ceil((f_range[1] - f_range[0]) / query_minimum_spacing[i])) + 1,
        )
        for i, f_range in enumerate(features_ranges)
    ]
    full_grid = np.array(np.meshgrid(*query_grid_axis))
    query_grid = full_grid.T.reshape(-1, n_features)
    return (full_grid, query_grid)


def raw_to_normalized_features(X, features_ranges):
    X_min = features_ranges[:, 0]
    X_max = features_ranges[:, 1]
    X_range = X_max - X_min
    X_range[X_range == 0] = 1
    X_norm = (X - X_min[np.newaxis, :]) / X_range[np.newaxis, :]
    return X_norm


def normalized_to_raw_features(X_norm, features_ranges):
    X_min = features_ranges[:, 0]
    X_max = features_ranges[:, 1]
    X_range = X_max - X_min
    X = (X_norm * X_range[np.newaxis, :]) + X_min[np.newaxis, :]
    return X


def plot_regression_images(
    regressor,
    x_plot,
    y_plot,
    full_grid,
    features_ranges,
    save_path,
    save_name,
    slow_plot,
    idx=0,
    fig=None,
):
    if not fig:
        fig = plt.figure(figsize=(9, 7))
        plt.show(block=False)
    else:
        plt.figure(fig.number)
        fig.clf()

    predicted_responses = np.array(
        [
            regressor.predict(raw_to_normalized_features(row, features_ranges))
            for row in np.swapaxes(full_grid, 0, 2)
        ]
    ).T.reshape(x_plot.shape)
    plt.pcolormesh(x_plot, y_plot, predicted_responses)
    X_norm = normalized_to_raw_features(regressor.X_training, features_ranges)
    plt.plot(X_norm[:, 0], X_norm[:, 1], "ro")
    plt.colorbar()
    fig.canvas.draw()

    if save_path:
        file_spec = "_initial" if not fig else "_query{0:04d}".format(idx)
        plt.savefig(os.path.join(save_path, save_name + file_spec))

    fig.canvas.flush_events()
    if slow_plot:
        plt.pause(0.1)

    return fig


def active_learner_regressor(
    function,
    features_ranges,
    initial_regressor,
    query_minimum_spacing,
    n_queries,
    query_strategy,
    n_initial=3,
    initialization_method=greedy_input_initialization,
    plot_regression=False,
    plot_brute_force=False,
    slow_plot=False,
    regression_history_path="",
    save_path="",
    save_name="case",
    only_new=False,
):
    n_features = len(features_ranges)
    assert len(query_minimum_spacing) == n_features
    assert features_ranges.shape[1] == 2

    if save_path and not os.path.exists(save_path):
        os.makedirs(save_path)
    if regression_history_path and not os.path.exists(regression_history_path):
        os.makedirs(regression_history_path)

    # Query grid

    full_grid, query_grid = create_query_grid(features_ranges, query_minimum_spacing)

    # Initial grid

    initial_idx, initial_features = initialization_method(query_grid, n_initial)
    initial_responses = np.array([function(xi) for xi in initial_features])

    # Active learner

    initial_features_norm = raw_to_normalized_features(
        initial_features, features_ranges
    )
    regressor = ActiveLearner(
        estimator=initial_regressor,
        query_strategy=query_strategy,
        X_training=initial_features_norm,
        y_training=initial_responses,
    )

    def query_and_teach(regressor, query_grid, function, features_ranges):
        query_idx, query_instance = regressor.query(
            raw_to_normalized_features(query_grid, features_ranges)
        )
        regressor.teach(
            np.array([query_instance]),
            np.array(
                [
                    function(
                        normalized_to_raw_features(
                            query_instance, features_ranges
                        ).flatten()
                    )
                ]
            ),
            only_new=only_new,
        )

    # Run learning

    x_plot, y_plot = full_grid[0:2, :]
    if plot_brute_force:
        full_responses = np.array(
            [[function(xi) for xi in row] for row in np.swapaxes(full_grid, 0, 2)]
        ).T.reshape(x_plot.shape)
        fig1 = plt.figure(figsize=(9, 7))
        plt.pcolor(x_plot, y_plot, full_responses)
        plt.colorbar()
        if save_path:
            plt.savefig(os.path.join(save_path, save_name + "_full"))

    if plot_regression:
        fig2 = plot_regression_images(
            regressor,
            x_plot,
            y_plot,
            full_grid,
            features_ranges,
            save_path,
            save_name,
            slow_plot,
        )
    if regression_history_path:
        dump(
            regressor,
            os.path.join(regression_history_path, save_name + "_initial.joblib"),
        )
        dump(
            features_ranges,
            os.path.join(regression_history_path, save_name + "_ranges.joblib"),
        )

    for idx in range(n_queries):

        query_and_teach(regressor, query_grid, function, features_ranges)

        if plot_regression:
            plot_regression_images(
                regressor,
                x_plot,
                y_plot,
                full_grid,
                features_ranges,
                save_path,
                save_name,
                slow_plot,
                idx,
                fig2,
            )
        if regression_history_path:
            dump(
                regressor,
                os.path.join(
                    regression_history_path,
                    save_name + "_query{0:04d}.joblib".format(idx),
                ),
            )
    if not slow_plot:
        plt.close("all")

    return regressor
