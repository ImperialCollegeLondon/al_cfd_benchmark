import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from active_learning_cfd.cfd_regressor import raw_to_normalized_features


def root_mean_squared_error(y_true, y_pred):
    output_errors = np.sqrt(mean_squared_error(y_true, y_pred))

    return output_errors


def mean_relative_error(y_true, y_pred):
    output_errors = np.average(np.abs(y_pred - y_true) / np.abs(y_true), axis=0)

    return output_errors


def max_relative_error(y_true, y_pred):
    output_errors = np.maximum(np.abs(y_pred - y_true) / np.abs(y_true), axis=0)

    return output_errors


def generate_reference_solution(function, method, X, n_samples):
    query_idx, query_list = method(X, n_samples)
    y_true = [function(query_instance) for query_instance in query_list]
    return (query_list, y_true)


def calculate_error(
    X_true, y_true, regressor, features_range, error_measure=mean_absolute_error
):
    y = regressor.predict(raw_to_normalized_features(X_true, features_range)).flatten()
    return error_measure(y_true, y)

