import numpy as np
from sklearn.metrics import pairwise_distances


def probabilistic_std_sampling(regressor, X):
    _, std = regressor.predict(X, return_std=True)
    query_idx = np.argmax(std)
    return query_idx, X[query_idx]


def random_sampling(regressor, X):
    query_idx = np.random.choice(np.arange(len(X)), 1, replace=False)[0]
    return query_idx, X[query_idx]


def greedy_sampling_input(regressor, X):
    dist_matrix = pairwise_distances(regressor.X_training, X)
    dist_to_training_set = np.amin(dist_matrix, axis=0)
    query_idx = np.argmax(dist_to_training_set)
    return query_idx, X[query_idx]


def greedy_sampling_output(regressor, X):
    y = regressor.predict(X)
    dist_matrix = pairwise_distances(
	regressor.y_training.reshape(-1, 1), y.reshape(-1, 1)
    )
    dist_to_training_set = np.amin(dist_matrix, axis=0)
    query_idx = np.argmax(dist_to_training_set)
    return query_idx, X[query_idx]


def greedy_sampling_input_output(regressor, X):
    y = regressor.predict(X)
    dist_x_matrix = pairwise_distances(regressor.X_training, X)
    dist_y_matrix = pairwise_distances(
        regressor.y_training.reshape(-1, 1), y.reshape(-1, 1)
    )
    dist_to_training_set = np.amin(dist_x_matrix * dist_y_matrix, axis=0)
    query_idx = np.argmax(dist_to_training_set)
    return query_idx, X[query_idx]
