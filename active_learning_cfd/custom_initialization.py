import numpy as np
from sklearn.metrics import pairwise_distances


def random_initialization(X, n_samples):
    query_idx = np.random.choice(np.arange(len(X)), size=n_samples, replace=False)
    return query_idx, X[query_idx]


def greedy_input_initialization(X, n_samples):
    query_idxs = []
    samples = []
    centroid = np.average(X, axis=0)
    dist_matrix = pairwise_distances([centroid], X)
    query_idx = np.argmin(dist_matrix, axis=1)[0]
    query_idxs.append(query_idx)
    samples.append(X[query_idx])
    for i in range(1, n_samples):
        dist_matrix = pairwise_distances(samples, X)
        dist_to_training_set = np.amin(dist_matrix, axis=0)
        query_idx = np.argmax(dist_to_training_set)
        query_idxs.append(query_idx)
        samples.append(X[query_idx])
    return query_idxs, np.array(samples)


def grid_initialization(X, n_samples):
    n_features = X.shape[1]
    n_samples_per_feat = np.power(n_samples, 1.0 / n_features)
    assert int(n_samples_per_feat) - n_samples_per_feat == 0.0
    feat_min = np.amin(X, axis=0)
    feat_max = np.amax(X, axis=0)
    grid_axis = np.linspace(feat_min, feat_max, num=int(n_samples_per_feat), axis=1)
    full_grid = np.array(np.meshgrid(*grid_axis))
    grid_linear = full_grid.T.reshape(-1, n_features)
    dist_matrix = pairwise_distances(grid_linear, X)
    query_idx = np.argmin(dist_matrix, axis=1)
    return query_idx, X[query_idx]
