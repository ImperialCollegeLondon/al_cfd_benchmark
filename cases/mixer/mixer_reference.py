import numpy as np

from active_learning_cfd.cfd_regressor import create_query_grid
from active_learning_cfd.custom_initialization import random_initialization
from active_learning_cfd.error_measures import generate_reference_solution

from mixer_runner import Mixer

function = Mixer()

reference_filename = "reference_solution.csv"

features_ranges = np.array(
    [np.log([0.1, 100]), [0.05, 0.4], [0.2, 1.5], [2, 6]]
)  # List of maximum and minimum values for each parameter
query_minimum_spacing = np.array(
    [0.1, 0.02, 0.05, 2]
)  # List of minimum spacing for each parameter

full_grid, query_grid = create_query_grid(features_ranges, query_minimum_spacing)
X_true, y_true = generate_reference_solution(
    function, random_initialization, query_grid, 100
)

full_matrix = np.hstack((X_true, np.array(y_true).reshape(-1, 1)))
np.savetxt(reference_filename, full_matrix, delimiter=",")
