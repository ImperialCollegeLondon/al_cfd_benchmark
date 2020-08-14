import numpy as np
import matplotlib.pyplot as plt

from active_learning_cfd.cfd_regressor import load_regression_history
from active_learning_cfd.error_measures import calculate_error

reference_filename = "reference_solution.csv"

reference_solution = np.genfromtxt(reference_filename, delimiter=",")
X_true = reference_solution[:, 0:-1]
y_true = reference_solution[:, -1]

strategy_list = (
    "gp_rbf_std",
    "gp_52_std",
    "lin_greedyio",
    "rfr_greedyio",
    "svr_greedyio",
    "nn_greedyio",
)

plt.figure()
for name in strategy_list:
    regression_history = load_regression_history("regression_" + name, "algebraic")

    error_list = np.array(
        [
            [n_samples, calculate_error(X_true, y_true, regressor)]
            for n_samples, regressor in regression_history
        ]
    )

    plt.plot(error_list[:, 0], error_list[:, 1], label=name)

plt.legend()
plt.show()
