import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, DotProduct
from sklearn.neural_network import MLPRegressor


def NNRegressor():
    return MLPRegressor(hidden_layer_sizes=(10,), max_iter=100000)


def SVRRBF():
    return SVR(gamma="scale")


def GaussianProcessRegressorRBF():
    return GaussianProcessRegressor(kernel=RBF(length_scale=0.1))


def GaussianProcessRegressorMatern52():
    return GaussianProcessRegressor(kernel=Matern(nu=2.5, length_scale=0.1))


def GaussianProcessRegressorCubic():
    return GaussianProcessRegressor(kernel=DotProduct() ** 3)


regressor_list = {
    "linear_regression": LinearRegression,
    "random_forest_regression": RandomForestRegressor,
    "svr": SVRRBF,
    "gaussian_process_rbf": GaussianProcessRegressorRBF,
    "gaussian_process_matern52": GaussianProcessRegressorMatern52,
    "gaussian_process_cubic": GaussianProcessRegressorCubic,
    "neural_network": NNRegressor,
}

