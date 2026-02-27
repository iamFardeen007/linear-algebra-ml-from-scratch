import numpy as np
from src.ml.gradient_descent import GradientDescent


class LinearRegression:
    def __init__(self, learning_rate=0.01, iterations=1000):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.theta = None
        self.cost_history = []

    def fit(self, X, y):
        m, n = X.shape

        # Add bias term
        X = np.c_[np.ones((m, 1)), X]

        self.theta = np.zeros(n + 1)

        def compute_cost(theta):
            predictions = X @ theta
            errors = predictions - y
            return (1 / (2 * m)) * np.sum(errors ** 2)

        def compute_gradient(theta):
            predictions = X @ theta
            errors = predictions - y
            return (1 / m) * (X.T @ errors)

        gd = GradientDescent(self.learning_rate, self.iterations)
        self.theta = gd.optimize(self.theta, compute_cost, compute_gradient)
        self.cost_history = gd.cost_history

    def predict(self, X):
        m = X.shape[0]
        X = np.c_[np.ones((m, 1)), X]
        return X @ self.theta