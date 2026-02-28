import numpy as np
from src.ml.gradient_descent import GradientDescent


class LogisticRegression:
    def __init__(self, learning_rate=0.01, iterations=1000):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.theta = None
        self.cost_history = []

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        m, n = X.shape

        # Add bias term
        X = np.c_[np.ones((m, 1)), X]
        self.theta = np.zeros(n + 1)

        def compute_cost(theta):
            z = X @ theta
            h = self.sigmoid(z)

            # Avoid log(0)
            epsilon = 1e-8
            cost = -(1/m) * np.sum(
                y * np.log(h + epsilon) +
                (1 - y) * np.log(1 - h + epsilon)
            )
            return cost

        def compute_gradient(theta):
            z = X @ theta
            h = self.sigmoid(z)
            return (1/m) * (X.T @ (h - y))

        gd = GradientDescent(self.learning_rate, self.iterations)
        self.theta = gd.optimize(self.theta, compute_cost, compute_gradient)
        self.cost_history = gd.cost_history

    def predict_proba(self, X):
        m = X.shape[0]
        X = np.c_[np.ones((m, 1)), X]
        return self.sigmoid(X @ self.theta)

    def predict(self, X):
        probabilities = self.predict_proba(X)
        return (probabilities >= 0.5).astype(int)

    def accuracy(self, X, y):
        predictions = self.predict(X)
        return np.mean(predictions == y)