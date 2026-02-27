class GradientDescent:
    def __init__(self, learning_rate=0.01, iterations=1000):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.cost_history = []

    def optimize(self, initial_theta, compute_cost, compute_gradient):
        """
        initial_theta: initial parameters
        compute_cost: function(theta) -> cost
        compute_gradient: function(theta) -> gradient
        """
        theta = initial_theta

        for _ in range(self.iterations):
            gradient = compute_gradient(theta)
            theta = theta - self.learning_rate * gradient
            cost = compute_cost(theta)
            self.cost_history.append(cost)

        return theta