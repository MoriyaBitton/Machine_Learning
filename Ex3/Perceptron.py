import numpy as np

class Perceptron:

    def __init__(self, epoch=100):

        self.epoch = epoch
        self.weights = None
        self.score = 100

    def activation(self, res):
        return -1.0 if (res < 0) else 1.0

    def fit(self, X, y):
        """
        Fit training data

        X : Training vectors
            X.shape: [#samples, #features]
        y : Target values
            y.shape: [#samples]
        """

        samples, n = X.shape

        # Inserting bias
        X_bias = np.ones((samples, n + 1))
        X_bias[:, 1:] = X
        X = X_bias

        # Initializing (n: features) + (1: bias) n to zeros
        self.weights = np.zeros(n + 1)

        # Training
        for t in range(self.epoch):
            self.error = 0

            for i, x_i in enumerate(X):

                pred = self.activation(x_i @ self.weights)

                # Updating weights
                if pred != y[i]:
                    self.error += 1
                    self.weights += ((y[i] - pred) / 2) * x_i

            self.score = (1 - self.error/samples) * 100

            if self.error == 0:
                self.epoch = t
                break

        return self.weights
