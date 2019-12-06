# Loss models for the NeuralNetworkLib
import numpy as np


class SumSquaresLoss:
    """
    Class that holds the loss function and gradient for the Sum of squared errors loss.

    """
    def loss_function(self, x, y):
        return (x - y) * (x - y)

    def loss_gradient(self, x, y):
        return 2 * (x - y)


class LogLoss:
    """
    Class that holds the loss function and gradient for the Sum of squared errors loss.

    """
    def loss_function(self, x, y):
        return -(y * np.log(x) + (1 - y) * np.log((1 - x)))

    def loss_gradient(self, x, y):
        return - ((y / max(x, 1e-8)) + ((1 - y) / max(1 - x, 1e-8)))


class CrossEntropyLoss:
    """
    Class that holds the loss function and gradient for the Sum of squared errors loss.

    """
    def loss_function(self, x, y):
        return -np.dot(y, np.log(x).T).squeeze()

    def loss_gradient(self, x, y):
        return -y / np.maximum(x, 1e-8)
