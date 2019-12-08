# Loss models for the NeuralNetworkLib
import numpy as np


class SumSquaresLoss:
    """
    Class that holds the loss function and gradient for the Sum of squared errors loss.

    """
    @staticmethod
    def loss_function(y_hat, y):
        return ((y_hat - y) * (y_hat - y)).mean()

    @staticmethod
    def loss_gradient(y_hat, y):
        return 2 * (y_hat - y)

    @staticmethod
    def grad_accumulator(grads):
        return np.mean(grads, axis=0)


class LogLoss:
    """
    Class that holds the loss function and gradient for the log loss.

    """
    @staticmethod
    def loss_function(x, y):
        return -(y * np.log(x) + (1 - y) * np.log((1 - x)))

    @staticmethod
    def loss_gradient(x, y):
        return - ((y / max(x, 1e-8)) + ((1 - y) / max(1 - x, 1e-8)))


class CrossEntropyLoss:
    """
    Class that holds the loss function and gradient for the cross entropy loss.

    """
    @staticmethod
    def loss_function(y_hat, y, clip_epsilon=1e-15):
        y_hat = np.clip(y_hat, clip_epsilon, 1-clip_epsilon)
        y_hat = y_hat.reshape(y_hat.shape[0], y_hat.shape[1], 1)
        y = y.reshape(y.shape[0], 1, y.shape[1])
        return -np.matmul(y, np.log(y_hat)).mean()

    @staticmethod
    def loss_gradient(y_hat, y, clip_epsilon=1e-15):
        y_hat = np.clip(y_hat, clip_epsilon, 1-clip_epsilon)
        y_hat = y_hat.reshape(y_hat.shape[0], 1, y_hat.shape[1])
        y = y.reshape(y.shape[0], 1, y.shape[1])
        return -y / y_hat

    @staticmethod
    def grad_accumulator(grads):
        return np.mean(grads, axis=0)
