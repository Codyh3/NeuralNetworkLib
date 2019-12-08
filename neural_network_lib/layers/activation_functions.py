import numpy as np


def outer_product(x):
    x = x.reshape(1, -1)
    return np.dot(x.T, x)


def outer_product_vect(x):
    return np.apply_along_axis(outer_product, -1, x)


def identity(x):
    return x


def pre_d_identity(x):
    return np.identity(x.shape[0])


def d_identity(x):
    return np.apply_along_axis(pre_d_identity, -1, x)


def ReLu(x):
    return np.maximum(0, x)


def pre_d_ReLu(x):
    return np.diag((1 * (x > 0)).reshape((-1,)))


def d_ReLu(x):
    return np.apply_along_axis(pre_d_ReLu, -1, x)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def pre_d_sigmoid(x):
    return np.diag((sigmoid(x) * (1 - sigmoid(x))).reshape((-1,)))


def d_sigmoid(x):
    return np.apply_along_axis(pre_d_sigmoid, -1, x)


def tanh(x):
    return np.tanh(x)


def pre_d_tanh(x):
    return np.diag((1 - tanh(x) * tanh(x)).reshape((-1,)))


def d_tanh(x):
    return np.apply_along_axis(pre_d_tanh, -1, x)


def softmax(x):
    exp_x = np.exp(x - x.max(axis=1, keepdims=True))
    return exp_x / exp_x.sum(axis=1, keepdims=True)


def pre_d_softmax(x):
    vector = np.exp(x - x.max()) / np.exp(x - x.max()).sum()
    return np.diag(vector) - outer_product(vector)


def d_softmax(x):
    return np.apply_along_axis(pre_d_softmax, -1, x)


def softplus(x):
    return np.log(1 + np.exp(x))


def pre_d_softplus(x):
    return np.diag(sigmoid(x).reshape(-1,))


def d_softplus(x):
    return np.apply_along_axis(pre_d_softplus, -1, x)
