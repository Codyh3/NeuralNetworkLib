import numpy as np


def ReLu(x):
    return np.maximum(0, x)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def tanh(x):
    return np.tanh(x)


class DenseLayer:
    """
    This is a dense layer of a feed forward neural network.

    Inputs:
        num_input_units (int) - The number of inputs to the layer
        num_output_units (int) - The number of outputs of the layer
        activation_function (numpy aware function or string) - The activation function to apply to the outputs (e.g.,
                                                     Relu, sigmoid, tanh etc...). This function must be applicable to
                                                     numpy arrays. You can specify "sigmoid", "tanh" or "ReLu" as
                                                     strings to use those functions. Defaults to ReLu
        weights (numpy array) - The weights for the layer in the form of a num_input_units x num_output_units numpy
                                array. If weights are none or not provided, then the layer will start with random
                                weights.
        bias (numpy array) - The bias terms in the form of a 1 x num_output_units numpy array. If the bias is none
                             or not provided, it is initially set to the zero vector.

    Output:
        Instance of a layer class. This will be combined in the NeuralNet class to form a 'deep' feed forward NN.

    """

    layer_type = 'Dense'

    def __init__(self, num_input_units, num_output_units, activation_function='', weights=None, bias=None):
        self.num_input_units = num_input_units
        self.num_output_units = num_output_units
        if isinstance(activation_function, str):
            if activation_function in ['tanh', 'Tanh']:
                self.activation_function = tanh
            elif activation_function in ['sigmoid', 'Sigmoid']:
                self.activation_function = sigmoid
            else:
                self.activation_function = ReLu
        else:
            self.activation_function = activation_function

        if isinstance(weights, np.ndarray):
            self.weights = weights
        else:
            self.weights = np.random.normal(loc=0.0, scale=0.1, size=(num_output_units, num_input_units))

        if isinstance(bias, np.ndarray):
            self.bias = bias
        else:
            self.bias = np.zeros(shape=(1, num_output_units))

    def set_weights(self, weights):
        if weights.shape == (self.num_output_units, self.num_input_units):
            self.weights = weights
        else:
            print("Unable to set weights as dimensions do not agree")

    def feed_forward(self, x):
        """
        Given a vector x in R^n, computes x.T * weights.T + bias and applies the activation function

        Inputs:
            x (numpy array) - A numpy array of shape (1, num_input_units)

        """
        return self.activation_function(np.dot(x, self.weights.T) + self.bias)
