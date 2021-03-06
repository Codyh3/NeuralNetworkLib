import numpy as np

from neural_network_lib.layers import activation_functions


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

    def __init__(self, num_input_units, num_output_units, activation_function='', weights=None, bias=None, st_dev=0.1):
        self.num_input_units = num_input_units
        self.num_output_units = num_output_units
        if isinstance(activation_function, str):
            if activation_function in ['tanh', 'Tanh']:
                self.activation_function = activation_functions.tanh
                self.dactivation_function = activation_functions.d_tanh
            elif activation_function in ['sigmoid', 'Sigmoid']:
                self.activation_function = activation_functions.sigmoid
                self.dactivation_function = activation_functions.d_sigmoid
            elif activation_function in ['identity', 'id', 'Id']:
                self.activation_function = activation_functions.identity
                self.dactivation_function = activation_functions.d_identity
            elif activation_function in ['softmax', 'Softmax']:
                self.activation_function = activation_functions.softmax
                self.dactivation_function = activation_functions.d_softmax
            elif activation_function in ['softplus', 'Softplus']:
                self.activation_function = activation_functions.softplus
                self.dactivation_function = activation_functions.d_softplus
            else:
                self.activation_function = activation_functions.ReLu
                self.dactivation_function = activation_functions.d_ReLu
        else:
            self.activation_function, self.dactivation_function = activation_function

        if isinstance(weights, np.ndarray):
            self.weights = weights
        elif isinstance(weights, list):
            self.weights = np.array(weights)
        else:
            self.weights = np.random.normal(loc=0.0, scale=st_dev, size=(num_input_units, num_output_units))

        if isinstance(bias, np.ndarray):
            self.bias = bias
        elif isinstance(bias, list):
            self.bias = np.array(bias)
        else:
            self.bias = np.zeros(shape=(1, num_output_units))

    def set_weights(self, weights):
        if weights.shape == (self.num_input_units, self.num_output_units):
            self.weights = weights
        else:
            print("Unable to set weights as dimensions do not agree")

    def set_bias(self, bias):
        if bias.shape == (1, self.num_output_units):
            self.bias = bias
        else:
            print("Unable to set bias as dimensions do not agree")

    def predict(self, x):
        """
        Given a vector x in R^n, computes x.T * weights.T + bias and applies the activation function

        Inputs:
            x (numpy array) - A numpy array of shape (1, num_input_units)

        """
        return self.activation_function(np.dot(x, self.weights) + self.bias)

    def feed_forward(self, x):
        """
        Given an input x, compute s = weights^T . x + bias and h = activation_function(s).

        Inputs:
            x (numpy array) - A numpy array of shape (1, num_input_units)

        Returns:
            s (numpy array) - weights^T . x + bias
            h (numpy array) - activation_function(s)
        """
        s = np.dot(x, self.weights) + self.bias
        return (s, self.activation_function(s))



# =============================================================================
# weights1 = np.array([[1,1], [1,1]])
# bias1 = np.array([[1,1]])
# layer1 = DenseLayer(num_input_units=2, num_output_units=2, weights=weights1, bias=bias1)
# 
# weights2 = np.array([[2,2], [2,2]])
# bias2 = np.array([[2,2]])
# layer2 = DenseLayer(num_input_units=2, num_output_units=2, weights=weights2, bias=bias2)
# 
# layer3 = crossover_layers(layer1, layer2)
# print(layer3.weights)
# print(layer3.bias)
# mutate_layer(layer3, 0.5)
# 
# mask = np.array([[0],[1]])
# =============================================================================
