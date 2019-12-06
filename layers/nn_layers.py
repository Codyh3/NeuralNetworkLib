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
    exp_x = np.exp(x)
    return exp_x / exp_x.sum(axis=1, keepdims=True)


def pre_d_softmax(x):
    vector = np.exp(x) / np.exp(x).sum()
    return np.diag(vector) - outer_product(vector)


def d_softmax(x):
    return np.apply_along_axis(pre_d_softmax, -1, x)


def softplus(x):
    return np.log(1 + np.exp(x))


def pre_d_softplus(x):
    return np.diag(sigmoid(x).reshape(-1,))


def d_softplus(x):
    return np.apply_along_axis(pre_d_softplus, -1, x)


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
                self.activation_function = tanh
                self.dactivation_function = d_tanh
            elif activation_function in ['sigmoid', 'Sigmoid']:
                self.activation_function = sigmoid
                self.dactivation_function = d_sigmoid
            elif activation_function in ['identity', 'id', 'Id']:
                self.activation_function = identity
                self.dactivation_function = d_identity
            elif activation_function in ['softmax', 'Softmax']:
                self.activation_function = softmax
                self.dactivation_function = d_softmax
            else:
                self.activation_function = ReLu
                self.dactivation_function = d_ReLu
        else:
            self.activation_function, self.dactivation_function = activation_function

        if isinstance(weights, np.ndarray):
            self.weights = weights
        else:
            self.weights = np.random.normal(loc=0.0, scale=st_dev, size=(num_input_units, num_output_units))

        if isinstance(bias, np.ndarray):
            self.bias = bias
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


def crossover_layers(layer_1, layer_2):
    """
    Given two layers of the same type, perform crossover and produce a new layer of the same type as the parents.

    Inputs:
        layer_1 (nn layer) - The first layer used in crossover.
        layer_2 (nn layer) - The second layer used in crossover.

    Returns:
        layer (nn layer) - A layer created from the crossover of the two input layers.

    """
    num_input_units = layer_1.num_input_units
    num_output_units = layer_1.num_output_units
    layer_type = layer_1.layer_type
    weights = np.zeros(shape=(num_output_units, num_input_units))
    bias = np.zeros(shape=(1, num_output_units))
    if (num_input_units != layer_2.num_input_units) or (num_output_units != layer_2.num_output_units) or (
            layer_type != layer_2.layer_type):
        print("Unable to perform crossover as dimensions and/or type of layers do not agree")
        return None
    if np.random.choice([1, 2]) == 1:
        activation_function = layer_1.activation_function
    else:
        activation_function = layer_2.activation_function
    weights_mask = np.random.choice([0, 1], size=(num_output_units, 1))
    weights = layer_1.weights * weights_mask + layer_2.weights * (1 - weights_mask)
    bias_mask = np.random.choice([0, 1], size=(1, num_output_units))
    bias = layer_1.bias * bias_mask + layer_2.bias * (1 - bias_mask)
    return DenseLayer(num_input_units, num_output_units, activation_function, weights, bias)


def mutate_layer(layer, mutation_probability=0.01):
    """
    Given the weights and bias of a layer, we randomly and indepently mutate each number with probability
    mutation_probability by replacing the weight/bias element by an observation from a uniform (-1, 1)
    random variable

    Inputs:
        layer (nn layer) - A layer object whose weights and bias are to be mutated.
        mutation_probability (float in (0, 1)) - The probability that a given element of the weights or bias will be
                                                 mutated.

    """
    weights_to_mutate = np.random.choice(a=[0, 1], size=(layer.num_output_units, layer.num_input_units),
                                         p=(1-mutation_probability, mutation_probability))
    #layer.set_weights(layer.weights * (1 - weights_to_mutate))
    layer.set_weights(layer.weights + (2 * np.random.random(
            size=(layer.num_output_units, layer.num_input_units)) - 1) * weights_to_mutate)

    bias_to_mutate = np.random.choice(a=[0, 1], size=(1, layer.num_output_units),
                                      p=(1-mutation_probability, mutation_probability))
    #layer.set_bias(layer.bias * (1 - bias_to_mutate))
    layer.set_bias(layer.bias + (2 * np.random.random(size=(1, layer.num_output_units)) - 1) * bias_to_mutate)
    return None



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


















