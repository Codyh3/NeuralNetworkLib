import numpy as np
import nn_layers


class NeuralNetwork:
    """
    A simple neural net object.

    """

    def __init__(self):
        self.layers = []

    def add_layer(self, layer=None, layer_type='Dense', num_input_units=None, num_output_units=1,
                  activation_function='', weights=None, bias=None):
        """
        Add a layer to the neural network. If a layer instance is passed into the layer keyword arg, then all other args
        will be ignored.

        Inputs:
            layer (nn_layers.layer) - A predefined layer already created
            layer_type (str) - The type of the layer to add (Dense is the only option for now).
            num_input_units (int) - The number of inputs to the layer
            num_output_units (int) - The number of outputs of the layer
            activation_function (numpy aware function or string) - The activation function to apply to the outputs (
                                                     e.g., Relu, sigmoid, tanh etc...). This function must be applicable
                                                     to numpy arrays. You can specify "sigmoid", "tanh" or "ReLu" as
                                                     strings to use those functions. Defaults to ReLu
            weights (numpy array) - The weights for the layer in the form of a num_input_units x num_output_units numpy
                                array. If weights are none or not provided, then the layer will start with random
                                weights.
            bias (numpy array) - The bias terms in the form of a 1 x num_output_units numpy array. If the bias is none
                             or not provided, it is initially set to the zero vector.
        """
        if layer is not None:
            if len(self.layers) == 0:
                self.layers.append(layer)
            else:
                if self.layers[-1].num_output_units != layer.num_input_units:
                    print("Unable to add layer as the output dimension of the previous layer in the network does not \
                          match the input dimension of the layer being added.")
                else:
                    self.layers.append(layer)
        else:
            if len(self.layers) == 0:
                if num_input_units is None:
                    print("First layer must have the number of input units defined.")
                else:
                    if layer_type == 'Dense':
                        self.layers.append(nn_layers.DenseLayer(num_input_units=num_input_units,
                                                                num_output_units=num_output_units,
                                                                activation_function=activation_function,
                                                                weights=weights,
                                                                bias=bias))
            else:
                if self.layers[-1].num_output_units != num_input_units:
                    print("Unable to add layer as the output dimension of the previous layer in the network does not \
                          match the input dimension of the layer being added.")
                else:
                    if layer_type == 'Dense':
                        self.layers.append(nn_layers.DenseLayer(num_input_units=num_input_units,
                                                                num_output_units=num_output_units,
                                                                activation_function=activation_function,
                                                                weights=weights,
                                                                bias=bias))

    def print_architecture(self):
        for layer in self.layers:
            print(f"{layer.layer_type}, {(layer.num_input_units, layer.num_output_units)}")

    def feed_forward(self, x):
        for layer in self.layers:
            x = layer.feed_forward(x)
        return x








nn = NeuralNetwork()
weights1 = np.array([[1, 2, 3], [6, 5, 4]])
bias1 = np.array([[-0.5, 0.5]])
nn.add_layer(layer_type='Dense', num_input_units=3, num_output_units=2, weights=weights1, bias=bias1)
weights2 = np.array([[-1, 1]])
bias2 = np.array([[-1.8]])
layer = nn_layers.DenseLayer(num_input_units=2, num_output_units=1, weights=weights2, bias=bias2)
nn.add_layer(layer=layer)




layer = nn.layers[0]
x = np.array([0,1,-1])
h = layer.feed_forward(x)
layer = nn.layers[-1]
layer.feed_forward(h)
