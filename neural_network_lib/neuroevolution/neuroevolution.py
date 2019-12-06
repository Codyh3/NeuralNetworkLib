#Neuroevolution methods
import numpy as np

from neural_network_lib.layers.layers import DenseLayer


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
    layer.set_weights(layer.weights + (2 * np.random.random(
            size=(layer.num_output_units, layer.num_input_units)) - 1) * weights_to_mutate)

    bias_to_mutate = np.random.choice(a=[0, 1], size=(1, layer.num_output_units),
                                      p=(1-mutation_probability, mutation_probability))
    layer.set_bias(layer.bias + (2 * np.random.random(size=(1, layer.num_output_units)) - 1) * bias_to_mutate)
    return None


def mate_neural_nets(NN1, NN2, mutation_probability):
    """
    Given two neural networks, perform crossover and mutation of all the layers to produce an offspring

    Inputs:
        NN1 (NeuralNetwork) - The first neural network parent.
        NN2 (NeuralNetwork) - The second neural network parent.
        mutation_probability (float in (0, 1)) - The probability that a given element of the weights or bias will be
                                                 mutated.

    Returns:
        NN (NeuralNetwork) - The neural network that results from performing crossover of the two parent NNs and
                             mutating the result.
    """
    NN = NeuralNetwork()
    for layer_1, layer_2 in zip(NN1.layers, NN2.layers):
        layer = crossover_layers(layer_1, layer_2)
        mutate_layer(layer, mutation_probability)
        NN.add_layer(layer=layer)
    return NN
