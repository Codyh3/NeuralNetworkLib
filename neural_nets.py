import numpy as np
import nn_layers


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


class NeuralNetwork:
    """
    A simple neural net object.

    """

    def __init__(self):
        self.layers = []

    def add_layer(self, layer=None, layer_type='Dense', num_input_units=None, num_output_units=1,
                  activation_function='', weights=None, bias=None, st_dev=0.1):
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
            st_dev (float) - The standard deviation of the normal random variable used to initialize the weights if None
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
                                                                bias=bias,
                                                                st_dev=st_dev))
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
                                                                bias=bias,
                                                                st_dev=st_dev))

    def print_architecture(self):
        for layer in self.layers:
            print(f"{layer.layer_type}, {(layer.num_input_units, layer.num_output_units)}")

    def predict(self, x):
        for layer in self.layers:
            x = layer.predict(x)
        return x

    def forward_prop(self, x):
        S = []
        Activations = [x]
        for layer in self.layers:
            s, h = layer.feed_forward(Activations[-1])
            S.append(s)
            Activations.append(h)
        return S, Activations

    def backward_prop(self, S, Activations, expected_output, loss_model):
        sensitivities = []
        L = len(self.layers)
        m, n = Activations[L].shape
        sensitivities.append(np.matmul(loss_model.loss_gradient(Activations[L], expected_output).reshape(m, 1, n),
                                       self.layers[L-1].dactivation_function(S[L-1])))
        for l in range(L-1, 0, -1):
            delta = np.matmul(sensitivities[-1],
                              np.matmul(self.layers[l].weights.T, self.layers[l-1].dactivation_function(S[l-1])))
            sensitivities.append(delta)
        sensitivities = sensitivities[::-1]
        return sensitivities

    def compute_gradient(self, Activations, sensitivities):
        weight_grads = []
        bias_grads = sensitivities
        for s, x in zip(sensitivities, Activations):
            m, n = x.shape
            x = x.reshape(m, 1, n)
            weight_grads.append(np.matmul(np.transpose(x, axes=[0, 2, 1]), s))
        return weight_grads, bias_grads

    def run_gradient_descent(self, loss_model, learning_rate=0.01, error_tolerance=0.01,
                             input_list=[], output_list=[]):
        total_error = 1e8
        N = len(output_list)
        weights_update = []
        bias_update = []
        max_epochs = 100000
        count = 0
        tot_err_array = []
        for i in range(len(self.layers)):
            weights_update.append(np.zeros_like(self.layers[i].weights))
            bias_update.append(np.zeros_like(self.layers[i].bias))
        while total_error > error_tolerance:
            for x, y in zip(input_list, output_list):
                S, Activations = self.forward_prop(x)
                sensitivities = self.backward_prop(S, Activations, y, loss_model)
                weight_grads, bias_grads = self.compute_gradient(Activations, sensitivities)
                for i, (wg, bg) in enumerate(zip(weight_grads, bias_grads)):
                    weights_update[i] = weights_update[i] + wg
                    bias_update[i] = bias_update[i] + bg
            for i in range(len(self.layers)):
                self.layers[i].set_weights(self.layers[i].weights - (learning_rate * weights_update[i] / N))
                self.layers[i].set_bias(self.layers[i].bias - (learning_rate * bias_update[i] / N))
                weights_update[i] = weights_update[i] * 0
                bias_update[i] = bias_update[i] * 0
            total_error = 0
            for x, y in zip(input_list, output_list):
                total_error += loss_model.loss_function(self.predict(x), y)
            total_error /= N
            tot_err_array.append(total_error)
            if count > max_epochs:
                break
            if count % 1000 == 0:
                print(f"{count} - {total_error}")
            count += 1
        return tot_err_array
        


def create_nn_from_arch(nn_architecture):
    """
    Given a nn_architecture, create a neural network.

    Inputs:
        nn_architecture (list of dicts) - list of dicts where each dict specifies the inputs to the NeuralNet.add_layer
                                          method. If an input is not specified in the dict, the default will be used.

    Returns:
        NN (NeuralNet) - A neural network created with the prescribe architecture defined in nn_architecture.

    """
    NN = NeuralNetwork()
    for layer in nn_architecture:
        NN.add_layer(**layer)
    return NN


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
        layer = nn_layers.crossover_layers(layer_1, layer_2)
        nn_layers.mutate_layer(layer, mutation_probability)
        NN.add_layer(layer=layer)
    return NN




# =============================================================================
# layer_1 = dict(
#         layer_type='Dense',
#         num_input_units=1,
#         num_output_units=2,
#         activation_function='tanh',
#         weights=np.array(np.array([[0.3, 0.4]])),
#         bias = np.array([[0.1, 0.2]])
#         )
# layer_2 = dict(
#         layer_type='Dense',
#         num_input_units=2,
#         num_output_units=1,
#         activation_function='tanh',
#         weights=np.array(np.array([[1], [-3]])),
#         bias = np.array([[0.2]])
#         )
# layer_3 = dict(
#         layer_type='Dense',
#         num_input_units=1,
#         num_output_units=1,
#         activation_function='tanh',
#         weights=np.array(np.array([[2]])),
#         bias = np.array([[1]])
#         )
# nn_arch = [layer_1, layer_2, layer_3]
# NN = create_nn_from_arch(nn_arch)
# 
# x = np.array([[2]])
# S, Activations = NN.forward_prop(x)
# expected_output = 1
# loss_model = SumSquaresLoss()
# sensitivities = NN.backward_prop(S, Activations, expected_output, loss_model)
# weight_grads, bias_grads = NN.compute_gradient(Activations, sensitivities)
# =============================================================================


layer_1 = dict(
        layer_type='Dense',
        num_input_units=2,
        num_output_units=2,
        activation_function='softmax',
        weights=np.array([[-0.04919933, 0.05634465], [-0.02162292, 0.17382581]]),
        bias=np.array([[-0.5, 1]])
        )
layer_2 = dict(
        layer_type='Dense',
        num_input_units=2,
        num_output_units=1,
        activation_function='identity',
        weights=np.array([[-0.07560012], [0.0986376]]),
        bias=np.array([[0]])
        )
nn_arch = [layer_1, layer_2]
NN = create_nn_from_arch(nn_arch)

x1 = np.array([[0, 0]])
x2 = np.array([[0, 1]])
x3 = np.array([[1, 0]])
x4 = np.array([[1, 1]])
loss_model = SumSquaresLoss()
input_list = [x1, x2, x3, x4]
output_list = [0, 1, 1, 0]

idx = 0
x = input_list[idx]
expected_output = output_list[idx]

S, Activations = NN.forward_prop(x)
sensitivities = NN.backward_prop(S, Activations, expected_output, loss_model)
weight_grads, bias_grads = NN.compute_gradient(Activations, sensitivities)

[array([[ 0.03705399, -0.04777052],
        [ 0.03705399, -0.04777052]]),
 array([[-0.94675111],
        [-1.09395155]])]

[array([[ 0.03705399, -0.04777052]]),
 array([[-1.96298467]])]


x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
loss_model = SumSquaresLoss()
y = np.array([[0], [1], [1], [0]])

S, Activations = NN.forward_prop(x)
expected_output = y
sensitivities = NN.backward_prop(S, Activations, expected_output, loss_model)
weight_grads, bias_grads = NN.compute_gradient(Activations, sensitivities)


sensitivities = []
L = len(NN.layers)
m, n = Activations[L].shape
sensitivities.append(np.matmul(loss_model.loss_gradient(Activations[L], expected_output).reshape(m, 1, n),
                               NN.layers[L-1].dactivation_function(S[L-1])))
for l in range(L-1, 0, -1):
    delta = np.matmul(sensitivities[-1],
                   np.matmul(NN.layers[l].weights.T, NN.layers[l-1].dactivation_function(S[l-1])))
    sensitivities.append(delta)
sensitivities = sensitivities[::-1]

weight_grads = []
bias_grads = sensitivities
for s, x in zip(sensitivities, Activations):
    m, n = x.shape
    x = x.reshape(m, 1, n)
    weight_grads.append(np.matmul(np.transpose(x, axes=[0, 2, 1]), s))
############################################################################################################
# =============================================================================
# h=0.0001
# act_1 = 'softmax'
# act_2 = 'identity'
# layer_1 = dict(
#         layer_type='Dense',
#         num_input_units=2,
#         num_output_units=2,
#         activation_function=act_1,
#         weights=np.array([[-0.04919933, 0.05634465], [-0.02162292, 0.17382581]]),
#         bias=np.array([[-0.5, 1]])
#         )
# layer_2 = dict(
#         layer_type='Dense',
#         num_input_units=2,
#         num_output_units=1,
#         activation_function=act_2,
#         weights=np.array([[-0.07560012], [0.0986376]]),
#         bias=np.array([[0+h]])
#         )
# nn_arch = [layer_1, layer_2]
# NN1 = create_nn_from_arch(nn_arch)
# 
# layer_1 = dict(
#         layer_type='Dense',
#         num_input_units=2,
#         num_output_units=2,
#         activation_function=act_1,
#         weights=np.array([[-0.04919933, 0.05634465], [-0.02162292, 0.17382581]]),
#         bias=np.array([[-0.5, 1]])
#         )
# layer_2 = dict(
#         layer_type='Dense',
#         num_input_units=2,
#         num_output_units=1,
#         activation_function=act_2,
#         weights=np.array([[-0.07560012], [0.0986376]]),
#         bias=np.array([[0-h]])
#         )
# nn_arch = [layer_1, layer_2]
# NN2 = create_nn_from_arch(nn_arch)
# 
# layer_1 = dict(
#         layer_type='Dense',
#         num_input_units=2,
#         num_output_units=2,
#         activation_function=act_1,
#         weights=np.array([[-0.04919933, 0.05634465], [-0.02162292, 0.17382581]]),
#         bias=np.array([[-0.5, 1]])
#         )
# layer_2 = dict(
#         layer_type='Dense',
#         num_input_units=2,
#         num_output_units=1,
#         activation_function=act_2,
#         weights=np.array([[-0.07560012], [0.0986376]]),
#         bias=np.array([[0]])
#         )
# nn_arch = [layer_1, layer_2]
# NN3 = create_nn_from_arch(nn_arch)
# 
# x1 = np.array([[0, 0]])
# x2 = np.array([[0, 1]])
# x3 = np.array([[1, 0]])
# x4 = np.array([[1, 1]])
# loss_model = SumSquaresLoss()
# input_list = [x1, x2, x3, x4]
# output_list = [0, 1, 1, 0]
# 
# idx = 0
# x = input_list[idx]
# expected_output = output_list[idx]
# 
# S, Activations = NN3.forward_prop(x)
# sensitivities = NN3.backward_prop(S, Activations, expected_output, loss_model)
# weight_grads, bias_grads = NN3.compute_gradient(Activations, sensitivities)
# 
# 
# y1 = loss_model.loss_function(NN1.predict(x), expected_output)
# y2 = loss_model.loss_function(NN2.predict(x), expected_output)
# print((y1 - y2) / (2 * h))
# 
# for idx in range(4):
#     x = input_list[idx]
#     expected_output = output_list[idx]
# 
#     y1 = loss_model.loss_function(NN1.predict(x), expected_output)
#     y2 = loss_model.loss_function(NN2.predict(x), expected_output)
#     print((y1 - y2) / (2 * h))
# =============================================================================
