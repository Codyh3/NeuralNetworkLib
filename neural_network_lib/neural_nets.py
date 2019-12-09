import numpy as np

from neural_network_lib.loss_models import loss_models
from neural_network_lib.layers import nn_layers


__version__ = '0.0.0'


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

    def get_architecture(self, include_weights=False):
        nn_architecture = []
        for layer in self.layers:
            temp_dict = dict(
                        layer_type=layer.layer_type,
                        num_input_units=layer.num_input_units,
                        num_output_units=layer.num_output_units,
                        activation_function=layer.activation_function.__name__,
                        )
            if include_weights:
                temp_dict['weights'] = layer.weights.tolist()
                temp_dict['bias'] = layer.bias.tolist()
            nn_architecture.append(temp_dict)
        return nn_architecture

    def predict(self, x):
        for layer in self.layers:
            x = layer.predict(x)
        return x

    def forward_prop(self, X):
        S = []
        Activations = [X]
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

    def compute_gradient(self, X, y, loss_model, accumulate_grads=True):
        S, Activations = self.forward_prop(X)
        sensitivities = self.backward_prop(S, Activations, y, loss_model)
        weight_grads = []
        bias_grads = []
        for s, x in zip(sensitivities, Activations):
            m, n = x.shape
            x = x.reshape(m, 1, n)
            temp_weight_grad = np.matmul(np.transpose(x, axes=[0, 2, 1]), s)
            if accumulate_grads:
                weight_grads.append(loss_model.grad_accumulator(temp_weight_grad))
                bias_grads.append(loss_model.grad_accumulator(s))
            else:
                weight_grads.append(temp_weight_grad)
                bias_grads.append(s)
        return weight_grads, bias_grads

    def run_batch_gradient_descent(self, X, y, loss_model, learning_rate=0.01,
                                   error_tolerance=0.01, max_epochs=100000):
        total_error = 1e16
        epoch = 0
        tot_err_array = []
        while total_error > error_tolerance:
            weight_grads, bias_grads = NN.compute_gradient(X, y, loss_model)
            for i in range(len(NN.layers)):
                NN.layers[i].set_weights(NN.layers[i].weights - (learning_rate * weight_grads[i]))
                NN.layers[i].set_bias(NN.layers[i].bias - (learning_rate * bias_grads[i]))
            total_error = loss_model.loss_function(NN.predict(X), y)
            tot_err_array.append(total_error)
            if epoch > max_epochs:
                break
            if epoch % 1000 == 0:
                print(f"{epoch} - {total_error}")
            epoch += 1
        return tot_err_array

    def run_mini_batch_gradient_descent(self, X, y, loss_model, batch_size=128, learning_rate=0.01,
                                        error_tolerance=0.01, max_epochs=100000):
        total_error = 1e16
        epoch = 0
        tot_err_array = []
        num_training_examples = X.shape[0]
        if num_training_examples % batch_size == 0:
            num_batch_loops = num_training_examples // batch_size
        else:
            num_batch_loops = num_training_examples // batch_size + 1
        while total_error > error_tolerance:
            for i in range(num_batch_loops):
                weight_grads, bias_grads = NN.compute_gradient(X[i * batch_size:(i + 1) * batch_size, :],
                                                               y[i * batch_size:(i + 1) * batch_size, :],
                                                               loss_model)
                for l in range(len(NN.layers)):
                    NN.layers[l].set_weights(NN.layers[l].weights - (learning_rate * weight_grads[l]))
                    NN.layers[l].set_bias(NN.layers[l].bias - (learning_rate * bias_grads[l]))
            total_error = loss_model.loss_function(NN.predict(X), y)
            tot_err_array.append(total_error)
            if epoch > max_epochs:
                break
            if epoch % 1000 == 0:
                print(f"{epoch} - {total_error}")
            epoch += 1
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


############################################################################################################
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
loss_model = loss_models.SumSquaresLoss()
input_list = [x1, x2, x3, x4]
output_list = [0, 1, 1, 0]

idx = 1
x = input_list[idx]
expected_output = output_list[idx]

weight_grads, bias_grads = NN.compute_gradient(x, expected_output, loss_model)


x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
loss_model = loss_models.SumSquaresLoss()
y = np.array([[0], [1], [1], [0]])
y_hat = NN.predict(x)
weight_grads, bias_grads = NN.compute_gradient(x, expected_output, loss_model)

############################################################################################################
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])
loss_model = loss_models.SumSquaresLoss()


def test_gradients(nn_arch, X, y, loss_model, h=0.001, tolerance=1e-8):
    NN = create_nn_from_arch(nn_arch)
    weight_grads, bias_grads = NN.compute_gradient(X, y, loss_model)
    for l, layer in enumerate(NN.layers):
        m, n = layer.num_input_units, layer.num_output_units
        for j in range(n):
            # Compare the bias gradients
            nn_arch_1 = NN.get_architecture(include_weights=True)
            nn_arch_2 = NN.get_architecture(include_weights=True)
            nn_arch_1[l]['bias'][0][j] = nn_arch_1[l]['bias'][0][j] + h
            nn_arch_2[l]['bias'][0][j] = nn_arch_2[l]['bias'][0][j] - h
            NN1 = create_nn_from_arch(nn_arch_1)
            NN2 = create_nn_from_arch(nn_arch_2)
            y1 = loss_model.loss_function(NN1.predict(X), y)
            y2 = loss_model.loss_function(NN2.predict(X), y)
            numerical_gradient = (y1 - y2) / (2 * h)
            if abs(bias_grads[l][0, j] - numerical_gradient) > tolerance:
                print(f"PROBLEM at layer {l}, location {0, j} for bias gradients:")
                print(f"numerical_grad={numerical_gradient}, bias_grad={bias_grads[l][0, j]}")
            for i in range(m):
                # Compare the weight gradients
                nn_arch_1 = NN.get_architecture(include_weights=True)
                nn_arch_2 = NN.get_architecture(include_weights=True)
                nn_arch_1[l]['weights'][i][j] = nn_arch_1[l]['weights'][i][j] + h
                nn_arch_2[l]['weights'][i][j] = nn_arch_2[l]['weights'][i][j] - h
                NN1 = create_nn_from_arch(nn_arch_1)
                NN2 = create_nn_from_arch(nn_arch_2)
                y1 = loss_model.loss_function(NN1.predict(X), y)
                y2 = loss_model.loss_function(NN2.predict(X), y)
                numerical_gradient = (y1 - y2) / (2 * h)
                if abs(weight_grads[l][i, j] - numerical_gradient) > tolerance:
                    print(f"PROBLEM at layer {l}, location {i, j} for weight gradients:")
                    print(f"numerical_grad={numerical_gradient}, weight_grad={weight_grads[l][i, j]}")


############################################################################################################
layer_1 = dict(
        layer_type='Dense',
        num_input_units=5,
        num_output_units=3,
        activation_function='softplus',
        st_dev=2
        )
layer_2 = dict(
        layer_type='Dense',
        num_input_units=3,
        num_output_units=1,
        activation_function='sigmoid',
        st_dev=2
        )
nn_arch = [layer_1, layer_2]
NN = create_nn_from_arch(nn_arch)

X = np.random.normal(size=(4, 5))
y = np.array([0, 2, 1, 2])
num_classes = np.max(y) + 1
y = np.eye(num_classes)[y]


loss_model = loss_models.CrossEntropyLoss()
weight_grads, bias_grads = NN.compute_gradient(X, y, loss_model)

test_gradients(nn_arch, X, y, loss_model, h=0.00001, tolerance=1e-15)

############################################################################################################
layer_1 = dict(
        layer_type='Dense',
        num_input_units=2,
        num_output_units=2,
        activation_function='softplus',
        )
layer_2 = dict(
        layer_type='Dense',
        num_input_units=2,
        num_output_units=1,
        activation_function='identity',
        )
nn_arch = [layer_1, layer_2]
NN = create_nn_from_arch(nn_arch)

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])
loss_model = loss_models.SumSquaresLoss()

tot_err_array = NN.run_gradient_descent(X, y, loss_model, learning_rate=0.01, error_tolerance=0.000001, max_epochs=100000)

NN.predict(X)

test_gradients(NN.get_architecture(True), X, y, loss_model, h=0.00001, tolerance=1e-8)


############################################################################################################
layer_1 = dict(
        layer_type='Dense',
        num_input_units=784,
        num_output_units=128,
        activation_function='softplus',
        )
layer_2 = dict(
        layer_type='Dense',
        num_input_units=128,
        num_output_units=10,
        activation_function='softmax',
        )
nn_arch = [layer_1, layer_2]
NN = create_nn_from_arch(nn_arch)

from tensorflow import keras
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
train_images = train_images / 255.0
test_images = test_images / 255.0
train_images = train_images.reshape((train_images.shape[0], -1))
test_images = test_images.reshape((test_images.shape[0], -1))

train_labels.shape
num_classes = np.max(train_labels) + 1
y = np.eye(num_classes)[train_labels]

loss_model = loss_models.CrossEntropyLoss()

tot_err_array = NN.run_mini_batch_gradient_descent(train_images, y, loss_model, batch_size=500,
                                                   learning_rate=0.01, error_tolerance=0.01, max_epochs=100000)

y_hat = NN.predict(train_images)
y_hat = np.argmax(y_hat, axis=1)
print(f"ACCURACY = {(y_hat==training_labels).mean()}")
