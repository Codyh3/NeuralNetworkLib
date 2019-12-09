"garbage file"

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