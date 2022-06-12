import loss
import activations
import numpy as np


def compare_ce_and_softmax_combined_to_separate():
    # a lot changed architecturally after this test was written, probably won't work
    ce = loss.CrossEntropy("cross_entropy")
    s = activations.Softmax("softmax")
    input_vec = np.random.uniform(size=(2, 10))
    label = np.zeros_like(input_vec)
    label[np.arange(len(input_vec)), input_vec.argmax(1)] = 1
    print(label)
    s_applied = s.forward(input_vec)
    print(input_vec)
    print(s_applied, s_applied.sum(axis=-1))
    ls_res = ce.loss_function(s_applied, label)
    print(ls_res)
    reg_err = ce.regular_error_function(s_applied, label)
    s_err = ce.error_function_with_softmax(s_applied, label)
    s_jac = s.regular_derivative(s_applied)
    # insert axis for matrix multiplication to get desired output shape
    reg_err = reg_err[:, np.newaxis, :]
    fake_activation_function = activations.Relu("relu")
    _, final_g = ce.error_function(s_applied, label, fake_activation_function)
    print(reg_err.shape)
    print(s_jac.shape)
    print(s_err)
    print(final_g)
    print(final_g.shape)
    # final_g == s_err: cross entropy error * softmax jacobian = short form from derivation of both together


def print_model_layer_shapes(model):
    for layer in model.layers:
        print(layer.input_shape, layer.output_shape)
        if layer.has_parameters:
            print(layer.weights.shape)
            if layer.use_bias:
                print(layer.bias.shape)
