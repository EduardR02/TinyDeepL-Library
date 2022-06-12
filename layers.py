import numpy as np
import activations


class Layer:
    def __init__(self, activation, input_shape, output_shape):
        # not every layer has activation, meaning it can be None
        self.activation = activation
        self.input_shape = None
        self.output_shape = None
        self.set_input_shape(input_shape)
        self.set_output_shape(output_shape)
        self.has_parameters = False     # signifies that a layer does not have values that can be trained,
        # like weights and bias
        self.init_activation()

    def forward(self, inputs):
        raise NotImplementedError()

    def forward_test_time(self, inputs):
        raise NotImplementedError()

    def backward(self, error, layer_outputs, previous_layer_outputs, mult_error_and_actv):
        raise NotImplementedError()

    def calculate_error(self, propagated_error):
        raise NotImplementedError()

    def apply_gradients(self, weight_gradients, bias_gradients):
        return

    def init_activation(self):
        if self.activation:
            if self.activation == "relu":
                self.activation = activations.Relu(self.activation)
            elif self.activation == "leaky_relu":
                self.activation = activations.Leaky_Relu(self.activation)
            elif self.activation == "sigmoid":
                self.activation = activations.Sigmoid(self.activation)
            elif self.activation == "softmax":
                self.activation = activations.Softmax(self.activation)
            else:
                self.activation = None

    def init_parameters(self):
        if self.has_parameters:
            raise NotImplementedError()
        print("Layer has no parameters, therefore cannot initialize them")

    def set_input_shape(self, input_shape):
        if input_shape and not isinstance(input_shape, tuple):
            raise TypeError("Input shape has to be a tuple")
        self.input_shape = input_shape

    def set_output_shape(self, output_shape):
        if output_shape and not isinstance(output_shape, tuple):
            raise TypeError("Output shape has to be a tuple")
        self.output_shape = output_shape

    def set_parameters(self, weights, bias):
        """
        This function will not deepcopy the given parameters, it will simply assign them
        """
        return

    def get_parameters(self):
        return None, None

    def mult_error_with_derivative_of_actv_func(self, error, output):
        actv_derivative = self.activation.derivative(output)
        # in case the derivative of the activation function is a jacobian matrix, and not a vector,
        # the dimension of the error need to be expanded for the matrix multiplication.
        # after the matrix multiplication the resulting redundant dimension needs to be removed
        if error.shape == actv_derivative.shape:
            return error * actv_derivative

        error = error[:, np.newaxis, :]
        return np.squeeze(np.matmul(error, actv_derivative), axis=1)


class FullyConnected(Layer):
    def __init__(self, neurons, activation="relu", use_bias=False, input_shape=None):
        # 1 is batch size, modify later when at that step (make variable or smth for vectorization)
        super().__init__(activation, input_shape, (1, neurons))
        self.neurons = neurons
        self.use_bias = use_bias
        self.weights = None
        self.bias = None
        self.has_parameters = True

    def forward(self, inputs):
        res = np.matmul(inputs, self.weights)
        if self.use_bias:
            res += self.bias
        res = self.activation.forward(res)
        return res

    def forward_test_time(self, inputs):
        return self.forward(inputs)

    def backward(self, error, layer_outputs, previous_layer_outputs, mult_error_and_actv):
        weight_gradients, bias_grads = self.calculate_weight_gradients(error, layer_outputs, previous_layer_outputs, mult_error_and_actv)
        return weight_gradients, bias_grads

    def calculate_error(self, propagated_error):
        error = np.matmul(propagated_error, np.transpose(self.weights))
        return error

    def calculate_weight_gradients(self, propagated_error, layer_outputs, previous_layer_outputs, mult_error_and_actv):
        """
        :param: mult_error_and_actv: Boolean that signifies if the derivative of the activation function and the
        propagated error need to be multiplied or not. This is necessary as for example in the case of softmax + CE loss
        they don't, as their derivatives are usually combined
        """
        full_error = propagated_error if mult_error_and_actv else self.mult_error_with_derivative_of_actv_func(propagated_error, layer_outputs)
        weight_gradients = np.matmul(np.transpose(previous_layer_outputs), full_error)
        # return full_error because that is also the bias_gradients, sum it because over batch
        return weight_gradients, np.sum(full_error, axis=0)

    def apply_gradients(self, weight_gradients, bias_gradients):
        # plus and not minus because that is handled in the optimizers
        self.weights += weight_gradients
        if self.use_bias:
            self.bias += bias_gradients

    def init_parameters(self):
        self.init_weights()
        if self.use_bias:
            self.init_bias()

    # call when input and output shapes are known
    def init_weights(self):
        self.weights = np.random.normal(0.0, pow(self.input_shape[-1], -0.5),
                                        (self.input_shape[-1], self.output_shape[-1]))

    def init_bias(self):
        self.bias = np.atleast_2d(np.zeros(self.output_shape[-1]))

    def set_parameters(self, weights, bias):
        if weights is not None:
            self.weights = weights
        if bias is not None and self.use_bias:
            self.bias = bias

    def get_parameters(self):
        return self.weights, self.bias


class InvertedDropout(Layer):
    # good stackoverflow explanation: https://stackoverflow.com/questions/54109617/implementing-dropout-from-scratch
    # dropout is what percentage of activations is kept!
    # input_shape parameter in case you want it as first layer, needs to match input in that case
    def __init__(self, dropout=1.0, input_shape=None):
        super().__init__(None, input_shape, input_shape)
        self.dropout = dropout
        self.dropout_mask = None
        if self.dropout > 1.0 or self.dropout < 0.0:
            raise ValueError("Dropout value has to be between 0.0 and 1.0")

    def forward(self, outputs):
        # neuron is kept with probability q = 1 - p, meaning p is how many are dropped, here self.dropout is q
        output_mask = np.random.rand(*outputs.shape) < self.dropout
        self.dropout_mask = output_mask
        masked_outputs = outputs * output_mask
        masked_outputs /= self.dropout   # this step is what makes it inverted dropout,
        # making it so nothing needs to be done at inference-time
        return masked_outputs

    def forward_test_time(self, inputs):
        # Do nothing, because inverted_dropout
        return inputs

    def backward(self, error, layer_outputs, previous_layer_outputs, mult_error_and_actv):
        # No parameters so nothing has to be set / returned
        return None, None

    def calculate_error(self, propagated_error):
        # the error is again masked ( the outputs that didnt contribute / were set to 0 are again set to 0)
        # and the output is scaled to make up for the masking
        return (propagated_error * self.dropout_mask) / self.dropout

    def set_input_shape(self, input_shape):
        super(InvertedDropout, self).set_input_shape(input_shape)
        super(InvertedDropout, self).set_output_shape(input_shape)

    def set_output_shape(self, output_shape):
        super(InvertedDropout, self).set_input_shape(output_shape)
        super(InvertedDropout, self).set_output_shape(output_shape)


# not correctly implemented yet, because for flatten to be useful other layers beside fully connected are necessary :)
class Flatten(Layer):
    # input_shape parameter in case you want it as first layer, needs to match input in that case
    # make sure to NOT flatten the batch dimension!
    def __init__(self, input_shape=None):
        # not sure about shapes here, check later
        super().__init__(None, None, None)
        if input_shape:
            self.set_input_shape(input_shape)

    def forward(self, inputs):
        return np.atleast_2d(inputs.flatten()).transpose()

    def forward_test_time(self, inputs):
        return self.forward(inputs)

    def backward(self, error, layer_outputs, previous_layer_outputs, mult_error_and_actv):
        raise NotImplementedError()

    def set_input_shape(self, input_shape):
        if not input_shape:
            return
        super(Flatten, self).set_input_shape(input_shape)
        elems = 1
        for i in input_shape:
            elems *= i
        super(Flatten, self).set_output_shape((elems, 1))


class BatchNormalization(Layer):
    def __init__(self, activation, input_shape, output_shape):
        super().__init__(None, None, None)
        raise NotImplementedError()
