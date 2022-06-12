import numpy as np
import scipy.special as sci


class Activation_Function:
    def __init__(self, name):
        self.name = name

    def forward(self, layer_outputs):
        raise NotImplementedError()

    def derivative(self, layer_activations):
        raise NotImplementedError()


class Relu(Activation_Function):
    def forward(self, layer_outputs):
        return np.maximum(0.0, layer_outputs)

    def derivative(self, layer_activations):
        x = layer_activations.copy()
        x[x < 0] = 0.0
        x[x > 0] = 1.0
        return x


class Leaky_Relu(Activation_Function):
    def __init__(self, name):
        super().__init__(name)
        self.alpha = 0.01

    def forward(self, layer_outputs):
        # values above zero remain unchanged, same as ReLu
        # values below zero are multiplied with an alpha value (to deal with the problem of dead relus)
        values_greater_zero = ((layer_outputs > 0) * layer_outputs)
        values_less_than_zero = ((layer_outputs <= 0) * layer_outputs * self.alpha)
        combined = values_greater_zero + values_less_than_zero
        return combined

    def derivative(self, layer_activations):
        # derivative of alpha * x = alpha
        x = layer_activations.copy()
        x[x < 0] = self.alpha
        x[x > 0] = 1.0
        return x


class Sigmoid(Activation_Function):
    def forward(self, layer_outputs):
        # scipy implementation is fast
        return sci.expit(layer_outputs)

    def derivative(self, layer_activations):
        # very nice derivation here:
        # https://towardsdatascience.com/derivative-of-the-sigmoid-function-536880cf918e
        return layer_activations * (1.0 - layer_activations)


class Softmax(Activation_Function):
    """
    Derivations: https://peterroelants.github.io/posts/cross-entropy-softmax/
    """
    def forward(self, layer_outputs):
        return self.forward_shifted(layer_outputs)

    def forward_shifted(self, layer_outputs):
        """
        same as regular softmax, but shifts the values down before exponentiation to prevent NaNs
        """
        exps = np.exp(layer_outputs - np.amax(layer_outputs, axis=-1, keepdims=True))
        return self.forward_division_step(exps)

    def forward_regular(self, layer_outputs):
        exps = np.exp(layer_outputs)
        return self.forward_division_step(exps)

    def forward_division_step(self, exps):
        # Because both forward_shifted and forward regular share this step, this function exists
        return exps / np.sum(exps, axis=-1, keepdims=True)

    def regular_derivative(self, layer_activations):
        """
        computes the softmax jacobian, which is necessary because each output of the softmax depends on all inputs

        softmax derivative when i==j: yi * (1 - yj)
        softmax derivative when i!=j: -yi * yj
        or
        yi * (kronecker_delta_ij - yj)
        :return: (batch_size, n, n) shape softmax jacobian
        """
        # sets up the identity matrix to compute the jacobian, because derivative only differs when i==j
        # shape is -1 because layer_activation is expected to be of shape = (batch_size, outputs)
        identity_matrix = np.identity(layer_activations.shape[-1])
        # expand for batch_size
        identity_matrix = np.stack([identity_matrix] * layer_activations.shape[0], axis=0)
        # transpose activations for subtraction from identity (batch size dimension is not transposed, only output dim)
        x = layer_activations[:, np.newaxis, :]
        # add axis to activations for column wise multiplication with resulting matrix
        y = layer_activations[:, :, np.newaxis]
        # this is the softmax derivative, resulting in it's jacobian
        identity_matrix = y * (identity_matrix - x)
        return identity_matrix

    def derivative(self, layer_activations):
        """
        softmax derivative when i==j: yi * (1 - yj)
        softmax derivative when i!=j: -yi * yj
        or
        yi * (kronecker_delta_ij - yj)
        resulting in jacobian

        with cross-entropy derivative becomes much simpler: output - label
        """
        return self.regular_derivative(layer_activations)
