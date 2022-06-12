import numpy as np


class Loss:
    def __init__(self, name):
        self.name = name
        self.epsilon = 1e-8

    def loss_function(self, output, label):
        """
        :return: The return value of the loss function will give the sum of elements in the batch
        """
        raise NotImplementedError()

    def error_function(self, output, label, activation_function):
        """
        Returns the derivative of the loss_function
        :return: Boolean if there is a "shortcut" combination with the activation function,
                 meaning that further transformations do not need to be applied.
                 numpy array containing the derivative of the loss function
        """
        raise NotImplementedError()


class CrossEntropy(Loss):
    def loss_function(self, output, label):
        """
        the calculated loss is divided by the number of samples
        small value added so value inside log never 0
        """
        loss = -np.sum(label * np.log(output + self.epsilon))
        return loss

    def error_function(self, output, label, activation_function):
        if activation_function.name == "softmax":
            return True, self.error_function_with_softmax(output, label)
        return False, self.regular_error_function(output, label)

    def error_function_with_softmax(self, output, label):
        return output - label

    def regular_error_function(self, output, label):
        # avoid division by 0, similar to loss function with log
        return -label / (output + self.epsilon)


class BinaryCrossEntropy(Loss):
    """
    Helpful stack overflow derivative:
    https://math.stackexchange.com/questions/2503428/derivative-of-binary-cross-entropy-why-are-my-signs-not-right
    """
    def loss_function(self, output, label):
        inside_of_sum_term = label * np.log(output + self.epsilon) + (1 - label) * np.log(1 - output + self.epsilon)
        # axis = -1 because axis 0 is batch dimension, and we need the mean for every sample
        loss = -np.sum(np.mean(inside_of_sum_term, axis=-1))
        return loss

    def error_function(self, output, label, activation_function):
        if activation_function.name == "sigmoid":
            return True, self.error_function_with_sigmoid(output, label)
        return False, self.regular_error_function(output, label)

    def error_function_with_sigmoid(self, output, label):
        return output - label

    def regular_error_function(self, output, label):
        # here you can clearly see how the sigmoid derivative cancels out :)
        # epsilon is to prevent division by 0
        return (output - label) / (output * (1 - output) + self.epsilon)


class MeanSquaredError(Loss):
    def loss_function(self, output, label):
        # already "divides" by the number of samples in a batch
        return np.sum(np.mean(np.square(label - output), axis=-1))

    def error_function(self, output, label, activation_function):
        return False, self.regular_error_function(output, label)

    def regular_error_function(self, output, label):
        """
        derivative of mse:
        -2 * (label - prediction) / output_nodes
        """
        return -2 * (label - output) / output.shape[-1]
