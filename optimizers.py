import numpy as np


class Optimizer:
    def __init__(self, name, layer_shapes, lr=1e-3):
        self.name = name
        self.lr = lr
        self.layer_shapes = layer_shapes

    def optimize(self, gradients):
        raise NotImplementedError()


class NotAnOptimizer(Optimizer):
    """
    This optimizer is supposed to just multiply the gradients by the learning rate.
    """
    def optimize(self, gradients):
        return [-gradients_layer * self.lr if gradients_layer is not None else None for gradients_layer in gradients]


class Adam(Optimizer):
    # Paper: https://arxiv.org/abs/1412.6980
    def __init__(self, name, layer_shapes, lr=1e-3, beta1=0.9, beta2=0.999, epsilon=1e-8):
        super().__init__(name, layer_shapes, lr)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.moment1 = []
        self.moment2 = []
        self.timestep = 0
        self.init_moments()

    def optimize(self, gradients):
        self.timestep += 1
        optimized_gradients = [None] * len(gradients)
        for i in range(len(gradients)):
            if self.moment1[i] is not None:
                self.moment1[i] = self.beta1 * self.moment1[i] + (1 - self.beta1) * gradients[i]
                self.moment2[i] = self.beta2 * self.moment2[i] + (1 - self.beta2) * gradients[i] * gradients[i]
                bias_corrected1 = self.moment1[i] / (1 - self.beta1 ** self.timestep)
                bias_corrected2 = self.moment2[i] / (1 - self.beta2 ** self.timestep)
                optimized_gradients[i] = -self.lr * bias_corrected1 / (np.sqrt(bias_corrected2) + self.epsilon)
            else:
                optimized_gradients[i] = None
        return optimized_gradients

    def init_moments(self):
        # Only instantiate matrices for layers that use the optimizer
        self.moment1 = [np.zeros(shape=layer_shape) if layer_shape is not None else None for layer_shape in
                        self.layer_shapes]
        self.moment2 = [np.zeros(shape=layer_shape) if layer_shape is not None else None for layer_shape in
                        self.layer_shapes]


class AdaGrad(Optimizer):
    pass


class RMSProp(Optimizer):
    pass
