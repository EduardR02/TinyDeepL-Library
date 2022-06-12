import file_manager
import loss
import optimizers
import metrics
import utils
import numpy as np


class Model:
    def __init__(self, name=None):
        self.name = name
        self.layers = []
        self.lr = None
        # bias and weight optimizers are split to reduce code duplication in optimizers, just create two instances here,
        # as both bias and weight optimizers do the same thing, but have different parameters due to shape difference
        self.optimizer = None
        self.bias_optimizer = None
        self.input_shape = None
        self.output_shape = None
        self.loss = None
        self.loss_metrics = None
        self.accuracy_metrics = None

    def train(self, data_samples, data_labels, shuffle=False,
              batch_size=1, epochs=1, validation_samples=None, validation_labels=None):
        for epoch in range(epochs):
            if shuffle:
                data_samples, data_labels = utils.shuffle_data(data_samples, data_labels)
            minibatches_samples, minibatches_labels = utils.create_minibatches(data_samples, data_labels, batch_size)
            self.train_loop(minibatches_samples, minibatches_labels, epoch)
            if validation_labels is not None and validation_samples is not None:
                self.test(validation_samples, validation_labels, batch_size=batch_size)

    def test(self, data_samples, data_labels, batch_size=1):
        minibatches_samples, minibatches_labels = utils.create_minibatches(data_samples, data_labels, batch_size)
        self.test_loop(minibatches_samples, minibatches_labels)

    def predict(self, data_sample):
        return self.feed_forward(data_sample)[-1]

    def finish(self, lr, optimizer, loss):
        self.lr = lr
        self.loss = loss
        self.init_loss()
        self.init_learning_rate()
        self.optimizer = self.init_optimizer(optimizer, True)   # True stands for weights, False for bias
        self.bias_optimizer = self.init_optimizer(optimizer, False)   # True stands for weights, False for bias
        self.loss_metrics = metrics.LossTracker(self.loss)
        self.accuracy_metrics = metrics.AccuracyTracker()

    # adds new layer to model, updates input/output shapes if necessary and initialized weights and biases
    def add(self, layer):
        if not self.layers:
            if not layer.input_shape:
                # if layer has an output shape, it is used as input shape also
                # example: if you have a fully_connected layer and specified 100 neurons, the output shape is set,
                # but not the input shape. But if you set the first layer to 100 neurons it is expected that the input
                # will match the neuron count(when no other input shape is provided),
                # therefore the input shape gets changed.
                if not layer.output_shape:
                    raise ValueError("First layer either needs a provided input shape or an appropriate layer has to "
                                     "be used")
                layer.set_input_shape(layer.output_shape)
            self.input_shape = layer.input_shape
        else:
            if not layer.input_shape:
                layer.set_input_shape(self.layers[-1].output_shape)
            elif layer.input_shape and layer.input_shape != self.layers[-1].output_shape:
                raise ValueError("Layer input shape does not match previous layer output shape")

        if not layer.output_shape:
            layer.set_output_shape(layer.input_shape)
        self.output_shape = layer.output_shape
        if layer.has_parameters:
            layer.init_parameters()
        self.layers.append(layer)

    def train_loop(self, samples, labels, epoch=0):
        self.reset_metrics()
        # len or shape depends on implementation of minibatches, either python list or numpy array
        for i in range(len(samples)):
            outputs = self.feed_forward(samples[i])
            weight_gradients, bias_grads = self.calculate_gradients(outputs, labels[i])
            # optimizers are applied in this step
            weight_gradients = self.optimizer.optimize(weight_gradients)
            bias_grads = self.bias_optimizer.optimize(bias_grads)
            self.apply_gradients(weight_gradients, bias_grads)
            # metrics and progress bar
            self.update_metrics(outputs[-1], labels[i])
            self.update_progress_bar((i + 1) * labels[0].shape[0], len(labels) * labels[0].shape[0], epoch)

    def test_loop(self, samples, labels):
        self.reset_metrics()
        for i in range(len(labels)):
            output = self.predict(samples[i])
            self.update_metrics(output, labels[i])
            self.update_progress_bar((i + 1) * labels[0].shape[0], len(labels) * labels[0].shape[0])

    def apply_gradients(self, weight_gradients, bias_gradients):
        """
        :param: weight_gradients: list of weights gradients where the index corresponds to the layer
        :param: bias_gradients: nested list of errors where the first element is a boolean, second the actual bias error
            incomplete, some layers have different params or do not have params, also not every layer has method
        """
        for i in range(len(self.layers)):
            self.layers[i].apply_gradients(weight_gradients[i], bias_gradients[i])

    def feed_forward(self, inputs):
        outputs = [inputs]
        for layer in self.layers:
            outputs.append(layer.forward(outputs[-1]))
        return outputs

    def calculate_gradients(self, outputs, label):
        # look up difference between if error function actually belongs to activation function or loss,
        # see how loss influences backprop
        error = self.calculate_error(outputs, label)
        weight_gradients = [None] * len(self.layers)
        for i in range(len(self.layers) - 1, -1, -1):
            # outputs is 1 longer than layers, because the inputs(sample) is included,
            # as it is necessary for backprop for the first layer of the network
            layer_weight_gradients, bias_grads = self.layers[i].backward(error[i][1], outputs[i + 1],
                                                                         outputs[i], error[i][0])
            weight_gradients[i] = layer_weight_gradients
            # remove now redundant boolean value and thereby nested list
            error[i] = bias_grads
        return weight_gradients, error

    def calculate_error(self, outputs, label):
        """
        Each entry in the error list contains two sub entries: The first is only relevant for the last layer,
        a boolean either True, meaning that the error does not need to be multiplied by the derivative of the activation
        function (like in softmax cross entropy case, because it is incorporated in their combined derivative),
        False meaning it does.
        Second entry is the error itself
        """
        error = [[None, None]] * len(self.layers)
        # determine the error of the model output
        error[-1] = list(self.loss.error_function(outputs[-1], label, self.layers[-1].activation))
        # skip the last element as it has been set already
        for i in range(len(self.layers) - 2, -1, -1):
            error[i] = [False, self.layers[i + 1].calculate_error(error[i + 1][1])]
        return error

    def update_metrics(self, output, label):
        # convenience method
        self.loss_metrics.update_metrics(output, label)
        self.accuracy_metrics.update_metrics(output, label)

    def update_progress_bar(self, current_element, total_elements, epoch=0):
        utils.progressBar(current_element, total_elements, epoch,
                          self.accuracy_metrics.get_metrics(), self.loss_metrics.get_metrics())
        if current_element >= total_elements:
            print("")

    def reset_metrics(self):
        self.loss_metrics.reset()
        self.accuracy_metrics.reset()

    def init_loss(self):
        if self.loss == "cross_entropy":
            self.loss = loss.CrossEntropy(self.loss)
        elif self.loss == "mse":
            self.loss = loss.MeanSquaredError(self.loss)
        elif self.loss == "binary_cross_entropy":
            self.loss = loss.BinaryCrossEntropy(self.loss)
        else:
            self.loss = None
            raise ValueError("No loss function has been provided in model.finish(), aborting")

    def init_learning_rate(self):
        if not self.lr:
            raise ValueError("No learning rate has been provided in model.finish(), aborting")

    def init_optimizer(self, optimizer_name, weights_else_bias):
        if weights_else_bias:
            layer_shapes = [layer.weights.shape if layer.has_parameters else None for layer in self.layers]
        else:
            layer_shapes = [layer.bias.shape if layer.has_parameters and layer.use_bias
                            else None for layer in self.layers]
        if optimizer_name == "adam":
            optimizer = optimizers.Adam(optimizer_name, layer_shapes, lr=self.lr)
        elif optimizer_name == "adagrad":
            optimizer = optimizers.AdaGrad(optimizer_name, layer_shapes, lr=self.lr)
        elif optimizer_name == "rmsprop":
            optimizer = optimizers.RMSProp(optimizer_name, layer_shapes, lr=self.lr)
        else:
            optimizer = optimizers.NotAnOptimizer("no_optimizer", layer_shapes, lr=self.lr)
        return optimizer

    def load_model_weights(self, name=None):
        # for the model weights to load properly the model needs to be built exactly like the one that was saved
        file_manager.ModelWeightsLoader.load(self, name)

    def save_model_weights(self, name=None):
        file_manager.ModelWeightsSaver.save(self, name)
