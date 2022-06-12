import numpy as np


class Tracker:
    def __init__(self):
        self.sample_amount = 0
        self.sample_sum = 0

    def update_metrics(self, output, label):
        raise NotImplementedError()

    def reset(self):
        self.sample_amount = 0
        self.sample_sum = 0

    def get_metrics(self):
        return self.sample_sum / self.sample_amount


class LossTracker(Tracker):
    def __init__(self, loss):
        super().__init__()
        self.loss = loss

    def update_metrics(self, output, label):
        # first dimension is always batch size
        self.sample_amount += label.shape[0]
        # loss function is expected to already average the values
        self.sample_sum += self.loss.loss_function(output, label)


class AccuracyTracker(Tracker):

    def update_metrics(self, output, label):
        self.sample_amount += label.shape[0]
        sum_batch_accuracy = self.compute_accuracy(output, label)
        self.sample_sum += sum_batch_accuracy

    def compute_accuracy(self, output, label):
        output = np.argmax(output, axis=-1)
        label = np.argmax(label, axis=-1)
        accuracy = np.mean(output == label) * label.shape[0]
        return accuracy
