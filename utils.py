import numpy as np
import time


def progressBar(current, total, epoch, acc, loss, bar_length=20):
    percent = float(current) * 100 / (total - 1)
    arrow = '=' * int(percent / 100 * bar_length - 1) + '>'
    spaces = ' ' * (bar_length - len(arrow))

    print("metrics[", "acc:", round(acc, 4), "; loss:", round(loss, 4), "]",
          "Epoch:", epoch + 1, ";", 'Progress: [%s%s] %d %%' % (arrow, spaces, percent), "out of", total, "  ", end='\r',)


def time_it(function, loops):
    total_time = 0
    for i in range(loops):
        t = time.time()
        function()
        total_time += time.time() - t
    print("Function took :", total_time / loops, "seconds on average")


# check if in place, doesn't really matter tho
def shuffle_data(samples, labels):
    indices = np.arange(labels.shape[0])
    np.random.shuffle(indices)
    labels = labels[indices]
    samples = samples[indices]
    return samples, labels


# turns each element in array to a one hot vector
def to_categorical(arr, classes_amt=None):
    input_shape = arr.shape
    arr = arr.ravel()
    found_classes = np.max(arr) + 1
    num_classes = classes_amt
    if not num_classes or num_classes < found_classes:
        num_classes = found_classes
    n = arr.shape[0]
    categorical = np.zeros((n, num_classes))
    categorical[np.arange(n), arr] = 1
    output_shape = input_shape + (num_classes,)
    categorical = np.reshape(categorical, output_shape)
    return categorical


def create_minibatches(data_samples, data_labels, batch_size):
    # idx_list tells array split where to split the arrays
    idx_list = list(range(batch_size, data_labels.shape[0], batch_size))
    batch_list_samples = np.array_split(data_samples, idx_list, axis=0)
    batch_list_labels = np.array_split(data_labels, idx_list, axis=0)
    return batch_list_samples, batch_list_labels
