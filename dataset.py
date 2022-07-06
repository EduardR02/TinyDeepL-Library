import mnist
import utils
from cifar10_web import cifar10
from matplotlib import pyplot as plt
import cv2
import numpy as np


def get_train_data_mnist():
    train_images = mnist.train_images()
    train_labels = utils.to_categorical(mnist.train_labels())
    train_images = train_images.reshape((-1, 784))
    train_images = ((train_images / 255.0) - 0.5) * 2
    return train_images, train_labels


def get_test_data_mnist():
    test_images = mnist.test_images()
    test_labels = utils.to_categorical(mnist.test_labels())
    test_images = test_images.reshape((-1, 784))
    test_images = ((test_images / 255.0) - 0.5) * 2
    return test_images, test_labels


def get_train_data_cifar10():
    # already preprocessed
    train_images, train_labels, test_images, test_labels = cifar10(path="C:/Users/Eduard/data/cifar10")
    train_images = train_images.astype("float64")
    print(train_images.shape, train_labels.shape, train_images[0], train_labels[0])
    return train_images, train_labels


def get_test_data_cifar10():
    train_images, train_labels, test_images, test_labels = cifar10(path="C:/Users/Eduard/data/cifar10")
    test_images = test_images.astype("float64")
    return test_images, test_labels


def data_augment_mnist(data, offset):
    # assume square image
    if len(data.shape) < 3:
        side_len = int(data.shape[-1]**0.5)
        data = data.reshape((-1, side_len, side_len))
    original_data = data
    data = np.concatenate((data, shift_data(original_data, offset, 0)), axis=0)
    data = np.concatenate((data, shift_data(original_data, -offset, 0)), axis=0)
    data = np.concatenate((data, shift_data(original_data, 0, offset)), axis=0)
    data = np.concatenate((data, shift_data(original_data, 0, -offset)), axis=0)
    print(data.shape)
    return data


def shift_data(data, shift_x, shift_y):
    new_data = np.copy(data)
    if shift_x > 0:
        new_data[:,:, shift_x:] = data[:,:, :-shift_x]
    elif shift_x < 0:
        new_data[:,:, :shift_x] = data[:, :, shift_x * -1:]
    elif shift_y > 0:
        new_data[:,shift_y:, :] = data[:,:-shift_y, :]
    else:
        new_data[:,:shift_y, :] = data[:, shift_y * -1:, :]
    return new_data


