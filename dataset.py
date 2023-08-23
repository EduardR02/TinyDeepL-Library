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


def data_augment_mnist(data, offset_x, offset_y):
    # assume square image
    if len(data.shape) < 3:
        side_len = int(data.shape[-1]**0.5)
        data = data.reshape((-1, side_len, side_len))
    original_data = data
    data = np.concatenate((data, shift_data(original_data, offset_x, 0)), axis=0)
    data = np.concatenate((data, shift_data(original_data, -offset_x, 0)), axis=0)
    data = np.concatenate((data, shift_data(original_data, 0, offset_y)), axis=0)
    data = np.concatenate((data, shift_data(original_data, 0, -offset_y)), axis=0)
    print(data.shape)
    return data


def squish_and_stretch_mnist(samples, labels, factor, runs_through_dataset):
    # use normal distribution to randomly squish and stretch, but altered by factor
    reshaped_samples = []

    def get_scaling_factor(distribution, j):
        # if above zero stretch, if below zero squish
        if distribution[j] > 0:
            return 1 + distribution[j] * factor, 1.0
        else:
            return 1.0, (1 - distribution[j] * factor)

    samples = samples.reshape((-1, 28, 28))
    for i in range(runs_through_dataset):
        distribution = np.random.normal(size=(samples.shape[0]))
        for j in range(samples.shape[0]):
            x_scaling, y_scaling = get_scaling_factor(distribution, j)
            # always only either stretch x or y, as after padding this translates
            # to either squishing or stretching depending on axis
            rescaled_image = cv2.resize(samples[j], None, fx=x_scaling, fy=y_scaling, interpolation=cv2.INTER_LINEAR)
            # now pad to get square image again
            if x_scaling == 1:
                # there is an off by one error here in case of odd numbers, but it doesnt matter because resize takes care of it
                padding = int(rescaled_image.shape[0] - rescaled_image.shape[1]) // 2
                # pad non altered axis with black
                rescaled_image = np.pad(rescaled_image, ((0, 0), (padding, padding)), mode="constant", constant_values=-1)
            else:
                padding = int(rescaled_image.shape[1] - rescaled_image.shape[0]) // 2
                rescaled_image = np.pad(rescaled_image, ((padding, padding), (0, 0)), mode="constant", constant_values=-1)
            # now resize to 28x28
            rescaled_image = cv2.resize(rescaled_image, (28, 28), interpolation=cv2.INTER_LINEAR)
            reshaped_samples.append(rescaled_image)
    reshaped_samples = np.stack(reshaped_samples, axis=0)
    labels = np.concatenate([labels for _ in range(runs_through_dataset)], axis=0)
    print(reshaped_samples.shape, labels.shape)
    return reshaped_samples, labels


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


