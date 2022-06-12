import mnist
import utils
from cifar10_web import cifar10


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
