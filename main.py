import numpy as np
import matplotlib.pyplot as plt
import model as model_file
import dataset
import layers


def test():
    input_size, output_size = 784, 10
    model = model_file.Model("Cool model")
    model.add(layers.FullyConnected(800, activation="relu", use_bias=True, input_shape=(1, input_size)))
    model.add(layers.FullyConnected(300, activation="relu", use_bias=True))
    model.add(layers.FullyConnected(300, activation="relu", use_bias=True))
    model.add(layers.FullyConnected(10, activation="softmax", use_bias=False))
    model.finish(2e-4, "adam", "cross_entropy")
    model.load_model_weights()
    train_samples, train_labels = dataset.get_train_data_mnist()
    train_samples, train_labels = dataset.data_augment_mnist(train_samples, 1, 1).reshape(-1, 784), np.concatenate(
        [train_labels for i in range(5)], axis=0)
    test_samples, test_labels = dataset.get_test_data_mnist()
    print(test_samples.shape, test_labels.shape)

    model.train(train_samples, train_labels, batch_size=512, epochs=10,
                shuffle=True
                , validation_samples=test_samples, validation_labels=test_labels
                )
    model.test(test_samples, test_labels, batch_size=512)
    model.save_model_weights()


def test_addition():
    # add two numbers by neural network, 2 inputs 1 output neuron
    input_size, output_size = 2, 1
    model = model_file.Model()
    model.add(layers.FullyConnected(output_size, activation="linear", use_bias=True, input_shape=(1, input_size)))
    model.finish(2, "adam", "mse")
    print(model.layers[0].get_parameters())
    # create training data
    summands = np.random.randint(low=-2000, high=2001, size=(100000, 2))
    sums = np.sum(summands, axis=-1).reshape(-1, 1) - 5
    data_split = summands.shape[0] // 10
    train_data, train_labels = summands[:-data_split], sums[:-data_split]
    test_data, test_labels = summands[-data_split:], sums[-data_split:]
    print(train_data.shape, train_labels.shape, test_data.shape, test_labels.shape)
    model.train(train_data, train_labels, batch_size=512, epochs=2, shuffle=False)
    model.test(test_data, test_labels, batch_size=512)
    print(model.layers[0].get_parameters())     # weights should both be 1 if everything works
    print_test = np.random.randint(low=-2000, high=2001, size=(10, 2))
    print(print_test)
    print(model.predict(print_test))


def show_mnist_loss_outliers():
    input_size, output_size = 784, 10
    model = model_file.Model("Show outliers")
    model.add(layers.FullyConnected(400, activation="relu", use_bias=True, input_shape=(1, input_size)))
    model.add(layers.FullyConnected(300, activation="relu", use_bias=True))
    model.add(layers.FullyConnected(100, activation="relu", use_bias=True))
    model.add(layers.FullyConnected(10, activation="softmax", use_bias=False))
    model.finish(2e-4, "adam", "cross_entropy")
    train_samples, train_labels = dataset.get_train_data_mnist()
    """train_samples, train_labels = dataset.data_augment_mnist(train_samples, 1, 1).reshape(-1, 784), np.concatenate(
        [train_labels for i in range(5)], axis=0)"""
    test_samples, test_labels = dataset.get_test_data_mnist()
    """model.train(train_samples, train_labels, batch_size=512, epochs=15,
                shuffle=True
                , validation_samples=test_samples, validation_labels=test_labels
                )"""
    model.load_model_weights("Show outliers")
    used_samples, used_labels = train_samples, train_labels
    outputs = model.predict(used_samples)
    # sort by loss, but also keep the index
    calc_loss = lambda x, y: -np.sum(x * np.log(y + 1e-10))
    # to look at smallest loss reverse sort order (remove minus sign in lambda)
    sorted_outputs = sorted([(i, calc_loss(outputs[i], used_labels[i])) for i in range(len(outputs))],
                            key=lambda x: -x[1])
    # plot the first 10 images with the highest loss, display the loss, the label, the prediction, and the output vector
    for i in range(10):
        index, loss = sorted_outputs[i]
        label = np.argmax(used_labels[index])
        prediction = np.argmax(outputs[index])
        output_vector_rounded = np.around(outputs[index], decimals=2)

        plt.subplot(2, 5, i + 1)
        plt.imshow(used_samples[index].reshape(28, 28), cmap="gray")
        plt.title(f"Loss: {loss:.2f}\nLabel: {label}\nPrediction: {prediction}")

        # Display the output vector as text
        counting_vector = np.arange(10)
        counting_vector_formatted = ' '.join(format(x, '.2f') for x in counting_vector)
        plt.xlabel(f"Output Vector:\n{output_vector_rounded}\n{counting_vector_formatted}")

    # plt.tight_layout()
    plt.show()


def max_loss_test():
    label = np.zeros((1, 10))
    output = np.zeros((1, 10))
    label[0, 0] = 1
    output[0, 2] = 1
    loss = -np.sum(label * np.log(output + 1e-10))
    print(loss)


if __name__ == "__main__":
    # test()
    show_mnist_loss_outliers()
