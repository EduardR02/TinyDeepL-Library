import numpy as np

import model as model_file
import dataset
import layers


if __name__ == "__main__":
    input_size, output_size = 784, 10
    model = model_file.Model("Cool model")
    model.add(layers.FullyConnected(300, activation="relu", use_bias=True, input_shape=(1, input_size)))
    model.add(layers.FullyConnected(300, activation="relu", use_bias=True))
    model.add(layers.FullyConnected(50, activation="relu", use_bias=True))
    model.add(layers.FullyConnected(10, activation="softmax", use_bias=False))
    model.finish(1e-3, "adam", "cross_entropy")
    # model.load_model_weights()
    train_samples, train_labels = dataset.get_train_data_mnist()
    train_samples, train_labels = dataset.data_augment_mnist(train_samples, 5).reshape(-1, 784), np.concatenate([train_labels for i in range(5)], axis=0)
    test_samples, test_labels = dataset.get_test_data_mnist()

    model.train(train_samples, train_labels, batch_size=512, epochs=10,
                shuffle=True
                # , validation_samples=test_samples, validation_labels=test_labels
                )
    model.test(test_samples, test_labels, batch_size=512)
    model.save_model_weights()
