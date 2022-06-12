import model
import dataset
import layers
import utils
import numpy as np
import tests


if __name__ == "__main__":
    input_size, output_size = 784, 10
    model = model.Model("Cool model")
    model.add(layers.FullyConnected(800, activation="relu", use_bias=True, input_shape=(1, input_size)))
    model.add(layers.FullyConnected(50, activation="relu", use_bias=True))
    model.add(layers.FullyConnected(10, activation="softmax", use_bias=False))
    model.finish(1e-3, "adam", "cross_entropy")
    # model.load_model_weights()
    train_samples, train_labels = dataset.get_train_data_mnist()
    test_samples, test_labels = dataset.get_test_data_mnist()

    model.train(train_samples, train_labels, batch_size=512, epochs=30,
                shuffle=True
                # , validation_samples=test_samples, validation_labels=test_labels
                )
    model.test(test_samples, test_labels, batch_size=512)
    model.save_model_weights()
