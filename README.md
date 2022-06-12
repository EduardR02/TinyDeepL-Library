# Tiny Deep Learning Library

Tiny Deep Learning Library is a Python project in which I wrote the basic functionalities of a Neural Network in Python.  
#### This Project has:
* Layers
    * Fully Connected
    * Dropout
* Omptimizers
    * Adam
* Activation Functions
    * ReLu
    * Sigmoid
    * Leaky ReLu
    * Softmax
* Loss
    * Cross Entropy
    * Binary Cross Entropy
    * MSE
* Vectorized mini-batches (using numpy)
* Saving and loading models
* Preloaded mnist and cifar10 datasets

This project does not use GPU.

## Why this is useful to me

Because layers are very modular, each derivative and implementation has to exist on its own, being complete on its own.
Everything had to work like legos, which is why I had to understand each part better. A good example where this helped a
lot is the usual softmax - cross-entropy output layer. Often books and explanations only show the combined result, but for me, I just had to understand how they work separately. Even if some combinations do not make sense (like relu and cross-entropy), I still wanted it to be possible for it to work (even if the results are useless).  
After my previous NN from scratch project, which I did the opposite way (because I was still learning the basics), meaning very rigid, and therefore less was possible, I knew that I still had some knowledge gaps because of this, which is why this project exists.

## Installation

To play around with this project clone it with git:

```bash
git clone https://github.com/EduardR02/TinyDeepL-Library.git
```
Furthermore you will need to install the requirements by running this in your python environment in the project directory:

```bash
pip3 install -r requirements.txt
```

## Usage

Simple single hidden layered neural network tutorial (something similar can be found in main.py)
The model class and its methods are inspired by the Keras sequential model API.

```python
import model
import layers

# initialize model, give your model and appropriate name,
# it will be the default when saving
my_model = model.Model("Single Hidden Layer NN")

# add input layer (specified as input shape), and first hidden layer
# here we will have a 10 - 5 - 2 architecture
my_model.add(layers.FullyConnected(neurons=5, activation="relu",
use_bias=True, input_shape=(1, 10)))

# add output layer
# each layer can have its own activation function and decide between using or not using bias
my_model.add(layers.FullyConnected(neurons=2, activation="softmax", use_bias=False))

# add optimizer, loss and learning rate 
# (only if you want to train the model, 
# if you just want to feed-forward this step is not necessary)
my_model.finish(1e-3, "adam", "cross_entropy")
```

If you now want to train the model, you can either use your own dataset or use dataset.py:
```python
import dataset

# get mnist dataset (dimensions will obviously not fit here as mnist is 784 on input and 10 on output)
train_samples, train_labels = dataset.get_train_data_mnist()
test_samples, test_labels = dataset.get_test_data_mnist()

# train the model
my_model.train(train_samples, train_labels, batch_size=512, epochs=30, shuffle=True)

# test accuracy on test data
# batch_size does not matter here, higher is just faster
my_model.test(test_samples, test_labels, batch_size=512)

# save model
my_model.save_model_weights()
```

## Contributing
This is just a personal project of mine for learning, but still, if you find any major mistakes and would like to fix them; help is welcome.

Adding additional optmizers / layers / activation functions is easy.
Inherit the base class, add your functionality and add your new class in the model.py file where it should be initialized.

## License
[MIT](https://choosealicense.com/licenses/mit/)
