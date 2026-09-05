# ./model/Net.py

import numpy as np
from model.Linear import Linear
from model.ReLU import ReLU
from model.CrossEntropyLoss import CrossEntropyLoss

class Net:

    def __init__(self, layer_sizes):
        """
        Initialize the neural network with the given layer sizes.
        layer_sizes: List of integers specifying the number of neurons in each layer.
        """

        self.layers = []

        for i in range(len(layer_sizes) - 1):
            # Add a linear layer for each consecutive pair of layer sizes
            self.layers.append(Linear(layer_sizes[i], layer_sizes[i + 1]))
            # Add a ReLU activation after all but the last layer
            if i < len(layer_sizes) - 2:
                self.layers.append(ReLU())

        self.loss_fn = CrossEntropyLoss()

    def forward(self, X):
        """
        Perform the forward pass through the network.
        X: Input data of shape (batch_size, input_dim)
        """

        for layer in self.layers:
            X = layer.forward(X)
        
        return X

    def backward(self, dLoss):
        """
        Perform the backward pass through the network.
        dLoss: Gradient of the loss with respect to the output of the network
        """

        for layer in reversed(self.layers):
            dLoss = layer.backward(dLoss)
        
        return dLoss

    def step(self, learning_rate):
        """
        Update the parameters of the network using gradient descent
        learning_rate: Learning rate for the parameter update
        """

        for layer in self.layers:
            if isinstance(layer, Linear): # Only update parameters for Linear layers
                layer.weights -= learning_rate * layer.dW
                layer.biases -= learning_rate * layer.db

    def save(self, path):
        """
        Export the network architecture and all trainable parameters to a
        numpy .npz file, so the trained model can be loaded by external programs.

        File layout:
            layer_sizes : int array, e.g. [784, 128, 64, 10]
            W{i}, b{i}  : weights (input_dim, output_dim) and biases
                          (1, output_dim) of the i-th Linear layer (i = 0, 1, ...)
        """
        linear_layers = [layer for layer in self.layers if isinstance(layer, Linear)]
        layer_sizes = [linear_layers[0].input_dim] + \
                      [layer.output_dim for layer in linear_layers]

        arrays = {}
        for i, layer in enumerate(linear_layers):
            arrays[f"W{i}"] = layer.weights
            arrays[f"b{i}"] = layer.biases

        np.savez(path, layer_sizes=np.array(layer_sizes), **arrays)

    @classmethod
    def load(cls, path):
        """
        Rebuild a Net from a parameter file previously written by save().
        The loaded parameters are writable, so training can be resumed.
        """
        with np.load(path) as data:
            layer_sizes = [int(size) for size in data["layer_sizes"]]
            net = cls(layer_sizes)
            linear_layers = [layer for layer in net.layers if isinstance(layer, Linear)]
            for i, layer in enumerate(linear_layers):
                weights = np.array(data[f"W{i}"])
                biases = np.array(data[f"b{i}"])
                if weights.shape != (layer.input_dim, layer.output_dim) or \
                   biases.shape != (1, layer.output_dim):
                    raise ValueError(
                        f"Shape mismatch for layer {i}: file has W{weights.shape}, "
                        f"b{biases.shape}, expected W{(layer.input_dim, layer.output_dim)}, "
                        f"b{(1, layer.output_dim)}"
                    )
                layer.weights = weights
                layer.biases = biases
        return net