# ./model/Linear.py

import numpy as np

class Linear:

    def __init__(self, input_dim, output_dim):
        """
        Initialize the linear layer with random weights and zero biases.
        input_dim: Number of input features
        output_dim: Number of output features
        self.weights: Weight matrix of shape (input_dim, output_dim)
        self.biases: Bias vector of shape (1, output_dim)
        """

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.weights = np.random.randn(input_dim, output_dim) * 0.01
        self.biases = np.zeros((1, output_dim))

        self.dW = np.zeros_like(self.weights)
        self.db = np.zeros_like(self.biases)

    def forward(self, X):
        """
        Perform the forward pass of the linear layer.
        X: Input data of shape (batch_size, input_dim)
        Z: Output data of shape (batch_size, output_dim)
        """
        self.X = X
        self.Z = np.dot(X, self.weights) + self.biases
        return self.Z

    def backward(self, dZ):
        """
        Perform the backward pass of the linear layer.
        dZ: Gradient of the loss with respect to the output Z, shape (batch_size, output_dim)
        dX: Gradient of the loss with respect to the input X, shape (batch_size, input_dim)
        """
        # Compute gradients. dZ already carries the 1/N mean-loss scaling from the
        # loss layer, so no additional division by the batch size is applied here.
        self.dW = np.dot(self.X.T, dZ)
        self.db = np.sum(dZ, axis=0, keepdims=True)
        dX = np.dot(dZ, self.weights.T)

        return dX