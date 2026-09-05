# ./model/ReLU.py

import numpy as np

class ReLU:

    def forward(self, X):
        """
        Perform the forward pass of the ReLU activation function.
        X: Input data of any shape
        A: Output data of the same shape as X
        """
        self.X = X
        self.A = np.maximum(0, X)
        return self.A

    def backward(self, dA):
        """
        Perform the backward pass of the ReLU activation function.
        dA: Gradient of the loss with respect to the output A, same shape as A
        dX: Gradient of the loss with respect to the input X, same shape as X
        """
        dX = dA * (self.X > 0)
        return dX