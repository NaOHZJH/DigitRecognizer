# ./model/Softmax.py

import numpy as np

class Softmax:

    def forward(self, X):
        """
        Perform the forward pass of the Softmax activation function.
        X: Input data of shape (batch_size, num_classes)
        A: Output data of the same shape as X
        """
        self.X = X
        exp_X = np.exp(X - np.max(X, axis=1, keepdims=True))  # For numerical stability
        self.A = exp_X / np.sum(exp_X, axis=1, keepdims=True)
        return self.A

    def backward(self, dA):
        """
        Perform the backward pass of the Softmax activation function.
        dA: Gradient of the loss with respect to the output A, shape (batch_size, num_classes)
        dX: Gradient of the loss with respect to the input X, shape (batch_size, num_classes)
        """
        batch_size = self.A.shape[0]
        dX = np.zeros_like(dA)

        for i in range(batch_size):
            a = self.A[i].reshape(-1, 1)
            jacobian_matrix = np.diagflat(a) - np.dot(a, a.T)
            dX[i] = np.dot(jacobian_matrix, dA[i])

        return dX