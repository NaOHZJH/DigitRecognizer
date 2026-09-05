# ./model/CrossEntropyLoss.py

import numpy as np

class CrossEntropyLoss:

    def forward(self, X, Y):
        """
        Compute the forward pass of the cross-entropy loss.
        X: Input data (logits) of shape (batch_size, num_classes)
        Y: True labels (one-hot encoded) of shape (batch_size, num_classes)
        """

        exp_X = np.exp(X - np.max(X, axis=1, keepdims=True))  # For numerical stability
        self.A = exp_X / np.sum(exp_X, axis=1, keepdims=True)

        N = X.shape[0]
        self.Y = Y
        self.loss = -np.sum(Y * np.log(self.A + 1e-15)) / N  # Add small value to avoid log(0)
        return self.loss

    def backward(self):
        """
        Compute the backward pass of the cross-entropy loss.
        Returns the gradient of the loss with respect to the input X.
        """
        N = self.Y.shape[0]
        dX = (self.A - self.Y) / N
        return dX