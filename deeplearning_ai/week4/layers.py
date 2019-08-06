import numpy as np


class AbstractLayer:
    def forward(self, X):
        pass

    def backward(self, dY):
        pass


class CostLayer:
    def forward(self, Y_hat, Y):
        pass

    def backward(self, Y_hat, Y):
        pass


class Relu(AbstractLayer):
    def __init__(self):
        self.mask = None

    def forward(self, X):
        self.mask = (X <= 0)
        out = X.copy()
        out[self.mask] = 0
        return out

    def backward(self, dY):
        dX = dY
        dX[self.mask] = 0
        return dX


class Sigmoid(AbstractLayer):
    def __init__(self):
        self.Y = None

    def forward(self, X):
        self.Y = 1. / (1. + np.exp(-X))
        return self.Y

    def backward(self, dY):
        return dY * (1 - self.Y) * self.Y


class Affine(AbstractLayer):
    def __init__(self, W, b):
        self.W = W
        self.b = b
        self.Y = None
        self.X = None
        self.m = None

    def forward(self, X):
        self.X = X
        self.m = X.shape[1]
        self.Y = np.dot(self.W, X) + self.b
        return self.Y

    def backward(self, dY):
        dW = np.dot(dY, self.X.T)
        db = np.sum(dY, axis=1, keepdims=True)
        dX = np.dot(self.W.T, dY)
        return dW, db, dX

    def backward_and_update_params(self, dY, lr=0.01):
        dW, db, dX = self.backward(dY)
        self.W = self.W - lr * dW
        self.b = self.b - lr * db
        dY = dX
        return dY


class LogCost(CostLayer):

    def forward(self, Y_hat, Y):
        m = Y.shape[1]
        return -(1. / m) * np.sum(Y * np.log(Y_hat) + (1. - Y) * np.log(1. - Y_hat))

    def backward(self, Y_hat, Y):
        m = Y.shape[1]
        return -(1./m)*(Y / Y_hat - (1. - Y) / (1. - Y_hat))
