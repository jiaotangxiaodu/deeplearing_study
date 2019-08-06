import numpy as np
import h5py
import matplotlib.pyplot as plt
import testCases
from dnn_utils import sigmoid, sigmoid_backward, relu, relu_backward
from layers import *
import lr_utils


class MultiLayersNN:

    def __init__(self, layer_dims, std_scalar=0.01):
        np.random.randn(1)
        assert len(layer_dims) > 1
        self.L = len(layer_dims) - 1
        self.layers = []
        # hidden layers
        for i in range(1, self.L):
            W = std_scalar * np.random.randn(layer_dims[i], layer_dims[i - 1])
            b = np.zeros((layer_dims[i], 1), dtype='float')
            affine = Affine(W, b)
            activation = Relu()

            self.layers.append(affine)
            self.layers.append(activation)

        # last layer
        W = std_scalar * np.random.randn(layer_dims[self.L], layer_dims[self.L - 1])
        b = np.zeros((layer_dims[self.L], 1), dtype='float')
        affine = Affine(W, b)
        activation = Sigmoid()

        self.layers.append(affine)
        self.layers.append(activation)

        assert len(self.layers) == self.L * 2

        # cost layer
        self.cost_layer = LogCost()

    def fit(self, train_x, train_y, iter_num=3000, lr=0.0075):
        for n in range(iter_num):
            A = train_x
            for layer in self.layers:
                A = layer.forward(A)
            if n % 1 == 0:
                cost = self.cost_layer.forward(A, train_y)
                print('iter=%i,cost=%f' % (n, cost))
            dY = self.cost_layer.backward(A, train_y)
            for i in reversed(range(self.L)):
                activation_layer = self.layers[2 * i + 1]
                affine_layer = self.layers[2 * i]
                dY = activation_layer.backward(dY)
                dY = affine_layer.backward_and_update_params(dY,lr)

    def predict(self, X):
        A = X
        for layer in self.layers:
            A = layer.forward(A)
        return A

    def accuracy(self, X, Y):
        Y_hat = self.predict(X)
        m = Y.shape[1]
        correct = np.sum(np.array(Y_hat>0.5,dtype='int') == Y, dtype='float')
        return correct / m


if __name__ == '__main__':
    train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = lr_utils.load_dataset()

    train_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
    test_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T

    train_x = train_x_flatten / 255
    train_y = train_set_y
    test_x = test_x_flatten / 255
    test_y = test_set_y

    assert train_x.shape == (12288, 209)
    assert train_y.shape == (1, 209)
    assert test_x.shape == (12288, 50)
    assert test_y.shape == (1, 50)

    n_x = train_x.shape[0]
    n_h = 7
    n_y = train_y.shape[0]

    model = MultiLayersNN((n_x, n_h, n_y))
    model.fit(train_x, train_y,iter_num=500)
    train_accuracy = model.accuracy(train_x, train_y)
    test_accuracy = model.accuracy(test_x, test_y)
    print('train_accuracy=%f'%train_accuracy)
    print('test_accuracy=%f'%test_accuracy)
