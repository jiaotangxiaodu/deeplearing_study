import numpy as np
import matplotlib.pyplot as plt
from testCases import *
import sklearn
import sklearn.datasets
import sklearn.linear_model
from planar_utils import plot_decision_boundary, sigmoid, load_planar_dataset, load_extra_datasets


def initialize_parameters(n_x, n_y, n_h1, std_scaler=0.01):
    W1 = std_scaler * np.random.randn(n_h1, n_x)
    B1 = np.zeros((n_h1, 1), dtype=float)
    W2 = std_scaler * np.random.randn(n_y, n_h1)
    B2 = np.zeros((n_y, 1), dtype=float)
    param = {'W1': W1, 'B1': B1, 'W2': W2, 'B2': B2}
    return param


def forward_propagation(X, params):
    W1 = params['W1']
    B1 = params['B1']
    W2 = params['W2']
    B2 = params['B2']

    Z1 = np.dot(W1, X) + B1
    A1 = np.tanh(Z1)

    Z2 = np.dot(W2, A1) + B2
    A2 = sigmoid(Z2)

    cache = {'Z1': Z1, 'A1': A1, 'Z2': Z2, 'A2': A2}
    return A2, cache


def backward_propagation(params, cache, X, Y):
    m = X.shape[1]

    W1 = params["W1"]
    W2 = params["W2"]

    A1 = cache["A1"]
    A2 = cache["A2"]

    dZ2 = A2 - Y
    dW2 = (1 / m) * np.dot(dZ2, A1.T)
    dB2 = (1 / m) * np.sum(dZ2, axis=1, keepdims=True)
    dZ1 = np.multiply(np.dot(W2.T, dZ2), 1 - np.power(A1, 2))
    dW1 = (1 / m) * np.dot(dZ1, X.T)
    dB1 = (1 / m) * np.sum(dZ1, axis=1, keepdims=True)
    grads = {"dW1": dW1,
             "dB1": dB1,
             "dW2": dW2,
             "dB2": dB2}

    return grads


def update_params(params,grads,lr=1.2):
    W1,W2,B1,B2 = params['W1'],params['W2'],params['B1'],params['B2']
    dW1, dW2, dB1, dB2 = grads['dW1'],grads['dW2'],grads['dB1'],grads['dB2']
    W1 = W1 - lr * dW1
    W2 = W2 - lr * dW2
    B1 = B1 - lr * dB1
    B2 = B2 - lr * dB2
    return {'W1': W1, 'B1': B1, 'W2': W2, 'B2': B2}


def predict(params, X):
    A2, cache = forward_propagation(X, params)
    return A2 > 0.5
    # cache = {'Z1': Z1, 'A1': A1, 'Z2': Z2, 'A2': A2}


if __name__ == '__main__':
    np.random.seed(1)

    X, Y = load_planar_dataset()

    m = X.shape[1]
    n_x = X.shape[0]
    n_y = Y.shape[0]
    n_h1 = 4



    # assert X.shape == (n_x, m)
    # assert Y.shape == (n_y, m)
    #
    # assert W1.shape == (n_h1, n_x)
    # assert B1.shape == (n_h1, 1)
    #
    # # assert Z1.shape == (n_h1, m)
    # # assert A1.shape == Z1.shape
    #
    # assert W2.shape == (n_y, n_h1)
    # assert B2.shape == (n_y, 1)

    n_iter = 10000
    lr = 8
    params = initialize_parameters(n_x, n_y, n_h1)
    W1, B1, W2, B2 = params['W1'], params['B1'], params['W2'], params['B2']

    plt_x = []
    plt_y = []
    for i in range(n_iter):
        A2, cache = forward_propagation(X, params)
        grads = backward_propagation(params, cache, X, Y)
        params = update_params(params,grads,lr=lr)
        if i % 100 == 0:
            cost = -(1. / m) * np.sum(Y * np.log(A2) + (1 - Y) * (np.log(1 - A2)))
            plt_x.append(i)
            plt_y.append(cost)
            print('lr=%s,cost=%s'%(lr,cost))
    # plt.ylim(0.2,0.25)
    plot_decision_boundary(lambda x: predict(params, x.T), X, Y)
    plt.title("Decision Boundary for hidden layer size " + str(4))
    plt.show()

    predictions = predict(params, X)
    print('准确率: %d' % float((np.dot(Y, predictions.T) + np.dot(1 - Y, 1 - predictions.T)) / float(Y.size) * 100) + '%')