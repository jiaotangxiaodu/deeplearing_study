import numpy as np
import matplotlib.pyplot as plt
import h5py
import lr_utils


def sigmoid(z):
    return 1. / (1. + np.exp(-z))


def initParam(input_size, output_size, weight_init_std=0.01):
    w = weight_init_std * np.random.randn(input_size, output_size)
    b = np.zeros((1, output_size), dtype='float')
    return w, b


def propagate(w, b, X, Y):
    m = X.shape[1]

    # 正向传播
    Z = np.dot(w.T, X) + b
    A = sigmoid(Z)
    cost = (-1. / m) * np.sum(Y * np.log(A) + (1 - Y) * (np.log(1 - A)))

    # 反向传播
    dw = (1. / m) * np.dot(X, (A - Y).T)
    db = (1. / m) * np.sum(A - Y)

    grads = {"dw": dw, "db": db}
    cost = np.squeeze(cost)
    return (grads, cost)

def optimize(w,b,X,Y,num_iter,lr,print_cost = False):
    costs = []

    for i in range(num_iter):
        grads , cost = propagate(w,b,X,Y)
        w -= grads['dw'] * lr
        b -= grads['db'] * lr

        if i % 100 == 0:
            costs.append(cost)
            if print_cost:
                print(cost)
    params = {'w':w,'b':b}
    return params,costs

def predict(w,b,X):
    A = np.dot(w.T,X) + b
    return np.array(A > 0,dtype='int')

if __name__ == '__main__':
    train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes = lr_utils.load_dataset()

    train_set_x_flatten = train_set_x_orig.reshape((train_set_x_orig.shape[0], -1)).T
    test_set_x_flatten = test_set_x_orig.reshape((test_set_x_orig.shape[0], -1)).T

    train_set_x = train_set_x_flatten / 255.
    test_set_x = test_set_x_flatten / 255.

    w, b = initParam(train_set_x.shape[0], 1)
    params,costs = optimize(w,b,train_set_x,train_set_y_orig,10000,0.01,True)
    plt.plot(costs)
    plt.show()

    print('-----')

    Y_predict_train = predict(w,b,train_set_x)
    train_accuracy = np.sum(Y_predict_train == train_set_y_orig) * 1. / np.sum(np.ones(train_set_y_orig.shape,dtype='float'))
    print("train accuracy:%s"%train_accuracy)

    Y_predict_test = predict(w, b, test_set_x)
    test_accuracy = np.sum(Y_predict_test == test_set_y_orig) * 1. / np.sum(np.ones(test_set_y_orig.shape, dtype='float'))
    print("test accuracy:%s" % test_accuracy)

