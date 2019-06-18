import numpy as np
import matplotlib.pyplot as plt

'''
这个神经网络模型模型有2层
输入层有3个神经元
输出层有2个神经元
没有隐藏层和偏置
所以模型只有一个2*3的权重模型W1
'''

def softmax(x):
    if x.ndim == 1:
        return np.exp(x) / np.sum(np.exp(x))
    return np.array([softmax(row) for row in x])

def sigmoid(x):
    return 1. / (1. + np.exp(-x))

def gradient(f,x,h=1e-6):
    g = np.zeros_like(x,dtype='float')
    x_line = x.reshape(-1,order='c')
    g_line = g.reshape(-1,order='c')
    for i in range(len(x_line)):
        tmp = x_line[i]
        x_line[i] = tmp + h
        y_high = f(x)
        x_line[i] = tmp - h
        y_low = f(x)
        x_line[i] = tmp
        g_line[i] = (y_high - y_low)/(2*h)
    return g

class SimpleNet:
    def __init__(self):
        self.param = {}
        self.param['W1'] = np.random.randn(2,3)

    def predict(self,X):
        W1 = self.param['W1']
        if np.ndim(X) > 1:
            return np.array([self.predict(row) for row in X])
        return X.dot(W1)

    def loss(self,X,y):
        sigma = 1e-8
        out_row = self.predict(X)
        out_sigmoid = sigmoid(out_row)
        return -np.sum(np.log(out_sigmoid[:,y])) / len(y)


if __name__ == '__main__':
    X = np.array([[1,2],[1,2],[2,4],[1,1]])
    W = np.array([[1.,2.,2.],[4.,5.,5.]])
    y = np.array([0,1,1,2])
    simple_net = SimpleNet()
    simple_net.param['W1'] = W
    eta = 0.5
    plt_arr = []
    for i in range(100000):
        g = gradient(lambda w: simple_net.loss(X, y), simple_net.param['W1'])
        simple_net.param['W1'] = - eta * g + simple_net.param['W1']
        loss_val = simple_net.loss(X,y)
        plt_arr.append(loss_val)
    plt.plot(plt_arr)
    plt.show()

