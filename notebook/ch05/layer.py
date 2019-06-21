import numpy as np


class Relu:
    def __init__(self):
        self.x = None
        self.mask = None

    def forward(self, x):
        self.mask = (x <= 0)
        return x * (1 - self.mask)

    def backward(self, dout):
        return dout * (1 - self.mask)


class Sigmoid:
    def __init__(self):
        self.y = None

    def forward(self, x):
        y = (1 + np.exp(-x)) ** (-1)
        self.y = y
        return y

    def backward(self, d_out):
        dx = d_out * (1. - self.y) * self.y
        return dx


class Affine:
    def __init__(self, W, b):
        self.W = W
        self.b = b
        self.X = None
        self.dW = None
        self.db = None

    def forward(self, X):
        self.X = X
        return X.dot(self.W) + self.b

    def backward(self, d_out):
        dx = d_out.dot(self.W.T)
        dW = self.X.T.dot(d_out)
        db = np.sum(d_out, axis=0)
        self.dW = dW
        self.db = db
        return dx

class SoftmaxWithLoss:

    def _softmax(self,x):
        if x.ndim == 2:
            x = x.T
            x = x - np.max(x, axis=0)
            y = np.exp(x) / np.sum(np.exp(x), axis=0)
            return y.T

        x = x - np.max(x)  # 溢出对策
        return np.exp(x) / np.sum(np.exp(x))

    def _cross_entropy_error(self,y, t):
        if y.ndim == 1:
            t = t.reshape(1, t.size)
            y = y.reshape(1, y.size)

        # 监督数据是one-hot-vector的情况下，转换为正确解标签的索引
        if t.size == y.size:
            t = t.argmax(axis=1)

        batch_size = y.shape[0]
        return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size

    def __init__(self):
        self.loss = None
        self.y = None
        self.t = None

    def forward(self,x,t):
        self.t = t
        self.y = self._softmax(x)
        self.loss = self._cross_entropy_error(self.y,self.t)
        return self.loss

    def backward(self,dout=1):
        batch_size = self.t.shape[0]
        if self.y.ndim != self.t.ndim:
            t_tmp = np.zeros(self.y.shape)
            t_tmp[self.t] = 1
            self.t = t_tmp
        dx = (self.y - self.t) / batch_size
        return dx

