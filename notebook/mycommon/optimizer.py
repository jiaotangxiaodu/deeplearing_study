import numpy as np

class GradientDescent:
    def update(self, grads, params):
        pass

class SGD(GradientDescent):
    def __init__(self, lr=1e-2):
        self.lr = lr

    def update(self, grads, params):
        for key in params.keys():
            params[key] += - self.lr * grads[key]


class Momentum(GradientDescent):
    def __init__(self, lr=1e-2, momentum=0.9):
        self.lr = lr
        self.momentum = momentum
        self.v = None

    def update(self, grads, params):
        if self.v is None:
            self.v = {}
            for key, val in params.items():
                self.v[key] = np.zeros_like(val)

        for key in params.keys():
            old_v_key = self.v[key]
            self.v[key] = old_v_key * self.momentum - grads[key] * self.lr
            params[key] += self.v[key]


class AdaGrad(GradientDescent):
    epsilon = 1e-7

    def __init__(self, lr=1e-2):
        self.lr = lr
        self.h = None

    def update(self, grads, params):
        if self.h is None:
            self.h = {}
            for key, val in params.items():
                self.h[key] = np.zeros_like(val)
        for key in params.keys():
            h_key = self.h[key]
            h_key += grads[key] * grads[key]
            self.h[key] = h_key
            params[key] -= self.lr * (1 / (np.sqrt(h_key) + self.epsilon)) * grads[key]


class Adam(GradientDescent):
    epsilon = 1e-7

    def __init__(self, lr=1e-3, beta1=0.9, beta2=0.999):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.m = None
        self.v = None
        self.iter = 0

    def update(self, grads, params):

        if self.m is None or self.v is None:
            self.m = {}
            self.v = {}
            for key, value in params.items():
                self.m[key] = np.zeros_like(value)
                self.v[key] = np.zeros_like(value)
        self.iter += 1
        lr_t = self.lr * np.sqrt(1. - self.beta2 ** self.iter) / (1. - self.beta1 ** self.iter)
        for key in params.keys():
            self.m[key] += (1 - self.beta1) * (grads[key] - self.m[key])
            self.v[key] += (1 - self.beta2) * (grads[key] * grads[key] - self.v[key])
            params[key] -= lr_t * self.m[key] / (np.sqrt(self.v[key] + self.epsilon))
