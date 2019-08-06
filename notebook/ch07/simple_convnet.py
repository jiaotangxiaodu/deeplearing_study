# coding: utf-8
import sys, os
sys.path.append(os.pardir)  # 为了导入父目录的文件而进行的设定
import pickle
import numpy as np
from collections import OrderedDict
from common.layers import *
from common.gradient import numerical_gradient

class SimpleConvent:

    def __init__(self,input_dim=(1,28,28),
                 conv_param={'filter_num':30,'filter_size':5,'pad':0,'stride':1},
                 hidden_size=100,output_size=10,weight_init_std=0.01):

        ## 处理超参数
        S = stride = conv_param['stride']
        P = pad = conv_param['pad']

        ## 生成初始参数
        FN = conv_param['filter_num']
        C = input_dim[0]
        FH =  conv_param['filter_size']
        FW = conv_param['filter_size']

        W = input_dim[1]
        H = input_dim[2]
        OH = (H + 2*P -FH)/S + 1
        OW = (W + 2*P -FW)/S + 1

        '''池化步幅'''
        PS = 2
        POW = int(OW / PS)
        POH = int(OH / PS)
        pool_output_size = FN * POW * POH

        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(FN, C, FH, FW)
        self.params['b1'] = weight_init_std * np.random.randn(FN)
        self.params['W2'] = weight_init_std * np.random.randn(pool_output_size,hidden_size)
        self.params['b2'] = weight_init_std * np.random.randn(hidden_size)
        self.params['W3'] = weight_init_std * np.random.randn(hidden_size,output_size)
        self.params['b3'] = weight_init_std * np.random.randn(output_size)

        pool_pad = 0

        self.layers = OrderedDict()
        self.layers['Conv1'] = Convolution(self.params['W1'],self.params['b1'],stride,pad)
        self.layers['Relu1'] = Relu()
        self.layers['Pool1'] = Pooling(PS,PS,PS,pool_pad)
        self.layers['Affine1'] = Affine(self.params['W2'],self.params['b2'])
        self.layers['Relu2'] = Relu()
        self.layers['Affine2'] = Affine(self.params['W3'],self.params['b3'])
        self.last_layer = SoftmaxWithLoss()

    def predict(self,x):
        for layer in self.layers.values():
            x = layer.forward(x)
        return x

    def loss(self,x,t):
        y = self.predict(x)
        return self.last_layer.forward(y,t)

    def accuracy(self, x, t, batch_size=100):
        if t.ndim != 1: t = np.argmax(t, axis=1)

        acc = 0.0

        for i in range(int(x.shape[0] / batch_size)):
            tx = x[i * batch_size:(i + 1) * batch_size]
            tt = t[i * batch_size:(i + 1) * batch_size]
            y = self.predict(tx)
            y = np.argmax(y, axis=1)
            acc += np.sum(y == tt)

        return acc / x.shape[0]

    def gradient(self,x,t):
        self.loss(x, t)
        dout = 1
        dout = self.last_layer.backward(dout)

        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)
        grads = {}
        grads['W1'],grads['b1'] = self.layers['Conv1'].dW,self.layers['Conv1'].db
        grads['W2'],grads['b2'] = self.layers['Affine1'].dW,self.layers['Affine1'].db
        grads['W3'],grads['b3'] = self.layers['Affine2'].dW,self.layers['Affine2'].db
        return grads

if __name__ == '__main__':
    from dataset.mnist import load_mnist
    import matplotlib.pyplot as plt
    # 读入数据
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True,flatten=False)

    network = SimpleConvent()

    iters_num = 10000  # 适当设定循环的次数
    train_size = x_train.shape[0]
    batch_size = 100
    learning_rate = 0.1

    train_loss_list = []
    train_acc_list = []
    test_acc_list = []

    iter_per_epoch = max(train_size / batch_size, 1)

    for i in range(iters_num):
        batch_mask = np.random.choice(train_size, batch_size)
        x_batch = x_train[batch_mask]
        t_batch = t_train[batch_mask]

        # 计算梯度
        # grad = network.numerical_gradient(x_batch, t_batch)
        grad = network.gradient(x_batch, t_batch)

        # 更新参数
        for key in ('W1', 'b1', 'W2', 'b2','W3', 'b3'):
            network.params[key] -= learning_rate * grad[key]

        loss = network.loss(x_batch, t_batch)
        train_loss_list.append(loss)

        if i % iter_per_epoch == 0:
            train_acc = network.accuracy(x_train, t_train)
            test_acc = network.accuracy(x_test, t_test)
            train_acc_list.append(train_acc)
            test_acc_list.append(test_acc)
            print("train acc, test acc | " + str(train_acc) + ", " + str(test_acc))

    # 绘制图形
    markers = {'train': 'o', 'test': 's'}
    x = np.arange(len(train_acc_list))
    plt.plot(x, train_acc_list, label='train acc')
    plt.plot(x, test_acc_list, label='test acc', linestyle='--')
    plt.xlabel("epochs")
    plt.ylabel("accuracy")
    plt.ylim(0, 1.0)
    plt.legend(loc='lower right')
    plt.show()