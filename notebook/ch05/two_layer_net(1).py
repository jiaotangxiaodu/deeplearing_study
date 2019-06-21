import numpy as np
from layer import Relu,Affine,SoftmaxWithLoss
from collections import OrderedDict

class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size,weight_init_std=0.01):
    # 初始化权重
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)
    # 生成层
        self.layers = OrderedDict()
        self.layers['Affine1'] = Affine(self.params['W1'], self.params['b1'])
        self.layers['Relu1'] = Relu()
        self.layers['Affine2'] = Affine(self.params['W2'], self.params['b2'])
        self.lastLayer = SoftmaxWithLoss()
    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)
        return x
    def predict0(self,x):
        for layer in self.layers.values():
            x = layer.forward(x)
        return np.argmax(x,axis=1)

    # x:输入数据, t:监督数据
    def loss(self, x, t):
        y = self.predict(x)
        return self.lastLayer.forward(y, t)
    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        if t.ndim != 1 : t = np.argmax(t, axis=1)
        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy
    def gradient(self, x, t):
        # forward
        self.loss(x, t)
        # backward
        dout = 1
        dout = self.lastLayer.backward(dout)
        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)
        # 设定
        grads = {}
        grads['W1'] = self.layers['Affine1'].dW
        grads['b1'] = self.layers['Affine1'].db
        grads['W2'] = self.layers['Affine2'].dW
        grads['b2'] = self.layers['Affine2'].db
        return grads

def plot_decision_boundary(model, axis):
    x0, x1 = np.meshgrid(
        np.linspace(axis[0], axis[1], int((axis[1] - axis[0]) * 100)).reshape(-1, 1),
        np.linspace(axis[2], axis[3], int((axis[3] - axis[2]) * 100)).reshape(-1, 1)
    )
    X_new = np.c_[x0.ravel(), x1.ravel()]
    y_predict = model.predict0(X_new)
    zz = y_predict.reshape(x0.shape)
    from matplotlib.colors import ListedColormap
    custom_cmap = ListedColormap(['#EF9A9A', '#FFF59D', '#90CAF9'])
    plt.contourf(x0, x1, zz, linewidth=5, cmap=custom_cmap)

if __name__ == '__main__':
    import sys, os
    sys.path.append(os.pardir)
    import numpy as np
    #from dataset.mnist import load_mnist
    import matplotlib.pyplot as plt

    # mnist
    # (x_train, t_train), (x_test, t_test) = \
    #     load_mnist(normalize=True, one_hot_label=True)
    # network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)
    # learning_rate = 0.1

    # 线性可分数据集
    np.random.seed(5)
    x_train = np.random.randn(6000,2) * .5 + .5
    t_train = np.array(x_train[:,0] + x_train[:,1] >.5,dtype='int')
    x_test = np.random.randn(1000,2) * .5 + .5
    t_test = np.array(x_test[:,0] + x_test[:,1] > .5,dtype='int')
    network = TwoLayerNet(input_size=2, hidden_size=100, output_size=2)
    learning_rate = 0.007


    iters_num = 1000
    train_size = x_train.shape[0]
    batch_size = 6000
    train_loss_list = []
    train_acc_list = []
    test_acc_list = []
    iter_per_epoch = max(train_size / batch_size, 1)
    for i in range(iters_num):
        '''
                batch_mask = np.random.choice(train_size, batch_size)
        if i < 10:
            print(batch_mask)
            print(batch_mask.shape)
        x_batch = x_train[batch_mask]
        t_batch = t_train[batch_mask]
        '''
        x_batch = x_train
        t_batch = t_train
        # 通过误差反向传播法求梯度
        grad = network.gradient(x_batch, t_batch)
        # 更新
        for key in ('W1', 'b1', 'W2', 'b2'):
            network.params[key] -= learning_rate * grad[key]
        loss = network.loss(x_batch, t_batch)
        train_loss_list.append(loss)

        if i % 2 == 0:
            loss_val = network.loss(x_train,t_train)
            print('itr:%d,loss:%f,train:%f,test:%f'%(i,loss_val,network.accuracy(x_train,t_train),network.accuracy(x_test,t_test)))
            # print('w1:%s,b1:%s,w2:%s,b2:%s'%(network.params['W1'],network.params['b1'],network.params['W2'],network.params['b2']))
            # print('---------------------------------------------')
        if network.accuracy(x_train,t_train)>0.95 and network.accuracy(x_test,t_test)>0.95:
            plot_decision_boundary(network,(-1,2,-1,2))
            plt.scatter(x_test[t_test == 0, 0], x_test[t_test == 0, 1], alpha=.5)
            plt.scatter(x_test[t_test == 1, 0], x_test[t_test == 1, 1], alpha=.5)
            plt.show()
            break



