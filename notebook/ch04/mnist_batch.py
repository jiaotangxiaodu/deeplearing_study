import numpy as np
import sys,os
sys.path.append(os.pardir)
from dataset.mnist import load_mnist
from two_layer_net import TwoLayerNet
# from two_layer_net import gradient


(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True)
train_loss_list = []

#超参数
iters_num = 1000
train_size = len(x_train)
batch_size = 100
eta = 0.1
network = TwoLayerNet(784,50,10)

for i in range(iters_num):
    batch_mask = np.random.choice(train_size,batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    grad = network.gradient(x_batch, t_batch)

    network.param['W1'] = network.param['W1'] - eta * grad['W1']
    network.param['b1'] = network.param['b1'] - eta * grad['b1']
    network.param['W2'] = network.param['W2'] - eta * grad['W2']
    network.param['b2'] = network.param['b2'] - eta * grad['b2']

    if i % 10 == 0:
        loss_val = network._loss(x_train,t_train)
        print('itr=%s,loss=%s'%(i,loss_val))
print(network.score(x_train,t_train))
print(network.score(x_test,t_test))
print(network.param)

