import numpy as np
from collections import OrderedDict
from layers import Affine, Sigmoid, Relu, SoftmaxWithLoss

'''
全连接的多层神经网络
'''


class MulLayerNet:
    """
    Parameters
    ----------
    input_size : 输入大小（MNIST的情况下为784）
    hidden_size_list : 隐藏层的神经元数量的列表（e.g. [100, 100, 100]）
    output_size : 输出大小（MNIST的情况下为10）
    activation : 'relu' or 'sigmoid'
    weight_init_std : 指定权重的标准差（e.g. 0.01）
        指定'relu'或'he'的情况下设定“He的初始值”
        指定'sigmoid'或'xavier'的情况下设定“Xavier的初始值”
    weight_decay_lambda : Weight Decay（L2范数）的强度
    """

    def __init__(self, input_size, output_size, hidden_size_list, activation='relu',
                 weight_init_std='relu',
                 weight_decay_lambda=0):
        self.input_size = input_size
        self.hidden_size_list = hidden_size_list
        self.output_size = output_size
        self.activation = activation
        self.weight_init_std = weight_init_std
        self.weight_decay_lambda = weight_decay_lambda

        self._init_weight()
        self._init_layers()

    def _init_weight(self):
        self.params = {}
        all_layers = [self.input_size] + self.hidden_size_list + [self.output_size]
        weight_init_std = self.weight_init_std
        for i in range(1, len(self.hidden_size_list)+2):
            if weight_init_std == 'relu' or weight_init_std == 'he':
                scalar = np.sqrt(2 / all_layers[i - 1])
            elif weight_init_std == 'sigmoid' or weight_init_std == 'xavier':
                scalar = np.sqrt(1 / all_layers[i - 1])
            else:
                scalar = weight_init_std

            self.params['W%d' % i] = np.random.randn(all_layers[i - 1], all_layers[i]) * scalar
            self.params['b%d' % i] = np.zeros(all_layers[i], dtype='float')

    def _init_layers(self):
        self.layers = OrderedDict()
        all_layers = [self.input_size] + self.hidden_size_list + [self.output_size]
        activation_dict = {'relu': Relu, 'sigmoid': Sigmoid}
        for i in range(1, len(self.hidden_size_list)+2):
            self.layers['Affine%d' % i] = Affine(self.params['W%d' % i], self.params['b%d' % i])
            self.layers['Activation%d' % i] = activation_dict[self.activation]()

        self.last_layers = SoftmaxWithLoss()

    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)
        return x

    def loss(self, x, t):
        y = self.predict(x)
        weight_decay = 0

        for param_index in range(1, len(self.hidden_size_list) + 2):
            param = self.params['W%d' % param_index]
            weight_decay += .5 * self.weight_decay_lambda * np.sum(param ** 2)

        return self.last_layers.forward(y, t) + weight_decay

    def accuracy(self, x, t):
        p = self.predict(x)
        y = np.argmax(p, axis=1)

        if t.ndim != 1:
            t = np.argmax(t, axis=1)
        return np.sum(y == t,dtype='float') / float(y.shape[0])

    '''
    求损失函数关于参数的梯度
    '''

    def gradient(self, x, t):
        self.loss(x, t)
        dout = 1.
        dout = self.last_layers.backward(dout)

        layers = list(self.layers.values())
        layers.reverse()

        for layer in layers:
            dout = layer.backward(dout)

        grads = {}
        for i in range(1, len(self.hidden_size_list) + 2):
            grads['W%d' % i] = self.layers['Affine%d' % i].dW + self.weight_decay_lambda*self.layers['Affine%d'%i].W
            grads['b%d' % i] = self.layers['Affine%d' % i].db
        return grads


if __name__ == '__main__':
    from dataset.mnist import load_mnist
    from common.optimizer import SGD,Momentum,AdaGrad,Adam
    from common.multi_layer_net import MultiLayerNet
    (x_train,t_train),(x_test,t_test) = load_mnist(normalize=True)
    multi_layer_net = MultiLayerNet(784,output_size=10, hidden_size_list=[100,100,100,100])
    mul_layer_net = MulLayerNet(784,output_size=10, hidden_size_list=[100,100,100,100])

    max_iterations = 2000
    train_size = x_train.shape[0]
    batch_size = 128
    optimizer_mul = AdaGrad()
    optimizer_multi = AdaGrad()

    for i in range(max_iterations):
        batch_mask = np.random.choice(train_size, batch_size)
        x_batch = x_train[batch_mask]
        t_batch = t_train[batch_mask]

        grads_mul = mul_layer_net.gradient(x_batch, t_batch)
        grads_multi = multi_layer_net.gradient(x_batch, t_batch)

        optimizer_mul.update(mul_layer_net.params, grads_mul)
        optimizer_multi.update(multi_layer_net.params, grads_multi)

        loss_mul = mul_layer_net.loss(x_batch, t_batch)
        loss_multi = multi_layer_net.loss(x_batch, t_batch)


        if i % 100 == 0:
            print("===========" + "iteration:" + str(i) + "===========")
            loss_mul = mul_layer_net.loss(x_batch, t_batch)
            loss_multi = multi_layer_net.loss(x_batch, t_batch)
            print('mul-loss' + ":" + str(loss_mul))
            print('multi-loss' + ":" + str(loss_multi))


    # optimizer_function = Adam()
    # batch_size = 128
    # for i in range(2000):
    #     batch_index = np.random.choice(x_train.shape[0],batch_size)
    #     x_batch = x_train[batch_idex]
    #     t_batch = t_train[batch_index]
    #     gradient = mul_layer_net.gradient(x_batch, t_batch)
    #     optimizer_function.update(gradient, mul_layer_net.params)
    #     if i % 100 ==0:
    #         print('itr:%d,accuracy:%f,loss:%f.'%(i,mul_layer_net.accuracy(x_train,t_train),mul_layer_net.loss(x_train,t_train)))
    # print(mul_layer_net.accuracy(x_test,t_test))



