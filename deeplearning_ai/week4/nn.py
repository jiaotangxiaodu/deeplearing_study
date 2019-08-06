import numpy as np
import h5py
import matplotlib.pyplot as plt
import testCases
from dnn_utils import sigmoid, sigmoid_backward, relu, relu_backward
import lr_utils

'''
n_x:输入数据的特征数
n_h:隐藏层的节点数
n_y:输出数据的特征数
std_scalar:参数w分布的标准差
'''


# def init_parameters(n_x, n_h, n_y, std_scalar=0.01):
#     W1 = std_scalar * np.random.randn(n_h, n_x)
#     b1 = np.zeros((n_h, 1), dtype='float')
#     W2 = std_scalar * np.random.randn(n_y, n_h)
#     b2 = np.zeros((n_y, 1), dtype='float')
#
#     assert W1.shape == (n_h, n_x)
#     assert b1.shape == (n_h, 1)
#     assert W2.shape == (n_y, n_h)
#     assert b2.shape == (n_y, 1)
#
#     return {'W1': W1, 'b1': b1, "W2": W2, "b2": b2}


def initialize_parameters_deep(layers_dims, std_scalar=0.01):
    """
    此函数是为了初始化多层网络参数而使用的函数。
    参数：
        layers_dims - 包含我们网络中每个图层的节点数量的列表

    返回：
        parameters - 包含参数“W1”，“b1”，...，“WL”，“bL”的字典：
                     W1 - 权重矩阵，维度为（layers_dims [1]，layers_dims [1-1]）
                     bl - 偏向量，维度为（layers_dims [1]，1）
    """
    assert len(layers_dims) > 1
    parameters = {}
    for i in range(1, len(layers_dims)):
        cur_W = std_scalar * np.random.randn(layers_dims[i], layers_dims[i - 1])
        cur_b = np.zeros((layers_dims[i], 1), dtype='float')
        parameters['W%i' % i] = cur_W
        parameters['b%i' % i] = cur_b
    return parameters


def linear_forward(A, W, b):
    Z = np.dot(W, A) + b
    cache = (A, W, b)
    return Z, cache


def linear_activation_forward(A_prev, W, b, activation):
    Z, linear_cache = linear_forward(A_prev, W, b)
    if activation == 'relu':
        A, activation_cache = relu(Z)
    elif activation == 'sigmoid':
        A, activation_cache = sigmoid(Z)
    else:
        A, activation_cache = Z,Z

    assert A.shape == Z.shape == (W.shape[0], A_prev.shape[1])
    cache = linear_cache, activation_cache
    return A, cache


def L_model_forward(X,parameters):
    caches = []
    A = X
    L = len(parameters) //2
    for i in range(1,L):
        cur_W = parameters['W%i'%i]
        cur_b = parameters['b%i'%i]
        A,cache = linear_activation_forward(A, cur_W, cur_b, 'relu')
        caches.append(cache)
    cur_W = parameters['W%i'%L]
    cur_b = parameters['b%i'%L]
    AL,cache = linear_activation_forward(A,cur_W,cur_b,'sigmoid')
    caches.append(cache)
    return AL,caches

def compute_cost(AL,Y):
    """
    :param AL:与标签预测相对应的概率向量，维度为（1，示例数量）shape = (1,m)
    :param Y:标签向量（例如：如果不是猫，则为0，如果是猫则为1），维度为（1,m）
    :return:cost - 交叉熵成本
    """
    m = Y.shape[1]
    cost = - (1./m) * np.sum(Y * np.log(AL) + (1-Y) * np.log((1-AL)))
    return np.squeeze(cost)

def linear_backward(dZ,cache):
    A_prev,W,b = cache
    m = A_prev.shape[1]
    # dW = np.dot(dZ, A_prev.T) / m
    # db = np.sum(dZ, axis=1, keepdims=True) / m
    # dA_prev = np.dot(W.T, dZ)

    dW = (1./m) * np.dot(dZ,A_prev.T)
    db = (1./m) * np.sum(dZ,axis=1,keepdims=True)
    dA_prev = np.dot(W.T,dZ)

    assert dW.shape == W.shape
    assert db.shape == b.shape
    assert dA_prev.shape == A_prev.shape

    return dA_prev,dW,db

def linear_activation_backward(dA,cache,activation='relu'):
    linear_cache,activation_cache = cache
    if activation == 'relu':
        dZ = relu_backward(dA,activation_cache)
    elif activation == 'sigmoid':
        dZ = sigmoid_backward(dA,activation_cache)
    else:
        dZ = dA
    return linear_backward(dZ,linear_cache)

def L_model_backward(AL,Y,caches):
    """
    对[LINEAR-> RELU] *（L-1） - > LINEAR - > SIGMOID组执行反向传播，就是多层网络的向后传播

    参数：
     AL - 概率向量，正向传播的输出（L_model_forward（））
     Y - 标签向量（例如：如果不是猫，则为0，如果是猫则为1），维度为（1，数量）
     caches - 包含以下内容的cache列表：
                 linear_activation_forward（"relu"）的cache，不包含输出层
                 linear_activation_forward（"sigmoid"）的cache

    返回：
     grads - 具有梯度值的字典
              grads [“dA”+ str（l）] = ...
              grads [“dW”+ str（l）] = ...
              grads [“db”+ str（l）] = ...
    """
    grads = {}
    L = len(caches)
    Y = Y.reshape(AL.shape)
    m = AL.shape[1]
    dAL = -(Y/AL - (1-Y)/(1-AL))

    last_cache = caches[L-1]
    dA_prev,dW,db = linear_activation_backward(dAL,last_cache,activation='sigmoid')
    grads['dA%i'%L],grads['dW%i'%L],grads['db%i'%L] = dA_prev,dW,db

    for l in reversed(range(L-1)):
        cur_cache = caches[l]
        dA_prev, dW, db = linear_activation_backward(grads["dA" + str(l + 2)], cur_cache, activation='relu')
        grads['dA%i' % (l+1)], grads['dW%i' % (l+1)], grads['db%i' % (l+1)] = dA_prev, dW, db
    return grads

def update_parameters(parameters,grads,learning_rate):
    L = len(parameters) // 2
    for l in range(L):
        parameters['W%i'%(l+1)] = parameters['W%i'%(l+1)] - grads['dW%i'%(l+1)] * learning_rate
        parameters['b%i' % (l + 1)] = parameters['b%i' % (l + 1)] - grads['db%i' % (l + 1)] * learning_rate
    return parameters


def two_layers_model(X,Y,layers_dims,learning_rate=0.0075,num_iterations=3000,print_cost=False,isPlot=True,std_scalar=0.01):
    """

        实现一个两层的神经网络，【LINEAR->RELU】 -> 【LINEAR->SIGMOID】
    参数：
        X - 输入的数据，维度为(n_x，例子数)
        Y - 标签，向量，0为非猫，1为猫，维度为(1,数量)
        layers_dims - 层数的向量，维度为(n_x,n_h,n_y)
        learning_rate - 学习率
        num_iterations - 迭代的次数
        print_cost - 是否打印成本值，每100次打印一次
        isPlot - 是否绘制出误差值的图谱
    返回:
        parameters - 一个包含W1，b1，W2，b2的字典变量
    """
    np.random.seed(1)
    n_x,n_h,n_y = layers_dims
    costs = []

    parameters = initialize_parameters_deep(layers_dims, std_scalar=std_scalar)

    for i in range(num_iterations):

        W1 = parameters['W1']
        b1 = parameters['b1']
        W2 = parameters['W2']
        b2 = parameters['b2']

        A1, cache1 = linear_activation_forward(X, W1, b1, 'relu')
        A2, cache2 = linear_activation_forward(A1,W2,b2,'sigmoid')
        AL = A2
        if i % 100 == 0:
            cost = compute_cost(AL, Y)
            costs.append(cost)
            if print_cost:
                print("i=%i,costs=%f"%(i,cost))
        grads = L_model_backward(AL, Y, (cache1, cache2))
        parameters['W1'] = parameters['W1'] - learning_rate * grads['dW1']
        parameters['b1'] = parameters['b1'] - learning_rate * grads['db1']
        parameters['W2'] = parameters['W2'] - learning_rate * grads['dW2']
        parameters['b2'] = parameters['b2'] - learning_rate * grads['db2']
    if isPlot:
        plt.plot(costs)
        plt.show()
    return parameters


def predict(X,y,parameters):
    m = y.shape[1]
    AL,caches = L_model_forward(X,parameters)
    p = np.array(AL>0.5,'int')
    print('accuracy=%f'%(np.sum(p == y) / m))
    return p

if __name__ == '__main__':
    train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = lr_utils.load_dataset()

    train_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
    test_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T

    train_x = train_x_flatten / 255
    train_y = train_set_y
    test_x = test_x_flatten / 255
    test_y = test_set_y

    n_x = 12288
    n_h = 7
    n_y = 1
    layers_dims = (n_x, n_h, n_y)

    parameters = two_layers_model(train_x, train_set_y, layers_dims=(n_x, n_h, n_y), num_iterations=10000,
                                 print_cost=True, isPlot=True)
    predict(train_x,train_y,parameters)
    predict(test_x,test_y,parameters)

