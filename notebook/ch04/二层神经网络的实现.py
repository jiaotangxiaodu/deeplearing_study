import numpy as np
import sys,os
sys.path.append(os.pardir)
from dataset.mnist import load_mnist
import matplotlib.pyplot as plt

def softmax(x):
    return np.exp(x) / (np.sum(np.exp(x)))

def cross_entropy_error(y,t,delta=1e-8):
    y2 = softmax(y + delta)
    return -t.dot(np.log(y2))

def gradient(f,x,epsilon=1e-6):
    g = np.zeros_like(x,dtype=float)
    x_line = x.reshape(-1,order='c')
    g_line = g.reshape(-1,order='c')

    for i in range(len(x_line)):
        tmp = x_line[i]
        x_line[i] = tmp + epsilon
        y_high = f(x)
        x_line[i] = tmp - epsilon
        y_low = f(x)
        x_line[i] = tmp
        g_line[i] = (y_high - y_low) / (2*epsilon)
    return g

def loss(w,x,t):
    return cross_entropy_error(np.sum(x.dot(w.T), axis=1), t)



if __name__ == '__main__':
    x =  np.array([[1.,2.],[3.,4.],[2.,2.]])
    max_iter = 1e4
    delta = 1e-6
    f = lambda w: loss(w, x, t)
    y = [0, 1, 0]

    eta_range = np.array([-3,-2,-1])
    for eta in [10.**i for i in eta_range]:
        w = np.array([[0., 0.], [0., 0.]])
        p = [softmax(row) for row in x.dot(w.T)]
        y_predict = np.argmax(p, axis=1)
        t = np.array(y_predict == y, dtype='int')
        last_loss = loss(w, x, t)
        img = np.zeros(shape=int(max_iter))
        iter = 0
        while iter < max_iter:
            g = gradient(f,w)
            w -= eta * g
            cur_loss = loss(w,x,t)
            # if last_loss - cur_loss < delta :
            #     break
            img[iter] = cur_loss
            iter += 1
            last_loss = cur_loss
        min = np.min(img)
        plt.plot(np.log(img - min),label=str(eta))
    plt.legend()
    plt.show()

