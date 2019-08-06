from optimizer import SGD,Momentum,AdaGrad,Adam
from dataset.mnist import load_mnist

(x_train,t_train),(x_test,t_test) = load_mnist(normalize=True)

