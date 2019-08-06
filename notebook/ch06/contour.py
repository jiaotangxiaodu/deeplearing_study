import matplotlib.pyplot as plt
import numpy as np

def f(x,y):
    return x**(-1)+y**(-1)

if __name__ == '__main__':

    x = np.linspace(-10,10,1000)
    y = np.linspace(-10,10,1000)
    X,Y = np.meshgrid(x,y)
    Z = f(X,Y)
    Z[Z > 1] = 0
    plt.xlim(-10,10)
    plt.ylim(-10,10)

    cs = plt.contour(X,Y,Z)
    plt.clabel(cs,inline=1,fontsize=10)
    plt.show()