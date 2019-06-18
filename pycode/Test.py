import numpy as np

def _gradient_vector(f,x,h=1e-6):
    g = np.zeros(len(x))
    for i,feature in enumerate(x):
        x[i] = feature + h
        y_high = f(x)
        x[i] = feature - h
        y_low = f(x)
        x[i] = feature
        g[i] = (y_high - y_low) / (2*h)
    return g

if __name__ == '__main__':
    p = np.array([[.1,.22,.68],[.6,.2,.2]])
    y = np.array([2,0])
    t = np.eye(p.shape[1], dtype='int')[y]
    mse = np.sum((p - t)**2)/2
    print(mse)






