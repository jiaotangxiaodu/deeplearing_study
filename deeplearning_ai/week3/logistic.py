import numpy as np
import matplotlib.pyplot as plt
from testCases import *
import sklearn
import sklearn.datasets
import sklearn.linear_model
from planar_utils import plot_decision_boundary,sigmoid,load_planar_dataset,load_extra_datasets

if __name__ == '__main__':
    np.random.seed(1)

    X,Y = load_planar_dataset()
    y = np.squeeze(Y)

    print(y.shape)
    print(X.shape)
    print(Y.shape)

    plt.scatter(X[0,y == 0], X[1,y == 0])
    plt.scatter(X[0,y == 1], X[1,y == 1])


    clf = sklearn.linear_model.LogisticRegressionCV()
    clf.fit(X.T,y.T)

    plot_decision_boundary(lambda x : clf.predict(x),X,Y)
    plt.show()
    LR_predictions = clf.predict(X.T)  # 预测结果
    print("逻辑回归的准确性： %d " % float((np.dot(Y, LR_predictions) +
                                   np.dot(1 - Y, 1 - LR_predictions)) / float(Y.size) * 100) +
          "% " + "(正确标记的数据点所占的百分比)")