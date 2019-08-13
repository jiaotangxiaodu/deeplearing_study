from init_utils import load_dataset
import matplotlib.pyplot as plt
from multi_layer_nn import MultiLayersNN
from plt_utils import plot_decision_boundary
from reg_utils import load_2D_dataset

if __name__ == '__main__':
    X_train,Y_train,X_test,Y_test = load_2D_dataset(is_plot=False)

    assert X_train.shape == (2, 211)
    assert Y_train.shape == (1, 211)
    assert X_test.shape == (2, 200)
    assert Y_test.shape == (1, 200)

    model = MultiLayersNN((X_train.shape[0],20,3,1))
    model.fit(X_train,Y_train,lr=0.1,print_cost_100=True,iter_num=10000)
    print(model.accuracy(X_train,Y_train))
    print(model.accuracy(X_test,Y_test))
    plot_decision_boundary(model.predict,(-1,1,-1,1),plt)
    plt.scatter(X_test[:, Y_test.squeeze() == 1][0,:], X_test[:, Y_test.squeeze() == 1][1,:])
    plt.scatter(X_test[:, Y_test.squeeze() == 0][0,:], X_test[:, Y_test.squeeze() == 0][1,:])
    plt.show()




