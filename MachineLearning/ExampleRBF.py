# Finds the Vector grouping using Radial Bias Function
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC


class DataSet:
    def __init__(self, X, Y, gamma_options):
        self.X = X
        self.Y = Y
        self.gamma_option = gamma_options


def map_rbf(data_set):
    plt.figure(1, figsize=(4 * len(data_set.gamma_option), 4))
    for i, gamma in enumerate(data_set.gamma_option, 1):
        svm = SVC(kernel='rbf', gamma=gamma)
        svm.fit(data_set.X, data_set.Y)
        plt.subplot(1, len(data_set.gamma_option), i)
        plt.scatter(data_set.X[:, 0], data_set.X[:, 1], c=data_set.Y, zorder=10, cmap=plt.cm.Paired)
        plt.axis('tight')
        XX, YY = np.mgrid[-3:3:200j, -3:3:200j]
        Z = svm.decision_function(np.c_[XX.ravel(), YY.ravel()])
        Z = Z.reshape(XX.shape)
        plt.pcolormesh(XX, YY, Z > 0, cmap=plt.cm.Paired)
        plt.contour(XX, YY, Z, colors=['k', 'k', 'k'], linestyles=['--', '-', '--'], levels=[-.5, 0, .5])
        plt.title('gamma = %d' % gamma)
        plt.show()


def test_sample_data():
    x = np.c_[
        (.3, -.8),
        (-1.5, -1),
        (-1.3, -.8),
        (-1.1, -1.3),
        (-1.2, -0.3),
        (-1.3, -.5),
        (-.6, 1.1),
        (-1.4, 2.2),
        (1, 1),
        (1.3, .8),
        (1.2, .5),
        (.2, -2),
        (.5, -2.4),
        (.2, -2.3),
        (0, -2.7),
        (1.2, 2.1)].T
    y = [-1] * 8 + [1] * 8
    gamma_option = [1, 2, 4]
    example_set = DataSet(x, y, gamma_option)
    map_rbf(example_set)


test_sample_data()
