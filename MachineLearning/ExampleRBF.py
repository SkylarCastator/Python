#Finds the Vector grouping using Radial Bias Function
import numpy as np
import matplotlib.pyplot as plt

class DataSet:
    def __init__(self, X, Y, gamma_options):
        self.X = X
        self.Y = Y
        self.gamma_options = gamma_options

def map_rbf(DataSet data_set):
    plt.figure(1, figsize=(4*len(data_set.gamma_option), 4))
    for i, gamma in enumerate(data_set.gamma_option, 1):
      svm =SVC(kernel='rbf', gamma=gamma)
      svm.fit(data_set.X,data_set.Y)
      plt.subplot(1, len(gamma_option), i)
      plt.scatter(X[:, 0], X[:, 1], c=Y, zorder=10, cmap=plt.cm.Paired)
      plt.axis('tight')
      XX,YY =np.mgrid[-3:3:200j, -3:3:200j]
      Z = svm.decision_funtion(np.c_[XX.ravel(),YY.ravel()])
      Z = Z.reshape(XX.shape)
      plt.pcolormesh(XX,YY, Z > 0, cmap=plt.cm.Paired)
      plt.contour(XX,YY,Z, colors=['k', 'k', 'k'], linestyles=['--', '-', '--'], levels=[-.5,0,.5])
      plt.title('gamma = %d' % gamma)
      plt.show()

def test_sample_data:
    X = np.c_[# negative vlass
                (.3, -.8),
                (-1.5, -1),
                (-1.3, -.8),
                (-1.1, -1.3),
                (-1.2, -0.3),
                (-1.3, -.5),
                (-.6, 1.1),
                (-1.4, 2.2),
                (1, 1),
                #positive class
                (1.3, .8)
                (1.2, .5)
                (.2, -2),
                (.5, -2.4),
                (.2, -2.3),
                (0, -2.7),
                (1.2, 2.1)].T
    Y = [-1] *8 + [1] * 8
    gamma_option = [1,2,4]
    example_set = DataSet(X,Y,gamma_option)
    map_rbf(example_set)
