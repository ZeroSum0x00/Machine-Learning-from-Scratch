import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from supervised_learning.logistic_regression import Gradient_Logistic_Regression
from utils.visualizer import Visualizer

np.random.seed(0)
matplotlib.rcParams['figure.figsize'] = (10.0, 8.0)

if __name__ == '__main__':
    X, y = make_moons(200, noise=0.20)

    # định nghĩa và fit hàm Logistic Regression
    model = Gradient_Logistic_Regression(learning_rate=0.001, batch_size=64, n_epochs=1000)
    model.fit(X, y)

    # tìm giá trị lớn nhất và nhỏ nhất ở 2 trục X, Y
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    h = 0.01

    # lấy tất cả các giá trị trong range(min, max, h)
    x_coordinate = np.arange(x_min, x_max, h)
    y_coordinate = np.arange(y_min, y_max, h)
    # print(x_coordinate)
    # print(x_coordinate.shape)
    # print(y_coordinate.shape)

    # trả về ma trận tọa độ từ vector tọa độ
    xx, yy = np.meshgrid(x_coordinate, y_coordinate)
    # print(xx)
    # print(xx.shape)
    # print(yy)
    # print(yy.shape)

    # flatten dữ liệu
    xx_flatten = xx.ravel()
    yy_flatten = yy.ravel()

    # np.column_stack((xx_flatten, yy_flatten)): Xếp mảng 2 mảng 1 chiều thành 1 mảng 2 chiều
    _, y_pred_label = model.predict(np.column_stack((xx_flatten, yy_flatten)))
    y_pred_label = y_pred_label.reshape(xx.shape)

    visual = Visualizer()
    visual.plot_contourf_2D(xx, yy, y_pred_label, cmap=plt.cm.Spectral)
    visual.plot_data_2D(X[:, 0], X[:, 1], c=y, s=60, marker='o', edgecolors='k', cmap=plt.cm.Spectral)
    visual.plot_saved('../saved_weights/logistic_regression.png')
    visual.plot_show('Logistic Regression', 'feature 1', 'feature 2')