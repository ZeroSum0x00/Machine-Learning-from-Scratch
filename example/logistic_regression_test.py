import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from supervised_learning.logistic_regression import Gradient_Logistic_Regression
from activations.activations import sigmoid
from utils.visualizer import Visualizer
from metrics import accuracy, r2, binary_confusion_matrix, precision, recall, f_score

np.random.seed(0)
matplotlib.rcParams['figure.figsize'] = (10.0, 8.0)


if __name__ == '__main__':
    # Tạo dataset
    X, y = make_moons(n_samples=500, noise=0.20)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

    # Định nghĩa và fit hàm Logistic Regression
    model = Gradient_Logistic_Regression(learning_rate=0.001, batch_size=64, n_epochs=1000, activation='sigmoid')
    model.fit(X, y)

    # Tìm giá trị lớn nhất và nhỏ nhất ở 2 trục X, Y
    x_min, x_max = X_train[:, 0].min() - 0.5, X_train[:, 0].max() + 0.5
    y_min, y_max = X_train[:, 1].min() - 0.5, X_train[:, 1].max() + 0.5
    h = 0.01

    # Lấy tất cả các giá trị trong range(min, max, h)
    x_coordinate = np.arange(x_min, x_max, h)
    y_coordinate = np.arange(y_min, y_max, h)

    # Trả về ma trận tọa độ từ vector tọa độ
    xx, yy = np.meshgrid(x_coordinate, y_coordinate)

    # Flatten dữ liệu
    xx_flatten = xx.ravel()
    yy_flatten = yy.ravel()

    # np.column_stack((xx_flatten, yy_flatten)): Xếp mảng 2 mảng 1 chiều thành 1 mảng 2 chiều
    _, y_pred_label = model.predict(np.column_stack((xx_flatten, yy_flatten)))
    y_pred_label = y_pred_label.reshape(xx.shape)

    # Visual dữ liệu
    visual = Visualizer()
    visual.plot_contourf_2D(xx, yy, y_pred_label, cmap=plt.cm.Spectral)
    visual.plot_data_2D(X[:, 0], X[:, 1], c=y, s=60, marker='o', edgecolors='k', cmap=plt.cm.Spectral)
    visual.plot_saved('../assets/logistic_regression_plot.png')
    visual.plot_show('Logistic Regression', 'feature 1', 'feature 2')

    # Evaluate
    _, y_pred = model.predict(X_test)
    print('Model accuracy score: ', accuracy(y_test, y_pred))
    print("Model r2 score: ", r2(y_test, y_pred))
    print("Confusion matrix: ", binary_confusion_matrix(y_test, y_pred, visual=True))
    print('Model precision score: ', precision(y_test, y_pred))
    print('Model recall score: ', recall(y_test, y_pred))
    print('Model f1 score score: ', f_score(y_test, y_pred))