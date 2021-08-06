import numpy as np
from utils.preprocess import Preprocessing
from utils.visualizer import Visualizer

if __name__ == '__main__':
    data = '../datasets/the_trang.csv'
    label = '../datasets/the_trang_labels.csv'
    dict_X = ['ChieuCao', 'CanNang']
    dict_y = ['Loai']
    pp = Preprocessing()
    pp.get_label(label)
    X_train, y_train = pp.get_data(data, dict_X=dict_X, dict_y=dict_y)
    X_test = np.array([[170, 52]])

    visual = Visualizer()
    visual.plot_data_2D(X_train[:, 0], X_train[:, 1])
    visual.plot_show('Visualization K-Means Clustering model', dict_X[0], dict_X[1])