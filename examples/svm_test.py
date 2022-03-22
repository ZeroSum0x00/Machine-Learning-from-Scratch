import numpy as np
from sklearn.model_selection import train_test_split
from utils.preprocess import Preprocessing
from utils.visualizer import Visualizer
from metrics.simple_metrics import accuracy
from supervised_learning.svm import SVM

def get_hyperplane_value(x, w, b, offset):
    return (-w[0] * x + b + offset) / w[1]

if __name__ == '__main__':
    data = '../datasets/the_trang_binary_classification.csv'
    dict_X = ['ChieuCao', 'CanNang']
    dict_y = ['Loai']

    pp = Preprocessing()
    X, y = pp.get_data(data, dict_X=dict_X, dict_y=dict_y, norm='0', norm_range=[-1, 1])
    y = y.reshape(-1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

    model = SVM()
    model.fit(X_train, y_train)
    weights = model.weights
    bias = model.bias
    y_pred = model.predict(X_test)

    print('Model accuracy score: ', accuracy(y_test.reshape(-1), y_pred))

    x0_1 = np.min(X_train[:, 0])
    x0_2 = np.max(X_train[:, 0])

    x1_1 = get_hyperplane_value(x0_1, weights, bias, 0)
    x1_2 = get_hyperplane_value(x0_2, weights, bias, 0)

    x1_1_margin = get_hyperplane_value(x0_1, weights, bias, -1)
    x1_2_margin = get_hyperplane_value(x0_2, weights, bias, -1)

    x1_1_padding = get_hyperplane_value(x0_1, weights, bias, 1)
    x1_2_padding = get_hyperplane_value(x0_2, weights, bias, 1)

    visual = Visualizer()
    visual.plot_data_2D(X[:, 0], X[:, 1], c=y, marker='o', edgecolors='k')
    visual.plot_line_2D([x0_1, x0_2], [x1_1, x1_2], label='Hyperplane', linestyle='--')
    visual.plot_line_2D([x0_1, x0_2], [x1_1_margin, x1_2_margin], label='Support Vectors 1')
    visual.plot_line_2D([x0_1, x0_2], [x1_1_padding, x1_2_padding], label='Support Vectors 2')
    visual.plot_saved('../assets/svm_plot.png')
    visual.plot_show('Visualization SVM model', dict_X[0], dict_X[1], legend_title='Note')
