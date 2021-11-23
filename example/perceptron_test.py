import numpy as np
from sklearn.model_selection import train_test_split
from metrics.simple_metrics import accuracy
from utils.visualizer import Visualizer
from utils.preprocess import Preprocessing
from supervised_learning.perceptron import Perceptron


if __name__ == '__main__':
    data = '../datasets/the_trang_binary_classification.csv'
    dict_X = ['ChieuCao', 'CanNang']
    dict_y = ['Loai']

    pp = Preprocessing()
    X, y = pp.get_data(data, dict_X=dict_X, dict_y=dict_y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

    model = Perceptron(learning_rate=0.01, n_iters=1000)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print('Model accuracy score: ', accuracy(y_test.reshape(-1), y_pred))

    x0_1 = np.min(X_train[:, 0])
    x0_2 = np.max(X_train[:, 0])

    x1_1 = (-model.weights[0] * x0_1 - model.bias) / model.weights[1]
    x1_2 = (-model.weights[0] * x0_2 - model.bias) / model.weights[1]

    visual = Visualizer()
    visual.plot_data_2D(X[:, 0], X[:, 1], c=y, marker='o', edgecolors='k')
    visual.plot_line_2D([x0_1, x0_2], [x1_1, x1_2])
    visual.plot_saved('../assets/perceptron_plot.png')
    visual.plot_show('Visualization Perceptron model', dict_X[0], dict_X[1])
