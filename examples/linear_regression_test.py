from sklearn import datasets
from sklearn.model_selection import train_test_split
from utils.preprocess import Preprocessing
from utils.visualizer import Visualizer
from supervised_learning.linear_regression import Normal_Equation_Linear_Regression
from supervised_learning.linear_regression import Gradient_Linear_Regression
from metrics.simple_metrics import mse


if __name__ == '__main__':
    data = '../datasets/the_trang_linear_regression.csv'
    dict_X = ['ChieuCao']
    dict_y = ['CanNang']

    pp = Preprocessing()
    X, y = pp.get_data(data, dict_X=dict_X, dict_y=dict_y, norm='0', norm_range=[-1, 1])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

    model1 = Normal_Equation_Linear_Regression(batch_size=64)
    model1.fit(X_train, y_train)
    predicted1 = model1.predict(X_test)
    loss1 = mse(y_test, predicted1)
    print('MSE losses score: ', loss1)

    model2 = Gradient_Linear_Regression(learning_rate=0.001, batch_size=64, n_epochs=1000)
    model2.fit(X_train, y_train)
    predicted2 = model2.predict(X_test)
    loss2 = mse(y_test, predicted2)
    print('MSE losses score: ', loss2)

    visual = Visualizer()
    visual.plot_data_2D(X[:, 0], y, marker='o', edgecolors='k')
    visual.plot_line_2D(X_test, predicted1, label='Normal Equation')
    visual.plot_line_2D(X_test, predicted2, label='Gradient Descent')
    visual.plot_saved('../assets/linear_regression_plot.png')
    visual.plot_show('Visualization Linear Regression model', dict_X[0], dict_y[0], legend_title='Method')

    pp.saved_data((predicted1, predicted2), '../saved_weights/linear_regression.plk')
