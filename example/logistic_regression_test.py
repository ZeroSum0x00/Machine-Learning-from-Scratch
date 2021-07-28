from sklearn import datasets
from sklearn.model_selection import train_test_split
from utils.preprocess import Preprocessing
from supervised_learning.logistic_regression import Gradient_Logistic_Regression
from metrics.simple_metrics import accuracy

if __name__ == '__main__':
    data = '../datasets/the_trang_logistic_regression.csv'
    dict_X = ['TheTrang']
    dict_y = ['Loai']

    pp = Preprocessing()
    X, y = pp.get_data(data, dict_X=dict_X, dict_y=dict_y, norm='3', norm_range=[-1, 1])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

    model = Gradient_Logistic_Regression(learning_rate=0.001, batch_size=64, n_epochs=1000)
    model.fit(X_train, y_train)
    y_pred, y_label = model.predict(X_test)
    print('y predict: ', y_pred.reshape(-1))
    print('Predict: ', y_label)
    print('Label: ', y_test.reshape(-1))
    print('accuracy score: ', accuracy(y_test.reshape(-1), y_label))

    # pp.saved_data((predicted1, predicted2), '../saved_weights/linear_regression.plk')
