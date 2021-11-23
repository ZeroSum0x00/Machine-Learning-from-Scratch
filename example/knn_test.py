import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from scipy.spatial.distance import cdist
from utils.visualizer import Visualizer
from utils.preprocess import Preprocessing
from supervised_learning.knn import KNearest_Neighbors


# iris = datasets.load_iris()
# X, y = iris.data, iris.target
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

data = '../datasets/the_trang.csv'
label = '../datasets/the_trang_labels.csv'
dict_X = ['ChieuCao', 'CanNang']
dict_y = ['Loai']
pp = Preprocessing()
pp.get_label(label)
X_train, y_train = pp.get_data(data, dict_X=dict_X, dict_y=dict_y)
X_test = np.array([[170, 52]])

model = KNearest_Neighbors(n_neighbors=21)
model.fit(X_train, y_train, distance_mode='cdist')
prediction_label, nearest_points = model.predict(X_test)
print('Prediction: ', prediction_label)
print('Label Prediction: ', pp.match_label(prediction_label))

nearest_points = nearest_points.reshape(nearest_points.shape[0], nearest_points.shape[-1])
max_radius = cdist(np.array([nearest_points[-1]]), X_test)

visual = Visualizer()
visual.plot_data_2D(X_train[:, 0], X_train[:, 1], y_train, edgecolors='k')
visual.plot_circle_2D(X_test, max_radius)
for i in range(nearest_points.shape[0]):
    visual.plot_line_2D((X_test[0][0], nearest_points[:, 0][i]), (X_test[0][1], nearest_points[:, 1][i]), )
visual.plot_point_2D(X_test, shape='Xr', label='Test Points')
visual.plot_saved('../assets/knn_plot.png')
visual.plot_show('Visualization K-Nearest Neighbors model', dict_X[0], dict_X[1], legend_title='Note')
