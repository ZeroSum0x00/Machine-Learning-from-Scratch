from utils.preprocess import Preprocessing
from utils.visualizer import Visualizer
from unsupervised_learning.kmean import KMeans_Clustering

if __name__ == '__main__':
    data = '../datasets/the_trang.csv'
    label = '../datasets/the_trang_labels.csv'
    dict_X = ['ChieuCao', 'CanNang']
    dict_y = ['Loai']
    pp = Preprocessing()
    pp.get_label(label)
    X, y = pp.get_data(data, dict_X=dict_X, dict_y=dict_y)

    k = 6
    model = KMeans_Clustering(n_clusters=k)
    center_points = model.predict(X=X)

    visual = Visualizer()
    for i in range(k):
        X_i = X[model.label_pred == i]
        visual.plot_data_2D(X_i[:, 0], X_i[:, 1], label=pp.label_mean[i])

    visual.plot_point_2D(center_points, shape='Xk', label='Center Points')
    visual.plot_saved('../saved_weights/kmean.png')
    visual.plot_show('Visualization K-Means Clustering model', dict_X[0], dict_X[1])