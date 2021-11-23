import numpy as np
from scipy.spatial.distance import cdist

class KMeans_Clustering:
    def __init__(self, n_clusters=3, max_iters=1500):
        self.n_clusters = n_clusters
        self.max_iters = max_iters
        self.center_points = None
        self.label_pred = None

    def predict(self, X):
        # Tạo self.n_clusters điểm bất kỳ, các điểm này thuộc các điểm có trên dữ liệu
        self.center_points = X[np.random.choice(X.shape[0], self.n_clusters, replace=False)]
        for iter in range(self.max_iters):
            # Tính khoảng cách từ các điểm có trên dữ liệu đến self.n_clusters điểm mới tạo
            distances = cdist(X, self.center_points)

            # Tìm ra tập label (là tập chứa vị trí điểm có khoảng cách đến tâm nhỏ nhất)
            self.label_pred = np.argmin(distances, axis=1)

            center_compare = self.center_points

            self.center_points = []
            for i in range(self.n_clusters):
                self.center_points.append(np.mean(X[self.label_pred == i], axis=0))

            self.center_points = np.array(self.center_points)

            if set(tuple(c) for c in self.center_points) == set(tuple(c) for c in center_compare):
                break

        return self.center_points
