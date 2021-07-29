import numpy as np
from scipy.spatial.distance import cdist
from utils.visualizer import Visualizer

class KNearest_Neighbors:
    def __init__(self, n_neighbors):
        self.n_neighbors = n_neighbors

    def fit(self, X, y, distance_mode='enclidean'):
        self.X_train = X
        self.y_train = y
        self.distance_mode = distance_mode

    def predict(self, X):
        if self.distance_mode == 'cdist':
            distances = cdist(self.X_train, X)

        elif self.distance_mode == 'enclidean':
            distances = [self._enclidean_distance(X, x_train) for x_train in self.X_train]
            distances = np.array(distances)

        elif self.distance_mode == 'linalg':
            distances = [self._linalg_norm(X, x_train) for x_train in self.X_train]
            distances = np.array(distances)

        elif self.distance_mode == 'linalg_T':
            distances = [self._linalg_norm_T(X, x_train) for x_train in self.X_train]
            distances = np.array(distances)

        elif self.distance_mode == 'einsum':
            distances = [self._sqrt_einsum(X, x_train) for x_train in self.X_train]
            distances = np.array(distances)

        k_indices = np.argsort(distances, axis=0)[:self.n_neighbors]
        k_nearest_labels = self.y_train[k_indices]

        nearest_points = self.X_train[k_indices]

        classes, counts = np.unique(k_nearest_labels, return_counts=True)
        y_pred = classes[np.argmax(counts)]
        return y_pred, nearest_points

    def _enclidean_distance(self, x, y):
        return np.sqrt(np.sum((x - y) ** 2))

    def _linalg_norm(self, x, y):
        return np.linalg.norm(x - y, axis=0)

    def _linalg_norm_T(self, x, y):
        return np.linalg.norm(x - y, axis=1)

    def _sqrt_einsum(self, x, y):
        x_min_y = x - y
        return np.sqrt(np.einsum('ij, ij->i', x_min_y, x_min_y))