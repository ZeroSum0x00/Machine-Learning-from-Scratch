import numpy as np

class Perceptron:
    def __init__(self, learning_rate=0.01, n_iters=1000, activation=None):
        self.learning_rate = learning_rate
        self.n_iters = n_iters
        self.weights = None
        self.bias = None
        if activation is not None:
            self.activation = activation
        else:
            self.activation = self._linear_activation

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        # y_ = np.array([1 if i > 0 else 0 for i in y])   # normalizer label to range(0, 1)
        y_ = np.where(y <= 0, 0, 1)     # normalizer label to range(0, 1)

        for _ in range(self.n_iters):
            for idx,  x_i in enumerate(X):
                linear_output = np.dot(x_i, self.weights) + self.bias
                y_pred = self.activation(linear_output)

                update = self.learning_rate * (y_[idx] - y_pred)
                self.weights += update * x_i
                self.bias += update

    def predict(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        y_pred = self.activation(linear_output)
        return y_pred

    def _linear_activation(self, x):
        return x
