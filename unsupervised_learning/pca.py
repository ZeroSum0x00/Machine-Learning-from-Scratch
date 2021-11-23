import numpy as np

class PCA:
    def __init__(self, n_components):
        self.n_components = n_components
        self.components = None
        self.mean = None

    def fit(self, X):
        # Caculation mean and substract from X data
        self.mean = np.mean(X, axis=0)
        X = X - self.mean

        # Caclulation covariance
        covariance = np.cov(X.T)

        # Caclulation eigenvectors and eigenvalues
        eigenvalues, eigenvectors = np.linalg.eig(covariance)

        # Sort eigenvectors
        eigenvectors = eigenvectors.T
        idxs = np.argsort(eigenvalues)[: : -1]
        eigenvalues = eigenvalues[idxs]
        eigenvectors = eigenvectors[idxs]

        # Store first n eigenvectors
        self.components = eigenvectors[0: self.n_components]

    def transform(self, X):
        X = X - self.mean
        return np.dot(X, self.components.T)

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, X):
        pass