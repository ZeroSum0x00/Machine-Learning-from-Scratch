import numpy as np
from sklearn import datasets
from utils.visualizer import Visualizer
from matplotlib import pyplot as plt
from unsupervised_learning.pca import PCA

if __name__ == '__main__':
    data = datasets.load_iris()
    X = data.data
    y = data.target

    model = PCA(n_components=2)
    model.fit(X)
    X_projected = model.transform(X)

    print('Shape of X: ', X.shape)
    print('Shape of transformed X: ', X_projected.shape)

    x1 = X_projected[:, 0]
    x2 = X_projected[:, 1]

    visual = Visualizer()
    visual.plot_data_2D(x1, x2, c=y, alpha=0.8, edgecolors='none', cmap=plt.cm.get_cmap('viridis', 3))
    visual.plot_saved('../assets/pca_plot.png')
    visual.plot_show(title='Dimensionality Reduction using PCA algorithm', xlabel='Principal Component 1', ylabel='Principal Component 1', colorbar=True)
