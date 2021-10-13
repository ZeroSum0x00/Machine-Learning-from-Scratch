from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d

class Visualizer:

    def plot_data_2D(self, data_axis_X, data_axis_y, c=None, s=None, marker=None, edgecolors=None, label=None, cmap=None):
        plt.scatter(data_axis_X, data_axis_y, c=c, s=s, marker=marker, edgecolors=edgecolors, label=label, cmap=plt.cm.Spectral)

    def plot_line_2D(self, X, y, label=None, linestyle=None):
        plt.plot(X, y, label=label, linestyle=linestyle)

    def plot_point_2D(self, points, shape='rx', label=None):
        plt.plot(points[:, 0], points[:, 1], shape, label=label)

    def plot_circle_2D(self, center_points, radius):
        circle1 = plt.Circle(xy=(center_points[:, 0], center_points[:, 1]), radius=radius, color='b', fill=False)
        plt.gca().add_patch(circle1)

    def plot_contourf_2D(self, xx, yy, Z, cmap=None):
        plt.contourf(xx, yy, Z, cmap=cmap)

    def create_plot_3D(self):
        self.ax = plt.axes(projection='3d')

    def plot_data_3D(self, X, y):
        self.ax.scatter3D(X[:, 0], X[:, 1], y, c=y, linewidth=0.5)

    def plot_line_3D(self, X, y):
        plt.plot(X[0], X[1], y, 'g')

    def plot_show(self, title='', xlabel='', ylabel='', legend_title=None):
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        if legend_title is not None:
            plt.legend(loc='best', title=legend_title)
        plt.show()

    def plot_saved(self, saved_path):
        plt.savefig(saved_path)