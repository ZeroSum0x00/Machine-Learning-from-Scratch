from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d

class Visualizer:

    def plot_data_2D(self, data_axis_X, data_axis_y, label=None, marker=None, edgecolors=None):
        plt.scatter(data_axis_X, data_axis_y, c=label, marker=marker, edgecolors=edgecolors)

    def plot_line_2D(self, X, y, label=None, linestyle=None):
        plt.plot(X, y, label=label, linestyle=linestyle)

    def plot_point_2D(self, points, shape='rx'):
        plt.plot(points[:, 0], points[:, 1], shape)

    def plot_circle_2D(self, center_points, radius):
        circle1 = plt.Circle(xy=(center_points[:, 0], center_points[:, 1]), radius=radius, color='b', fill=False)
        plt.gca().add_patch(circle1)

    def create_plot_3D(self):
        self.ax = plt.axes(projection='3d')

    def plot_data_3D(self, X, y):
        self.ax.scatter3D(X[:, 0], X[:, 1], y, c=y, linewidth=0.5)

    def plot_line_3D(self, X, y):
        plt.plot(X[0], X[1], y, 'g')

    def plot_show(self, title='', xlabel='', ylabel=''):
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.legend(loc='best')
        plt.show()

    def plot_saved(self, saved_path):
        plt.savefig(saved_path)