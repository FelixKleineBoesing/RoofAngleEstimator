import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np


def get_2d_scatter_plot(x, y):
    return plt.plot(x, y, "rx")[0].get_figure()


def get_3d_scatter_with_prediction_planes(scatter_points: np.ndarray, planes: np.ndarray):
    assert scatter_points.shape[1] == 3
    assert planes.shape[1] == 3
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(scatter_points[:, 0], scatter_points[:, 1], scatter_points[:, 2])
    ax.scatter(planes[:, 0], planes[:, 1], planes[:, 2])
    return fig
