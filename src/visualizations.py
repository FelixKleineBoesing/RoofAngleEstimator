import matplotlib.pyplot as plt


def get_scatter_plot(x, y):
    return plt.plot(x, y, "rx")[0].get_figure()