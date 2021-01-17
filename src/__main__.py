import os

import pandas as pd
import tensorflow as tf
from src.estimator import AngleEstimator
from src.visualizations import get_scatter_plot


physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
   tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

my_devices = tf.config.experimental.list_physical_devices(device_type='CPU')
tf.config.experimental.set_visible_devices(devices=my_devices[0], device_type='CPU')
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

def main():
    roof_points_path = "../data/points.csv"
    roof_points = pd.read_csv(roof_points_path).values[:, 1:]
    get_scatter_plot(roof_points[:, 0], roof_points[:, 1]).show()
    estimator = AngleEstimator()
    estimator.get_side_angles(x_y=roof_points[:, 0:2], z=roof_points[:, 2])


if __name__ == "__main__":
    main()