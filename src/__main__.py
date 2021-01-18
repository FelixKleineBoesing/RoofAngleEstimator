from datetime import datetime
import os
from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.python.framework.ops import disable_eager_execution

from src.estimator import AngleEstimator
from src.visualizations import get_2d_scatter_plot, get_3d_scatter_with_prediction_planes

physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
   tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

my_devices = tf.config.experimental.list_physical_devices(device_type='CPU')
tf.config.experimental.set_visible_devices(devices=my_devices[0], device_type='CPU')
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


def main():
    roof_points_path = Path("..", "data", "points.csv")
    tensorboard_dir = Path("..", "data", "tensorboard_log", datetime.now().strftime("%Y%m%d-%H%M%S"))

    roof_points = pd.read_csv(roof_points_path).values[:, 1:]
    get_2d_scatter_plot(roof_points[:, 0], roof_points[:, 1]).show()
    estimator = AngleEstimator(tensorboard_dir=tensorboard_dir, number_epochs=50)
    estimator.get_side_angles(x_y=roof_points[:, 0:2], z=roof_points[:, 2])
    fig = estimator.plot_3d_prediction_plot(data=roof_points)
    fig.show()


if __name__ == "__main__":
    main()