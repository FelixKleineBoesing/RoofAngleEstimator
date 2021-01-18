from pathlib import Path
import numpy as np
from typing import Tuple
from tensorflow.python.data import Dataset
import tensorflow as tf
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.losses import mse
from tensorflow.keras.optimizers import Optimizer, Adam
from tensorflow.keras.activations import linear, sigmoid
from tensorflow.keras.callbacks import TensorBoard

from src.helpers import create_train_test_splits, MinMaxTransformer
from src.visualizations import get_3d_scatter_with_prediction_planes


class AngleEstimator(Model):

    def __init__(self, number_sides: int = 2, number_epochs: int = 1000, batch_size: int = 32,
                 optimizer: Optimizer = None, tensorboard_dir: Path = None):
        super().__init__()
        if optimizer is None:
            optimizer = Adam(
                learning_rate=tf.Variable(0.1, dtype=tf.float64),
                beta_1=tf.Variable(0.9, dtype=tf.float64),
                beta_2=tf.Variable(0.999, dtype=tf.float64),
                epsilon=tf.Variable(1e-7, dtype=tf.float64)
            )
            optimizer.iterations
        self.optimizer = optimizer
        self.number_sides = number_sides
        self.input_layer = Dense(2, activation=linear, use_bias=False, dtype=tf.float64)
        self.hidden_layer = Dense(number_sides + 10, activation=linear, use_bias=False, dtype=tf.float64)
        self.output_layer = Dense(1, activation=linear, use_bias=False, dtype=tf.float64)
        self.batch_size = batch_size
        self.number_epochs = number_epochs
        self.tensorboard_dir = tensorboard_dir
        self.build((None, 2))
        self._setup_tensorboard_callback()

    def _setup_tensorboard_callback(self):
        if self.tensorboard_dir is not None:
            self.tensorboard_callback = TensorBoard(self.tensorboard_dir)
            self.tensorboard_callback.set_model(self)

    def call(self, inputs):
        x = self.input_layer(inputs)
        x = self.hidden_layer(x)
        y = self.output_layer(x)
        return y

    def _train_step(self, inputs, output):
        with tf.GradientTape() as g:
            z = self.call(inputs)
            loss = mse(output, z)

        gradients = g.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        return tf.reduce_sum(loss)

    def get_side_angles(self, x_y, z):
        data = np.concatenate((x_y, np.reshape(z, (-1, 1))), axis=1)
        transformer = MinMaxTransformer(column_wise=False)
        transformer.transform(data)
        x_y, z = data[:, 0:2], data[:, 2]
        input_splits, output_splits = create_train_test_splits(x_y, z, ratio=0.0)
        buffer_size = input_splits[0].shape[0]
        dataset = Dataset.from_tensor_slices((input_splits[0], output_splits[0])).shuffle(buffer_size=buffer_size)
        dataset = dataset.batch(self.batch_size)
        steps_per_epoch = max(buffer_size // self.batch_size, 1)
        self._train(dataset=dataset, steps_per_epoch=steps_per_epoch)

    def _train(self, dataset, steps_per_epoch):
        for epoch in range(self.number_epochs):
            for (batch, (input, target)) in enumerate(dataset.take(steps_per_epoch)):
                loss = self._train_step(input, target)
                print("Epoch: {} Batch: {} Loss: {}".format(epoch, batch, loss))

    def plot_3d_prediction_plot(self, data, plane_steps: int = 50):
        transformer = MinMaxTransformer(column_wise=False)
        transformer.transform(data)
        plane_input = self._get_prediction_array(x_range=(np.min(data[:, 0]), np.max(data[:, 0])),
                                                 y_range=(np.min(data[:, 1]), np.max(data[:, 1])),
                                                 steps=plane_steps)
        z_hat = self.call(plane_input).numpy()
        plane_data = np.concatenate((plane_input, z_hat), axis=1)
        return get_3d_scatter_with_prediction_planes(data, plane_data)

    def _get_prediction_array(self, x_range: Tuple, y_range: Tuple, steps: int = 50):
        prediction_array = np.zeros((steps ** 2, 2))
        x_stepsize = self._get_step_size(x_range, steps)
        y_stepsize = self._get_step_size(y_range, steps)
        x = x_range[0]
        k = 0
        for i in range(steps):
            y = y_range[0]
            for j in range(steps):
                prediction_array[k, 0] = x
                prediction_array[k, 1] = y
                k += 1
                y += y_stepsize
            x += x_stepsize

        return prediction_array

    @staticmethod
    def _get_step_size(axis_range: Tuple, steps: int):
        return (axis_range[1] - axis_range[0]) / steps