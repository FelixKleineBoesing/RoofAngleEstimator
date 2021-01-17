import pandas as pd
import numpy as np
from typing import Union

from tensorflow.python.data import Dataset
import tensorflow as tf
from tensorflow.keras.utils import normalize
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.losses import mse
from tensorflow.keras.optimizers import Optimizer, Adam
from tensorflow.keras.activations import linear
from src.helpers import create_train_test_splits, MinMaxTransformer


class AngleEstimator(Model):

    def __init__(self, number_sides: int = 2, number_epochs: int = 1000, batch_size: int = 2048,
                 optimizer: Optimizer = None):
        super().__init__()
        if optimizer is None:
            optimizer = Adam(learning_rate=tf.Variable(0.1),
                             beta_1=tf.Variable(0.9),
                             beta_2=tf.Variable(0.999),
                             epsilon=tf.Variable(1e-7)
                             )
            optimizer.iterations
        self.optimizer = optimizer
        self.number_sides = number_sides
        self.input_layer = Dense(2, activation=linear, use_bias=False)
        self.hidden_layer = Dense(number_sides, activation=linear, use_bias=False)
        self.output_layer = Dense(1, activation=linear, use_bias=False)
        self.batch_size = batch_size
        self.number_epochs = number_epochs

    def call(self, inputs):
        x = self.input_layer(inputs)
        #x = self.hidden_layer(x)
        return self.output_layer(x)

    def _train_step(self, inputs, output):
        with tf.GradientTape() as g:
            z = self.call(inputs)
            loss = mse(output, z)

        gradients = g.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        return tf.reduce_sum(loss)

    def get_side_angles(self, x_y, z):
        #data = np.concatenate((x_y, z.reshape((-1, 1))), axis=1)
        #transformer = MinMaxTransformer()
        #transformer.transform(data)
        #x_y, z = data[:, 0:2], data[:, 2]
        input_splits, output_splits = create_train_test_splits(x_y, z, ratio=0.0)
        buffer_size = input_splits[0].shape[0]
        dataset = Dataset.from_tensor_slices((input_splits[0], output_splits[0])).shuffle(buffer_size=buffer_size)
        dataset = dataset.batch(self.batch_size)
        steps_per_epoch = max(buffer_size // self.batch_size, 1)

        for epoch in range(self.number_epochs):
            for (batch, (input, target)) in enumerate(dataset.take(steps_per_epoch)):
                loss = self._train_step(input, target)
                print("Epoch: {} Batch: {} Loss: {}".format(epoch, batch, loss))
        z_hat = self.call(x_y).numpy()
