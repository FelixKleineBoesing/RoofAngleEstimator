import abc

import tensorflow as tf
import numpy as np
from typing import Union
import pandas as pd


arraylike = Union[pd.Series, np.ndarray, list]


def create_train_test_splits(*args: Union[np.ndarray, tf.Tensor, list], ratio: float = 0.3, ):
    assert len(args) > 0, "please supply at least one data container"
    len_arr = get_number_obs(args[0])
    for arg in args:
        assert len_arr == get_number_obs(arg), "All data containers must have the same shape or length"
    test_size = int(ratio * len_arr)
    indices = np.random.choice(np.arange(len_arr), size=test_size, replace=False)
    mask = np.ones(len_arr, dtype=bool)
    mask[indices] = False

    splits = []
    for arg in args:
        if isinstance(arg, list):
            arg = np.array(arg)
            split = arg[mask].to_list(), arg[~mask].to_list()
        else:
            split = arg[mask], arg[~mask]
        splits.append(split)
    if len(args) == 1:
        return splits[0]
    else:
        return splits


def get_number_obs(obj):
    if hasattr(obj, "shape"):
        return obj.shape[0]
    elif hasattr(obj, "__len__"):
        return len(obj)
    else:
        raise ValueError("Could not get number observations. Obj has no attribute length or shape")


class Scaler:

    @abc.abstractmethod
    def transform(self, x) -> (arraylike, dict):
        pass

    @abc.abstractmethod
    def reform(self, x, **kwargs) -> arraylike:
        pass


class Transformer:

    def __init__(self, scaler: Scaler, column_wise: bool = True):
        self.memory = []
        self.scaler = scaler
        self.column_wise = column_wise

    def transform(self, X):
        if self.column_wise:
            for j in range(X.shape[1]):
                X[:, j], params = self.scaler.transform(X[:, j])
                self.memory.append(params)
        else:
            X[:], params = self.scaler.transform(X)
            self.memory.append(params)

    def reform(self, X):
        if self.column_wise:
            for j in range(X.shape[1]):
                X[:, j] = self.scaler.reform(X[:, j], **self.memory[j])
        else:
            X[:] = self.scaler.reform(X, **self.memory[0])


class _Standardizer(Scaler):

    def transform(self, x):
        mu = np.mean(x)
        sd = np.std(x)
        return (x - mu) / sd, {"mu": mu, "sd": sd}

    def reform(self, x, mu, sd):
        return x * sd + mu


class _MinMaxScaler(Scaler):

    def transform(self, x):
        min_ = np.min(x)
        max_ = np.max(x)
        return (x - min_) / (max_ - min_), {"min_": min_, "max_": max_}

    def reform(self, x, min_, max_):
        return x * (max_ - min_) + min_


class StandardizationTransformer(Transformer):

    def __init__(self, column_wise: bool = True):
        super().__init__(scaler=_Standardizer(), column_wise=column_wise)


class MinMaxTransformer(Transformer):

    def __init__(self, column_wise: bool = True):
        super().__init__(scaler=_MinMaxScaler(), column_wise=column_wise)

