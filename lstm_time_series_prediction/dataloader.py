import logging
import pickle
from typing import List

import numpy as np
import torch

logger = logging.getLogger(__name__)


def _split_train_test(data: np.array, window_size=10, step_size=10, test_pct=0.3) -> List[torch.Tensor]:
    net_size = len(data[0])
    data = torch.tensor([data[:, step_size * idx: window_size + step_size * idx]
                         for idx in range((net_size - window_size) // step_size)], dtype=torch.float32).transpose(1, 2)

    train_size = int(data.shape[0] * (1 - test_pct))
    train_set = data[:train_size]
    test_set = data[train_size:]

    return [train_set[:-1], train_set[1:], test_set[:-1], test_set[1:]]


def preprocess_dataset(raw_data: np.ndarray, work_dir='../', window_size=30, step_size=30):
    """ Preprocess input raw_data, raw_data.shape = (m, n)
        n - number of features, m - number of points

    :param raw_data: Input array with data, [(src, dst, ts), ...]. Note: raw_dataset should be sorted by 'ts' key
    :param work_dir: working directory to models directory, work_dir/models
    :param window_size: input for neural network, amount of sequenced points to predict a future
    :param step_size: step time to predict future. let y - prediction, t(y) = t(x) + t(step_size)
    :return: X_train, y_train, X_test, y_test
    """

    n_features = raw_data.shape[1]
    mean = raw_data.mean(axis=1, keepdims=True)
    std = raw_data.std(axis=1, keepdims=True)
    std[std < 1e-3] = 1.0
    ip_frames_per_period = (raw_data - mean) / std

    with open('{}/models/params.pkl'.format(work_dir), 'wb') as params_file:
        pickle.dump({'mean': mean.squeeze(), 'std': std.squeeze(),
                     'window_size': window_size, 'step_size': step_size,
                     'n_features': n_features},
                    params_file)

    # TODO: add k-fold and test, eval, train sets
    return _split_train_test(ip_frames_per_period, window_size, step_size)
