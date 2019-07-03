import logging
import pathlib
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

    :param raw_data: Input array with data
    :param work_dir: working directory to models directory, work_dir/models
    :param window_size: input for neural network, amount of sequenced points to predict a future
    :param step_size: step time to predict future. let y - prediction, t(y) = t(x) + t(step_size)
    :return: X_train, y_train, X_test, y_test
    """

    n_features = raw_data.shape[1]
    params_dir = '{}/models'.format(work_dir)
    pathlib.Path(params_dir).mkdir(parents=True, exist_ok=True)

    with open('{}/params.pkl'.format(params_dir), 'wb') as params_file:
        pickle.dump({'window_size': window_size, 'step_size': step_size, 'n_features': n_features},
                    params_file)

    # TODO: add k-fold and test, eval, train sets
    return _split_train_test(raw_data, window_size, step_size)
