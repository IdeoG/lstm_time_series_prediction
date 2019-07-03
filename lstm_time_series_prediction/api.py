import logging

import numpy as np

from lstm_time_series_prediction import utils, train, dataloader

logger = logging.getLogger(__name__)


class TimeSeriesPrediction:

    def __init__(self, work_dir='../', use_cuda=False):
        if utils._model is None or work_dir != '../':
            utils.prepare_model(work_dir, use_cuda)
        model_params = utils._model_params

        self.period = model_params['period']
        self.step_size = model_params['step_size']
        self.window_size = model_params['window_size']

        self.yi_prev = None

    @staticmethod
    def train(
            raw_dataset: np.ndarray,
            lr=1e-2,
            n_epochs=200,
            work_dir='../',
            window_size=30,
            step_size=30,
            use_cuda=False):
        """ Train model

        :param raw_dataset: Input array with data, raw_dataset.shape = (m, n),
                            where n - number of features, m - number of points
        :param work_dir: working directory to models directory, work_dir/models
        :param window_size: input for neural network, amount of sequenced points to predict a future
        :param step_size: step time to predict future. let y - prediction, t(y) = t(x) + t(step_size)
        :param lr: Learning rate during train phase
        :param n_epochs: Number of epochs to train
        :param use_cuda:
        """

        datasets = dataloader.preprocess_dataset(raw_dataset, work_dir, window_size=window_size, step_size=step_size)
        train.train(datasets, lr, n_epochs, work_dir, use_cuda)

    def inference(self, xi: np.ndarray):
        """ Predict future sequence. xi = x(t) -> yi = t + "step_size"

        :param xi: Input array with data, xi.shape = (window_size, n_features)
        :return: yi is predicted sequence shifted at "step_size", yi.shape = (step_size, n_features)
        """

        loss = utils.calculate_loss(self.yi_prev, xi[:self.step_size])

        yi = utils.inference(xi)[self.window_size - self.step_size:]
        self.yi_prev = yi
        return xi, yi, loss
