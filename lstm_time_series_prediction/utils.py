import logging
import pickle
import time

import numpy as np
import torch

from lstm_time_series_prediction.model import SequenceModel

logger = logging.getLogger(__name__)

_model_path = 'models/model.pth'
_model_params_path = 'models/params.pkl'
_model = None
_model_device = 'cpu'
_model_params = {}


def time_it(func):
    def wrapper(*args, **kwargs):
        start_tick = time.time()
        result = func(*args, **kwargs)
        delta_tick = time.time() - start_tick
        logger.debug("{}: Time in method = {} seconds".format(func.__name__, round(delta_tick, 3)))
        return result

    return wrapper


@time_it
def prepare_model(work_dir, use_cuda=False):
    """ Load model from checkpoint
    :param work_dir: path to models dir -> work_dir/models/model.pth
    :param use_cuda:
    :return:
    """
    global _model, _model_params, _model_device, _model_params_path

    state_dict_path = "{}/{}".format(work_dir, _model_path)
    params_dict_path = '{}/{}'.format(work_dir, _model_params_path)

    with open(params_dict_path, 'rb') as params_file:
        _model_params = pickle.load(params_file)

    if torch.cuda.is_available() and use_cuda:
        _model_device = 'cuda'

    _model = SequenceModel(n_features=_model_params['n_features'])
    _model.load_state_dict(torch.load(state_dict_path))
    _model.eval()
    _model.to(_model_device)

    logger.debug("base: model loaded with path = {}".format(state_dict_path))


@time_it
def inference(xi: np.ndarray) -> np.ndarray:
    """ Predict future sequence. xi = x(t) -> yi = x(t) + "step_size"
    :param xi: xi is input sequence, xi.shape = (window_size, n_features)
    :return: yi is predicted sequence shifted on "step_size", yi.shape = (window_size, n_features)
    """
    global _model, _model_params, _model_device

    with torch.no_grad():
        xi = torch.tensor(xi, dtype=torch.float32).to(_model_device).unsqueeze(dim=0)
        yi = _model(xi).detach().cpu().numpy().squeeze(axis=0)

    return yi


def calculate_loss(a: np.ndarray, b: np.ndarray):
    if a is None or b is None:
        return -1.0

    return (np.square(a - b)).mean()
