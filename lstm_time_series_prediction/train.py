import pathlib
import pickle
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from tqdm import tqdm

from lstm_time_series_prediction.model import SequenceModel


def train(
        datasets,
        lr=3e-1,
        n_epochs=100,
        work_dir='../',
        use_cuda=False
):
    train_input, train_target, test_input, test_target = datasets
    n_features = train_input.shape[-1]

    model_device = 'cuda' if torch.cuda.is_available() and use_cuda else 'cpu'

    model = SequenceModel(n_features=n_features, device=model_device).to(model_device)
    criterion = nn.SmoothL1Loss()
    optimizer = optim.LBFGS(model.parameters(), lr=lr)

    printable_time = datetime.now().strftime('%m%d %H%M%S')
    log_dir = '{}/meta/logs/{}_lr={}'.format(work_dir, printable_time, lr)
    model_dir = '{}/meta/models/{}_lr={}'.format(work_dir, printable_time, lr)
    pathlib.Path(model_dir).mkdir(parents=True, exist_ok=True)

    params_dir = '{}/models'.format(work_dir)
    with open('{}/params.pkl'.format(params_dir), 'rb') as params_file:
        data_params = pickle.load(params_file)

    min_test_loss = -1
    with SummaryWriter(log_dir) as writer:
        for epoch in tqdm(range(n_epochs)):

            train_loss = _epoch_model_train(optimizer, train_input, train_target, model, criterion, model_device)
            y, test_loss = _epoch_model_eval(criterion, test_input, test_target, model, model_device)
            figure = _epoch_save_results(y, test_target, test_input, data_params)

            writer.add_figure('data/visual', figure, epoch)
            writer.add_scalars('data/losses', {'train': train_loss, 'test': test_loss}, epoch)

            if test_loss < min_test_loss or epoch == 0:
                min_test_loss = test_loss
                torch.save(model.state_dict(), '{}/model_{}_{}.pth'.format(model_dir, epoch, round(min_test_loss, 3)))


def _epoch_save_results(y, y_gt, y_in, params, n_samples=5):
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    colors = ['r', 'g', 'b', 'y', 'm', 'c', 'k']
    n_features = y.shape[-1]
    window_size = y.shape[1]
    n_samples = min([n_samples, y.shape[0]])

    gs = GridSpec(n_features, 1)
    figure = plt.figure(figsize=(20, 10))

    for func_idx in range(n_features):
        ax = figure.add_subplot(gs[func_idx])
        ax.grid(True)

        for idx, (sample_idx, color) in enumerate(zip(range(n_samples), colors)):
            data = y[sample_idx, :, func_idx] * params['std'][func_idx] + params['mean'][func_idx]
            data_in = y_in[sample_idx, :, func_idx].numpy() * params['std'][func_idx] + params['mean'][func_idx]
            data_gt = y_gt[sample_idx, :, func_idx].numpy() * params['std'][func_idx] + params['mean'][func_idx]

            ax.plot(np.arange(-window_size + 1, 1), data_in, color, linewidth=4.0, label='input')
            ax.plot(np.arange(window_size // 2), data_gt[window_size // 2:], color + '--',
                    linewidth=3.0, label='ground true')
            ax.plot(np.arange(window_size // 2), data[window_size // 2:], color + ':',
                    linewidth=2.0, label='prediction')

            if not idx:
                ax.legend(loc='upper left')

    return figure


def _epoch_model_eval(criterion, m_test_input, m_test_target, model, device):
    model.eval()
    with torch.no_grad():
        pred = model(m_test_input.to(device))
        loss = criterion(pred, m_test_target.to(device))
        y = pred.detach().cpu().numpy()

    test_loss = loss.item()
    return y, test_loss


def _epoch_model_train(optimizer, m_train_input, m_train_target, model, criterion, device):
    def closure():
        optimizer.zero_grad()

        out = model(m_train_input.to(device))
        loss = criterion(out, m_train_target.to(device))
        losses.append(loss.item())
        loss.backward()
        return loss

    model.train()
    losses = []
    optimizer.step(closure)

    train_loss = sum(losses) / len(losses)
    return train_loss
