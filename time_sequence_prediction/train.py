import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib.gridspec import GridSpec

from time_sequence_generation.model import SequenceModel

matplotlib.use('Agg')


def generate_train_test(train_pct=0.9, window_size=64, step_size=32, net_size=25600):
    x = np.linspace(0, step_size * window_size, net_size)

    y1 = np.sin(x + 1)
    y2 = np.cos(2 * x)
    y3 = (x % 4) / 4
    y4 = (16 - ((x + 2) % 4) ** 2) / 16
    y5 = (x % 2) > 1

    y = np.array([y1, y2, y3, y4, y5])

    data = torch.tensor([y[:, step_size * idx: window_size + step_size * idx]
                         for idx in range((net_size - window_size) // step_size)], dtype=torch.float32).transpose(1, 2)

    train_size = int(data.shape[0] * train_pct)
    y_train = data[:train_size]
    y_test = data[train_size:]

    return y_train[:-1], y_train[1:], y_test[:-1], y_test[1:]


def train():
    train_input, train_target, test_input, test_target = generate_train_test()

    model = SequenceModel().cuda()

    criterion = nn.MSELoss()
    optimizer = optim.LBFGS(model.parameters(), lr=0.2)

    n_epochs = 100
    future = 10
    n_samples = 4

    for epoch in range(n_epochs):
        print(f"Epoch {epoch + 1} out of {n_epochs}")

        _epoch_model_train(optimizer, train_input, train_target, model, criterion)
        y = _epoch_model_eval(criterion, test_input, test_target, model, future)
        _epoch_save_results(epoch, y, test_target, test_input, future, n_samples)


def _epoch_save_results(epoch, y, y_gt, y_in, future, n_samples):
    colors = ['r', 'g', 'b', 'y', 'm', 'c', 'k']
    n_functions = y.shape[-1]
    window_size = y.shape[1] - future

    gs = GridSpec(n_functions, 1)
    figure = plt.figure(figsize=(20, 10))

    for func_idx in range(n_functions):
        ax: plt.Axes = figure.add_subplot(gs[func_idx])
        ax.grid(True)

        for idx, (sample_idx, color) in enumerate(zip(range(n_samples), colors)):
            data = y[sample_idx, :, func_idx]
            data_in = y_in[sample_idx, :, func_idx].tolist()
            data_gt = y_gt[sample_idx, :, func_idx].tolist()

            ax.plot(np.arange(-window_size + 1, 1), data_in, color, linewidth=4.0, label='input')
            ax.plot(np.arange(window_size // 2), data_gt[window_size // 2:], color + '--',
                    linewidth=3.0, label='ground true')
            ax.plot(np.arange(window_size // 2), data[window_size // 2:-future], color + ':',
                    linewidth=2.0, label='prediction')

            if not idx:
                ax.legend(loc='upper left')

    figure.savefig(f'images/predict {epoch}.png')
    plt.close()


def _epoch_model_eval(criterion, m_test_input, m_test_target, model, future):
    model.eval()
    with torch.no_grad():
        pred = model(m_test_input.cuda(), future=future)
        loss = criterion(pred[:, :-future], m_test_target.cuda())
        print('test loss:', loss.item())
        y = pred.detach().cpu().numpy()

    return y


def _epoch_model_train(optimizer, m_train_input, m_train_target, model, criterion):
    model.train()

    def closure():
        optimizer.zero_grad()

        out = model(m_train_input.cuda())
        loss = criterion(out, m_train_target.cuda())
        print('loss:', loss.item())
        loss.backward()
        return loss

    optimizer.step(closure)


if __name__ == '__main__':
    train()
