import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib.gridspec import GridSpec

from time_series_generation.model import Sequence

matplotlib.use('Agg')


def generate_train_test(train_pct=0.9, window_size=64, step_size=32, net_size=25600):
    x = np.linspace(0, step_size * window_size // 2, net_size)

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


def _epoch_save_results(epoch, y, y_gt, y_in, future):
    n_samples = 4
    colors = ['r', 'g', 'b', 'y']
    n_functions = 5
    gs = GridSpec(n_functions, 1)
    figure = plt.figure(figsize=(30, 20))

    for func_idx in range(n_functions):
        ax: plt.Axes = figure.add_subplot(gs[func_idx])

        for sample_idx, color in zip(range(n_samples), colors):
            data = y[sample_idx, :, func_idx]
            data_in = y_in[sample_idx, :, func_idx].tolist()
            data_gt = y_gt[sample_idx, :, func_idx].tolist()

            ax.plot(np.arange(-len(data_in), 0), data_in, color, linewidth=2.0)
            ax.plot(np.arange(len(data_gt) // 2), data_gt[:len(data_gt) // 2], color + '--', linewidth=2.0)

            ax.plot(np.arange(-len(data[:-future]) // 2, len(data[:-future]) // 2), data[:-future], color + ':',
                    linewidth=2.0)
            ax.grid(True)

    figure.savefig(f'images/double predict {epoch}.png')
    plt.close()


def train():
    m_train_input, m_train_target, m_test_input, m_test_target = generate_train_test()

    model = Sequence().cuda()

    criterion = nn.MSELoss()
    optimizer = optim.LBFGS(model.parameters(), lr=0.2)

    n_epoches = 100

    for epoch in range(n_epoches):
        print(f"Epoch {epoch + 1} out of {n_epoches}")
        future = 10

        _epoch_model_train(optimizer, m_train_input, m_train_target, model, criterion)
        y = _epoch_model_eval(criterion, m_test_input, m_test_target, model, future)
        _epoch_save_results(epoch, y, m_test_target, m_test_input, future)


def _epoch_model_eval(criterion, m_test_input, m_test_target, model, future):
    with torch.no_grad():
        pred = model(m_test_input.cuda(), future=future)
        loss = criterion(pred[:, :-future], m_test_target.cuda())
        print('test loss:', loss.item())
        y = pred.detach().cpu().numpy()

    return y


def _epoch_model_train(optimizer, m_train_input, m_train_target, model, criterion):
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
