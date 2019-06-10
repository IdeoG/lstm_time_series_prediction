import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib.gridspec import GridSpec

from model import Sequence

matplotlib.use('Agg')


def generate_train_test(train_pct=0.7, window_size=200, step_size=20, net_size=10000):
    x = np.linspace(0, 200, net_size)

    y1 = np.sin(x)
    y2 = np.sin(2 * x)
    y3 = x % 10

    y = np.array([y1, y2, y3])

    data = torch.tensor([y[:, step_size * idx: window_size + step_size * idx]
                         for idx in range((net_size - window_size) // step_size)]).transpose(1, 2)

    train_size = int(data.shape[0] * train_pct)
    y_train = data[:train_size]
    y_test = data[train_size:]

    return y_train[:-1], y_train[1:], y_test[:-1], y_test[1:]


def _epoch_save_results(epoch, y, y_gt, future):
    n_samples = 4
    colors = ['r', 'g', 'b', 'y']
    n_functions = 3
    gs = GridSpec(n_functions, 1)
    figure = plt.figure(figsize=(30, 20))

    for func_idx in range(n_functions):
        ax: plt.Axes = figure.add_subplot(gs[func_idx])

        for sample_idx, color in zip(range(n_samples), colors):
            data = y[sample_idx, :, func_idx]
            data_gt = y_gt[sample_idx, :, func_idx].tolist()

            ax.plot(np.arange(len(data[:-future])), data[:-future], color + '--', linewidth=2.0)
            ax.plot(np.arange(len(data_gt)), data_gt, color, linewidth=2.0)
            ax.plot(np.arange(len(data[:-future]), len(data[:-future]) + future), data[-future:], color + ':',
                    linewidth=2.0)

    figure.savefig(f'images/double predict {epoch}.png')
    plt.close()


def train():
    m_train_input, m_train_target, m_test_input, m_test_target = generate_train_test()

    model = Sequence()
    model.double().cuda()

    criterion = nn.MSELoss()
    optimizer = optim.LBFGS(model.parameters(), lr=1)

    n_epoches = 100

    for epoch in range(n_epoches):
        print(f"Epoch {epoch + 1} out of {n_epoches}")
        future = 10

        _epoch_model_train(optimizer, m_train_input, m_train_target, model, criterion)
        y = _epoch_model_eval(criterion, m_test_input, m_test_target, model, future)
        _epoch_save_results(epoch, y, m_test_target, future)


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
