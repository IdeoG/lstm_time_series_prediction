import torch
import torch.nn as nn


class SequenceModel(nn.Module):

    def __init__(self, n_features=5, hidden_size=128):

        super(SequenceModel, self).__init__()

        self.n_features = n_features
        self.hidden_size = hidden_size

        self.lstm1 = nn.LSTMCell(self.n_features, self.hidden_size)
        self.lstm2 = nn.LSTMCell(self.hidden_size, self.hidden_size)
        self.linear = nn.Linear(self.hidden_size, self.n_features)

    def forward(self, x: torch.Tensor, future=0):
        outputs = []

        h_t = torch.zeros(x.size(0), self.hidden_size).cuda()
        c_t = torch.zeros(x.size(0), self.hidden_size).cuda()
        h_t2 = torch.zeros(x.size(0), self.hidden_size).cuda()
        c_t2 = torch.zeros(x.size(0), self.hidden_size).cuda()

        for i, input_t in enumerate(x.chunk(x.size(1), dim=1)):
            input_t = input_t.squeeze(dim=1)

            h_t, c_t = self.lstm1(input_t, (h_t, c_t))
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))

            output = self.linear(h_t2)
            outputs += [output]

        for i in range(future):
            h_t, c_t = self.lstm1(output, (h_t, c_t))
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))

            output = self.linear(h_t2)
            outputs += [output]

        outputs = torch.stack(outputs, 1)
        return outputs
