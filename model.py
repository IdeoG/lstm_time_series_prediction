import torch
import torch.nn as nn


class Sequence(nn.Module):
    def __init__(self):
        super(Sequence, self).__init__()
        self.lstm1 = nn.LSTMCell(3, 101)
        self.lstm2 = nn.LSTMCell(101, 101)
        self.linear = nn.Linear(101, 3)

    def forward(self, input: torch.Tensor, future=0):
        outputs = []
        h_t = torch.zeros(input.size(0), 101, dtype=torch.double).cuda()
        c_t = torch.zeros(input.size(0), 101, dtype=torch.double).cuda()
        h_t2 = torch.zeros(input.size(0), 101, dtype=torch.double).cuda()
        c_t2 = torch.zeros(input.size(0), 101, dtype=torch.double).cuda()

        for i, input_t in enumerate(input.chunk(input.size(1), dim=1)):
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
