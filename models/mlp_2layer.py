import torch.nn as nn


class TwoLayerMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden1 = nn.Linear(20, 8)
        self.hidden2 = nn.Linear(8, 10)
        self.output = nn.Linear(10, 1)

        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.tanh(self.hidden1(x))
        x = self.tanh(self.hidden2(x))
        x = self.sigmoid(self.output(x))

        return x

    def __repr__(self):
        return "TwoLayerMLP"
