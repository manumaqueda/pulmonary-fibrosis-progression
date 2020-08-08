import torch.nn as nn
import torch.nn.functional as F


class QuantileModel(nn.Module):
    def __init__(self, in_tabular_features=9, out_quantiles=3):
        super(QuantileModel, self).__init__()
        self.fc1 = nn.Linear(in_tabular_features, 100)
        self.fc2 = nn.Linear(100, 100)
        self.fc3 = nn.Linear(100, out_quantiles)

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = self.fc3(x)
        return x