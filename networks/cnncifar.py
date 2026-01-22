import torch
import torch.nn as nn
from networks.basenetwork import init_weights, conv3x3


class CNNCifar(nn.Module):
    def __init__(self, out_dim: int, in_channels=3, hidden_dim=64):
        super(CNNCifar, self).__init__()

        self.features = nn.Sequential(
            conv3x3(in_channels, hidden_dim),
            conv3x3(hidden_dim, hidden_dim),
            conv3x3(hidden_dim, hidden_dim),
        )

        self.linears = nn.Sequential(
            nn.Linear(hidden_dim * 2 * 2 * 4, hidden_dim * 2 * 2),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Linear(hidden_dim * 2 * 2, hidden_dim * 2),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Linear(hidden_dim * 2, out_dim),
        )

        self.apply(init_weights)

    def forward(self, x):
        features = self.extract_features(x)
        logits = self.linears(features)
        return logits

    def extract_features(self, x):
        return self.features(x).flatten(1)
