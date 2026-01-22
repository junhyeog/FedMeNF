import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


@torch.no_grad()
def init_weights(m):
    if type(m) == nn.Linear:
        if hasattr(m, "weight"):
            # nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            # nn.init.xavier_normal_(m.weight, gain=nn.init.calculate_gain('leaky_relu', 0.01))
            nn.init.xavier_normal_(m.weight)
            # m.weight.fill_(0.01)
        if hasattr(m, "bias") and m.bias is not None:
            # m.bias.fill_(0.001)
            nn.init.normal_(m.bias, mean=0, std=1e-3)
            # m.bias.fill_(0.01)

    if type(m) == nn.Conv2d:
        if hasattr(m, "weight"):
            # gain = torch.nn.init.calculate_gain('relu')
            # nn.init.xavier_normal_(m.weight, gain=gain)
            # nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            nn.init.xavier_normal_(m.weight)
            # m.weight.fill_(0.01)
        if hasattr(m, "bias") and m.bias is not None:
            #    m.bias.fill_(0.01)
            nn.init.normal_(m.bias, mean=0, std=1e-3)


class FCmodel(nn.Module):
    def __init__(self, input_size, output_size, num_layers, hdim, last_act=None):
        super().__init__()

        self.fc = []
        if num_layers == 0:
            raise ValueError("num_layers should be greater than 0")
        elif num_layers == 1:
            self.fc.append(nn.Linear(input_size, output_size))
        else:
            for i in range(num_layers):
                if i == 0:
                    self.fc.append(nn.Linear(input_size, hdim))
                    self.fc.append(nn.ReLU())
                elif i == num_layers - 1:
                    self.fc.append(nn.Linear(hdim, output_size))
                else:
                    self.fc.append(nn.Linear(hdim, hdim))
                    self.fc.append(nn.ReLU())

        if last_act == "sigmoid":
            self.fc.append(nn.Sigmoid())

        self.fc = nn.Sequential(*self.fc)

    def forward(self, x):
        y = self.fc(x)
        return y


def conv3x3(in_channels, out_channels, **kwargs):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, **kwargs),
        # nn.BatchNorm2d(out_channels, track_running_stats=False),
        nn.ReLU(),
        nn.MaxPool2d(2),
    )
