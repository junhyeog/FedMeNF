from collections import OrderedDict
from copy import deepcopy
from typing import List

import torch

from utils.misc import copy_model_params


class IdentityNet(torch.nn.Module):
    def __init__(self, base_model: torch.nn.Module):
        super().__init__()
        self.identity = torch.nn.Identity()
        self.base_model_params = copy_model_params(base_model.state_dict(), requires_grad=False)
        self.params = torch.nn.ParameterList(
            [p.clone().detach().requires_grad_(True) for n, p in self.base_model_params.items()]
        )
        self.param_keys = list(self.base_model_params.keys())

    def forward(self):
        # return copy_model_params(self.params)
        params = OrderedDict(zip(self.param_keys, self.params))
        return params
