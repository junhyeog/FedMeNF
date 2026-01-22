from argparse import Namespace
from collections import OrderedDict

import torch

from models.img.img_fedavg_maml import Img_FedAvg_MAML


class Img_Scaffold_MAML(Img_FedAvg_MAML):
    def __init__(self, args: Namespace, log_dict: dict = {}):
        super().__init__(args, log_dict=log_dict)

        # Scaffold
        self.c_global = OrderedDict({k: torch.zeros_like(p) for k, p in self.hpnet().items()})
        self.c_locals = OrderedDict(
            {
                i: OrderedDict({k: torch.zeros_like(p) for k, p in self.c_global.items()})
                for i in range(len(self.clients))
            }
        )
        self.c_deltas = OrderedDict(
            {
                i: OrderedDict({k: torch.zeros_like(p) for k, p in self.c_global.items()})
                for i in range(len(self.clients))
            }
        )

        self.client_update = self.get_client_update_scaffold_fn(self.client_update)
        self.server_aggregation = self.server_aggregation_scaffold
