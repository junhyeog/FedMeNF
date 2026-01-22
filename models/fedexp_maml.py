from argparse import Namespace
from typing import List

from models.fedavg_maml import FedAvg_MAML


class FedExP_MAML(FedAvg_MAML):
    def __init__(self, args: Namespace, log_dict: dict = {}):
        _log_dict = {
            "fedexp_epsilon": "fe",
            "fedexp_type": "ty",
        }
        log_dict.update(_log_dict)
        super().__init__(args, log_dict=log_dict)

        self.fedexp_epsilon = args.fedexp_epsilon

    def server_aggregation(self, client_ids: List[int]):
        return self.server_aggregation_fedexp(client_ids)
