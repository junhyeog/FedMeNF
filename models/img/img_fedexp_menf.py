from argparse import Namespace
from typing import List

from models.img.img_fedavg_menf import Img_FedAvg_MeNF


class Img_FedExP_MeNF(Img_FedAvg_MeNF):
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
