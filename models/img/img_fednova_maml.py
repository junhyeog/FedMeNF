from argparse import Namespace
from typing import List

from models.img.img_fedavg_maml import Img_FedAvg_MAML


class Img_FedNova_MAML(Img_FedAvg_MAML):
    def __init__(self, args: Namespace, log_dict: dict = {}):
        super().__init__(args, log_dict=log_dict)

    def server_aggregation(self, client_ids: List[int]):
        return self.server_aggregation_fednova(client_ids)
