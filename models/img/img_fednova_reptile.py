from argparse import Namespace
from typing import List

from models.img.img_fedavg_reptile import Img_FedAvg_Reptile


class Img_FedNova_Reptile(Img_FedAvg_Reptile):
    def __init__(self, args: Namespace, log_dict: dict = {}):
        super().__init__(args, log_dict=log_dict)

    def server_aggregation(self, client_ids: List[int]):
        return self.server_aggregation_fednova(client_ids)
