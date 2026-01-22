from argparse import Namespace
from typing import List

from models.fedavg_menf import FedAvg_MeNF


class FedNova_MeNF(FedAvg_MeNF):
    def __init__(self, args: Namespace, log_dict: dict = {}):
        super().__init__(args, log_dict=log_dict)

    def server_aggregation(self, client_ids: List[int]):
        return self.server_aggregation_fednova(client_ids)
