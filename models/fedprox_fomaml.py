from argparse import Namespace

from models.fedprox_maml import FedProx_MAML


class FedProx_FOMAML(FedProx_MAML):
    def __init__(self, args: Namespace, log_dict: dict = {}):
        super().__init__(args, log_dict=log_dict)
        self.inner_loop_create_graph = False
        self.log(f"[+] inner_loop_create_graph: {self.inner_loop_create_graph}")
