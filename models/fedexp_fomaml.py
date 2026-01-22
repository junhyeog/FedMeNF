from argparse import Namespace

from models.fedexp_maml import FedExP_MAML


class FedExP_FOMAML(FedExP_MAML):
    def __init__(self, args: Namespace, log_dict: dict = {}):
        super().__init__(args, log_dict=log_dict)
        self.inner_loop_create_graph = False
        self.log(f"[+] inner_loop_create_graph: {self.inner_loop_create_graph}")
