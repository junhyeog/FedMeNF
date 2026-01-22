from argparse import Namespace

from models.img.img_fedprox_maml import Img_FedProx_MAML


class Img_FedProx_FOMAML(Img_FedProx_MAML):
    def __init__(self, args: Namespace, log_dict: dict = {}):
        super().__init__(args, log_dict=log_dict)
        self.inner_loop_create_graph = False
        self.log(f"[+] inner_loop_create_graph: {self.inner_loop_create_graph}")
