from argparse import Namespace

from models.scaffold_maml import Scaffold_MAML


class Scaffold_FOMAML(Scaffold_MAML):
    def __init__(self, args: Namespace, log_dict: dict = {}):
        super().__init__(args, log_dict=log_dict)
        self.inner_loop_create_graph = False
        self.log(f"[+] inner_loop_create_graph: {self.inner_loop_create_graph}")
