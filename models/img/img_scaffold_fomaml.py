from argparse import Namespace

from models.img.img_scaffold_maml import Img_Scaffold_MAML


class Img_Scaffold_FOMAML(Img_Scaffold_MAML):
    def __init__(self, args: Namespace, log_dict: dict = {}):
        super().__init__(args, log_dict=log_dict)
        self.inner_loop_create_graph = False
        self.log(f"[+] inner_loop_create_graph: {self.inner_loop_create_graph}")
