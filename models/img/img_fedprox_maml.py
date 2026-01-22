from argparse import Namespace
from collections import OrderedDict
from typing import List

import torch
from einops import rearrange, repeat

from hypernets import IdentityNet
from models.img.img_fedavg_maml import Img_FedAvg_MAML
from networks import SirenNet
from utils.clipping import clip_norm_
from utils.logger import MetricLogger
from utils.misc import copy_model_params


class Img_FedProx_MAML(Img_FedAvg_MAML):
    def __init__(self, args: Namespace, log_dict: dict = {}):
        _log_dict = {
            "fedprox_mu": "fm",
        }
        log_dict.update(_log_dict)
        super().__init__(args, log_dict)

        self.fedprox_mu = args.fedprox_mu

    def outer_loop_batch(
        self,
        client_params,
        client_params_batch,
        ray_bs,
        gradclip,
        lr,
        task_data,
        client_id,
    ):
        metric = MetricLogger()

        task_id, s_set, q_set = task_data
        task_bs, v_q, h_q, w_q, in_size_q = q_set.shape

        coords, pixels = self.sample_coords_batch(s_set, ray_bs)

        rgbs = self.functional_batch(client_params_batch, coords)
        loss = torch.nn.functional.mse_loss(rgbs, pixels)

        # FedProx
        global_params = copy_model_params(self.hpnet(), requires_grad=False)
        global_params_flat = torch.cat([repeat(w, "... -> b ...", b=task_bs).view(-1) for w in global_params.values()])
        proximal = (
            self.fedprox_mu
            * 0.5
            * torch.norm(torch.cat([w.view(-1) for w in client_params.values()]) - global_params_flat) ** 2
        )
        loss += proximal

        grad = torch.autograd.grad(loss, client_params.values(), create_graph=False)

        clip_norm_(grad, gradclip)
        for (n, p), g in zip(client_params.items(), grad):
            client_params[n] = p - lr * g

        metric.update(loss=loss.item())

        return client_params, metric
