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


class Img_FedAvg_MeNF(Img_FedAvg_MAML):
    def __init__(self, args: Namespace, log_dict: dict = {}):
        _log_dict = {
            "triplet_gamma": "tg",
        }
        log_dict.update(_log_dict)
        super().__init__(args, log_dict)

        self.triplet_gamma = args.triplet_gamma

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

        # * positive loss
        loss_pos = torch.nn.functional.mse_loss(rgbs, pixels)

        # * negative loss ! (global <-> gt)
        global_params_batch = OrderedDict({n: repeat(p, "... -> b ...", b=task_bs) for n, p in client_params.items()})
        g_rgbs = self.functional_batch(global_params_batch, coords)

        loss_neg = torch.nn.functional.mse_loss(g_rgbs, pixels)

        # * menf loss
        loss = loss_pos - self.triplet_gamma * loss_neg

        grad = torch.autograd.grad(loss, client_params.values(), create_graph=False)

        clip_norm_(grad, gradclip)
        for (n, p), g in zip(client_params.items(), grad):
            client_params[n] = p - lr * g

        metric.update(loss=loss.item(), loss_pos=loss_pos.item(), loss_neg=loss_neg.item())

        return client_params, metric
