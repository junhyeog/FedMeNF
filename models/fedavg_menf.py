from argparse import Namespace
from collections import OrderedDict
from typing import List

import numpy as np
import torch
from einops import rearrange, repeat
from torch.func import functional_call, vmap

from hypernets import IdentityNet
from models.fedavg_maml import FedAvg_MAML
from networks import SimpleNeRF
from utils.clipping import clip_norm_
from utils.logger import MetricLogger
from utils.misc import copy_model_params, make_grid
from utils.rendering import (get_rays_shapenet_batch, sample_points_batch,
                             tensor_linspace, test_model_batch,
                             volume_render_batch)

# from models import BaseModel, FedAvg_MAML, FedAvg_Reptile


class FedAvg_MeNF(FedAvg_MAML):
    def __init__(self, args: Namespace, log_dict: dict = {}):
        _log_dict = {
            "triplet_gamma": "tg",
        }
        log_dict.update(_log_dict)
        super().__init__(args, log_dict=log_dict)  # should be first

        self.triplet_gamma = args.triplet_gamma

    def outer_loop_batch(
        self,
        client_params,
        client_params_batch,
        ray_bs,
        num_points_per_ray,
        gradclip,
        lr,
        imgs,
        poses,
        hwf,
        bound,
        client_id,
    ):
        metric = MetricLogger()
        task_bs, num_views, H, W, _ = imgs.shape
        pixels = rearrange(imgs, "b v h w c -> b (v h w) c")  # [task_bs, num_views * H * W, 3]
        rays_o, rays_d = get_rays_shapenet_batch(hwf, poses)  # [task_bs, num_views, H, W, 3]
        rays_o = rearrange(rays_o, "b v h w c -> b (v h w) c")  # [task_bs, num_views * H * W, 3]
        rays_d = rearrange(rays_d, "b v h w c -> b (v h w) c")  # [task_bs, num_views * H * W, 3]
        num_total_rays = rays_d.shape[1]

        indices = torch.randint(num_total_rays, size=[task_bs, ray_bs], device=rays_o.device)
        indices = repeat(indices, "... -> ... c", c=3)
        raybatch_o = torch.gather(rays_o, 1, indices)  # [task_bs, ray_bs, 3]
        raybatch_d = torch.gather(rays_d, 1, indices)  # [task_bs, ray_bs, 3]
        pixelbatch = torch.gather(pixels, 1, indices)  # [task_bs, ray_bs, 3]
        t_vals, xyz = sample_points_batch(
            raybatch_o, raybatch_d, bound[:, 0], bound[:, 1], num_points_per_ray, perturb=True
        )  # [task_bs, ray_bs, num_points_per_ray], [task_bs, ray_bs, num_points_per_ray, 3]

        rgbs, sigmas = self.functional_batch(client_params_batch, xyz)
        colors = volume_render_batch(rgbs, sigmas, t_vals, white_bkgd=self.args.white_bkgd)  # [task_bs, ray_bs, 3]

        # * positive loss
        loss_pos = torch.nn.functional.mse_loss(colors, pixelbatch)

        # * negative loss ! (global <-> gt)
        global_params_batch = OrderedDict({n: repeat(p, "... -> b ...", b=task_bs) for n, p in client_params.items()})
        g_rgbs, g_sigmas = self.functional_batch(global_params_batch, xyz)
        g_colors = volume_render_batch(
            g_rgbs, g_sigmas, t_vals, white_bkgd=self.args.white_bkgd
        )  # [task_bs, ray_bs, 3]

        loss_neg = torch.nn.functional.mse_loss(g_colors, pixelbatch)

        # * menf loss
        loss = loss_pos - self.triplet_gamma * loss_neg

        grad = torch.autograd.grad(loss, client_params.values(), create_graph=False)

        clip_norm_(grad, gradclip)
        for (n, p), g in zip(client_params.items(), grad):
            client_params[n] = p - lr * g

        metric.update(loss=loss.item(), loss_pos=loss_pos.item(), loss_neg=loss_neg.item())

        return client_params, metric
