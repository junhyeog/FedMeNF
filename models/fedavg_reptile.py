from argparse import Namespace
from collections import OrderedDict
from typing import List

import numpy as np
import torch
from einops import rearrange, repeat
from torch.func import functional_call, vmap

from hypernets import IdentityNet
# from models import FedAvg_MAML
from models.fedavg_maml import FedAvg_MAML
from networks import SimpleNeRF
from utils.clipping import clip_norm_
from utils.logger import MetricLogger
from utils.misc import copy_model_params, make_grid, rotate_poses


class FedAvg_Reptile(FedAvg_MAML):
    def __init__(self, args: Namespace, log_dict: dict = {}):
        _log_dict = {
            "use_q": "uq",
        }
        log_dict.update(_log_dict)
        super().__init__(args, log_dict=log_dict)  # should be first

        self.use_q = getattr(args, "use_q", 0)

    def outer_loop_batch(
        self,
        client_params,
        client_params_batch,
        gradclip,
        lr,
    ):
        metric = MetricLogger()

        grad = []
        for (n, p), (n_batch, p_batch) in zip(client_params.items(), client_params_batch.items()):
            assert n == n_batch
            assert p.shape == p_batch.mean(0).shape
            grad.append(p - p_batch.mean(0))

        clip_norm_(grad, gradclip)
        for (n, p), g in zip(client_params.items(), grad):
            client_params[n] = p - lr * g

        metric.update(loss=0)

        return client_params, metric

    def client_update(
        self,
        client_id: int,
        epochs: int,
        load_global_model: bool,
        update_client_params: bool,
        epoch_to_iter: bool = False,
    ):
        client = self.clients[client_id]
        metric = MetricLogger()
        dl_train = client.dl_train

        if load_global_model:
            client.model_params = self.hpnet()
        client_params = copy_model_params(client.model_params)

        # meta-train
        # for _ in range(epochs):
        #     for task_data in dl_train:
        #         imgs_s, poses_s, imgs_q, poses_q, hwf, bound = task_data
        num_iters = epochs if epoch_to_iter else epochs * len(dl_train) * client.train_num_views["total"]["query"]
        for _ in range(num_iters):
            task_id, imgs_s, poses_s, imgs_q, poses_q, hwf, bound = next(dl_train.__iter__())
            # [task_bs, num_support_views, H, W, 3], [task_bs, num_support_views, 4, 4],
            # [task_bs, num_query_views, H, W, 3], [task_bs, num_query_views, 4, 4],
            # [task_bs, 4], [task_bs, 2]
            task_bs = imgs_s.shape[0]

            # support set
            assert task_bs == 1
            num_support_views = client.num_views[task_id[0].item()]["support"]
            imgs_s = imgs_s[:, :num_support_views]
            poses_s = poses_s[:, :num_support_views]

            # query set
            num_query_views = client.num_views[task_id[0].item()]["query"]
            imgs_q = imgs_q[:, :num_query_views]
            poses_q = poses_q[:, :num_query_views]

            imgs = imgs_s
            poses = poses_s

            client_params_batch = OrderedDict(
                {n: repeat(p, "... -> b ...", b=task_bs) for n, p in client_params.items()}
            )

            # 1. meta-train train using all data (support set + query set) (inner loop)
            client_params_batch, metric_inner = self.inner_loop_batch(
                client_params_batch,
                self.args.inner_loop_steps,
                self.args.client_ray_bs,
                self.args.num_points_per_ray,
                gradclip=self.args.client_gradclip,
                lr=self.args.client_inner_lr,
                imgs=imgs,
                poses=poses,
                hwf=hwf,
                bound=bound,
                create_graph=False,
            )

            # 2. meta-train test (outer loop)
            client_params, metric_outer = self.outer_loop_batch(
                client_params,
                client_params_batch,
                gradclip=self.args.client_gradclip,
                lr=self.args.client_lr,
            )

            metric.update(weight=task_bs, prefix_inner=metric_inner, metric_outer=metric_outer)

        if update_client_params:
            client.model_params = client_params

        # torch.cuda.empty_cache()
        return client_params, metric
