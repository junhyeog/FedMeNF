from argparse import Namespace
from collections import OrderedDict
from typing import List

import torch
from einops import rearrange, repeat

from hypernets import IdentityNet
from models.img.img_fedavg_maml import Img_FedAvg_MAML
from utils.clipping import clip_norm_
from utils.logger import MetricLogger
from utils.misc import copy_model_params


class Img_FedAvg_Reptile(Img_FedAvg_MAML):
    def __init__(self, args: Namespace, log_dict: dict = {}):
        super().__init__(args, log_dict)

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
        num_iters = epochs if epoch_to_iter else epochs * len(dl_train)
        for _ in range(num_iters):
            task_data = next(dl_train.__iter__())
            task_bs = task_data[1].shape[0]

            client_params_batch = OrderedDict(
                {n: repeat(p, "... -> b ...", b=task_bs) for n, p in client_params.items()}
            )

            # 2. meta-train train using support set (inner loop)
            client_params_batch, metric_inner = self.inner_loop_batch(
                client_params_batch,
                self.args.inner_loop_steps,
                self.args.client_ray_bs,
                gradclip=self.args.client_gradclip,
                lr=self.args.client_inner_lr,
                task_data=task_data,
                create_graph=False,
            )

            # 3. meta-train test using query set (outer loop)
            client_params, metric_outer = self.outer_loop_batch(
                client_params,
                client_params_batch,
                gradclip=self.args.client_gradclip,
                lr=self.args.client_lr,
            )

            metric.update(weight=task_bs, prefix_inner=metric_inner, metric_outer=metric_outer)

        if update_client_params:
            client.model_params = copy_model_params(client_params)

        # torch.cuda.empty_cache()
        return client_params, metric
