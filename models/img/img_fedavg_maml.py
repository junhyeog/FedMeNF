from argparse import Namespace
from collections import OrderedDict
from typing import List

import torch
from einops import rearrange, repeat

from hypernets import IdentityNet
from models.img.img_basemodel import Img_BaseModel
from networks import SirenNet
from utils.clipping import clip_norm_
from utils.logger import MetricLogger
from utils.misc import copy_model_params
from utils.rendering import (get_rays_shapenet_batch, sample_points_batch,
                             volume_render_batch)


class Img_FedAvg_MAML(Img_BaseModel):
    def __init__(self, args: Namespace, log_dict: dict = {}):
        super().__init__(args, log_dict)

        self.model = SirenNet(3, args.hidden_dim, 3, args.num_layers, w0=30.0).to(args.device)
        self.hpnet = IdentityNet(self.model).to(args.device)
        self.hpnet_optimizer = torch.optim.SGD(self.hpnet.parameters(), lr=1.0, momentum=0.0, weight_decay=0.0)

        # set client model to the global model
        for client in self.clients:
            client.model_params = copy_model_params(self.hpnet())

        self.inner_loop_create_graph = True

    def inner_loop_batch(
        self,
        client_params_batch,
        steps,
        ray_bs,
        gradclip,
        lr,
        task_data,
        create_graph,
    ):
        metric = MetricLogger()

        task_id, s_set, q_set = task_data
        task_bs, v_s, h_s, w_s, in_size_s = s_set.shape

        for step in range(steps):

            coords, pixels = self.sample_coords_batch(s_set, ray_bs)

            rgbs = self.functional_batch(client_params_batch, coords)
            loss = torch.nn.functional.mse_loss(rgbs, pixels)

            grad = torch.autograd.grad(loss, client_params_batch.values(), create_graph=create_graph)

            clip_norm_(grad, gradclip)
            for (n, p), g in zip(client_params_batch.items(), grad):
                client_params_batch[n] = p - lr * g

            metric.update(loss=loss.item())

        if "loss" not in metric:
            metric.update(loss=0.0)
        return client_params_batch, metric

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

        grad = torch.autograd.grad(loss, client_params.values(), create_graph=False)

        clip_norm_(grad, gradclip)
        for (n, p), g in zip(client_params.items(), grad):
            client_params[n] = p - lr * g

        metric.update(loss=loss.item())

        return client_params, metric

    def outer_loop_scaffold_batch(
        self,
        client_params,
        client_params_batch,
        ray_bs,
        gradclip,
        lr,
        task_data,
        client_id,
    ):
        assert hasattr(self, "c_global")
        assert hasattr(self, "c_locals")

        metric = MetricLogger()

        task_id, s_set, q_set = task_data
        task_bs, v_q, h_q, w_q, in_size_q = q_set.shape

        coords, pixels = self.sample_coords_batch(s_set, ray_bs)

        rgbs = self.functional_batch(client_params_batch, coords)
        loss = torch.nn.functional.mse_loss(rgbs, pixels)

        grad = torch.autograd.grad(loss, client_params.values(), create_graph=False)

        clip_norm_(grad, gradclip)
        for (n, p), g in zip(client_params.items(), grad):
            # client_params[n] = p - lr * g

            # wt = [w - self.args.inner_lr * g for w, g in zip(wt, grad)]
            # for i in range(len(wt)):
            #     wt[i] = wt[i] - self.args.inner_lr * (grad[i] - self.c_locals[client_id][i] + self.c_global[i])

            client_params[n] = p - lr * (g - self.c_locals[client_id][n] + self.c_global[n])

        metric.update(loss=loss.item())

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
                create_graph=self.inner_loop_create_graph,
            )

            # 3. meta-train test using query set (outer loop)
            client_params, metric_outer = self.outer_loop_batch(
                client_params,
                client_params_batch,
                self.args.client_ray_bs,
                gradclip=self.args.client_gradclip,
                lr=self.args.client_lr,
                task_data=task_data,
                client_id=client_id,
            )

            metric.update(weight=task_bs, prefix_inner=metric_inner, metric_outer=metric_outer)

        if update_client_params:
            client.model_params = copy_model_params(client_params)

        # torch.cuda.empty_cache()
        return client_params, metric

    def server_aggregation(self, client_ids: List[int]):
        self.server_aggregation_fedavg(client_ids)
