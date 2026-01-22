from argparse import Namespace
from collections import OrderedDict
from typing import List

import torch
from einops import rearrange, repeat

from hypernets import IdentityNet
from models.basemodel import BaseModel
from networks import SimpleNeRF
from utils.clipping import clip_norm_
from utils.logger import MetricLogger
from utils.misc import copy_model_params
from utils.rendering import get_rays_shapenet_batch, sample_points_batch, volume_render_batch


class FedAvg_MAML(BaseModel):
    def __init__(self, args: Namespace, log_dict: dict = {}):
        super().__init__(args, log_dict)

        self.model = SimpleNeRF(3, args.max_freq, args.num_freqs, args.hidden_dim, args.num_layers, 4).to(args.device)
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
        num_points_per_ray,
        gradclip,
        lr,
        imgs,
        poses,
        hwf,
        bound,
        create_graph,
    ):
        metric = MetricLogger()
        task_bs, num_views, H, W, _ = imgs.shape
        rays_o, rays_d = get_rays_shapenet_batch(hwf, poses)  # [task_bs, num_views, H, W, 3]
        pixels = rearrange(imgs, "b v h w c -> b (v h w) c")  # [task_bs, num_views * H * W, 3]
        rays_o = rearrange(rays_o, "b v h w c -> b (v h w) c")  # [task_bs, num_views * H * W, 3]
        rays_d = rearrange(rays_d, "b v h w c -> b (v h w) c")  # [task_bs, num_views * H * W, 3]
        num_total_rays = rays_d.shape[1]

        for step in range(steps):
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
            loss = torch.nn.functional.mse_loss(colors, pixelbatch)

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
        loss = torch.nn.functional.mse_loss(colors, pixelbatch)

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
        num_points_per_ray,
        gradclip,
        lr,
        imgs,
        poses,
        hwf,
        bound,
        client_id,
    ):
        assert hasattr(self, "c_global")
        assert hasattr(self, "c_locals")

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
        loss = torch.nn.functional.mse_loss(colors, pixelbatch)

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

            client_params_batch = OrderedDict(
                {n: repeat(p, "... -> b ...", b=task_bs) for n, p in client_params.items()}
            )

            # 2. meta-train train using support set (inner loop)
            client_params_batch, metric_inner = self.inner_loop_batch(
                client_params_batch,
                self.args.inner_loop_steps,
                self.args.client_ray_bs,
                self.args.num_points_per_ray,
                gradclip=self.args.client_gradclip,
                lr=self.args.client_inner_lr,
                imgs=imgs_s,
                poses=poses_s,
                hwf=hwf,
                bound=bound,
                create_graph=self.inner_loop_create_graph,
            )

            # 3. meta-train test using query set (outer loop)
            client_params, metric_outer = self.outer_loop_batch(
                client_params,
                client_params_batch,
                self.args.client_ray_bs,
                self.args.num_points_per_ray,
                gradclip=self.args.client_gradclip,
                lr=self.args.client_lr,
                imgs=imgs_q,
                poses=poses_q,
                hwf=hwf,
                bound=bound,
                client_id=client_id,
            )

            metric.update(weight=task_bs, prefix_inner=metric_inner, metric_outer=metric_outer)

        if update_client_params:
            client.model_params = copy_model_params(client_params)

        # torch.cuda.empty_cache()
        return client_params, metric

    def server_aggregation(self, client_ids: List[int]):
        self.server_aggregation_fedavg(client_ids)
