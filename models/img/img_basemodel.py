import argparse
import os
import uuid
from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import List

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from torch.func import functional_call, vmap

# from tensor import SummaryWriter
from models.basemodel import BaseModel
from utils.logger import Logger, MetricLogger
from utils.misc import copy_model_params, find_folder, make_grid, set_seed
from utils.rendering import img_test_model


class Img_BaseModel(BaseModel):
    def __init__(self, args: argparse.Namespace, log_dict: dict = {}):
        _log_dict = {
            "num_layers": "nl",
            "hidden_dim": "hd",
            "num_rounds": "r",
            "epoch_to_iter": "eti",
            "client_epochs": "ep",
            "client_lr": "lr",
            "client_inner_lr": "ilr",
            "tto_lr": "tlr",
            "task_bs": "tbs",
            "client_ray_bs": "rbs",
            "inner_loop_steps": "is",
            # "num_support_views": "nsv",
            # "num_query_views": "nqv",
            # "num_test_support_views": "tsv",
            "adapt_optim": "opt",
        }
        log_dict.update(_log_dict)
        super().__init__(args, log_dict=log_dict)

    def shape_to_coords(self, shape, device):
        coords = []
        for i in range(len(shape)):
            coords.append(torch.linspace(-1.0, 1.0, shape[i], device=device))
        return torch.stack(torch.meshgrid(*coords), dim=-1)

    def sample_coords(self, data, ray_bs=-1):
        coords = self.shape_to_coords(data.shape[:-1], device=data.device)

        if ray_bs == -1:  # use all coords and same shape with data [v h w in]
            return coords

        # sample coords. use torch.gather
        v, h, w, out_size = data.shape

        data_flat = rearrange(data, "v h w c -> (v h w) c")
        coords_flat = rearrange(coords, "v h w c -> (v h w) c")

        num_total_rays = len(coords_flat)
        num_sampled_rays = min(ray_bs, num_total_rays)

        indices = torch.randint(num_total_rays, size=[num_sampled_rays], device=data.device)
        indices = repeat(indices, "... -> ... c", c=out_size)
        values_flat = torch.gather(data_flat, 0, indices)
        coords_flat = torch.gather(coords_flat, 0, indices)

        return coords_flat, values_flat

    def sample_coords_batch(self, data, ray_bs=-1):
        results = [self.sample_coords(d, ray_bs) for d in data]
        return torch.stack([i[0] for i in results], dim=0), torch.stack([i[1] for i in results], dim=0)

    def adapt_one_client_one_task(
        self,
        model_params: str | OrderedDict,
        task_data,
        accumulated_steps,
        verbose: bool = False,
    ):
        metric_list = []
        synth_list = []
        gt_list = []

        # arguments
        test_ray_bs = self.args.test_ray_bs
        ray_bs = self.args.client_ray_bs
        gradclip = self.args.client_gradclip
        lr = self.args.tto_lr
        steps = accumulated_steps[-1]
        adapt_optim = getattr(self.args, "adapt_optim", "sgd")

        # set model and optimizer
        client_params = copy_model_params(model_params)
        self.model.load_state_dict(client_params, strict=True)
        self.model.train()
        optimizer = self.get_optimizer(adapt_optim, list(self.model.parameters()), lr=lr)

        # data
        task_id, s_set, q_set = task_data

        if 0 in accumulated_steps:
            metric, synths = img_test_model(self, s_set, q_set, test_ray_bs)
            metric.update(loss=0)
            metric_list.append(metric)
            sampled_view = torch.randint(0, len(synths), (1,))
            synth_list.append(synths[sampled_view])
            gt_list.append(q_set[sampled_view])

        set_seed(self.args.seed)
        for step in range(1, steps + 1):
            optimizer.zero_grad()

            coords, values = self.sample_coords(s_set, ray_bs=ray_bs)  # [bs in], [bs out]

            rgbs = self.model(coords)
            loss = torch.nn.functional.mse_loss(rgbs, values)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), gradclip)
            optimizer.step()

            if step in accumulated_steps:
                metric, synths = img_test_model(self, s_set, q_set, test_ray_bs)
                metric.update(loss=loss.item())
                metric_list.append(metric)
                # sample one view
                sampled_view = torch.randint(0, len(synths), (1,))
                synth_list.append(synths[sampled_view])
                gt_list.append(s_set[sampled_view])

                if verbose:
                    self.log(f"[+] One client one task tto done. tto steps:{step}, psnr: {metric.psnr.avg:.4f}")

        model_params = copy_model_params(self.model.state_dict())
        extra = {
            "gt_list": gt_list,
            "synth_list": synth_list,
        }
        return model_params, metric_list, extra

    def adapt_multi_client_all_tasks(
        self,
        client_ids: List[int],
        model_params: str | OrderedDict,
        accumulated_steps,
        type: str = "test",  # client dataloader type
        tag: str = None,  # if None, do not write to tensorboard
        postfix: str = "",
        num_test_support_views: int = None,
        tto0: bool = False,
        round: int = None,
    ):
        assert num_test_support_views is None  # for img
        if tto0:
            assert accumulated_steps == [0]

        params_per_client_list = []
        metric_per_client_list = []
        synth_per_client_list = []
        gt_per_client_list = []
        for client_id in client_ids:
            client = self.clients[client_id]
            dl = client.get_dl(type)

            if model_params == "local":  # use the client model
                client_params = client.model_params
            elif model_params == "global":  # use the global model
                client_params = self.hpnet()
            elif isinstance(model_params, OrderedDict):  # use the given model_params
                client_params = model_params
            client_params_init = copy_model_params(client_params)

            params_per_task_list = []
            metric_per_task_list = []
            synth_per_task_list = []
            gt_per_task_list = []
            weight_per_task_list = []

            for batch_idx, task_data in enumerate(dl):
                task_id, s_set, q_set = task_data
                task_bs = s_set.shape[0]

                for task_idx in range(task_bs):
                    task_id_i = task_id[task_idx]
                    s_set_i = s_set[task_idx]
                    q_set_i = q_set[task_idx]

                    one_task_data = (task_id_i, s_set_i, q_set_i)
                    model_params_i = copy_model_params(client_params_init)
                    model_params_i, metric_list, extra = self.adapt_one_client_one_task(
                        model_params_i,
                        one_task_data,
                        accumulated_steps,
                        verbose=client_id == client_ids[0] and batch_idx == 0 and task_idx == 0,
                    )

                    params_per_task_list.append(model_params_i)
                    metric_per_task_list.append(metric_list)
                    synth_per_task_list.append(extra["synth_list"])  # [H, W, 3]
                    gt_per_task_list.append(extra["gt_list"])
                    weight_per_task_list.append(1)  # do not have num_support_views => 1

            params_per_client_list.append(params_per_task_list)

            client_synth_list = synth_per_task_list[0]  # only the first task
            client_gt_list = gt_per_task_list[0]  # only the first task

            # merge metric per client
            client_metric_list = []
            for a_s_idx, a_s in enumerate(accumulated_steps):
                metric = MetricLogger()
                for metric_list, weight in zip(metric_per_task_list, weight_per_task_list):
                    metric.update(**metric_list[a_s_idx], weight=weight)  # do not have num_support_views => 1
                client_metric_list.append(metric)

            metric_per_client_list.append(client_metric_list)
            synth_per_client_list.append(client_synth_list)
            gt_per_client_list.append(client_gt_list)

        # end for all clients

        # merge metric per a_s
        metric = MetricLogger()
        for a_s_idx, a_s in enumerate(accumulated_steps):
            metric_a_s = MetricLogger()
            for client_id, client_metric in zip(client_ids, metric_per_client_list):
                metric_a_s.update(**client_metric[a_s_idx], weight=sum(client_metric[a_s_idx].psnr.weights))

            metric.update(**{f"prefix_tto_{a_s:06}{f'_{postfix}' if postfix else ''}": metric_a_s})

            self.log(
                f"Steps:{a_s}, ({"tto0" if tto0 else "step"}) loss: {metric_a_s.loss.avg:.4f}, psnr: {metric_a_s.psnr.avg:.4f}"
            )

            if tag is not None:  # write to tensorboard
                self.logger.write_scalars(
                    f"{tag}{f'_{postfix}' if postfix else ''}",
                    step=round if tto0 else a_s,
                    **metric_a_s,
                )

                num_imgs = min(4, len(gt_per_client_list))
                # sample client index
                # indices = torch.randperm(len(gt_per_client_list))[:num_imgs]
                indices = range(num_imgs)
                gts = torch.stack([gt_per_client_list[i][a_s_idx] for i in indices], dim=0)
                synths = torch.stack([synth_per_client_list[i][a_s_idx] for i in indices], dim=0)
                grid = make_grid([gts, synths], num_imgs=num_imgs)

                self.writer.add_image(
                    f"{tag}{f'_{postfix}' if postfix else ''}/synths",
                    grid,
                    round if tto0 else a_s,
                )

        return metric, params_per_client_list
