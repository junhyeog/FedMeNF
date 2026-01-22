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
from clients.client import Client
from clients.clients import Clients
from utils.logger import Logger, MetricLogger
from utils.misc import copy_model_params, find_folder, make_grid, set_seed
from utils.rendering import get_rays_shapenet, sample_points, test_model, test_model_batch, volume_render


class BaseModel(ABC):
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
            # "client_ray_bs": "rbs",
            # "num_points_per_ray": "np",
            "inner_loop_steps": "is",
            # "num_support_views": "nsv",
            # "num_query_views": "nqv",
            "num_test_support_views": "tsv",
            "adapt_optim": "opt",
        }
        log_dict.update(_log_dict)

        # set exp
        exp_wo_uuid = f"{args.algo}"
        for k, v in log_dict.items():
            if getattr(args, k, None) is not None:
                exp_wo_uuid += f"_{v}:{eval(f'args.{k}')}"
        log_dir_wo_exp = f"./runs/{args.tag}/u:{args.num_clients},{args.num_ood_clients},{args.num_participants}/{args.dataset}/d:{int(args.iid)},{args.alpha}_no:{args.num_objects_per_client},{args.num_test_objects_per_client}"
        if args.load_ckpt:
            try:
                if getattr(args, "log_dir", None) is None:
                    folder = find_folder(log_dir_wo_exp, startswith=exp_wo_uuid)
                else:
                    folder = args.log_dir
                # get the last checkpoint
                files = sorted(os.listdir(folder))
                ckpt_files = [f for f in files if f.startswith("ckpt")]
                if len(ckpt_files) == 0:
                    raise ValueError(f"[-] load_ckpt: no ckpt files in {folder}")

                max_ckpt = max([int(f.split("_")[1].split(".")[0]) for f in ckpt_files])
                ckpt_file = f"{folder}/ckpt_{max_ckpt:06}.pt"
                ckpt = torch.load(ckpt_file)
                print(f"[+] load_ckpt: {ckpt_file}")
                if getattr(args, "log_dir", None) is None:
                    # check args if log_dir is not given
                    whitelist = ["trial_number", "uuid", "load_ckpt"]
                    flag = True
                    for k, v in vars(args).items():
                        if k in whitelist:
                            continue
                        if k not in ckpt["args"]:
                            print(f"(now) {k} not in (loaded) ckpt['args']")
                            flag = False
                            continue
                        if v != getattr(ckpt["args"], k):
                            print(f"(now) {k}: {v} != (loaded) {getattr(ckpt["args"], k)}")
                            flag = False
                    if not flag:
                        raise ValueError(f"[-] load_ckpt: args mismatch")

                    print(f"[+] load_ckpt: args match")

                # load models
                # self.model.load_state_dict(ckpt["model"])
                self.hpnet.load_state_dict(ckpt["hpnet"])
                if "hpnet_optimizer" in ckpt:
                    self.hpnet_optimizer.load_state_dict(ckpt["hpnet_optimizer"])
                # update args (namespace)
                for k, v in vars(ckpt["args"]).items():
                    if k in ["load_ckpt"]:
                        continue
                    setattr(args, k, v)
                args.max_ckpt = max_ckpt

            except Exception as e:
                print(e)
                print(f"[-] load_ckpt: failed")

        if getattr(args, "log_dir", None) is None:
            # set log_dir
            self.uuid = str(uuid.uuid1())[:8]
            args.uuid = self.uuid
            exp = exp_wo_uuid + f"_{getattr(args, 'trial_number', None)}_{args.uuid}"
            args.exp = exp
            args.log_dir = f"{log_dir_wo_exp}/{args.exp}"
            if not os.path.exists(args.log_dir):
                os.makedirs(args.log_dir)

        self.logger = Logger(args.log_dir)
        self.log = self.logger.log
        self.log(f"[+] exp: {args.exp}")
        self.writer = self.logger.writer
        self.criteria = torch.nn.CrossEntropyLoss()
        self.args = args
        self.log(f"[+] preparing clients...")
        self.clients = Clients(self)
        self.log(f"[+] clients prepared")

        self.round = None  # global communication round

        if getattr(args, "tto_step_period", -1) == -1:
            if args.tto_steps == 0:
                args.steps_per_round = [0]
            else:
                args.steps_per_round = [0, 1] + [pow(2, i) for i in range(np.log2(args.tto_steps).astype(int))]
        else:
            r = args.tto_steps % args.tto_step_period
            q = args.tto_steps // args.tto_step_period
            args.steps_per_round = [args.tto_step_period * i for i in range(q + 1)]
            if r:
                args.steps_per_round.append(r)

        args.accumulated_steps = np.cumsum(args.steps_per_round)

        self.set_custom_scalars()

    def set_custom_scalars(self):
        accumulated_steps = self.args.accumulated_steps
        tto_rounds = [
            round
            for round in range(self.args.num_rounds)
            if (round + 1) % (self.args.print_test) == 0 or (round + 1) == self.args.num_rounds
        ]
        layout = {
            # draw tto plots per step at once
            "tto_per_step": {
                "psnr": ["Multiline", [f"test/tto_{step:06}_psnr" for step in accumulated_steps]],
                "lpips": ["Multiline", [f"test/tto_{step:06}_lpips" for step in accumulated_steps]],
                "ssim": ["Multiline", [f"test/tto_{step:06}_ssim" for step in accumulated_steps]],
                "inner_loss": [
                    "Multiline",
                    [f"test/tto_{step:06}_inner_loss" for step in accumulated_steps],
                ],
                "loss": ["Multiline", [f"test/tto_{step:06}_loss" for step in accumulated_steps]],
            },
            # draw tto plots per round at once
            "tto_per_round": {
                "psnr": ["Multiline", [f"test_round_{round:06}/psnr" for round in tto_rounds]],
                "lpips": ["Multiline", [f"test_round_{round:06}/lpips" for round in tto_rounds]],
                "ssim": ["Multiline", [f"test_round_{round:06}/ssim" for round in tto_rounds]],
                "inner_loss": [
                    "Multiline",
                    [f"test_round_{round:06}/inner_loss" for round in tto_rounds],
                ],
                "loss": ["Multiline", [f"test_round_{round:06}/loss" for round in tto_rounds]],
            },
        }
        self.writer.add_custom_scalars(layout)

    def get_optimizer(self, optim, params, lr, weight_decay=0.01, differentiable=False):
        set_seed(self.args.seed)
        optimizer = {
            "sgd": torch.optim.SGD(params, lr=lr, momentum=0.0, weight_decay=0.0, differentiable=differentiable),
            "sgdm": torch.optim.SGD(
                params, lr=lr, momentum=0.9, weight_decay=weight_decay, differentiable=differentiable
            ),
            "adam": torch.optim.Adam(params, lr=lr, weight_decay=weight_decay, differentiable=differentiable),
            "adamw": torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay, differentiable=differentiable),
            "nadamw": torch.optim.NAdam(
                params, lr=lr, decoupled_weight_decay=True, weight_decay=weight_decay, differentiable=differentiable
            ),
        }[optim]
        return optimizer

    def functional(self, params, x):
        return functional_call(self.model, params, x, strict=True)

    def functional_batch(self, params_batch, x_batch):  # using vmap
        return vmap(lambda params, x: functional_call(self.model, params, x, strict=True))(params_batch, x_batch)

    @abstractmethod
    def inner_loop_batch(self):
        pass

    @abstractmethod
    def outer_loop_batch(self):
        pass

    @abstractmethod
    def client_update(self):
        pass

    # ! Scaffold
    def client_update_c(self, client_id: int, client_params: OrderedDict):
        assert hasattr(self, "c_global")
        assert hasattr(self, "c_locals")

        update_cnt = (
            self.args.client_epochs
            if self.args.epoch_to_iter
            else self.args.client_epochs
            * len(self.clients[client_id].dl_train)
            * self.clients[client_id].train_num_views["total"]["query"]
        )

        global_params = copy_model_params(self.hpnet())

        c_plus = OrderedDict()
        for k_g, v_g in self.c_global.items():
            _c_plus = (
                self.c_locals[client_id][k_g]
                - v_g
                + (global_params[k_g] - client_params[k_g]) / (update_cnt * self.args.client_lr)
            )
            c_plus[k_g] = _c_plus.detach().clone()
        c_delta = OrderedDict({k: v - self.c_locals[client_id][k] for k, v in c_plus.items()})
        self.c_locals[client_id] = c_delta
        return c_delta

    def c_aggregation(self, client_ids: List[int]):
        for i, idx in enumerate(client_ids):
            self.c_global = OrderedDict(
                {k: v + self.c_deltas[idx][k] / len(self.clients) for k, v in self.c_global.items()}
            )
        return

    def get_client_update_scaffold_fn(self, client_update_fn):
        def client_update_scaffold(
            client_id: int,
            epochs: int,
            load_global_model: bool,
            update_client_params: bool,
            epoch_to_iter: bool = False,
        ):
            client_params, metric = client_update_fn(
                client_id, epochs, load_global_model, update_client_params, epoch_to_iter
            )

            # Scaffold
            c_delta = self.client_update_c(client_id, client_params)
            self.c_deltas[client_id] = c_delta

            return client_params, metric

        return client_update_scaffold

    def server_aggregation_scaffold(self, client_ids: torch.List[int]):
        # Scaffold
        self.c_aggregation(client_ids)
        self.server_aggregation_fedavg(client_ids, weighted=False)

    # ! server_aggregation
    def server_aggregation_fedavg(self, client_ids: List[int], weighted: bool = True):
        self.hpnet_optimizer.zero_grad()

        if weighted:
            # weights = torch.Tensor([self.clients[id].train_data_size for id in client_ids]).cuda()
            weights = torch.Tensor([self.clients[id].train_num_views["total"]["support"] for id in client_ids]).cuda()
            weights = weights / weights.sum()
        else:
            weights = torch.ones(len(client_ids)).cuda() / len(client_ids)

        global_model_params = self.hpnet()
        for id, weight in zip(client_ids, weights):
            client = self.clients[id]
            # delta_params: global model - client model
            assert global_model_params.keys() == client.model_params.keys()
            delta_params = OrderedDict({n: p - client.model_params[n] for n, p in global_model_params.items()})
            grad_hnet = torch.autograd.grad(
                global_model_params.values(), self.hpnet.parameters(), delta_params.values()
            )

            for (n, p), g in zip(self.hpnet.named_parameters(), grad_hnet):
                if p.grad == None:
                    p.grad = torch.zeros_like(p)
                if g == None:
                    g = torch.zeros_like(p)
                p.grad = p.grad + g * weight

        torch.nn.utils.clip_grad_norm_(self.hpnet.parameters(), self.args.server_gradclip)
        self.hpnet_optimizer.step()

    def server_aggregation_fednova(self, client_ids: List[int]):
        self.hpnet_optimizer.zero_grad()

        # weights = torch.Tensor([self.clients[id].train_data_size for id in client_ids]).cuda()
        weights = torch.Tensor([self.clients[id].train_num_views["total"]["support"] for id in client_ids]).cuda()
        weights = weights / weights.sum()

        # FedNova
        tau_eff = 0.0
        update_cnts = []
        for i, id in enumerate(client_ids):
            # update_cnts: number of updates for each client
            update_cnt = (
                self.args.client_epochs
                if self.args.epoch_to_iter
                else self.args.client_epochs
                * len(self.clients[id].dl_train)
                * self.clients[id].train_num_views["total"]["query"]
            )
            tau_eff += weights[i] * update_cnt
            update_cnts.append(update_cnt)

        global_model_params = self.hpnet()
        for id, weight, update_cnt in zip(client_ids, weights, update_cnts):
            client = self.clients[id]
            # delta_params: global model - client model
            assert global_model_params.keys() == client.model_params.keys()
            delta_params = OrderedDict({n: p - client.model_params[n] for n, p in global_model_params.items()})
            grad_hnet = torch.autograd.grad(
                global_model_params.values(), self.hpnet.parameters(), delta_params.values()
            )

            for (n, p), g in zip(self.hpnet.named_parameters(), grad_hnet):
                if p.grad == None:
                    p.grad = torch.zeros_like(p)
                if g == None:
                    g = torch.zeros_like(p)
                # FedNova
                # p.grad = p.grad + g * weight # original
                p.grad = p.grad + g * weight / update_cnt * tau_eff

        # write eta_g to tensorboard
        self.writer.add_scalar("train/tau_eff", tau_eff.item(), self.round)

        torch.nn.utils.clip_grad_norm_(self.hpnet.parameters(), self.args.server_gradclip)
        self.hpnet_optimizer.step()

    def server_aggregation_fedexp(self, client_ids: List[int]):
        assert hasattr(self, "fedexp_epsilon")

        self.hpnet_optimizer.zero_grad()

        # weights = torch.Tensor([self.clients[id].train_data_size for id in client_ids]).cuda()
        weights = torch.Tensor([self.clients[id].train_num_views["total"]["support"] for id in client_ids]).cuda()
        weights = weights / weights.sum()

        # FedExp
        grad_norm_avg = 0.0
        grad_avg = 0.0

        global_model_params = self.hpnet()
        for id, weight in zip(client_ids, weights):
            client = self.clients[id]
            # delta_params: global model - client model
            assert global_model_params.keys() == client.model_params.keys()
            delta_params = OrderedDict({n: p - client.model_params[n] for n, p in global_model_params.items()})

            # FedExp
            grad = torch.cat([-g.flatten() for g in delta_params.values()])
            grad_norm = torch.linalg.norm(grad) ** 2
            grad_avg += grad * weight
            grad_norm_avg += grad_norm * weight

            grad_hnet = torch.autograd.grad(
                global_model_params.values(), self.hpnet.parameters(), delta_params.values()
            )

            for (n, p), g in zip(self.hpnet.named_parameters(), grad_hnet):
                if p.grad == None:
                    p.grad = torch.zeros_like(p)
                if g == None:
                    g = torch.zeros_like(p)
                p.grad = p.grad + g * weight

        torch.nn.utils.clip_grad_norm_(self.hpnet.parameters(), self.args.server_gradclip)

        # FedExp
        grad_avg_norm = torch.linalg.norm(grad_avg) ** 2

        if self.args.fedexp_type == "paper":
            eta_g = (
                0.5 * grad_norm_avg / (grad_avg_norm * (len(client_ids) + self.args.fedexp_epsilon))
            )  # paper => eta_g == 1
        elif self.args.fedexp_type == "github":
            eta_g = (
                0.5 * grad_norm_avg / (grad_avg_norm + len(client_ids) * self.fedexp_epsilon)
            )  # github => also eta_g == 1
        else:
            raise ValueError(f"Unknown fedexp_type: {self.args.fedexp_type}")

        eta_g = torch.max(eta_g, torch.ones_like(eta_g))  # set eta_g = max(eta_g, 1)

        # set server_optimizer lr to eta_g
        for param_group in self.hpnet_optimizer.param_groups:
            param_group["lr"] = eta_g.item()

        # write eta_g to tensorboard
        self.writer.add_scalar("train/eta_g", eta_g.item(), self.round)

        self.hpnet_optimizer.step()

    # ! for tto (reconstruction)
    def build_client_batched_params(self, client_id: int, model_params, type: str = "test"):
        # Define client_test_params = list of batched model_params (include all models for all tasks in dl_test)
        client = self.clients[client_id]
        dl = client.get_dl(type)

        if model_params == "local":  # use the client model
            client_params = client.model_params
        elif model_params == "global":  # use the global model
            client_params = self.hpnet()
        elif isinstance(model_params, OrderedDict):  # use the given model_params
            client_params = model_params
        client_params = copy_model_params(client_params)

        client_batched_params = []
        for batch_idx, task_data in enumerate(dl):
            task_id, imgs_s, poses_s, imgs_q, poses_q, hwf, bound = task_data
            task_bs = imgs_s.shape[0]

            client_params_batch = OrderedDict(
                {n: repeat(p, "... -> b ...", b=task_bs) for n, p in client_params.items()}
            )

            client_batched_params.append(client_params_batch)

        return client_batched_params

    def client_test_one_client(
        self,
        client_id: int,
        steps: int,
        client_test_params: List[OrderedDict],
        type: str = "test",
        recon: str = "q",
        num_test_support_views: int = None,
    ):
        client = self.clients[client_id]
        dl = client.get_dl(type)
        metric = MetricLogger()
        imgs_reon_list, synths_list = [], []

        for batch_idx, task_data in enumerate(dl):
            task_id, imgs_s, poses_s, imgs_q, poses_q, hwf, bound = task_data
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

            imgs_tto = imgs_s
            poses_tto = poses_s

            if num_test_support_views is not None:
                assert num_test_support_views <= imgs_tto.shape[1]
                imgs_tto = imgs_tto[:, :num_test_support_views]
                poses_tto = poses_tto[:, :num_test_support_views]

            # 2. meta-test train using support set (inner loop: optimize with support views)
            client_params_batch = client_test_params[batch_idx]

            set_seed(self.args.seed)
            client_params_batch, metric_inner = self.inner_loop_batch(
                client_params_batch,
                steps,
                self.args.client_ray_bs,
                self.args.num_points_per_ray,
                gradclip=self.args.client_gradclip,
                lr=self.args.tto_lr,
                imgs=imgs_tto,
                poses=poses_tto,
                hwf=hwf,
                bound=bound,
                create_graph=False,
            )

            client_test_params[batch_idx] = client_params_batch

            # 3. reconstruct the scene using the client model (meta-test test: test with novel views)
            if recon == "s":
                imgs_recon = imgs_s
                poses_recon = poses_s
            elif recon == "q":
                imgs_recon = imgs_q
                poses_recon = poses_q
            elif recon == "all":
                imgs_recon = torch.cat([imgs_s, imgs_q], dim=1)
                poses_recon = torch.cat([poses_s, poses_q], dim=1)
            else:
                raise ValueError(f"Unknown recon type: {recon}")

            metric_test, synths = test_model_batch(
                self.functional_batch,
                client_params_batch,
                imgs_recon,
                poses_recon,
                hwf,
                bound,
                self.args.num_points_per_ray,
                self.args.test_ray_bs,
                self.args.white_bkgd,
            )
            metric.update(weight=imgs_tto.shape[:2].numel(), prefix_inner=metric_inner, metric_test=metric_test)
            imgs_reon_list.append(imgs_recon[:, 0])
            synths_list.append(synths[:, 0])

        # metric.update(psnr_max=metric.psnr.max, ssim_max=metric.ssim.max, lpips_min=metric.lpips.min)

        return (
            client_test_params,
            metric,
            torch.concat(imgs_reon_list, dim=0),
            torch.concat(synths_list, dim=0),
        )

    def client_test_multiple_client(
        self,
        client_ids: List[int],
        steps: int,
        client_test_params_list: List[List[OrderedDict]],  # client_test_params for each client
        type: str = "test",  # client dataloader type
        recon: str = "q",
        num_test_support_views: int = None,
    ):
        metric = MetricLogger()
        imgs_q_list = []
        synths_list = []
        for client_idx, client_id in enumerate(client_ids):
            client_test_params, metric_client, imgs_q_client, synths_client = self.client_test_one_client(
                client_id,
                steps,
                client_test_params_list[client_idx],
                type=type,
                recon=recon,
                num_test_support_views=num_test_support_views,
            )
            client_test_params_list[client_idx] = client_test_params
            metric.update(**metric_client, weight=sum(metric_client.psnr.weights))
            # imgs_q_client, synths_client: [B, H, W, 3]
            imgs_q_list.append(imgs_q_client)
            synths_list.append(synths_client)
        # psnr_max = metric.psnr_max.max
        # ssim_max = metric.ssim_max.max
        # lpips_min = metric.lpips_min.min
        # metric.psnr_max.reset()
        # metric.ssim_max.reset()
        # metric.lpips_min.reset()
        # metric.update(psnr_max=psnr_max, lpips_min=lpips_min, ssim_max=ssim_max)

        # imgs_q = torch.stack(imgs_q_list, dim=0)
        # synths = torch.stack(synths_list, dim=0)

        # return client_test_params_list, metric, imgs_q, synths
        return client_test_params_list, metric, imgs_q_list, synths_list

    def client_test_multiple_client_multiple_rounds(
        self,
        client_ids: List[int],
        steps_per_round: List[int],
        client_test_params_list: List[List[OrderedDict]] = None,  # client_test_params for each client
        model_params: str | OrderedDict = "global",
        type: str = "test",  # client dataloader type
        recon: str = "q",
        tag: str = None,  # if None, do not write to tensorboard
        postfix: str = None,
        num_test_support_views: int = None,
        save_ckpt: bool = False,
    ):
        # ! round = tto round
        metric = MetricLogger()
        if client_test_params_list is None:
            client_test_params_list = []
            for client_id in client_ids:
                client_test_params_list.append(
                    self.build_client_batched_params(client_id, model_params=model_params, type=type)
                )

        indices = None
        for round_idx, steps in enumerate(steps_per_round):
            client_test_params_list, metric_round, imgs_q_list, synths_list = self.client_test_multiple_client(
                client_ids,
                steps,
                client_test_params_list,
                type=type,
                recon=recon,
                num_test_support_views=num_test_support_views,
            )
            metric.update(
                **{
                    f"prefix_tto_{self.args.accumulated_steps[round_idx]:06}{f'_{postfix}' if postfix else ''}": metric_round
                }
            )
            # imgs_q_list = list of [B, H, W, 3]

            if tag is not None:  # write to tensorboard
                num_imgs = 4
                row_lists = [imgs_q_list, synths_list]
                row_lists = [torch.concat(row_list, dim=0) for row_list in row_lists]
                if indices is None:
                    # indices = torch.randperm(len(row_lists[0]))[:num_imgs]
                    indices = range(num_imgs)
                grid = make_grid([row_list[indices] for row_list in row_lists], num_imgs=num_imgs)

                # num_imgs = 4
                # imgs = torch.concat(imgs_q_list, dim=0)
                # synths = torch.concat(synths_list, dim=0)
                # if indices is None:
                #     indices = torch.randperm(len(imgs))[:num_imgs]
                # grid = make_grid([imgs[indices], synths[indices]], num_imgs=num_imgs)

                self.writer.add_image(
                    f"{tag}{f'_{postfix}' if postfix else ''}/synths",
                    grid,
                    self.args.accumulated_steps[round_idx],
                )
                self.logger.write_scalars(
                    f"{tag}{f'_{postfix}' if postfix else ''}",
                    step=self.args.accumulated_steps[round_idx],
                    **metric_round,
                )
                self.log(
                    f"Steps:{self.args.accumulated_steps[round_idx]}, (step) loss: {metric_round.loss.avg:.4f}, inner_loss: {metric_round.inner_loss.avg:.4f}, psnr: {metric_round.psnr.avg:.4f}"
                )
            if save_ckpt:
                # ! save
                torch.save(
                    {
                        "args": self.args,
                        # "model": ml.model.state_dict(),
                        "hpnet": self.hpnet.state_dict(),
                        "hpnet_optimizer": self.hpnet_optimizer.state_dict(),
                        "client_model_params_list": [client.model_params for client in self.clients],
                        "client_test_params_list": client_test_params_list,
                    },
                    f"{self.args.log_dir}/ckpt_{0:06}_tto_{self.args.accumulated_steps[round_idx]:06}.pt",
                )
        return metric, client_test_params_list

    def client_test_multiple_client_tto0(
        self,
        client_ids: List[int],
        client_test_params_list: List[List[OrderedDict]] = None,  # client_test_params for each client
        model_params: str | OrderedDict = "local",
        type: str = "train",  # client dataloader type
        recon: str = "q",
        tag: str = None,  # if None, do not write to tensorboard
        round: int = 0,
    ):
        if client_test_params_list is None:
            client_test_params_list = []
            for client_id in client_ids:
                client_test_params_list.append(
                    self.build_client_batched_params(client_id, model_params=model_params, type=type)
                )

        indices = None

        client_test_params_list, metric_round, imgs_q_list, synths_list = self.client_test_multiple_client(
            client_ids, 0, client_test_params_list, type=type, recon=recon
        )
        # imgs_q_list = list of [B, H, W, 3]

        if tag is not None:  # write to tensorboard
            num_imgs = 4
            row_lists = [imgs_q_list, synths_list]
            row_lists = [torch.concat(row_list, dim=0) for row_list in row_lists]
            if indices is None:
                # indices = torch.randperm(len(row_lists[0]))[:num_imgs]
                indices = range(num_imgs)
            grid = make_grid([row_list[indices] for row_list in row_lists], num_imgs=num_imgs)

            # num_imgs = 4
            # imgs = torch.concat(imgs_q_list, dim=0)
            # synths = torch.concat(synths_list, dim=0)
            # if indices is None:
            #     indices = torch.randperm(len(imgs))[:num_imgs]
            # grid = make_grid([imgs[indices], synths[indices]], num_imgs=num_imgs)

            self.writer.add_image(
                f"{tag}/synths",
                grid,
                round,
            )
            self.logger.write_scalars(
                f"{tag}",
                round,
                **metric_round,
            )
            self.log(
                # f"{tag}: psnr: {metric_round.psnr.avg:.4f}, psnr_max: {metric_round.psnr.max:.4f}"
                f"{tag}: psnr: {metric_round.psnr.avg:.4f}"
            )
        return metric_round, client_test_params_list

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
        num_points_per_ray = self.args.num_points_per_ray
        gradclip = self.args.client_gradclip
        lr = self.args.tto_lr
        white_bkgd = self.args.white_bkgd
        steps = accumulated_steps[-1]
        adapt_optim = getattr(self.args, "adapt_optim", "sgd")

        # set model and optimizer
        client_params = copy_model_params(model_params)
        self.model.load_state_dict(client_params, strict=True)
        self.model.train()
        optimizer = self.get_optimizer(adapt_optim, list(self.model.parameters()), lr=lr)

        # data
        task_id, imgs_s, poses_s, imgs_q, poses_q, hwf, bound = task_data
        # [num_support_views, H, W, 3], [num_support_views, 4, 4],
        # [num_query_views, H, W, 3], [num_query_views, 4, 4],
        # [4], [2]

        rays_o, rays_d = get_rays_shapenet(hwf, poses_s)  # [num_views, H, W, 3]
        pixels = rearrange(imgs_s, "v h w c -> (v h w) c")  # [num_views * H * W, 3]
        rays_o = rearrange(rays_o, "v h w c -> (v h w) c")  # [num_views * H * W, 3]
        rays_d = rearrange(rays_d, "v h w c -> (v h w) c")  # [num_views * H * W, 3]
        num_total_rays = rays_d.shape[0]

        if 0 in accumulated_steps:
            metric, synths = test_model(
                self.model, imgs_q, poses_q, hwf, bound, num_points_per_ray, test_ray_bs, white_bkgd
            )
            metric.update(loss=0)
            metric_list.append(metric)
            # sample one view
            sampled_view = torch.randint(0, len(imgs_q), (1,))
            synth_list.append(synths[sampled_view])
            gt_list.append(imgs_q[sampled_view])

        set_seed(self.args.seed)
        for step in range(1, steps + 1):
            optimizer.zero_grad()

            indices = torch.randint(num_total_rays, size=[ray_bs], device=rays_o.device)
            indices = repeat(indices, "... -> ... c", c=3)
            raybatch_o = torch.gather(rays_o, 0, indices)  # [ray_bs, 3]
            raybatch_d = torch.gather(rays_d, 0, indices)  # [ray_bs, 3]
            pixelbatch = torch.gather(pixels, 0, indices)  # [ray_bs, 3]
            t_vals, xyz = sample_points(
                raybatch_o, raybatch_d, bound[0], bound[1], num_points_per_ray, perturb=True
            )  # [ray_bs, num_points_per_ray], [ray_bs, num_points_per_ray, 3]

            rgbs, sigmas = self.model(xyz)
            colors = volume_render(rgbs, sigmas, t_vals, white_bkgd=white_bkgd)  # [ray_bs, 3]
            loss = torch.nn.functional.mse_loss(colors, pixelbatch)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), gradclip)
            optimizer.step()

            if step in accumulated_steps:
                metric, synths = test_model(
                    self.model, imgs_q, poses_q, hwf, bound, num_points_per_ray, test_ray_bs, white_bkgd
                )
                metric.update(loss=loss.item())
                metric_list.append(metric)
                # sample one view
                sampled_view = torch.randint(0, len(imgs_q), (1,))
                synth_list.append(synths[sampled_view])
                gt_list.append(imgs_q[sampled_view])

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
                task_id, imgs_s, poses_s, imgs_q, poses_q, hwf, bound = task_data
                task_bs = imgs_s.shape[0]

                for task_idx in range(task_bs):
                    task_id_i = task_id[task_idx]
                    imgs_s_i = imgs_s[task_idx]
                    poses_s_i = poses_s[task_idx]
                    imgs_q_i = imgs_q[task_idx]
                    poses_q_i = poses_q[task_idx]
                    hwf_i = hwf[task_idx]
                    bound_i = bound[task_idx]

                    # support set
                    num_support_views = client.num_views[task_id_i.item()]["support"]
                    imgs_s_i = imgs_s_i[:num_support_views]
                    poses_s_i = poses_s_i[:num_support_views]

                    # query set
                    num_query_views = client.num_views[task_id_i.item()]["query"]
                    imgs_q_i = imgs_q_i[:num_query_views]
                    poses_q_i = poses_q_i[:num_query_views]

                    if num_test_support_views is not None:
                        assert num_test_support_views <= imgs_q_i.shape[0]
                        imgs_q_i = imgs_q_i[:num_test_support_views]
                        poses_q_i = poses_q_i[:num_test_support_views]

                    one_task_data = (task_id_i, imgs_s_i, poses_s_i, imgs_q_i, poses_q_i, hwf_i, bound_i)
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
                    # weight_per_task_list.append(num_support_views)
                    weight_per_task_list.append(1)

            params_per_client_list.append(params_per_task_list)

            client_synth_list = synth_per_task_list[0]  # only the first task
            client_gt_list = gt_per_task_list[0]  # only the first task

            # merge metric per client
            client_metric_list = []
            for a_s_idx, a_s in enumerate(accumulated_steps):
                metric = MetricLogger()
                for metric_list, weight in zip(metric_per_task_list, weight_per_task_list):
                    metric.update(**metric_list[a_s_idx], weight=weight)  # weight: num_support_views
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

            self.log(f"Steps:{a_s}, (step) loss: {metric_a_s.loss.avg:.4f}, psnr: {metric_a_s.psnr.avg:.4f}")

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
