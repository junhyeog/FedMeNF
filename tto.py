import argparse
import importlib

import numpy as np
import torch

from models.fedavg_maml import FedAvg_MAML
from utils.logger import MetricLogger
from utils.misc import set_seed


def test(
    ml: FedAvg_MAML,
    round: int,
    tto0: bool = False,
    postfix: str = None,
    num_test_support_views=None,
):
    set_seed(ml.args.seed)
    if (round + 1) % (ml.args.save_period) == 0 or (round + 1) == ml.args.num_rounds or round < 0:
        participant_ids = range(ml.args.num_clients)
        steps_per_round = ml.args.steps_per_round
    else:
        num_test_clients = (ml.args.num_clients + 5 - 1) // 5
        participant_ids = np.random.choice(range(ml.args.num_clients), num_test_clients, replace=False)
        steps_per_round = ml.args.steps_per_round[: max(11, len(ml.args.steps_per_round) // 2)]

    if tto0:
        # * tto step 0: reconstruction at the server
        metric_test_tto0_s, _ = ml.client_test_multiple_client_tto0(
            client_ids=range(ml.args.num_clients),  # all clients
            client_test_params_list=None,
            model_params="local",
            type="train",
            recon="s",
            tag="tto0_q_tto",
            round=round,
        )

        metric_test_tto0_q, _ = ml.client_test_multiple_client_tto0(
            client_ids=range(ml.args.num_clients),  # all clients
            client_test_params_list=None,
            model_params="local",
            type="train",
            recon="q",
            tag="tto0_q_tto",
            round=round,
        )

        # tto0 for both support set and query set
        metric_test_tto0 = MetricLogger()
        metric_test_tto0.update(**metric_test_tto0_s, weight=ml.args.num_support_views)
        metric_test_tto0.update(**metric_test_tto0_q, weight=ml.args.num_query_views)
        ml.logger.write_scalars("tto0", round, **metric_test_tto0)

    # * tto
    metric_test, client_test_params_list = ml.adapt_multi_client_all_tasks(
        client_ids=participant_ids,
        model_params="global",
        accumulated_steps=ml.args.accumulated_steps,
        tag=f"test_round_{round:06}",
        postfix=postfix,
        num_test_support_views=num_test_support_views,
    )

    ml.logger.write_scalars("test", round, **metric_test)

    # ! OOD
    metric_ood, client_ood_params_list = None, None
    if ml.args.num_ood_clients > 0:
        participant_ids = range(ml.args.num_clients, ml.args.num_total_clients)
        if ml.args.adapt_optim == "sgd":
            metric_ood, client_ood_params_list = ml.client_test_multiple_client_multiple_rounds(
                client_ids=participant_ids,
                steps_per_round=steps_per_round,
                client_test_params_list=None,
                model_params="global",
                tag=f"OOD_round_{round:06}",
                postfix=postfix,
                num_test_support_views=num_test_support_views,
            )
        else:
            metric_ood, client_ood_params_list = ml.adapt_multi_client_all_tasks(
                client_ids=participant_ids,
                model_params="global",
                accumulated_steps=ml.args.accumulated_steps,
                tag=f"OOD_round_{round:06}",
                postfix=postfix,
                num_test_support_views=num_test_support_views,
            )

        ml.logger.write_scalars("OOD", round, **metric_ood)
    return metric_test, client_test_params_list, metric_ood, client_ood_params_list


if __name__ == "__main__":

    def type_fn(type_obj):
        return lambda x: None if x == "None" else type_obj(x)

    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_dir", type=str)
    parser.add_argument("--round", type=int)
    parser.add_argument("--num_test_support_views", type=type_fn(int), default=None)
    parser.add_argument("--test_ray_bs", type=type_fn(int), default=None)
    parser.add_argument("--test_task_bs", type=type_fn(int), default=None)
    parser.add_argument("--tto_lr", type=type_fn(float), default=None)
    parser.add_argument("--tto0", action="store_true")

    parser.add_argument("--adapt_optim", type=type_fn(str), default=None)
    parser.add_argument("--tto_steps", type=type_fn(int), default=None)
    parser.add_argument("--tto_step_period", type=type_fn(int), default=None)
    parser.add_argument("--seed", type=type_fn(int), default=None)

    tto_args = parser.parse_args()

    ckpt_path = f"{tto_args.ckpt_dir}/ckpt_{tto_args.round:06}.pt"
    ckpt = torch.load(ckpt_path, weights_only=False)

    # * update args
    args = ckpt["args"]

    update_args = {
        "num_test_support_views": {
            "to_be_updated": False,
            "postfix": True,
        },
        "adapt_optim": {
            "to_be_updated": True,
            "postfix": True,
        },
        "tto_lr": {
            "to_be_updated": True,
            "postfix": True,
        },
        "test_task_bs": {
            "to_be_updated": True,
            "postfix": False,
        },
        "test_ray_bs": {
            "to_be_updated": True,
            "postfix": False,
        },
        "tto_steps": {
            "to_be_updated": True,
            "postfix": False,
        },
        "tto_step_period": {
            "to_be_updated": True,
            "postfix": False,
        },
        "seed": {
            "to_be_updated": True,
            "postfix": True,
        },
    }

    for key, value in update_args.items():
        if getattr(tto_args, key) is not None:
            orig_val = getattr(args, key)
            new_val = getattr(tto_args, key)
            if orig_val != new_val:
                if value["to_be_updated"]:
                    print(f"[+] Update {key}: {orig_val} -> {new_val}")
                    setattr(args, key, new_val)
                else:
                    print(f"[-] Skip {key}: {orig_val} -> {new_val} (but, used as tto_args)")
            else:  # orig_val == new_val
                update_args[key]["postfix"] = False
        else:
            update_args[key]["postfix"] = False

    # * build model
    # set device
    if getattr(args, "device", None) is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    set_seed(args.seed)

    models_modules = importlib.import_module("models")
    model_classes = {k.lower(): v for k, v in models_modules.__dict__.items() if isinstance(v, type)}
    ml: FedAvg_MAML = model_classes[args.algo](args)

    # * load model
    # ml.model.load_state_dict(ckpt["model"])
    ml.hpnet.load_state_dict(ckpt["hpnet"])
    # client_test_params_list = ckpt["client_test_params_list"]
    # client_ood_params_list = ckpt["client_ood_params_list"]

    postfix = "tto_"
    for key, value in update_args.items():
        if not value["postfix"]:
            continue
        short_k = "".join([s[0] for s in key.split("_")])
        postfix += f"{short_k}_{getattr(args, key)}_"

    metric_test, client_test_params_list, metric_ood, client_ood_params_list = test(
        ml,
        tto_args.round,
        tto0=tto_args.tto0,
        num_test_support_views=tto_args.num_test_support_views,
        postfix=postfix,
    )

    # save
    to_saved = {
        "args": ml.args,
        "tto_args": tto_args,
        # "model": ml.model.state_dict(),
        "hpnet": ml.hpnet.state_dict(),
        "hpnet_optimizer": ml.hpnet_optimizer.state_dict(),
        "client_model_params_list": [client.model_params for client in ml.clients],
        "client_test_params_list": client_test_params_list,
        "client_ood_params_list": client_ood_params_list,
    }
    file_name = ckpt_path.replace(".pt", f"_{postfix}tto_steps_{args.tto_steps}.pt")
    torch.save(
        to_saved,
        file_name,
    )
    ml.log(f"Saved: {file_name}")
