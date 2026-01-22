import argparse
import importlib
import os
import shutil
import time
import uuid
import warnings
from datetime import datetime
from modulefinder import ModuleFinder

import numpy as np
import optuna
import torch
import torchvision
from einops import rearrange
# from optuna.storages import RetryFailedTrialCallback
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm, trange

from clients import Clients
from get_params import get_params
from models import BaseModel, FedAvg_MAML
from utils.args import args_parser
from utils.data_utils import DatasetSplit, get_data, plot_data_partition
from utils.logger import MetricLogger
from utils.misc import (get_time, get_vram_usage_gib, make_grid, round_sig,
                        set_seed)

warnings.filterwarnings("ignore", category=UserWarning, message="TypedStorage is deprecated")


def train(ml: FedAvg_MAML, round: int, pbar_round):
    metric_train = MetricLogger()
    participant_ids = np.random.RandomState(ml.args.seed + round).choice(
        range(ml.args.num_clients), ml.args.num_participants, replace=False
    )

    for client_id in participant_ids:
        client_params, metric_client = ml.client_update(
            client_id,
            ml.args.client_epochs,
            load_global_model=True,
            update_client_params=True,
            epoch_to_iter=getattr(ml.args, "epoch_to_iter", False),
        )

        metric_train.update(**metric_client, weight=ml.clients[client_id].train_num_views["total"]["support"])

    ml.server_aggregation(participant_ids)

    # report training
    pbar_round.set_description(
        f"Round:{round}, (train) loss: {metric_train.loss.avg:.4f}, inner_loss: {metric_train.inner_loss.avg:.4f}"
    )
    ml.log(f"Round:{round}, (train) loss: {metric_train.loss.avg:.4f}, inner_loss: {metric_train.inner_loss.avg:.4f}")
    if (round + 1) % ml.args.print_train == 0 or (round + 1) == ml.args.num_rounds:
        ml.logger.write_scalars("train", round, **metric_train)
    return metric_train


def test(ml: FedAvg_MAML, round: int):
    set_seed(ml.args.seed)
    participant_ids = range(ml.args.num_clients)
    
    # use adapt_multi_client_all_tasks
    metric_test_tto0_q, _ = ml.adapt_multi_client_all_tasks(
        client_ids=participant_ids,
        model_params="local",
        accumulated_steps=[0],
        type="train",
        tag="tto0_q",
        tto0=True,
        round=round,
    )

    # * tto
    if ((round + 1) % ml.args.tto_period == 0 or (round + 1) == ml.args.num_rounds) and round >= 0:
        steps_per_round = ml.args.steps_per_round
        accumulated_steps = ml.args.accumulated_steps
    else:  # no inner steps
        steps_per_round = [0]
        accumulated_steps = [0]

    ml.log(
        f"[+] Start tto at round {round}. steps_per_round: {steps_per_round}, accumulated_steps: {accumulated_steps}"
    )

    metric_test, client_test_params_list = ml.adapt_multi_client_all_tasks(
        client_ids=participant_ids,
        model_params="global",
        accumulated_steps=accumulated_steps,
        tag=f"test_round_{round:06}",
    )

    ml.logger.write_scalars("test", round, **metric_test)

    # ! OOD
    metric_ood, client_ood_params_list = None, None
    if ml.args.num_ood_clients > 0:
        participant_ids = range(ml.args.num_clients, ml.args.num_total_clients)
        metric_ood, client_ood_params_list = ml.adapt_multi_client_all_tasks(
            client_ids=participant_ids,
            model_params="global",
            accumulated_steps=accumulated_steps,
            tag=f"OOD_round_{round:06}",
        )

        ml.logger.write_scalars("OOD", round, **metric_ood)
    return metric_test, client_test_params_list, metric_ood, client_ood_params_list


def main(args: argparse.Namespace, trial: optuna.Trial = None):
    # optuna
    if trial is not None:
        optuna_seed = datetime.now().microsecond
        set_seed(optuna_seed)
        print(f"[+] optuna_seed: {optuna_seed}")
        if hasattr(args, "trial_number"):
            delattr(args, "trial_number")
        params = get_params(args, trial)
        for k, v in params.items():
            if isinstance(v, float):
                params[k] = round_sig(v, 2)
        vars(args).update(params)

        for k, v in vars(args).items():
            trial.set_user_attr(k, v)

        # avoid duplicate sets
        for previous_trial in trial.study.trials:
            if (
                previous_trial.state == optuna.trial.TrialState.COMPLETE
                and trial.params == previous_trial.params  # check if the same hparams
                and trial.user_attrs == previous_trial.user_attrs  # check if the same args
            ):
                trial.set_user_attr("duplicated", True)
                print(
                    f"[+] Duplicated trial: previous trial number={previous_trial.number}, present trial number={trial.number}"
                )
                print(f"[+] return previous_trial.values: {previous_trial.values}")
                return previous_trial.values

        args.trial_number = trial.number

    # set device
    if getattr(args, "device", None) is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    set_seed(args.seed)

    models_modules = importlib.import_module("models")
    model_classes = {k.lower(): v for k, v in models_modules.__dict__.items() if isinstance(v, type)}

    log_dict = {}
    if args.exp_keys:
        k_list = args.exp_keys.split(",")
        for k in k_list:
            if k not in vars(args):
                continue
            log_dict[k] = "".join([s[0] for s in k.split("_")])

    ml: FedAvg_MAML = model_classes[args.algo](args, log_dict)

    # save code
    code_dir = os.path.join(args.log_dir, "code")
    mf = ModuleFinder([os.getcwd()])
    mf.run_script(__file__)
    for name, module in mf.modules.items():
        if module.__file__ is None:
            continue
        rel_path = os.path.relpath(module.__file__)
        new_path = os.path.join(code_dir, rel_path)
        new_dirname = os.path.dirname(new_path)
        os.makedirs(new_dirname, mode=0o750, exist_ok=True)
        shutil.copy2(rel_path, new_path)

    # check args
    ml.log(f"[+] Args (trial: {getattr(ml.args, 'trial_number', None)}) " + "-" * 62)
    for h in vars(ml.args):
        ml.log(f" | {h:30}: {getattr(ml.args, h)}")
    ml.log(" +" + "-" * 71)

    # log data partition
    for client_id in range(ml.args.num_total_clients):
        train_size = ml.clients[client_id].train_data_size
        test_size = ml.clients[client_id].test_data_size
        ml.logger.write_scalars(
            "data_partition",
            client_id,
            train_size=train_size,
            test_size=test_size,
        )

    start_round = 0 if not ml.args.load_ckpt else getattr(ml.args, "max_ckpt", -1) + 1
    ml.log(f"[+] Start training from round {start_round} to {ml.args.num_rounds}")
    pbar_round = tqdm(range(start_round, ml.args.num_rounds), desc="Round")

    # # ! test & OOD at start_round
    metric_test, client_test_params_list, metric_ood, client_ood_params_list = test(ml, start_round - 1)
    ml.args.max_ckpt = -1
    last_saved_ckpt = f"{ml.args.log_dir}/ckpt_{-1:06}.pt"

    to_saved = {
        "args": ml.args,
        # "model": ml.model.state_dict(),
        "hpnet": ml.hpnet.state_dict(),
        "hpnet_optimizer": ml.hpnet_optimizer.state_dict(),
        "client_model_params_list": [client.model_params for client in ml.clients],
        "client_test_params_list": client_test_params_list,
        "client_ood_params_list": client_ood_params_list,
    }

    torch.save(
        to_saved,
        last_saved_ckpt,
    )

    for round in pbar_round:
        ml.round = round

        # ! train
        metric_train = train(ml, round, pbar_round)

        # ! test & OOD
        if (round + 1) % ml.args.print_test == 0 or (round + 1) == ml.args.num_rounds:
            metric_test, client_test_params_list, metric_ood, client_ood_params_list = test(ml, round)

        # ! save
        if (round + 1) % (ml.args.save_period) == 0 or (round + 1) == ml.args.num_rounds:
            ml.args.max_ckpt = round
            last_saved_ckpt = f"{ml.args.log_dir}/ckpt_{round:06}.pt"

            to_saved = {
                "args": ml.args,
                # "model": ml.model.state_dict(),
                "hpnet": ml.hpnet.state_dict(),
                "hpnet_optimizer": ml.hpnet_optimizer.state_dict(),
                "client_model_params_list": [client.model_params for client in ml.clients],
                "client_test_params_list": client_test_params_list,
                "client_ood_params_list": client_ood_params_list,
            }

            torch.save(
                to_saved,
                last_saved_ckpt,
            )

        # ml.log(f"[+] Round:{round}, GPU memory usage: {get_vram_usage_gib()}")

    ml.log("[+] Done main")
    torch.cuda.empty_cache()
    ml.logger.writer.flush()
    ml.logger.writer.close()
    ml.logger.log_file.close()

    last_tto_psnr_test = metric_test[sorted([i for i in metric_test.keys() if "psnr" in i])[-1]]
    last_tto_psnr_ood = 0
    if metric_ood is not None:
        last_tto_psnr_ood = metric_ood[sorted([i for i in metric_ood.keys() if "psnr" in i])[-1]]

    return last_tto_psnr_test, last_tto_psnr_ood


def objective(args, trial):
    # return main(args, trial)
    try:
        return main(args, trial)
    except Exception as e:
        error_message = f"[{get_time()}] ERROR: {e}"
        with open(f"{args.log_dir}/log.txt", "a") as f:
            f.write(f"{error_message}\n")
        print(error_message)
        raise e


if __name__ == "__main__":
    from time import sleep

    secs = 3 + getattr(os, "getppid", lambda: 0)() % 3
    print(f"[+] Sleeping for {secs} seconds...")
    sleep(secs)

    args = args_parser()

    # optuna
    study_path = f"studys/{args.tag}/u:{args.num_clients},{args.num_ood_clients},{args.num_participants}/{args.dataset}/d:{int(args.iid)},{args.alpha}_no:{args.num_objects_per_client},{args.num_test_objects_per_client}"
    study_name = f"{args.algo}"

    os.makedirs(study_path, exist_ok=True)
    samplers = {
        "rand": optuna.samplers.RandomSampler(seed=args.seed),
        "tpe": optuna.samplers.TPESampler(multivariate=True, seed=args.seed),
        # "grid": optuna.samplers.GridSampler(),
    }
    storage = optuna.storages.RDBStorage(
        url=f"sqlite:///{study_path}/{study_name}.db",
        heartbeat_interval=60,
        grace_period=120,
        # failed_trial_callback=RetryFailedTrialCallback(max_retry=3),
    )

    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        load_if_exists=True,
        directions=["maximize", "maximize"],
        sampler=samplers[args.sampler],
    )

    study.optimize(lambda t: objective(args, t), n_trials=args.num_trials)
