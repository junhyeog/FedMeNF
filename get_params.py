import argparse

import optuna


def get_params(args: argparse.Namespace, trial: optuna.Trial) -> dict:
    algo = args.algo.lower()
    dataset = args.dataset.lower()
    tag = args.tag

    if "cli" in tag:
        print("[+] not use get_params.py")
        return {}

    print("[+] not use get_params.py")
    return {}
