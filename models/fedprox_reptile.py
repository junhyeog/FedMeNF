from argparse import Namespace

from models.fedavg_reptile import FedAvg_Reptile
from utils.clipping import clip_norm_
from utils.logger import MetricLogger
from utils.misc import copy_model_params


class FedProx_Reptile(FedAvg_Reptile):
    def __init__(self, args: Namespace, log_dict: dict = {}):
        _log_dict = {
            "fedprox_mu": "fm",
        }
        log_dict.update(_log_dict)
        super().__init__(args, log_dict=log_dict)

        self.fedprox_mu = args.fedprox_mu

    def outer_loop_batch(
        self,
        client_params,
        client_params_batch,
        gradclip,
        lr,
    ):
        assert hasattr(self, "fedprox_mu")

        metric = MetricLogger()

        # FedProx
        global_params = copy_model_params(self.hpnet(), requires_grad=False)

        grad = []
        for (n, p), (n_batch, p_batch) in zip(client_params.items(), client_params_batch.items()):
            assert n == n_batch
            assert p.shape == p_batch.mean(0).shape
            g = p - p_batch.mean(0)

            # FedProx
            g += self.fedprox_mu * (p - global_params[n])

            grad.append(g)

        clip_norm_(grad, gradclip)
        for (n, p), g in zip(client_params.items(), grad):
            client_params[n] = p - lr * g

        metric.update(loss=0)

        return client_params, metric
