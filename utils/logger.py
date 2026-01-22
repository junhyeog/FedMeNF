import logging
import os
import sys
from argparse import Namespace
from collections import OrderedDict, defaultdict, deque
from datetime import datetime
from logging.handlers import RotatingFileHandler

import optuna
import torch
from torch.utils.tensorboard import SummaryWriter

from utils.misc import get_time

# from einops import rearrange

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class STDLOG(object):
    def __init__(self, log_file):
        self.terminal = sys.stdout
        self.log_file = log_file

    def write(self, message):
        self.terminal.write(message)
        self.terminal.flush()
        self.log_file.write(message)
        self.log_file.flush()

    def flush(self):
        self.terminal.flush()
        self.log_file.flush()
        pass


# https://gist.github.com/gyglim/1f8dfb1b5c82627ae3efcfbbadb9f514
class Logger(object):
    def __init__(self, log_dir: str):

        # tensorBoard
        self.writer = SummaryWriter(log_dir)

        # log file
        self.log_file = open(os.path.join(log_dir, "log.txt"), "a")

        # logger = logging.getLogger(args.uuid)
        # logger.setLevel(logging.NOTSET)
        # formatter = logging.Formatter(fmt="[%(asctime)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

        # streamHandler = logging.StreamHandler(sys.stdout)
        # streamHandler.setFormatter(formatter)
        # logger.addHandler(streamHandler)

        # fileMaxBytes = 100 * 1024 * 1024  # 100MiB
        # fileHandler = RotatingFileHandler(
        #     os.path.join(args.log_dir, "log.txt"), mode="a", maxBytes=fileMaxBytes, backupCount=0
        # )
        # fileHandler.setFormatter(formatter)
        # logger.addHandler(fileHandler)

        # optuna.logging.enable_propagation()

        # self.logger = logger
        # args.logger = logger

    def log(self, string):
        # self.logger.log(logging.INFO, string)
        parsed_string = f"[{get_time()}] {string}"

        self.log_file.write(f"{parsed_string}\n")
        self.log_file.flush()

        print(parsed_string)
        sys.stdout.flush()

    def write_scalars(self, tag, step, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.writer.add_scalar(f"{tag}/{k}", v, step, new_style=True)
        self.writer.flush()


# https://github.com/NVIDIA/DeepLearningExamples/blob/master/PyTorch/Segmentation/MaskRCNN/pytorch/maskrcnn_benchmark/utils/metric_logger.py
class Value(object):
    def __init__(self):
        self.values = []
        self.weights = []
        self.total = 0.0
        self.total_weight = 0.0
        self.count = 0

    def update(self, value, weight=1):
        self.values.append(value)
        self.weights.append(weight)
        self.count += 1
        self.total += value * weight
        self.total_weight += weight

    def reset(self):
        self.values.clear()
        self.weights.clear()
        self.total = 0.0
        self.total_weight = 0.0
        self.count = 0

    @property
    def median(self):
        d = torch.tensor(self.values, dtype=torch.float32)
        return d.median().item()

    @property
    def avg(self):
        return self.total / self.total_weight

    @property
    def max(self):
        return max(self.values)

    @property
    def min(self):
        return min(self.values)


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(Value)
        self.delimiter = delimiter

    def update(self, weight=1, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, MetricLogger):
                if k.startswith("prefix_"):
                    self.update(
                        weight=weight, **{f"{k[7:]}_{_k}": _v for _k, _v in v.get_dict().items()}
                    )
                else:
                    self.update(weight=weight, **v.get_dict())
            else:
                if isinstance(v, torch.Tensor):
                    v = v.item()
                assert isinstance(v, (float, int))
                self.meters[k].update(v, weight=weight)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append("{}: {:.4f}".format(name, meter.avg))
        return self.delimiter.join(loss_str)

    def get_dict(self, property="avg"):
        loss_dict = {}
        for name, meter in self.meters.items():
            loss_dict[name] = eval(f"meter.{property}")
        return loss_dict

    def reset(self):
        for meter in self.meters.values():
            meter.reset()

    def __getitem__(self, key):
        return self.meters[key].avg

    def keys(self):
        return self.meters.keys()

    def __contains__(self, key):
        return key in self.meters.keys()

    @property
    def dict(self):
        return self.get_dict()
