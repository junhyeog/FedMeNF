from collections import OrderedDict

import torch
from torch.utils.data import DataLoader, Dataset

from utils.logger import MetricLogger, Value
from utils.misc import copy_model_params


class Client:
    def __init__(self, id: int, dl_train: DataLoader, dl_test: DataLoader):
        self.id = id
        self.dl_train = dl_train
        self.dl_test = dl_test
        self.model_params = OrderedDict()
        self.train_data_size = len(dl_train.dataset)
        self.test_data_size = len(dl_test.dataset)
        # self.downloaded_bytes = Value()
        # self.uploaded_bytes = Value()

    def get_dl(self, type: str):
        if type == "train":
            return self.dl_train
        elif type == "test":
            return self.dl_test
        else:
            raise ValueError(f"Unknown dataloader type: {type}")

    # def download_model_params(self, model_params, requires_grad=True):
    #     self.model_params = copy_model_params(model_params, requires_grad)
    #     self.update_download(self.model_params)
    #     return self.model_params

    # def upload_model_params(self, model_params, requires_grad=True):
    #     client_model_params = copy_model_params(model_params, requires_grad)
    #     self.update_upload(client_model_params)
    #     return client_model_params

    # def get_bytes_of(self, obj):
    #     if isinstance(obj, torch.Tensor):
    #         return obj.element_size() * obj.nelement() / 1024 / 1024
    #     elif isinstance(obj, (int, float)):
    #         return 4 / 1024 / 1024
    #     elif isinstance(obj, dict):
    #         size = 0
    #         for k, v in obj.items():
    #             size += self.get_bytes_of(v)
    #         return size
    #     elif hasattr(obj, "__iter__"):
    #         size = 0
    #         for v in obj:
    #             size += self.get_bytes_of(v)
    #         return size
    #     else:
    #         raise ValueError(f"Unknown type: {type(obj)}")

    # def update_download(self, obj):
    #     self.downloaded_bytes.update(self.get_bytes_of(obj))

    # def update_upload(self, obj):
    #     self.uploaded_bytes.update(self.get_bytes_of(obj))
