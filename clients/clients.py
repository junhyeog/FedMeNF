import numpy as np
import torch
from einops import repeat
from torch.utils.data import DataLoader, RandomSampler

from clients.client import Client
from utils.data_utils import (DatasetSplit, get_data,
                              get_data_facescape_modanerf,
                              get_data_facescape_mofanerf_manual)
from utils.misc import rotate_poses
from utils.sampling import dirichlet_distribution_noniid_slice, iid


class Clients:

    def __init__(self, ml):
        dataset = getattr(ml.args, "dataset", None)

        nerf_datasets = ["facescape_mofanerf", "facescape_mofanerf_manual", "cars", "lamps", "chairs"]
        img_datasets = ["petface_cat", "golfdb8"]

        if dataset in nerf_datasets:
            self.nerf_init(ml)

        elif dataset in img_datasets:
            self.img_init(ml)
        else:
            raise ValueError(f"Invalid data type: {dataset}")

        return

    def img_init(self, ml):
        num_total_clients = ml.args.num_clients + ml.args.num_ood_clients
        self._clients = []
        dataset = ml.args.dataset.lower()
        self.ds_train, self.ds_test, self.dict_train, self.dict_test = get_data(ml)
        ml.args.num_classes = len(self.ds_train.classes)
        for client_id in range(num_total_clients):
            ds_train = DatasetSplit(self.ds_train, self.dict_train[client_id])
            dl_train = DataLoader(
                ds_train,
                batch_size=ml.args.task_bs,
                shuffle=True,
            )
            dl_test = DataLoader(
                DatasetSplit(self.ds_test, self.dict_test[client_id]),
                batch_size=ml.args.test_task_bs,
                shuffle=False,
            )
            self._clients.append(Client(client_id, dl_train, dl_test))

        # check client's data size
        for client in self._clients:
            assert client.train_data_size == len(self.dict_train[client.id])
            assert client.test_data_size == len(self.dict_test[client.id])

        # dummy train_num_views, test_num_views
        for client in self._clients:
            train_data_size = client.train_data_size
            test_data_size = client.test_data_size

            client.num_views = {"total": {"support": train_data_size + test_data_size, "query": 1 + 1}}
            client.train_num_views = {"total": {"support": train_data_size, "query": 1}}
            client.test_num_views = {"total": {"support": test_data_size, "query": 1}}

    def nerf_init(self, ml):
        num_total_clients = ml.args.num_clients + ml.args.num_ood_clients
        self._clients = []
        dataset = ml.args.dataset.lower()
        if dataset == "facescape_mofanerf":
            ds_train_list, ds_test_list = get_data_facescape_modanerf(ml)
            for client_id in range(num_total_clients):
                dl_train = DataLoader(
                    ds_train_list[client_id],
                    batch_size=ml.args.task_bs,
                    shuffle=True,
                )
                dl_test = DataLoader(
                    ds_test_list[client_id],
                    batch_size=ml.args.test_task_bs,
                    shuffle=False,
                )
                self._clients.append(Client(client_id, dl_train, dl_test))
        elif dataset == "facescape_mofanerf_manual":
            ds_train_list, ds_test_list = get_data_facescape_mofanerf_manual(ml)
            for client_id in range(num_total_clients):
                dl_train = DataLoader(
                    ds_train_list[client_id],
                    batch_size=ml.args.task_bs,
                    shuffle=True,
                )
                dl_test = DataLoader(
                    ds_test_list[client_id],
                    batch_size=ml.args.test_task_bs,
                    shuffle=False,
                )
                self._clients.append(Client(client_id, dl_train, dl_test))
        else:
            self.ds_train, self.ds_test, self.dict_train, self.dict_test = get_data(ml)
            ml.args.num_classes = len(self.ds_train.classes)
            for client_id in range(num_total_clients):
                ds_train = DatasetSplit(self.ds_train, self.dict_train[client_id])
                dl_train = DataLoader(
                    ds_train,
                    batch_size=ml.args.task_bs,
                    shuffle=True,
                )
                dl_test = DataLoader(
                    DatasetSplit(self.ds_test, self.dict_test[client_id]),
                    batch_size=ml.args.test_task_bs,
                    shuffle=False,
                )
                self._clients.append(Client(client_id, dl_train, dl_test))

            # check client's data size
            for client in self._clients:
                assert client.train_data_size == len(self.dict_train[client.id])
                assert client.test_data_size == len(self.dict_test[client.id])

        if ml.args.views_iid:
            for client in self._clients:
                client.num_views = {"total": {"support": 0, "query": 0}}
                client.train_num_views = {"total": {"support": 0, "query": 0}}
                client.train_task_ids = []
                for data in client.dl_train:
                    task_id = data[0].item()
                    client.train_num_views[task_id] = {
                        "support": ml.args.num_support_views,
                        "query": ml.args.num_query_views,
                    }
                    client.train_num_views["total"] = {
                        "support": client.train_num_views["total"]["support"] + ml.args.num_support_views,
                        "query": client.train_num_views["total"]["query"] + ml.args.num_query_views,
                    }
                    client.num_views[task_id] = {"support": ml.args.num_support_views, "query": ml.args.num_query_views}
                    client.num_views["total"] = {
                        "support": client.num_views["total"]["support"] + ml.args.num_support_views,
                        "query": client.num_views["total"]["query"] + ml.args.num_query_views,
                    }
                    client.train_task_ids.append(task_id)
                client.test_num_views = {"total": {"support": 0, "query": 0}}
                client.test_task_ids = []
                for data in client.dl_test:
                    task_id = data[0].item()
                    client.test_num_views[task_id] = {
                        "support": ml.args.num_test_support_views,
                        "query": ml.args.num_test_query_views,
                    }
                    client.test_num_views["total"] = {
                        "support": client.test_num_views["total"]["support"] + ml.args.num_test_support_views,
                        "query": client.test_num_views["total"]["query"] + ml.args.num_test_query_views,
                    }
                    client.num_views[task_id] = {
                        "support": ml.args.num_test_support_views,
                        "query": ml.args.num_test_query_views,
                    }
                    client.num_views["total"] = {
                        "support": client.num_views["total"]["support"] + ml.args.num_test_support_views,
                        "query": client.num_views["total"]["query"] + ml.args.num_test_query_views,
                    }
                    client.test_task_ids.append(task_id)
        else:
            num_train_objects = num_total_clients * ml.args.num_objects_per_client
            num_test_objects = num_total_clients * ml.args.num_test_objects_per_client

            total_support_views = ml.args.mean_num_support_views * num_train_objects
            total_query_views = ml.args.mean_num_query_views * num_train_objects
            total_test_support_views = ml.args.mean_num_test_support_views * num_test_objects
            total_test_query_views = ml.args.mean_num_test_query_views * num_test_objects

            # train
            dict_s_train = dirichlet_distribution_noniid_slice(
                np.zeros(total_support_views), num_train_objects, ml.args.views_alpha
            )
            n_s_train = [min(len(dict_s_train[o_i]), ml.args.num_support_views) for o_i in range(num_train_objects)]
            s_train_dist = [[0 for _ in range(n_s_train[o_i])] for o_i in range(num_train_objects)]
            dict_q_train = dirichlet_distribution_noniid_slice(
                np.zeros(total_query_views),
                num_train_objects,
                ml.args.views_alpha,
                prior=s_train_dist,
            )
            n_q_train = [min(len(dict_q_train[o_i]), ml.args.num_query_views) for o_i in range(num_train_objects)]

            # test
            dict_s_test = dirichlet_distribution_noniid_slice(
                np.zeros(total_test_support_views), num_test_objects, ml.args.views_alpha
            )
            n_s_test = [min(len(dict_s_test[o_i]), ml.args.num_test_support_views) for o_i in range(num_test_objects)]
            s_test_dist = [[0 for _ in range(n_s_test[o_i])] for o_i in range(num_test_objects)]
            dict_q_test = dirichlet_distribution_noniid_slice(
                np.zeros(total_test_query_views),
                num_test_objects,
                ml.args.views_alpha,
                prior=s_test_dist,
            )
            n_q_test = [min(len(dict_q_test[o_i]), ml.args.num_test_query_views) for o_i in range(num_test_objects)]
            #
            ml.log(f"n_s_train: {n_s_train}")
            ml.log(f"n_q_train: {n_q_train}")
            ml.log(f"n_s_test: {n_s_test}")
            ml.log(f"n_q_test: {n_q_test}")
            #
            train_obj_idx = 0
            test_obj_idx = 0
            for client in self._clients:
                # train
                client.num_views = {"total": {"support": 0, "query": 0}}
                client.train_num_views = {"total": {"support": 0, "query": 0}}
                client.train_task_ids = []
                for data in client.dl_train:
                    task_id = data[0].item()
                    client.train_num_views[task_id] = {
                        "support": n_s_train[train_obj_idx],
                        "query": n_q_train[train_obj_idx],
                    }
                    client.train_num_views["total"] = {
                        "support": client.train_num_views["total"]["support"] + n_s_train[train_obj_idx],
                        "query": client.train_num_views["total"]["query"] + n_q_train[train_obj_idx],
                    }
                    client.num_views[task_id] = {
                        "support": n_s_train[train_obj_idx],
                        "query": n_q_train[train_obj_idx],
                    }
                    client.num_views["total"] = {
                        "support": client.num_views["total"]["support"] + n_s_train[train_obj_idx],
                        "query": client.num_views["total"]["query"] + n_q_train[train_obj_idx],
                    }
                    client.train_task_ids.append(task_id)
                    train_obj_idx += 1

                # test
                client.test_num_views = {"total": {"support": 0, "query": 0}}
                client.test_task_ids = []
                for data in client.dl_test:
                    task_id = data[0].item()
                    client.test_num_views[task_id] = {
                        "support": n_s_test[test_obj_idx],
                        "query": n_q_test[test_obj_idx],
                    }
                    client.test_num_views["total"] = {
                        "support": client.test_num_views["total"]["support"] + n_s_test[test_obj_idx],
                        "query": client.test_num_views["total"]["query"] + n_q_test[test_obj_idx],
                    }
                    client.num_views[task_id] = {
                        "support": n_s_test[test_obj_idx],
                        "query": n_q_test[test_obj_idx],
                    }
                    client.num_views["total"] = {
                        "support": client.num_views["total"]["support"] + n_s_test[test_obj_idx],
                        "query": client.num_views["total"]["query"] + n_q_test[test_obj_idx],
                    }
                    client.test_task_ids.append(task_id)
                    test_obj_idx += 1

        # log data partition
        obj_idx = 0
        for client in self._clients:
            for k, v in client.train_num_views.items():
                ml.logger.write_scalars(
                    "num_views/train",
                    obj_idx,
                    support=v["support"],
                    query=v["query"],
                )
                obj_idx += 1

        obj_idx = 0
        for client in self._clients:
            for k, v in client.test_num_views.items():
                ml.logger.write_scalars(
                    "num_views/test",
                    obj_idx,
                    support=v["support"],
                    query=v["query"],
                )
                obj_idx += 1

    def __len__(self):
        return len(self._clients)

    def __getitem__(self, idx: int) -> Client:
        return self._clients[idx]
