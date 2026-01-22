import json
import random
from collections import OrderedDict
from copy import deepcopy
from pathlib import Path
from typing import Tuple, Union

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import ConcatDataset, Dataset, Subset
from torchvision import datasets, transforms

from utils.facescape_mofanerf import (FaceScapeMoFaNeRF,
                                      build_facescape_modanerf)
from utils.golfdb import GolfDB8
from utils.misc import set_seed
from utils.petface import PetFace
# from leaf import CusteomLEAF
from utils.sampling import dirichlet_distribution_noniid_slice, iid
from utils.shapenet import ShapenetDataset

trans_cifar10_train = transforms.Compose(
    [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)
trans_cifar10_test = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)
trans_cifar100_train = transforms.Compose(
    [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276]),
    ]
)
trans_cifar100_test = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276]),
    ]
)

trans_tensor = transforms.Compose(
    [
        transforms.ToTensor(),
    ]
)


def get_data_facescape_mofanerf_manual(ml):
    set_seed(ml.args.seed)  ## !
    ml.args.white_bkgd = False
    pose_scale = 10.0

    root = "data/inr/facescape/mofanerf/train/multi-view-images"
    facescape_modanerf = FaceScapeMoFaNeRF(root)

    # manual setting
    person_ids = ml.args.person_ids
    expression_ids_per_person_train = ml.args.expression_ids_per_person_train
    expression_ids_per_person_test = ml.args.expression_ids_per_person_test

    ds_train_list = build_facescape_modanerf(
        root,
        person_ids,
        expression_ids_per_person_train,
        ml.args.num_support_views,
        ml.args.num_query_views,
        type="all",
        res_scale=ml.args.res_scale,
        pose_scale=pose_scale,
    )
    ds_test_list = build_facescape_modanerf(
        root,
        person_ids,
        expression_ids_per_person_test,
        ml.args.num_test_support_views,
        ml.args.num_test_query_views,
        type="all",
        res_scale=ml.args.res_scale,
        pose_scale=pose_scale,
    )
    return ds_train_list, ds_test_list


def get_data_facescape_modanerf(ml):
    set_seed(ml.args.seed)  ## !
    ml.args.white_bkgd = False
    pose_scale = 10.0

    total_train_objects = ml.args.num_objects_per_client * ml.args.num_total_clients
    total_test_objects = ml.args.num_test_objects_per_client * ml.args.num_total_clients

    if ml.args.iid:
        dict_clients_train = iid(range(total_train_objects), ml.args.num_total_clients)
        dict_clients_test = iid(range(total_test_objects), ml.args.num_total_clients)
        ml.log(f"[+] IID distribution")
    else:
        dict_clients_train = dirichlet_distribution_noniid_slice(
            np.zeros(total_train_objects, dtype=int), ml.args.num_total_clients, ml.args.alpha
        )
        if ml.args.test_dist == "dirichlet":
            dict_clients_test = dirichlet_distribution_noniid_slice(
                np.zeros(total_test_objects, dtype=int), ml.args.num_total_clients, ml.args.alpha
            ) 
        elif ml.args.test_dist == "consistent":
            train_label_distribution = [
                np.zeros(len(dict_clients_train[client_idx]), dtype=int)
                for client_idx in range(ml.args.num_total_clients)
            ]
            dict_clients_test = dirichlet_distribution_noniid_slice(
                np.zeros(total_test_objects, dtype=int),
                ml.args.num_total_clients,
                ml.args.alpha,
                prior=train_label_distribution,
            )
        elif ml.args.test_dist == "uniform":
            dict_clients_test = iid(range(total_test_objects), ml.args.num_total_clients)
        else:
            raise NotImplementedError("test_dist should be dirichlet, consistent, or uniform")
        ml.log(f"[+] Non-IID distribution (alpha: {ml.args.alpha}, test_dist: {ml.args.test_dist})")

    root = "data/inr/facescape/mofanerf/train/multi-view-images"
    facescape_modanerf = FaceScapeMoFaNeRF(root)
    all_person_ids = facescape_modanerf.all_person_ids
    all_expression_ids = facescape_modanerf.all_expression_ids

    for i in range(ml.args.num_total_clients):
        len_train = len(dict_clients_train[i])
        len_test = len(dict_clients_test[i])
        extra = len_train + len_test - len(all_expression_ids)  # = 20
        if extra > 0:
            train_extra = extra // 2
            test_extra = extra - train_extra
            dict_clients_train[i] = dict_clients_train[i][train_extra:]
            dict_clients_test[i] = dict_clients_test[i][test_extra:]

    ml.log(f"[+] Number of samples in random clients")
    ml.log(f"    - trian:\n{[len(dict_clients_train[i]) for i in range(ml.args.num_total_clients)]}")
    ml.log(f"    - test:\n{[len(dict_clients_test[i]) for i in range(ml.args.num_total_clients)]}")

    person_ids = np.random.permutation(all_person_ids)[: ml.args.num_total_clients]
    # make sure that facescape_modanerf.publishable_list is in person_ids
    for person_id in facescape_modanerf.publishable_list:
        if person_id in person_ids:
            continue
        # find id for exchange
        for i, pid in enumerate(person_ids):
            if pid not in facescape_modanerf.publishable_list:
                person_ids[i] = person_id
                break
    expression_ids_per_person_train = []
    expression_ids_per_person_test = []
    for i, person_id in enumerate(person_ids):
        len_train = len(dict_clients_train[i])
        len_test = len(dict_clients_test[i])
        assert len_train + len_test <= len(
            all_expression_ids
        ), f"(total exp) {len(all_expression_ids)} < (train) {len_train} + (test) {len_test}"
        all_expression_ids = np.random.permutation(all_expression_ids)
        expression_ids_per_person_train.append(all_expression_ids[:len_train])
        expression_ids_per_person_test.append(all_expression_ids[len_train : len_train + len_test])

    ds_train_list = build_facescape_modanerf(
        root,
        person_ids,
        expression_ids_per_person_train,
        ml.args.num_support_views,
        ml.args.num_query_views,
        type="all",
        res_scale=ml.args.res_scale,
        pose_scale=pose_scale,
    )
    ds_test_list = build_facescape_modanerf(
        root,
        person_ids,
        expression_ids_per_person_test,
        ml.args.num_test_support_views,
        ml.args.num_test_query_views,
        type="all",
        res_scale=ml.args.res_scale,
        pose_scale=pose_scale,
    )
    return ds_train_list, ds_test_list


def get_data(ml) -> Tuple[Dataset, Dataset, dict, dict]:
    set_seed(ml.args.seed)
    num_total_clients = ml.args.num_clients + ml.args.num_ood_clients
    dataset = ml.args.dataset.lower()
    if dataset == "cifar10":
        ds_train = datasets.CIFAR10("data/cifar10", train=True, download=True, transform=trans_cifar10_train)
        ds_train.targets = np.array(ds_train.train_labels)
        ds_train.classes = np.sort(np.unique(ds_train.targets))
        ds_test = datasets.CIFAR10("data/cifar10", train=False, download=True, transform=trans_cifar10_test)
        ds_test.targets = np.array(ds_test.test_labels)
        ds_test.classes = np.sort(np.unique(ds_test.targets))
    elif dataset == "cifar100":
        ds_train = datasets.CIFAR100("data/cifar10", train=True, download=True, transform=trans_cifar100_train)
        ds_train.targets = np.array(ds_train.train_labels)
        ds_train.classes = np.sort(np.unique(ds_train.targets))
        ds_test = datasets.CIFAR100("data/cifar10", train=False, download=True, transform=trans_cifar100_test)
        ds_test.targets = np.array(ds_test.test_labels)
        ds_test.classes = np.sort(np.unique(ds_test.targets))
    elif dataset == "golfdb8":
        golfdb8 = GolfDB8(
            "data/golfdb",
            num_total_clients,
            num_objects_per_client=ml.args.num_objects_per_client,
            test_ratio=0.25,
            transform=trans_tensor,
            device=ml.args.device,
        )
        ml.log(f"[+] GolfDB8 dataset is loaded")
        ml.log(f"[+] player_to_id: {golfdb8.player_to_id}")
        ml.log(f"[+] client_id_to_player: {golfdb8.client_id_to_player}")
        ml.log(f"[+] player_to_client_id: {golfdb8.player_to_client_id}")
        ml.log(f"[+] client_id_to_id: {golfdb8.client_id_to_id}")
        ml.log(f"[+] id_list: {golfdb8.id_list}")
        ml.log(f"[+] id_to_idx: {golfdb8.id_to_idx}")
        ml.log(f"[+] dict_train: {golfdb8.dict_train}")
        ml.log(f"[+] dict_test: {golfdb8.dict_test}")

        return golfdb8, golfdb8, golfdb8.dict_train, golfdb8.dict_test
    elif dataset == "petface_cat":
        petface = PetFace(
            "data/PetFace",
            "cat",
            num_total_clients,
            num_objects_per_client=ml.args.num_objects_per_client,
            test_ratio=0.25,
            transform=trans_tensor,
            device=ml.args.device,
        )
        ml.log(f"[+] PetFace dataset is loaded")
        ml.log(f"[+] path_dict: {petface.path_dict}")
        ml.log(f"[+] id_list: {petface.id_list}")
        ml.log(f"[+] pet_id_to_client_id: {petface.pet_id_to_client_id}")
        ml.log(f"[+] client_id_to_pet_id: {petface.client_id_to_pet_id}")
        ml.log(f"[+] client_id_to_paths: {petface.client_id_to_paths}")
        ml.log(f"[+] client_id_to_idxs: {petface.client_id_to_idxs}")
        ml.log(f"[+] file_list: {petface.file_list}")
        ml.log(f"[+] file_to_idx: {petface.file_to_idx}")
        ml.log(f"[+] dict_train: {petface.dict_train}")
        ml.log(f"[+] dict_test: {petface.dict_test}")

        return petface, petface, petface.dict_train, petface.dict_test

    elif dataset in ["cars", "chairs", "lamps"]:
        ml.args.white_bkgd = True

        dataset_root = {
            "cars": "data/inr/learnit_Data/shapenet/cars/02958343",
            "chairs": "data/inr/learnit_Data/shapenet/chairs/03001627",
            "lamps": "data/inr/learnit_Data/shapenet/lamps/03636649",
        }[dataset]

        splits_path = {
            "cars": "data/inr/learnit_Data/shapenet/car_splits.json",
            "chairs": "data/inr/learnit_Data/shapenet/chair_splits.json",
            "lamps": "data/inr/learnit_Data/shapenet/lamp_splits.json",
        }[dataset]

        root_path = Path(dataset_root)
        splits_path = Path(splits_path)
        with open(splits_path, "r") as splits_file:
            splits = json.load(splits_file)

        # train: 3423, val: 10, test: 100
        all_folders = []
        for k, v in splits.items():
            all_folders.extend([root_path.joinpath(foldername) for foldername in v])
        all_folders = sorted(all_folders)
        # filter out folders that do not exist and not contain transforms.json
        all_folders = [
            folder for folder in all_folders if folder.exists() and (folder.joinpath("transforms.json")).exists()
        ]

        num_train_objects = ml.args.num_objects_per_client * ml.args.num_total_clients
        num_test_objects = ml.args.num_test_objects_per_client * ml.args.num_total_clients
        assert num_train_objects + num_test_objects <= len(
            all_folders
        ), f"(total) {len(all_folders)} < (train) {num_train_objects} + (test) {num_test_objects}"
        all_folders = np.random.permutation(all_folders)
        all_train_folders = all_folders[:num_train_objects]
        all_test_folders = all_folders[num_train_objects : num_train_objects + num_test_objects]

        ds_train = ShapenetDataset(
            all_train_folders,
            ml.args.num_support_views,
            ml.args.num_query_views,
            0,
            device=ml.args.device,
        )
        ds_test = ShapenetDataset(
            all_test_folders,
            ml.args.num_test_support_views,
            ml.args.num_test_query_views,
            0,
            device=ml.args.device,
        )
        ds_train.targets = np.zeros(len(ds_train), dtype=int)
        ds_train.classes = np.array([0])
        ds_test.targets = np.zeros(len(ds_test), dtype=int)
        ds_test.classes = np.array([0])
    else:
        raise ValueError("[!] Dataset not available")

    ml.log(f"[+] {dataset} dataset is loaded")
    ml.log(f"[+] Train: {len(ds_train)}, Test: {len(ds_test)}")

    if ml.args.iid:
        dict_clients_train = iid(ds_train, ml.args.num_total_clients)
        dict_clients_test = iid(ds_test, ml.args.num_total_clients)
        print(f"[+] IID distribution")
    else:
        dict_clients_train = dirichlet_distribution_noniid_slice(
            np.array(ds_train.targets), ml.args.num_total_clients, ml.args.alpha
        )
        if ml.args.test_dist == "dirichlet":
            dict_clients_test = dirichlet_distribution_noniid_slice(
                np.array(ds_test.targets), ml.args.num_total_clients, ml.args.alpha
            )
        elif ml.args.test_dist == "consistent":
            train_label_distribution = [
                [ds_train.targets[idx] for idx in dict_clients_train[client_idx]]
                for client_idx in range(ml.args.num_total_clients)
            ]
            dict_clients_test = dirichlet_distribution_noniid_slice(
                np.array(ds_test.targets),
                ml.args.num_total_clients,
                ml.args.alpha,
                prior=train_label_distribution,
            )
        elif ml.args.test_dist == "uniform":
            dict_clients_test = iid(ds_test, ml.args.num_total_clients)
        else:
            raise NotImplementedError("test_dist should be dirichlet, consistent, or uniform")

        ml.log(f"[+] Non-IID distribution (alpha: {ml.args.alpha}, test_dist: {ml.args.test_dist})")

        # for debug: print the number of samples in random clients
        # print the number of samples in random clients
        ml.log(f"[+] Number of samples in random clients")
        ml.log(f"    - trian:\n{[len(dict_clients_train[i]) for i in range(ml.args.num_total_clients)]}")
        ml.log(f"    - test:\n{[len(dict_clients_test[i]) for i in range(ml.args.num_total_clients)]}")

    return ds_train, ds_test, dict_clients_train, dict_clients_test


class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = np.array(idxs)
        self.targets = np.array(dataset.targets)[self.idxs]
        self.classes = dataset.classes

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        data_point = self.dataset[self.idxs[item]]
        return data_point


def plot_data_partition(
    dataset,
    dict_clients,
    num_classes,
    num_sample_clients,
    writer=None,
    tag="Data Partition",
):
    dict_clients_targets = {}
    targets = np.array(dataset.targets)

    dict_clients_targets = {client_idx: targets[data_idxs] for client_idx, data_idxs in dict_clients.items()}

    s = torch.stack(
        [
            torch.bincount(torch.tensor(data_idxs), minlength=num_classes)
            for client_idx, data_idxs in dict_clients_targets.items()
        ]
    )
    ss = torch.cumsum(s, 1)
    cmap = plt.cm.get_cmap("hsv", num_classes)
    fig, ax = plt.subplots(figsize=(20, num_sample_clients))
    ax.barh(
        [f"Client {i:3d}" for i in range(num_sample_clients)],
        s[:num_sample_clients, 0],
        color=cmap(0),
    )
    for c in range(1, num_classes):
        ax.barh(
            [f"Client {i:3d}" for i in range(num_sample_clients)],
            s[:num_sample_clients, c],
            left=ss[:num_sample_clients, c - 1],
            color=cmap(c),
        )
    # plt.show()
    if writer is not None:
        writer.add_figure(tag, fig)
