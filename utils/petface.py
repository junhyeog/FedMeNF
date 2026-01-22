import csv
import json
import os
import random
from collections import OrderedDict
from copy import deepcopy
from pathlib import Path
from typing import Tuple, Union

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from einops import rearrange, repeat
from PIL import Image
from torch.utils.data import ConcatDataset, Dataset, Subset
from torchvision import datasets, transforms

from utils.misc import set_seed

# from leaf import CusteomLEAF


class PetFace(Dataset):

    def __init__(
        self,
        root: str,
        category: str,
        n_pets: int = -1,
        num_objects_per_client: int = -1,
        test_ratio: float = 0.25,
        transform=None,
        device="cuda",
    ):
        # assert split in ["train", "test"]
        self.root = root
        self.n_pets = n_pets
        self.test_ratio = test_ratio
        self.transform = transform
        self.device = device
        self.num_objects_per_client = num_objects_per_client

        img_path = os.path.join(root, f"images/{category}")
        self.img_path = img_path

        exclude_path = os.path.join(root, f"excluded/{category}_excluded_images.json")
        with open(exclude_path, mode="r", encoding="utf-8") as file:
            exclude = json.load(file)

        id_list = sorted(list(exclude.keys()))

        path_dict = {}
        for id in id_list:
            exclude_file = exclude[id]
            id_dir = os.path.join(img_path, str(id))
            files = sorted(os.listdir(id_dir))
            files = [f for f in files if f not in exclude_file]
            if len(files) < 2:
                continue

            # shuffle files
            files = np.random.permutation(files)
            if num_objects_per_client > 0:
                files = files[:num_objects_per_client]

            path_dict[id] = [os.path.join(id_dir, f) for f in files]

        assert len(path_dict) >= n_pets

        id_list = np.random.permutation(list(path_dict.keys()))[:n_pets]

        pet_id_to_client_id = {}
        client_id_to_pet_id = {}
        client_id_to_paths = {}
        client_id_to_idxs = {}
        file_list = []
        file_to_idx = {}
        data = []
        file_id_list = []

        for i, pet_id in enumerate(id_list):
            exsisting_files = path_dict[pet_id]
            pet_id_to_client_id[pet_id] = i
            client_id_to_pet_id[i] = pet_id
            client_id_to_paths[i] = exsisting_files
            client_id_to_idxs[i] = []
            for j, file in enumerate(path_dict[pet_id]):
                file_list.append(file)
                idx = len(file_list) - 1
                file_to_idx[file] = idx
                client_id_to_idxs[i].append(idx)
                img = Image.open(file).convert("RGB")
                if self.transform is not None:
                    img = self.transform(img)
                img = rearrange(img, "c h w -> 1 h w c").to(device)
                data.append(img)
                file_id_list.append(f"{pet_id}_{file.split('/')[-1].split('.')[0]}")

        # dict: client_id -> list of idx
        dict_train = {}
        dict_test = {}
        for i in range(n_pets):
            idxs = client_id_to_idxs[i]
            idxs = np.random.permutation(idxs)
            n_test = int(len(idxs) * test_ratio)
            n_test = max(n_test, 1)
            dict_train[i] = idxs[n_test:]
            dict_test[i] = idxs[:n_test]

        self.path_dict = path_dict
        self.id_list = id_list
        self.pet_id_to_client_id = pet_id_to_client_id
        self.client_id_to_pet_id = client_id_to_pet_id
        self.client_id_to_paths = client_id_to_paths
        self.client_id_to_idxs = client_id_to_idxs
        self.file_list = file_list
        self.file_to_idx = file_to_idx
        self.dict_train = dict_train
        self.dict_test = dict_test
        self.data = data
        self.file_id_list = file_id_list
        self.targets = [file.split("/")[-2] for file in file_list]
        self.classes = list(set(self.targets))

    def __len__(self):
        return len(self.file_id_list)

    def __getitem__(self, idx):
        id = self.file_id_list[idx]
        data = self.data[idx]
        return id, data, data
