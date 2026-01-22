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


class GolfDB8(Dataset):

    def __init__(
        self,
        root: str,
        n_players: int = -1,
        num_objects_per_client: int = -1,
        test_ratio: float = 0.25,
        transform=None,
        device="cuda",
    ):
        # assert split in ["train", "test"]
        self.root = root
        self.n_players = n_players
        self.test_ratio = test_ratio
        self.transform = transform
        self.device = device

        img_path = os.path.join(root, "images_160")
        self.img_path = img_path

        metadata_path = os.path.join(root, "golfDB_slice.csv")

        use_ids_path = os.path.join(root, "use_ids.txt")
        with open(use_ids_path, mode="r", encoding="utf-8") as file:
            use_ids = [int(line) for line in file]

        metadata = {}
        player_to_id = {}
        with open(metadata_path, mode="r", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            # 'id', 'youtube_id', 'player', 'sex', 'club', 'view', 'slow', 'events', 'bbox', 'split'
            # 'id', 'youtube_id', 'player', 'sex', 'club', 'view'
            for row in reader:
                row.pop("")
                row["id"] = int(row["id"])
                # row["slow"] = int(row["slow"])
                # row["events"] = [int(i) for i in row["events"][1:-1].split()]
                # row["bbox"] = [float(i) for i in row["bbox"][1:-1].split()]
                # row["split"] = int(row["split"])

                if row["id"] not in use_ids:
                    continue

                metadata[row["id"]] = row

                if row["player"] not in player_to_id:
                    player_to_id[row["player"]] = []
                player_to_id[row["player"]].append(row["id"])

        player_list = [k for k, v in player_to_id.items() if len(v) > 1]
        player_list = np.random.permutation(player_list)

        assert len(player_list) >= n_players

        # limit number of ids per player to num_objects_per_client
        if num_objects_per_client > 0:
            for k, v in player_to_id.items():
                v = np.random.permutation(v)
                player_to_id[k] = v[:num_objects_per_client]

        client_id_to_player = {}
        player_to_client_id = {}
        client_id_to_id = {}

        for i in range(n_players):
            player = player_list[i]
            client_id_to_player[i] = player
            player_to_client_id[player] = i
            client_id_to_id[i] = player_to_id[player]

        id_list = []
        for i in range(n_players):
            id_list.extend(client_id_to_id[i])

        id_to_idx = {id: idx for idx, id in enumerate(id_list)}

        # dict: client_id -> list of idx
        dict_train = {}
        dict_test = {}
        for i in range(n_players):
            ids = client_id_to_id[i]
            idxs = [id_to_idx[id] for id in ids]
            idxs = np.random.permutation(idxs)
            n_test = int(len(idxs) * test_ratio)
            n_test = max(n_test, 1)
            dict_train[i] = idxs[n_test:]
            dict_test[i] = idxs[:n_test]

        self.metadata = metadata
        self.player_to_id = player_to_id
        self.client_id_to_player = client_id_to_player
        self.player_to_client_id = player_to_client_id
        self.client_id_to_id = client_id_to_id
        self.id_list = id_list
        self.id_to_idx = id_to_idx
        self.dict_train = dict_train
        self.dict_test = dict_test
        self.targets = [metadata[id]["player"] for id in id_list]
        self.classes = list(set(self.targets))

        # data
        data = []
        for id in id_list:
            img_list = []
            for i in range(8):
                img = Image.open(os.path.join(img_path, f"{id}_{i}.png")).convert("RGB")
                if self.transform is not None:
                    img = self.transform(img)
                img_list.append(img)
            data.append(torch.stack(img_list, dim=0))
        data = rearrange(data, "n v c h w -> n v h w c").to(device)
        self.data = data

    def __len__(self):
        return len(self.id_list)

    def __getitem__(self, idx):
        id = self.id_list[idx]
        data = self.data[idx]
        return id, data, data
