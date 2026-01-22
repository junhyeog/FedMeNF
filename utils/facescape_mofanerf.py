import json
import os
from pathlib import Path
from typing import List

import cv2
import imageio
import numpy as np
import torch
from torch.utils.data import Dataset


class FaceScapeMoFaNeRF:
    def __init__(self, root):
        self.root = root
        self.all_expression_names = [
            "neutral",
            "smile",
            "mouth_stretch",
            "anger",
            "jaw_left",
            "jaw_right",
            "jaw_forward",
            "mouth_left",
            "mouth_right",
            "dimpler",
            "chin_raiser",
            "lip_puckerer",
            "lip_funneler",
            "sadness",
            "lip_roll",
            "grin",
            "cheek_blowing",
            "eye_closed",
            "brow_raiser",
            "brow_lower",
        ]
        self.all_expression_ids = list(range(len(self.all_expression_names)))
        # l_dir = os.listdir(root)
        l_dir = sorted(os.listdir(root))  # !!
        l_img = []
        l_train = []
        l_val = []
        l_test = []
        l_all = []
        for i in l_dir:
            if i.find("train") != -1:
                l_train.append(i)
            elif i.find("val") != -1:
                l_val.append(i)
            elif i.find("test") != -1:
                l_test.append(i)
            elif i.find("all") != -1:
                l_all.append(i)
            else:
                l_img.append(i)
        assert len(l_train) == len(l_val) == len(l_test) == len(l_all) == 294

        self.all_person_ids = sorted([int(i) for i in l_img])
        for person_id in self.all_person_ids:
            for type, l in zip(["train", "val", "test", "all"], [l_train, l_val, l_test, l_all]):
                assert f"transforms_{type}_{person_id}.json" in l

        # self.publishable_list = [122, 212, 340, 344, 393, 395, 421, 527, 594, 610]
        self.publishable_list = [122, 340, 344, 393, 395, 421, 527, 610]


# https://github.com/zhuhao-nju/mofanerf/blob/11f9965acfa43ffa50eadc2c3000fd4bd69c96b4/run_train.py#L25
class FaceScapeMoFaNeRFDataset(Dataset, FaceScapeMoFaNeRF):
    """
    returns the images, poses and instrinsics of a partiucular scene
    # oad img, accoding to meta, 100images for trains, 13vals, 25 tests # TODO: check
    """

    def __init__(
        self,
        root: str,
        person_id: int,
        expression_ids: List[int],
        num_support_views: int,
        num_query_views: int,
        type: str,
        res_scale: int,
        pose_scale: float = 10.0,
        load_imgs: bool = True,
    ):
        """
        Args:
            root (str): root path of the dataset
            person_id (int): persion id
            expression_ids (list): list of expression_id. dataset gets len(expression_ids) objects
            num_support_views (int): number of support views to return for each scene
            num_query_views (int): number of query views to return for each scene
            type (str): specifies whether to return "train", "val" or "test" dataset
        """
        Dataset.__init__(self)
        FaceScapeMoFaNeRF.__init__(self, root)
        self.root = root
        self.person_id = person_id
        assert person_id in self.all_person_ids
        self.expression_ids = sorted(expression_ids)
        for exp_id in self.expression_ids:
            assert exp_id < len(self.all_expression_names)
        self.exp_id_to_idx = {exp_id: idx for idx, exp_id in enumerate(self.expression_ids)}
        self.num_support_views = num_support_views
        self.num_query_views = num_query_views
        self.num_total_views = num_support_views + num_query_views
        self.type = type
        self.res_scale = res_scale
        self.pose_scale = pose_scale
        self.load_imgs = load_imgs

        self.preprocessing()

    def _load_imgs(self):
        all_imgs_list = []
        all_poses_list = []
        # exp_idxs = np.random.permutation(range(len(self.expression_ids)))
        exp_idxs = range(len(self.expression_ids))
        for exp_idx in exp_idxs:
            all_img_paths = self.all_img_paths[exp_idx]
            all_imgs = []
            all_poses = self.all_poses[exp_idx]
            for img_path in all_img_paths:
                img = imageio.imread(img_path)
                if self.res_scale != 1:
                    img = cv2.resize(img, (self.W, self.H), interpolation=cv2.INTER_AREA)
                img = torch.as_tensor(img, dtype=torch.float)
                all_imgs.append(img)

            all_imgs = torch.stack(all_imgs, dim=0) / 255.0
            # not white background
            all_imgs = all_imgs[..., :3]

            all_imgs_list.append(all_imgs)
            all_poses_list.append(all_poses)

        self.all_imgs_list = all_imgs_list
        self.all_poses_list = all_poses_list

    def preprocessing(self):
        # * metadata
        with open(
            os.path.join(self.root, f"transforms_{self.type}_{self.person_id}.json"),
            "r",
            encoding="utf-8",
        ) as fp:
            metadata = json.load(fp)
        self.metadata = metadata

        # * H, W
        self.H = 512
        self.W = 512

        # * near, far
        # https://github.com/zhuhao-nju/mofanerf/blob/11f9965acfa43ffa50eadc2c3000fd4bd69c96b4/run_train.py#L194
        self.near = 8.0 / self.pose_scale
        self.far = 26.0 / self.pose_scale
        self.bound = torch.as_tensor([self.near, self.far], dtype=torch.float).cuda()

        # * focal
        camera_angle_x = float(self.metadata["camera_angle_x"])
        self.camera_angle_x = torch.as_tensor(camera_angle_x, dtype=torch.float)
        self.focal = 0.5 * self.W / torch.tan(0.5 * self.camera_angle_x)

        # * res_scale
        self.H = self.H // self.res_scale
        self.W = self.W // self.res_scale
        self.focal = self.focal / self.res_scale

        # * hwf
        self.hwf = torch.as_tensor([self.H, self.W, self.focal], dtype=torch.float).cuda()

        # * all_img_paths, all_poses
        all_img_paths = [[] for _ in self.expression_ids]
        all_poses = [[] for _ in self.expression_ids]
        frame_idxs = np.random.permutation(range(len(self.metadata["frames"])))
        for frame_idx in frame_idxs:
            frame = self.metadata["frames"][frame_idx]
            exp_id = int(frame["expression"])
            # check valid expression
            if exp_id not in self.expression_ids:
                continue
            exp_idx = self.exp_id_to_idx[exp_id]
            # check if the number of images for the expression is enough
            if len(all_img_paths[exp_idx]) >= self.num_total_views:
                continue
            img_path = f"{self.root}{frame['file_path']}.png"  # file_path includes /, so no need to add it
            all_img_paths[exp_idx].append(img_path)
            pose = frame["transform_matrix"]
            all_poses[exp_idx].append(pose)
        self.all_img_paths = all_img_paths
        self.all_poses = torch.as_tensor(all_poses, dtype=torch.float).cuda()  # [num_exp, num_img, 4, 4]

        # scaling poses
        self.all_poses[:, :, :3] = self.all_poses[:, :, :3] / self.pose_scale

        if self.load_imgs:
            self._load_imgs()

    def __getitem__(self, idx):
        if self.load_imgs:
            all_imgs = self.all_imgs_list[idx]
            all_poses = self.all_poses_list[idx]
        else:
            all_img_paths = self.all_img_paths[idx]
            all_imgs = []
            all_poses = self.all_poses[idx]
            for img_path in all_img_paths:
                img = imageio.imread(img_path)
                if self.res_scale != 1:
                    img = cv2.resize(img, (self.W, self.H), interpolation=cv2.INTER_AREA)
                img = torch.as_tensor(img, dtype=torch.float)
                all_imgs.append(img)

            all_imgs = torch.stack(all_imgs, dim=0) / 255.0
            # not white background
            all_imgs = all_imgs[..., :3]

        # split the images and poses into support and query sets
        imgs_s, imgs_q = torch.split(all_imgs, [self.num_support_views, self.num_query_views], dim=0)
        poses_s, poses_q = torch.split(all_poses, [self.num_support_views, self.num_query_views], dim=0)
        return (
            torch.tensor(idx, dtype=torch.long),
            imgs_s.cuda(),
            poses_s.cuda(),
            imgs_q.cuda(),
            poses_q.cuda(),
            self.hwf.cuda(),
            self.bound.cuda(),
        )

    def __len__(self):
        return len(self.expression_ids)


def build_facescape_modanerf(
    root: str,
    person_ids: List[int],
    expression_ids_per_person: dict[List[int]],
    num_support_views: int,
    num_query_views: int,
    type: str = "all",
    res_scale: int = 2,
    pose_scale: float = 10.0,
):
    """
    Args:
        root: root path of the dataset
        person_ids: list of person_ids
        expression_ids_per_person:
        num_support_views: num of support views to return from a single scene
        num_query_views: num of query views to return from a single scene
        type: specifies whether to return ""all", train", "val" or "test" dataset
    """
    print(f"[-] person_ids: {person_ids}")
    print(f"[-] expression_ids_per_person: {expression_ids_per_person}")
    print(f"[-] num_support_views: {num_support_views}")
    print(f"[-] num_query_views: {num_query_views}")
    print(f"[-] type: {type}")
    print(f"[-] res_scale: {res_scale}")
    print(f"[-] pose_scale: {pose_scale}")

    ds_list = []
    for i, person_id in enumerate(person_ids):
        dataset = FaceScapeMoFaNeRFDataset(
            root=root,
            person_id=person_id,
            expression_ids=expression_ids_per_person[i],
            num_support_views=num_support_views,
            num_query_views=num_query_views,
            type=type,
            res_scale=res_scale,
            pose_scale=pose_scale,
        )
        ds_list.append(dataset)
    return ds_list
