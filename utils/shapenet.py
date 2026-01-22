import json
from pathlib import Path

import imageio
import numpy as np
import torch
from torch.utils.data import Dataset

from utils.misc import rotate_poses


# https://github.com/sanowar-raihan/nerf-meta/blob/main/datasets/shapenet.py
class ShapenetDataset(Dataset):
    """
    returns the images, poses and instrinsics of a partiucular scene
    """

    def __init__(self, all_folders, num_support_views, num_query_views, init_rotate_prob=0, device="cuda"):
        """
        Args:
            all_folders (list): list of folder paths. each folder contains indiviual scene info
            num_support_views (int): number of support views to return for each scene
            num_query_views (int): number of query views to return for each scene
        """
        super().__init__()
        self.all_folders = all_folders
        self.num_support_views = num_support_views
        self.num_query_views = num_query_views
        self.num_total_views = num_support_views + num_query_views
        self.init_rotate_prob = init_rotate_prob
        self.device = device

        all_imgs_list = []
        all_poses_list = []
        hwf_list = []
        bound_list = []
        all_uuids = []

        for folderpath in all_folders:
            meta_path = folderpath.joinpath("transforms.json")
            with open(meta_path, "r") as meta_file:
                meta_data = json.load(meta_file)

            all_imgs = []
            all_poses = []

            frame_idxs = np.random.permutation(range(len(meta_data["frames"])))
            frame_idxs = frame_idxs[: self.num_total_views]
            for frame_idx in frame_idxs:
                frame = meta_data["frames"][frame_idx]

                img_name = f"{Path(frame['file_path']).stem}.png"
                img_path = folderpath.joinpath(img_name)
                img = imageio.imread(img_path)
                all_imgs.append(torch.as_tensor(img, dtype=torch.float))

                pose = frame["transform_matrix"]
                all_poses.append(torch.as_tensor(pose, dtype=torch.float))

            all_imgs = torch.stack(all_imgs, dim=0) / 255.0
            # composite the images to a white background
            all_imgs = all_imgs[..., :3] * all_imgs[..., -1:] + 1 - all_imgs[..., -1:]

            all_poses = torch.stack(all_poses, dim=0)

            # all images of a scene has the same camera intrinsics
            H, W = all_imgs[0].shape[:2]
            camera_angle_x = meta_data["camera_angle_x"]
            camera_angle_x = torch.as_tensor(camera_angle_x, dtype=torch.float)

            # camera angle equation: tan(angle/2) = (W/2)/focal
            focal = 0.5 * W / torch.tan(0.5 * camera_angle_x)
            hwf = torch.as_tensor([H, W, focal], dtype=torch.float)

            # all shapenet scenes are bounded between 2. and 6.
            near = 2.0
            far = 6.0
            bound = torch.as_tensor([near, far], dtype=torch.float)

            all_imgs_list.append(all_imgs)
            all_poses_list.append(all_poses)
            hwf_list.append(hwf)
            bound_list.append(bound)
            all_uuids.append(meta_data["model"])

        self.all_imgs = torch.stack(all_imgs_list, dim=0).to(device)  # (N, V, H, W, 3)
        self.all_poses = torch.stack(all_poses_list, dim=0).to(device)  # (N, V, 4, 4)
        self.hwf = torch.stack(hwf_list, dim=0).to(device)  # (N, 3)
        self.bound = torch.stack(bound_list, dim=0).to(device)  # (N, 2)
        self.all_uuids = all_uuids

        # pose rotation
        if self.init_rotate_prob > 0:
            print(f"[+] Rotating the poses (p:{self.init_rotate_prob})")
            for object_idx, poses in enumerate(self.all_poses):
                p = np.random.rand()
                if p < self.init_rotate_prob:
                    dict_angles = {k: np.random.uniform(0, np.pi * 2) for k in ["x", "y", "z"]}
                    self.all_poses[object_idx] = rotate_poses(self.all_poses[object_idx], dict_angles)

    def __getitem__(self, idx):
        # split the images and poses into support and query sets
        imgs_s, imgs_q = torch.split(self.all_imgs[idx], [self.num_support_views, self.num_query_views], dim=0)
        poses_s, poses_q = torch.split(self.all_poses[idx], [self.num_support_views, self.num_query_views], dim=0)
        hwf = self.hwf[idx]
        bound = self.bound[idx]
        return (
            torch.tensor(idx, dtype=torch.long),
            imgs_s.to(self.device),
            poses_s.to(self.device),
            imgs_q.to(self.device),
            poses_q.to(self.device),
            hwf.to(self.device),
            bound.to(self.device),
        )

    def __len__(self):
        return len(self.all_folders)
