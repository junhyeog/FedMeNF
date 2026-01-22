import math
import os
import random
import time
from collections import OrderedDict

import numpy as np
import torch
import torchvision
from einops import rearrange
from scipy.spatial.transform import Rotation as R
from torchmetrics.image import (PeakSignalNoiseRatio,
                                StructuralSimilarityIndexMeasure)
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity


def round_sig(x, sig):
    return round(x, sig - int(math.floor(math.log10(abs(x)))) - 1)


def set_seed(seed):
    """for reproducibility
    :param seed:
    :return:
    """
    np.random.seed(seed)
    random.seed(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_time():
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())


def copy_model_params(params: OrderedDict, requires_grad=True):
    new_params = OrderedDict()
    for k, v in params.items():
        new_params[k] = v.clone().detach().requires_grad_(requires_grad)
    return new_params


def get_vram_usage_gib():
    free, total = torch.cuda.mem_get_info()
    free = free / 1024**3
    total = total / 1024**3
    remaining = total - free
    return f"{remaining:.2f} / {total:.2f} (GiB)"


def make_grid(batched_recon_images_list, num_imgs=4):
    for i in range(len(batched_recon_images_list)):
        imgs = batched_recon_images_list[i]
        if imgs.dim() == 5:
            pattern = "b n h w c -> (b n) c h w"
        elif imgs.dim() == 4:
            pattern = "n h w c -> n c h w"
        elif imgs.dim() == 3:
            pattern = "h w c -> 1 c h w"
        else:
            raise ValueError(f"[!] Invalid shape: {imgs.shape()}")
        batched_recon_images_list[i] = rearrange(imgs, pattern)[:num_imgs].clamp(0, 1).cpu()
    grid = torchvision.utils.make_grid(
        torch.cat(batched_recon_images_list, dim=0), nrow=len(batched_recon_images_list[0])
    )
    return grid


def allclose_dict(dict1, dict2):
    # check keys
    if dict1.keys() != dict2.keys():
        return False
    for k, v in dict1.items():
        if not torch.allclose(v, dict2[k]):
            return False
    return True


def allclose_one_test_params(params):
    for k, v in params.items():
        for i in range(1, len(v)):
            if not torch.allclose(v[0], v[i]):
                return False


def allclose_test_params(params1, params2):
    if len(params1) != len(params2):
        return False
    for (k1, v1), (k2, v2) in zip(params1.items(), params2.items()):
        if k1 != k2:
            return False
        if not torch.allclose(v1[0], v2[0]):
            return False


def psnr_fn(pred, target):
    """pred, target should be [0, 1]

    Args:
        pred (torch.tensor): [N, C, H, W]
        target (torch.tensor): [N, C, H, W]

    Returns:
        torch.tensor: []
    """
    _psnr = PeakSignalNoiseRatio(data_range=(0, 1)).cuda()
    return _psnr(pred, target)


def ssim_fn(pred, target):
    """pred, target should be [0, 1]

    Args:
        pred (torch.tensor): [N, C, H, W]
        target (torch.tensor): [N, C, H, W]

    Returns:
        torch.tensor: []
    """
    _ssim = StructuralSimilarityIndexMeasure(data_range=(0, 1)).cuda()
    return _ssim(pred, target)


def lpips_fn(pred, target):
    """pred, target should be [0, 1]

    Args:
        pred (torch.tensor): [N, C, H, W]
        target (torch.tensor): [N, C, H, W]

    Returns:
        torch.tensor: []
    """
    _lpips = LearnedPerceptualImagePatchSimilarity(normalize=True).cuda()
    return _lpips(pred, target)


def find_folder(root_dir, startswith=None, endswith=None):
    assert startswith != endswith and (startswith or endswith)
    matching_folders = []
    for root, dirs, files in os.walk(root_dir):
        for folder in dirs:
            flag = True
            if startswith is not None and not folder.startswith(startswith):
                flag = False
            if endswith is not None and not folder.endswith(endswith):
                flag = False
            if flag:
                matching_folders.append(os.path.join(root, folder))
    assert len(matching_folders) == 1, f"Found {len(matching_folders)} folders: {matching_folders}"
    return matching_folders[0]


def rotate_poses(poses, dict_angles):
    """rotate poses

    Args:
        poses (torch.tensor): transform_matrix [B, 4, 4]
        dict_angles (dict): {rotation direction (x, y, z): rotation angle (radian)}
    Returns:
        _type_: _description_
    """
    device = poses.device
    dtype = poses.dtype
    poses = poses.clone().cpu().numpy()

    combined_rotations = R.from_matrix(poses[:, :3, :3])

    camera_positions = poses[:, :3, 3]

    for seq, angle in dict_angles.items():
        new_rotation = R.from_euler(seq, angle)

        combined_rotations = new_rotation * combined_rotations

    distances = np.linalg.norm(camera_positions, axis=1)
    rotated_directions = combined_rotations.apply(np.array([0, 0, -1]))
    new_camera_positions = -rotated_directions * distances[:, np.newaxis]

    new_poses = np.eye(4).reshape(1, 4, 4).repeat(len(poses), axis=0)
    new_poses[:, :3, :3] = combined_rotations.as_matrix()
    new_poses[:, :3, 3] = new_camera_positions

    new_poses = torch.tensor(new_poses, dtype=dtype, device=device)
    return new_poses
