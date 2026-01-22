import numpy as np
import torch
import torchvision
from einops import rearrange

from utils.logger import MetricLogger
from utils.misc import lpips_fn, psnr_fn, ssim_fn

############ functa (minimal nerf) https://github.com/google-deepmind/functa/blob/667f3a0c4ad59d5585db2c0807de60934bf1f398/minimal_nerf.py

############ nerf-meta # https://github.com/sanowar-raihan/nerf-meta/blob/main/models/rendering.py


def get_rays_shapenet(hwf, poses):
    """
    shapenet camera intrinsics are defined by H, W and focal.
    this function can handle multiple camera poses at a time.
    Args:
        hwf (3,): H, W, focal
        poses (N, 4, 4): pose for N number of images

    Returns:
        rays_o (N, H, W, 3): ray origins
        rays_d (N, H, W, 3): ray directions
    """
    if poses.ndim == 2:
        poses = poses.unsqueeze(dim=0)  # if poses has shape (4, 4)
        # make it (1, 4, 4)

    H, W, focal = hwf
    xx, yy = torch.meshgrid(
        torch.arange(0.0, W, device=focal.device),
        torch.arange(0.0, H, device=focal.device),
        indexing="xy",
    )
    direction = torch.stack(
        [(xx - 0.5 * W) / focal, -(yy - 0.5 * H) / focal, -torch.ones_like(xx)], dim=-1
    )  # (H, W, 3)

    rays_d = torch.einsum("hwc, nrc -> nhwr", direction, poses[:, :3, :3])  # (N, H, W, 3)
    rays_d = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)

    rays_o = poses[:, :3, -1]  # (N, 3)
    rays_o = rays_o[:, None, None, :].expand_as(rays_d)  # (N, H, W, 3)

    return rays_o, rays_d


def get_rays_shapenet_batch(hwf, poses):
    """
    Args:
        hwf (B, 3): H, W, focal
        poses (B, N, 4, 4): pose for N number of images

    Returns:
        rays_o (B, N, H, W, 3): ray origins
        rays_d (B, N, H, W, 3): ray directions
    """
    rays = [get_rays_shapenet(hwf[i], poses[i]) for i in range(hwf.shape[0])]
    rays_o, rays_d = zip(*rays)
    rays_o = torch.stack(rays_o, dim=0)
    rays_d = torch.stack(rays_d, dim=0)
    return rays_o, rays_d


def get_rays_tourism(H, W, kinv, pose):
    """
    phototourism camera intrinsics are defined by H, W and kinv.
    Args:
        H: image height
        W: image width
        kinv (3, 3): inverse of camera intrinsic
        pose (4, 4): camera extrinsic
    Returns:
        rays_o (H, W, 3): ray origins
        rays_d (H, W, 3): ray directions
    """
    yy, xx = torch.meshgrid(torch.arange(0.0, H, device=kinv.device), torch.arange(0.0, W, device=kinv.device))
    pixco = torch.stack([xx, yy, torch.ones_like(xx)], dim=-1)

    directions = torch.matmul(pixco, kinv.T)  # (H, W, 3)

    rays_d = torch.matmul(directions, pose[:3, :3].T)
    rays_d = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)  # (H, W, 3)

    rays_o = pose[:3, -1].expand_as(rays_d)  # (H, W, 3)

    return rays_o, rays_d


def sample_points(rays_o, rays_d, near, far, num_samples, perturb=False):
    """
    Sample points along the ray
    Args:
        rays_o (num_rays, 3): ray origins
        rays_d (num_rays, 3): ray directions
        near (float): near plane
        far (float): far plane
        num_samples (int): number of points to sample along each ray
        perturb (bool): if True, use randomized stratified sampling
    Returns:
        t_vals (num_rays, num_samples): sampled t values
        coords (num_rays, num_samples, 3): coordinate of the sampled points
    """
    num_rays = rays_o.shape[0]

    t_vals = torch.linspace(near, far, num_samples, device=rays_o.device)
    t_vals = t_vals.expand(num_rays, num_samples)  # t_vals has shape (num_samples)
    # we must broadcast it to (num_rays, num_samples)
    if perturb:
        rand = torch.rand_like(t_vals) * (far - near) / num_samples
        t_vals = t_vals + rand

    coords = rays_o.unsqueeze(dim=-2) + t_vals.unsqueeze(dim=-1) * rays_d.unsqueeze(dim=-2)

    return t_vals, coords


# https://github.com/zhaobozb/layout2im/blob/031f65bb6e5735eb91dd766d865d964041a597e9/models/bilinear.py#L246
def tensor_linspace(start, end, steps=10):
    """
    Vectorized version of torch.linspace.

    Inputs:
    - start: Tensor of any shape
    - end: Tensor of the same shape as start
    - steps: Integer

    Returns:
    - out: Tensor of shape start.size() + (steps,), such that
      out.select(-1, 0) == start, out.select(-1, -1) == end,
      and the other elements of out linearly interpolate between
      start and end.
    """
    assert start.size() == end.size()
    view_size = start.size() + (1,)
    w_size = (1,) * start.dim() + (steps,)
    out_size = start.size() + (steps,)

    start_w = torch.linspace(1, 0, steps=steps, device=start.device)
    start_w = start_w.view(w_size).expand(out_size)
    end_w = torch.linspace(0, 1, steps=steps, device=start.device)
    end_w = end_w.view(w_size).expand(out_size)

    start = start.contiguous().view(view_size).expand(out_size)
    end = end.contiguous().view(view_size).expand(out_size)

    out = start_w * start + end_w * end
    return out


def sample_points_batch(rays_o, rays_d, near, far, num_samples, perturb=False):
    """
    Sample points along the ray
    Args:
        rays_o (B, num_rays, 3): ray origins
        rays_d (B, num_rays, 3): ray directions
        near (B): tensor of near plane
        far (B): tensor of far plane
        num_samples (int): number of points to sample along each ray
        perturb (bool): if True, use randomized stratified sampling
    Returns:
        t_vals (B, num_rays, num_samples): sampled t values
        coords (B, num_rays, num_samples, 3): coordinate of the sampled points
    """
    b, num_rays = rays_o.shape[:2]

    t_vals = tensor_linspace(near, far, steps=num_samples)
    if t_vals.dim() == 2:  # (B, num_samples)
        t_vals = t_vals.unsqueeze(dim=1).expand(b, num_rays, num_samples)
    # (B, num_rays, num_samples)

    if perturb:
        # rand = torch.rand_like(t_vals) * (far-near)/num_samples
        rand = torch.rand_like(t_vals)  # TODO: fix range to (-1, 1)?
        d = (far - near) / num_samples
        if d.dim() == 1:
            d = d.reshape(b, 1, 1)
        elif d.dim() == 2:
            d = d.reshape(b, num_rays, 1)

        rand = rand * d

        t_vals = t_vals + rand
        del rand
    coords = rays_o.unsqueeze(dim=-2) + t_vals.unsqueeze(dim=-1) * rays_d.unsqueeze(
        dim=-2
    )  # (B, num_rays, num_samples, 3)

    return t_vals, coords  # (B, num_rays, num_samples), (B, num_rays, num_samples, 3)


def volume_render(rgbs, sigmas, t_vals, white_bkgd=False):
    """
    Volume rendering function.
    Args:
        rgbs (num_rays, num_samples, 3): colors
        sigmas (num_rays, num_samples): densities
        t_vals (num_rays, num_samples): sampled t values
        white_bkgd (bool): if True, assume white background
    Returns:
        color (num_rays, 3): color of the ray
    """
    # for phototourism, final delta is infinity to capture background
    # https://github.com/tancik/learnit/issues/4

    if white_bkgd:
        bkgd = 1e-3
    else:
        bkgd = 1e10

    deltas = t_vals[:, 1:] - t_vals[:, :-1]
    delta_final = bkgd * torch.ones_like(deltas[:, -1:])
    deltas = torch.cat([deltas, delta_final], dim=-1)  # (num_rays, num_samples)

    alphas = 1 - torch.exp(-deltas * sigmas)
    transparencies = torch.cat(
        [torch.ones_like(alphas[:, :1]), torch.cumprod(1 - alphas[:, :-1] + 1e-10, dim=-1)], dim=-1
    )
    weights = alphas * transparencies  # (num_rays, num_samples)

    # color = torch.sum(rgbs*weights.unsqueeze(-1), dim=-2)
    color = torch.einsum("rsc, rs -> rc", rgbs, weights)

    if white_bkgd:
        # composite the image to a white background
        color = color + 1 - weights.sum(dim=-1, keepdim=True)

    return color


def volume_render_batch(rgbs, sigmas, t_vals, white_bkgd=False):
    """
    Volume rendering function.
    Args:
        rgbs (B, num_rays, num_samples, 3): colors
        sigmas (B, num_rays, num_samples): densities
        t_vals (B, num_rays, num_samples): sampled t values
        white_bkgd (bool): if True, assume white background
    Returns:
        color (B, num_rays, 3): color of the ray
    """
    # for phototourism, final delta is infinity to capture background

    if white_bkgd:
        bkgd = 1e-3
    else:
        bkgd = 1e10

    deltas = t_vals[:, :, 1:] - t_vals[:, :, :-1]  # (B, num_rays, num_samples-1)
    delta_final = bkgd * torch.ones_like(deltas[:, :, -1:])  # (B, num_rays, 1)
    deltas = torch.cat([deltas, delta_final], dim=-1)  # (B, num_rays, num_samples)

    alphas = 1 - torch.exp(-deltas * sigmas)  # (B, num_rays, num_samples)
    transparencies = torch.cat(
        [
            torch.ones_like(alphas[:, :, :1]),  # (B, num_rays, 1)
            torch.cumprod(1 - alphas[:, :, :-1] + 1e-10, dim=-1),  # (B, num_rays, num_samples-1)
        ],
        dim=-1,
    )  # (B, num_rays, num_samples)
    weights = alphas * transparencies  # (B, num_rays, num_samples)

    color = torch.einsum("brsc, brs -> brc", rgbs, weights)  # (B, num_rays, 3)

    if white_bkgd:
        # composite the image to a white background
        color = color + 1 - weights.sum(dim=-1, keepdim=True)  # (B, num_rays, 3)

    return color  # (B, num_rays, 3)


@torch.no_grad()
def infer_model_batch(
    functional_batch,
    client_params_batch,
    poses,
    hwf,
    bound,
    num_points_per_ray,
    ray_bs,
    white_bkgd,
):
    rays_o, rays_d = get_rays_shapenet_batch(hwf, poses)  # [task_bs, num_views, H, W, 3]
    _, num_views, H, W, _ = rays_o.shape
    rays_o = rearrange(rays_o, "b v h w c -> b (v h w) c")  # [task_bs, num_views * H * W, 3]
    rays_d = rearrange(rays_d, "b v h w c -> b (v h w) c")  # [task_bs, num_views * H * W, 3]
    num_total_rays = rays_d.shape[1]
    # [task_bs, num_total_rays, num_points_per_ray], [task_bs, num_total_rays, num_points_per_ray, 3]

    synths = []
    for i in range(0, num_total_rays, ray_bs):
        start_idx, end_idx = i, i + ray_bs

        t_vals, xyz = sample_points_batch(
            rays_o[:, start_idx:end_idx],
            rays_d[:, start_idx:end_idx],
            bound[:, 0],
            bound[:, 1],
            num_points_per_ray,
            perturb=True,
        )
        rgbs, sigmas = functional_batch(client_params_batch, xyz)
        colors = volume_render_batch(rgbs, sigmas, t_vals, white_bkgd=white_bkgd)  # [task_bs, ray_bs, 3]
        synths.append(colors)

    synths = torch.cat(synths, dim=1)  # [task_bs, num_total_rays, 3]
    synths = rearrange(synths, "b (v h w) c -> b v h w c", v=num_views, h=H, w=W)

    return synths


@torch.no_grad()
def test_model_batch(
    functional_batch,
    client_params_batch,
    imgs,
    poses,
    hwf,
    bound,
    num_points_per_ray,
    ray_bs,
    white_bkgd,
):
    metric = MetricLogger()

    synths = infer_model_batch(
        functional_batch,
        client_params_batch,
        poses,
        hwf,
        bound,
        num_points_per_ray,
        ray_bs,
        white_bkgd,
    )
    imgs_ = rearrange(imgs, "b v h w c -> (b v) c h w").clamp(0, 1)
    synths_ = rearrange(synths, "b v h w c -> (b v) c h w").clamp(0, 1)

    loss = torch.nn.functional.mse_loss(synths_, imgs_)
    psnr = psnr_fn(synths_, imgs_)
    ssim = ssim_fn(synths_, imgs_)
    lpips = lpips_fn(synths_, imgs_)

    metric.update(loss=loss.item(), psnr=psnr.item(), ssim=ssim.item(), lpips=lpips.item())
    return metric, synths


@torch.no_grad()
def infer_model(
    model,
    poses,
    hwf,
    bound,
    num_points_per_ray,
    ray_bs,
    white_bkgd,
):
    rays_o, rays_d = get_rays_shapenet(hwf, poses)  # [num_views, H, W, 3]
    num_views, H, W, _ = rays_o.shape
    rays_o = rearrange(rays_o, "v h w c -> (v h w) c")  # [num_views * H * W, 3]
    rays_d = rearrange(rays_d, "v h w c -> (v h w) c")  # [num_views * H * W, 3]
    num_total_rays = rays_d.shape[0]
    # [num_total_rays, num_points_per_ray], [num_total_rays, num_points_per_ray, 3]

    synths = []
    for i in range(0, num_total_rays, ray_bs):
        start_idx, end_idx = i, i + ray_bs

        t_vals, xyz = sample_points(
            rays_o[start_idx:end_idx],
            rays_d[start_idx:end_idx],
            bound[0],
            bound[1],
            num_points_per_ray,
            perturb=True,
        )
        rgbs, sigmas = model(xyz)
        colors = volume_render(rgbs, sigmas, t_vals, white_bkgd=white_bkgd)  # [ray_bs, 3]
        synths.append(colors)

    synths = torch.cat(synths, dim=0)  # [num_total_rays, 3]
    synths = rearrange(synths, "(v h w) c -> v h w c", v=num_views, h=H, w=W)

    return synths


@torch.no_grad()
def test_model(
    model,
    imgs,
    poses,
    hwf,
    bound,
    num_points_per_ray,
    ray_bs,
    white_bkgd,
):
    training = model.training
    model.eval()
    metric = MetricLogger()

    synths = infer_model(
        model,
        poses,
        hwf,
        bound,
        num_points_per_ray,
        ray_bs,
        white_bkgd,
    )
    imgs_ = rearrange(imgs, "v h w c -> v c h w").clamp(0, 1)
    synths_ = rearrange(synths, "v h w c -> v c h w").clamp(0, 1)

    loss = torch.nn.functional.mse_loss(synths_, imgs_)
    psnr = psnr_fn(synths_, imgs_)
    ssim = ssim_fn(synths_, imgs_)
    lpips = lpips_fn(synths_, imgs_)

    metric.update(loss=loss.item(), psnr=psnr.item(), ssim=ssim.item(), lpips=lpips.item())
    model.train(training)
    return metric, synths



@torch.no_grad()
def img_infer_model(
    ml,
    q_set,
    ray_bs,
):
    # s_set: [v, 3, h, w]
    # q_set: [v, 3, h, w]
    model = ml.model
    synths = []

    total_coords = ml.sample_coords(q_set, ray_bs=-1)  # [v h w in]
    v, h, w, in_size = total_coords.shape
    total_coords = rearrange(total_coords, "v h w in -> (v h w) in")

    for i in range(0, len(total_coords), ray_bs):
        _coords = total_coords[i : i + ray_bs]  # [ray_bs, in]
        _rgbs = model(_coords)
        synths.append(_rgbs)

    synths = torch.cat(synths, dim=0)  # [(v h w) out]
    synths = rearrange(synths, "(v h w) c -> v h w c", v=v, h=h, w=w)

    return synths


@torch.no_grad()
def img_test_model(
    ml,
    s_set,
    q_set,
    ray_bs,
):
    model = ml.model
    training = model.training
    model.eval()
    metric = MetricLogger()

    synths = img_infer_model(ml, q_set, ray_bs)
    # s_set: [v h w 3]
    # q_set: [v h w 3]
    ###

    _synths = rearrange(synths, "v h w c -> v c h w").clamp(0, 1)
    _q_set = rearrange(q_set, "v h w c -> v c h w").clamp(0, 1)

    loss = torch.nn.functional.mse_loss(_synths, _q_set)
    psnr = psnr_fn(_synths, _q_set)
    ssim = ssim_fn(_synths, _q_set)
    lpips = lpips_fn(_synths, _q_set)

    metric.update(loss=loss.item(), psnr=psnr.item(), ssim=ssim.item(), lpips=lpips.item())
    model.train(training)
    return metric, synths  # [v h w 3]
