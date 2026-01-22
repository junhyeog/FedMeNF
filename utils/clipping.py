import torch


# https://discuss.pytorch.org/t/how-to-clip-grad-norm-grads-from-torch-autograd-grad/137816
def clip_norm_coef(grads, max_norm: float, detach: bool, norm_type: float = 2.0):
    """This code looks similar to torch.nn.utils.clip_grad_norm_ and clip_norm_,
    but it is very different because it does not detach grads(important to MAML algo).
    return A scalar coefficient that normalizes the norm of gradients to the max_norm
    """
    max_norm = float(max_norm)
    norm_type = float(norm_type)

    device = grads[0].device
    total_norm = torch.norm(
        torch.stack([torch.norm(g.detach() if detach else g, norm_type).to(device) for g in grads]),
        norm_type,
    )
    clip_coef = max_norm / (total_norm + 1e-6)

    clip_coef_clamped = torch.clamp(clip_coef, max=1.0)

    return clip_coef_clamped.to(device)


def clip_norm_(grads, max_norm: float, norm_type: float = 2.0):
    """This code is based on torch.nn.utils.clip_grad_norm_(inplace function that does gradient clipping to max_norm).
    the input of torch.nn.utils.clip_grad_norm_ is parameters
    but the input of clip_norm_ is list of gradients
    """
    max_norm = float(max_norm)
    norm_type = float(norm_type)

    device = grads[0].device
    total_norm = torch.norm(
        torch.stack([torch.norm(g.detach(), norm_type).to(device) for g in grads if g is not None]),
        norm_type,
    )
    clip_coef = max_norm / (total_norm + 1e-6)

    clip_coef_clamped = torch.clamp(clip_coef, max=1.0)
    for g in grads:
        if g is not None:
            g.detach().mul_(clip_coef_clamped.to(g.device))

    return total_norm
