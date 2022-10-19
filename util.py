import torch
import torch.autograd as autograd


def tensor2var(x, device, grad=False):
    """
    put tensor to gpu, and set grad to false
    Args:
        x (tensor): input tensor
        grad (bool, optional):  Defaults to False.
    Returns:
        tensor: tensor in gpu and set grad to false
    """
    x = x.to(device)
    x.requires_grad_(grad)
    return x


def compute_gradient_penalty(D, real_images, fake_images, device):
    """
    compute the gradient penalty where from the wgan-gp
    Args:
        D ([Moudle]): Discriminator
        real_images ([tensor]): real images from the dataset
        fake_images ([tensor]): fake images from teh G(z)
    Returns:
        [tensor]: computed the gradient penalty
    """
    # compute gradient penalty
    alpha = torch.rand(real_images.size(0), 1, 1, 1).to(device).expand_as(real_images)
    # (*, 1, 64, 64)
    interpolated = (
        alpha * real_images.data + ((1 - alpha) * fake_images.data)
    ).requires_grad_(True)
    # (*,)
    out = D(interpolated)
    # get gradient w,r,t. interpolates
    grad = autograd.grad(
        outputs=out,
        inputs=interpolated,
        grad_outputs=torch.ones(out.size()).to(device),
        retain_graph=True,
        create_graph=True,
        only_inputs=True,
    )[0]

    # grad_flat = grad.view(grad.size(0), -1)
    # grad_l2norm = torch.sqrt(torch.sum(grad_flat ** 2, dim=1))

    # grad_l2norm = torch.linalg.vector_norm(grad, ord=2, dim=[1,2,3]) # start version 1.10~
    grad_l2norm = grad.norm(2, dim=[1, 2, 3])

    gradient_penalty = torch.mean((grad_l2norm - 1) ** 2)

    return gradient_penalty
