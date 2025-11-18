import torch
import json
from cryo_sbi.inference.models import build_models


@torch.no_grad()
def compute_latent_repr(
    estimator: torch.nn.Module,
    images: torch.Tensor,
    batch_size: int = 100,
    device: str = "cpu",
) -> torch.Tensor:
    """
    Computes the latent representation of images.

    Args:
        estimator (torch.nn.Module): Posterior model for which to compute the latent representation.
        images (torch.Tensor): The images to compute the latent representation for.
        batch_size (int, optional): The batch size to use. Defaults to 100.
        device (str, optional): The device to use. Defaults to "cpu".

    Returns:
        torch.Tensor: The latent representation of the images.
    """

    latent_space_samples = []

    if images.shape[0] > batch_size and batch_size > 0:
        images = torch.split(images, split_size_or_sections=batch_size, dim=0)
    else:
        batch_size = images.shape[0]
        images = [images]

    for image_batch in images:
        samples = estimator.embedding(image_batch.to(device, non_blocking=True)).cpu()
        latent_space_samples.append(samples.reshape(image_batch.shape[0], -1))

    return torch.cat(latent_space_samples, dim=0)


def load_classifier(
    config_file_path: str, estimator_path: str, device: str = "cpu"
) -> torch.nn.Module:
    """
    Loads a trained estimator.

    Args:
        config_file_path (str): Path to the config file used to train the estimator.
        estimator_path (str): Path to the estimator.
        device (str, optional): The device to use. Defaults to "cpu".

    Returns:
        torch.nn.Module: The loaded estimator.
    """

    train_config = json.load(open(config_file_path))
    estimator = build_models.build_classifier(train_config)
    estimator.load_state_dict(
        torch.load(estimator_path, map_location=torch.device(device))
    )
    estimator.to(device)
    estimator.eval()

    return estimator
