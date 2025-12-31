import torch
import json
from cryo_sbi.inference.models import build_models


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
