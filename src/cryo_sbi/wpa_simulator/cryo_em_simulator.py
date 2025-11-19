import json
import numpy as np
import torch

from cryo_sbi.wpa_simulator.ctf import apply_ctf
from cryo_sbi.wpa_simulator.image_generation import project_density
from cryo_sbi.wpa_simulator.noise import add_noise
from cryo_sbi.wpa_simulator.normalization import gaussian_normalize_image
from cryo_sbi.inference.priors import get_image_priors
from cryo_sbi.utils.check_config import check_image_params


def cryo_em_simulator(
    models,
    index,
    quaternion,
    sigma,
    shift,
    defocus,
    b_factor,
    amp,
    snr,
    num_pixels,
    pixel_size,
):
    """
    Simulates a bacth of cryo-electron microscopy (cryo-EM) images of a set of given coars-grained models.

    Args:
        models (torch.Tensor): A tensor of coars grained models (num_models, 3, num_beads).
        index (torch.Tensor): A tensor of indices to select the models to simulate.
        quaternion (torch.Tensor): A tensor of quaternions to rotate the models.
        sigma (float): The standard deviation of the Gaussian kernel used to project the density.
        shift (torch.Tensor): A tensor of shifts to apply to the models.
        defocus (float): The defocus value of the contrast transfer function (CTF).
        b_factor (float): The B-factor of the CTF.
        amp (float): The amplitude contrast of the CTF.
        snr (float): The signal-to-noise ratio of the simulated image.
        num_pixels (int): The number of pixels in the simulated image.
        pixel_size (float): The size of each pixel in the simulated image.

    Returns:
        torch.Tensor: A tensor of the simulated cryo-EM image.
    """
    if index.ndim == 2:
        models_selected = models[index[:, 0], index[:, 1]]
    elif index.ndim == 1:
        models_selected = models[index]
    image = project_density(
        models_selected,
        quaternion,
        sigma,
        shift,
        num_pixels,
        pixel_size,
    )
    image = apply_ctf(image, defocus, b_factor, amp, pixel_size)
    image = add_noise(image, snr)
    image = gaussian_normalize_image(image)
    return image


class CryoEmSimulator:
    def __init__(self, config_fname: str, device: str = "cpu"):
        self._device = device
        self._load_params(config_fname)
        self._load_models()
        self._priors = get_image_priors(
            self.num_models, self.num_representatives, self._config, device=device
        )
        self._num_pixels = torch.tensor(
            self._config["N_PIXELS"], dtype=torch.float32, device=device
        )
        self._pixel_size = torch.tensor(
            self._config["PIXEL_SIZE"], dtype=torch.float32, device=device
        )

    def _load_params(self, config_fname: str) -> None:
        """
        Loads the parameters from the config file into a dictionary.

        Args:
            config_fname (str): Path to the configuration file.

        Returns:
            None
        """

        config = json.load(open(config_fname))
        check_image_params(config)
        self._config = config

    def _load_models(self) -> None:
        """
        Loads the models from the model file specified in the config file.

        Returns:
            None

        """
        if self._config["MODEL_FILE"].endswith("npy"):
            models = (
                torch.from_numpy(
                    np.load(self._config["MODEL_FILE"]),
                )
                .to(self._device)
                .to(torch.float32)
            )
        elif self._config["MODEL_FILE"].endswith("pt"):
            models = (
                torch.load(self._config["MODEL_FILE"])
                .to(self._device)
                .to(torch.float32)
            )

        else:
            raise NotImplementedError(
                "Model file format not supported. Please use .npy or .pt."
            )

        self._models = models

        if self._models.ndim == 3:
            self.num_models = self._models.shape[0]
            self.num_representatives = None
        elif self._models.ndim == 4:
            self.num_models = self._models.shape[0]
            self.num_representatives = self._models.shape[1]
        else:
            raise ValueError(
                "Models are not of shape (models, 3, atoms) or (models, representatives, 3, atoms)."
            )

        assert (
            self._models.ndim == 3 or self._models.ndim == 4
        ), "Models are not of shape (models, 3, atoms) or (models, representatives, 3, atoms)."
        assert self._models.shape[-2] == 3, "Models are not of shape (..., 3, atoms)."

    def simulate(self, num_sim, indices=None, return_parameters=False, batch_size=None):
        """
        Simulate cryo-EM images using the specified models and prior distributions.

        Args:
            num_sim (int): The number of images to simulate.
            indices (torch.Tensor, optional): The indices of the images to simulate. If None, all images are simulated.
            return_parameters (bool, optional): Whether to return the sampled parameters used for simulation.
            batch_size (int, optional): The batch size to use for simulation. If None, all images are simulated in a single batch.

        Returns:
            torch.Tensor or tuple: The simulated images as a tensor of shape (num_sim, num_pixels, num_pixels),
            and optionally the sampled parameters as a tuple of tensors.
        """

        parameters = self._priors.sample((num_sim,))
        indices = parameters[0] if indices is None else indices
        if indices is not None:
            assert isinstance(
                indices, torch.Tensor
            ), "Indices are not a torch.tensor, converting to torch.tensor."
            parameters[0] = indices

        images = []
        if batch_size is None:
            batch_size = num_sim
        for i in range(0, num_sim, batch_size):
            batch_indices = indices[i : i + batch_size]
            batch_parameters = [param[i : i + batch_size] for param in parameters[1:]]
            batch_images = cryo_em_simulator(
                self._models,
                batch_indices,
                *batch_parameters,
                self._num_pixels,
                self._pixel_size,
            )
            images.append(batch_images.cpu())

        images = torch.cat(images, dim=0)

        if return_parameters:
            return images.cpu(), parameters
        else:
            return images.cpu()
