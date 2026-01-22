import torch



def apply_ctf(
    image: torch.Tensor,
    defocus,        # µm
    b_factor,
    amp,
    pixel_size,     # Å
    voltage_kv,     # kV
) -> torch.Tensor:

    num_batch, num_pixels, _ = image.shape

    freq_pix_1d = torch.fft.fftfreq(
        num_pixels, d=pixel_size, device=image.device
    )
    x, y = torch.meshgrid(freq_pix_1d, freq_pix_1d, indexing="ij")

    freq2_2d = (x**2 + y**2).expand(num_batch, -1, -1)

    # Electron wavelength (Å)
    V = voltage_kv * 1e3
    m_ec2 = 511e3
    lambda_e = 12.3986 / torch.sqrt(V * (1.0 + V / (2.0 * m_ec2)))

    # Phase prefactor: 2π λ Δf
    phase = 2.0 * torch.pi * lambda_e * defocus * 1e4

    env = torch.exp(-0.5 * b_factor * freq2_2d)

    chi = 0.5 * phase * freq2_2d

    ctf = (
        -amp * torch.cos(chi)
        - torch.sqrt(1 - amp**2) * torch.sin(chi)
    )

    ctf = ctf * env / amp

    image_ctf = torch.fft.ifft2(torch.fft.fft2(image) * ctf).real
    return image_ctf

