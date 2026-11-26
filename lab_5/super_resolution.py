import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def bicubic(low_res):
    return F.interpolate(low_res, scale_factor=2, mode="bicubic", align_corners=False)


class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(nn.Conv2d(3, 24, 3, padding=1), nn.ReLU())
        self.encoder = nn.Sequential(
            nn.Conv2d(24, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(56, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 24, 3, padding=1), nn.ReLU(),
            nn.Conv2d(24, 12, 3, padding=1), nn.PixelShuffle(2),
        )
        nn.init.zeros_(self.decoder[-2].weight)
        nn.init.zeros_(self.decoder[-2].bias)

    def forward(self, low_res):
        features = self.features(low_res)
        latent = self.encoder(features)
        decoded = F.interpolate(latent, size=features.shape[-2:], mode="nearest")
        correction = self.decoder(torch.cat((features, decoded), dim=1))
        return bicubic(low_res) + correction


def psnr(estimate, target):
    mse = (estimate - target).square().mean().item()
    return -10 * np.log10(mse)


def display_rgb(batch, lo, hi):
    pixels = batch[0].permute(1, 2, 0).numpy()
    return np.clip((pixels - lo) / (hi - lo), 0, 1)
