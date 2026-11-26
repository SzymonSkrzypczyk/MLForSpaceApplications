import numpy as np
from torch import nn
from torch.nn import functional as F


def bicubic(low_res):
    return F.interpolate(low_res, scale_factor=2, mode="bicubic", align_corners=False)


def psnr(estimate, target):
    mse = (estimate - target).square().mean().item()
    return -10 * np.log10(mse)


def display_rgb(batch, lo, hi):
    pixels = batch[0].permute(1, 2, 0).numpy()
    return np.clip((pixels - lo) / (hi - lo), 0, 1)


class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
        )
        self.decoder = nn.Conv2d(16, 3, 3, padding=1)
        nn.init.zeros_(self.decoder.weight)
        nn.init.zeros_(self.decoder.bias)

    def forward(self, low_res):
        features = self.encoder(low_res)
        features = F.interpolate(features, scale_factor=2, mode="bilinear", align_corners=False)
        return bicubic(low_res) + self.decoder(features)
