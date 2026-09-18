"""Synthetic layer-wise volume generation and PyTorch datasets."""

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class Config:
    n_builds: int = 120
    depth: int = 64
    height: int = 128
    width: int = 128
    defective_build_probability: float = 0.65
    background_mean: float = 0.10
    material_mean: float = 0.65
    noise_std: float = 0.04
    seed: int = 42


def generate_build(config, defective=True, rng=None):
    if rng is None:
        rng = np.random.default_rng()

    depth, height, width = config.depth, config.height, config.width
    volume = rng.normal(
        config.background_mean, config.noise_std, (depth, height, width)
    ).astype(np.float32)
    mask = np.zeros((depth, height, width), dtype=np.uint8)

    yy, xx = np.ogrid[:height, :width]
    cy, cx = height // 2, width // 2
    radius = min(height, width) * 0.38
    component = (yy - cy) ** 2 + (xx - cx) ** 2 <= radius ** 2

    for z in range(depth):
        layer_shift = rng.normal(0, 0.025)
        values = rng.normal(
            config.material_mean + layer_shift,
            config.noise_std,
            component.sum(),
        )
        volume[z][component] = values

    if defective:
        zz, yy3, xx3 = np.ogrid[:depth, :height, :width]
        for _ in range(int(rng.integers(1, 5))):
            cz = int(rng.integers(8, depth - 8))
            dy = int(radius * 0.55)
            cy_d = int(rng.integers(cy - dy, cy + dy))
            cx_d = int(rng.integers(cx - dy, cx + dy))
            rz = int(rng.integers(2, 7))
            ry = int(rng.integers(3, 10))
            rx = int(rng.integers(3, 10))

            ellipsoid = (
                ((zz - cz) / rz) ** 2
                + ((yy3 - cy_d) / ry) ** 2
                + ((xx3 - cx_d) / rx) ** 2
                <= 1
            )
            defect = ellipsoid & component[None, :, :]
            mask[defect] = 1
            intensity = rng.uniform(0.20, 0.45)
            volume[defect] = rng.normal(intensity, config.noise_std, defect.sum())

    return np.clip(volume, 0.0, 1.0), mask


class SliceDataset(Dataset):
    def __init__(self, volumes, masks, build_ids):
        self.volumes = volumes
        self.masks = masks
        self.samples = [
            (int(build_id), z)
            for build_id in build_ids
            for z in range(volumes.shape[1])
        ]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        build_id, z = self.samples[index]
        image = torch.from_numpy(self.volumes[build_id, z].copy()).float().unsqueeze(0)
        label = torch.tensor(float(self.masks[build_id, z].any()), dtype=torch.float32)
        return image, label, build_id, z
