import numpy as np
import rasterio


def load_water_scene(path):
    with rasterio.open(path) as source:
        cube = source.read((1, 2, 3, 4)).transpose(1, 2, 0)
        valid = (source.read(5) == 6) & (cube > 0).all(axis=-1)
    return {"cube": cube, "valid": valid}


def patch_features(scenes, areas, patch_size):
    features, labels = [], []
    for label, boxes in areas.items():
        scene = scenes[label]
        for r0, r1, c0, c1 in boxes:
            for row in range(r0, r1, patch_size):
                for col in range(c0, c1, patch_size):
                    patch = np.s_[row:row + patch_size, col:col + patch_size]
                    valid = scene["valid"][patch]
                    if valid.mean() < 0.95:
                        continue
                    pixels = scene["cube"][patch][valid]
                    features.append(np.r_[pixels.mean(axis=0), pixels.std(axis=0)])
                    labels.append(label)
    return np.asarray(features), np.asarray(labels)
