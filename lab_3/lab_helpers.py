import numpy as np
from PIL import Image


def load_scene(area, image_dir, bands):
    arrays = []
    for band in bands:
        path = image_dir / "sentinel_2" / area / f"{band}.tif"
        with Image.open(path) as image:
            arrays.append(np.array(image, dtype=float))
    cube = np.stack(arrays, axis=-1)
    return {"cube": cube, "valid": np.all(cube != 0, axis=-1)}


def rgb_display(scene):
    rgb = scene["cube"][..., [2, 1, 0]]  # red, green, blue
    lo, hi = np.percentile(rgb[scene["valid"]], [2, 98], axis=0)
    rgb = np.clip((rgb - lo) / np.maximum(hi - lo, 1), 0, 1)
    rgb[~scene["valid"]] = 0.5
    return rgb


def slices(box):
    r0, r1, c0, c1 = box
    return np.s_[r0:r1, c0:c1]


def colorize(labels):
    rgb = np.full((*labels.shape, 3), 0.6)
    rgb[labels == 0] = (0, 0, 1)
    rgb[labels == 1] = (1, 0, 0)
    return rgb

