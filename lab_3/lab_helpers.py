import numpy as np
from PIL import Image


def load_scene(area, image_dir, bands):
    arrays, valid, reference = [], None, None
    for band in bands:
        path = image_dir / "sentinel_2" / area / f"{band}.tif"
        with Image.open(path) as image:
            # Preserve the original 16-bit values; do not convert to RGB/8-bit.
            values = np.array(image, dtype=np.float64)
            tags = image.tag_v2
            grid = (values.shape,) + tuple(
                tags.get(t) for t in (33550, 33922, 34264, 34735)
            )
            nodata = tags.get(42113)  # GeoTIFF GDAL_NODATA tag
        if values.ndim != 2:
            raise ValueError(f"Expected one band in {path}")
        if reference is not None and grid != reference:
            raise ValueError(f"Bands are not on the same grid: {area}")
        reference = grid
        good = np.isfinite(values)
        if nodata is not None:
            good &= values != float(nodata)
        valid = good if valid is None else valid & good
        arrays.append(values)
    cube = np.stack(arrays, axis=-1)
    return {"cube": cube, "valid": valid}


def rgb_display(scene):
    rgb = scene["cube"][..., [2, 1, 0]]  # red, green, blue
    lo, hi = np.percentile(rgb[scene["valid"]], [2, 98], axis=0)
    rgb = np.clip((rgb - lo) / np.maximum(hi - lo, 1), 0, 1)
    rgb[~scene["valid"]] = 0.5
    return rgb


def slices(box):
    r0, r1, c0, c1 = box
    return np.s_[r0:r1, c0:c1]


def overlaps(a, b):
    return max(a[0], b[0]) < min(a[1], b[1]) and max(a[2], b[2]) < min(a[3], b[3])


def colorize(labels):
    rgb = np.full((*labels.shape, 3), 0.6)
    rgb[labels == 0] = (0, 0, 1)
    rgb[labels == 1] = (1, 0, 0)
    return rgb

