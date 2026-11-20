"""Build the patch feature table from public Sentinel-2 and WorldCover rasters."""

import csv
import json
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
from rasterio.warp import transform, transform_bounds
from rasterio.windows import Window

PATCH_SIZE = 200
GRID_SIZE = 12
CLASS_FRACTION = 0.55
CLEAR_FRACTION = 0.95
CLEAR_CLASSES = (4, 5, 6)
EARTH_SEARCH = "https://earth-search.aws.element84.com/v1"
WORLDCOVER = "https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map"
COLLECTION = "sentinel-2-l2a"
REGIONS = {
    # Longitude, latitude, split
    "Krakow": (19.945, 50.064, "train"),
    "Katowice": (19.023, 50.259, "train"),
    "Warsaw": (20.930, 52.230, "train"),
    "Wroclaw": (17.036, 51.127, "train"),
    "Lodz": (19.457, 51.769, "validation"),
    "Lublin": (22.630, 51.246, "validation"),
    "Poznan": (16.925, 52.406, "test"),
    "Gdansk": (18.621, 54.360, "test"),
    "Szczecin": (14.553, 53.429, "test"),
}
LAB_DIR = Path(__file__).resolve().parent


def read_json(url):
    with urlopen(url, timeout=45) as response:
        return json.load(response)


def crop_window(dataset, lon, lat):
    x, y = transform("EPSG:4326", dataset.crs, [lon], [lat])
    row, col = dataset.index(x[0], y[0])
    size = PATCH_SIZE * GRID_SIZE
    return Window(col - size // 2, row - size // 2, size, size)


def read_on_grid(url, crs, affine, size):
    with rasterio.open(url) as source:
        with WarpedVRT(
            source, crs=crs, transform=affine, width=size, height=size,
            resampling=Resampling.nearest, nodata=0,
        ) as aligned:
            return aligned.read(1)


def worldcover_tiles(crs, bounds):
    left, bottom, right, top = transform_bounds(crs, "EPSG:4326", *bounds)
    for latitude in range(3 * int(bottom // 3), 3 * int(top // 3) + 3, 3):
        for longitude in range(3 * int(left // 3), 3 * int(right // 3) + 3, 3):
            # The lab regions are all north of the equator and east of Greenwich
            yield f"ESA_WorldCover_10m_2021_v200_N{latitude:02d}E{longitude:03d}_Map.tif"


def patch_features(red, nir, scl):
    clear = np.isin(scl, CLEAR_CLASSES) & (red > 0) & (nir > 0)
    clear_fraction = clear.mean()
    if clear_fraction < CLEAR_FRACTION:
        return None
    return red[clear].mean(), nir[clear].mean(), clear_fraction


def summarize_patch(cover, red, nir, scl):
    """Return the label and band statistics for one eligible patch."""
    forest = (cover == 10).mean()
    urban = (cover == 50).mean()
    if max(forest, urban) < CLASS_FRACTION:
        return None

    features = patch_features(red, nir, scl)
    if features is None:
        return None

    mean_red, mean_nir, clear = features
    return {
        "label": "forest" if forest > urban else "urban",
        "forest_fraction": forest,
        "urban_fraction": urban,
        "clear_fraction": clear,
        "mean_red": mean_red,
        "mean_nir": mean_nir,
    }


def collect_patches(region, longitude, latitude, split, item):
    assets = item["assets"]
    with rasterio.open(assets["red"]["href"]) as source:
        window = crop_window(source, longitude, latitude)
        if source.res != (10, 10) or window.intersection(
            Window(0, 0, source.width, source.height)
        ) != window:
            raise ValueError(f"{item['id']} does not cover the 10 m crop for {region}")
        crs = source.crs
        affine = source.window_transform(window)
        bounds = source.window_bounds(window)

    size = int(window.width)
    cover = np.zeros((size, size), dtype=np.uint8)
    tiles = tuple(worldcover_tiles(crs, bounds))
    for tile in tiles:
        values = read_on_grid(f"{WORLDCOVER}/{tile}", crs, affine, size)
        cover[values != 0] = values[values != 0]

    scl = read_on_grid(assets["scl"]["href"], crs, affine, size)
    red = read_on_grid(assets["red"]["href"], crs, affine, size)
    nir = read_on_grid(assets["nir"]["href"], crs, affine, size)
    rows = []
    for row in range(0, size, PATCH_SIZE):
        for col in range(0, size, PATCH_SIZE):
            patch = np.s_[row:row + PATCH_SIZE, col:col + PATCH_SIZE]
            summary = summarize_patch(cover[patch], red[patch], nir[patch], scl[patch])
            if summary is None:
                continue
            rows.append({
                "region": region, "split": split, "scene_id": item["id"],
                "date": item["properties"]["datetime"][:10], "epsg": crs.to_epsg(),
                "row": int(window.row_off) + row, "col": int(window.col_off) + col,
                **summary,
            })

    return rows


def main():
    scene_file = LAB_DIR / "scenes.json"
    scene_ids = json.loads(scene_file.read_text(encoding="utf-8"))
    rows = []
    with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", GDAL_HTTP_TIMEOUT=45):
        for region, (lon, lat, split) in REGIONS.items():
            item = read_json(f"{EARTH_SEARCH}/collections/{COLLECTION}/items/{scene_ids[region]}")
            patches = collect_patches(region, lon, lat, split, item)
            print(region, item["id"], len(patches), "patches", flush=True)
            rows.extend(patches)

    with (LAB_DIR / "patches.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
