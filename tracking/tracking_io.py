import json
from pathlib import Path

import numpy as np
import polars as pl
import zarr
from btrack.io import HDF5FileHandler

from tracking.conversions import to_si

# %% Image, label & feature loaders

def load_params(fn_params):
    with open(fn_params) as fh:
        params_dict = json.load(fh)
        params = {
            k: v
            for k, v in params_dict.items()
            if k not in ["tracks_out", "params_out"]
        }
    return params

def load_tracks(fn, parameters=False):
    fn = Path(fn)
    with HDF5FileHandler(fn, "r") as f:
        tracks = f.tracks
    
    if parameters:
        fn_params = fn.parent / f"{'_'.join(fn.stem.split('_')[:-1] + ['params'])}.json"
        params = load_params(fn_params)
        return tracks, params
    return tracks
            
def params_to_name(params, override_repr: tuple[str, ...] | None = None):
    repr_params = params["_repr_params"] if override_repr is None else override_repr
    return "; ".join(f"{k}={v}" for k, v in params.items() if k in repr_params)
    
def paramss_to_names(paramss):
    return [
        "; ".join(f"{k}={v}" for k, v in params.items() if k in params["_repr_params"])
        for params in paramss
    ]

def load_labels(fn, scale=(1, 1.0, 0.65, 0.65), dims=("t", "z", "y", "x"), n_max=None):
    zarray = zarr.open(fn)
    n_images = len(list(zarray))
    if n_max is not None:
        n_images = n_max
    shp = zarray[0].shape

    lbls = np.empty((n_images, *shp), dtype=np.uint16)
    for i in range(n_images):
        print(f"loading labelimage t={i}")
        lbls[i, :] = zarray[i][...]
    return to_si(lbls, dims=dims, scale=scale)


def label_generator(
    fn, scale=(1.0, 0.65, 0.65), dims=("z", "y", "x"), l_coords=("nuclei",)
):
    zarray = zarr.open(fn)
    n_images = len(list(zarray))

    for i in range(n_images):
        print(f"loading labelimage t={i}")
        yield to_si(zarray[i][...], dims=dims, scale=scale).expand_dims(
            {"t": [i], "l": list(l_coords)}
        ).squeeze()


def load_image(
    fn,
    level=1,
    scale=(1.0, 0.65, 0.65),
    dims=("t", "c", "z", "y", "x"),
    c_coords=("H1A",),
    n_max=None,
):
    arr_img = zarr.open(fn)
    if n_max is not None:
        img = arr_img[str(level)][:n_max, ...]
        
    else:
        img = arr_img[str(level)][...]
    if img.ndim != len(dims):
        img = np.expand_dims(img, axis=1)
    return to_si(img, dims=dims, scale=scale, c_coords=c_coords).squeeze()


def image_generator(
    fn, level=0, scale=(1.0, 0.65, 0.65), dims=("c", "z", "y", "x"), c_coords=("H1A",)
):
    arr_img = zarr.open(fn)
    img = arr_img[level]
    for i in range(img.shape[0]):
        print(f"loading image t={i}")
        yield to_si(img[i, ...], scale=scale, dims=dims, c_coords=c_coords).expand_dims(
            {"t": [i]}
        ).squeeze()


def load_features(fn):
    return pl.read_parquet(fn)
