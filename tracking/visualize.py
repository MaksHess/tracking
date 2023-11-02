# %%
import json
from pathlib import Path

import btrack
import napari
import numpy as np
import polars as pl
from btrack.utils import tracks_to_napari
from tracking.tracking_io import load_features, load_image, load_labels

# %% specify paths
# fn_labels = Path(r"M:\marvwy\VisiScope\20231026H1A488_compressed\20231026H1A1_s1_segmentation.zarr")
fn_image = Path(r"E:\sshami\Visiscope\20231026H1A488_compressed\20231026H1A1_s2.zarr")
fn_features = Path(
    r"M:\marvwy\VisiScope\20231026H1A488_compressed\20231026H1A1_s2_clean.parquet"
)
# fn_features = Path(r"C:\Users\hessm\Documents\zfish_local\live_data\imaris_sub.parquet")
# %%
import zarr

out_fn = r"C:\Users\hessm\Documents\zfish_local\live_data\20231026H1A1_s2.zarr"

arr = zarr.open(out_fn)
# %%
img_z = zarr.array(np.asarray(img), chunks=(1, 251, 256, 256))

arr.create_dataset('2', data=   img_z)
# %% load the first 10 images/labels and features
# lbls = load_labels(fn_labels, n_max=10)
img = load_image(fn_image, level=2, scale=(1.0, 2.6, 2.6))
df = load_features(fn_features)
# %%
# viewer = napari.Viewer()
# viewer.add_labels(np.asarray(lbls), scale=(1, 1.0, 0.65, 0.65))
# viewer.add_image(np.asarray(img), scale=(1, 1.0, 2.6, 2.6))
# viewer.add_points(df.select(["t", "z", "y", "x"]), size=3)
# imshow_spatial_image(lbls, viewer)
# imshow_spatial_image(img, viewer)


# %%

# %% Load tracks
import json
from pathlib import Path

from btrack.io import HDF5FileHandler

files = Path(
    r"C:\Users\hessm\Documents\zfish_local\live_data\tracking_results\config_file_cell_s2"
)

trackss = []
params = []
for fn in list(sorted(files.glob("*.h5"), key=lambda x: int(x.stem.split("_")[-2]))):
    print(fn)
    fn_params = fn.parent / f"{'_'.join(fn.stem.split('_')[:-1] + ['params'])}.json"

    with open(fn_params) as fh:
        params_dict = json.load(fh)

        params.append(
            {
                k: v
                for k, v in params_dict.items()
                if k not in ["tracks_out", "params_out"]
            }
        )
    with HDF5FileHandler(fn, "r") as f:
        tracks = f.tracks
        trackss.append(tracks)


names = [
    "; ".join(f"{k}={v}" for k, v in param.items() if k in param["_repr_params"])
    for param in params
]

# %% Visualize all tracks
import napari


def color_cycle():
    while True:
        yield "green"
        yield "blue"
        yield "red"
        # yield "red"
        # yield "blue"
        # yield "green"


cycle = color_cycle()

viewer = napari.Viewer()
# viewer.add_image(np.asarray(img), scale=(1, 1.0, 2.6, 2.6))
viewer.add_points(df.select(["t", "z", "y", "x"]).to_numpy(), size=3, face_color='magenta')
for name, tracks in zip(names[:], trackss[:]):

    print(name)
    data, properties, graph = tracks_to_napari(tracks)
    viewer.add_tracks(
        data,
        properties=properties,
        graph=graph,
        name=name,
        visible=False,
        colormap='turbo',
        # color_by="Roundness",
        tail_length=4,
    )
