# %%
import json
from pathlib import Path

import btrack
import napari
import numpy as np
import polars as pl
from btrack.utils import tracks_to_napari
from tracking_io import load_features, load_image, load_labels

# %% specify paths
# fn_labels = Path(r"M:\marvwy\VisiScope\20231026H1A488_compressed\20231026H1A1_s1_segmentation.zarr")
# fn_image = Path(r"E:\sshami\Visiscope\20231026H1A488_compressed\20231026H1A1_s1.zarr")
fn_features = Path(r"M:\marvwy\VisiScope\20231026H1A488_compressed\20231026H1A1_s1.parquet")

# %% load the first 10 images/labels and features
# lbls = load_labels(fn_labels, n_max=10)
# img = load_image(fn_image, level=2, scale=(1.0, 2.6, 2.6), n_max=10)
df = load_features(fn_features)
# %%

viewer = napari.Viewer()
# viewer.add_labels(np.asarray(lbls), scale=(1, 1.0, 0.65, 0.65))
# viewer.add_image(np.asarray(img), scale=(1, 1.0, 2.6, 2.6))
viewer.add_points(df.select(['t', 'z', 'y', 'x']))
# imshow_spatial_image(lbls, viewer)
# imshow_spatial_image(img, viewer)


# %% Load tracks
import json
from pathlib import Path

from btrack.io import HDF5FileHandler
from zfish.tracking.tracking import Parameters

files = Path(
    r"C:\Users\hessm\Documents\Programming\Python\zfish\zfish\tracking\local_data\tracking_results\run_2"
)

trackss = []
params = []
for fn in sorted(files.glob("*.h5"), key=lambda x: int(x.stem.split("_")[-2])):
    print(fn)
    fn_params = fn.parent / f"{'_'.join(fn.stem.split('_')[:-1] + ['params'])}.json"

    with HDF5FileHandler(fn, "r") as f:
        tracks = f.tracks
        trackss.append(tracks)
    with open(fn_params) as fh:
        params_dict = json.load(fh)
        params.append(
                {
                    k: v
                    for k, v in params_dict.items()
                    if k not in ["tracks_out", "params_out"]
                }
        )

names = ['; '.join(f"{k}={v}" for k, v in param.items() if k in param['_repr_params']) for param in params]

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
viewer.add_points(df.select(["t", "z", "y", "x"]).to_numpy(), size=3)
for name, tracks in zip(names[:], trackss[:]):
    # for track in trackss:
    print(name)
    data, properties, graph = tracks_to_napari(tracks)
    viewer.add_tracks(
        data,
        properties=properties,
        graph=graph,
        name=name,
        visible=False,
        colormap=next(cycle),
        color_by="Roundness",
        tail_length=4,
    )     

# # %%
# pp, vv = split_events_from_graph(graph, tracks)
# # %%
# viewer = napari.Viewer()
# viewer.add_points(df.select(["t", "z", "y", "x"]).to_numpy(), size=3)
# viewer.add_points(pp, size=7, face_color='red')
# viewer.add_vectors(vv, edge_width=3)
# names = [
#     f"d_th={params.dist_thresh}; t_th={params.time_thresh}; u_meth={params.update_method}"
#     for params in configs
# ]

# # %% Visualize individual tracks
# roots = [track.root for track in tracks if track.root != track.ID]

# root_ids = pl.Series("root_id", roots).value_counts()["root_id"].to_list()

# root_trees = get_tree_by_ids(root_ids, tracks)

# viewer = napari.Viewer()
# viewer.add_points(df.select(["t", "z", "y", "x"]).to_numpy(), size=3)
# viewer.add_tracks(**tree_to_napari(root_trees))

# # %%
# track = tracks[0]
# dir(track)
# %%


