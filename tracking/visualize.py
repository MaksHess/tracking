# %%
import json
from pathlib import Path

import btrack
import napari
import numpy as np
import polars as pl

# %% Load tracks
from btrack.io import HDF5FileHandler
from btrack.utils import tracks_to_napari

from tracking.tracking_io import load_features, load_image, load_labels
from tracking.utils import tracks_to_polars

idx = 1
LOAD_IMAGES = True
LOAD_LABELS = False
MAX_LABELS = None

files = Path(
    rf"Z:\hmax\Visiscope\20240425H1A488_compressed\tracking_results\new_seg_s{idx}"
)
fn_features = Path(
    rf"Z:\hmax\Visiscope\20240425H1A488_compressed\20240425_H1A488_s{idx}.parquet"
)
fn_image = Path(rf"Z:\hmax\Visiscope\20240425H1A488_compressed\20240425_H1A488_s{idx}.zarr")
fn_labels = Path(rf"Z:\hmax\Visiscope\20240425H1A488_compressed\20240425_H1A488_s{idx}_segmentation.zarr")

if LOAD_IMAGES:
    img = load_image(fn_image, level=2, scale=(1.0, 1.3, 1.3))
    
if LOAD_LABELS:
    lbls = load_labels(fn_labels, n_max=MAX_LABELS)

df = load_features(fn_features)
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
# viewer.add_image(np.asarray(img), scale=(1, 1.0, 1.3, 1.3))
# viewer.add_labels(np.asarray(lbls), scale=(1, 1.0, 0.65, 0.65))
points_layer = viewer.add_points(df.select(["t", "z", "y", "x"]).to_numpy(), size=4, face_color='img_label', properties={'img_label': df["img_label"]})
for name, tracks in zip(names[-2:], trackss[-2:]):
    print(name)
    data, properties, graph = tracks_to_napari(tracks)
    properties['ones'] = np.ones(properties['t'].shape)
    properties['ones'][0] = 0
    viewer.add_tracks(
        data,
        properties=properties,
        graph=graph,
        name=name,
        visible=False,
        colormap='turbo',
        # color_by="Roundness",
        color_by='ones',
        tail_length=5,
    )


# %%
df_tracks = tracks_to_polars(trackss[-2]).select(["ID", "t", "z", "y", "x", "img_label"])
max_id = df_tracks.select('ID').max().item()
# %%
high_quality = [
    2,
    42,
    44,
    179,
    7,
    166,
    376,
    359,
    660,
    658,
    182,
    10,
    11,
    54,
    1310,
    1313,
    9,
    51,
    52,
    174,
    184,
    # 167,
    419,
    415,
    778,
    782,
    # 1395,
    
]

tracklet_mergers = [ # (tracklet_id, tracklet_id)
  (11, 31),
  (54, 1123),
  (778, 1138),
  (782, 1148),
]

tracklet_splits = [ # (tracklet_id, [t_new_tracklet])
    (44, [9, 17, 26]),
    (42, [9, 17, 25]),
    (179, [17, ]),
    (427, [26]),
    (7, [2, ]),
    (10, [2, 9, 19, 25]),
    (213, [17]),
    (182, [16, 24]),
    (376, [24, 31, 43, 67]),
    (660, [31, 43]),
    (54, [9, 16, 24]),
    (1310, [51]),
    (1313, [48]),
    (52, [9, ]),
    (782, [34]),
    (778, [34, 54]),
    (174, [17, 25]),
    (184, [17, 25]),
    (419, [25]),
    # (376, [31]),
]

tracklet_graph = {
    179: [44],
    891: [179],
    427: [179],
    840: [427],
    64: [10],
    213: [10],
    749: [10],
}

tracklet_graph = {}
# OLD
# tracklet_mergers = [ # (tracklet_id, tracklet_id)
#     (65, 119),
#     (1230, 1249),
#     (30, 53),
#     (26, 46),
#     (22, 36),
#     (64, 117, 133),
# ]

# tracklet_splits = [ # (tracklet_id, [t_new_tracklet])
#     (659, [51]),
#     (100, [16, 24]),
#     (513, [42]),
#     (512, [43]),
#     (64, [17]),
# ]

# tracklet_graph = {
#     65: [16],
#     181: [65],
#     1230: [659],
#     100: [26],
#     187: [64],
# }

from functools import reduce
from operator import add

select_ids = tuple(high_quality) + reduce(add, tracklet_mergers, tuple()) + tuple(e[0] for e in tracklet_splits) + tuple(reduce(add, tracklet_graph.values(), tuple())) + tuple(tracklet_graph.keys())

def select_tracklets(df_tracks, tracklet_ids) -> pl.DataFrame:
    return df_tracks.filter(pl.col("ID").is_in(tracklet_ids))

def merge_tracklets(df_tracks, tracklet_mergers) -> pl.DataFrame:
    merge_dict = {}
    for merger in tracklet_mergers:
        for child in merger[1:]:
            merge_dict[child] = merger[0]
    return df_tracks.with_columns(pl.col('ID').replace(merge_dict))


def split_tracklets(df_tracks, tracklet_splits, max_id) -> tuple[pl.DataFrame, dict[int, list[int]]]:
    df_out = df_tracks.clone()
    new_graph_entries = {}
    current_new_id = max_id + 1
    for old_id, t_splits in tracklet_splits:
        new_ids = [old_id] + list(range(current_new_id, current_new_id+len(t_splits)))
        labels = [str(e) for e in new_ids]
        df_out = df_out.with_columns(pl.when(pl.col('ID')==old_id).then(pl.col('t').cut(t_splits, left_closed=True, labels=labels).cast(pl.Int32)).otherwise(pl.col('ID')).alias('ID'))
        current_new_id = max(new_ids) + 1
        for a, b in zip(new_ids, new_ids[1:]):
            new_graph_entries[b] = [a]
    return df_out, new_graph_entries
# %%
df_tracklet_clean, new_graph = split_tracklets(merge_tracklets(select_tracklets(df_tracks, select_ids), tracklet_mergers), tracklet_splits, max_id)
properties = {'ones': np.ones(len(df_tracklet_clean))}
# %%
viewer = napari.Viewer()
viewer.add_image(np.asarray(img), scale=(1, 1.0, 1.3, 1.3))
# viewer.add_labels(np.asarray(lbls), scale=(1, 1.0, 0.65, 0.65))
viewer.add_tracks(data=df_tracklet_clean.select(['ID', 't', 'z', 'y', 'x']).to_numpy(), properties=properties, graph=new_graph)
# %%
manual_graph = {
    44: [2],
    42: [2],
    182: [3534],
    166: [3534],
    376: [166],
    359: [166],
    213: [3535],
    179: [44],
    427: [179],
    660: [359],
    658: [359],
    54: [11],
    1310: [3550],
    1313: [3550],
    51: [9],
    52: [9],
    174: [51],
    184: [51],
    419: [3553],
    415: [3553],
    778: [415],
    782: [415],
}
# %%
full_graph = {**manual_graph, **new_graph}
viewer = napari.Viewer()
viewer.add_image(np.asarray(img), scale=(1, 1.0, 1.3, 1.3))
# viewer.add_labels(np.asarray(lbls), scale=(1, 1.0, 0.65, 0.65))
viewer.add_tracks(data=df_tracklet_clean.select(['ID', 't', 'z', 'y', 'x']).to_numpy(), properties=properties, graph=full_graph)
# %%
from sklearn.neighbors import NearestNeighbors

nns = NearestNeighbors(n_neighbors=1).fit(df.select(['t', 'z', 'y', 'x']).to_numpy())

neighbor_indices = nns.kneighbors(df_tracklet_clean.select(['t', 'z', 'y', 'x']).to_numpy())

assert neighbor_indices[0].max() < 1e-4, "Nearest neighbor should be almost zero!"

df_out = df_tracklet_clean.with_columns(df[neighbor_indices[1].flatten()].drop(['t', 'z', 'y', 'x']))
# %%
import seaborn as sns

sns.scatterplot(x=df_out['t'], y=df_out['Roundness'])
# %%
full_graph_ = {k: v[0] for k, v in full_graph.items()}

def root_mapping(graph):
    root_nodes = set(full_graph_.values()) - set(full_graph_.keys())
    root_graph = {}
    generation_graph = {n: 0 for n in root_nodes}
    
    
    for k in graph:
        current_node = k
        generation_counter = 1
        while True:
            root_candidate = graph[current_node]
            if root_candidate in root_nodes:
                root_graph[k] = root_candidate
                generation_graph[k] = generation_counter
                break
            current_node = root_candidate
            generation_counter += 1
    return root_graph, generation_graph


root_graph, generation_graph = root_mapping(full_graph_)

df_ts = pl.read_parquet(fn_features.parent / f"{fn_features.stem}_timestamps.parquet")
df_out = df_out.with_columns(pl.col('ID').replace(full_graph_).alias('parent'), pl.col('ID').replace(root_graph).alias('root'), pl.col('ID').replace(generation_graph).alias('generation')).join(df_ts.with_columns(pl.col('t').cast(pl.Int32)), on='t', how='left')
# %%
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(10, 3), dpi=300)
sns.scatterplot(data=df_out.filter(pl.col('root') == 9).with_columns(pl.col('PhysicalSize')), x='delta_min', y='PhysicalSize', hue='ID', palette='Set1', legend=False)
# sns.scatterplot(data=df_out.filter(pl.col('root') == 9), x='delta_min', y='PhysicalSize', hue='generation', palette='Set1')
# sns.scatterplot(data=df_out.filter(pl.col('root') == 10), x='delta_min', y='PhysicalSize', hue='generation', palette='viridis')
# sns.scatterplot(data=df_out.filter(pl.col('root') == 7), x='t', y='PhysicalSize', hue='ID', palette='Set1', legend=False)
# sns.scatterplot(data=df_out, x='delta_min', y='PhysicalSize', hue='generation', palette='Set1')
# %%
sns.scatterplot(data=df_out.filter(pl.col('root') == 10), x='t', y='PhysicalSize', hue='generation', palette='viridis')
# %%
sns.scatterplot(df_out.with_columns((pl.col('delta_min') - pl.col('delta_min').min().over('ID')).alias('track_time'), pl.col('generation').cast(pl.String)), x='track_time', y='PhysicalSize', hue='generation')
# %%
import hvplot.polars

df_out.with_columns((pl.col('delta_min') - pl.col('delta_min').min().over('ID')).alias('track_time')).hvplot.scatter(x='track_time', y='PhysicalSize', color='ID', by='generation', subplots=True).cols(1)
# %%
df_out.write_parquet('tracks_clean.parquet')