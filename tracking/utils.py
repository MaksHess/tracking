from copy import deepcopy
from functools import reduce
from operator import add
from typing import Any

import numpy as np
import polars as pl
from btrack.utils import tracks_to_napari


def get_tree_by_ids(ids, tracks):
    trees = [get_tree_by_id(id_, tracks) for id_ in ids]
    return reduce(add, trees, [])


def get_tree_by_id(id_, tracks):
    root = get_track_by_id(id_, tracks)
    return extract_tree_recursive(root, tracks)


def get_track_by_id(id_, tracks):
    for track in tracks:
        if track["ID"] == id_:
            return track


def extract_tree_recursive(node, tracks, aggregator=None):
    if aggregator is None:
        aggregator = []
        aggregator.append(node)

    for child_id in node.children:
        print(child_id)
        child = get_track_by_id(child_id, tracks)
        aggregator.append(child)
        extract_tree_recursive(child, tracks, aggregator=aggregator)
    return aggregator


def tree_to_napari(tree):
    return dict(zip(["data", "properties", "graph"], tracks_to_napari(tree)))


def napari_tree_to_bbx(tree_napari, xyz_buffer=20):
    _, tmin, zmin, ymin, xmin = tree_napari["data"].min(axis=0)
    _, tmax, zmax, ymax, xmax = tree_napari["data"].max(axis=0)
    return {
        "t": slice(tmin, tmax),
        "z": slice(zmin - xyz_buffer, zmax + xyz_buffer),
        "y": slice(ymin - xyz_buffer, ymax + xyz_buffer),
        "x": slice(xmin - xyz_buffer, xmax + xyz_buffer),
    }


def napari_tree_to_point_selector(tree_napari, xyz_buffer=20):
    _, tmin, zmin, ymin, xmin = tree_napari["data"].min(axis=0)
    _, tmax, zmax, ymax, xmax = tree_napari["data"].max(axis=0)
    return (
        pl.col("t").is_between(tmin, tmax)
        & pl.col("z").is_between(zmin - xyz_buffer, zmax + xyz_buffer)
        & pl.col("y").is_between(ymin - xyz_buffer, ymax + xyz_buffer)
        & pl.col("x").is_between(xmin - xyz_buffer, xmax + xyz_buffer)
    )

def split_events_from_graph(graph, tracks):
    split_events = []
    split_points = []
    for k, vs in graph.items():
        for v in vs:
            tt = get_track_by_id(k, tracks)
            tf = get_track_by_id(v, tracks)
            p_start = np.array([tf.t[-1], tf.z[-1], tf.y[-1], tf.x[-1]])
            p_end = np.array([tt.t[0], tt.z[0], tt.y[0], tt.x[0]])
            vec = np.array([p_start, p_end-p_start])
            split_events.append(vec)
            split_points.append(p_start)
    return split_points, split_events


def remove_tracks(tracks_napari: dict[str, Any], remove: tuple[int]):
    data, features = remove_nodes_from_data_and_features(
        tracks_napari["data"], tracks_napari["properties"], remove
    )
    tracks_out = {
        "data": data,
        "properties": features,
        "graph": remove_nodes_from_graph(tracks_napari["graph"], remove),
    }
    return tracks_out


def remove_nodes_from_data_and_features(data, features, drop=tuple()):
    selector = np.where(~np.isin(data[:, 0], drop))[0]
    data_out = data[selector, :]
    features_out = {k: v[selector] for k, v in features.items()}
    return data_out, features_out


def remove_nodes_from_graph(graph, drop=tuple()):
    graph = {k: v for k, v in deepcopy(graph).items() if k not in drop}
    for lst in graph.values():
        for d in drop:
            if d in lst:
                lst.remove(d)
    return graph


# # %%
# from zfish.visualize.imshow import imshow_spatial_image

# root = get_track_by_id(159, tracks)
# tree = extract_tree_recursive(root, tracks)
# tree_napari = dict(zip(["data", "properties", "graph"], tracks_to_napari(tree)))


# viewer = napari.Viewer()
# imshow_spatial_image(image.sel(**napari_tree_to_bbx(tree_napari)), viewer)
# viewer.add_points(
#     df.select(["t", "z", "y", "x"]).filter(napari_tree_to_point_selector(tree_napari)),
#     size=3,
# )
# viewer.add_tracks(**tree_napari)
