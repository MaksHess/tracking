# %%
import json
from collections.abc import Iterable
from itertools import pairwise
from pathlib import Path

import matplotlib.pyplot as plt
import napari
import numpy as np
import polars as pl
import polars.selectors as cs
import seaborn as sns
from matplotlib.lines import Line2D

from tracking.conversions import to_si
from tracking.tracking_io import (
    load_features,
    load_image,
    load_params,
    load_tracks,
    params_to_name,
)
from tracking.utils import (
    IDX,
    get_tracklet_by_id,
    get_tree_by_id,
    get_tree_by_ids,
    split_events_from_graph,
    tracks_to_polars,
    tree_to_napari,
)

# %%
df_features = load_features(
    r"C:\Users\hessm\Documents\zfish_local\live_data\s2_clean.parquet"
)
# %%
from zfish.features.neighborhood.neighborhoods import (
    NeighborhoodQueryObject,
    aggregation_functions,
)
from zfish.features.polars_utils import stack_column_name_to_column

# %%
nq = NeighborhoodQueryObject.from_dataframe(
    df_features,
    label_columns=("t", "img_label"),
    centroid_column=["z", "y", "x"],
    region_id_column="t",
)
# %%
df_closest_neighbors = nq.knn(
    [1, 3, 5, 10], self_loops=False, distance=True
).aggregate_weights(aggregation_functions.Max, return_label=True)
# %%
distance_to = 'KNNd:5'
fig, ax = plt.subplots(dpi=200, figsize=(20, 8))
to_plot = df_closest_neighbors.pipe(
    stack_column_name_to_column, column_name="neighborhood", index=["t", "img_label"]
).filter(pl.col('neighborhood')==distance_to)
sns.boxplot(data=to_plot.to_pandas(), x="t", y="Max")

locs, labels = plt.xticks()
plt.xticks(locs, labels, rotation=90)
ax.set_title(distance_to)

# %%
img_fn = Path(r"C:\Users\hessm\Documents\zfish_local\live_data\20231026H1A1_s2.zarr")
img = load_image(img_fn, level=2, scale=(1.0, 2.3, 2.3))
# %%
fld_tracks = Path(
    r"C:\Users\hessm\Documents\zfish_local\live_data\tracking_results\distance"
)
fld_tracks2 = Path(
    r"C:\Users\hessm\Documents\zfish_local\live_data\tracking_results\features"
)

fld_tracks3 = Path(
    r"C:\Users\hessm\Documents\zfish_local\live_data\tracking_results\distance_v2"
)

fld_tracks4 = Path(
    r"C:\Users\hessm\Documents\zfish_local\live_data\tracking_results\distance_v3"
)

fld_tracks5 = Path(
    r"C:\Users\hessm\Documents\zfish_local\live_data\tracking_results\tracklets_radius_max_lost"
)
fld_tracks6 = Path(
    r"C:\Users\hessm\Documents\zfish_local\live_data\tracking_results\tracklets_prob_not_assign"
)
fld_tracks7 = Path(
    r"C:\Users\hessm\Documents\zfish_local\live_data\tracking_results\cell_config_s2_v2_f80"
)


flds = [fld_tracks7]

# fns = [fn for fld in flds for fn in fld.glob("*h5")]
# fns = [fn for fn in fld_tracks3.glob('*.h5')]
# override_repr = {"max_search_radius": float, "max_lost": int, "prob_not_assign": float}


def load_multiple_tracks(
    flds, trackss_pl=None, trackss=None, override_repr: dict[str, type] = None
):
    if trackss_pl is None:
        trackss_pl = {}
    if trackss is None:
        trackss = {}

    fns = [fn for fld in flds for fn in fld.glob("*h5")]
    fns_params = [
        fn.parent / f"{'_'.join(fn.stem.split('_')[:-1] + ['params'])}.json"
        for fn in fns
    ]

    for fn_params, fn in zip(fns_params, fns):
        print(f"loading {fn.stem}")
        params = load_params(fn_params)
        if override_repr is None:
            override_repr = {
                k: type(v) for k, v in params.items() if k in params["_repr_params"]
            }
        else:
            override_repr = {
                **{k: v for k, v in params.items() if k in params["_repr_params"]},
                **override_repr,
            }
        name = params_to_name(params, override_repr=list(override_repr.keys()))
        print(name)
        if name in trackss.keys() and name in trackss_pl.keys():
            print("skip, already loaded...")
            continue

        tracks = load_tracks(fn)

        trackss_pl[name] = tracks_to_polars(tracks)
        trackss[name] = tracks

    return trackss_pl, trackss, override_repr


# trackss_pl, trackss, override_repr = load_multiple_tracks(flds)

trackss_pl, trackss, override_repr = load_multiple_tracks(
    flds, trackss_pl=trackss_pl, trackss=trackss, override_repr=override_repr
)
# %%
trackss


# %%
def tracklet_time_table(df):
    df_start_stop = (
        df.sort(["ID", "t"])
        .select(IDX)
        .group_by(["root", "parent", "ID", "generation"], maintain_order=True)
        .agg(pl.col("t").min().alias("t_start"), pl.col("t").max().alias("t_stop"))
    )

    df_root_stop = df_start_stop.join(
        df_start_stop.select(pl.col("ID"), pl.col("t_start").alias("t_start_root")),
        left_on="root",
        right_on="ID",
        how="left",
    )

    df_times = df_root_stop.with_columns(
        (pl.col("t_stop") - pl.col("t_start")).alias("tracklet_time"),
        (pl.col("t_stop") - pl.col("t_start_root")).alias("total_time"),
    )
    return df_times


def tree_time_table(df):
    df_tracklet_times = tracklet_time_table(df)
    return (
        df_tracklet_times.group_by("root")
        .agg(
            pl.col("t_start").min(),
            pl.col("t_stop").max(),
            pl.count().alias("tracklet_count"),
            pl.col("tracklet_time").sum(),
        )
        .sort("tracklet_time", descending=True)
    )


def sort_trees(df_times):
    draw_order_roots = (
        df_times.group_by("root")
        .agg(pl.col("generation").max(), pl.col("total_time").max())
        .sort(["generation", "total_time"], descending=True)
    )
    draw_order_total_time = (
        df_times.group_by("root", maintain_order=True)
        .agg(pl.col("tracklet_time").sum())
        .sort("tracklet_time", descending=True)
    )
    return draw_order_total_time["root"]


def sort_tracklets(df_times):
    raise NotImplementedError


def _tree_to_dict(df_tree, current_id, parent_node=None):
    if parent_node is None:
        df_parent_node = df_tree.filter(pl.col("ID") == current_id)
        parent_node = {
            "id": current_id,
            "t": df_parent_node.select(["t_start", "t_stop"]).rows()[0],
            "children": [],
        }

    for child_id in df_tree.filter(pl.col("parent") == current_id)["ID"]:
        df_child_node = df_tree.filter(pl.col("ID") == child_id)
        child_node = {
            "id": child_id,
            "t": df_child_node.select(["t_start", "t_stop"]).rows()[0],
            "children": [],
        }
        parent_node["children"].append(child_node)
        _tree_to_dict(df_tree, child_id, parent_node=child_node)

    return parent_node


def tree_to_drawing_dict(df_nodes, root_id=None):
    if root_id is None:
        assert (
            len(df_nodes["root"].unique()) == 1
        ), "Need to provide `root` id if more than one tree in your dataframe."
        root_id = df_nodes["root"].take(0).item()

    df_tree_time = tracklet_time_table(df_nodes.filter(pl.col("root") == root_id))
    acc = {
        "id": root_id,
        "t": df_tree_time.select(["t_start", "t_stop"]).rows()[0],
        "children": [],
    }
    return _tree_to_drawing_dict_recursive(
        df_tree_time.filter(pl.col("ID") != root_id), root_id, acc
    )


def _tree_to_drawing_dict_recursive(
    df_tree, current_id, acc, features=("t_start", "t_stop")
):
    for child_id in df_tree.filter(pl.col("parent") == current_id)["ID"]:
        df_child_node = df_tree.filter(pl.col("ID") == child_id)
        child_node = {
            "id": child_id,
            "children": [],
            "t": df_child_node.select(features).rows()[0],
        }
        acc["children"].append(child_node)
        _tree_to_drawing_dict_recursive(df_tree, child_id, acc=child_node)

    return acc


def tree_to_dict2(df_nodes, root_id):
    df_tree = (
        df_nodes.lazy()
        .filter(pl.col("root") == root_id)
        .filter(pl.col("ID") != pl.col("parent"))
        .select(pl.col("parent"), pl.col("ID"))
        .unique(["parent", "ID"])
        .collect()
    )
    tree = _tree_to_dict_recursive(df_tree, root_id)
    return tree


def _tree_to_dict_recursive(df_tree, current_id, acc=None):
    if acc is None:
        acc = {
            "id": current_id,
            "children": [],
        }

    for child_id in df_tree.filter(pl.col("parent") == current_id)["ID"]:
        child_node = {
            "id": child_id,
            "children": [],
        }
        acc["children"].append(child_node)
        _tree_to_dict_recursive(df_tree, child_id, acc=child_node)

    return acc


def draw_recursive(
    tree,
    highlight_ids=None,
    highlight_kwargs=None,
    current_x=0,
    current_level=0,
    dx_0=0.2,
    last_x=None,
    last_y=None,
    ax=None,
):
    if ax is None:
        ax = plt.gca()
    if last_x is not None:
        h_line = Line2D(
            xdata=(last_x, current_x), ydata=(last_y, tree["t"][0]), c="0.7"
        )
        ax.add_line(h_line)

    if highlight_ids is not None and tree["id"] in highlight_ids:
        if highlight_kwargs is None:
            highlight_kwargs = {"c": "darkred", "lw": 1.5}
        line = Line2D(xdata=(current_x, current_x), ydata=tree["t"], **highlight_kwargs)
    else:
        line = Line2D(xdata=(current_x, current_x), ydata=tree["t"])
    ax.add_line(line)
    for i, child in enumerate(tree["children"]):
        if i % 2 == 0:
            draw_recursive(
                child,
                highlight_ids=highlight_ids,
                highlight_kwargs=highlight_kwargs,
                current_x=current_x + (dx_0 / 2**current_level),
                current_level=current_level + 1,
                last_x=current_x,
                last_y=tree["t"][1],
            )
        else:
            draw_recursive(
                child,
                highlight_ids=highlight_ids,
                highlight_kwargs=highlight_kwargs,
                current_x=current_x - (dx_0 / 2**current_level),
                current_level=current_level + 1,
                last_x=current_x,
                last_y=tree["t"][1],
            )
    ax.margins(x=0.1, y=0.1)
    return ax


def draw_forest(
    df,
    root_ids,
    highlight_ids=None,
    highlight_kwargs=None,
    n_per_row=30,
    n_rows=6,
    shift_x=1,
    title=None,
):
    fig, axs = plt.subplots(n_rows, 1, figsize=(20, 4 * n_rows), dpi=300)
    for i, ax in enumerate(axs):
        plt.sca(ax)
        current_x = 0
        for root_id in root_ids[i * n_per_row : (i + 1) * n_per_row]:
            tree = tree_to_drawing_dict(df, root_id)
            draw_recursive(
                tree,
                current_x=current_x,
                ax=ax,
                highlight_ids=highlight_ids,
                highlight_kwargs=highlight_kwargs,
            )
            current_x += shift_x

        ax.margins(x=0.01, y=0.1)
    if title is not None:
        fig.suptitle(title)
        fig.tight_layout()


def summary(df, short=True):
    summary_dict = site_summary_dict(df)
    for c1, c2 in zip(["nodes", "tracklets", "trees"], ["dummy", "generation", ""]):
        print(f"{summary_dict[c1]:>6} {c1:<11}{summary_dict.get(c2, ''):>5} {c2:<12}")
    print()


def site_summary_dict(df, short=True):
    stats = {}
    stats["nodes"] = df.height

    tracklet_ids = df["ID"].unique()
    stats["tracklets"] = len(tracklet_ids)

    root_ids = df["root"].unique()
    stats["trees"] = len(root_ids)

    stats["dummy"] = int(df["dummy"].sum())
    stats["generation"] = f"[{df['generation'].min()}-{df['generation'].max()}]"

    stats = {**stats, **{k: v for k, v in df["fate"].value_counts().rows()}}

    for g in df["generation"].unique():
        df_g = df.filter(pl.col("generation") == g)
        name = f"tracklets-g{g}"
        stats[name] = len(df_g["ID"].unique())
        stats[f"dummy-g{g}"] = int(df_g["dummy"].sum())

    # print(f'{"nodes:":<15}{df.height:>5}')
    generation_counts = (
        df.select(pl.col("generation").value_counts())
        .unnest("generation")
        .sort("generation")
    )
    for g, c in generation_counts.rows():
        name = f"nodes-g{g}"
        stats[name] = c
        # print(f'{name:<15}{c:>5}')

    track_length_quantiles = (
        df.group_by("ID")
        .count()
        .select(
            pl.col("count").quantile(i).alias(f"q{i:.1f}")
            for i in np.arange(0, 1.1, 0.1)
        )
    )
    for q_name, value in zip(
        track_length_quantiles.columns, track_length_quantiles.rows()[0]
    ):
        stats[f"tracklet_length-{q_name}"] = value

    tree_length_quantiles = (
        df.group_by("root")
        .count()
        .select(
            pl.col("count").quantile(i).alias(f"q{i:.1f}")
            for i in np.arange(0, 1.1, 0.1)
        )
    )
    for q_name, value in zip(
        tree_length_quantiles.columns, tree_length_quantiles.rows()[0]
    ):
        stats[f"tree_length-{q_name}"] = value
    return stats


FEATURES = (
    cs.by_name("nodes")
    | cs.by_name("nodes_root_perc")
    | cs.by_name("tracklets")
    | cs.by_name("tracklets_root_perc")
    | cs.by_name("tracklet_len_median")
    | cs.by_name("tree_len_median")
    | cs.matches("Fates.")
)

OLD_FEATURES = (
    cs.matches("tracklets")
    | cs.matches("nodes")
    | cs.matches("dummy")
    | cs.matches("tracklet_length")
    | cs.matches("tree_length")
)


def summary_new_format(stats):
    out_stats = stats.select(
        ~OLD_FEATURES,
        pl.col("nodes"),
        (pl.col("nodes-g0") / pl.col("nodes")).alias("nodes_root_perc"),
        pl.col("tracklets"),
        (pl.col("tracklets-g0") / pl.col("tracklets")).alias("tracklets_root_perc"),
        pl.col("tracklet_length-q0.5").alias("tracklet_len_median"),
        pl.col("tree_length-q0.5").alias("tree_len_median"),
    )
    return out_stats


def summaries(dfs: dict[str, pl.DataFrame], t_max=None) -> pl.DataFrame:
    stats_dicts = []
    for name, df in dfs.items():
        if t_max is None:
            t_max = df["t"].max()
        summary_dict = site_summary_dict(df.filter(pl.col("t") <= t_max))
        for e in name.split("; "):
            k, v = e.split("=")
            summary_dict[k] = v
        summary_dict["name"] = name

        stats_dicts.append(summary_dict)

    stats_frame = pl.DataFrame(stats_dicts)
    # return stats_frame
    stats = stats_frame.select(
        pl.col("name"),
        *[cs.by_name(k).cast(v) for k, v in override_repr.items()],
        OLD_FEATURES,
        cs.matches("Fates."),
    )
    return summary_new_format(stats)


def get_n_best_trees(name, n_trees=10, plot_trees=True):
    df = trackss_pl[name]
    tracks = trackss[name]
    df_times = tracklet_time_table(df)
    root_ids = sort_trees(df_times)
    if plot_trees:
        draw_forest(df_times, root_ids, title=name)
    best_trees = get_tree_by_ids(root_ids[:n_trees], tracks)
    return tree_to_napari(best_trees)


def select_lone_objects(df):
    df_gen = {
        g: df.filter(pl.col("generation") == g) for g in df["generation"].unique()
    }
    # return df_gen
    return df_gen[0].join(df_gen[1], left_on="ID", right_on="parent", how="anti")


def select_capped_objects(df):
    max_gen = df["generation"].max()
    df_capped = []
    for gl, gu in pairwise(range(1, max_gen + 1)):
        df_capped.append(
            df.filter(pl.col("generation") == gl).join(
                df.filter(pl.col("generation") == gu),
                left_on="ID",
                right_on="parent",
                how="semi",
            )
        )
    if not df_capped:
        return pl.DataFrame(schema=df.schema)
    return pl.concat(df_capped)


def get_capped_trees(name):
    df = trackss_pl[name]
    df_capped = select_capped_objects(df)
    root_ids_capped = df_capped["root"].unique(maintain_order=True)
    trees = get_tree_by_ids(root_ids_capped, trackss[name])
    return tree_to_napari(trees)


def get_trees(name):
    return tree_to_napari(trackss[name])


def get_child_ids(df, ids):
    df_idx = (
        df.lazy()
        .select(["parent", "ID"])
        .filter(pl.col("parent") != pl.col("ID"))
        .unique(subset=["parent", "ID"])
        .collect()
    )
    acc = []
    child_ids = _get_child_ids_recursive(df_idx, ids, acc)
    return child_ids


def _get_child_ids_recursive(df, ids, acc):
    df_direct_children = df.filter(pl.col("parent").is_in(ids))
    child_ids = df_direct_children.select("ID").to_series()
    if child_ids.is_empty():
        return acc
    acc.extend(child_ids)
    return _get_child_ids_recursive(df, child_ids, acc)


def remove_tracklet_(df, id_):
    df_id_removed = df.filter(pl.col("ID") != id_)
    children = df_id_removed.filter(pl.col("parent") == id_)["ID"]
    # Children need their parent set to their id
    return df_id_removed.with_columns(
        pl.when(pl.col("parent") == id_)
        .then(pl.col("ID"))
        .otherwise(pl.col("parent"))
        .alias("parent"),
    ).with_columns(
        *[
            pl.when(pl.col("ID").is_in(get_child_ids(df, child_id)))
            .then(child_id)
            .otherwise(pl.col("root"))
            for child_id in children
        ]
    )


def remove_tracklet_(df, id_):
    df_id_removed = df.filter(pl.col("ID") != id_)
    children = df_id_removed.filter(pl.col("parent") == id_)["ID"]
    grand_childrens = [get_child_ids(df_id_removed, child) for child in children]
    # Children need their parent set to their id
    df_id_removed = df_id_removed.with_columns(
        pl.when(pl.col("parent") == id_)
        .then(pl.col("ID"))
        .otherwise(pl.col("parent"))
        .alias("parent"),
        *(
            pl.when(pl.col("ID").is_in(grand_children + [child]))
            .then(pl.lit(child))
            .otherwise(pl.col("root"))
            .alias("root")
            for child, grand_children in zip(children, grand_childrens)
        ),
    )


def remove_branches(df, ids):
    """no rewiring necessary if all children removed"""
    if not isinstance(ids, Iterable) or isinstance(ids, str):
        ids = [ids]
    child_ids = get_child_ids(df, ids)
    return df.filter(~pl.col("ID").is_in(child_ids + ids))


def keep_branches(df, ids):
    """rewiring not implemented -> dangling roots."""
    if not isinstance(ids, Iterable) or isinstance(ids, str):
        ids = [ids]
    child_ids = get_child_ids(df, ids)
    return df.filter(pl.col("ID").is_in(child_ids + ids))


def remove_tracklet(df, id_):
    """rewiring implemented."""
    df_id_removed = df.filter(pl.col("ID") != id_)
    children = df_id_removed.filter(pl.col("parent") == id_)["ID"]
    # Children need their parent set to their id
    df_id_removed = df_id_removed.with_columns(
        pl.when(pl.col("parent") == id_)
        .then(pl.col("ID"))
        .otherwise(pl.col("parent"))
        .alias("parent"),
    )
    for child in children:
        df_id_removed = df_id_removed.with_columns(
            pl.when(pl.col("ID").is_in(get_child_ids(df_id_removed, child) + [child]))
            .then(pl.lit(child))
            .otherwise(pl.col("root"))
            .alias("root")
        )
    return df_id_removed


def plot_sample_capped_forest(stats):
    for name in stats["name"]:
        print(name)
        df = trackss_pl[name]
        df_capped = select_capped_objects(df)
        root_ids_capped = df_capped["root"].unique(maintain_order=True)
        ids_capped = df_capped["ID"].unique(maintain_order=True)
        draw_forest(df, root_ids_capped, highlight_ids=ids_capped, n_rows=4, title=name)


def plot_sample_longest_tree_forest(stats):
    for name in stats["name"]:
        print(name)
        df = trackss_pl[name]
        df_tree_times = tree_time_table(df)

        draw_forest(df, df_tree_times["root"], highlight_ids=None, n_rows=4, title=name)


def print_status(stats):
    for name in stats["name"]:
        df = trackss_pl[name]
        df_capped = select_capped_objects(df)
        root_ids_capped = df_capped["root"].unique(maintain_order=True)
        ids_capped = df_capped["ID"].unique(maintain_order=True)
        branches = df.filter(pl.col("fate") == "Fates.DIVIDE")

        print(name)
        print(f"n branchings:      {len(branches)}")
        print(f"n trees w/ capped: {len(root_ids_capped)}")
        print(f"n capped:          {len(ids_capped)}")
        print()


# %%
fate_counts = (
    trackss_pl[name].unique("ID")["fate"].value_counts().transpose(column_names="fate")
)
fate_counts
# columns = fate_counts['fate']
# fate_counts.select('counts').transpose()
# %%
stats = summaries(trackss_pl, t_max=None)
# stats_sample = stats.filter(pl.col('lambda_lin'))
stats
# %%
to_plot = stats.with_columns(
    pl.col("dist_thresh") + pl.Series(np.random.randn(stats.height) * 0.2)
).to_pandas()  # .plot(x='dist_thresh', y='nodes_root_perc', c='lambda_branch', kind='scatter')

g = sns.FacetGrid(to_plot, col="lambda_branch", col_wrap=3)

g.map_dataframe(
    sns.scatterplot,
    x="dist_thresh",
    y="tracklets_root_perc",
    hue="max_search_radius",
    style="lambda_link",
    palette="Set1",
)
g.add_legend()


g = sns.FacetGrid(to_plot, col="lambda_branch", col_wrap=3)

g.map_dataframe(
    sns.scatterplot,
    x="dist_thresh",
    y="nodes_root_perc",
    hue="max_search_radius",
    style="lambda_link",
    palette="Set1",
)
g.add_legend()

g = sns.FacetGrid(to_plot, col="lambda_branch", col_wrap=3)

g.map_dataframe(
    sns.scatterplot,
    x="dist_thresh",
    y="Fates.DIVIDE",
    hue="max_search_radius",
    style="lambda_link",
    palette="Set1",
)
g.add_legend()
# %%
stats_sample = stats.filter(
    pl.col("dist_thresh") == 20.0,
    pl.col("lambda_link") == 1000,
    pl.col("lambda_branch") == 1000,
)
print(stats_sample)
names = stats_sample["name"].to_list()
points = trackss_pl[names[0]].select(pl.col("t", "z", "y", "x"))
# %%
# plot_sample_capped_forest(stats_sample)
plot_sample_longest_tree_forest(stats_sample)
# %%
viewer = napari.Viewer()
viewer.add_image(img, scale=(1, 1.0, 2.6, 2.6), contrast_limits=(80, 1000))
viewer.add_points(points, size=3, face_color="magenta")
# for name in stats_select["name"]:
for name in names:
    viewer.add_tracks(
        **get_trees(name),
        name=name,
        tail_length=6,
        visible=False,
    )
# %%
df.group_by("t").count().with_columns(
    (pl.col("t") * 97).alias("t_min") / 60
).to_pandas().plot(x="t_min", y="count", kind="scatter")
# %%
df_tree = df.filter(pl.col("root") == 3809)
df_tree
# %%
from scipy.spatial.distance import cdist

for id_, df_tracklet in df_tree.select(["ID", "t", "z", "y", "x"]).group_by("ID"):
    points = df_tracklet.select(["z", "y", "x"])
    dists = cdist(points[:-1], points[1:])


# %%
def compute_distances(df_tracklet):
    return df_tracklet.with_columns(
        pl.sum_horizontal(
            (pl.col("z", "y", "x").shift(-1) - pl.col("z", "y", "x")).pow(2)
        ).sqrt()
    )["sum"][:-1]


# %%
name_disconnected = stats["name"][-1]
name_connected = stats["name"][1]

df_dis = trackss_pl[name_disconnected]
df_conn = trackss_pl[name_connected]

tracks_conn = trackss[name]
tracks_nap_conn = tree_to_napari(tracks_conn)
graph = tracks_nap_conn["graph"]

# %%
sns.kdeplot(df_times["total_time"])


# %%
def compute_all_tracklet_distances(df_nodes, threshold=(pl.col("total_time") > 5)):
    df_times = tracklet_time_table(df_nodes)
    tracklet_times = []
    for t_id in df_times.filter(threshold)["ID"]:
        tracklet_times.append(compute_distances(df_nodes.filter(pl.col("ID") == t_id)))
    all_tracklet_times = [e for tt in tracklet_times for e in tt]
    return tracklet_times, all_tracklet_times


# %%
tt_dis, att_dis = compute_all_tracklet_distances(df_dis)
tt_conn, att_conn = compute_all_tracklet_distances(df_conn)
# %%
sns.kdeplot(att_dis)
sns.kdeplot(att_conn)
# %%

# %%
sum(pl.Series(att_dis) > 10)
# %%
fig, ax = plt.subplots(dpi=300)
for td in tracklet_times:
    t = np.arange(len(td)) - td.arg_max()
    plt.plot(t, td, lw=0.4, c="0.7", alpha=0.3)

# %%
name = stats.sort("nodes_root_perc")["name"][0]
# %%
all_times = [e for tt in tracklet_times for e in tt]
# %%
sns.kdeplot(all_times)

# %% Visualize individual tracks
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
