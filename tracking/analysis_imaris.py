# %%
import pandas as pd
import polars as pl
import polars.selectors as cs

fn_pos = r"C:\Users\hessm\Downloads\20231026H1A1_s1-t50_pos.csv"
fn_gen = r"C:\Users\hessm\Downloads\20231026H1A1_s1-t50_generation.csv"
fn_excel = r"C:\Users\hessm\Downloads\20231026H1A1_s1-t50_BIO325.xls"
fn_excel = r"C:\Users\hessm\Downloads\20231026H1A1_s1-t50.xlsx"
# %%
df_pos = pl.DataFrame(pd.read_excel(fn_excel, "Position", header=1))
df_gen = pl.DataFrame(pd.read_excel(fn_excel, "Generation", header=1))
df_dis = pl.DataFrame(pd.read_excel(fn_excel, "Displacement", header=1))
df_pos
# %%
rename_map = {
    "Time": "t",
    "Position Z": "z",
    "Position Y": "y",
    "Position X": "x",
    "TrackID": "root",
    "Generation": "generation",
    "Displacement X": "dx",
    "Displacement Y": "dy",
    "Displacement Z": "dz",
}


df = (
    df_pos
    .join(df_gen.select(pl.col("Generation"), pl.col("ID")), on="ID")
    .join(df_dis.select(cs.matches('Displacement'), cs.by_name('ID')), on='ID')
    .rename(rename_map)
    .select(pl.col("root", "t", "z", "y", "x", "generation", "dz", "dy", "dx", "ID"))
)

df.select(*['t', 'z', 'y', 'x'], pl.col('root').alias('_root'), pl.col('generation').alias('_generation'), pl.col('ID').alias('_id')).write_parquet(r"C:\Users\hessm\Documents\zfish_local\live_data\imaris_full.parquet")
# %%
# for root_id, df_tree in df.group_by('root', maintain_order=True):
#     for generation, df_gen in df_tree.groupby('generation', maintain_order=True):
#         print(root_id, generation)

root_id = 1000000017
df_tree = df.filter(pl.col('root')==root_id).filter(pl.col('generation').is_between(1, 3))
child_positions = df_tree.select(pl.col('ID').alias('ID_child'), *[(pl.col(f"{dim}")+pl.col(f"d{dim}")).alias(f"{dim}") for dim in ['z', 'y', 'x']])
# %%
points = df_tree.select(['t', 'z', 'y', 'x'])
points_shadow = points.with_columns(pl.col('t')+1)
points_ghost = points.with_columns(pl.col('t')-1)
dirs = df_tree.select(pl.lit(1).alias('t'), *['dz', 'dy', 'dx'])
vecs = np.stack([points, dirs],axis=1)
# %%
df.groupby(['root', 't']).count().sort(['root', 't'])
# %%
tree_id = df.group_by("root").count().sort("count", descending=True)["root"][0]

df_tree = df.filter(pl.col("root") == tree_id)
# %%
df_tree.filter(pl.col("t") > 9)
# %%
import napari

viewer = napari.Viewer()
viewer.add_points(points, size=3)
viewer.add_points(points_shadow, size=1)
viewer.add_points(points_ghost, size=1)
viewer.add_vectors(vecs)
