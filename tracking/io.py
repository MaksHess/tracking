import numpy as np
import polars as pl

# %% Feature extraction function
LABEL_FEATURES = [
    "PhysicalSize",
    "Elongation",
    "Flatness",
    "Roundness",
    "Perimeter",
    "EquivalentSphericalPerimeter",
    "EquivalentSphericalRadius",
    "Centroid",
    "PrincipalAxes",
    "EquivalentEllipsoidDiameter",
]
INTENSITY_FEATURES = [
    "Mean",
    "Median",
    "Kurtosis",
    "Skewness",
    "Maximum",
    "Minimum",
    "Variance",
    "StandardDeviation",
]

RENAME_MAP = {
    "Centroid.x": "x",
    "Centroid.y": "y",
    "Centroid.z": "z",
    "label": "img_label",
}


def _extract_features(lbls, img=None, write_to=None):
    from zfish.features.label import get_label_features
    from zfish.features.polars_utils import unnest_all_structs

    dfs = []
    for i in range(len(lbls)):
        print(f"extracting label features t={i}")
        df = get_label_features(
            lbls[i],
            features=LABEL_FEATURES,
        )

        if img is not None:
            from zfish.features.intensity import get_intensity_features

            print(f"extracting intensity features t={i}")
            df_intensity = get_intensity_features(
                lbls[i], img[i], features=INTENSITY_FEATURES
            )
            df = df.join(df_intensity, on="label")

        dfs.append(
            df.pipe(unnest_all_structs)
            .rename(RENAME_MAP)
            .with_columns(pl.lit(i).alias("t"))
        )

    df_out = pl.concat(dfs)
    if write_to is not None:
        df_out.write_parquet(write_to)
    return df_out


def extract_features(labels_gen, image_gen, out_path=None):
    from zfish.features.intensity import get_intensity_features
    from zfish.features.label import get_label_features
    from zfish.features.polars_utils import unnest_all_structs

    dfs = []
    for lbls, img in zip(labels_gen, image_gen):
        df_labels = get_label_features(lbls, features=LABEL_FEATURES)
        df_intensity = get_intensity_features(lbls, img, features=INTENSITY_FEATURES)
        dfs.append(
            df_labels.join(df_intensity, on="label")
            .pipe(unnest_all_structs)
            .rename(RENAME_MAP)
            .with_columns(pl.lit(lbls.t.item()).alias("t"))
        )
    df_out = pl.concat(dfs)
    if out_path is not None:
        df_out.write_parquet(out_path)
    return df_out


# %% Image, label & feature loaders
def load_labels(fn, scale=(1, 1.0, 0.65, 0.65), dims=("t", "z", "y", "x"), n_max=None):
    import zarr

    from zfish.image.image import to_si

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
    import zarr

    from zfish.image.image import to_si

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
    import zarr

    from zfish.image.image import to_si

    arr_img = zarr.open(fn)
    if n_max is not None:
        img = arr_img[level][:n_max, ...]
    else:
        img = arr_img[level][...]
    return to_si(img, dims=dims, scale=scale, c_coords=c_coords).squeeze()


def image_generator(
    fn, level=0, scale=(1.0, 0.65, 0.65), dims=("c", "z", "y", "x"), c_coords=("H1A",)
):
    import zarr

    from zfish.image.image import to_si

    arr_img = zarr.open(fn)
    img = arr_img[level]
    for i in range(img.shape[0]):
        print(f"loading image t={i}")
        yield to_si(img[i, ...], scale=scale, dims=dims, c_coords=c_coords).expand_dims(
            {"t": [i]}
        ).squeeze()


def load_features(fn):
    import polars as pl

    return pl.read_parquet(fn)
