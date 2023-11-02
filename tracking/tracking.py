"""Code for tracking of cells based on features extraction from label images.
Example usage:

python tracking.py -c /path/to/defaults.json -i /path/to/input.parquet "my_first_experiment" 
"""
import argparse
import datetime
import json
import logging
from dataclasses import asdict, dataclass, field
from itertools import chain, product
from pathlib import Path
from pprint import pformat, pprint
from typing import TypeAlias

import btrack
import polars as pl
from btrack.constants import BayesianUpdates
from sklearn.preprocessing import PowerTransformer, RobustScaler, StandardScaler
from utils import apply_transformer_to_dataframe

logger = logging.getLogger(__name__)

Volume: TypeAlias = tuple[tuple[float, float], ...]

LOG_TRANSFORM_PATTERNS = [
    "^.*_Mean$",
    "^.*_Median$",
    "^.*_Maximum$",
    "^.*_Minimum$",
    "^.*_StandardDeviation$",
    "^.*_Variance$",
]


FEATURES = [
    "Elongation",
    "EquivalentSphericalPerimeter",
    "EquivalentSphericalRadius",
    "Flatness",
    "Perimeter",
    "PhysicalSize",
    "Roundness",
    "EquivalentEllipsoidDiameter.a",
    "EquivalentEllipsoidDiameter.b",
    "EquivalentEllipsoidDiameter.c",
    "PrincipalAxes.a-x",
    "PrincipalAxes.a-y",
    "PrincipalAxes.a-z",
    "PrincipalAxes.b-x",
    "PrincipalAxes.b-y",
    "PrincipalAxes.b-z",
    "PrincipalAxes.c-x",
    "PrincipalAxes.c-y",
    "PrincipalAxes.c-z",
    "H1A_Kurtosis",
    "H1A_Maximum",
    "H1A_Mean",
    "H1A_Median",
    "H1A_Minimum",
    "H1A_Skewness",
    "H1A_StandardDeviation",
    "H1A_Variance",
]

TRACKER_CONFIG = [
    "features",
    "update_method",
    "volume",
    "tracking_updates",
    "max_search_radius",
    "optimizer_options",
]

MOTION_CONFIG = ["max_lost", "prob_not_assign"]

HYPOTHESIS_CONFIG = [
    "lambda_time",
    "lambda_dist",
    "lambda_link",
    "lambda_branch",
    "theta_dist",
    "theta_time",
    "dist_thresh",
    "time_thresh",
    "apop_thresh",
    "segmentation_miss_rate",
    "apoptosis_rate",
    "relax",
]
OTHER_CONFIG = ["optimize", "_repr_params"]


# Available parameters
@dataclass
class Parameters:
    # Tracker
    optimize: bool = True
    features: tuple[str, ...] = tuple()
    update_method: BayesianUpdates = 0  # APPROXIMATE: 1, EXACT: 0
    volume: Volume = ((0, 665.6), (0, 665.6), (0, 251.0))
    tracking_updates: tuple[str, ...] = ("motion",)  # ("motion", "visual")
    max_search_radius: float = 15
    optimizer_options: dict[str, int] = None

    # Motion model
    max_lost: int = 1
    prob_not_assign: float = 0.1

    # Hypothesis model
    lambda_time: float = 5.0
    lambda_dist: float = 3.0
    lambda_link: float = 10.0
    lambda_branch: float = 50.0

    theta_dist: float = 20.0
    theta_time: float = 5.0
    dist_thresh: float = 40.0
    time_thresh: float = 2.0
    apop_thresh: float = 5
    segmentation_miss_rate: float = 0.1
    apoptosis_rate: float = 0.001
    relax: bool = True

    _repr_params: tuple[str, ...] = tuple()

    def __post_init__(self):
        if self.optimizer_options is None:
            self.optimizer_options = {"tm_lim": 100_000}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment_name", type=str)
    parser.add_argument("-n", "--tracking_to_frame_n", type=int, default=None)
    parser.add_argument("-i", "--input_path", type=str)
    parser.add_argument(
        "-c",
        "--base_config_path",
        type=str,
        default=r"C:\Users\hessm\Documents\zfish_local\live_data\my_cell_config.json",
    )
    args = parser.parse_args()
    
    if args.base_config_path is None:
        from btrack import datasets
        args.base_config_path = datasets.ce
    output_folder = (
        Path(args.input_path).parent / "tracking_results" / (f"{args.experiment_name}" + ("" if args.tracking_to_frame_n is None else f"_f{args.tracking_to_frame_n}"))
    )
    output_folder.mkdir(exist_ok=True, parents=True)

    # Load features & generate tracking objects.
    df = pl.read_parquet(args.input_path)
    if args.tracking_to_frame_n is not None:
        df = df.filter(pl.col('t') <= args.tracking_to_frame_n)
    
    # Preprocess Features.
    df = apply_transformer_to_dataframe(df, StandardScaler(), columns=FEATURES)
    df = df.select(["t", "z", "y", "x"] + [f for f in FEATURES if f in df.columns])
    objs = btrack.io.objects_from_array(df.to_numpy(), default_keys=df.columns)

    # Specify the experiment to run using (multiple) parameter_gen
    parameter_generators = [
        parameter_gen(
            max_search_radius=(6.0, 8.0, 10.0, 12.0),
            dist_thresh=(10.0, 12.0, 14.0, 16.0, 18.0, 20.0),
            time_thresh=(1.0,),
            max_lost=(1,),
            lambda_link=(1.0, 5.0, 1000.0),
            lambda_branch=(1.0, 5.0, 1000.0),
            # lambda_link=(20.0, ),
            # lambda_branch=(40.0, ),
            prob_not_assign=(0.1, ),
            # max_search_radius=(11.0,),
            # update_method=(0,),
            # optimize=(True,),
            # apop_thresh=(0, 1, 2, 5, 10),
            # segmentation_miss_rate=(0.01, 0.05, 0.1, 0.3),
            # apoptosis_rate=(0.001, 0.01, 0.05, 0.1),
        ),
        # parameter_gen(
        #     time_thresh=(1.0,),
        #     dist_thresh=(16.0,),
        #     lambda_link=(20.0, ),
        #     lambda_branch=(50.0, ),
        #     max_search_radius=(11.0,),
        #     update_method=(0,),
        #     tracking_updates=(("motion", "visual"),),
        #     features=(
        #         tuple(),
        #         ("PhysicalSize",),
        #         ("Roundness",),
        #         ('Flatness', ),
        #         ("H1A_Median",),
        #         ("H1A_Median", "H1A_Skewness", "H1A_Kurtosis"),
        #         (
        #             "H1A_Median",
        #             "H1A_Kurtosis",
        #             "H1A_Skewness",
        #             "H1A_StandardDeviation",
        #             "H1A_Variance",
        #         ),
        #         (
        #             "Elongation",
        #             "EquivalentEllipsoidDiameter.a",
        #             "EquivalentEllipsoidDiameter.b",
        #             "EquivalentEllipsoidDiameter.c",
        #             "EquivalentSphericalPerimeter",
        #             "EquivalentSphericalRadius",
        #             "Flatness",
        #             "Perimeter",
        #             "PhysicalSize",
        #         ),
        #         (
        #             "PrincipalAxes.a-x",
        #             "PrincipalAxes.a-y",
        #             "PrincipalAxes.a-z",
        #             "PrincipalAxes.b-x",
        #             "PrincipalAxes.b-y",
        #             "PrincipalAxes.b-z",
        #             "PrincipalAxes.c-x",
        #             "PrincipalAxes.c-y",
        #             "PrincipalAxes.c-z",
        #         ),
        #         (
        #             "Elongation",
        #             "EquivalentEllipsoidDiameter.a",
        #             "EquivalentEllipsoidDiameter.b",
        #             "EquivalentEllipsoidDiameter.c",
        #             "EquivalentSphericalPerimeter",
        #             "EquivalentSphericalRadius",
        #             "Flatness",
        #             "Perimeter",
        #             "PhysicalSize",
        #             "PrincipalAxes.a-x",
        #             "PrincipalAxes.a-y",
        #             "PrincipalAxes.a-z",
        #             "PrincipalAxes.b-x",
        #             "PrincipalAxes.b-y",
        #             "PrincipalAxes.b-z",
        #             "PrincipalAxes.c-x",
        #             "PrincipalAxes.c-y",
        #             "PrincipalAxes.c-z",
        #         ),
        #         FEATURES,
        #     ),
        # ),
        # parameter_gen(
        #     lambda_link=(2.0, 5.0, 10.0, 20.0, 40.0, 80.0),
        #     dist_thresh=(15.0,),
        # )
        # parameter_gen(
        #     time_thresh=(1.0, 2.0),
        #     dist_thresh=(15.0,),
        #     update_method=(0,),
        #     tracking_updates=(("motion", "visual"),),
        #     features=(
        #         tuple(),
        #         ("PhysicalSize",),
        #         ("Roundness", ),
        #         ("Flatness", ),
        #         tuple((f"EquivalentEllipsoidDiameter.{e}" for e in ["a", "b", "c"])),
        #         tuple(FEATURES)
        #     ),
        # ),
    ]

    # Select one parameter file based on slurm array id.
    all_parameters = list(chain(*parameter_generators))

    for i, parameters in enumerate(all_parameters):
        base_name = Path(args.input_path).stem
        tracks_out_file = output_folder / f"{base_name}_{i:03d}_tracks.h5"
        config_out_file = output_folder / f"{base_name}_{i:03d}_config.json"
        params_out_file = output_folder / f"{base_name}_{i:03d}_params.json"

        # Load base configuration (most of it overwritten in this script!).
        base_config = btrack.config.load_config(args.base_config_path)

        pprint(base_config)
        # Overwrite parameters in base_config & save the result
        for k, v in asdict(parameters).items():
            if k in TRACKER_CONFIG:
                setattr(base_config, k, v)
            elif k in MOTION_CONFIG:
                setattr(base_config.motion_model, k, v)
            elif k in HYPOTHESIS_CONFIG:
                setattr(base_config.hypothesis_model, k, v)
            elif k in OTHER_CONFIG:
                continue
            else:
                raise ValueError(f"Unknown argument {k}")

        pprint(base_config)
        with open(config_out_file, "w") as f:
            f.write(base_config.json(indent=2))
        with open(params_out_file, "w") as f:
            json.dump(asdict(parameters), f, indent=2)

        # Run the tracker
        with btrack.BayesianTracker() as tracker:
            tracker.configure(base_config)
            tracker.append(objs)
            tracker.track()
            if parameters.optimize:
                optimized = tracker.optimise()

            tracker.export(tracks_out_file, obj_type="obj_type_1")


def parameter_gen(
    optimize: tuple[bool, ...] | None = None,
    features: tuple[tuple[str, ...], ...] | None = None,
    update_method: tuple[BayesianUpdates, ...] | None = None,
    volume: tuple[Volume, ...] | None = None,
    tracking_updates: tuple[tuple[str, ...], ...] | None = None,
    max_search_radius: tuple[float, ...] | None = None,
    max_lost: tuple[int, ...] | None = None,
    prob_not_assign: tuple[float, ...] | None = None,
    lambda_time: float | None = None,
    lambda_dist: float | None = None,
    lambda_link: float | None = None,
    lambda_branch: float | None = None,
    theta_dist: float | None = None,
    theta_time: float | None = None,
    dist_thresh: float | None = None,
    time_thresh: float | None = None,
    apop_thresh: float | None = None,
    segmentation_miss_rate: float | None = None,
    apoptosis_rate: float | None = None,
    relax: bool | None = None,
):
    params_set = {
        k: v for k, v in locals().items() if v is not None and k != "repr_params"
    }
    repr_params = tuple([k for k, v in params_set.items() if len(v) > 1])
    for value_pair in product(*params_set.values()):
        yield Parameters(
            **{k: v for k, v in zip(params_set.keys(), value_pair)},
            _repr_params=repr_params,
        )


if __name__ == "__main__":
    main()
