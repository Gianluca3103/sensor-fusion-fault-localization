import argparse

from Fault_Localization_Model.config.defaults import config_defaults, load_json_config


def parse_args():
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", default=None, help="Optional JSON config file with dataset-generation defaults.")
    pre_args, _ = pre_parser.parse_known_args()
    defaults = config_defaults(load_json_config(pre_args.config))

    parser = argparse.ArgumentParser(
        parents=[pre_parser],
        description="Create K-Radar LiDAR reliability/fault heatmaps inside the radar-overlap field of view.",
    )
    parser.add_argument("--data-root", default=defaults["data_root"])
    parser.add_argument("--output-root", default=defaults["output_root"])
    parser.add_argument("--num-samples", type=int, default=defaults["num_samples"])
    parser.add_argument("--frames", type=int, nargs="*", default=None)
    parser.add_argument(
        "--temporal-split",
        choices=["train", "val", "test"],
        default=None,
        help="Use the first train ratio, next validation ratio, or final test ratio from every K-Radar sequence.",
    )
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--faults", nargs="*", default=defaults["faults"])
    parser.add_argument("--severities", type=int, nargs="*", default=defaults["severities"])
    parser.add_argument(
        "--fault-plan",
        nargs="*",
        default=defaults["fault_plan"],
        help="Exact mixed-severity plan, e.g. fog_sim:3 rain_sim:5 snow_sim:5 lidar_crosstalk_noise:1 fov_filter:1.",
    )
    parser.add_argument(
        "--no-shuffle",
        action="store_true",
        default=not bool(defaults["shuffle"]),
        help="Keep frame/fault/severity order deterministic.",
    )
    parser.add_argument(
        "--no-previews",
        action="store_true",
        help="Skip the six diagnostic PNGs per sample and save only training data plus manifests.",
    )
    parser.add_argument(
        "--remove-added-points",
        action=argparse.BooleanOptionalAction,
        default=bool(defaults["remove_added_points"]),
        help=(
            "Remove injected points with no clean-source identity before "
            "rasterizing the faulty LiDAR BEV. Missing-point faults remain."
        ),
    )
    parser.add_argument("--grid-size", type=int, default=defaults["grid_size"])
    parser.add_argument("--x-min", type=float, default=defaults["x_min"])
    parser.add_argument("--x-max", type=float, default=defaults["x_max"])
    parser.add_argument("--y-min", type=float, default=defaults["y_min"])
    parser.add_argument("--y-max", type=float, default=defaults["y_max"])
    parser.add_argument("--resolution", type=float, default=defaults["resolution"])
    parser.add_argument("--min-range", type=float, default=defaults["min_range"])
    parser.add_argument("--max-range", type=float, default=defaults["max_range"])
    parser.add_argument(
        "--observability-num-z-bins",
        type=int,
        default=defaults["observability_num_z_bins"],
        help="Temporary vertical bins used only by clean-LiDAR observability.",
    )
    parser.add_argument(
        "--observability-ray-support-tau",
        type=float,
        default=defaults["observability_ray_support_tau"],
        help="Tau in ray_support = 1 - exp(-ray_count / tau).",
    )
    parser.add_argument(
        "--movement-tolerance-m",
        type=float,
        default=defaults["movement_tolerance_m"],
        help="Maximum clean-to-faulty displacement treated as unchanged. Defaults to 0.05 m.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=defaults["num_workers"],
        help="Number of parallel sample-generation worker processes. Use 1 for the original sequential behavior.",
    )
    parser.add_argument(
        "--source-batch-size",
        type=int,
        default=32,
        help=(
            "Maximum samples per chronological worker batch. Random sample "
            "selection, indexes, seeds, and filenames remain unchanged."
        ),
    )
    parser.add_argument("--seed", type=int, default=defaults["seed"])
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args()
