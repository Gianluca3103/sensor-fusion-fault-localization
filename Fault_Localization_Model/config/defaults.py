from pathlib import Path
import json


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_KRADAR_ROOT = REPO_ROOT.parent / "K-Radar_Data"
DEFAULT_INJECTOR_ROOT = REPO_ROOT / "Weather_Injector" / "3D_Corruptions_AD"
DEFAULT_FOG_ROOT = REPO_ROOT / "Weather_Injector" / "LiDAR_fog_sim"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "Fault_Localization_Model" / "grid_reliability_heatmaps"


def load_json_config(path):
    if path is None:
        return {}
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    try:
        with path.open("r", encoding="utf-8") as file:
            config = json.load(file)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON config {path}: {exc}") from exc
    if not isinstance(config, dict):
        raise ValueError(f"Config root in {path} must be a JSON object")
    return config

#Receives the loaded dictionary, checks every dotted_key, and if missing returns the default
def config_get(config, dotted_key, default=None):
    value = config
    for key in dotted_key.split("."): #checks if key exists, if key exists extract value
        if not isinstance(value, dict) or key not in value:
            return default
        value = value[key]
    return value

#Default configuration values
def config_defaults(config):
    return {
        "data_root": config_get(config, "paths.data_root", str(DEFAULT_KRADAR_ROOT)),
        "output_root": config_get(config, "paths.output_root", str(DEFAULT_OUTPUT_ROOT)),
        "num_samples": config_get(config, "generation.num_samples", 24),
        "seed": config_get(config, "generation.seed", 42),
        "shuffle": config_get(config, "generation.shuffle", True),
        "fault_plan": config_get(config, "faults.plan", None),
        "faults": config_get(config, "faults.names", None),
        "severities": config_get(config, "faults.severities", None),
        "num_workers": config_get(config, "generation.num_workers", 1),
        "remove_added_points": config_get(
            config, "generation.remove_added_points", False
        ),
        "observability_num_z_bins": config_get(
            config, "observability.num_z_bins", 16
        ),
        "observability_ray_support_tau": config_get(
            config, "observability.ray_support_tau", 4.0
        ),
        # Keep reliability/fault evidence cell-aligned with the 320 x 320 BEV.
        "grid_size": config_get(config, "bev.grid_size", 320),
        "x_min": config_get(config, "bev.x_min", 0.0),
        "x_max": config_get(config, "bev.x_max", 64.0),
        "y_min": config_get(config, "bev.y_min", -32.0),
        "y_max": config_get(config, "bev.y_max", 32.0),
        "resolution": config_get(config, "bev.resolution", 0.20),
        "min_range": config_get(config, "bev.min_range", 1.0),
        "max_range": config_get(config, "bev.max_range", 118.037109375),
        "movement_tolerance_m": config_get(config, "reliability.movement_tolerance_m", 0.05),
    }
