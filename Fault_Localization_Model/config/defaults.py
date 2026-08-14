from pathlib import Path
import json


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INJECTOR_ROOT = REPO_ROOT / "Weather_Injector" / "3D_Corruptions_AD"
DEFAULT_FOG_ROOT = REPO_ROOT / "Weather_Injector" / "LiDAR_fog_sim"


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
