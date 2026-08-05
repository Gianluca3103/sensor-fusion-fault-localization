import logging
import sys

from Fault_Localization_Model.config.defaults import config_get, load_json_config
from Fault_Localization_Model.config.validation import (
    require_directory,
    require_positive,
    require_range,
)


def setup_logging(level_name="INFO"):
    level = getattr(logging, str(level_name).upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
        force=True,
    )
