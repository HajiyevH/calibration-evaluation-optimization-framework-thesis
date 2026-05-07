"""CLI entry point for launching experiments from the Streamlit UI."""

import sys
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from calibri.framework.config import load_config
from calibri.framework.runner import run_experiment

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
        stream=sys.stdout,
    )
    config_path = Path(sys.argv[2])  # sys.argv[1] is "run"
    config = load_config(config_path)
    run_experiment(config)
