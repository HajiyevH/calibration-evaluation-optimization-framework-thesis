"""CLI entry point for the Calibri experiment framework.

Usage (after pip install -e .):
    calibri-run run <config.yaml> [--runs-dir DIR] [--cli-path PATH]
    calibri-run validate <config.yaml>

Legacy usage:
    python scripts/run_experiment.py run <config.yaml>
"""

import argparse
import sys
from pathlib import Path

from calibri.framework.config import load_config, config_to_dict
from calibri.framework.runner import run_experiment

import yaml


def cmd_validate(args):
    """Validate a config file and print the resolved config."""
    try:
        config = load_config(args.config)
    except (ValueError, FileNotFoundError) as e:
        print(f"INVALID: {e}", file=sys.stderr)
        return 1

    print("Config is valid.")
    print()
    print(yaml.dump(config_to_dict(config), default_flow_style=False, sort_keys=False))
    return 0


def cmd_run(args):
    """Run an experiment from a config file."""
    try:
        config = load_config(args.config)
    except (ValueError, FileNotFoundError) as e:
        print(f"Config error: {e}", file=sys.stderr)
        return 1

    runs_dir = Path(args.runs_dir) if args.runs_dir else None
    cli_path = Path(args.cli_path) if args.cli_path else None

    try:
        summary = run_experiment(config, base_dir=runs_dir, cli_path=cli_path)
    except Exception as e:
        print(f"Experiment failed: {e}", file=sys.stderr)
        return 1

    if summary.get("solver", {}).get("success"):
        return 0
    return 1


def main():
    parser = argparse.ArgumentParser(
        description="Calibri experiment framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # validate
    p_validate = subparsers.add_parser("validate", help="Validate a config file")
    p_validate.add_argument("config", help="Path to experiment YAML config")

    # run
    p_run = subparsers.add_parser("run", help="Run an experiment")
    p_run.add_argument("config", help="Path to experiment YAML config")
    p_run.add_argument("--runs-dir", default=None, help="Base directory for run output")
    p_run.add_argument("--cli-path", default=None, help="Path to ba_cli.exe")

    args = parser.parse_args()

    if args.command == "validate":
        sys.exit(cmd_validate(args))
    elif args.command == "run":
        sys.exit(cmd_run(args))


if __name__ == "__main__":
    main()
