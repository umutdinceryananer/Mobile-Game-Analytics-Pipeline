from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import typer
from loguru import logger

from mobile_game_analytics_pipeline.config import (
    DATA_DIR,
    PROCESSED_DATA_DIR,
    PROJ_ROOT,
    RAW_DATA_DIR,
)

app = typer.Typer(
    help="CLI wrapper around data/make_dataset.py to rebuild the synthetic dataset."
)


def _resolve_raw_env(raw_csv: Path | None) -> tuple[Path, Path]:
    if raw_csv is None:
        raw_dir = RAW_DATA_DIR
        raw_file = raw_dir / "cookie_cats.csv"
        if not raw_file.exists():
            raise FileNotFoundError(
                f"Default raw file not found at {raw_file}. Provide --raw-csv path."
            )
        return raw_dir, raw_file
    raw_csv = raw_csv.resolve()
    if not raw_csv.exists():
        raise FileNotFoundError(raw_csv)
    if raw_csv.name != "cookie_cats.csv":
        raise typer.BadParameter(
            "Raw dataset must be named 'cookie_cats.csv' (expected by make_dataset.py)."
        )
    return raw_csv.parent, raw_csv


@app.command()
def main(
    config_path: Path = typer.Option(
        Path("data/config/synthetic.yaml"),
        "--config",
        help="Path to synthetic configuration YAML.",
    ),
    raw_csv: Path
    | None = typer.Option(
        None,
        "--raw-csv",
        help="Path to cookie_cats.csv (defaults to data/raw/cookie_cats.csv).",
    ),
    processed_dir: Path = typer.Option(
        Path("data/processed"),
        "--processed-dir",
        help="Directory where processed outputs will be written.",
    ),
) -> None:
    """Trigger data/make_dataset.py within the current virtualenv."""

    processed_dir = processed_dir.resolve()
    processed_dir.mkdir(parents=True, exist_ok=True)

    config_path = config_path.resolve()
    if not config_path.exists():
        raise FileNotFoundError(config_path)

    raw_dir, raw_file = _resolve_raw_env(raw_csv)

    script_path = PROJ_ROOT / "data" / "make_dataset.py"
    if not script_path.exists():
        raise FileNotFoundError(script_path)

    env = os.environ.copy()
    env["SYNTH_CONFIG"] = str(config_path)
    env["DATA_RAW_DIR"] = str(raw_dir)
    env["DATA_PROCESSED_DIR"] = str(processed_dir)

    logger.info(f"Running synthetic data generation via {script_path}")
    subprocess.run([sys.executable, str(script_path)], check=True, env=env)
    logger.success(f"Dataset rebuilt at {processed_dir}")


if __name__ == "__main__":
    app()
