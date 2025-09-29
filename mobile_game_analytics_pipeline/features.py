from __future__ import annotations

from pathlib import Path

import pandas as pd
import typer
from loguru import logger

from mobile_game_analytics_pipeline.config import PROCESSED_DATA_DIR

app = typer.Typer(
    help="Create modeling-ready feature and label tables from clean_data.csv"
)

NUMERIC_COLUMNS = [
    "session_count",
    "purchase",
    "CAC",
    "revenue",
    "ROI",
    "retention_1",
]

CATEGORICAL_COLUMNS = [
    "acquisition_channel",
    "country",
    "platform",
    "version",
]

ID_COLUMNS = ["userid"]


def build_features(
    input_path: Path,
    features_path: Path,
    labels_path: Path,
) -> tuple[Path, Path]:
    if not input_path.exists():
        raise FileNotFoundError(input_path)

    df = pd.read_csv(input_path, low_memory=False)
    missing = [
        col
        for col in NUMERIC_COLUMNS + CATEGORICAL_COLUMNS + ID_COLUMNS + ["retention_7"]
        if col not in df.columns
    ]
    if missing:
        raise ValueError(f"Input dataset missing columns: {missing}")

    features = df[ID_COLUMNS + NUMERIC_COLUMNS + CATEGORICAL_COLUMNS].copy()
    features_path.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(features_path, index=False)

    labels = pd.DataFrame(
        {
            "userid": df["userid"],
            "retention_7": df["retention_7"].astype(int),
            "churn_flag": (~df["retention_7"].astype(bool)).astype(int),
        }
    )
    labels_path.parent.mkdir(parents=True, exist_ok=True)
    labels.to_csv(labels_path, index=False)

    return features_path, labels_path


@app.command()
def main(
    input_path: Path = typer.Option(
        PROCESSED_DATA_DIR / "clean_data.csv",
        "--input",
        help="Path to clean_data.csv produced by the dataset step.",
    ),
    features_path: Path = typer.Option(
        PROCESSED_DATA_DIR / "features.csv",
        "--features",
        help="Path where the feature table will be written.",
    ),
    labels_path: Path = typer.Option(
        PROCESSED_DATA_DIR / "labels.csv",
        "--labels",
        help="Path where the label table (including churn_flag) will be written.",
    ),
) -> None:
    features_path, labels_path = build_features(
        input_path.resolve(), features_path.resolve(), labels_path.resolve()
    )
    logger.success(f"Features saved to {features_path}")
    logger.success(f"Labels saved to {labels_path}")


if __name__ == "__main__":
    app()
