from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
import typer
from loguru import logger

from mobile_game_analytics_pipeline.config import MODELS_DIR, PROCESSED_DATA_DIR

app = typer.Typer(help="Generate churn predictions using a trained model pipeline.")


@app.command()
def main(
    features_path: Path = typer.Option(
        PROCESSED_DATA_DIR / "features.csv",
        "--features",
        help="Feature table used for inference.",
    ),
    model_path: Path = typer.Option(
        MODELS_DIR / "churn_model.pkl",
        "--model-path",
        help="Path to the trained model pipeline (joblib pickle).",
    ),
    output_path: Path = typer.Option(
        PROCESSED_DATA_DIR / "predictions.csv",
        "--output",
        help="Destination CSV containing churn probabilities.",
    ),
) -> None:
    features_path = features_path.resolve()
    model_path = model_path.resolve()
    output_path = output_path.resolve()

    if not features_path.exists():
        raise FileNotFoundError(features_path)
    if not model_path.exists():
        raise FileNotFoundError(model_path)

    features = pd.read_csv(features_path, low_memory=False)
    if "userid" in features.columns:
        ids = features["userid"]
    else:
        ids = pd.Series(range(len(features)), name="userid")

    model = joblib.load(model_path)
    churn_prob = model.predict_proba(
        features.drop(columns=[c for c in ["userid"] if c in features.columns])
    )[:, 1]
    predictions = pd.DataFrame(
        {
            "userid": ids,
            "churn_probability": churn_prob,
            "churn_prediction": (churn_prob >= 0.5).astype(int),
        }
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output_path, index=False)
    logger.success(f"Predictions saved to {output_path}")


if __name__ == "__main__":
    app()
