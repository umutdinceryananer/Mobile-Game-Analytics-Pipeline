from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import typer
from loguru import logger
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from mobile_game_analytics_pipeline.config import (
    FIGURES_DIR,
    MODELS_DIR,
    PROCESSED_DATA_DIR,
    REPORTS_DIR,
)

app = typer.Typer(help="Train churn model and persist metrics/artefacts.")

NUMERIC_COLUMNS = [
    "session_count",
    "purchase",
    "CAC",
    "revenue",
    "ROI",
    "retention_1",
]
CATEGORICAL_COLUMNS = ["acquisition_channel", "country", "platform", "version"]


def _load_dataset(
    features_path: Path, labels_path: Path
) -> tuple[pd.DataFrame, pd.Series]:
    features = pd.read_csv(features_path, low_memory=False)
    labels = pd.read_csv(labels_path, low_memory=False)
    if "churn_flag" not in labels.columns:
        raise ValueError(
            "Labels file must contain a 'churn_flag' column (0=retained,1=churn)"
        )
    return features, labels["churn_flag"].astype(int)


def _build_models(random_state: int) -> dict[str, Pipeline]:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, NUMERIC_COLUMNS),
            ("cat", categorical_pipeline, CATEGORICAL_COLUMNS),
        ]
    )

    models = {
        "log_reg": Pipeline(
            steps=[
                ("prep", preprocessor),
                (
                    "model",
                    LogisticRegression(
                        max_iter=200,
                        class_weight="balanced",
                        random_state=random_state,
                    ),
                ),
            ]
        ),
        "xgb": Pipeline(
            steps=[
                ("prep", preprocessor),
                (
                    "model",
                    XGBClassifier(
                        objective="binary:logistic",
                        eval_metric="logloss",
                        subsample=0.8,
                        colsample_bytree=0.8,
                        max_depth=4,
                        learning_rate=0.1,
                        n_estimators=300,
                        reg_lambda=1.0,
                        random_state=random_state,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
    }
    return models


def _evaluate_models(models, X_train, y_train) -> pd.DataFrame:
    scoring = {
        "roc_auc": "roc_auc",
        "pr_auc": "average_precision",
        "balanced_accuracy": "balanced_accuracy",
        "accuracy": "accuracy",
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    records = []
    for name, pipeline in models.items():
        results = cross_validate(
            pipeline,
            X_train,
            y_train,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
            return_train_score=False,
        )
        for fold_idx in range(cv.n_splits):
            records.append(
                {
                    "model": name,
                    "fold": fold_idx + 1,
                    "roc_auc": results["test_roc_auc"][fold_idx],
                    "pr_auc": results["test_pr_auc"][fold_idx],
                    "balanced_accuracy": results["test_balanced_accuracy"][fold_idx],
                    "accuracy": results["test_accuracy"][fold_idx],
                }
            )
    return pd.DataFrame(records)


def _summarise_metrics(
    models, X_train, X_test, y_train, y_test
) -> tuple[dict, str, dict]:
    metrics = {}
    test_predictions = {}
    for name, pipeline in models.items():
        pipeline.fit(X_train, y_train)
        y_proba = pipeline.predict_proba(X_test)[:, 1]
        y_pred = (y_proba >= 0.5).astype(int)
        metrics[name] = {
            "roc_auc": roc_auc_score(y_test, y_proba),
            "pr_auc": average_precision_score(y_test, y_proba),
            "accuracy": accuracy_score(y_test, y_pred),
            "balanced_accuracy": balanced_accuracy_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred),
        }
        test_predictions[name] = y_proba
    best_model = max(metrics, key=lambda m: metrics[m]["roc_auc"])
    return metrics, best_model, test_predictions[best_model]


def _plot_roc_pr(y_test, y_proba, auc_score, ap_score, output_path: Path) -> None:
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    baseline = y_test.mean()

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    ax[0].plot(fpr, tpr, label=f"ROC (AUC={auc_score:.3f})")
    ax[0].plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax[0].set_xlabel("False Positive Rate")
    ax[0].set_ylabel("True Positive Rate")
    ax[0].set_title("ROC Curve")
    ax[0].legend(loc="lower right")

    ax[1].plot(recall, precision, label=f"PR (AP={ap_score:.3f})")
    ax[1].hlines(
        baseline, 0, 1, linestyle="--", color="gray", label=f"Baseline={baseline:.3f}"
    )
    ax[1].set_xlabel("Recall")
    ax[1].set_ylabel("Precision")
    ax[1].set_title("Precision-Recall Curve")
    ax[1].legend(loc="upper right")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


@app.command()
def main(
    features_path: Path = typer.Option(
        PROCESSED_DATA_DIR / "features.csv",
        "--features",
        help="CSV produced by the features step.",
    ),
    labels_path: Path = typer.Option(
        PROCESSED_DATA_DIR / "labels.csv",
        "--labels",
        help="Labels CSV containing churn_flag column.",
    ),
    model_path: Path = typer.Option(
        MODELS_DIR / "churn_model.pkl",
        "--model-path",
        help="Where to persist the best performing model pipeline.",
    ),
    reports_dir: Path = typer.Option(
        REPORTS_DIR / "tables",
        "--reports-dir",
        help="Directory for tabular artefacts.",
    ),
    figures_dir: Path = typer.Option(
        FIGURES_DIR,
        "--figures-dir",
        help="Directory for figures/plots.",
    ),
    random_state: int = typer.Option(
        42, "--seed", help="Random seed for splits and models."
    ),
) -> None:
    features_path = features_path.resolve()
    labels_path = labels_path.resolve()
    model_path = model_path.resolve()
    reports_dir = reports_dir.resolve()
    figures_dir = figures_dir.resolve()

    X, y = _load_dataset(features_path, labels_path)
    missing_cols = [
        col for col in NUMERIC_COLUMNS + CATEGORICAL_COLUMNS if col not in X.columns
    ]
    if missing_cols:
        raise ValueError(f"Feature table missing columns: {missing_cols}")

    X_train, X_test, y_train, y_test = train_test_split(
        X[NUMERIC_COLUMNS + CATEGORICAL_COLUMNS],
        y,
        test_size=0.3,
        stratify=y,
        random_state=random_state,
    )

    models = _build_models(random_state)
    backtest_df = _evaluate_models(models, X_train, y_train)
    reports_dir.mkdir(parents=True, exist_ok=True)
    backtest_path = reports_dir / "backtest_scores.csv"
    backtest_df.to_csv(backtest_path, index=False)
    logger.info(f"Cross-validation scores saved to {backtest_path}")

    metrics_summary, best_model_name, y_test_proba = _summarise_metrics(
        models, X_train, X_test, y_train, y_test
    )
    metrics_path = reports_dir / "model_metrics.json"
    metrics_path.write_text(json.dumps(metrics_summary, indent=2), encoding="utf-8")
    logger.info(f"Test metrics saved to {metrics_path}")

    best_pipeline = models[best_model_name]
    best_pipeline.fit(X_train, y_train)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_pipeline, model_path)
    logger.success(f"Best model ({best_model_name}) stored at {model_path}")

    roc_pr_path = figures_dir / "roc_pr_curves.png"
    _plot_roc_pr(
        y_test,
        y_test_proba,
        metrics_summary[best_model_name]["roc_auc"],
        metrics_summary[best_model_name]["pr_auc"],
        roc_pr_path,
    )
    logger.info(f"ROC/PR curve written to {roc_pr_path}")


if __name__ == "__main__":
    app()
