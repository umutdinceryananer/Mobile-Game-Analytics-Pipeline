import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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

SEED = 42
PROJ_ROOT = Path.cwd()
while PROJ_ROOT != PROJ_ROOT.parent and not (PROJ_ROOT / "data").exists():
    PROJ_ROOT = PROJ_ROOT.parent
DATA_PATH = PROJ_ROOT / "data" / "processed" / "clean_data.csv"
REPORTS_DIR = PROJ_ROOT / "reports"
TABLES_DIR = REPORTS_DIR / "tables"
FIGURES_DIR = REPORTS_DIR / "figures"
TABLES_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA_PATH, low_memory=False)
df["churn_flag"] = (~df["retention_7"].astype(bool)).astype(int)

numeric_features = [
    "session_count",
    "purchase",
    "CAC",
    "revenue",
    "ROI",
    "retention_1",
]

categorical_features = [
    "acquisition_channel",
    "country",
    "platform",
    "version",
]

feature_cols = numeric_features + categorical_features
X = df[feature_cols]
y = df["churn_flag"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=SEED
)

numeric_pipeline = Pipeline(
    steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
)

categorical_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_pipeline, numeric_features),
        ("cat", categorical_pipeline, categorical_features),
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
                    random_state=SEED,
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
                    random_state=SEED,
                    n_jobs=-1,
                ),
            ),
        ]
    ),
}

scoring = {
    "roc_auc": "roc_auc",
    "pr_auc": "average_precision",
    "balanced_accuracy": "balanced_accuracy",
    "accuracy": "accuracy",
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
records = []
for name, pipeline in models.items():
    cv_results = cross_validate(
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
                "roc_auc": cv_results["test_roc_auc"][fold_idx],
                "pr_auc": cv_results["test_pr_auc"][fold_idx],
                "balanced_accuracy": cv_results["test_balanced_accuracy"][fold_idx],
                "accuracy": cv_results["test_accuracy"][fold_idx],
            }
        )

backtest_df = pd.DataFrame(records)
backtest_path = TABLES_DIR / "backtest_scores.csv"
backtest_df.to_csv(backtest_path, index=False)

metrics_summary = {}
for name, pipeline in models.items():
    pipeline.fit(X_train, y_train)
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)
    metrics_summary[name] = {
        "roc_auc": roc_auc_score(y_test, y_proba),
        "pr_auc": average_precision_score(y_test, y_proba),
        "accuracy": accuracy_score(y_test, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
    }

metrics_path = TABLES_DIR / "model_metrics.json"
metrics_path.write_text(json.dumps(metrics_summary, indent=2), encoding="utf-8")

best_model_name = max(metrics_summary, key=lambda m: metrics_summary[m]["roc_auc"])
best_pipeline = models[best_model_name]
best_pipeline.fit(X_train, y_train)
y_test_proba = best_pipeline.predict_proba(X_test)[:, 1]

fpr, tpr, _ = roc_curve(y_test, y_test_proba)
precision, recall, _ = precision_recall_curve(y_test, y_test_proba)

fig, ax = plt.subplots(1, 2, figsize=(12, 5))
ax[0].plot(
    fpr,
    tpr,
    label=f"{best_model_name} (AUC={metrics_summary[best_model_name]['roc_auc']:.3f})",
)
ax[0].plot([0, 1], [0, 1], linestyle="--", color="gray")
ax[0].set_xlabel("False Positive Rate")
ax[0].set_ylabel("True Positive Rate")
ax[0].set_title("ROC Curve")
ax[0].legend(loc="lower right")

baseline = y_test.mean()
ax[1].plot(
    recall,
    precision,
    label=f"{best_model_name} (AP={metrics_summary[best_model_name]['pr_auc']:.3f})",
)
ax[1].hlines(
    baseline, 0, 1, linestyle="--", color="gray", label=f"Baseline={baseline:.3f}"
)
ax[1].set_xlabel("Recall")
ax[1].set_ylabel("Precision")
ax[1].set_title("Precision-Recall Curve")
ax[1].legend(loc="upper right")
fig.tight_layout()
roc_pr_path = FIGURES_DIR / "roc_pr_curves.png"
fig.savefig(roc_pr_path, dpi=150, bbox_inches="tight")
plt.close(fig)

segment_summary = (
    X_test.assign(churn_prob=y_test_proba)
    .groupby(["acquisition_channel", "platform"], as_index=False)["churn_prob"]
    .agg(
        [
            ("n_users", "size"),
            ("avg_churn_prob", "mean"),
            ("median_churn_prob", "median"),
        ]
    )
)
segment_summary = segment_summary.sort_values("avg_churn_prob", ascending=False)
segment_summary.to_csv(TABLES_DIR / "churn_risk_segments.csv", index=False)

print("Best model:", best_model_name)
print("Test metrics:", metrics_summary[best_model_name])
print("Top segment head:\n", segment_summary.head())
