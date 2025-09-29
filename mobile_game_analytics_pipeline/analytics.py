from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import typer
from loguru import logger

from mobile_game_analytics_pipeline.config import (
    FIGURES_DIR,
    PROCESSED_DATA_DIR,
    REPORTS_DIR,
)

app = typer.Typer(help="Generate funnel, ROI/ROAS, and retention artefacts")


def _load_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path, low_memory=False)
    expected = {
        "userid",
        "session_count",
        "retention_1",
        "retention_7",
        "purchase",
        "acquisition_channel",
        "CAC",
        "revenue",
        "platform",
    }
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"Input dataset missing required columns: {sorted(missing)}")
    return df


def _ensure_dirs(*paths: Path) -> None:
    for p in paths:
        p.mkdir(parents=True, exist_ok=True)


def generate_funnel(df: pd.DataFrame, tables_dir: Path, figures_dir: Path) -> None:
    funnel_flags = pd.DataFrame(
        {
            "install": np.ones(len(df), dtype=int),
            "onboarding": (df["session_count"].fillna(0) > 0).astype(int),
            "d1": df["retention_1"].fillna(False).astype(int),
            "d7": df["retention_7"].fillna(False).astype(int),
            "purchase": df["purchase"].fillna(False).astype(int),
        }
    )
    stage_labels = {
        "install": "Install",
        "onboarding": "Onboarding",
        "d1": "D1 Return",
        "d7": "D7 Return",
        "purchase": "Purchase",
    }
    total_users = len(funnel_flags)
    stage_counts = {
        stage_labels[key]: funnel_flags[key].sum() for key in funnel_flags.columns
    }

    def safe_div(num, denom):
        return float(num) / denom if denom else np.nan

    funnel_summary = pd.DataFrame(
        {
            "n_users": [total_users],
            "rate_install": [safe_div(stage_counts["Install"], total_users)],
            "rate_onboarding_from_install": [
                safe_div(stage_counts["Onboarding"], stage_counts["Install"])
            ],
            "rate_d1_from_onboarding": [
                safe_div(stage_counts["D1 Return"], stage_counts["Onboarding"])
            ],
            "rate_d7_from_d1": [
                safe_div(stage_counts["D7 Return"], stage_counts["D1 Return"])
            ],
            "rate_purchase_overall": [safe_div(stage_counts["Purchase"], total_users)],
            "rate_purchase_from_d7": [
                safe_div(stage_counts["Purchase"], stage_counts["D7 Return"])
            ],
        }
    )
    funnel_path = tables_dir / "funnel.csv"
    funnel_summary.round(6).to_csv(funnel_path, index=False)
    logger.info(f"Funnel summary saved to {funnel_path}")

    stage_order = ["Install", "Onboarding", "D1 Return", "D7 Return", "Purchase"]
    funnel_long = pd.DataFrame(
        {
            "stage": stage_order,
            "users": [stage_counts[stage] for stage in stage_order],
        }
    )
    funnel_long["pct_of_installs"] = funnel_long["users"] / total_users
    funnel_long["conversion_from_previous"] = funnel_long["users"].div(
        funnel_long["users"].shift(fill_value=total_users)
    )
    funnel_long_path = tables_dir / "funnel_long.csv"
    funnel_long.round(6).to_csv(funnel_long_path, index=False)
    logger.info(f"Funnel long-format table saved to {funnel_long_path}")

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(
        data=funnel_long,
        y="stage",
        x="pct_of_installs",
        order=list(reversed(stage_order)),
        color="#1f77b4",
        ax=ax,
    )
    ax.set_xlabel("Share of installs")
    ax.set_ylabel("")
    ax.set_xlim(0, 1)
    for idx, row in funnel_long.iterrows():
        ax.text(
            row["pct_of_installs"] + 0.01,
            len(stage_order) - 1 - idx,
            f"{row['pct_of_installs']:.1%}",
            va="center",
        )
    ax.set_title("User Funnel Conversion")
    fig.tight_layout()
    funnel_fig_path = figures_dir / "funnel.png"
    fig.savefig(funnel_fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Funnel chart saved to {funnel_fig_path}")


def generate_roi(df: pd.DataFrame, tables_dir: Path, figures_dir: Path) -> None:
    roi_by_channel = df.groupby("acquisition_channel", as_index=False).agg(
        users=("userid", "nunique"),
        revenue=("revenue", "sum"),
        ad_spend=("CAC", "sum"),
    )
    denom = roi_by_channel["ad_spend"].replace(0, pd.NA)
    roi_by_channel["roi"] = (
        roi_by_channel["revenue"] - roi_by_channel["ad_spend"]
    ) / denom
    roi_by_channel["roas"] = roi_by_channel["revenue"] / denom
    roi_by_channel = roi_by_channel.sort_values("roas", ascending=False)
    roi_path = tables_dir / "roi_by_channel.csv"
    roi_by_channel.round(6).to_csv(roi_path, index=False)
    logger.info(f"ROI table saved to {roi_path}")

    roi_long = roi_by_channel.melt(
        id_vars=["acquisition_channel", "users"],
        value_vars=["revenue", "ad_spend", "roi", "roas"],
        var_name="metric",
        value_name="value",
    )
    roi_long_path = tables_dir / "roi_by_channel_long.csv"
    roi_long.round(6).to_csv(roi_long_path, index=False)
    logger.info(f"ROI long-format table saved to {roi_long_path}")

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(
        data=roi_by_channel, x="roas", y="acquisition_channel", palette="viridis", ax=ax
    )
    ax.set_xlabel("ROAS")
    ax.set_ylabel("Channel")
    ax.set_title("Channel ROAS")
    ax.axvline(1.0, color="red", linestyle="--", linewidth=1, label="Break-even ROAS")
    ax.legend(loc="lower right")
    for idx, row in roi_by_channel.iterrows():
        ax.text(row["roas"] + 0.05, idx, f"{row['roas']:.2f}", va="center")
    fig.tight_layout()
    roi_fig_path = figures_dir / "roi_by_channel.png"
    fig.savefig(roi_fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"ROI chart saved to {roi_fig_path}")

    installs = df.groupby("acquisition_channel", as_index=False)["userid"].nunique()
    installs_path = tables_dir / "installs_by_channel.csv"
    installs.rename(columns={"userid": "users"}).to_csv(installs_path, index=False)
    logger.info(f"Installs by channel saved to {installs_path}")


def generate_retention(df: pd.DataFrame, tables_dir: Path, figures_dir: Path) -> None:
    channel_retention = df.groupby("acquisition_channel", as_index=False).agg(
        users=("userid", "nunique"),
        d1_rate=("retention_1", "mean"),
        d7_rate=("retention_7", "mean"),
    )
    channel_retention["d7_from_d1"] = channel_retention["d7_rate"] / channel_retention[
        "d1_rate"
    ].replace(0, np.nan)
    channel_retention = channel_retention.sort_values("d7_rate", ascending=False)
    retention_path = tables_dir / "retention_by_channel.csv"
    channel_retention.round(6).to_csv(retention_path, index=False)
    logger.info(f"Retention by channel saved to {retention_path}")

    cohort_version = df.groupby(["version", "acquisition_channel"], as_index=False).agg(
        users=("userid", "nunique"),
        d1_rate=("retention_1", "mean"),
        d7_rate=("retention_7", "mean"),
    )
    cohort_version_path = tables_dir / "retention_cohort_by_version.csv"
    cohort_version.round(6).to_csv(cohort_version_path, index=False)
    logger.info(f"Retention cohort table saved to {cohort_version_path}")

    heatmap_data = cohort_version.pivot_table(
        index="acquisition_channel", columns="version", values="d7_rate"
    )
    plt.figure(figsize=(10, 4 + len(heatmap_data) * 0.3))
    sns.heatmap(
        heatmap_data,
        cmap="YlGnBu",
        annot=True,
        fmt=".1%",
        cbar_kws={"label": "D7 retention"},
    )
    plt.title("D7 Retention by Version and Channel")
    plt.xlabel("Version")
    plt.ylabel("Acquisition channel")
    plt.tight_layout()
    heatmap_path = figures_dir / "retention_heatmap.png"
    plt.savefig(heatmap_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Retention heatmap saved to {heatmap_path}")


@app.command()
def main(
    input_path: Path = typer.Option(
        PROCESSED_DATA_DIR / "clean_data.csv",
        "--input",
        help="Source dataset (clean_data.csv).",
    ),
    tables_dir: Path = typer.Option(
        REPORTS_DIR / "tables",
        "--tables-dir",
        help="Destination for analytics tables.",
    ),
    figures_dir: Path = typer.Option(
        FIGURES_DIR,
        "--figures-dir",
        help="Destination for analytics figures.",
    ),
) -> None:
    input_path = input_path.resolve()
    tables_dir = tables_dir.resolve()
    figures_dir = figures_dir.resolve()
    _ensure_dirs(tables_dir, figures_dir)
    df = _load_dataset(input_path)

    generate_funnel(df, tables_dir, figures_dir)
    generate_roi(df, tables_dir, figures_dir)
    generate_retention(df, tables_dir, figures_dir)
    logger.success("Analytics tables and figures refreshed.")


if __name__ == "__main__":
    app()
