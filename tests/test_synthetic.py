import numpy as np
import pandas as pd
import pytest


# Helper to load processed parquet
def load_df():
    return pd.read_parquet("data/processed/events.parquet")


def test_required_columns():
    """Check if all synthetic columns exist"""
    df = load_df()
    required = {"acquisition_channel", "CAC", "revenue", "ROI", "country", "platform"}
    missing = required - set(df.columns)
    assert not missing, f"Missing columns: {missing}"


def test_determinism():
    """Ensure deterministic assignment (same userid -> same channel/country/platform)"""
    df1 = load_df()
    df2 = load_df()
    merged = df1.merge(df2, on="userid", suffixes=("_a", "_b"))
    assert (merged["acquisition_channel_a"] == merged["acquisition_channel_b"]).all()
    assert (merged["country_a"] == merged["country_b"]).all()
    assert (merged["platform_a"] == merged["platform_b"]).all()


def test_country_distribution():
    """Country distribution should roughly match expected ratios"""
    df = load_df()
    shares = df["country"].value_counts(normalize=True)
    assert abs(shares["USA"] - 0.7727) < 0.02
    assert abs(shares["Mexico"] - 0.1364) < 0.02
    assert abs(shares["Brazil"] - 0.0909) < 0.02


def test_platform_distribution():
    """Platform install share should roughly match expected ratios"""
    df = load_df()
    shares = df["platform"].value_counts(normalize=True)
    assert abs(shares["App Store"] - 0.25) < 0.02
    assert abs(shares["Google Play"] - 0.75) < 0.02


def test_platform_revenue_share():
    """Revenue share across platforms should match target ratio (26M vs 9M)"""
    df = load_df()
    rev = df.groupby("platform")["revenue"].sum()
    shares = rev / rev.sum()
    assert abs(shares["App Store"] - 0.7429) < 0.02
    assert abs(shares["Google Play"] - 0.2571) < 0.02


def test_no_negative_values():
    """Revenue and CAC should not be negative; ROI should be finite"""
    df = load_df()
    assert (df["CAC"] >= 0).all()
    assert (df["revenue"] >= 0).all()
    assert np.isfinite(df["ROI"]).all()
