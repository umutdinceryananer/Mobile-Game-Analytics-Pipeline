# tests/test_synthetic.py

import numpy as np
import pandas as pd
import pytest

# ---- Config (tek noktadan değiştir) ----
# User-level dataset columns you actually have:
REQUIRED_COLS = {
    "userid",  # NOTE: user_id değil
    "version",
    "session_count",
    "retention_1",
    "retention_7",
    "acquisition_channel",
    "country",
    "platform",
    "purchase",
    "CAC",
    "revenue",
    "ROI",
}

CHANNEL_WHITELIST = {"Instagram", "TikTok", "Organic", "Facebook"}

TARGET_COUNTRY_SHARES = {"USA": 0.7727, "Mexico": 0.1364, "Brazil": 0.0909}
TARGET_PLATFORM_SHARES = {"App Store": 0.25, "Google Play": 0.75}
PLATFORM_REVENUE_SHARE = {"App Store": 0.7429, "Google Play": 0.2571}

TOL = 0.02  # makul tolerans


@pytest.fixture(scope="module")
def df():
    # Tercihen Parquet (hızlı/şemasal), yoksa CSV fallback
    try:
        return pd.read_parquet("data/processed/events.parquet")
    except Exception:
        return pd.read_csv("data/processed/clean_data.csv", low_memory=False)


def test_required_columns(df):
    missing = REQUIRED_COLS - set(df.columns)
    assert not missing, f"Missing columns: {missing}"


def test_no_nan_inf_and_nonnegatives(df):
    assert np.isfinite(df["revenue"]).all(), "revenue has NaN/Inf"
    assert np.isfinite(df["ROI"]).all(), "ROI has NaN/Inf"
    assert (df["CAC"] >= 0).all(), "negative CAC"
    assert (df["revenue"] >= 0).all(), "negative revenue"


def test_channel_whitelist(df):
    unknown = set(df["acquisition_channel"].dropna().unique()) - CHANNEL_WHITELIST
    assert not unknown, f"Unexpected acquisition_channel values: {sorted(unknown)}"


def test_country_distribution(df):
    shares = df["country"].value_counts(normalize=True)
    for k, target in TARGET_COUNTRY_SHARES.items():
        assert k in shares.index, f"country '{k}' missing in data"
        assert (
            abs(shares[k] - target) < TOL
        ), f"{k} share {shares[k]:.4f} vs target {target:.4f}"


def test_platform_distribution(df):
    shares = df["platform"].value_counts(normalize=True)
    for k, target in TARGET_PLATFORM_SHARES.items():
        assert k in shares.index, f"platform '{k}' missing in data"
        assert (
            abs(shares[k] - target) < TOL
        ), f"{k} share {shares[k]:.4f} vs target {target:.4f}"


def test_platform_revenue_share(df):
    rev = df.groupby("platform")["revenue"].sum()
    shares = rev / rev.sum()
    for k, target in {"App Store": 0.7429, "Google Play": 0.2571}.items():
        assert k in shares.index, f"platform '{k}' missing for revenue share"
        assert (
            abs(shares[k] - target) < TOL
        ), f"{k} revenue share {shares[k]:.4f} vs target {target:.4f}"


def test_user_level_roi_formula(df):
    """
    User-level ROI genellikle (revenue - CAC) / CAC şeklindedir.
    (Toplam ROI için: sum(revenue) ve sum(CAC) ile aynı formül.)
    """
    denom = df["CAC"].replace(0, np.nan)
    roi_calc = (df["revenue"] - df["CAC"]) / denom
    # Yalnızca tanımlı (CAC>0) gözlemlerle karşılaştır
    mask = denom.notna() & df["ROI"].notna()
    diff = (df.loc[mask, "ROI"] - roi_calc[mask]).abs()
    assert diff.fillna(0).lt(1e-6).all(), "ROI formula mismatch beyond tolerance"


@pytest.mark.skip(reason="User-level schema; enable when event-level available.")
def test_event_date_dtype_and_range(df):
    # Eğer ileride event-level veriye geçersen bu testi aktif edersin.
    assert "event_date" in df.columns and "event_name" in df.columns
