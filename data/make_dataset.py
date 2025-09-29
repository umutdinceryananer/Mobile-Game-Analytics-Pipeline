# mobile_game_analytics_pipeline/data/make_dataset.py

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from dotenv import load_dotenv

# --------------------------------------------------------------------------------------
# Environment & paths
# --------------------------------------------------------------------------------------
load_dotenv()

PROJ_ROOT = Path(__file__).resolve().parents[1]


def _resolve_path(path_value: str | None, default: Path) -> Path:
    path = Path(path_value) if path_value else default
    if not path.is_absolute():
        path = (PROJ_ROOT / path).resolve()
    return path


RAW_DIR = _resolve_path(os.getenv("DATA_RAW_DIR"), PROJ_ROOT / "data" / "raw")
PROCESSED_DIR = _resolve_path(
    os.getenv("DATA_PROCESSED_DIR"), PROJ_ROOT / "data" / "processed"
)
CONFIG_PATH = _resolve_path(
    os.getenv("SYNTH_CONFIG"), PROJ_ROOT / "data" / "config" / "synthetic.yaml"
)


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------


def _assign_categorical(
    ids: pd.Series,
    labels: list[str],
    probs: list[float],
    mode: str,
    rng: np.random.Generator,
    salt: str = "",
) -> pd.Series:
    labels = [str(label).strip() for label in labels]
    probs_arr = np.asarray(probs, dtype=float)
    if probs_arr.sum() <= 0:
        raise ValueError("Probabilities must sum to a positive value.")
    probs_arr = probs_arr / probs_arr.sum()

    if mode == "hash":
        cumulative = probs_arr.cumsum()
        denominator = 16 ** len(hashlib.sha256().hexdigest())

        def pick_label(value: object) -> str:
            key = f"{value}{salt}"
            h_val = hashlib.sha256(key.encode()).hexdigest()
            fraction = int(h_val, 16) / denominator
            if fraction >= 1.0:
                fraction = np.nextafter(1.0, 0.0)
            idx = int(np.searchsorted(cumulative, fraction, side="right"))
            if idx >= len(labels):
                idx = len(labels) - 1
            return labels[idx]

        assigned = ids.apply(pick_label)
    else:
        assigned = pd.Series(
            rng.choice(labels, size=len(ids), p=probs_arr), index=ids.index
        )

    return assigned.astype(str).str.strip()


def _sigmoid(x: np.ndarray | float) -> np.ndarray | float:
    return 1.0 / (1.0 + np.exp(-x))


def _lift_from_mapping(series: pd.Series, mapping: dict | None) -> np.ndarray:
    if not mapping:
        return np.zeros(len(series), dtype=float)
    return series.map(mapping).fillna(0.0).to_numpy(dtype=float)


def _apply_retention_enrichment(
    df: pd.DataFrame, cfg: dict, rng: np.random.Generator
) -> pd.DataFrame:
    # Generate retention_1 and retention_7 flags using logistic-style models.
    d1_cfg = cfg.get("d1") or {}
    d7_cfg = cfg.get("d7") or {}

    if d1_cfg:
        log_sessions = np.log1p(df["session_count"].fillna(0).astype(float))
        purchase = df.get("purchase", pd.Series(0, index=df.index)).astype(float)

        logit = float(d1_cfg.get("logit_intercept", -0.5))
        logit += float(d1_cfg.get("session_log1p_weight", 0.2)) * log_sessions
        logit += float(d1_cfg.get("purchase_weight", 0.8)) * purchase
        logit += _lift_from_mapping(
            df["acquisition_channel"], d1_cfg.get("channel_lift")
        )
        logit += _lift_from_mapping(df["platform"], d1_cfg.get("platform_lift"))
        if "version" in df.columns:
            logit += _lift_from_mapping(df["version"], d1_cfg.get("version_lift"))

        noise_std = float(d1_cfg.get("noise_std", 0.0))
        if noise_std > 0:
            logit += rng.normal(0.0, noise_std, size=len(df))

        prob = _sigmoid(logit)
        prob = np.clip(
            prob,
            float(d1_cfg.get("min_rate", 0.05)),
            float(d1_cfg.get("max_rate", 0.95)),
        )
        df["retention_1"] = rng.binomial(1, prob, size=len(df))

    if d7_cfg:
        if "retention_1" not in df.columns:
            raise ValueError("retention_1 column required before computing retention_7")

        log_sessions = np.log1p(df["session_count"].fillna(0).astype(float))
        log_revenue = np.log1p(df.get("revenue", 0).clip(lower=0).astype(float))
        d1_values = df["retention_1"].astype(float)

        logit = float(d7_cfg.get("logit_intercept", -1.0))
        logit += float(d7_cfg.get("d1_weight", 1.0)) * d1_values
        logit += float(d7_cfg.get("session_log1p_weight", 0.1)) * log_sessions
        logit += float(d7_cfg.get("revenue_log1p_weight", 0.05)) * log_revenue
        logit += _lift_from_mapping(
            df["acquisition_channel"], d7_cfg.get("channel_lift")
        )
        logit += _lift_from_mapping(df["platform"], d7_cfg.get("platform_lift"))
        if "version" in df.columns:
            logit += _lift_from_mapping(df["version"], d7_cfg.get("version_lift"))

        noise_std = float(d7_cfg.get("noise_std", 0.0))
        if noise_std > 0:
            logit += rng.normal(0.0, noise_std, size=len(df))

        prob = _sigmoid(logit)
        prob = np.clip(
            prob,
            float(d7_cfg.get("min_rate", 0.02)),
            float(d7_cfg.get("max_rate", 0.8)),
        )
        df["retention_7"] = rng.binomial(1, prob, size=len(df))

    return df


def add_synthetic_columns(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """
    Adds acquisition_channel, CAC, revenue, ROI, country, and platform columns in a
    deterministic, config-driven way.
    Assumptions for Cookie Cats-like data:
      - user id column is 'userid'
      - session count proxy can be 'session_count' or 'sum_gamerounds'
    """
    rng = np.random.default_rng(cfg.get("seed", 42))

    # -----------------------------
    # 0) Normalize critical columns
    # -----------------------------
    # user id
    if "userid" not in df.columns:
        # try common fallbacks
        id_candidates = ["user_id", "userId", "uid", "player_id"]
        found = [c for c in id_candidates if c in df.columns]
        if not found:
            raise ValueError(
                "userid column needed (or one of user_id/userId/uid/player_id)."
            )
        df = df.rename(columns={found[0]: "userid"})

    # session_count (use sum_gamerounds if present)
    if "session_count" not in df.columns:
        if "sum_gamerounds" in df.columns:
            df = df.rename(columns={"sum_gamerounds": "session_count"})
        else:
            df["session_count"] = 0

    # --------------------------------
    # 1) acquisition_channel per config
    # --------------------------------
    ch_cfg = cfg["acquisition_channel"]
    ch_labels = ch_cfg["labels"]
    ch_probs = ch_cfg["probs"]
    assign_mode = ch_cfg.get("assign_mode", "hash")

    df["acquisition_channel"] = _assign_categorical(
        df["userid"],
        labels=ch_labels,
        probs=ch_probs,
        mode=assign_mode,
        rng=rng,
        salt="::acquisition_channel",
    )

    # --------------------------------
    # 1a) country per config
    # --------------------------------
    country_cfg = cfg.get("geo", {}).get("country")
    if not country_cfg:
        raise ValueError("geo.country config section is required.")

    country_labels = country_cfg["labels"]
    country_probs = country_cfg["probs"]
    country_assign_mode = country_cfg.get("assign_mode", "hash")

    df["country"] = _assign_categorical(
        df["userid"],
        labels=country_labels,
        probs=country_probs,
        mode=country_assign_mode,
        rng=rng,
        salt="::country",
    )

    # --------------------------------
    # 1b) platform per config
    # --------------------------------
    platform_cfg = cfg.get("platform")
    if not platform_cfg:
        raise ValueError("platform config section is required.")

    platform_labels = platform_cfg["labels"]
    platform_probs = platform_cfg["probs"]
    platform_assign_mode = platform_cfg.get("assign_mode", "hash")

    df["platform"] = _assign_categorical(
        df["userid"],
        labels=platform_labels,
        probs=platform_probs,
        mode=platform_assign_mode,
        rng=rng,
        salt="::platform",
    )

    # --------------------------------
    # 1c) Purchase probability enrichment
    # --------------------------------
    if "purchase" not in df.columns:
        purchase_cfg = cfg.get("purchase", {})
        base_rate = float(purchase_cfg.get("base_rate", 0.02))
        session_weight = float(purchase_cfg.get("session_log1p_weight", 0.03))
        session_center = float(purchase_cfg.get("session_center", 10.0))
        min_rate = float(purchase_cfg.get("min_rate", 0.0))
        max_rate = float(purchase_cfg.get("max_rate", 0.5))
        random_jitter = float(purchase_cfg.get("random_jitter", 0.0))
        channel_lift = purchase_cfg.get("channel_lift", {}) or {}
        platform_lift = purchase_cfg.get("platform_lift", {}) or {}
        version_lift = purchase_cfg.get("version_lift", {}) or {}

        session_values = df["session_count"].fillna(0).to_numpy(dtype=float)
        session_center = session_center if session_center > 0 else 1.0
        log_component = np.log1p(session_values / session_center)

        prob = base_rate + session_weight * log_component

        if channel_lift:
            prob += (
                df["acquisition_channel"]
                .map(channel_lift)
                .fillna(0.0)
                .to_numpy(dtype=float)
            )
        if platform_lift:
            prob += df["platform"].map(platform_lift).fillna(0.0).to_numpy(dtype=float)
        if version_lift and "version" in df.columns:
            prob += df["version"].map(version_lift).fillna(0.0).to_numpy(dtype=float)

        if random_jitter > 0:
            prob += rng.normal(0.0, random_jitter, size=len(df))

        prob = np.clip(prob, min_rate, max_rate)
        df["purchase"] = rng.binomial(1, prob)

    normalized_platform_weights = None
    platform_weights_cfg = platform_cfg.get("revenue_weights")
    target_share_cfg = platform_cfg.get("revenue_target_shares")

    if platform_weights_cfg and target_share_cfg:
        raise ValueError(
            "Use either platform.revenue_weights or platform.revenue_target_shares, not both."
        )

    if target_share_cfg:
        shares = {}
        for label in platform_labels:
            if label not in target_share_cfg:
                raise ValueError(
                    f"Revenue target share missing for platform '{label}'."
                )
            share = float(target_share_cfg[label])
            if share <= 0:
                raise ValueError("Revenue target shares must be positive.")
            shares[label] = share
        total_share = sum(shares.values())
        if total_share <= 0:
            raise ValueError("Revenue target shares must sum to a positive value.")

        shares = {label: share / total_share for label, share in shares.items()}

        normalized_platform_weights = {}
        for label, prob in zip(platform_labels, platform_probs):
            prob_value = float(prob)
            if prob_value <= 0:
                raise ValueError(
                    "Platform probabilities must be positive when revenue shares are applied."
                )
            normalized_platform_weights[label] = shares[label] / prob_value
    elif platform_weights_cfg:
        weights = {}
        for label in platform_labels:
            if label not in platform_weights_cfg:
                raise ValueError(
                    f"Platform revenue weight missing for label '{label}'."
                )
            weight = float(platform_weights_cfg[label])
            if weight <= 0:
                raise ValueError("Platform revenue weights must be positive.")
            weights[label] = weight
        total_weight = sum(weights.values())
        if total_weight <= 0:
            raise ValueError("Platform revenue weights must sum to a positive value.")

        normalized_platform_weights = {}
        for label, prob in zip(platform_labels, platform_probs):
            prob_value = float(prob)
            if prob_value <= 0:
                raise ValueError(
                    "Platform probabilities must be positive when revenue weights are applied."
                )
            normalized_platform_weights[label] = (
                weights[label] / total_weight
            ) / prob_value
    # --------------------------------
    # 2) CAC mapping (early validation)
    # --------------------------------
    cac_map = cfg["cac"]
    labels_set = set(df["acquisition_channel"].unique())
    cac_keys = set(cac_map.keys())
    missing = labels_set - cac_keys
    if missing:
        raise ValueError(
            f"CAC mapping is missing for channel(s): {sorted(missing)}.\n"
            f"Please add them under 'cac:' in your config. Seen labels: {sorted(labels_set)}"
        )

    df["CAC"] = df["acquisition_channel"].map(cac_map).astype(float)

    # -----------------------------
    # 3) Revenue (IAP + ad revenue)
    # -----------------------------
    rev_cfg = cfg["revenue"]
    shape = rev_cfg["iap_gamma_shape"]
    scale = rev_cfg["iap_gamma_scale"]
    ad_per_sess = rev_cfg["ad_revenue_per_session"]
    use_user_seed = rev_cfg.get("use_user_seed", True)

    if use_user_seed:

        def user_gamma(uid):
            s = int(hashlib.sha256(str(uid).encode()).hexdigest()[:8], 16)
            r = np.random.default_rng(s)
            return r.gamma(shape, scale)

        iap = df.apply(
            lambda r: user_gamma(r["userid"]) if r["purchase"] == 1 else 0.0,
            axis=1,
        )
    else:
        iap = np.where(df["purchase"] == 1, rng.gamma(shape, scale, size=len(df)), 0.0)

    ad_base = np.log1p(df["session_count"].fillna(0).astype(float))
    ad_rev = ad_base * ad_per_sess

    df["revenue"] = (pd.Series(iap, index=df.index) + ad_rev).clip(lower=0.0)
    if normalized_platform_weights:
        multipliers = df["platform"].map(normalized_platform_weights).astype(float)
        df["revenue"] = df["revenue"] * multipliers

    retention_cfg = cfg.get("retention")
    if retention_cfg:
        df = _apply_retention_enrichment(df, retention_cfg, rng)

    # -----------------------------
    # 4) ROI (safe division)
    # -----------------------------
    denom = df["CAC"].replace(0.0, np.nan)  # avoid inf
    df["ROI"] = ((df["revenue"] - df["CAC"]) / denom).fillna(0.0)

    return df


def validate_processed(df: pd.DataFrame, cfg: dict):
    """Lightweight schema/logic checks with actionable messages."""
    assert "acquisition_channel" in df, "acquisition_channel column is missing."
    assert {"CAC", "revenue", "ROI"}.issubset(
        df.columns
    ), "CAC/revenue/ROI columns are missing."
    assert "country" in df, "country column is missing."
    assert "platform" in df, "platform column is missing."

    if df["CAC"].isna().any():
        bad = df.loc[df["CAC"].isna(), "acquisition_channel"].value_counts()
        raise AssertionError(
            f"CAC is NaN for some rows. Offending channels:\n{bad.head(10)}"
        )

    if (df["CAC"] < 0).any():
        bad = df.loc[df["CAC"] < 0, ["acquisition_channel", "CAC"]].head(10)
        raise AssertionError(f"Negative CAC values detected (first 10 rows):\n{bad}")

    if (df["revenue"] < 0).any():
        raise AssertionError("Negative revenue detected.")

    if not np.isfinite(df["ROI"]).all():
        raise AssertionError("Non-finite ROI detected (check CAC=0 handling).")

    platform_cfg = cfg.get("platform", {})
    revenue_weights = platform_cfg.get("revenue_weights")
    target_shares = platform_cfg.get("revenue_target_shares")
    if revenue_weights and target_shares:
        raise AssertionError(
            "Define either platform.revenue_weights or platform.revenue_target_shares, not both."
        )
    if revenue_weights:
        labels = platform_cfg.get("labels", [])
        missing_weights = [label for label in labels if label not in revenue_weights]
        if missing_weights:
            raise AssertionError(
                "Platform revenue weights missing for: " f"{sorted(missing_weights)}"
            )
        non_positive = {
            label: revenue_weights[label]
            for label in labels
            if float(revenue_weights[label]) <= 0
        }
        if non_positive:
            raise AssertionError(
                "Platform revenue weights must be positive. "
                f"Found non-positive values: {non_positive}"
            )
    if target_shares:
        labels = platform_cfg.get("labels", [])
        missing_shares = [label for label in labels if label not in target_shares]
        if missing_shares:
            raise AssertionError(
                "Platform revenue target shares missing for: "
                f"{sorted(missing_shares)}"
            )
        non_positive = {
            label: target_shares[label]
            for label in labels
            if float(target_shares[label]) <= 0
        }
        if non_positive:
            raise AssertionError(
                "Platform revenue target shares must be positive. "
                f"Found non-positive values: {non_positive}"
            )
        total_share = sum(float(target_shares[label]) for label in labels)
        if total_share <= 0:
            raise AssertionError("Revenue target shares must sum to a positive value.")


def save_meta(meta_path: Path, cfg: dict, input_name: str) -> None:
    """Write a small JSON with provenance and config hash."""
    meta = {
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "config_hash": hashlib.sha256(yaml.dump(cfg).encode()).hexdigest(),
        "raw_source": input_name,
        "notes": "Added acquisition_channel, CAC, revenue, ROI, country, platform (synthetic).",
    }
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # Load config
    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Load raw data
    raw_file = RAW_DIR / "cookie_cats.csv"
    df = pd.read_csv(raw_file)

    # Basic datetime cast (if present)
    if "install_date" in df.columns:
        df["install_date"] = pd.to_datetime(df["install_date"], errors="coerce")

    # Add synthetic columns
    df = add_synthetic_columns(df, cfg)

    # Validate
    validate_processed(df, cfg)

    # Save outputs
    out_path = PROCESSED_DIR / "events.parquet"
    df.to_parquet(out_path, index=False)
    csv_path = PROCESSED_DIR / "clean_data.csv"
    df.to_csv(csv_path, index=False)
    save_meta(
        PROCESSED_DIR / "_meta.synthetic.json",
        cfg,
        raw_file.name,
    )
    print(f"Processed saved -> {out_path}")
    print(f"CSV saved -> {csv_path}")


if __name__ == "__main__":
    main()
