# mobile_game_analytics_pipeline/data/make_dataset.py

import hashlib
import json
import os
from datetime import datetime

import numpy as np
import pandas as pd
import yaml
from dotenv import load_dotenv

# --------------------------------------------------------------------------------------
# Environment & paths
# --------------------------------------------------------------------------------------
load_dotenv()
RAW_DIR = os.getenv("DATA_RAW_DIR", "data/raw")
PROCESSED_DIR = os.getenv("DATA_PROCESSED_DIR", "data/processed")
# You preferred keeping config under data/, so default there:
CONFIG_PATH = os.getenv("SYNTH_CONFIG", "data/config/synthetic.yaml")


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------
def stable_bucket(key: str, buckets: int) -> int:
    """Deterministic bucket assignment for a given key (SHA-256 based)."""
    h = hashlib.sha256(str(key).encode()).hexdigest()
    return int(h[:8], 16) % buckets


def add_synthetic_columns(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """
    Adds acquisition_channel, CAC, revenue, ROI columns in a deterministic, config-driven way.
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

    # purchase flag (if missing, synthesize a reasonable proxy)
    if "purchase" not in df.columns:
        sc = df["session_count"].fillna(0).astype(float)
        # Higher sessions -> higher purchase prob, cap at 35%
        p = (sc / (sc.max() + 1.0)).clip(0.0, 0.35)
        df["purchase"] = rng.binomial(1, p)

    # --------------------------------
    # 1) acquisition_channel per config
    # --------------------------------
    ch_labels = cfg["acquisition_channel"]["labels"]
    ch_probs = cfg["acquisition_channel"]["probs"]
    assign_mode = cfg["acquisition_channel"].get("assign_mode", "hash")

    if assign_mode == "hash":
        buckets = len(ch_labels)
        idx = df["userid"].apply(lambda x: stable_bucket(x, buckets))
        df["acquisition_channel"] = idx.apply(lambda i: ch_labels[i])
    else:  # "random" (deterministic via global seed)
        df["acquisition_channel"] = rng.choice(ch_labels, size=len(df), p=ch_probs)

    # normalize channel strings
    df["acquisition_channel"] = df["acquisition_channel"].astype(str).str.strip()

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
    shape, scale = rev_cfg["iap_gamma_shape"], rev_cfg["iap_gamma_scale"]
    ad_per_sess = rev_cfg["ad_revenue_per_session"]
    use_user_seed = rev_cfg.get("use_user_seed", True)

    # IAP revenue (only for purchase=1)
    if use_user_seed:
        # per-user deterministic gamma
        def user_gamma(uid):
            s = int(hashlib.sha256(str(uid).encode()).hexdigest()[:8], 16)
            r = np.random.default_rng(s)
            return r.gamma(shape, scale)

        iap = df.apply(
            lambda r: user_gamma(r["userid"]) if r["purchase"] == 1 else 0.0, axis=1
        )
    else:
        iap = np.where(df["purchase"] == 1, rng.gamma(shape, scale, size=len(df)), 0.0)

    # Ad revenue (session-based; log transform helps with heavy tails)
    ad_base = np.log1p(df["session_count"].fillna(0).astype(float))
    ad_rev = ad_base * ad_per_sess

    df["revenue"] = (pd.Series(iap, index=df.index) + ad_rev).clip(lower=0.0)

    # -----------------------------
    # 4) ROI (safe division)
    # -----------------------------
    denom = df["CAC"].replace(0.0, np.nan)  # avoid inf
    df["ROI"] = ((df["revenue"] - df["CAC"]) / denom).fillna(0.0)

    return df


def validate_processed(df: pd.DataFrame):
    """Lightweight schema/logic checks with actionable messages."""
    assert "acquisition_channel" in df, "acquisition_channel column is missing."
    assert {"CAC", "revenue", "ROI"}.issubset(
        df.columns
    ), "CAC/revenue/ROI columns are missing."

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


def save_meta(meta_path: str, cfg: dict, input_name: str):
    """Write a small JSON with provenance and config hash."""
    meta = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "config_hash": hashlib.sha256(yaml.dump(cfg).encode()).hexdigest(),
        "raw_source": input_name,
        "notes": "Added acquisition_channel, CAC, revenue, ROI (synthetic).",
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


def main():
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    # Load config
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Load raw data
    raw_file = os.path.join(RAW_DIR, "cookie_cats.csv")
    df = pd.read_csv(raw_file)

    # Basic datetime cast (if present)
    if "install_date" in df.columns:
        df["install_date"] = pd.to_datetime(df["install_date"], errors="coerce")

    # Add synthetic columns
    df = add_synthetic_columns(df, cfg)

    # Validate
    validate_processed(df)

    # Save outputs
    out_path = os.path.join(PROCESSED_DIR, "events.parquet")
    df.to_parquet(out_path, index=False)
    save_meta(
        os.path.join(PROCESSED_DIR, "_meta.synthetic.json"),
        cfg,
        os.path.basename(raw_file),
    )
    print(f"Processed saved -> {out_path}")


if __name__ == "__main__":
    main()
