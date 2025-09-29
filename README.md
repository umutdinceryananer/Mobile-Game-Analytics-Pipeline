# Mobile Game Analytics Pipeline

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

## Executive Summary

This repository demonstrates a comprehensive analytics workflow for marketing performance analysis, covering user funnel optimization, retention cohorts, trend analysis, and predictive modeling. The project showcases modern data analysis techniques including SQL-driven reporting, dashboard development, data integrity monitoring, and ROI forecasting to extract valuable insights from marketing data and drive strategic decision-making.

### What this project demonstrates?

* Actionable analytics: Clear funnel and ROI/ROAS insights that translate into concrete recommendations for marketing and product teams.
* Cohort retention: D1/D7 cohort analysis by acquisition channel and platform, highlighting where to invest or iterate on creatives/targeting.
* Forecasting/modeling: A small but production-minded forecasting or churn prototype to illustrate predictive capability.
* Reporting craft: Reproducible SQL, cleaned notebooks, and exportable figures suitable for dashboards and executive communication.

### Data & Scope

* Dataset: Cookie Cats (Kaggle) user-level/mobile game telemetry, enriched with synthetic user acquisition attributes (e.g., acquisition_channel, CAC/ad spend, revenue fields) to enable ROI/ROAS analysis.
* KPIs: D1/D7 retention, conversion funnel step rates, ROI/ROAS by channel, ARPDAU/Revenue trends (optional), and a focused prediction target (e.g., D7 or revenue proxy).
* Decisions supported: Budget reallocation across channels, creative/testing priorities, onboarding/FTUE optimizations, and retention-oriented product bets.

### Deliverables

#### Notebooks

* 0.9-SQL-Validation-and-Samples.ipynb → sql_analysis.ipynb
    * Runs 3 canonical queries (daily installs, funnel step rates, ROI/ROAS) and validates notebook vs. SQL outputs.

* 1.0-EDA-and-Funnel.ipynb → eda.ipynb
    * Defines the funnel (install → onboarding → D1 → purchase), produces the first KPI table and a funnel chart.

* 2.0-ROI-and-ROAS-by-Channel.ipynb → roas_analysis.ipynb
    * Computes ROI/ROAS by acquisition channel (+optional platform), exports ranked tables and visuals, ends with 3 actionable recommendations.

* 2.1-Retention-Cohorts.ipynb
    * D1/D7 cohort heatmaps and top/bottom channel lists, with short commentary on implications.

* 3.0-Forecast-or-Churn.ipynb (new, compact)
    * Option A: Revenue/ARPDAU forecasting (Prophet/SARIMAX) with backtests (SMAPE/MAPE).
    * Option B: Churn classification (LogReg + XGBoost/LightGBM) with AUC/PR-AUC, calibration, and Top-K lift.

--------

## Usage & Reproducibility

This section explains **how to run the project end‑to‑end**, how SQL/Notebooks are wired, and where artifacts are exported. It is written to be **copy‑paste runnable** on a fresh machine.

---

### Prerequisites

* Python **3.10+**
* `pip` (or `pipx` if preferred)
* Git
* (Optional) **DuckDB** is installed via `requirements.txt` and used to run SQL over CSV/Parquet without a DB server.

---

### Quickstart

**Create and activate a virtual environment**

```bash
# macOS/Linux
python -m venv .venv && source .venv/bin/activate

# Windows (PowerShell)
python -m venv .venv; .\.venv\Scripts\Activate.ps1
```

**Install dependencies**

```bash
pip install -r requirements.txt
```

---

### Project Layout (runnable paths)

```
mobile_game_analytics_pipeline/   # checks, features, modeling helpers
notebooks/                        # 0.9 (SQL) • 1.0 (EDA/Funnel) • 2.0 (ROI/ROAS) • 2.1 (Cohorts) • 3.0 (Forecast/Churn)
references/sql/                   # canonical SQL files: daily_installs, funnel_step_rates, roi_by_channel
reports/                          # figures/, tables/, executive_summary.md
data/
  ├─ raw/
  ├─ interim/
  └─ processed/                  # e.g., clean_data.csv or events.parquet
```

---

### Data Inputs

* Primary input: `data/processed/clean_data.csv` (or `data/processed/events.parquet`).
* Notebook 0.9 (SQL) automatically **creates a DuckDB view `events`** from one of these files.
* If neither file is present, notebooks will raise a clear error.

---

### Running Order

Run notebooks **top‑down** in this order:

1. **`0.9-SQL-Validation-and-Samples.ipynb`**
   Loads SQL from `references/sql/` and executes via DuckDB. Exports validation tables to `reports/tables/`.

2. **`1.0-EDA-and-Funnel.ipynb`**
   Defines the funnel and exports `funnel.csv` and `funnel.png`.

3. **`2.0-ROI-and-ROAS-by-Channel.ipynb`**
   Computes ROI/ROAS by channel; exports `roi_by_channel.csv` and `roi_by_channel.png`.

4. **`2.1-Retention-Cohorts.ipynb`**
   Produces D1/D7 cohort heatmaps; exports `retention_by_channel.csv` and `retention_heatmap.png`.

5. **`3.0-Forecast-or-Churn.ipynb`**
   Either forecasting (e.g., Prophet/SARIMAX) or churn classification (LogReg + XGBoost/LightGBM). Exports `model_metrics.json` and one key figure (`forecast_plot.png` or `roc_pr_curves.png`).

---

### SQL Usage (Externalized & Reused)

SQL is stored in **`references/sql/`** and loaded from notebooks at runtime.

* `references/sql/daily_installs.sql`
* `references/sql/funnel_step_rates.sql`
* `references/sql/roi_by_channel.sql`

---

### Exports (Artifacts)

All notebooks **export tables and figures** to a consistent location:

* **Tables:** `reports/tables/`
  * `funnel.csv`, `funnel_long.csv`, `roi_by_channel.csv`, `roi_by_channel_long.csv`, `retention_by_channel.csv`, `retention_cohort_by_version.csv`

* **Figures:** `reports/figures/`
  * `funnel.png`, `roi_by_channel.png`, `retention_heatmap.png`, `forecast_plot.png` or `roc_pr_curves.png`

* **Summary:** `reports/executive_summary.md` (curated manually from notebook findings)

---

### Data Quality Checks (Optional but Recommended)

In here you must run test_synthetic.py in order to make sure quality of the clean_data
---

### Reproducibility Notes

* **Determinism:** set seeds for model training and sampling (e.g., `numpy`, `random`, model libraries).
* **No Leakage:** split by **time** for validation (rolling or holdout) and build features only from the past window.
* **Versioning:** tag releases when major deliverables change (e.g., `v0.1` MVP, `v0.2` cohorts deep dive, `v0.3` modeling).
* **Environment:** keep `requirements.txt` up to date; pin critical libs if needed for a clean re‑run.

---

## Part 3 — Results, Visuals & Next Steps

This section summarizes key insights, embeds exported figures, and outlines limitations and next actions. Replace the placeholder metrics below once the latest notebooks are executed.

---

### Results Overview (Key Findings)

**Acquisition & Funnel**

* *Conversion funnel:* Install -> Onboarding -> D1 return -> Purchase. Current run shows:
  * **Install -> Onboarding:** ~`95.6%` (baseline FTUE completion)
  * **Onboarding -> D1:** ~`47.6%` (early retention health)
  * **D1 -> Purchase:** ~`16.8%` (monetization gate)
* *Action:* Focus UX experiments on the largest drop (e.g., onboarding), and validate with an A/B test.

**ROI/ROAS by Channel**

* **Top channel:** `Organic` delivers **ROAS ~`1.85`**, **ROI ~`+85%`**, and is the only source above break-even.
* **Underperformers:** Paid UA (e.g., `TikTok` ROAS ~`0.34`, `Instagram` ROAS ~`0.23`, `Facebook` ROAS ~`0.19`) remains below break-even.
* *Action:* Reallocate **+10-20% UA budget** to top channels; test creative iteration for low-ROAS channels before further spend.

**Retention Cohorts (D1/D7)**

* **D1 retention:** overall average ~`45.5%`; `Organic` leads at ~`47.7%`.
* **D7 retention:** ~`33.3%` of installs return on day 7, and `Organic` keeps ~`73.8%` of its D1 returners through D7.
* *Action:* Prioritize **best-quality sources** (high D7) for long-term value; refine onboarding for channels with high D1 but weak D7.

**Prediction/Forecast (optional path)**

*Churn model (LogReg + XGBoost):* ROC-AUC ≈ `0.607`, PR-AUC ≈ `0.580`, accuracy ≈ `0.602`; top 10% risk bucket captures ~`78%` of churn (lift ≈ `1.17×`).
*Key segments:* Highest churn risk clusters in `Facebook` and `TikTok` installs on `Google Play`; see `reports/tables/churn_risk_segments.csv` for channel/platform drill-down.
*Artifacts:* `reports/tables/backtest_scores.csv`, `reports/tables/model_metrics.json`, `reports/tables/churn_risk_segments.csv`, `reports/figures/roc_pr_curves.png`.

> Scores are measured on the synthetic demo dataset; expect lower performance on production data.

> Record the finalized numbers in `reports/executive_summary.md` as a single‑page narrative for reviewers.

---

### Visuals (Exported Artifacts)

> All figures are generated by notebooks and exported under `reports/figures/`. Update after re‑running notebooks.

* **Funnel**
  `![Funnel](reports/figures/funnel.png)`

* **ROI by Channel**
  `![ROI by Channel](reports/figures/roi_by_channel.png)`

* **Retention Cohort Heatmap (D1/D7)**
  `![Retention Cohort Heatmap](reports/figures/retention_heatmap.png)`

* **Modeling / Forecasting**
  `![Churn ROC/PR](reports/figures/roc_pr_curves.png)`  *or*  `![Forecast](reports/figures/forecast_plot.png)`

---

### Dashboard (Tableau)

* **Recommended tabs:** Funnel - ROI/ROAS - Retention (D1/D7) - Churn/Forecast
* **Export:** Add 2-3 screenshots to `reports/figures/` using the pattern `dashboard_funnel.png`, `dashboard_retention.png`, `dashboard_roi.png`, `dashboard_modeling.png`.
* **(If published)** Include a share link here once available.

<div class='tableauPlaceholder' id='viz1759097764622' style='position: relative'><noscript><a href='#'><img alt='Mobile Game UA Performance Overview' src='https://public.tableau.com/static/images/Mo/MobileGameUAPerformanceOverview/Dashboard1/1.png' style='border: none' /></a></noscript><object class='tableauViz' style='display:none;'>
  <param name='host_url' value='https%3A%2F%2Fpublic.tableau.com%2F' />
  <param name='embed_code_version' value='3' />
  <param name='site_root' value='' />
  <param name='name' value='MobileGameUAPerformanceOverview/Dashboard1' />
  <param name='tabs' value='no' />
  <param name='toolbar' value='yes' />
  <param name='static_image' value='https://public.tableau.com/static/images/Mo/MobileGameUAPerformanceOverview/Dashboard1/1.png' />
  <param name='animate_transition' value='yes' />
  <param name='display_static_image' value='yes' />
  <param name='display_spinner' value='yes' />
  <param name='display_overlay' value='yes' />
  <param name='display_count' value='yes' />
  <param name='language' value='en-US' />
  <param name='filter' value='publish=yes' />
</object></div>

---

### Limitations & Assumptions

* **Synthetic enrichment:** User acquisition fields (e.g., `acquisition_channel`, CAC/ad spend) are enriched and may not reflect production distributions.
* **Schema/coverage:** Missing events or short time windows can bias retention and ROI estimates; metrics are indicative.
* **Attribution simplification:** Channel attribution is 1‑touch in this demo; multi‑touch or MMM would alter ROI interpretation.
* **Model scope:** Forecasts/classifiers are compact prototypes (no hyper‑intensive tuning). Calibration and backtesting are included to keep results honest.

---

### License & Credits

* Code is released under **MIT License** (see `LICENSE`).
* Dataset: Cookie Cats (Kaggle) — used here for educational/demo purposes with synthetic UA enrichment.

---
