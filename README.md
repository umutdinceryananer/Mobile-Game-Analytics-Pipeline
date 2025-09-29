# Mobile Game Analytics Pipeline

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![Status](https://img.shields.io/badge/status-active-success.svg)
![DuckDB](https://img.shields.io/badge/DuckDB-SQL-yellow?logo=duckdb&logoColor=white)

## Executive Summary

This repository demonstrates a comprehensive analytics workflow for marketing performance analysis, covering user funnel optimization, retention cohorts, trend analysis, and predictive modeling. The project showcases modern data analysis techniques including SQL-driven reporting, dashboard development, data integrity monitoring, and ROI forecasting to extract valuable insights from marketing data and drive strategic decision-making.

### What this project demonstrates?

* Actionable analytics: funnel, ROI/ROAS, retention, and churn signals tied directly to UA decisions.
* Cohort retention: D1/D7 cohort analysis by acquisition channel and platform highlights where to invest or iterate.
* Predictive modeling: churn-classification pipeline (LogReg + XGBoost) with documented lift and Tableau integration.
* Reporting craft: reproducible SQL, polished notebooks, and exportable figures/dashboard embeds.

### Latest Metrics (synthetic run)

* **Funnel:** Install -> Onboarding ~95.6%, Onboarding -> D1 ~47.6%, D1 -> Purchase ~16.8%.
* **Retention:** Overall D7 ~33.3%; Organic keeps ~73.8% of its D1 returners (D7 ~35.2%).
* **ROI:** Organic ROAS ~1.85 (ROI ~+85%); paid channels range 0.19-0.34 ROAS (ROI -66% to -81%).
* **Churn model:** ROC-AUC ~0.61, PR-AUC ~0.58, accuracy ~0.60; top 10% risk bucket captures ~78% of churn (lift ~1.17x).

### Data & Scope

* Dataset: [Cookie Cats (Kaggle)](https://www.kaggle.com/datasets/yufengsui/mobile-games-ab-testing) user-level/mobile game telemetry, enriched with synthetic user acquisition attributes (e.g., acquisition_channel, CAC/ad spend, revenue fields) to enable ROI/ROAS analysis.
* KPIs: D1/D7 retention, conversion funnel step rates, ROI/ROAS by channel, ARPDAU/Revenue trends (optional), and a focused prediction target (e.g., D7 or revenue proxy).
* Decisions supported: Budget reallocation across channels, creative/testing priorities, onboarding/FTUE optimizations, and retention-oriented product bets.

### Deliverables

#### Notebooks

* 0.9-SQL-Validation-and-Samples.ipynb -> sql_analysis.ipynb
    * Runs 3 canonical queries (daily installs, funnel step rates, ROI/ROAS) and validates notebook vs. SQL outputs.

* 1.0-EDA-and-Funnel.ipynb -> eda.ipynb
    * Defines the funnel (install -> onboarding -> D1 -> purchase), produces the first KPI table and a funnel chart.

* 2.0-ROI-and-ROAS-by-Channel.ipynb -> roas_analysis.ipynb
    * Computes ROI/ROAS by acquisition channel (+optional platform), exports ranked tables and visuals, ends with 3 actionable recommendations.

* 2.1-Retention-Cohorts.ipynb
    * D1/D7 cohort heatmaps and top/bottom channel lists, with short commentary on implications.

* 3.0-Churn-Model.ipynb
    * Builds the churn feature set, trains LogReg & XGBoost, records ROC/PR metrics, and exports risk segments + Tableau-ready artefacts.

--------

## Usage & Reproducibility

This section explains **how to run the project end'to'end**, how SQL/Notebooks are wired, and where artifacts are exported. It is written to be **copy'paste runnable** on a fresh machine.

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


### Command-line Workflow

After installing requirements, you can reproduce the pipeline without opening notebooks:

1. `make data` - rebuilds `clean_data.csv` and `events.parquet` via the Typer CLI wrapper.
2. `make features` - creates `features.csv` and `labels.csv` with the churn flag.
3. `make train` - trains the churn models, writes metrics to `reports/tables/` and stores the best pipeline under `models/churn_model.pkl`.
4. *(Optional)* `make predict` - uses the trained model to generate `predictions.csv` with churn probabilities.

Shortcut: `make pipeline` runs steps 1-3 sequentially.

Each rule calls `python -m mobile_game_analytics_pipeline.<command>` under the hood, so you can invoke them directly if you prefer finer control over paths.

If you dont have make and dont want to install it you can use given commands;

```
# When Virtual Environment is Active
python -m mobile_game_analytics_pipeline.dataset
python -m mobile_game_analytics_pipeline.features
python -m mobile_game_analytics_pipeline.modeling.train
python -m mobile_game_analytics_pipeline.modeling.predict  # opsiyonel

```

### Project Layout (runnable paths)

```
mobile_game_analytics_pipeline/
├─ mobile_game_analytics_pipeline/
│  ├─ __init__.py
│  ├─ config.py
│  ├─ dataset.py               # Typer command: rebuild synthetic data
│  ├─ features.py              # Typer command: create features & labels
│  ├─ modeling/
│  │  ├─ __init__.py
│  │  ├─ train.py              # Typer command: train churn model
│  │  └─ predict.py            # Typer command: generate predictions
│  └─ …
├─ data/
│  ├─ raw/
│  │  └─ cookie_cats.csv
│  ├─ processed/               # clean_data.csv, events.parquet, features.csv, labels.csv
│  ├─ config/
│  │  └─ synthetic.yaml
│  └─ make_dataset.py
├─ notebooks/
│  ├─ 0.9-SQL-Validation-and-Samples.ipynb
│  ├─ 1.0-EDA-and-Funnel.ipynb
│  ├─ 2.0-ROI-and-ROAS-by-Channel.ipynb
│  ├─ 2.1-Retention-Cohorts.ipynb
│  └─ 3.0-Churn-Model.ipynb
├─ references/
│  └─ sql/
├─ reports/
│  ├─ tables/
│  ├─ figures/
│  └─ executive_summary.md
├─ tests/
│  └─ test_synthetic.py
├─ Makefile
├─ requirements.txt
└─ README.md
```

---

### Data Inputs

* Primary input: `data/processed/clean_data.csv` (or `data/processed/events.parquet`).
* Notebook 0.9 (SQL) automatically **creates a DuckDB view `events`** from one of these files.
* If neither file is present, notebooks will raise a clear error.

---

### Running Order

Run notebooks **top'down** in this order:

1. **`0.9-SQL-Validation-and-Samples.ipynb`**
   Loads SQL from `references/sql/` and executes via DuckDB. Exports validation tables to `reports/tables/`.

2. **`1.0-EDA-and-Funnel.ipynb`**
   Defines the funnel and exports `funnel.csv` and `funnel.png`.

3. **`2.0-ROI-and-ROAS-by-Channel.ipynb`**
   Computes ROI/ROAS by channel; exports `roi_by_channel.csv` and `roi_by_channel.png`.

4. **`2.1-Retention-Cohorts.ipynb`**
   Produces D1/D7 cohort heatmaps; exports `retention_by_channel.csv` and `retention_heatmap.png`.

5. **`3.0-Forecast-or-Churn.ipynb`**
   Churn classification (LogReg + XGBoost/LightGBM). Exports `model_metrics.json` and one key figure (`roc_pr_curves.png`).

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

### Data Quality Checks & Testing

```bash
pytest -q
```

* `tests/test_synthetic.py` validates schema, channel/country/platform distributions, ROI formula, and retention ratios for the synthetic dataset.
* CLI smoke test: `make pipeline` (or run the Typer commands manually) regenerates data, features, and churn model artefacts end-to-end.

#### Enabling Pre-commit Hooks

This repository uses `pre-commit` hooks (formatting, lint checks, etc.). A fresh clone does not include the hook automatically; run the following once after setting up your environment:

---

### Reproducibility Notes

* **Determinism:** set seeds for model training and sampling (e.g., `numpy`, `random`, model libraries).
* **No Leakage:** split by **time** for validation (rolling or holdout) and build features only from the past window.
* **Versioning:** tag releases when major deliverables change (e.g., `v0.1` MVP, `v0.2` cohorts deep dive, `v0.3` modeling).
* **Environment:** keep `requirements.txt` up to date; pin critical libs if needed for a clean re'run.

---

## Part 3 " Results, Visuals & Next Steps

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

**Prediction/Forecast **

*Churn model (LogReg + XGBoost):* ROC-AUC  `0.607`, PR-AUC  `0.580`, accuracy  `0.602`; top 10% risk bucket captures ~`78%` of churn (lift  `1.17`).
*Key segments:* Highest churn risk clusters in `Facebook` and `TikTok` installs on `Google Play`; see `reports/tables/churn_risk_segments.csv` for channel/platform drill-down.
*Artifacts:* `reports/tables/backtest_scores.csv`, `reports/tables/model_metrics.json`, `reports/tables/churn_risk_segments.csv`, `reports/figures/roc_pr_curves.png`.

> Scores are measured on the synthetic demo dataset; expect lower performance on production data.

> Record the finalized numbers in `reports/executive_summary.md` as a single'page narrative for reviewers.

---

### Visuals (Exported Artifacts)

* **Funnel**
  <img width="1185" height="731" alt="funnel" src="https://github.com/user-attachments/assets/504a297a-5d47-4c4b-bf0d-dae14a6b4f10" />

* **ROI by Channel**
  <img width="1182" height="731" alt="roi_by_channel" src="https://github.com/user-attachments/assets/3dee4aca-f0bf-4e68-8d4c-1d2720f27bca" />

* **Retention Cohort Heatmap (D1/D7)**
  <img width="1408" height="761" alt="retention_heatmap" src="https://github.com/user-attachments/assets/dfa39f95-d226-4744-bcb6-5cc17447732c" />

* **Modeling **
  <img width="1784" height="731" alt="roc_pr_curves" src="https://github.com/user-attachments/assets/54a1dec0-2464-4f8d-b1b3-e62ac78406b8" />

---

### Dashboard (Tableau)
* [**Tableau Link**](https://public.tableau.com/views/MobileGameUAStory/TableauStory?:language=en-US&:sid=&:redirect=auth&:display_count=n&:origin=viz_share_link)

<img width="1415" height="1063" alt="Tableau Story Part 1" src="https://github.com/user-attachments/assets/61a34373-ea43-46af-ba2c-79cc4bfa10c5" />
<img width="1415" height="1063" alt="Tableau Story Part 2" src="https://github.com/user-attachments/assets/5ab6fbba-b391-47a5-9d66-a86f7545c683" />

### Limitations & Assumptions

* **Synthetic enrichment:** User acquisition fields (e.g., `acquisition_channel`, CAC/ad spend) are enriched and may not reflect production distributions.
* **Schema/coverage:** Missing events or short time windows can bias retention and ROI estimates; metrics are indicative.
* **Attribution simplification:** Channel attribution is 1'touch in this demo; multi'touch or MMM would alter ROI interpretation.
* **Model scope:** Forecasts/classifiers are compact prototypes (no hyper-intensive tuning). Calibration and backtesting are included to keep results honest.

---

### License & Credits

* Code is released under **MIT License** (see `LICENSE`).
* Dataset: Cookie Cats (Kaggle) " used here for educational/demo purposes with synthetic UA enrichment.

---
