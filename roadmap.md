## Project Purpose

Analyze mobile game user behavior using the enriched dataset to meet data science expectations typically found in job postings:

* Conversion funnel analysis
* ROI and marketing performance evaluation
* Retention and churn analysis
* Revenue forecasting
* SQL, Python, and dashboard development

---

## Timeline (3-Day Plan)

### Day 1 — Data Exploration & Preparation (completed)

Processed the Cookie Cats dataset, verified no missing values or duplicate user IDs.
Added synthetic columns deterministically, driven by configuration:
* [x] acquisition_channel (Instagram, Facebook, TikTok, Organic)
* [x] CAC (channel-based customer acquisition cost)
* [x] purchase (session-based probability)
* [x] revenue (in-app purchases via gamma distribution + ad revenue per session)
* [x] ROI (return on investment per user)
* [x] country (USA, Mexico, Brazil, with realistic distribution)
* [x] platform (App Store, Google Play, calibrated to installs and revenue split)

Updated `synthetic.yaml`: increased CAC variance, scaled up revenue parameters, calibrated platform revenue shares to match 26M vs 9M distribution.
Added automated tests (pytest, 6 tests) to validate schema, distributions, and revenue shares. All tests passed.

Initial EDA results:
* [x] 90,189 rows, 12 columns
* [x] Global purchase rate: 5.58%
* [x] Retention: D1 ≈ 44.5%, D7 ≈ 18.6%
* [x] Average sessions: 51.9 (median 16, heavy-tailed, >425 sessions flagged as outliers)
* [x] Channel, platform, and country distributions match configuration
* [x] Platform performance: App Store ≈ 24.9% of installs but ≈ 74.4% of revenue; Google Play ≈ 75.1% of installs and ≈ 25.6% of revenue
* [x] Revenue and ROI distributions widened; ROI remains negative on average but is suitable for scenario analysis

---

### Day 2 — Funnel & ROI Analysis

* [ ] Funnel conversion analysis (ad click → install → session → purchase)
* [ ] Cohort retention (D1, D7)
* [ ] ROI calculation (per channel and platform)
* [ ] Visualizations: matplotlib/plotly + Tableau

**Success Criteria:**
* [ ] Funnel shows conversion percentages at each stage
* [ ] Cohort retention table is generated
* [ ] ROI is calculated per channel

---

### Day 3 — Prediction, Forecasting & Dashboard

* [ ] Define churn label (e.g., user inactive for 7+ days)
* [ ] Train churn prediction models (Logistic Regression, XGBoost)
* [ ] Revenue forecasting (Prophet or ARIMA)
* [ ] Tableau dashboard → Funnel, retention, churn risk, forecast
* [ ] Executive Summary report (Markdown/PDF)

**Input:** `data/processed/clean_data.csv`
**Output:** `modeling.ipynb`, dashboard screenshots, `executive_summary.md`

**Success Criteria:**
* [ ] Churn model AUC ≥ 0.75
* [ ] Forecast plots show trend and seasonality
* [ ] Dashboard contains at least 4 key KPI charts

---

## Business Concepts

* **Conversion Funnel** → User journey (ad click → install → session → purchase)
* **Retention (D1, D7, D30)** → Percentage of users retained at different time horizons
* **Churn** → Percentage of users leaving the game
* **Cohort Analysis** → Retention/LTV differences by signup date
* **ARPU (Average Revenue Per User)** → Average revenue across all users
* **ARPPU** → Average revenue among paying users
* **LTV (Lifetime Value)** → Total expected user revenue
* **CPI (Cost Per Install)** → Acquisition cost per user
* **ROI (Return on Investment)** → (LTV – CPI) / CPI

---

## Data Dictionary

| Column              | Description                                   | Type        |
| ------------------- | --------------------------------------------- | ----------- |
| user_id             | User identifier                               | String      |
| install_date        | Date of installation                          | Date        |
| acquisition_channel | Marketing channel (Instagram, Facebook, etc.) | Categorical |
| CAC                 | Customer acquisition cost                     | Numeric     |
| revenue             | Total user revenue                            | Numeric     |
| sessions            | Total session count                           | Integer     |
| retention_d1/d7/d30 | Retention flags                               | Boolean     |
| churn_flag          | Churn indicator                               | Boolean     |
| country             | User country (USA, Mexico, Brazil)            | Categorical |
| platform            | Platform (App Store, Google Play)             | Categorical |

---

## Executive Summary (Example)

* Funnel conversion rate: 12% of users convert install → purchase
* D7 retention: 23% → churn is relatively high
* Churn model AUC: 0.78 → lower session count strongly correlates with churn risk
* Revenue forecast: revenue trend expected to increase by ~5% over the next 30 days

---

## Next Development Steps

* Add user segmentation in dashboard (by country, platform, device)
* Compare churn models across different ML algorithms
* Tune synthetic revenue distribution parameters for more realism
* Apply survival analysis for LTV estimation

---

## Changelog

- 2025-09-27: Added synthetic country and platform columns in `data/make_dataset.py`, including platform-based revenue scaling.
- 2025-09-27: Defined `geo.country` and `platform` sections in `data/config/synthetic.yaml`.
