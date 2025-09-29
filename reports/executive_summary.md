# Executive Summary

## Key KPIs (latest synthetic build)

- **Funnel:** Onboarding completion ~95.6%, D1 retention from onboarding ~47.6%, and D1-to-purchase conversion ~16.8%.
- **Retention:** Overall D7 retention ~33.3%; Organic keeps ~73.8% of its D1 returners (D7 ~ 35.2%).
- **Revenue & ROI:** Organic ROAS ~ 1.85 (ROI ~ +85%); paid channels remain below break-even (ROAS 0.19-0.34).
- **Churn model:** ROC-AUC ~ 0.61, PR-AUC ~ 0.58, accuracy ~ 0.60; the top 10% risk bucket captures ~ 78% of churn (lift ~ 1.17x).

## Recommended Actions

1. Focus onboarding experiments to lift the 52% drop between install and D1.
2. Reallocate budget toward high-ROAS sources (Organic); iterate creatives for TikTok/Instagram before additional spend.
3. Target retention campaigns at high-risk Google Play cohorts flagged by the churn model.

## Artefacts

- Tables: `reports/tables/funnel.csv`, `roi_by_channel.csv`, `retention_by_channel.csv`, `churn_risk_segments.csv`, `model_metrics.json`.
- Figures: `reports/figures/funnel.png`, `roi_by_channel.png`, `retention_heatmap.png`, `roc_pr_curves.png`.
- Dashboards: Tableau overview and churn dashboards (see README embeds).
