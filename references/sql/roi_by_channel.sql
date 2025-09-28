-- Channel-level ROI = (SUM(revenue) - SUM(CAC)) / SUM(CAC)
-- ROAS = SUM(revenue) / SUM(CAC)
WITH agg AS (
  SELECT
    acquisition_channel,
    COUNT(*)                  AS users,
    SUM(COALESCE(revenue,0))  AS revenue,
    SUM(COALESCE(cac,0))      AS ad_spend
  FROM users
  GROUP BY 1
)
SELECT
  acquisition_channel,
  users,
  revenue,
  ad_spend,
  CASE WHEN ad_spend > 0 THEN (revenue - ad_spend) / ad_spend END AS roi,
  CASE WHEN ad_spend > 0 THEN revenue / ad_spend END               AS roas
FROM agg
ORDER BY roas DESC NULLS LAST;
