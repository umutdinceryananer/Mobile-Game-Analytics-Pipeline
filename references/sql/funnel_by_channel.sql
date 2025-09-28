WITH per_user AS (
  SELECT
    acquisition_channel,
    retention_1,
    retention_7,
    purchase
  FROM users
),
agg AS (
  SELECT
    acquisition_channel,
    COUNT(*) AS n_users,
    SUM(CASE WHEN retention_1 THEN 1 ELSE 0 END) AS s_d1,
    SUM(CASE WHEN retention_7 THEN 1 ELSE 0 END) AS s_d7,
    SUM(CASE WHEN purchase = 1 THEN 1 ELSE 0 END) AS s_purchase
  FROM per_user
  GROUP BY 1
)
SELECT
  acquisition_channel,
  n_users,
  s_d1 * 1.0 / NULLIF(n_users, 0)  AS rate_d1_from_install,
  s_d7 * 1.0 / NULLIF(s_d1, 0)     AS rate_d7_from_d1,
  s_purchase * 1.0 / NULLIF(n_users, 0) AS rate_purchase_overall
FROM agg
ORDER BY rate_d1_from_install DESC;
