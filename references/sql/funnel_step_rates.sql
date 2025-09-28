WITH base AS (
  SELECT
    COUNT(*) AS n_users,
    SUM(CASE WHEN retention_1 THEN 1 ELSE 0 END) AS s_d1,
    SUM(CASE WHEN retention_7 THEN 1 ELSE 0 END) AS s_d7,
    SUM(CASE WHEN purchase = 1 THEN 1 ELSE 0 END) AS s_purchase
  FROM users
)
SELECT
  n_users,
  1.0 AS rate_install,                                       -- tüm kullanıcılar
  s_d1 * 1.0 / NULLIF(n_users, 0)           AS rate_d1_from_install,
  s_d7 * 1.0 / NULLIF(s_d1, 0)              AS rate_d7_from_d1,
  s_purchase * 1.0 / NULLIF(n_users, 0)     AS rate_purchase_overall,
  s_purchase * 1.0 / NULLIF(s_d7, 0)        AS rate_purchase_from_d7
FROM base;
