WITH casted AS (
  SELECT
    1 AS install_flag,
    CASE WHEN retention_1 IN (1, TRUE) THEN 1 ELSE 0 END AS d1_flag,
    CASE WHEN retention_7 IN (1, TRUE) THEN 1 ELSE 0 END AS d7_flag,
    CASE WHEN purchase    IN (1, TRUE) THEN 1 ELSE 0 END AS purchase_flag
  FROM events
),
steps AS (
  SELECT
    COUNT(*)  AS n_users,
    SUM(install_flag)  AS s_install,
    SUM(d1_flag)       AS s_d1,
    SUM(d7_flag)       AS s_d7,
    SUM(purchase_flag) AS s_purchase
  FROM casted
)
SELECT
  n_users,
  s_install * 1.0 / NULLIF(n_users, 0)   AS rate_install,
  s_d1      * 1.0 / NULLIF(s_install, 0) AS rate_d1_from_install,
  s_d7      * 1.0 / NULLIF(s_d1, 0)      AS rate_d7_from_d1,
  s_purchase* 1.0 / NULLIF(s_d7, 0)      AS rate_purchase_from_d7
FROM steps;
