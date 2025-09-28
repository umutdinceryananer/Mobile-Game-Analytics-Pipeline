-- Each row is a user; count users per acquisition channel
SELECT
  acquisition_channel,
  COUNT(*) AS users
FROM users
GROUP BY 1
ORDER BY users DESC;
