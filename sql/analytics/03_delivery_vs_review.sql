-- Do late deliveries get worse reviews?
SELECT CASE WHEN is_late THEN 'late' ELSE 'on time' END AS delivery, COUNT(*) AS orders,
       ROUND(AVG(review_score), 2) AS avg_review_score
FROM analytics.fact_orders WHERE delivered_ts IS NOT NULL AND review_score IS NOT NULL GROUP BY 1;
