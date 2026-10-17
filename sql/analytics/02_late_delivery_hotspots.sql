-- Sellers with the worst late-delivery rate (minimum volume to avoid noise)
SELECT seller_key, seller_state, delivered_orders, ROUND(late_delivery_rate::numeric, 3) AS late_rate,
       ROUND(avg_review_score::numeric, 2) AS avg_review
FROM analytics.mart_seller_performance
WHERE delivered_orders >= 30
ORDER BY late_delivery_rate DESC LIMIT 20;
