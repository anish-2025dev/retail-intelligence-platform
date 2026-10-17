-- Share of unique customers with more than one order
WITH per_customer AS (
    SELECT c.customer_unique_id, COUNT(*) AS orders
    FROM analytics.fact_orders f JOIN analytics.dim_customers c USING (customer_key)
    WHERE f.order_status NOT IN ('canceled','unavailable') GROUP BY 1
)
SELECT COUNT(*) AS customers, AVG((orders > 1)::int) AS repeat_rate FROM per_customer;
