-- Which categories earn the most, and how does each change month to month? (CTE + window functions)
WITH monthly AS (
    SELECT month_start, category, SUM(revenue) AS revenue
    FROM analytics.mart_category_monthly_sales GROUP BY 1, 2
), ranked AS (
    SELECT *, RANK() OVER (PARTITION BY month_start ORDER BY revenue DESC) AS rank_in_month,
           revenue / NULLIF(LAG(revenue) OVER (PARTITION BY category ORDER BY month_start), 0) - 1 AS mom_growth
    FROM monthly
)
SELECT * FROM ranked WHERE rank_in_month <= 5 ORDER BY month_start, rank_in_month;
