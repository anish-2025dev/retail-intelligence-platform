select c.customer_state,
       count(*) as delivered_orders,
       avg(case when is_late then 1.0 else 0.0 end) as late_rate,
       avg(delivery_days) as avg_delivery_days,
       avg(review_score) filter (where is_late) as avg_review_when_late,
       avg(review_score) filter (where not is_late) as avg_review_when_on_time
from {{ ref('fact_orders') }} o
join {{ ref('dim_customers') }} c using (customer_key)
where o.delivered_ts is not null
group by 1
