with seller_orders as (
    select distinct fi.seller_key, fi.order_id from {{ ref('fact_order_items') }} fi
)
select s.seller_key, ds.seller_state,
       count(*) as orders,
       count(o.delivered_ts) as delivered_orders,
       avg(case when o.is_late then 1.0 else 0.0 end) filter (where o.delivered_ts is not null) as late_delivery_rate,
       avg(o.review_score) as avg_review_score
from seller_orders s
join {{ ref('fact_orders') }} o using (order_id)
join {{ ref('dim_sellers') }} ds on ds.seller_key = s.seller_key
group by 1, 2
