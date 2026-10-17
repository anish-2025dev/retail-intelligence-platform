select d.year_month, d.month_start, p.category,
       count(distinct fi.order_id) as orders,
       sum(fi.price) as revenue,
       sum(fi.price) - lag(sum(fi.price)) over (partition by p.category order by d.month_start) as revenue_change_vs_prev_month
from {{ ref('fact_order_items') }} fi
join {{ ref('dim_products') }} p using (product_key)
join {{ ref('dim_date') }} d using (date_key)
join {{ ref('fact_orders') }} o using (order_id)
where o.order_status not in ('canceled', 'unavailable')
group by 1, 2, 3
