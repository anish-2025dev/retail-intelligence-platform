-- One row per order. Items, payments and reviews are pre-aggregated to avoid join fan-out.
with items as (
    select order_id, count(*) as item_count, sum(price) as items_value, sum(freight_value) as freight_value
    from {{ ref('stg_order_items') }} group by 1
), pay as (
    select order_id, sum(payment_value) as payment_value from {{ ref('stg_payments') }} group by 1
), rev as (
    select order_id, avg(review_score)::numeric(3,2) as review_score from {{ ref('stg_reviews') }} group by 1
)
select
    o.order_id,
    o.customer_id as customer_key,
    o.purchase_ts::date as date_key,
    o.purchase_ts,
    o.order_status,
    i.item_count,
    i.items_value,
    i.freight_value,
    i.items_value + i.freight_value as order_value,
    pay.payment_value,
    o.delivered_ts,
    o.estimated_delivery_ts,
    -- delivery metrics only exist for orders that were actually delivered
    case when o.delivered_ts is not null then extract(epoch from (o.delivered_ts - o.purchase_ts)) / 86400 end as delivery_days,
    case when o.delivered_ts is not null then (o.delivered_ts > o.estimated_delivery_ts) end as is_late,
    rev.review_score
from {{ ref('stg_orders') }} o
left join items i using (order_id)
left join pay using (order_id)
left join rev using (order_id)
