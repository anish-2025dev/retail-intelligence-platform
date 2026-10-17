select i.order_id, i.order_item_id, i.product_id as product_key, i.seller_id as seller_key,
       o.date_key, i.price, i.freight_value
from {{ ref('stg_order_items') }} i
join {{ ref('fact_orders') }} o using (order_id)
