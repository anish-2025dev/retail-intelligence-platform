select product_id as product_key, category from {{ ref('stg_products') }}
