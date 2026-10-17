select p.product_id,
       coalesce(t.product_category_name_english, p.product_category_name, 'unknown') as category
from {{ source('raw', 'products') }} p
left join {{ source('raw', 'category_translation') }} t using (product_category_name)
