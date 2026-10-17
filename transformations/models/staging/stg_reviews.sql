select review_id, order_id, review_score::int as review_score, review_creation_date::timestamp as review_created_ts
from {{ source('raw', 'reviews') }}
