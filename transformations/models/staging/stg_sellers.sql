select seller_id, seller_city, upper(seller_state) as seller_state
from {{ source('raw', 'sellers') }}
