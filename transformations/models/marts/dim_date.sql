with bounds as (
    select min(purchase_ts)::date as d0, max(estimated_delivery_ts)::date as d1 from {{ ref('stg_orders') }}
)
select d::date as date_key,
       extract(year from d)::int as year, extract(month from d)::int as month,
       date_trunc('month', d)::date as month_start, extract(isodow from d)::int as iso_weekday,
       to_char(d, 'YYYY-MM') as year_month
from bounds, generate_series(d0, d1, interval '1 day') as d
