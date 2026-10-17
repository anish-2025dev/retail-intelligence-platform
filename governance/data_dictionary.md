# Data dictionary

Source: Olist Brazilian E-Commerce Public Dataset. Raw tables are loaded to schema `raw` **after** validation
(see `data_quality/validation.py`). Timestamps are stored as parsed timestamps; money as text in the source CSV.

## raw.orders  (PK: order_id)
| Column | Type | Meaning | Nulls | Valid values / rule |
|---|---|---|---|---|
| order_id | text | Unique order identifier | No | Unique |
| customer_id | text | FK -> customers. One per order (not per person) | No | Must exist in customers |
| order_status | text | Lifecycle stage | No | created, approved, invoiced, processing, shipped, delivered, unavailable, canceled |
| order_purchase_timestamp | timestamp | When the order was placed | No (rejected if null) | Every pair of present timestamps must be in order: purchase <= approved <= carrier <= delivered |
| order_approved_at | timestamp | Payment approval | Yes | |
| order_delivered_carrier_date | timestamp | Handed to carrier | Yes | |
| order_delivered_customer_date | timestamp | Delivered to customer | Yes | Expected null for non-delivered orders |
| order_estimated_delivery_date | timestamp | Delivery promise to the customer | Expected non-null (warned if missing) | |

## raw.customers  (PK: customer_id)
`customer_unique_id` identifies the *person* (use for repeat-purchase and RFM). `customer_id` is per order.

## raw.order_items  (PK: order_id + order_item_id)
| Column | Meaning | Rule |
|---|---|---|
| product_id / seller_id | FKs to products / sellers | Must exist |
| price | Item price (BRL) | > 0 |
| freight_value | Shipping cost (BRL) | >= 0 |

## raw.payments  (PK: order_id + payment_sequential)
`payment_value` >= 0. An order can have several payments (instalments, vouchers).

## raw.reviews
`review_score` integer 1-5. Unique on (review_id, order_id): review_id alone is not unique in the source data.

## raw.products / raw.sellers / raw.category_translation
Category names are Portuguese; `category_translation` maps them to English. Missing categories become `unknown` in staging.

## Analytics layer (dbt)
| Model | Grain | Notes |
|---|---|---|
| fact_orders | 1 row / order | `delivery_days`, `is_late` are null unless delivered |
| fact_order_items | 1 row / order item | |
| dim_customers, dim_products, dim_sellers, dim_date | 1 row / entity | |
| mart_category_monthly_sales | category x month | excludes canceled/unavailable |
| mart_seller_performance | seller | late rate over delivered orders only |
| mart_delivery_vs_reviews | customer state | |

## Limitations
- There is no cost data, so **profit cannot be calculated**; the platform reports revenue, freight and payment value.
- `customer_unique_id` has few repeat buyers; "not purchased recently" is not the same as "churned".
- Timestamps carry no time-zone information.

## Governance controls
- Every rule, failure count and action: `audit.validation_results`; rejected rows: `audit.rejected_records`.
- Every run: `audit.pipeline_runs` (start/end, read/inserted/rejected, failures, status, error).
- Reproducibility: all transformations are version-controlled SQL (dbt) and Python; `make` targets rebuild from raw CSV.
