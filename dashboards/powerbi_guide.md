# Power BI setup

Power BI Desktop runs on Windows. Connect with *Get data -> PostgreSQL database* (`localhost:5432`, db `retail`,
Import mode) and load these `analytics` tables:
`fact_orders, dim_customers, dim_products, dim_sellers, dim_date, mart_category_monthly_sales,
mart_seller_performance, mart_delivery_vs_reviews, ml_customer_segments, ml_sales_forecast`.

If PostgreSQL isn't reachable from your machine, load the CSVs in `outputs/` instead.
Relationships: fact_orders[date_key]->dim_date, fact_orders[customer_key]->dim_customers.

## Pages
1. **Executive overview** - cards: orders, revenue, AOV, on-time %, avg review; monthly sales line; slicers on date/state.
2. **Customers** - segments from `ml_customer_segments` (customers, avg monetary, recency); map by state; repeat rate.
3. **Products and sellers** - category revenue and monthly trend; seller late-rate table (min 30 delivered orders); late vs review scatter/bars.
4. **Forecast and recommendations** - `ml_sales_forecast` line chart, colour by `kind` (actual / backtest_prediction / forecast);
   table from `forecast_model_metrics.csv`; text box with recommendations.

Starter DAX:
```
Orders = DISTINCTCOUNT(fact_orders[order_id])
Revenue = SUM(fact_orders[order_value])
AOV = DIVIDE([Revenue], [Orders])
On-time % = DIVIDE(CALCULATE(COUNTROWS(fact_orders), fact_orders[is_late] = FALSE()), CALCULATE(COUNTROWS(fact_orders), NOT ISBLANK(fact_orders[delivered_ts])))
Avg review = AVERAGE(fact_orders[review_score])
```
Label forecast values clearly as predictions, and show the forecast error metric on the page.
