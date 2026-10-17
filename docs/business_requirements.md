# Business requirements (hypothetical client engagement)

**Client:** a Brazilian online marketplace (modelled with the public Olist data).
**Problem:** leadership wants to understand sales trends, customer behaviour and delivery performance, and plan demand.

## Questions to answer
1. How have sales evolved by month, region and category? Where is growth or decline?
2. Which customer groups should be prioritised for retention/marketing?
3. Which sellers or regions deliver late, and does lateness hurt review scores?
4. What sales should we expect over the next 8 weeks, and how accurate is the forecast?

## KPIs
Orders, revenue, average order value, on-time delivery rate (delivered orders only), average review score, repeat-purchase rate.

## Data
Orders, items, payments, reviews, customers, products, sellers (Olist, 2016-2018). See `governance/data_dictionary.md`.

## Assumptions and limitations
- No cost data -> no profit analysis. Revenue = item price (+ freight where stated).
- Canceled/unavailable orders are excluded from sales and ML.
- Forecast is at weekly, whole-business level; category-level forecasting is an extension.
- Findings are observational: they show association, not proven causes.

## Success criteria
Reproducible pipeline; data-quality report with every rejected record explained; dashboard answering questions 1-4;
forecast that is evaluated against a naive baseline on a chronological hold-out; 3-5 evidence-backed recommendations.
