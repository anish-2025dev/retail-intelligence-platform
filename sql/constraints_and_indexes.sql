-- Applied after the raw load (see ingestion/load.py). Raw tables are loaded already validated,
-- so primary/foreign keys can be enforced as a second line of defence.
ALTER TABLE raw.customers   ADD PRIMARY KEY (customer_id);
ALTER TABLE raw.products    ADD PRIMARY KEY (product_id);
ALTER TABLE raw.sellers     ADD PRIMARY KEY (seller_id);
ALTER TABLE raw.orders      ADD PRIMARY KEY (order_id);
ALTER TABLE raw.order_items ADD PRIMARY KEY (order_id, order_item_id);

ALTER TABLE raw.orders      ADD FOREIGN KEY (customer_id) REFERENCES raw.customers(customer_id);
ALTER TABLE raw.order_items ADD FOREIGN KEY (order_id)    REFERENCES raw.orders(order_id);
ALTER TABLE raw.order_items ADD FOREIGN KEY (product_id)  REFERENCES raw.products(product_id);
ALTER TABLE raw.order_items ADD FOREIGN KEY (seller_id)   REFERENCES raw.sellers(seller_id);
ALTER TABLE raw.payments    ADD FOREIGN KEY (order_id)    REFERENCES raw.orders(order_id);
ALTER TABLE raw.reviews     ADD FOREIGN KEY (order_id)    REFERENCES raw.orders(order_id);

CREATE INDEX IF NOT EXISTS idx_orders_purchase ON raw.orders (order_purchase_timestamp);
CREATE INDEX IF NOT EXISTS idx_orders_customer ON raw.orders (customer_id);
CREATE INDEX IF NOT EXISTS idx_items_product   ON raw.order_items (product_id);
CREATE INDEX IF NOT EXISTS idx_items_seller    ON raw.order_items (seller_id);
