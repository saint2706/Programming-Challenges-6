SELECT
    order_id,
    COLUMNS('_amount$') * 2
FROM orders
