SELECT
    order_id,
    net_amount + tax_amount AS gross_amount
FROM orders
