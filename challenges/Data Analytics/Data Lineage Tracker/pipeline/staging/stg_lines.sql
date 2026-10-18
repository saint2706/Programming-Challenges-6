-- One row per invoice line of a real product. Non-product codes (POST, DOT, M, BANK CHARGES,
-- ADJUST, ...) are filtered out here, so stock_code shapes every column without being copied to one.
SELECT
    invoice                        AS invoice_no,
    stock_code                     AS product_code,
    trim(description)              AS product_name,
    quantity,
    invoice_date                   AS invoiced_at,
    CAST(invoice_date AS DATE)     AS invoice_day,
    price                          AS unit_price,
    customer_id,
    country,
    invoice LIKE 'C%'              AS is_cancellation,
    quantity * price               AS line_amount
FROM raw_transactions
WHERE regexp_matches(stock_code, '^[0-9]{5}')
