-- One row per invoice. customer_id and country are constant within an invoice; min() keeps the
-- result deterministic if a file ever disagrees.
SELECT
    invoice_no,
    min(customer_id)          AS customer_id,
    min(country)              AS country,
    min(invoiced_at)          AS invoiced_at,
    count(*)                  AS n_lines,
    sum(quantity)             AS n_units,
    sum(line_amount)          AS invoice_amount,
    bool_or(is_cancellation)  AS is_cancellation
FROM stg_lines
GROUP BY invoice_no
