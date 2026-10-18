-- A customer's purchases in order, with the gap to the previous one and a running total.
WITH orders AS (
    SELECT customer_id, invoice_no, invoiced_at, invoice_amount
    FROM int_invoices
    WHERE NOT is_cancellation
      AND customer_id IS NOT NULL
)
SELECT
    customer_id,
    invoice_no,
    invoiced_at,
    invoice_amount,
    row_number() OVER (PARTITION BY customer_id ORDER BY invoiced_at, invoice_no) AS order_seq,
    date_diff(
        'day',
        lag(invoiced_at) OVER (PARTITION BY customer_id ORDER BY invoiced_at, invoice_no),
        invoiced_at
    ) AS days_since_prev,
    sum(invoice_amount) OVER (
        PARTITION BY customer_id ORDER BY invoiced_at, invoice_no
    ) AS running_revenue
FROM orders
