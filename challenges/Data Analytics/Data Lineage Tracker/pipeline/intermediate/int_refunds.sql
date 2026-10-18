SELECT
    customer_id,
    count(*)             AS n_cancelled_invoices,
    sum(-invoice_amount) AS refunded_amount
FROM int_invoices
WHERE is_cancellation
  AND customer_id IS NOT NULL
GROUP BY customer_id
