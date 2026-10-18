-- One row per customer with at least one purchase. p.* passes the CTE's columns through, and
-- COALESCE makes refunded_amount depend on int_refunds only when a customer has refunds.
WITH per_customer AS (
    SELECT
        customer_id,
        count(*)            AS n_orders,
        sum(invoice_amount) AS gross_revenue,
        min(invoiced_at)    AS first_order_at,
        max(invoiced_at)    AS last_order_at,
        max(order_seq)      AS last_order_seq
    FROM int_customer_orders
    GROUP BY customer_id
)
SELECT
    p.*,
    COALESCE(r.refunded_amount, 0)                       AS refunded_amount,
    p.gross_revenue - COALESCE(r.refunded_amount, 0)     AS net_revenue,
    CASE
        WHEN p.n_orders = 1 THEN 'one-time'
        WHEN p.n_orders < 5 THEN 'repeat'
        ELSE 'loyal'
    END                                                  AS segment
FROM per_customer AS p
LEFT JOIN int_refunds AS r USING (customer_id)
