-- Recency, frequency and monetary value as of the last invoice day, scored 1-5.
WITH snapshot AS (
    SELECT max(CAST(invoiced_at AS DATE)) AS as_of FROM int_invoices
),
scored AS (
    SELECT
        c.customer_id,
        date_diff('day', CAST(c.last_order_at AS DATE), s.as_of) AS recency_days,
        c.n_orders                                               AS frequency,
        c.net_revenue                                            AS monetary
    FROM mart_customers AS c
    CROSS JOIN snapshot AS s
)
SELECT
    customer_id,
    recency_days,
    frequency,
    monetary,
    6 - ntile(5) OVER (ORDER BY recency_days, customer_id) AS r_score,
    ntile(5) OVER (ORDER BY frequency, customer_id)        AS f_score,
    ntile(5) OVER (ORDER BY monetary, customer_id)         AS m_score
FROM scored
