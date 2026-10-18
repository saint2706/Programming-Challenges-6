SELECT
    date_trunc('month', i.invoiced_at)                                AS month,
    COALESCE(c.region, 'Unknown')                                     AS region,
    sum(i.invoice_amount) FILTER (WHERE NOT i.is_cancellation)        AS sales,
    sum(-i.invoice_amount) FILTER (WHERE i.is_cancellation)           AS refunds,
    count(DISTINCT i.invoice_no)                                      AS n_invoices,
    count(DISTINCT i.customer_id)                                     AS n_customers
FROM int_invoices AS i
LEFT JOIN stg_countries AS c ON i.country = c.country_name
GROUP BY 1, 2
