SELECT
    product_code,
    min(product_name)                                    AS product_name,
    min(unit_price) FILTER (WHERE unit_price > 0)        AS min_price,
    median(unit_price)                                   AS median_price,
    sum(quantity) FILTER (WHERE NOT is_cancellation)     AS units_sold,
    count(DISTINCT customer_id)                          AS n_customers
FROM stg_lines
GROUP BY product_code
