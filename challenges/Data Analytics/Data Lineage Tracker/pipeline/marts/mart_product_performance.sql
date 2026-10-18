SELECT
    product_code,
    product_name,
    units_sold,
    n_customers,
    median_price,
    min_price,
    rank() OVER (ORDER BY units_sold DESC NULLS LAST, product_code) AS units_rank,
    units_sold * median_price                                       AS est_revenue
FROM int_products
