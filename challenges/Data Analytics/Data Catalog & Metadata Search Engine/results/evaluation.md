# Catalog evaluation on the demo estate (10 datasets, 4 source kinds)

## Foreign-key inference (no source declares any)

| Precision | Recall | True positives | False positives | Missed |
| --------- | ------ | -------------- | --------------- | ------ |
| 1.0       | 1.0    | 11             | 0               | 0      |

## PII tagging

| Precision | Recall | Tagged columns                                                                                                                                            |
| --------- | ------ | --------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1.0       | 1.0    | customer.c_address, customer.c_phone, supplier.s_address, supplier.s_phone, users.email, users.first_name, users.ip_address, users.last_name, users.phone |

## Search (16 labelled queries)

| Mode   | MRR   | Hit@5 |
| ------ | ----- | ----- |
| like   | 0.736 | 15/16 |
| fts    | 0.637 | 13/16 |
| hybrid | 0.906 | 15/16 |

## Top 3 hybrid hits per query

| Query           | Top 3                                                             |
| --------------- | ----------------------------------------------------------------- |
| customer email  | users.email, users.customer_id, customer.*                        |
| phone number    | users.phone, customer.c_phone, supplier.s_phone                   |
| ship date       | lineitem.l_shipdate, users.signup_date, lineitem.l_shipmode       |
| extended price  | lineitem.l_extendedprice, lineitem.*, orders.o_totalprice         |
| order priority  | orders.o_orderpriority, orders.o_orderkey, orders.*               |
| supplier        | supplier.*, supplier.s_name, supplier.s_suppkey                   |
| nation          | nation.*, nation.n_nationkey, customer.c_nationkey                |
| discount        | lineitem.l_discount, lineitem.*                                   |
| ip address      | users.ip_address, supplier.s_address, customer.c_address          |
| signup          | users.signup_date, users.*                                        |
| region          | region.*, region.r_regionkey, nation.n_regionkey                  |
| retail price    | part.p_retailprice, part.*, orders.o_totalprice                   |
| events          | events.*, events.event_id, events.event                           |
| price           | orders.o_totalprice, part.p_retailprice, lineitem.l_extendedprice |
| segment         | customer.c_mktsegment                                             |
| account balance | customer.c_comment, orders.o_comment, part.p_comment              |
