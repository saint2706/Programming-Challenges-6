# Customer Segmentation Explorer (RFM Analysis)

**Category:** Data Analytics
**Difficulty:** I

**Status:** Implemented (Python)

Source modules live in `src/rfm_explorer/`; the tests are in `tests/`.

Recency / Frequency / Monetary scoring of real customers, with a Streamlit dashboard you can
re-score live and a check of whether the resulting segments predict what customers do next. It uses
the UCI *Online Retail II* transactions (a UK gift-ware wholesaler, 2009-12-01 to 2011-12-09, CC BY
4.0): 1,044,848 invoice lines after removing the duplicated overlap between the two workbook sheets.

```bash
uv run streamlit run src/rfm_explorer/app.py     # the dashboard
uv run rfm score --method quintile               # segments table + out/rfm_customers.csv
uv run rfm validate --horizon-days 90            # do segments predict the next 90 days?
uv run rfm ledger                                # what was dropped before scoring, and why
uv run pytest -q
```

The committed `sample_data/transactions_sample.parquet` (every row of 600 random customers) makes all
of this work out of the box. To use the full file: `uv run --group fetch python -m
rfm_explorer.fetch_data` (about 45 MB download, written to `data/`, which is not committed). The
numbers below are from the full file.

## What it does

| Piece      | What it gives you                                                                                                                                                                                                 |
| ---------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Cleaning   | One ledger of every dropped row (count and pounds) so the totals can be trusted.                                                                                                                                  |
| RFM        | Recency in whole days since the last *purchase*, frequency as distinct invoices, monetary net of refunds (or gross), as of any snapshot date and look-back window. Nothing after the snapshot is visible.         |
| Scoring    | `quintile` (equal-population, tie-safe), `fixed` (editable business thresholds), `kmeans` (log-scaled clusters, ranked and described in words).                                                                   |
| Segments   | A 5x5 `(R, round((F+M)/2))` grid giving the 11 usual names (Champions, Loyal, At Risk, Can't Lose Them, Hibernating, Lost...). It is a labelling convention in `scoring.SEGMENT_GRID`, not a result; edit it.     |
| Dashboard  | Sidebar: snapshot date, look-back, net/gross, scoring method, thresholds, k, country filter. Tabs: segment size vs money, the R x FM grid, customer table with CSV download, validation, and the cleaning ledger. |
| Validation | Score as of a past date using only earlier data, then compare what the segments did in the next 30 to 180 days.                                                                                                   |

## What the data says

**The business is concentrated.** At the final snapshot (look-back one year, 4,261 customers who
bought) the top 20% of customers by spend hold 74% of the money, the top 1% hold 30%, and the ten
biggest customers alone hold 17%. The quintile **Champions** are 22% of customers and 66% of money.
That share is stable in customers (21 to 22% for look-backs of 90 days to all history) but not in
money (60% at a 90-day look-back, about 68% with all history), so quote the window with the number.

**The common tie-handling recipe gives identical customers different scores.** 35.5% of customers
have exactly one purchase. The usual `qcut(rank(method="first"))` recipe cuts that block by row
order: of the 1,514 one-purchase customers, 852 get a frequency score of 1 and 662 get 2, with
nothing different about them. Here, tied values always share a score. The cost is that the groups are
uneven, and **no customer ever gets a frequency score of 2** (the score-1 block is 35% and the next
block lands in group 3). That is what honouring ties costs, and a test pins it.

**Methods disagree, a lot.** Adjusted Rand Index between the segmentations: quintile vs fixed 0.42,
quintile vs k-means 0.29, fixed vs k-means 0.34. The R, F and M scores themselves match between
quintile and fixed for only 30%, 50% and 86% of customers. Which method you pick changes who gets
called a Champion. Net vs gross monetary matters much less: 1.3% of customers change segment, because
refunds are small next to purchases for almost everyone. (11 customers have a net value of zero or less.)

**There is no natural number of clusters.** Silhouette is highest at k = 2 (0.43) and flat at
0.30 to 0.34 for k from 3 to 8, so k = 5 is a convention, and the k-means segments are a convenience
rather than a discovery.

**Do segments predict what comes next?** Yes, strongly. Scored as of 2011-09-11 (4,302 customers,
51% buy again in the next 90 days), the quintile segments separate cleanly: **Champions buy again
84% of the time and average £1,931 over the next 90 days, 64% of all future spend; Lost customers buy
again 21% of the time and average £60.** Down the list the repeat rate falls from 84% to 21%, with a few small inversions (Promising and Hibernating are close).

**But the full RFM score is not better than money alone.** Share of future spend earned by the top
20% of customers (random = 20%), with paired bootstrap intervals of the difference from the standard
quintile R+F+M score:

| Ordering             | Top-20% share (90 days) | Difference from quintile R+F+M, 95% interval |
| -------------------- | ----------------------- | -------------------------------------------- |
| monetary only        | 65.8%                   | -0.3 to +4.0 points                          |
| frequency + monetary | 63.6%                   | -0.2 to +1.3                                 |
| fixed R+F+M          | 63.3%                   | -0.1 to +0.9                                 |
| **quintile R+F+M**   | 63.0%                   | (reference)                                  |
| k-means cluster rank | 61.6%                   | -2.2 to -0.5                                 |
| frequency only       | 61.3%                   | -3.2 to -0.9                                 |
| recency only         | 52.5%                   | -16.0 to -5.1                                |

Recency alone is clearly the weakest signal. Monetary alone is at least as good as the full score at
every horizon: ahead by 0.4 to 4.8 points at 30 days, 0.9 to 5.2 at 60 and 0.3 to 4.6 at 180, and
indistinguishable at 90 (the table). So for *ranking by future spend* the R and F dimensions add little once past spend
is known. RFM earns its keep as a way to *describe and act on* customers (who is lapsing, who is
new), not as a better predictor. These are bootstrap intervals over customers in one dataset, one
business and one period. Late September to early December is peak season for a gift wholesaler, so
repeat rates are high; do not carry the absolute numbers to another business.

**One customer can make a segment look like the best.** In the 90-day check, "Promising" (recent
customers with one small purchase, 200 people) shows £946 mean future spend, above Loyal's £494. One
customer in it spent £168,470 afterwards: 89% of the segment's total. Without the top five customers
the mean is £77 and the median is £0. Mean spend per segment is not robust in a heavy-tailed
wholesale business; look at the repeat rate and medians too.

## Data decisions

| Decision                                   | Why                                                                                                                                                                                                                                                                              |
| ------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Drop overlapping sheet rows **by invoice** | Both workbook sheets contain 2010-12-01 to 2010-12-09: 1,088 identical invoices, 22,523 rows. Dropping exact duplicate rows would also delete real repeats (the same product twice on one invoice), so the overlap is removed by invoice number in `fetch_data`.                 |
| Drop non-product stock codes               | Postage, manual lines, discounts, fees, samples and test rows are not customer purchases: 5,993 rows, net -£70,350. A product code is five digits plus up to two letters.                                                                                                        |
| Drop price <= 0                            | 5,938 rows of stock write-offs and adjustments.                                                                                                                                                                                                                                  |
| Drop rows with no customer id              | 227,089 rows (21.7% of the file) cannot be segmented. They are **13.6% of revenue** (£2.57M), which is why the dashboard shows the ledger: segment totals describe 86%, not all, of the business.                                                                                |
| Keep returns (`C` invoices) and net them   | A refund reduces monetary value but is not a visit, so it never changes recency or frequency. A refund cannot be tied to the purchase it reverses, so a customer who refunds more than they bought in the window has a negative net value (kept; clipped to 0 only for k-means). |
| Recency uses whole days to the *snapshot*  | A purchase the day before the snapshot has recency 1; activity on or after the snapshot day is invisible. Tests check that adding future rows cannot change any score.                                                                                                           |
| Default look-back of one year              | RFM against all history lets a 2009 purchase count as "recent monetary". The look-back is a sidebar control.                                                                                                                                                                     |

## Design notes

**Leak-proof by construction.** `build_rfm` filters to `[snapshot - lookback, snapshot)` before
anything is computed, and the validation scores the past and measures the next period from the same
table. A test appends future transactions and checks the scores do not move.

**Top-share with ties.** A coarse score (R+F+M takes about a dozen values) has big tied groups. When
the top 20% cut falls inside one, `top_share` counts the group in proportion to how much of it fits,
so neither a method with many ties nor one with few is rewarded or punished by where the ties happen
to sit. The bootstrap resamples customers and evaluates every ordering on the same resample, which
is what makes the *differences* reliable (the intervals on the individual shares overlap heavily).

**Fixed thresholds are not derived from the data.** The defaults (recency 30/90/180/270 days,
frequency 1/2/4/9, spend £200/500/1,000/2,500) were picked by eye from this dataset's quantiles. They
are the point of `fixed`: stable cut-offs you choose and can explain, edited live in the sidebar.

**k-means** runs on standardized `log1p` of the three values (so a £280,000 customer does not own a
cluster) with 10 restarts and a fixed seed. Clusters are ranked by centroid (recent, frequent,
high-spend first) and named in words, for example `K1 recent·frequent·high-spend`.

## Not done

- No predictive model of any kind: validation only checks the segments against what happened, which
  keeps this a segmentation challenge rather than a churn model.
- No customer lifetime value and no probabilistic (BG/NBD) repeat-purchase model; those are the
  next step if prediction were the goal.
- Product categories are not used, so there is no per-segment basket analysis.
- Guest checkouts (13.6% of revenue) cannot be segmented at all; only the ledger reports them.

## Sources

- Chen, D. (2019). *Online Retail II* data set. UCI Machine Learning Repository,
  <https://archive.ics.uci.edu/dataset/502/online+retail+ii> (CC BY 4.0). Download SHA-256 in
  `sample_data/SOURCES.json`.
- Hughes, A. M. (1994). *Strategic Database Marketing*: the original recency / frequency / monetary
  scoring idea.
- The segment names follow the widely used 11-segment grid popularised by Putler and the `rfm` R
  package; the exact cell assignments here are this project's own.
