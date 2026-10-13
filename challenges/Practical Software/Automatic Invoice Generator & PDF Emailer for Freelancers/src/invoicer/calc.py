"""Invoice totals. Pure functions over integers; no database, no I/O.

Rules (the common EU/B2B convention, stated so they can be audited):

* a line's net amount is ``round(quantity * unit price)`` to the currency's minor unit;
* an invoice-level discount is spread over the lines in proportion to their net amounts using
  largest-remainder allocation, so the shares add up to the discount exactly;
* tax is computed once per tax rate on the *sum of discounted nets of that rate*, not per line,
  which is how VAT breakdowns on e-invoices are defined (EN 16931 BG-23).
"""

from collections.abc import Sequence
from dataclasses import dataclass
from decimal import Context, Decimal, localcontext
from functools import wraps
from typing import Literal

from invoicer.money import BP_SCALE, QTY_SCALE, MoneyError, round_half_up

DiscountKind = Literal["percent", "fixed"]


@dataclass(frozen=True)
class LineInput:
    qty_e4: int
    unit_minor: int
    tax_bp: int


@dataclass(frozen=True)
class Discount:
    kind: DiscountKind
    value: int  # basis points for "percent", minor units for "fixed"


@dataclass(frozen=True)
class LineResult:
    net: int  # before discount
    discount: int  # share of the invoice discount
    taxable: int  # net - discount
    tax_bp: int


@dataclass(frozen=True)
class TaxGroup:
    rate_bp: int
    taxable: int
    tax: int


@dataclass(frozen=True)
class Totals:
    lines: tuple[LineResult, ...]
    subtotal: int  # sum of line nets
    discount: int
    net: int  # subtotal - discount
    tax_groups: tuple[TaxGroup, ...]
    tax: int
    total: int  # net + tax


def _wide(fn):
    """Run ``fn`` with 60 significant digits. The default 28 would silently round a product of
    the largest allowed quantity and price (up to 10**28)."""

    @wraps(fn)
    def inner(*args, **kwargs):
        with localcontext(Context(prec=60)):
            return fn(*args, **kwargs)

    return inner


@_wide
def line_net(line: LineInput) -> int:
    return round_half_up(Decimal(line.qty_e4) * line.unit_minor / QTY_SCALE)


@_wide
def allocate(total: int, weights: Sequence[int]) -> list[int]:
    """Split ``total`` over ``weights`` proportionally; the parts always sum to ``total``."""
    weight_sum = sum(weights)
    if total == 0 or weight_sum == 0:
        return [0] * len(weights)
    exact = [Decimal(total) * w / weight_sum for w in weights]
    shares = [int(x) for x in exact]  # floor: all weights are non-negative
    leftover = total - sum(shares)
    # Hand the remaining minor units to the largest fractional parts (first wins on ties).
    order = sorted(
        range(len(weights)), key=lambda i: (exact[i] - shares[i], -i), reverse=True
    )
    for i in order[:leftover]:
        shares[i] += 1
    return shares


@_wide
def compute(lines: Sequence[LineInput], discount: Discount | None = None) -> Totals:
    for line in lines:
        if line.qty_e4 <= 0:
            raise MoneyError("quantity must be greater than zero")
        if line.unit_minor < 0:
            raise MoneyError("unit price must not be negative")
        if not 0 <= line.tax_bp <= BP_SCALE:
            raise MoneyError("tax rate must be between 0 and 100 %")
    nets = [line_net(line) for line in lines]
    subtotal = sum(nets)

    if discount is None:
        discount_total = 0
    elif discount.kind == "percent":
        if not 0 <= discount.value <= BP_SCALE:
            raise MoneyError("discount percentage must be between 0 and 100")
        discount_total = round_half_up(Decimal(subtotal) * discount.value / BP_SCALE)
    else:
        if not 0 <= discount.value <= subtotal:
            raise MoneyError(
                "a fixed discount cannot be negative or exceed the subtotal"
            )
        discount_total = discount.value

    shares = allocate(discount_total, nets)
    results = tuple(
        LineResult(net, share, net - share, line.tax_bp)
        for line, net, share in zip(lines, nets, shares, strict=True)
    )
    taxable_by_rate: dict[int, int] = {}
    for r in results:
        taxable_by_rate[r.tax_bp] = taxable_by_rate.get(r.tax_bp, 0) + r.taxable
    groups = tuple(
        TaxGroup(rate, taxable, round_half_up(Decimal(taxable) * rate / BP_SCALE))
        for rate, taxable in sorted(taxable_by_rate.items())
    )
    net = subtotal - discount_total
    tax = sum(g.tax for g in groups)
    return Totals(results, subtotal, discount_total, net, groups, tax, net + tax)
