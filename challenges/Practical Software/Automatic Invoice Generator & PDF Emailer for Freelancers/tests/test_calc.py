import random

import pytest
from invoicer.calc import Discount, LineInput, allocate, compute, line_net
from invoicer.money import MoneyError


def L(qty_e4, unit, tax=0):
    return LineInput(qty_e4, unit, tax)


def test_line_net_rounds_half_up():
    assert line_net(L(10_000, 1999)) == 1999
    assert line_net(L(15_000, 333)) == 500  # 499.5 -> 500
    assert line_net(L(5_000, 1)) == 1  # 0.5 -> 1
    assert line_net(L(33_333, 100)) == 333  # 333.33


def test_simple_invoice():
    t = compute([L(100_000, 10_000, 1900)])  # 10 x 100.00 @ 19 %
    assert (t.subtotal, t.discount, t.net, t.tax, t.total) == (
        100_000,
        0,
        100_000,
        19_000,
        119_000,
    )
    assert [(g.rate_bp, g.taxable, g.tax) for g in t.tax_groups] == [
        (1900, 100_000, 19_000)
    ]


def test_tax_is_rounded_per_rate_group_not_per_line():
    """Three 0.33 lines at 20 %: per line each tax is 0.07 (0.066) = 0.21, but the group tax on
    0.99 is 0.198 -> 0.20. The e-invoice VAT breakdown is per rate, so the group figure wins."""
    t = compute([L(10_000, 33, 2000)] * 3)
    assert t.tax == 20
    assert len(t.tax_groups) == 1


def test_mixed_rates_get_separate_groups_sorted_by_rate():
    t = compute([L(10_000, 1000, 1900), L(10_000, 500, 0), L(10_000, 2000, 700)])
    assert [g.rate_bp for g in t.tax_groups] == [0, 700, 1900]
    assert t.tax == sum(g.tax for g in t.tax_groups) == 140 + 190
    assert t.total == t.net + t.tax


def test_percent_discount_reduces_taxable_amounts():
    t = compute([L(10_000, 10_000, 2000)], Discount("percent", 1000))  # 10 % off 100.00
    assert (t.discount, t.net, t.tax, t.total) == (1000, 9000, 1800, 10_800)
    assert t.lines[0].taxable == 9000


def test_fixed_discount_is_spread_proportionally_and_exactly():
    t = compute(
        [L(10_000, 5000, 1900), L(10_000, 3000, 700), L(10_000, 2000, 0)],
        Discount("fixed", 1001),
    )
    assert sum(r.discount for r in t.lines) == 1001
    assert [r.discount for r in t.lines] == [
        501,
        300,
        200,
    ]  # 500.5 rounds toward the larger remainder
    assert t.net == 10_000 - 1001


def test_discount_validation():
    line = [L(10_000, 1000)]
    for bad in (
        Discount("percent", 10_001),
        Discount("percent", -1),
        Discount("fixed", 1001),
        Discount("fixed", -1),
    ):
        with pytest.raises(MoneyError):
            compute(line, bad)
    assert compute(line, Discount("fixed", 1000)).total == 0
    assert compute(line, Discount("percent", 10_000)).total == 0


@pytest.mark.parametrize(
    "line",
    [L(0, 100), L(-1, 100), L(10_000, -1), L(10_000, 100, -1), L(10_000, 100, 10_001)],
)
def test_invalid_lines(line):
    with pytest.raises(MoneyError):
        compute([line])


def test_empty_invoice_is_zero():
    t = compute([])
    assert (t.subtotal, t.total, t.tax_groups) == (0, 0, ())


def test_allocate_examples():
    assert allocate(100, [1, 1, 1]) == [34, 33, 33]
    assert allocate(0, [5, 5]) == [0, 0]
    assert allocate(7, [0, 0]) == [0, 0]
    assert allocate(10, [0, 5]) == [0, 10]
    assert allocate(1, [1, 1, 1]) == [1, 0, 0]


def test_allocate_always_sums_exactly():
    rng = random.Random(2026)
    for _ in range(500):
        weights = [rng.randint(0, 100_000) for _ in range(rng.randint(1, 12))]
        total = rng.randint(0, sum(weights))
        parts = allocate(total, weights)
        assert sum(parts) == (total if sum(weights) else 0)
        assert all(p >= 0 for p in parts)
        # Never far from the exact proportional share.
        if sum(weights):
            for part, w in zip(parts, weights, strict=True):
                assert abs(part - total * w / sum(weights)) < 1


def test_totals_identities_hold_on_random_invoices():
    rng = random.Random(7)
    for _ in range(300):
        lines = [
            L(
                rng.randint(1, 500_000),
                rng.randint(0, 200_000),
                rng.choice([0, 500, 700, 1900, 2000, 2500]),
            )
            for _ in range(rng.randint(1, 8))
        ]
        discount = rng.choice([None, Discount("percent", rng.randint(0, 10_000))])
        t = compute(lines, discount)
        assert t.subtotal - t.discount == t.net
        assert t.net + t.tax == t.total
        assert sum(r.taxable for r in t.lines) == t.net
        assert sum(g.taxable for g in t.tax_groups) == t.net
        assert sum(r.discount for r in t.lines) == t.discount
        assert t.net >= 0
