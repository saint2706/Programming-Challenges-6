# Josephus Problem Solver

**Category:** Algorithmic Challenges
**Difficulty:** B (brief: "simulate with a circular list, then derive the closed form")

**Status:** Implemented (Python)

n people stand in a circle; every k-th is removed until one remains. The brief
says simulate, then derive the closed form — and the honest answer to the
second half is that **the closed form only exists for k = 2**. For general k
there is a recurrence, and a much faster algorithm built on it, but no known
closed form; finding one is [still an open
problem](https://arxiv.org/pdf/2411.16696).

So instead of one derivation, this is a ladder of five, each earning its place:

| Method                | Time       | Space | Gives you                                                 |
| --------------------- | ---------- | ----- | --------------------------------------------------------- |
| `simulate`            | O(n·k)     | O(n)  | Full death order; the oracle everything is tested against |
| `elimination_order`   | O(n log n) | O(n)  | Full death order, k-independent                           |
| `survivor_recurrence` | O(n)       | O(1)  | The survivor                                              |
| `survivor_fast`       | O(k log n) | O(1)  | The survivor, for astronomical n                          |
| `survivor_pow2`       | O(1)       | O(1)  | The survivor, k = 2 only                                  |

`survivor(n, k)` dispatches to whichever is cheapest.

## The k = 2 closed form is a bit rotation

With n = 2^m + L, the survivor is J(n) = 2L + 1. Written in binary that's
something better than a formula — **move the leading 1 bit to the end**:

```
n = 1 0 1 1 0   (22)
J = 0 1 1 0 1   (13)
```

Which collapses to one subtraction, because shifting left and setting the low
bit gives 2n + 1, moving the leading 1 from 2^m to 2^(m+1):

```python
j = ((n << 1) | 1) - (1 << n.bit_length())
```

*Why* 2L + 1: after one lap around 2^m + L people, exactly L have died and the
count resumes at position 2L + 1 with a power-of-two circle left. And in a
power-of-two circle the person the count starts on always survives — each lap
halves the circle while keeping that person first, all the way down to one.

It costs O(1) at any size, which means it works on integers no simulation could
touch:

```python
survivor_pow2(10**1000 + 12345)   # instant
```

## The general-k recurrence, and why it's true

```
J(1) = 0
J(i) = (J(i-1) + k) mod i
```

The derivation is one observation. After the first person dies — position k−1,
counting from 0 — what's left is *the same problem* with n−1 people, except the
circle now starts at position k. Relabel that smaller circle to start at 0 and
every answer shifts back by k; undoing the relabelling is exactly
`(J(n-1) + k) mod n`.

## Making it O(k log n)

The recurrence walks one person at a time, which is wasteful: while i is much
larger than k, whole laps of the circle behave identically and can be jumped in
one arithmetic step. The [iterative form of that
idea](https://arxiv.org/pdf/2411.16696) is startlingly small:

```
x = 0
repeat  x ← x + ⌊x / (k−1)⌋ + 1  until x ≥ n(k−1)
survivor (0-indexed) = n·k − x − 1
```

Each step multiplies x by roughly k/(k−1), so **n enters the cost only
logarithmically** — n can be astronomical as long as k is not. It's also
iterative rather than recursive, so there's no stack depth to blow at n = 10^18:

```
Method                         n       k       time   survivor
simulate                 200,000       7    0.0697s   153463
survivor_recurrence      200,000       7    0.0137s   153463
survivor_fast            200,000       7    0.0000s   153463
survivor_fast              10^18       7    0.0001s   459717405259920305
survivor_pow2              10^18       2    0.0000s   847078495393153025
survivor                 10^1000       3    0.0050s   438615908...09234016
```

The catch is sharper than it looks, and it cost me a hung test suite — see
below. `survivor` dispatches around it.

## The whole permutation in O(n log n)

Simulation's O(n·k) is fine for k = 3 and hopeless for k = 10^6. But the circle
is only ever asked one question — *"who is the j-th person still standing?"* —
which is exactly the `select` an order-statistic structure provides. A Fenwick
tree over n slots each holding 1 gives `remove` and `select` in O(log n), and
**k drops out of the complexity entirely**.

The `select` is worth a look. The naive version binary-searches on prefix sums
for O(log² n); this one descends the tree directly by binary lifting, taking
each power-of-two jump whose subtree doesn't yet cover j — a single O(log n)
pass over the same array. Initialization is O(n) rather than n insertions,
because node i covers exactly `i & -i` slots and all of them hold 1:

```python
self.tree = [i & -i for i in range(n + 1)]
```

### But the asymptotically better version loses — until it doesn't

```
      n                k    simulate     Fenwick   winner
100,000                7      0.020s      0.322s   simulate
100,000            1,000      0.088s      0.329s   simulate
 50,000           25,000      0.187s      0.150s   Fenwick
100,000    1,000,000,000      0.655s      0.323s   Fenwick
```

`deque.rotate` is a C `memmove` while the Fenwick tree is interpreted Python,
so the O(n·k) version stays ahead until k reaches roughly n/2. Note the fourth
row: `rotate` costs `min(k mod len, len − k mod len)`, so a *huge* k is no worse
than a random one — it saturates rather than exploding. That's precisely where
the k-independent algorithm takes over, and precisely why "asymptotically
better" needed measuring rather than assuming.

## Generalizations

**Arbitrary starting position.** Counting can begin anywhere; the answer just
rotates. `start=s` is supported on every method and cross-validated for every
s in 1..n.

**Multiple survivors.** The same relabelling argument that gives the
single-survivor recurrence works on a whole set at once — run m positions
through it in lockstep, starting from the m that remain at the end. Which
settles the historical question:

```python
>>> survivors(41, 3, 2)
[16, 31]
```

Josephus and 40 others, every third man killed, two meant to live. Positions
16 and 31.

## Watching it happen

```
$ uv run python josephus.py 12 3 --animate
n=12 k=3: survivor is position 10
           start   1  2  3  4  5  6  7  8  9 10 11 12
    1     kill 3   1  2 --  4  5  6  7  8  9 10 11 12
    2     kill 6   1  2 --  4  5 --  7  8  9 10 11 12
    3     kill 9   1  2 --  4  5 --  7  8 -- 10 11 12
    4    kill 12   1  2 --  4  5 --  7  8 -- 10 11 --
    5     kill 4   1  2 -- --  5 --  7  8 -- 10 11 --
    6     kill 8   1  2 -- --  5 --  7 -- -- 10 11 --
    7     kill 1  --  2 -- --  5 --  7 -- -- 10 11 --
    8     kill 7  --  2 -- --  5 -- -- -- -- 10 11 --
    9     kill 2  -- -- -- --  5 -- -- -- -- 10 11 --
   10    kill 11  -- -- -- --  5 -- -- -- -- 10 -- --
   11     kill 5  -- -- -- -- -- -- -- -- -- 10 -- --
   12 survivor 10  -- -- -- -- -- -- -- -- -- -- -- --
```

## Correctness

`simulate` is the oracle: a `deque` doing literally what the problem says. Every
other method is checked against it exhaustively —

```
$ uv run python josephus.py --verify
all methods agree for n in 1..120, k in 1..20, and every start position
```

— which is 2,400 (n, k) pairs × 5 methods, plus the full elimination order, plus
1, 2 and 3 survivors, plus every starting position for a sample of circles. The
pytest suite adds 68 more: the bit-rotation identity for every n < 5000, k > n,
k = 1, n = 1, random agreement over 400 cases, and a check that `survivor_fast`
really is logarithmic in n (10^12 must cost under 2.5× what 10^6 does — the
test that would catch a hidden loop over n).

## The edge case that hung my own test suite

`survivor_fast` is O(k log n), and I had written the cost off as "fine, that's
what `survivor()` dispatches around". Then a test called
`survivor_fast(2, 10**12)` directly and pytest ran for ten minutes before I
killed it.

The arithmetic: x grows from k−1 to n(k−1) — a factor of n — at a rate of
k/(k−1) per step, so the real cost is **(k−1)·ln(n)** iterations. For n = 2,
k = 10¹² that is 7 × 10¹¹ steps. Not slow; never.

Three changes came out of it:

1. **Skip the linear prefix.** While x < k−1 the update is exactly `x += 1`, so
   a naive loop burns its first k−1 iterations counting one at a time. Seeding
   x past that analytically is two lines, and turns `survivor_fast(1, 10**12)`
   from an unbounded walk into three arithmetic operations.
2. **Refuse rather than hang.** `max_steps` (default 10⁷) estimates the work up
   front and raises with a message pointing at `survivor()`. Pass `None` to
   insist. A public function that silently runs for four minutes is a worse
   failure than one that declines.
3. **Fix the dispatch estimate.** It had been `(k−1)·ln(n·k)`, which
   over-charges the fast path. `survivor()` now uses the accurate model and
   passes `max_steps=None`, because its own comparison *is* the guard.

Separately: `start` wraps rather than being range-checked — `start=0` and
`start=n` name the same person — and all five methods agree on that over the
whole range −2n .. 3n, which is now a test.

## Where this is actually used

Honestly: nobody has the literal Josephus problem at work. It is a puzzle with a
two-thousand-year-old cover story. What is worth having is the machinery
underneath, and two pieces of it transfer directly.

**"Remove the k-th remaining element", repeatedly.** The Fenwick tree behind
`elimination_order` answers *find and delete the k-th survivor* in O(log n),
which is a common primitive with a shortage of good names:

- sampling without replacement in a prescribed order, and dealing algorithms
  that count rather than index — the "under-down" deal is a Josephus process,
  which is why a magician's stacked deck is computable in closed form;
- fair queueing and round-robin scheduling where participants leave: token
  passing on a ring, leader election over a shrinking membership, and load
  shedding that has to stay deterministic across replicas;
- editor and undo-system operations phrased as "the k-th remaining line", where
  the naive list rebuild costs O(n) per deletion;
- rank and select over a dynamic set generally — the same structure as an
  order-statistic tree, and the same Fenwick tree that
  [Count Inversions](../Count%20Inversions%20in%20an%20Array/) uses for an
  entirely different question.

**Jumping a recurrence over its constant stretches.** `survivor_fast` gets from
O(n) to O(k log n) by noticing that `J(i) = (J(i-1) + k) mod i` only wraps
occasionally, and computing *where* the next wrap happens rather than stepping
to it. That is a general technique for any recurrence with long arithmetic
stretches between irregularities, and it is the only reason the survivor of a
circle of 10¹⁰⁰⁰ people is computable at all.

**And the k = 2 bit rotation earns its place as a demonstration.** A closed form
that turns out to be "move the leading 1 bit to the end" is the cleanest small
argument for thinking in binary representations — the same instinct that finds
`x & (x-1)`, `i & -i` (which the Fenwick tree here runs on), and popcount
tricks.

Where circular counting does appear literally: consistent-hashing rings,
round-robin DNS and load balancers all walk a circle this way. They rarely
remove on a count, so they need the walk and not the survivor formula.

## Run it

```bash
cd "challenges/Algorithmic Challenges/Josephus Problem Solver"

uv run python josephus.py 41 3                     # survivor is position 31
uv run python josephus.py 41 3 --survivors 2        # [16, 31]
uv run python josephus.py 12 3 --animate
uv run python josephus.py 1000000000000000000 7     # instant
uv run python josephus.py 100 7 --order
uv run python josephus.py --verify
uv run python josephus.py --benchmark

uv run --with pytest pytest -q                      # 68 tests
```

Standard library only. `--method {auto,simulate,recurrence,fast,pow2}` forces a
particular ladder rung if you want to compare them by hand.

## Sources

- [Josephus problem — Algorithms for Competitive Programming](https://cp-algorithms.com/others/josephus_problem.html)
- [A New O(k log n) Algorithm for the Josephus Problem](https://arxiv.org/pdf/2411.16696)
