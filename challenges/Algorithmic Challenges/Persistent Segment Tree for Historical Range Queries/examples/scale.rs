//! Time and retained heap for every structure, as markdown tables.
//!
//! `cargo run --release --example scale [updates|kth|lazy|gc|all]` (default `all`).
//!
//! Everything runs in one process with a counting allocator, so "bytes" is the heap a structure
//! actually keeps alive (vector capacity slack included), measured as the change in live bytes
//! while the structure exists. Instances are seeded (xorshift written out below), and while a
//! table is built the structures are cross-checked against each other: an answer that differs
//! panics instead of printing a number.

use std::alloc::{GlobalAlloc, Layout, System};
use std::hint::black_box;
use std::sync::atomic::{AtomicUsize, Ordering::Relaxed};
use std::time::{Duration, Instant};

use persegtree::lazy::Sum;
use persegtree::oracle::{distinct_by_hashing, kth_by_sorting};
use persegtree::{
    DistinctIndex, FatTree, KthIndex, LazyAdd, Persistent, PushDown, SumI64, Version, WaveletMatrix,
};

struct Counting;
static LIVE: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let p = unsafe { System.alloc(layout) };
        if !p.is_null() {
            let now = LIVE.fetch_add(layout.size(), Relaxed) + layout.size();
            PEAK.fetch_max(now, Relaxed);
        }
        p
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) };
        LIVE.fetch_sub(layout.size(), Relaxed);
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let p = unsafe { System.realloc(ptr, layout, new_size) };
        if !p.is_null() {
            if new_size >= layout.size() {
                let grew = new_size - layout.size();
                let now = LIVE.fetch_add(grew, Relaxed) + grew;
                PEAK.fetch_max(now, Relaxed);
            } else {
                LIVE.fetch_sub(layout.size() - new_size, Relaxed);
            }
        }
        p
    }
}

#[global_allocator]
static ALLOC: Counting = Counting;

/// xorshift64*: written out so a seed names the same workload in every build.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    fn below(&mut self, n: usize) -> usize {
        ((self.next() as u128 * n as u128) >> 64) as usize
    }
}

fn live() -> usize {
    LIVE.load(Relaxed)
}

fn mb(bytes: usize) -> String {
    format!("{:.1} MB", bytes as f64 / 1048576.0)
}

fn ns(d: Duration, count: usize) -> String {
    let v = d.as_secs_f64() * 1e9 / count as f64;
    if v >= 1e6 {
        format!("{:.2} ms", v / 1e6)
    } else if v >= 1e3 {
        format!("{:.2} us", v / 1e3)
    } else {
        format!("{v:.0} ns")
    }
}

fn timed<T>(f: impl FnOnce() -> T) -> (T, Duration) {
    let t = Instant::now();
    let out = f();
    (out, t.elapsed())
}

fn random_range(rng: &mut Rng, n: usize, len: usize) -> (usize, usize) {
    let len = len.min(n);
    let l = rng.below(n - len + 1);
    (l, l + len)
}

// ---------------------------------------------------------------------------------------------
// Point updates and historical sums
// ---------------------------------------------------------------------------------------------

fn updates_and_sums(n: usize, m: usize) {
    println!("\n### n = {n} positions, {m} point updates, each from the previous version\n");
    let mut rng = Rng(0x9E37_79B9_7F4A_7C15 ^ n as u64);
    let positions: Vec<usize> = (0..m).map(|_| rng.below(n)).collect();
    let values: Vec<i64> = (0..m).map(|_| rng.below(1000) as i64).collect();
    let zeros = vec![0i64; n];
    let long = n / 2;
    let queries: Vec<(usize, usize, usize)> = (0..2000)
        .map(|_| {
            let (l, r) = random_range(&mut rng, n, 64);
            (1 + rng.below(m), l, r)
        })
        .collect();
    let long_queries: Vec<(usize, usize, usize)> = (0..40)
        .map(|_| {
            let (l, r) = random_range(&mut rng, n, long);
            (1 + rng.below(m), l, r)
        })
        .collect();

    println!("| structure | update | bytes / update | sum of 64 (version random) | sum of n/2 |");
    println!("| --- | ---: | ---: | ---: | ---: |");
    let mut reference: Option<Vec<i64>> = None;
    let mut check = |name: &str, answers: Vec<i64>| match &reference {
        None => reference = Some(answers),
        Some(want) => assert_eq!(&answers, want, "{name} disagrees with the first structure"),
    };

    // Path copying, without and with reference counts.
    for counting in [false, true] {
        let mut t = Persistent::<SumI64>::from_values(&zeros, counting);
        // The baseline comes before the reservation so that the reserved room counts as what the
        // updates cost. Reserving exactly avoids the `Vec`'s doubling, which would otherwise count
        // unused capacity (up to 2x). Fat nodes cannot do this, so they keep their slack.
        let after_build = live();
        t.reserve_updates(m);
        let mut v = t.first();
        let (_, dt) = timed(|| {
            for i in 0..m {
                v = t.update(v, positions[i], values[i]);
            }
        });
        let per = (live() - after_build) as f64 / m as f64;
        let (answers, dq) = timed(|| {
            queries
                .iter()
                .map(|&(ver, l, r)| t.query(Version(ver as u32), l, r))
                .collect::<Vec<_>>()
        });
        let (long_answers, dl) = timed(|| {
            long_queries
                .iter()
                .map(|&(ver, l, r)| t.query(Version(ver as u32), l, r))
                .collect::<Vec<_>>()
        });
        let name = if counting {
            "path copying + reference counts"
        } else {
            "path copying"
        };
        println!(
            "| {name} | {} | {per:.0} | {} | {} |",
            ns(dt, m),
            ns(dq, queries.len()),
            ns(dl, long_queries.len())
        );
        check(name, answers.iter().chain(&long_answers).copied().collect());
    }

    // Fat nodes.
    {
        let mut t = FatTree::new(&zeros);
        let after_build = live();
        let (_, dt) = timed(|| {
            for i in 0..m {
                t.update(positions[i], values[i]);
            }
        });
        let per = (live() - after_build) as f64 / m as f64;
        let (answers, dq) = timed(|| {
            queries
                .iter()
                .map(|&(ver, l, r)| t.query(ver as u32, l, r))
                .collect::<Vec<_>>()
        });
        let (long_answers, dl) = timed(|| {
            long_queries
                .iter()
                .map(|&(ver, l, r)| t.query(ver as u32, l, r))
                .collect::<Vec<_>>()
        });
        println!(
            "| fat nodes | {} | {per:.0} | {} | {} |",
            ns(dt, m),
            ns(dq, queries.len()),
            ns(dl, long_queries.len())
        );
        check(
            "fat nodes",
            answers.iter().chain(&long_answers).copied().collect(),
        );
    }

    // Persistent vectors: structural sharing, but no range aggregates, so a sum is a scan.
    {
        let mut cur: imbl::Vector<i64> = zeros.iter().copied().collect();
        let after_build = live();
        let mut versions: Vec<imbl::Vector<i64>> = Vec::with_capacity(m + 1);
        versions.push(cur.clone());
        let (_, dt) = timed(|| {
            for i in 0..m {
                cur.set(positions[i], values[i]);
                versions.push(cur.clone());
            }
        });
        let per = (live() - after_build) as f64 / m as f64;
        let scan = |ver: usize, l: usize, r: usize| (l..r).map(|i| versions[ver][i]).sum::<i64>();
        let (answers, dq) = timed(|| {
            queries
                .iter()
                .map(|&(v, l, r)| scan(v, l, r))
                .collect::<Vec<_>>()
        });
        let (long_answers, dl) = timed(|| {
            long_queries
                .iter()
                .map(|&(v, l, r)| scan(v, l, r))
                .collect::<Vec<_>>()
        });
        println!(
            "| `imbl::Vector` (scan) | {} | {per:.0} | {} | {} |",
            ns(dt, m),
            ns(dq, queries.len()),
            ns(dl, long_queries.len())
        );
        check(
            "imbl",
            answers.iter().chain(&long_answers).copied().collect(),
        );
    }
    {
        let mut cur: rpds::Vector<i64> = zeros.iter().copied().collect();
        let after_build = live();
        let mut versions: Vec<rpds::Vector<i64>> = Vec::with_capacity(m + 1);
        versions.push(cur.clone());
        let (_, dt) = timed(|| {
            for i in 0..m {
                cur = cur.set(positions[i], values[i]).expect("position in range");
                versions.push(cur.clone());
            }
        });
        let per = (live() - after_build) as f64 / m as f64;
        let scan = |ver: usize, l: usize, r: usize| {
            (l..r)
                .map(|i| *versions[ver].get(i).expect("in range"))
                .sum::<i64>()
        };
        let (answers, dq) = timed(|| {
            queries
                .iter()
                .map(|&(v, l, r)| scan(v, l, r))
                .collect::<Vec<_>>()
        });
        let (long_answers, dl) = timed(|| {
            long_queries
                .iter()
                .map(|&(v, l, r)| scan(v, l, r))
                .collect::<Vec<_>>()
        });
        println!(
            "| `rpds::Vector` (scan) | {} | {per:.0} | {} | {} |",
            ns(dt, m),
            ns(dq, queries.len()),
            ns(dl, long_queries.len())
        );
        check(
            "rpds",
            answers.iter().chain(&long_answers).copied().collect(),
        );
    }

    // A full copy per version, on a few versions only (memory is n * 8 bytes each).
    {
        let copies = 200.min(m);
        let after_base = live();
        let mut versions: Vec<Vec<i64>> = vec![zeros.clone()];
        let (_, dt) = timed(|| {
            for i in 0..copies {
                let mut next = versions.last().expect("seeded").clone();
                next[positions[i]] = values[i];
                versions.push(next);
            }
        });
        let per = (live() - after_base) as f64 / (copies + 1) as f64;
        println!(
            "| full copy per version | {} | {per:.0} | - | - |",
            ns(dt, copies)
        );
    }
}

// ---------------------------------------------------------------------------------------------
// k-th smallest
// ---------------------------------------------------------------------------------------------

fn kth_table(n: usize, sigma: usize) {
    println!("\n### n = {n} numbers, {sigma} possible values\n");
    let mut rng = Rng(0xD1B5_4A32_D192_ED03 ^ (n * 31 + sigma) as u64);
    let a: Vec<i64> = (0..n).map(|_| rng.below(sigma) as i64).collect();
    let short = 1000.min(n);
    let long = n / 2;
    let make = |rng: &mut Rng, count: usize, len: usize| -> Vec<(usize, usize, usize)> {
        (0..count)
            .map(|_| {
                let (l, r) = random_range(rng, n, len);
                (l, r, rng.below(r - l))
            })
            .collect()
    };
    let qs = make(&mut rng, 20_000, short);
    let ql = make(&mut rng, 20_000, long);

    println!("| engine | build | bytes | bytes / number | k-th, range 1000 | k-th, range n/2 |");
    println!("| --- | ---: | ---: | ---: | ---: | ---: |");
    let mut expected: Vec<Option<i64>> = Vec::new();
    {
        let before = live();
        let (idx, build) = timed(|| KthIndex::new(&a));
        let bytes = live() - before;
        let (rs, ds) = timed(|| {
            qs.iter()
                .map(|&(l, r, k)| idx.kth_smallest(l, r, k))
                .collect::<Vec<_>>()
        });
        let (rl, dl) = timed(|| {
            ql.iter()
                .map(|&(l, r, k)| idx.kth_smallest(l, r, k))
                .collect::<Vec<_>>()
        });
        println!(
            "| persistent count tree | {:.0} ms | {} | {:.1} | {} | {} |",
            build.as_secs_f64() * 1e3,
            mb(bytes),
            bytes as f64 / n as f64,
            ns(ds, qs.len()),
            ns(dl, ql.len())
        );
        expected.extend(rs.iter().take(200).copied());
        expected.extend(rl.iter().take(200).copied());
    }
    {
        let mut sorted = a.clone();
        sorted.sort_unstable();
        sorted.dedup();
        let rank: Vec<u32> = a
            .iter()
            .map(|x| sorted.binary_search(x).unwrap() as u32)
            .collect();
        let before = live();
        let (wm, build) = timed(|| WaveletMatrix::new(&rank));
        let bytes = live() - before;
        let value = |r: Option<u32>| r.map(|rk| sorted[rk as usize]);
        let (rs, ds) = timed(|| {
            qs.iter()
                .map(|&(l, r, k)| value(wm.kth_smallest(l, r, k)))
                .collect::<Vec<_>>()
        });
        let (rl, dl) = timed(|| {
            ql.iter()
                .map(|&(l, r, k)| value(wm.kth_smallest(l, r, k)))
                .collect::<Vec<_>>()
        });
        println!(
            "| wavelet matrix | {:.0} ms | {} | {:.2} | {} | {} |",
            build.as_secs_f64() * 1e3,
            mb(bytes),
            bytes as f64 / n as f64,
            ns(ds, qs.len()),
            ns(dl, ql.len())
        );
        let mut got: Vec<Option<i64>> = rs.iter().take(200).copied().collect();
        got.extend(rl.iter().take(200).copied());
        assert_eq!(
            got, expected,
            "wavelet matrix disagrees with the persistent index"
        );
    }
    {
        let few_s = &qs[..200];
        let few_l = &ql[..20];
        let (rs, ds) = timed(|| {
            few_s
                .iter()
                .map(|&(l, r, k)| kth_by_sorting(&a, l, r, k))
                .collect::<Vec<_>>()
        });
        let (rl, dl) = timed(|| {
            few_l
                .iter()
                .map(|&(l, r, k)| kth_by_sorting(&a, l, r, k))
                .collect::<Vec<_>>()
        });
        assert_eq!(rs, expected[..200], "sorting disagrees on short ranges");
        assert_eq!(rl, expected[200..220], "sorting disagrees on long ranges");
        println!(
            "| sort the range | - | {} | 8.0 | {} | {} |",
            mb(n * 8),
            ns(ds, few_s.len()),
            ns(dl, few_l.len())
        );
    }
}

fn distinct_table(n: usize, sigma: usize) {
    let mut rng = Rng(0xA076_1D64_78BD_642F ^ (n + sigma) as u64);
    let a: Vec<i64> = (0..n).map(|_| rng.below(sigma) as i64).collect();
    let before = live();
    let (idx, build) = timed(|| DistinctIndex::new(&a));
    let bytes = live() - before;
    let queries: Vec<(usize, usize)> = (0..20_000)
        .map(|_| random_range(&mut rng, n, 1000.min(n)))
        .collect();
    let (got, dq) = timed(|| {
        queries
            .iter()
            .map(|&(l, r)| idx.distinct(l, r))
            .collect::<Vec<_>>()
    });
    let few = &queries[..200];
    let (want, dh) = timed(|| {
        few.iter()
            .map(|&(l, r)| distinct_by_hashing(&a, l, r))
            .collect::<Vec<_>>()
    });
    assert_eq!(&got[..200], &want[..]);
    println!(
        "| {n} | {sigma} | {:.0} ms | {} | {} | {} | {} |",
        build.as_secs_f64() * 1e3,
        mb(bytes),
        idx.nodes(),
        ns(dq, queries.len()),
        ns(dh, few.len())
    );
}

// ---------------------------------------------------------------------------------------------
// Lazy range updates
// ---------------------------------------------------------------------------------------------

fn lazy_table(n: usize, m: usize) {
    println!(
        "\n### n = {n} elements, {m} random range updates (range length up to n), sum queries\n"
    );
    let mut rng = Rng(0xE703_7ED1_A0B4_28DB);
    let ops: Vec<(usize, usize, i64, bool)> = (0..m)
        .map(|_| {
            let (a, b) = (rng.below(n + 1), rng.below(n + 1));
            (
                a.min(b),
                a.max(b).max(a.min(b) + 1).min(n),
                rng.below(100) as i64 - 50,
                rng.below(2) == 0,
            )
        })
        .collect();
    let qs: Vec<(usize, usize, usize)> = (0..20_000)
        .map(|_| {
            let len = 1 + rng.below(n);
            let (l, r) = random_range(&mut rng, n, len);
            (1 + rng.below(m), l, r)
        })
        .collect();
    let init = vec![0i64; n];

    println!("| variant | update | nodes / update | bytes / update | sum query |");
    println!("| --- | ---: | ---: | ---: | ---: |");
    let mut reference: Option<Vec<i64>> = None;
    let mut check = |name: &str, answers: Vec<i64>| match &reference {
        None => reference = Some(answers),
        Some(want) => assert_eq!(&answers, want, "{name} disagrees"),
    };

    {
        let mut t = LazyAdd::<Sum>::new(&init, false);
        let (nodes0, bytes0) = (t.live_nodes(), live());
        let mut v = t.first();
        let (_, dt) = timed(|| {
            for &(l, r, d, _) in &ops {
                v = t.range_add(v, l, r, d);
            }
        });
        let (nodes, bytes) = (t.live_nodes() - nodes0, live() - bytes0);
        let (answers, dq) = timed(|| {
            qs.iter()
                .map(|&(ver, l, r)| t.query(Version(ver as u32), l, r))
                .collect::<Vec<_>>()
        });
        println!(
            "| add, permanent marks | {} | {:.1} | {:.0} | {} |",
            ns(dt, m),
            nodes as f64 / m as f64,
            bytes as f64 / m as f64,
            ns(dq, qs.len())
        );
        check("permanent marks", answers);
    }
    {
        let mut t = PushDown::new(&init, false);
        let (nodes0, bytes0) = (t.live_nodes(), live());
        let mut v = t.first();
        let (_, dt) = timed(|| {
            for &(l, r, d, _) in &ops {
                v = t.range_add(v, l, r, d);
            }
        });
        let (nodes, bytes) = (t.live_nodes() - nodes0, live() - bytes0);
        let (answers, dq) = timed(|| {
            qs.iter()
                .map(|&(ver, l, r)| t.query(Version(ver as u32), l, r))
                .collect::<Vec<_>>()
        });
        println!(
            "| add, push-down | {} | {:.1} | {:.0} | {} |",
            ns(dt, m),
            nodes as f64 / m as f64,
            bytes as f64 / m as f64,
            ns(dq, qs.len())
        );
        check("push-down add", answers);
    }
    {
        let mut t = PushDown::new(&init, false);
        let (nodes0, bytes0) = (t.live_nodes(), live());
        let mut v = t.first();
        let (_, dt) = timed(|| {
            for &(l, r, d, assign) in &ops {
                v = if assign {
                    t.range_assign(v, l, r, d)
                } else {
                    t.range_add(v, l, r, d)
                };
            }
        });
        let (nodes, bytes) = (t.live_nodes() - nodes0, live() - bytes0);
        let (answers, dq) = timed(|| {
            qs.iter()
                .map(|&(ver, l, r)| t.query(Version(ver as u32), l, r))
                .collect::<Vec<_>>()
        });
        black_box(answers);
        println!(
            "| half add, half assign, push-down | {} | {:.1} | {:.0} | {} |",
            ns(dt, m),
            nodes as f64 / m as f64,
            bytes as f64 / m as f64,
            ns(dq, qs.len())
        );
    }
}

// ---------------------------------------------------------------------------------------------
// Garbage collection
// ---------------------------------------------------------------------------------------------

fn gc_table(n: usize, m: usize) {
    println!("\n### n = {n} positions, {m} point updates, a window of recent versions retained\n");
    let mut rng = Rng(0x8EBC_6AF0_9C88_C6E3);
    let positions: Vec<usize> = (0..m).map(|_| rng.below(n)).collect();
    let zeros = vec![0i64; n];
    println!("| retention | update | live nodes | arena slots | arena bytes |");
    println!("| --- | ---: | ---: | ---: | ---: |");

    // No collection at all.
    for counting in [false, true] {
        let mut t = Persistent::<SumI64>::from_values(&zeros, counting);
        t.reserve_updates(m);
        let mut v = t.first();
        let (_, dt) = timed(|| {
            for (i, &p) in positions.iter().enumerate() {
                v = t.update(v, p, i as i64);
            }
        });
        let s = t.stats();
        let name = if counting {
            "keep every version, reference counts on"
        } else {
            "keep every version"
        };
        println!(
            "| {name} | {} | {} | {} | {} |",
            ns(dt, m),
            s.live_nodes,
            s.slots,
            mb(s.bytes)
        );
    }
    // Reference-counted window.
    for window in [1usize, 16, 256, 4096] {
        let mut t = Persistent::<SumI64>::from_values(&zeros, true);
        let mut live_versions: std::collections::VecDeque<Version> = [t.first()].into();
        let (_, dt) = timed(|| {
            for (i, &p) in positions.iter().enumerate() {
                let v = t.update(*live_versions.back().expect("nonempty"), p, i as i64);
                live_versions.push_back(v);
                if live_versions.len() > window {
                    t.retire(live_versions.pop_front().expect("nonempty"));
                }
            }
        });
        let s = t.stats();
        println!(
            "| last {window}, reference counts | {} | {} | {} | {} |",
            ns(dt, m),
            s.live_nodes,
            s.slots,
            mb(s.bytes)
        );
    }
    // Mark and compact: no counts, retire in bulk, compact once.
    {
        let mut t = Persistent::<SumI64>::from_values(&zeros, false);
        let mut v = t.first();
        let mut all = vec![v];
        for (i, &p) in positions.iter().enumerate() {
            v = t.update(v, p, i as i64);
            all.push(v);
        }
        let before = t.stats();
        let keep = all.len() - 16;
        for &old in &all[..keep] {
            t.retire(old);
        }
        let (_, dt) = timed(|| t.compact());
        let after = t.stats();
        println!(
            "| last 16, retire in bulk then compact once ({} to {} slots in {:.0} ms) | - | {} | {} | {} |",
            before.slots,
            after.slots,
            dt.as_secs_f64() * 1e3,
            after.live_nodes,
            after.slots,
            mb(after.bytes)
        );
        assert_eq!(t.query(v, 0, n), {
            let mut a = vec![0i64; n];
            for (i, &p) in positions.iter().enumerate() {
                a[p] = i as i64;
            }
            a.iter().sum::<i64>()
        });
    }
}

fn main() {
    let which = std::env::args().nth(1).unwrap_or_else(|| "all".into());
    if matches!(which.as_str(), "updates" | "all") {
        println!("## Point updates and historical sums");
        updates_and_sums(1 << 16, 1 << 16);
        updates_and_sums(1 << 20, 1 << 18);
    }
    if matches!(which.as_str(), "kth" | "all") {
        println!("\n## Range k-th smallest");
        kth_table(1 << 16, 1 << 16);
        kth_table(1 << 20, 1 << 20);
        kth_table(1 << 20, 256);
        println!(
            "\n### Distinct values in a range of 1000 (persistent index, online) against hashing 200 ranges\n"
        );
        println!("| n | values | build | bytes | nodes | query | hashing |");
        println!("| ---: | ---: | ---: | ---: | ---: | ---: | ---: |");
        distinct_table(1 << 16, 1 << 16);
        distinct_table(1 << 20, 1 << 20);
        distinct_table(1 << 20, 1000);
    }
    if matches!(which.as_str(), "lazy" | "all") {
        println!("\n## Range updates");
        lazy_table(1 << 16, 1 << 16);
        lazy_table(1 << 20, 1 << 17);
    }
    if matches!(which.as_str(), "gc" | "all") {
        println!("\n## Garbage collection");
        gc_table(1 << 16, 1 << 18);
    }
}
