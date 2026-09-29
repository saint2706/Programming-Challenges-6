//! Three things an order-statistic tree does that `BTreeMap` cannot.

use std::collections::BTreeMap;
use std::time::Instant;

use ost::OrderStatBTree;
use ost::rng::SplitMix64;

fn main() {
    leaderboard();
    sliding_window_median();
    why_std_btreemap_is_not_enough();
}

/// Rank / percentile queries on a live score table.
fn leaderboard() {
    println!("== leaderboard ==");
    let mut rng = SplitMix64::new(42);
    let mut scores: OrderStatBTree<u32> = OrderStatBTree::new();
    for _ in 0..100_000 {
        scores.insert(rng.below(10_000) as u32);
    }
    let mine = 9_000;
    let better = scores.len() - scores.rank_upper(&mine);
    println!(
        "score {mine}: {better} players are strictly better; percentile {:.1}",
        100.0 * scores.rank(&mine) as f64 / scores.len() as f64
    );
    let n = scores.len();
    for (label, q) in [("p50", 0.5), ("p90", 0.9), ("p99", 0.99)] {
        println!(
            "{label} = {}",
            scores.select(((n - 1) as f64 * q) as usize).unwrap()
        );
    }
    println!(
        "players scoring 4000..=4100: {} (counted, not iterated)",
        scores.count_range(4000..=4100)
    );
    println!("height {} for {n} keys", scores.height());
}

/// Median of the last `w` values in O(log w) per step.
fn sliding_window_median() {
    println!("\n== sliding-window median ==");
    let stream: Vec<u32> = vec![5, 2, 8, 1, 9, 3, 7, 4, 6, 10, 0, 11];
    let w = 5;
    let mut win: OrderStatBTree<u32, 4> = OrderStatBTree::new();
    for (i, &x) in stream.iter().enumerate() {
        win.insert(x);
        if i >= w {
            win.remove(&stream[i - w]);
        }
        if i + 1 >= w {
            println!(
                "window ending at {i}: median {}",
                win.select(w / 2).unwrap()
            );
        }
    }
}

/// `BTreeMap` has no rank: the best it can do is walk.
fn why_std_btreemap_is_not_enough() {
    println!("\n== rank on 1M keys: order-stat B-tree vs std BTreeMap ==");
    let n = 1_000_000u32;
    let tree = OrderStatBTree::<u32, 16>::from_sorted((0..n).collect());
    let map: BTreeMap<u32, ()> = (0..n).map(|k| (k, ())).collect();
    let probe = 750_000;

    let t = Instant::now();
    let r1 = tree.rank(&probe);
    let fast = t.elapsed();

    let t = Instant::now();
    let r2 = map.range(..probe).count();
    let slow = t.elapsed();

    assert_eq!(r1, r2);
    println!(
        "rank({probe}) = {r1}: {fast:?} vs {slow:?} ({:.0}x)",
        slow.as_secs_f64() / fast.as_secs_f64().max(1e-9)
    );
}
