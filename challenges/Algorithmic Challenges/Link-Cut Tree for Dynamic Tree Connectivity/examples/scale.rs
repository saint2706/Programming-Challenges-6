//! Wall time and peak heap per structure, as a markdown table.
//!
//! `cargo run --release --example scale [n]` (default 1,000,000). For every shape and structure it
//! links a whole tree, then replays a cyclic cut-and-relink stream with 10% and 90% connectivity
//! queries and with 90% queries on 64 hot vertices. Times are nanoseconds per operation; memory
//! is per vertex: the live heap after the replays (`steady`) and the high-water mark (`peak`),
//! which also counts the temporary values array a weighted structure is built from.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering::Relaxed};
use std::time::Instant;

use lct::shapes::{Op, Shape, cyclic_stream, generate, replay};
use lct::{EulerTourTree, Forest, PathLct, SubtreeLct, UfoTree};

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
        LIVE.fetch_sub(layout.size(), Relaxed);
        unsafe { System.dealloc(ptr, layout) };
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let p = unsafe { System.realloc(ptr, layout, new_size) };
        if !p.is_null() {
            let now = if new_size >= layout.size() {
                LIVE.fetch_add(new_size - layout.size(), Relaxed) + new_size - layout.size()
            } else {
                LIVE.fetch_sub(layout.size() - new_size, Relaxed) - (layout.size() - new_size)
            };
            PEAK.fetch_max(now, Relaxed);
        }
        p
    }
}

#[global_allocator]
static ALLOC: Counting = Counting;

fn time_replay<F: Forest>(f: &mut F, stream: &[Op]) -> f64 {
    replay(f, stream); // warm-up; the stream undoes itself
    let t = Instant::now();
    replay(f, stream);
    t.elapsed().as_nanos() as f64 / stream.len() as f64
}

fn row<F: Forest>(
    name: &str,
    n: usize,
    edges: &[(usize, usize)],
    streams: &[Vec<Op>; 3],
    note: impl Fn(&F) -> String,
) {
    let base = LIVE.load(Relaxed);
    PEAK.store(base, Relaxed);
    let mut f = F::new(n);
    let t = Instant::now();
    for &(u, v) in edges {
        assert!(f.link(u, v));
    }
    let build = t.elapsed().as_nanos() as f64 / edges.len() as f64;
    let times: Vec<f64> = streams.iter().map(|s| time_replay(&mut f, s)).collect();
    let steady = (LIVE.load(Relaxed) - base) as f64 / n as f64;
    let peak = (PEAK.load(Relaxed) - base) as f64 / n as f64;
    println!(
        "| {name} | {build:.0} | {:.0} | {:.0} | {:.0} | {steady:.0} / {peak:.0} | {} |",
        times[0],
        times[1],
        times[2],
        note(&f)
    );
}

fn main() {
    let n: usize = std::env::args()
        .nth(1)
        .map_or(1_000_000, |a| a.parse().expect("n"));
    let ops = (n / 10).clamp(2_000, 100_000);
    for shape in [Shape::Random, Shape::Path, Shape::Star, Shape::Binary] {
        let edges = generate(shape, n, 1);
        let streams = [(10, 0), (90, 0), (90, 64)]
            .map(|(q, hot)| cyclic_stream::<PathLct>(n, &edges, ops, q, hot, 7));
        println!("\n**{}**, n = {n}\n", shape.name());
        println!(
            "| structure | build ns/edge | churn 10% queries | 90% queries | 90% on 64 hot | heap B/vertex steady / peak | note |"
        );
        println!("|---|---|---|---|---|---|---|");
        row::<PathLct>("lct", n, &edges, &streams, |_| String::new());
        row::<SubtreeLct>("subtree-lct", n, &edges, &streams, |_| String::new());
        row::<EulerTourTree>("ett", n, &edges, &streams, |_| String::new());
        row::<UfoTree>("ufo", n, &edges, &streams, |f| {
            format!("height {}", f.height())
        });
    }
}
