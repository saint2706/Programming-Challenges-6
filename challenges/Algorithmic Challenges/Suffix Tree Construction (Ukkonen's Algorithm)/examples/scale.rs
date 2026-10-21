//! Build time, peak heap and query time for every algorithm, as markdown tables.
//!
//! `uv run --no-project python scripts/fetch_corpus.py` first for the real texts, then
//! `cargo run --release --example scale [SIZE ...]` (default 1,000,000 4,000,000 bytes).
//!
//! For each text prefix of SIZE bytes it times: the suffix array alone (own SA-IS and the
//! `libsais` C library, single-threaded); the three suffix-tree builders (Ukkonen with sibling
//! lists and with a hash map, SA-IS + LCP + tree, and PWOTD with and without its partition);
//! and then 200,000 pattern searches on the tree against a binary search on the suffix array.
//! Heap numbers are the high-water mark of live bytes while a builder runs, per input byte,
//! counting the text it owns. A last table shows top-down construction going quadratic.

use std::alloc::{GlobalAlloc, Layout, System};
use std::hint::black_box;
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering::Relaxed};
use std::time::Instant;

use libsais::SuffixArrayConstruction;
use sufftree::array_index::SuffixArrayIndex;
use sufftree::lcp::lcp_phi;
use sufftree::pwotd;
use sufftree::sais::suffix_array_u8;
use sufftree::satree::sa_lcp_bytes;
use sufftree::texts::{fibonacci, run, uniform};
use sufftree::ukkonen::{HashChildren, ListChildren, Ukkonen};
use sufftree::{Bytes, SuffixTree};

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

/// Run `f`, returning its result, seconds taken and peak extra heap bytes while it ran.
fn measure<T>(f: impl FnOnce() -> T) -> (T, f64, usize) {
    let base = LIVE.load(Relaxed);
    PEAK.store(base, Relaxed);
    let t = Instant::now();
    let out = f();
    let secs = t.elapsed().as_secs_f64();
    (out, secs, PEAK.load(Relaxed) - base)
}

fn libsais_sa(text: &[u8]) -> usize {
    let sa: Vec<i32> = SuffixArrayConstruction::for_text(text)
        .in_owned_buffer()
        .single_threaded()
        .run()
        .expect("libsais")
        .into_vec();
    sa.len()
}

// The five ways of getting a frozen suffix tree.

fn ukkonen_list(bytes: &[u8]) -> SuffixTree<Bytes> {
    let text = Bytes::new(bytes);
    Ukkonen::<ListChildren>::build(&text).freeze(text)
}

fn ukkonen_hash(bytes: &[u8]) -> SuffixTree<Bytes> {
    let text = Bytes::new(bytes);
    Ukkonen::<HashChildren>::build(&text).freeze(text)
}

fn via_sais(bytes: &[u8]) -> SuffixTree<Bytes> {
    let (sa, lcp) = sa_lcp_bytes(bytes);
    SuffixTree::from_sa_lcp(Bytes::new(bytes), &sa, &lcp)
}

fn top_down(bytes: &[u8]) -> SuffixTree<Bytes> {
    pwotd::build(Bytes::new(bytes))
}

fn plain_top_down(bytes: &[u8]) -> SuffixTree<Bytes> {
    pwotd::build_with(Bytes::new(bytes), Some(0))
}

struct Corpus {
    name: String,
    bytes: Vec<u8>,
}

impl Corpus {
    /// Top-down construction is quadratic on highly repetitive text; the last table shows it.
    fn skips_top_down(&self) -> bool {
        self.name == "fibonacci" && self.bytes.len() > 100_000
    }
}

fn load(size: usize) -> Vec<Corpus> {
    let mut out = Vec::new();
    for name in ["dna", "english", "sources", "proteins"] {
        if let Ok(all) = std::fs::read(Path::new("data").join(name)) {
            out.push(Corpus {
                name: name.to_string(),
                bytes: all[..size.min(all.len())].to_vec(),
            });
        }
    }
    out.push(Corpus {
        name: "uniform4".into(),
        bytes: uniform(size, 4, 1),
    });
    out.push(Corpus {
        name: "fibonacci".into(),
        bytes: fibonacci(size),
    });
    out
}

fn patterns(text: &[u8], count: usize, len: usize) -> Vec<&[u8]> {
    let step = (text.len() - len) / count;
    (0..count)
        .map(|i| &text[i * step..i * step + len])
        .collect()
}

fn time_table(corpora: &[Corpus]) {
    println!(
        "| text | SA-IS (own) | libsais | Ukkonen list | Ukkonen hash | SA+LCP+tree | PWOTD | WOTD | nodes/byte |"
    );
    println!("|---|---|---|---|---|---|---|---|---|");
    for c in corpora {
        let (_, own, _) = measure(|| suffix_array_u8(&c.bytes));
        let (_, lib, _) = measure(|| libsais_sa(&c.bytes));
        let (list, t_list, _) = measure(|| ukkonen_list(&c.bytes));
        let (hash, t_hash, _) = measure(|| ukkonen_hash(&c.bytes));
        let (tree, t_sa, _) = measure(|| via_sais(&c.bytes));
        assert_eq!(
            list.node_count(),
            tree.node_count(),
            "{}: Ukkonen disagrees",
            c.name
        );
        assert_eq!(
            hash.node_count(),
            tree.node_count(),
            "{}: Ukkonen hash disagrees",
            c.name
        );
        let (pw, wotd) = if c.skips_top_down() {
            ("(quadratic)".to_string(), "(quadratic)".to_string())
        } else {
            let (a, t_pw, _) = measure(|| top_down(&c.bytes));
            let (b, t_wotd, _) = measure(|| plain_top_down(&c.bytes));
            assert_eq!(
                a.node_count(),
                tree.node_count(),
                "{}: PWOTD disagrees",
                c.name
            );
            assert_eq!(
                b.node_count(),
                tree.node_count(),
                "{}: WOTD disagrees",
                c.name
            );
            (format!("{t_pw:.2}"), format!("{t_wotd:.2}"))
        };
        println!(
            "| {} | {own:.2} | {lib:.2} | {t_list:.2} | {t_hash:.2} | {t_sa:.2} | {pw} | {wotd} | {:.2} |",
            c.name,
            tree.node_count() as f64 / c.bytes.len() as f64
        );
    }
}

fn memory_table(corpora: &[Corpus]) {
    println!(
        "\n| text | Ukkonen list: live tree | Ukkonen list: peak | Ukkonen hash: peak | SA+LCP+tree: peak | PWOTD: peak | frozen tree + text | own SA-IS + LCP: peak | libsais: peak |"
    );
    println!("|---|---|---|---|---|---|---|---|---|");
    for c in corpora {
        let n = c.bytes.len() as f64;
        let live = {
            let text = Bytes::new(c.bytes.as_slice());
            Ukkonen::<ListChildren>::build(&text).heap_bytes()
        };
        let (_, _, m_list) = measure(|| ukkonen_list(&c.bytes));
        let (_, _, m_hash) = measure(|| ukkonen_hash(&c.bytes));
        let (tree, _, m_sa) = measure(|| via_sais(&c.bytes));
        let m_pw = if c.skips_top_down() {
            "n/a".to_string()
        } else {
            format!("{:.0}", measure(|| top_down(&c.bytes)).2 as f64 / n)
        };
        let (_, _, m_own) = measure(|| {
            let sa = suffix_array_u8(&c.bytes);
            lcp_phi(&c.bytes, &sa).len()
        });
        let (_, _, m_lib) = measure(|| libsais_sa(&c.bytes));
        println!(
            "| {} | {:.0} | {:.0} | {:.0} | {:.0} | {m_pw} | {:.0} | {:.0} | {:.0} |",
            c.name,
            live as f64 / n,
            m_list as f64 / n,
            m_hash as f64 / n,
            m_sa as f64 / n,
            (tree.heap_bytes() + c.bytes.len()) as f64 / n,
            m_own as f64 / n,
            m_lib as f64 / n
        );
    }
}

fn query_table(corpora: &[Corpus]) {
    println!("\n| text | suffix tree ns/query | suffix array binary search ns/query |");
    println!("|---|---|---|");
    for c in corpora {
        let tree = via_sais(&c.bytes);
        let index = SuffixArrayIndex::new(c.bytes.clone());
        let pats = patterns(&c.bytes, 200_000, 16);
        let (_, q_tree, _) =
            measure(|| pats.iter().map(|p| black_box(tree.count(p))).sum::<usize>());
        let (_, q_sa, _) = measure(|| {
            pats.iter()
                .map(|p| black_box(index.count(p)))
                .sum::<usize>()
        });
        println!(
            "| {} | {:.0} | {:.0} |",
            c.name,
            q_tree * 1e9 / pats.len() as f64,
            q_sa * 1e9 / pats.len() as f64
        );
    }
}

type Maker = fn(usize) -> Vec<u8>;

fn quadratic_table() {
    println!("\n### Top-down construction on repetitive text (seconds)\n");
    println!("| text | n | Ukkonen list | SA+LCP+tree | PWOTD |");
    println!("|---|---|---|---|---|");
    let makers: [(&str, Maker); 2] = [("fibonacci", fibonacci), ("run of one symbol", run)];
    for (name, make) in makers {
        for n in [4_000usize, 8_000, 16_000, 32_000] {
            let bytes = make(n);
            let (_, t_ukk, _) = measure(|| ukkonen_list(&bytes));
            let (_, t_sa, _) = measure(|| via_sais(&bytes));
            let (_, t_pw, _) = measure(|| top_down(&bytes));
            println!("| {name} | {n} | {t_ukk:.4} | {t_sa:.4} | {t_pw:.4} |");
        }
    }
}

fn main() {
    let sizes: Vec<usize> = std::env::args()
        .skip(1)
        .map(|a| a.parse().expect("size"))
        .collect();
    let sizes = if sizes.is_empty() {
        vec![1_000_000, 4_000_000]
    } else {
        sizes
    };
    for size in sizes {
        println!("\n### {size} bytes (seconds)\n");
        let corpora = load(size);
        time_table(&corpora);
        memory_table(&corpora);
        query_table(&corpora);
    }
    quadratic_table();
}
