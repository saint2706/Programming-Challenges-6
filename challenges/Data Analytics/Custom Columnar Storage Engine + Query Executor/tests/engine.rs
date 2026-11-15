//! End-to-end checks: every query result is compared with the same answer computed by plain
//! row-at-a-time Rust over the generated data.

use std::collections::{BTreeMap, HashMap};
use std::path::PathBuf;

use colstore::encoding;
use colstore::format::{self, Table};
use colstore::{Batch, Catalog, Column, QueryResult, Value, query};

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
}

fn tmpdir(tag: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("colstore-{tag}-{}", std::process::id()));
    std::fs::create_dir_all(&d).unwrap();
    d
}

struct Data {
    id: Vec<i64>,
    region: Vec<String>,
    qty: Vec<i64>,
    price: Vec<f64>,
    day: Vec<String>,
    cust: Vec<i64>,
    cust_id: Vec<i64>,
    cust_name: Vec<String>,
}

const REGIONS: [&str; 4] = ["east", "north", "south", "west"];

fn setup(tag: &str) -> (Catalog, Data) {
    let mut r = Rng(42);
    let n = 20_000;
    let d = Data {
        id: (0..n).collect(),
        region: (0..n)
            .map(|_| REGIONS[r.below(4) as usize].to_string())
            .collect(),
        qty: (0..n).map(|_| r.below(50) as i64 + 1).collect(),
        price: (0..n).map(|_| (r.below(100_000) as f64) / 100.0).collect(),
        day: (0..n)
            .map(|i| format!("2024-{:02}-{:02}", 1 + i * 12 / n, 1 + r.below(28)))
            .collect(),
        cust: (0..n).map(|_| r.below(300) as i64).collect(),
        cust_id: (0..250).collect(), // customers 250..299 have no row: inner join drops them
        cust_name: (0..250).map(|i| format!("c{i:03}")).collect(),
    };
    let dir = tmpdir(tag);
    let sales = Batch {
        names: ["id", "region", "qty", "price", "day", "cust"]
            .map(String::from)
            .to_vec(),
        columns: vec![
            Column::Int(d.id.clone()),
            Column::Str(d.region.clone()),
            Column::Int(d.qty.clone()),
            Column::Float(d.price.clone()),
            Column::Str(d.day.clone()),
            Column::Int(d.cust.clone()),
        ],
    };
    let custs = Batch {
        names: vec!["cust_id".into(), "cust_name".into()],
        columns: vec![
            Column::Int(d.cust_id.clone()),
            Column::Str(d.cust_name.clone()),
        ],
    };
    format::write(&dir.join("sales.col"), &sales, 1000).unwrap();
    format::write(&dir.join("custs.col"), &custs, 64).unwrap();
    let cat = Catalog::from([
        ("sales".into(), dir.join("sales.col")),
        ("custs".into(), dir.join("custs.col")),
    ]);
    (cat, d)
}

fn rows(cat: &Catalog, sql: &str) -> (Batch, colstore::exec::Stats) {
    match query(cat, sql).unwrap_or_else(|e| panic!("{sql}: {e}")) {
        QueryResult::Rows(b, s) => (b, s),
        QueryResult::Explain(_) => panic!("expected rows"),
    }
}

fn close(a: f64, b: f64) -> bool {
    (a - b).abs() <= 1e-9 * a.abs().max(b.abs()).max(1.0)
}

#[test]
fn encodings_round_trip_and_pick_the_smallest() {
    let mut r = Rng(7);
    let cases: Vec<(Column, &str)> = vec![
        (
            Column::Int((0..5000).map(|i| 1_000_000 + i * 3).collect()),
            "delta",
        ),
        (Column::Int(vec![7; 5000]), "for"),
        (
            Column::Int((0..5000).map(|i| (i / 500) * 1_000_000_000_000).collect()),
            "rle",
        ),
        (
            Column::Int((0..5000).map(|_| r.below(1000) as i64 - 500).collect()),
            "for",
        ),
        (
            Column::Int(
                (0..5000)
                    .map(|_| [i64::MIN, 0, i64::MAX][r.below(3) as usize])
                    .collect(),
            ),
            "dict",
        ),
        (
            Column::Str((0..5000).map(|i| REGIONS[i % 4].to_string()).collect()),
            "dict",
        ),
        (
            Column::Str((0..500).map(|i| format!("unique-{i}")).collect()),
            "plain",
        ),
        (
            Column::Float((0..500).map(|_| r.next() as f64 / 3.0).collect()),
            "plain",
        ),
        (
            Column::Float(vec![1.5, -0.0, f64::NAN, f64::INFINITY]),
            "plain",
        ),
        (Column::Int(vec![i64::MIN, i64::MAX, 0]), "plain"),
    ];
    for (col, want) in cases {
        let (bytes, tag) = encoding::encode(&col);
        assert_eq!(encoding::NAMES[tag as usize], want, "{:?}", col.get(0));
        let back = encoding::decode(&bytes, col.dtype()).unwrap();
        match (&col, &back) {
            (Column::Float(a), Column::Float(b)) => {
                assert!(a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits()))
            }
            _ => assert_eq!(col, back),
        }
    }
}

#[test]
fn random_int_columns_round_trip() {
    let mut r = Rng(99);
    for _ in 0..300 {
        let n = r.below(300) as usize;
        let spread = 1u64 << r.below(63);
        let col = Column::Int(
            (0..n)
                .map(|_| (r.below(spread) as i64).wrapping_sub(spread as i64 / 2))
                .collect(),
        );
        let (bytes, _) = encoding::encode(&col);
        assert_eq!(encoding::decode(&bytes, col.dtype()).unwrap(), col);
    }
}

#[test]
fn corrupt_files_error_instead_of_panicking() {
    let (cat, _) = setup("corrupt");
    let good = std::fs::read(&cat["sales"]).unwrap();
    let dir = tmpdir("corrupt-out");
    let p = dir.join("bad.col");
    std::fs::write(&p, &good[..good.len() - 3]).unwrap();
    assert!(Table::open(&p).is_err());
    std::fs::write(&p, b"not a col file at all").unwrap();
    assert!(Table::open(&p).unwrap_err().contains("magic"));
    // Flip bytes inside the first chunk: decoding must fail cleanly or produce some column.
    let mut bad = good.clone();
    for b in &mut bad[4..40] {
        *b = 0xff;
    }
    std::fs::write(&p, &bad).unwrap();
    let mut t = Table::open(&p).unwrap();
    let _ = t.read_chunk(0, 0);
    for b in [
        &[0u8; 1][..],
        &[1, 255, 255, 255, 255][..],
        &[9, 0, 0, 0, 0][..],
    ] {
        assert!(encoding::decode(b, colstore::DataType::Int).is_err());
    }
}

#[test]
fn filter_project_order_limit() {
    let (cat, d) = setup("fpl");
    let (b, _) = rows(
        &cat,
        "SELECT id, qty * price AS revenue FROM sales WHERE region = 'west' AND qty >= 40 ORDER BY revenue DESC, id LIMIT 7",
    );
    let mut want: Vec<(i64, f64)> = (0..d.id.len())
        .filter(|&i| d.region[i] == "west" && d.qty[i] >= 40)
        .map(|i| (d.id[i], d.qty[i] as f64 * d.price[i]))
        .collect();
    want.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
    want.truncate(7);
    assert_eq!(b.names, ["id", "revenue"]);
    assert_eq!(
        b.columns[0],
        Column::Int(want.iter().map(|w| w.0).collect())
    );
    assert_eq!(
        b.columns[1],
        Column::Float(want.iter().map(|w| w.1).collect())
    );
}

#[test]
fn group_by_aggregates() {
    let (cat, d) = setup("agg");
    let (b, _) = rows(
        &cat,
        "SELECT region, count(*) AS n, sum(qty) AS q, avg(price) AS p, min(day), max(price) \
         FROM sales WHERE day BETWEEN '2024-03-01' AND '2024-06-30' GROUP BY region ORDER BY region",
    );
    let mut want: BTreeMap<&str, (i64, i64, f64, String, f64)> = BTreeMap::new();
    for i in 0..d.id.len() {
        if d.day[i].as_str() >= "2024-03-01" && d.day[i].as_str() <= "2024-06-30" {
            let e = want
                .entry(&d.region[i])
                .or_insert((0, 0, 0.0, "9".into(), f64::MIN));
            e.0 += 1;
            e.1 += d.qty[i];
            e.2 += d.price[i];
            if d.day[i] < e.3 {
                e.3 = d.day[i].clone();
            }
            e.4 = e.4.max(d.price[i]);
        }
    }
    assert_eq!(b.rows(), 4);
    for (r, (region, (n, q, p, mn, mx))) in want.iter().enumerate() {
        assert_eq!(b.columns[0].get(r), Value::Str(region.to_string()));
        assert_eq!(b.columns[1].get(r), Value::Int(*n));
        assert_eq!(b.columns[2].get(r), Value::Int(*q));
        assert!(close(b.columns[3].get(r).as_f64().unwrap(), p / *n as f64));
        assert_eq!(b.columns[4].get(r), Value::Str(mn.clone()));
        assert_eq!(b.columns[5].get(r), Value::Float(*mx));
    }
    assert_eq!(b.names[4], "MIN(day)");
}

#[test]
fn expressions_over_aggregates_and_global_aggregate() {
    let (cat, d) = setup("expr");
    let (b, _) = rows(
        &cat,
        "SELECT count(*), sum(qty) / count(*) + 1 AS x, max(id) - min(id) FROM sales",
    );
    let n = d.id.len() as i64;
    let q: i64 = d.qty.iter().sum();
    assert_eq!(b.columns[0], Column::Int(vec![n]));
    assert!(close(
        b.columns[1].get(0).as_f64().unwrap(),
        q as f64 / n as f64 + 1.0
    ));
    assert_eq!(b.columns[2], Column::Int(vec![n - 1]));
    let (b, _) = rows(
        &cat,
        "SELECT count(*), sum(qty) FROM sales WHERE qty > 1000",
    );
    assert_eq!(b.columns[0], Column::Int(vec![0]));
}

#[test]
fn hash_join_with_group_by() {
    let (cat, d) = setup("join");
    let (b, _) = rows(
        &cat,
        "SELECT cust_name, sum(qty) AS q FROM sales JOIN custs ON sales.cust = custs.cust_id \
         WHERE region <> 'east' AND cust_id < 10 GROUP BY cust_name ORDER BY q DESC, cust_name LIMIT 5",
    );
    let mut want: HashMap<i64, i64> = HashMap::new();
    for i in 0..d.id.len() {
        if d.region[i] != "east" && d.cust[i] < 10 {
            *want.entry(d.cust[i]).or_default() += d.qty[i];
        }
    }
    let mut w: Vec<(String, i64)> = want
        .into_iter()
        .map(|(c, q)| (format!("c{c:03}"), q))
        .collect();
    w.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
    w.truncate(5);
    assert_eq!(
        b.columns[0],
        Column::Str(w.iter().map(|x| x.0.clone()).collect())
    );
    assert_eq!(b.columns[1], Column::Int(w.iter().map(|x| x.1).collect()));

    let (b, _) = rows(
        &cat,
        "SELECT count(*) FROM sales JOIN custs ON cust = cust_id",
    );
    assert_eq!(
        b.columns[0],
        Column::Int(vec![d.cust.iter().filter(|&&c| c < 250).count() as i64])
    );
}

#[test]
fn zone_maps_prune_row_groups_without_changing_results() {
    let (cat, d) = setup("zone");
    let (b, s) = rows(
        &cat,
        "SELECT sum(qty) FROM sales WHERE id >= 5000 AND id < 7000",
    );
    assert_eq!(
        b.columns[0],
        Column::Int(vec![d.qty[5000..7000].iter().sum()])
    );
    assert_eq!((s.row_groups_read, s.row_groups_total), (2, 20));
    let (_, s) = rows(&cat, "SELECT count(*) FROM sales WHERE 15000 < id");
    assert_eq!(s.row_groups_read, 5);
    // `day` is only roughly sorted, so a month filter still skips most groups.
    let (b, s) = rows(&cat, "SELECT count(*) FROM sales WHERE day >= '2024-12-01'");
    assert_eq!(
        b.columns[0],
        Column::Int(vec![
            d.day.iter().filter(|x| x.as_str() >= "2024-12-01").count() as i64
        ])
    );
    assert!(s.row_groups_read <= 3, "{s:?}");
    // OR is not pushed into the zone map, so everything is read.
    let (_, s) = rows(
        &cat,
        "SELECT count(*) FROM sales WHERE id < 10 OR id > 19990",
    );
    assert_eq!(s.row_groups_read, 20);
}

#[test]
fn select_star_and_limit_stop_early() {
    let (cat, _) = setup("star");
    let (b, s) = rows(&cat, "select * from sales limit 3");
    assert_eq!(b.names, ["id", "region", "qty", "price", "day", "cust"]);
    assert_eq!(b.rows(), 3);
    assert_eq!(s.row_groups_read, 1);
}

#[test]
fn explain_shows_pushdown() {
    let (cat, _) = setup("explain");
    let QueryResult::Explain(text) = query(
        &cat,
        "EXPLAIN SELECT cust_name, sum(qty) FROM sales JOIN custs ON cust = cust_id WHERE id < 2000 AND qty > cust_id GROUP BY cust_name",
    )
    .unwrap() else {
        panic!()
    };
    assert!(text.contains("HashAggregate group=[cust_name]"), "{text}");
    assert!(text.contains("Filter (qty > cust_id)"), "{text}");
    assert!(text.contains("HashJoin cust = cust_id"), "{text}");
    assert!(
        text.contains("Scan sales columns=[id, qty, cust] filter=(id < 2000) row_groups=2/20"),
        "{text}"
    );
}

#[test]
fn errors_are_reported() {
    let (cat, _) = setup("err");
    for (sql, msg) in [
        ("select nope from sales", "unknown column"),
        ("select region, qty from sales group by region", "GROUP BY"),
        ("select * from sales where region > 3", "cannot compare"),
        ("select region + 1 from sales", "string"),
        ("select * from missing", "unknown table"),
        ("select sum(region) from sales", "string"),
        ("select * from sales join sales on id = id", "both"),
        (
            "select * from sales join custs on cust = cust_name",
            "types differ",
        ),
        (
            "select * from custs join sales on cust_id = cust_name",
            "JOIN",
        ),
    ] {
        let e = query(&cat, sql).unwrap_err();
        assert!(e.contains(msg), "{sql}: {e}");
    }
}
