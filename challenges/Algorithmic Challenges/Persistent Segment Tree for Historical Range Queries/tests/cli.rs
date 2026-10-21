//! The `pst` binary, end to end.

use std::process::Command;

const BIN: &str = env!("CARGO_BIN_EXE_pst");

/// Output with terminal styling stripped: on GitHub Actions clap emits ANSI colour codes that can
/// split a flag name across escape sequences.
fn plain(s: &[u8]) -> String {
    let text = String::from_utf8_lossy(s);
    let mut out = String::new();
    let mut chars = text.chars().peekable();
    while let Some(c) = chars.next() {
        if c == '\u{1b}' {
            for n in chars.by_ref() {
                if n.is_ascii_alphabetic() {
                    break;
                }
            }
        } else {
            out.push(c);
        }
    }
    out
}

fn run(args: &[&str]) -> (bool, String, String) {
    let out = Command::new(BIN).args(args).output().expect("binary runs");
    (out.status.success(), plain(&out.stdout), plain(&out.stderr))
}

fn temp(name: &str, text: &str) -> String {
    let path = std::env::temp_dir().join(format!("pst-test-{}-{name}", std::process::id()));
    std::fs::write(&path, text).unwrap();
    path.to_str().unwrap().to_string()
}

#[test]
fn every_engine_gives_the_same_kth_smallest() {
    let file = temp("a.txt", "5 1 4 1 5 9 2 6 -3 5");
    for engine in ["persistent", "wavelet", "sort"] {
        for (k, want) in [(0, "-3"), (3, "2"), (6, "5"), (8, "9")] {
            let (ok, out, err) =
                run(&["kth", &file, "1", "10", &k.to_string(), "--engine", engine]);
            assert!(ok, "{engine}: {err}");
            // 1 4 1 5 9 2 6 -3 5 sorted: -3 1 1 2 4 5 5 6 9
            assert_eq!(out.trim(), want, "{engine} k={k}");
        }
    }
}

#[test]
fn kth_reports_bad_ranges_and_bad_k() {
    let file = temp("b.txt", "1, 2, 3, 4");
    let (ok, _, err) = run(&["kth", &file, "2", "9", "0"]);
    assert!(!ok && err.contains("does not fit 4 numbers"), "{err}");
    let (ok, _, err) = run(&["kth", &file, "0", "4", "4"]);
    assert!(!ok && err.contains("out of range"), "{err}");
    let bad = temp("c.txt", "1 two 3");
    let (ok, _, err) = run(&["kth", &bad, "0", "3", "0"]);
    assert!(!ok && err.contains("\"two\" is not an integer"), "{err}");
}

#[test]
fn distinct_agrees_with_brute_force() {
    let file = temp("d.txt", "3 3 1 2 3 1 4 4 4 2");
    for (l, r, want) in [(0, 10, "4"), (2, 6, "3"), (6, 9, "1"), (3, 3, "0")] {
        for brute in [false, true] {
            let mut args = vec!["distinct", &file[..], "x", "y"];
            let (l, r) = (l.to_string(), r.to_string());
            args[2] = &l;
            args[3] = &r;
            if brute {
                args.push("--brute-force");
            }
            let (ok, out, err) = run(&args);
            assert!(ok, "{err}");
            assert_eq!(out.trim(), want, "{l}..{r} brute={brute}");
        }
    }
}

#[test]
fn history_script_replays_time_travel_queries() {
    let script = temp(
        "h.txt",
        "# a ledger of 6 accounts\nsize 6\nset 10 0 100\nset 10 1 50\nset 20 0 70\nset 30 5 9\n\
         sum 5 0 6\nsum 10 0 6\nsum 25 0 6\nget 25 0\nsum 99 0 6\nstats\nforget 25\nsum 5 0 6\nsum 25 0 6\nstats\n",
    );
    let (ok, out, err) = run(&["history", &script]);
    assert!(ok, "{err}");
    let lines: Vec<&str> = out.lines().collect();
    assert_eq!(lines[0], "sum 5 0..6 = 0");
    assert_eq!(lines[1], "sum 10 0..6 = 150");
    assert_eq!(lines[2], "sum 25 0..6 = 120");
    assert_eq!(lines[3], "get 25 0 = 70");
    assert_eq!(lines[4], "sum 99 0..6 = 129");
    assert!(
        lines[5].starts_with("stats: 5 versions created, 5 retained"),
        "{}",
        lines[5]
    );
    assert!(
        lines[6].starts_with("forget 25: retired 3 versions, 2 retained"),
        "{}",
        lines[6]
    );
    assert_eq!(lines[7], "sum 5 0..6 = (history forgotten)");
    assert_eq!(lines[8], "sum 25 0..6 = 120");
    assert!(
        lines[9].starts_with("stats: 5 versions created, 2 retained"),
        "{}",
        lines[9]
    );
}

#[test]
fn history_script_errors_name_the_line() {
    let cases = [
        ("set 1 0 5\n", "line 1: the script must start with `size N`"),
        (
            "size 4\nset 1 9 5\n",
            "line 2: position 9 out of range for size 4",
        ),
        (
            "size 4\nset 5 0 1\nset 3 0 1\n",
            "line 3: time 3 is before the last change at 5",
        ),
        (
            "size 4\nsum 1 3 2\n",
            "line 2: range 3..2 does not fit size 4",
        ),
        (
            "size 4\nfrobnicate 1\n",
            "line 2: unknown command \"frobnicate\"",
        ),
        ("size 4\nset 1 0\n", "line 2: `set` needs more arguments"),
        ("size 0\n", "line 1: size must be positive"),
        (
            "size 4\nset 1 0 9223372036854775807\n",
            "line 2: value 9223372036854775807 is beyond +-2^40",
        ),
    ];
    for (text, want) in cases {
        let file = temp("e.txt", text);
        let (ok, _, err) = run(&["history", &file]);
        assert!(!ok && err.contains(want), "{text:?}: {err}");
    }
}

#[test]
fn sizes_shows_the_wavelet_matrix_is_smaller() {
    let numbers: Vec<String> = (0..5000).map(|i| ((i * 7919) % 1009).to_string()).collect();
    let file = temp("s.txt", &numbers.join(" "));
    let (ok, out, err) = run(&["sizes", &file]);
    assert!(ok, "{err}");
    let bytes = |name: &str| -> usize {
        out.lines()
            .find(|l| l.starts_with(name))
            .and_then(|l| l.split_whitespace().nth(1))
            .and_then(|b| b.parse().ok())
            .unwrap_or_else(|| panic!("no row for {name} in {out}"))
    };
    assert!(bytes("wavelet") * 10 < bytes("persistent"), "{out}");
    assert!(out.starts_with("5000 numbers, 1009 distinct"), "{out}");
}

proptest::proptest! {
    #![proptest_config(proptest::prelude::ProptestConfig::with_cases(60))]

    /// A script is untrusted input: whatever it says, the binary must exit cleanly with a message,
    /// never panic.
    #[test]
    fn history_never_panics_on_token_soup(
        lines in proptest::collection::vec(
            "(size|set|get|sum|forget|stats|bogus)( -?[0-9]{1,12}){0,4}( #x)?",
            0..12,
        ),
    ) {
        let mut text = String::from("size 5\n");
        text.push_str(&lines.join("\n"));
        let file = temp("fuzz.txt", &text);
        let (_, _, err) = run(&["history", &file]);
        proptest::prop_assert!(!err.contains("panicked"), "{text:?}: {err}");
    }
}
