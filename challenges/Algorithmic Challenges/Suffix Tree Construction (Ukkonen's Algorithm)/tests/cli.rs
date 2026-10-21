//! The `sufftree` binary, end to end.

use std::process::Command;

const BIN: &str = env!("CARGO_BIN_EXE_sufftree");

fn run(args: &[&str]) -> (bool, String, String) {
    let out = Command::new(BIN).args(args).output().expect("binary runs");
    (
        out.status.success(),
        String::from_utf8_lossy(&out.stdout).into_owned(),
        String::from_utf8_lossy(&out.stderr).into_owned(),
    )
}

fn temp(name: &str, text: &[u8]) -> String {
    let path = std::env::temp_dir().join(format!("sufftree-test-{}-{name}", std::process::id()));
    std::fs::write(&path, text).unwrap();
    path.to_str().unwrap().to_string()
}

#[test]
fn every_algorithm_finds_the_same_occurrences() {
    let file = temp("mississippi.txt", b"mississippi");
    for algo in ["ukkonen", "ukkonen-hash", "sais", "pwotd", "wotd"] {
        let (ok, out, err) = run(&["find", &file, "ssi", "i", "zz", "--algo", algo]);
        assert!(ok, "{algo}: {err}");
        assert_eq!(
            out,
            "\"ssi\": 2 occurrences [2, 5]\n\"i\": 4 occurrences [1, 4, 7, 10]\n\"zz\": 0 occurrences []\n",
            "{algo}"
        );
    }
}

#[test]
fn build_lrs_unique_lcs_and_ms() {
    let a = temp("a.txt", b"xxabcdeyy");
    let b = temp("b.txt", b"zzabcdeqq");
    let (ok, out, _) = run(&["build", &a, "--algo", "sais"]);
    assert!(ok && out.starts_with("9 bytes -> "), "{out}");
    let (ok, out, _) = run(&["lrs", &a]);
    assert!(ok && out.starts_with("length 1"), "{out}"); // "x" or "y"; "xx" occurs once
    let (ok, out, _) = run(&["lcs", &a, &b]);
    assert!(
        ok && out.contains("length 5") && out.contains("\"abcde\""),
        "{out}"
    );
    let (ok, out2, _) = run(&["lcs", &a, &b, "--array"]);
    assert!(ok && out2.contains("length 5"), "{out2}");
    let (ok, out, _) = run(&["unique", &a]);
    assert!(ok && out.starts_with("length 1"), "{out}");
    let (ok, out, _) = run(&["ms", &a, &b]);
    assert!(ok && out.contains("longest match: 5"), "{out}");
}

#[test]
fn online_reports_growth() {
    let file = temp("online.txt", b"abababab");
    let (ok, out, _) = run(&["online", &file, "ab", "--every", "4"]);
    assert!(ok);
    let lines: Vec<&str> = out.lines().collect();
    assert_eq!(lines.len(), 2);
    assert!(
        lines[0].contains("2 occurrences") && lines[1].contains("4 occurrences"),
        "{out}"
    );
}

#[test]
fn errors_are_reported_not_panicked() {
    let (ok, _, err) = run(&["build", "/definitely/not/here"]);
    assert!(!ok && err.contains("cannot read"), "{err}");
    let one = temp("one.txt", b"abc");
    let (ok, _, err) = run(&["lcs", &one]);
    assert!(!ok && err.contains("at least two"), "{err}");
    let (ok, _, err) = run(&["online", &one, "a", "--every", "0"]);
    assert!(!ok && err.contains("at least 1"), "{err}");
}
