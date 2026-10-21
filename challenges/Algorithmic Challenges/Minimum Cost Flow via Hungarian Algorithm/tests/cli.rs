//! The `mcflow` binary, end to end.

use std::process::Command;

const BIN: &str = env!("CARGO_BIN_EXE_mcflow");

/// Output with terminal styling stripped: on GitHub Actions clap emits ANSI colour codes that
/// split `--algo` across escape sequences.
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
    let path = std::env::temp_dir().join(format!("mcflow-test-{}-{name}", std::process::id()));
    std::fs::write(&path, text).unwrap();
    path.to_str().unwrap().to_string()
}

#[test]
fn solve_prints_cost_and_pairs() {
    let file = temp("three.csv", "4,1,3\n2,0,5\n3,2,2\n");
    for algo in [
        "hungarian",
        "crouse",
        "jv",
        "lapmod",
        "auction",
        "ssp",
        "cost-scaling",
        "network-simplex",
    ] {
        let (ok, out, err) = run(&["solve", &file, "--algo", algo, "--show"]);
        assert!(ok, "{algo}: {err}");
        assert!(out.starts_with("cost 5 (3x3,"), "{algo}: {out}");
        assert!(out.contains("row 1 -> column 0 (cost 2)"), "{algo}: {out}");
    }
}

#[test]
fn a_rectangular_and_an_infeasible_instance() {
    let wide = temp("wide.csv", "9,1,9,9\n9,9,2,9\n");
    let (ok, out, _) = run(&["solve", &wide]);
    assert!(ok && out.starts_with("cost 3 (2x4,"), "{out}");
    let bad = temp("bad.csv", "1,x\n2,x\n");
    let (ok, _, err) = run(&["solve", &bad]);
    assert!(!ok && err.contains("infeasible"), "{err}");
}

#[test]
fn verify_proves_optimality_two_ways_when_the_solver_has_duals() {
    let file = temp("v.csv", "4,1,3\n2,0,5\n3,2,2\n");
    let (ok, out, err) = run(&["verify", &file, "--algo", "hungarian"]);
    assert!(ok, "{err}");
    assert!(
        out.contains("dual certificate and negative-cycle search agree"),
        "{out}"
    );
    let (ok, out, _) = run(&["verify", &file, "--algo", "auction"]);
    assert!(ok && out.contains("reports no duals"), "{out}");
}

#[test]
fn gen_then_compare_gives_one_cost_for_every_algorithm() {
    let file = temp("gen.csv", "");
    let (ok, _, err) = run(&[
        "gen",
        "assignment",
        "uniform",
        "--n",
        "25",
        "--seed",
        "3",
        "--out",
        &file,
    ]);
    assert!(ok, "{err}");
    let (ok, out, err) = run(&["compare", &file]);
    assert!(ok, "{err}");
    let costs: Vec<&str> = out
        .lines()
        .skip(1)
        .map(|l| l.split_whitespace().nth(1).unwrap())
        .collect();
    assert_eq!(costs.len(), 8, "{out}");
    assert!(costs.iter().all(|&c| c == costs[0]), "{out}");
}

#[test]
fn convert_to_min_is_the_assignment_to_flow_reduction() {
    let csv = temp("c.csv", "4,1,3\n2,0,5\n3,2,2\n");
    let (ok, min_text, err) = run(&["convert", &csv, "--to", "min"]);
    assert!(ok, "{err}");
    assert!(min_text.starts_with("p min 6 9\n"), "{min_text}");
    let min = temp("c.min", &min_text);
    for algo in ["ssp", "cost-scaling", "network-simplex"] {
        let (ok, out, err) = run(&["solve", &min, "--algo", algo]);
        assert!(
            ok && out.starts_with("cost 5 (6 nodes, 9 arcs,"),
            "{algo}: {out} {err}"
        );
    }
    let (ok, _, err) = run(&["solve", &min, "--algo", "hungarian"]);
    assert!(!ok && err.contains("assignments only"), "{err}");
}

#[test]
fn flow_generation_and_comparison() {
    let file = temp("net.min", "");
    let (ok, _, err) = run(&[
        "gen",
        "flow",
        "--nodes",
        "30",
        "--arcs",
        "100",
        "--negative",
        "--out",
        &file,
    ]);
    assert!(ok, "{err}");
    let (ok, out, err) = run(&["compare", &file]);
    assert!(ok, "{err}");
    let costs: Vec<&str> = out
        .lines()
        .skip(1)
        .map(|l| l.split_whitespace().nth(1).unwrap())
        .collect();
    assert_eq!(costs.len(), 3, "{out}");
    assert!(costs.iter().all(|&c| c == costs[0]), "{out}");
    let (ok, out, err) = run(&["verify", &file, "--algo", "network-simplex"]);
    assert!(ok && out.contains("optimal"), "{out} {err}");
}

#[test]
fn unknown_algorithm_names_the_choices() {
    let file = temp("u.csv", "1\n");
    let (ok, _, err) = run(&["solve", &file, "--algo", "magic"]);
    assert!(!ok && err.contains("hungarian"), "{err}");
}
