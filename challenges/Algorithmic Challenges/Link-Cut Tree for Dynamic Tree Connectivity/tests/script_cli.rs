//! The script language and the `lct` binary, end to end.

use std::process::Command;

use lct::connectivity::{random_events, solve_offline, solve_rebuild};
use lct::script::{
    ScriptError, Structure, generate_script, parse_events, render_events, run_script,
};
use lct::shapes::Shape;

const BIN: &str = env!("CARGO_BIN_EXE_lct");

fn run(args: &[&str]) -> (bool, String, String) {
    let out = Command::new(BIN).args(args).output().expect("binary runs");
    (
        out.status.success(),
        String::from_utf8_lossy(&out.stdout).into_owned(),
        String::from_utf8_lossy(&out.stderr).into_owned(),
    )
}

fn temp(name: &str, text: &str) -> std::path::PathBuf {
    let path = std::env::temp_dir().join(format!("lct-test-{}-{name}", std::process::id()));
    std::fs::write(&path, text).unwrap();
    path
}

#[test]
fn a_small_script_runs_on_the_subtree_link_cut_tree() {
    let script = "5\nlink 0 1\nlink 1 2\nlink 3 4\nlink 0 2\nset 0 5\nset 1 -2\nset 2 7\n\
                  path 0 2\nadd 0 2 10\npath 0 2\nsub 0 1\ncomp 0\nconn 0 4\ncut 1 2\nconn 0 2\ncut 1 2\n";
    let got = run_script(script, Structure::SubtreeLct).unwrap();
    assert_eq!(
        got,
        [
            "link: ok",
            "link: ok",
            "link: ok",
            "link: rejected",
            "set: ok",
            "set: ok",
            "set: ok",
            "path: sum=10 min=-2 max=7 count=3",
            "add: ok",
            "path: sum=40 min=8 max=17 count=3",
            "sub: sum=25 count=2",
            "comp: sum=40 count=3",
            "conn: false",
            "cut: ok",
            "conn: false",
            "cut: rejected",
        ]
        .map(String::from)
    );
}

#[test]
fn structures_that_cannot_answer_say_so() {
    let script = "3\nlink 0 1\npath 0 1\n";
    assert_eq!(
        run_script(script, Structure::Ett).unwrap()[1],
        "path: not supported by ett"
    );
    assert_eq!(
        run_script("3\nlink 0 1\nsub 0 1\n", Structure::Lct).unwrap()[1],
        "sub: not supported by lct"
    );
    assert_eq!(
        run_script("3\nlink 0 1\nset 0 1\n", Structure::Ufo).unwrap()[1],
        "set: not supported by ufo"
    );
}

/// A structure may decline a command, but must never answer one differently from the oracle.
fn assert_consistent(script: &str, structure: Structure) {
    let want = run_script(script, Structure::Naive).unwrap();
    let got = run_script(script, structure).unwrap();
    assert_eq!(got.len(), want.len());
    for (g, w) in got.iter().zip(&want) {
        assert!(
            g.contains("not supported") || g == w,
            "{structure:?}: {g} vs {w}"
        );
    }
}

#[test]
fn every_structure_agrees_with_the_oracle_on_a_generated_script() {
    for (i, shape) in [Shape::Random, Shape::Path, Shape::Star]
        .into_iter()
        .enumerate()
    {
        let script = generate_script(shape, 60, 600, 5 + i as u64);
        for structure in [Structure::Lct, Structure::SubtreeLct, Structure::Ufo] {
            assert_consistent(&script, structure);
        }
        // The Euler tour tree has no path add, so its values drift from the oracle's once the
        // script uses one; give it the same script without those lines.
        let without_add: String = script
            .lines()
            .filter(|l| !l.starts_with("add "))
            .map(|l| {
                format!(
                    "{l}
"
                )
            })
            .collect();
        assert_consistent(&without_add, Structure::Ett);
        assert_consistent(&without_add, Structure::SubtreeLct);
    }
}

#[test]
fn bad_scripts_name_the_line() {
    let at = |s: &str| run_script(s, Structure::Lct).unwrap_err();
    assert_eq!(
        at(""),
        ScriptError {
            line: 1,
            message: "empty script: the first line must be the vertex count".into()
        }
    );
    assert_eq!(at("3\nlink 0 3").line, 2);
    assert_eq!(at("3\nlink 0 x").message, "vertex is not a number");
    assert_eq!(at("3\n# comment\n\nfly 0 1").line, 4);
    assert_eq!(at("0\n").line, 1);
}

#[test]
fn events_round_trip_and_solve() {
    let events = random_events(25, 40, 800, 30, 9);
    let text = render_events(25, &events);
    let (n, parsed) = parse_events(&text).unwrap();
    assert_eq!((n, &parsed), (25, &events));
    assert_eq!(
        solve_offline(n, &parsed).unwrap().answers,
        solve_rebuild(n, &parsed).unwrap()
    );
    assert!(parse_events("3\nfly 0 1").is_err());
}

#[test]
fn cli_run_gen_and_connect() {
    let (ok, script, _) = run(&[
        "gen", "--shape", "random", "--n", "30", "--ops", "200", "--seed", "3",
    ]);
    assert!(ok);
    let file = temp("script.txt", &script);
    let (ok, lct_out, _) = run(&["run", file.to_str().unwrap(), "--structure", "subtree-lct"]);
    assert!(ok);
    let (ok, naive_out, _) = run(&["run", file.to_str().unwrap(), "--structure", "naive"]);
    assert!(ok);
    assert_eq!(lct_out, naive_out);

    let (ok, events, _) = run(&["gen-events", "--n", "40", "--events", "500", "--seed", "4"]);
    assert!(ok);
    let file = temp("events.txt", &events);
    let (ok, answers, log) = run(&["connect", file.to_str().unwrap(), "--verify"]);
    assert!(ok, "{log}");
    assert!(log.contains("verified against union-find"));
    assert!(answers.lines().all(|l| l == "true" || l == "false"));

    let (ok, _, log) = run(&["run", "/definitely/not/here.txt"]);
    assert!(!ok && log.contains("cannot read"));
    let bad = temp("bad.txt", "3\nlink 0 9\n");
    let (ok, _, log) = run(&["run", bad.to_str().unwrap()]);
    assert!(!ok && log.contains("line 2"));
    let (ok, _, log) = run(&["gen", "--n", "0"]);
    assert!(!ok && log.contains("at least 1"));
}
