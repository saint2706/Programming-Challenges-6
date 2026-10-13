use std::path::PathBuf;
use std::process::Command;

use hld::script::{Structure, run_script};

const SAMPLE: &str = "\
# a small vertex-weighted tree
5
3 1 4 1 5
0 1
0 2
2 3
2 4
path 1 3
add 1 3 10
get 3
set 2 100
sub 2
lca 1 4
kth 3 1
kth 3 9
";

fn run(structure: Structure) -> Vec<String> {
    run_script(SAMPLE, structure, false, 0).unwrap()
}

#[test]
fn sample_script_has_the_hand_computed_answers() {
    let out = run(Structure::Naive);
    // path 1-0-2-3 = 1+3+4+1 = 9; after +10 vertex 3 is 11; set 2 -> 100;
    // subtree of 2 = {2,3,4} = 100 + 11 + 5.
    assert_eq!(out[0], "sum=9 min=1 max=4 count=4");
    assert_eq!(out[1], "11");
    assert_eq!(out[2], "sum=116 min=5 max=100 count=3");
    assert_eq!(out[3], "0");
    assert_eq!(out[4], "2");
    assert_eq!(out[5], "none");
}

#[test]
fn every_structure_prints_the_same_answers() {
    let reference = run(Structure::Naive);
    for s in [Structure::Classic, Structure::TopTree] {
        assert_eq!(run(s), reference, "{}", s.name());
    }
}

#[test]
fn balanced_has_no_subtree_queries_and_says_so_with_the_line_number() {
    let err = run_script(SAMPLE, Structure::Balanced, false, 0).unwrap_err();
    assert!(err.contains("line 12") && err.contains("balanced"), "{err}");
    let without_sub: String = SAMPLE
        .lines()
        .filter(|l| !l.starts_with("sub"))
        .collect::<Vec<_>>()
        .join("\n");
    assert_eq!(
        run_script(&without_sub, Structure::Balanced, false, 0)
            .unwrap()
            .len(),
        run(Structure::Naive).len() - 1
    );
}

#[test]
fn edge_mode_script() {
    let script =
        "4\n0 1 5\n0 2 7\n2 3 1\npath 1 3\nadd 1 3 2\npath 1 3\nsub 0\nset 2 100\nget 2\nkth 3 2\n";
    for s in Structure::ALL {
        if s == Structure::Balanced {
            continue;
        }
        let out = run_script(script, s, true, 0).unwrap();
        assert_eq!(out[0], "sum=13 min=1 max=7 count=3", "{}", s.name());
        assert_eq!(out[1], "sum=19 min=3 max=9 count=3");
        assert_eq!(out[2], "sum=19 min=3 max=9 count=3");
        assert_eq!(out[3], "100");
        assert_eq!(out[4], "0");
    }
}

#[test]
fn malformed_scripts_report_the_offending_line() {
    let cases: &[(&str, &str)] = &[
        ("", "empty input"),
        ("0\n", "at least one vertex"),
        ("3\n1 2\n0 1\n1 2\n", "expected 3 weights"),
        (
            "2\n1 2\n0 1\npath 0 7\n",
            "line 4: vertex 7 is out of range",
        ),
        ("2\n1 2\n0 1\nfly 0 1\n", "line 4: unknown command 'fly'"),
        ("2\n1 2\n0 1\nset 5 1\n", "index 5 is out of range"),
        ("2\n1 2\n0 1\npath 0 1 9\n", "extra tokens"),
        ("2\n1 2\n0 1\nadd 0 1 x\n", "delta must be a number"),
        ("3\n1 2 3\n0 1\n", "fewer edge lines"),
        ("3\n1 2 3\n0 1\n1 0\n", "cycle"),
        ("2\n1 2\n0 0\n", "itself"),
    ];
    for (script, needle) in cases {
        let err = run_script(script, Structure::TopTree, false, 0).unwrap_err();
        assert!(
            err.contains(needle),
            "{script:?}: expected {needle:?} in {err:?}"
        );
    }
    assert!(
        run_script(SAMPLE, Structure::TopTree, false, 9)
            .unwrap_err()
            .contains("root 9")
    );
}

// ---- the binary ---------------------------------------------------------------------------

fn bin() -> Command {
    Command::new(env!("CARGO_BIN_EXE_hld"))
}

fn temp_file(name: &str, contents: &str) -> PathBuf {
    let path = std::env::temp_dir().join(format!("hld-test-{}-{name}", std::process::id()));
    std::fs::write(&path, contents).unwrap();
    path
}

#[test]
fn cli_runs_a_script_and_all_structures_agree() {
    let script = Command::new(env!("CARGO_BIN_EXE_hld"))
        .args([
            "gen",
            "--shape",
            "adversarial",
            "--n",
            "500",
            "--queries",
            "400",
            "--seed",
            "3",
        ])
        .output()
        .unwrap();
    assert!(script.status.success());
    let path = temp_file("gen.txt", &String::from_utf8(script.stdout).unwrap());
    let run = |structure: &str| {
        let out = bin()
            .args(["run", path.to_str().unwrap(), "--structure", structure])
            .output()
            .unwrap();
        assert!(
            out.status.success(),
            "{structure}: {}",
            String::from_utf8_lossy(&out.stderr)
        );
        String::from_utf8(out.stdout).unwrap()
    };
    let reference = run("naive");
    assert!(reference.lines().count() > 100);
    assert_eq!(run("classic"), reference);
    assert_eq!(run("toptree"), reference);
    std::fs::remove_file(path).ok();
}

#[test]
fn cli_edge_mode_and_errors() {
    let path = temp_file("edges.txt", "3\n0 1 5\n1 2 7\npath 0 2\n");
    let out = bin()
        .args(["run", path.to_str().unwrap(), "--edges"])
        .output()
        .unwrap();
    assert_eq!(
        String::from_utf8_lossy(&out.stdout).trim(),
        "sum=12 min=5 max=7 count=2"
    );

    let bad = temp_file("bad.txt", "2\n1 2\n0 1\npath 0 9\n");
    let out = bin().args(["run", bad.to_str().unwrap()]).output().unwrap();
    assert_eq!(out.status.code(), Some(2));
    assert!(String::from_utf8_lossy(&out.stderr).contains("line 4"));

    let out = bin()
        .args(["run", "/definitely/not/here.txt"])
        .output()
        .unwrap();
    assert_eq!(out.status.code(), Some(2));
    assert!(String::from_utf8_lossy(&out.stderr).contains("cannot read"));

    let out = bin()
        .args(["run", bad.to_str().unwrap(), "--structure", "fast"])
        .output()
        .unwrap();
    assert!(!out.status.success());
    let out = bin().args(["gen", "--n", "0"]).output().unwrap();
    assert_eq!(out.status.code(), Some(2));
    for p in [path, bad] {
        std::fs::remove_file(p).ok();
    }
}
