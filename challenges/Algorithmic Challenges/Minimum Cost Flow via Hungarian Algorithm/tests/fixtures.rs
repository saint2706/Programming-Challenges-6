//! Reference answers from SciPy and NetworkX (`scripts/make_fixtures.py`), which share no code
//! with this crate: every solver must reproduce them, including which instances are infeasible.

use std::path::PathBuf;

use mcflow::dimacs::{parse_matrix, parse_min};
use mcflow::flow;
use mcflow::verify::{certify, certify_flow};
use mcflow::{Algorithm, Error, FlowAlgorithm, solve_dense};

fn fixtures() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures")
}

fn expected() -> Vec<(String, String, String)> {
    let text = std::fs::read_to_string(fixtures().join("expected.csv"))
        .expect("run scripts/make_fixtures.py");
    text.lines()
        .map(|l| {
            let mut f = l.split(',');
            let mut next = || f.next().unwrap().to_string();
            (next(), next(), next())
        })
        .collect()
}

#[test]
fn every_assignment_solver_reproduces_scipy() {
    let mut count = 0;
    for (kind, file, want) in expected() {
        if kind != "matrix" {
            continue;
        }
        let d = parse_matrix(&std::fs::read_to_string(fixtures().join(&file)).unwrap()).unwrap();
        let g = d.to_sparse();
        for algo in Algorithm::ALL {
            match solve_dense(&d, algo) {
                Ok(a) => {
                    assert_eq!(a.cost.to_string(), want, "{algo} on {file}");
                    certify(&g, &a).unwrap_or_else(|e| panic!("{algo} on {file}: {e}"));
                }
                Err(Error::Infeasible) => assert_eq!(want, "infeasible", "{algo} on {file}"),
                Err(Error::Unsupported(_)) => {
                    assert!(
                        algo == Algorithm::Jv && d.has_forbidden(),
                        "{algo} on {file}"
                    );
                }
                Err(e) => panic!("{algo} on {file}: {e}"),
            }
        }
        count += 1;
    }
    assert!(count >= 80, "only {count} matrix fixtures");
}

#[test]
fn every_flow_solver_reproduces_networkx() {
    let mut count = 0;
    for (kind, file, want) in expected() {
        if kind != "flow" {
            continue;
        }
        let net = parse_min(&std::fs::read_to_string(fixtures().join(&file)).unwrap()).unwrap();
        for algo in FlowAlgorithm::ALL {
            match flow::solve(&net, algo) {
                Ok(f) => {
                    assert_eq!(f.cost.to_string(), want, "{} on {file}", algo.name());
                    certify_flow(&net, &f)
                        .unwrap_or_else(|e| panic!("{} on {file}: {e}", algo.name()));
                }
                Err(Error::Infeasible) => {
                    assert_eq!(want, "infeasible", "{} on {file}", algo.name())
                }
                Err(e) => panic!("{} on {file}: {e}", algo.name()),
            }
        }
        count += 1;
    }
    assert!(count >= 20, "only {count} network fixtures");
}
