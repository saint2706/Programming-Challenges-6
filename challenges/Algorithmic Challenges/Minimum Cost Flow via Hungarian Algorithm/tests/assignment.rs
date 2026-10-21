//! Every assignment solver against a brute-force oracle and against two proofs of optimality.

mod common;

use common::{brute_force, small};
use mcflow::instances::{self, machol_wien, machol_wien_optimum};
use mcflow::verify::{certify, check_assignment, check_duals};
use mcflow::{Algorithm, Dense, Error, FORBIDDEN, Sparse, solve_dense, solve_sparse};
use proptest::prelude::*;

/// Runs `algo` on `d` and checks the answer against the oracle's `expected` (None = infeasible).
fn check(d: &Dense, algo: Algorithm, expected: Option<i64>) {
    let sparse = d.to_sparse();
    let what = format!("{algo} on {}x{}: {d:?}", d.rows(), d.cols());
    match solve_dense(d, algo) {
        Ok(a) => {
            let want = expected.unwrap_or_else(|| panic!("{what}: solved an infeasible instance"));
            assert_eq!(a.cost, want, "{what}");
            check_assignment(&sparse, &a).unwrap_or_else(|e| panic!("{what}: {e}"));
            certify(&sparse, &a).unwrap_or_else(|e| panic!("{what}: {e}"));
            if a.duals.is_some() {
                check_duals(&sparse, &a).unwrap_or_else(|e| panic!("{what}: {e}"));
            }
        }
        Err(Error::Unsupported(_)) => {
            assert!(
                algo == Algorithm::Jv && d.has_forbidden(),
                "{what}: unexpected Unsupported"
            );
        }
        Err(e) => {
            assert_eq!(e, Error::Infeasible, "{what}");
            assert!(
                expected.is_none(),
                "{what}: reported infeasible, oracle found {expected:?}"
            );
        }
    }
}

#[test]
fn hungarian_tie_breaks_agree_on_cost() {
    use mcflow::hungarian::{TieBreak, optimal_cost};
    for seed in 0..40 {
        let rows = 1 + seed as usize % 7;
        let cols = 1 + (seed as usize / 3) % 7;
        let d = small(rows, cols, 0, 3, [0, 30][seed as usize % 2], seed);
        let first = optimal_cost(&d, TieBreak::First);
        let free = optimal_cost(&d, TieBreak::PreferFree);
        assert_eq!(first, free, "seed {seed}");
        assert_eq!(first.ok(), brute_force(&d), "seed {seed}");
    }
}

#[test]
fn textbook_three_by_three() {
    let d = Dense::from_rows(&[vec![4, 1, 3], vec![2, 0, 5], vec![3, 2, 2]]).unwrap();
    for algo in Algorithm::ALL {
        let a = solve_dense(&d, algo).unwrap();
        assert_eq!(a.cost, 5, "{algo}"); // 1 + 2 + 2
    }
}

#[test]
fn empty_and_single_cell() {
    let empty = Dense::new(0, 0, vec![]).unwrap();
    let tall = Dense::new(3, 0, vec![]).unwrap();
    let one = Dense::new(1, 1, vec![-7]).unwrap();
    for algo in Algorithm::ALL {
        assert_eq!(solve_dense(&empty, algo).unwrap().cost, 0, "{algo}");
        assert_eq!(solve_dense(&tall, algo).unwrap().cost, 0, "{algo}");
        assert_eq!(solve_dense(&one, algo).unwrap().cost, -7, "{algo}");
    }
}

#[test]
fn every_solver_matches_the_oracle_on_a_seed_sweep() {
    for seed in 0..120 {
        let (rows, cols) = (1 + seed as usize % 6, 1 + (seed as usize / 6) % 7);
        let forbid = [0, 0, 25, 50][seed as usize % 4];
        let d = small(rows, cols, -9, 9, forbid, seed);
        let expected = brute_force(&d);
        for algo in Algorithm::ALL {
            check(&d, algo, expected);
        }
    }
}

#[test]
fn ties_everywhere() {
    // All costs equal: every assignment is optimal, which stresses tie-breaking in each solver.
    for (r, c) in [(1, 1), (4, 4), (3, 6), (6, 3)] {
        let d = Dense::from_fn(r, c, |_, _| 5).unwrap();
        for algo in Algorithm::ALL {
            let a = solve_dense(&d, algo).unwrap();
            assert_eq!(a.cost, 5 * r.min(c) as i64, "{algo} {r}x{c}");
            check_assignment(&d.to_sparse(), &a).unwrap();
        }
    }
}

#[test]
fn infeasible_instances_are_reported() {
    // Two rows that can only use the same single column.
    let d = Dense::from_rows(&[vec![1, FORBIDDEN], vec![2, FORBIDDEN]]).unwrap();
    assert_eq!(brute_force(&d), None);
    for algo in Algorithm::ALL {
        match solve_dense(&d, algo) {
            Err(Error::Infeasible) => {}
            Err(Error::Unsupported(_)) if algo == Algorithm::Jv => {}
            other => panic!("{algo}: {other:?}"),
        }
    }
    // An empty row.
    let d = Dense::from_rows(&[vec![FORBIDDEN, FORBIDDEN], vec![1, 2]]).unwrap();
    for algo in Algorithm::ALL {
        assert!(solve_dense(&d, algo).is_err(), "{algo}");
    }
}

#[test]
fn negative_and_large_costs() {
    let big = 1i64 << 32;
    let d = Dense::from_rows(&[vec![-big, big, 0], vec![big, -big, 7], vec![0, 7, -big]]).unwrap();
    for algo in Algorithm::ALL {
        assert_eq!(solve_dense(&d, algo).unwrap().cost, -3 * big, "{algo}");
    }
}

#[test]
fn costs_beyond_the_exact_range_are_rejected() {
    assert!(matches!(
        Dense::new(1, 1, vec![(1 << 32) + 1]),
        Err(Error::OutOfRange(_))
    ));
}

#[test]
fn machol_wien_has_a_closed_form_optimum() {
    for n in [1, 2, 3, 10, 64, 200] {
        let d = machol_wien(n);
        for algo in Algorithm::ALL {
            if matches!(algo, Algorithm::NetworkSimplex | Algorithm::CostScaling) && n > 64 {
                continue; // quadratic arcs; covered at smaller sizes
            }
            let a = solve_dense(&d, algo).unwrap();
            assert_eq!(a.cost, machol_wien_optimum(n), "{algo} n={n}");
        }
    }
}

#[test]
fn medium_instances_agree_across_solvers_and_pass_both_certificates() {
    for (name, d) in [
        ("uniform", instances::uniform(60, 60, 1000, 11)),
        ("narrow", instances::uniform(60, 60, 3, 12)),
        ("geometric", instances::geometric(60, 500, 13)),
        ("separable", instances::separable(60, 1000, 5, 14)),
        ("wide", instances::uniform(40, 70, 500, 15)),
        ("tall", instances::uniform(70, 40, 500, 16)),
    ] {
        let sparse = d.to_sparse();
        let reference = solve_dense(&d, Algorithm::Hungarian).unwrap();
        certify(&sparse, &reference).unwrap();
        check_duals(&sparse, &reference).unwrap();
        for algo in Algorithm::ALL {
            let a = solve_dense(&d, algo).unwrap();
            assert_eq!(a.cost, reference.cost, "{algo} on {name}");
            certify(&sparse, &a).unwrap_or_else(|e| panic!("{algo} on {name}: {e}"));
        }
    }
}

#[test]
fn sparse_solvers_agree_with_dense_ones_under_forbidden_edges() {
    for seed in 0..12 {
        let base = instances::uniform(30, 30, 200, seed);
        let d = instances::forbid(&base, 60, seed + 100);
        let reference = solve_dense(&d, Algorithm::Hungarian).unwrap();
        let g = d.to_sparse();
        for algo in Algorithm::ALL {
            if algo == Algorithm::Jv {
                continue; // no forbidden edges
            }
            let a = solve_sparse(&g, algo).unwrap();
            assert_eq!(a.cost, reference.cost, "{algo} seed {seed}");
            certify(&g, &a).unwrap();
        }
    }
}

#[test]
fn planted_sparse_instances_are_solved_and_certified() {
    for n in [20, 64, 150] {
        let g = instances::dimacs_sparse(n, n as u64);
        let reference = solve_sparse(&g, Algorithm::Lapmod).unwrap();
        check_duals(&g, &reference).unwrap();
        for algo in [Algorithm::Auction, Algorithm::Hungarian, Algorithm::Ssp] {
            let a = solve_sparse(&g, algo).unwrap();
            assert_eq!(a.cost, reference.cost, "{algo} n={n}");
        }
        certify(&g, &reference).unwrap();
    }
}

#[test]
fn rectangular_duals_have_the_right_signs() {
    // Wide: free columns must carry dual 0 and every column dual <= 0.
    let wide = instances::uniform(8, 20, 100, 5);
    let tall = wide.transposed();
    for algo in [
        Algorithm::Hungarian,
        Algorithm::Crouse,
        Algorithm::Jv,
        Algorithm::Lapmod,
    ] {
        for d in [&wide, &tall] {
            let a = solve_dense(d, algo).unwrap();
            check_duals(&d.to_sparse(), &a).unwrap_or_else(|e| panic!("{algo}: {e}"));
        }
    }
}

#[test]
fn a_wrong_assignment_is_caught_by_both_certificates() {
    let d = Dense::from_rows(&[vec![1, 100], vec![100, 1]]).unwrap();
    let g = d.to_sparse();
    let mut a = solve_dense(&d, Algorithm::Hungarian).unwrap();
    assert_eq!(a.cost, 2);
    // Swap the columns: legal, but not optimal.
    a.row_to_col = vec![Some(1), Some(0)];
    a.cost = 200;
    check_assignment(&g, &a).unwrap();
    assert!(certify(&g, &a).is_err());
    assert!(check_duals(&g, &a).is_err());
}

#[test]
fn unsupported_inputs_say_so() {
    let d = Dense::from_rows(&[vec![1, FORBIDDEN], vec![2, 3]]).unwrap();
    assert!(matches!(
        solve_dense(&d, Algorithm::Jv),
        Err(Error::Unsupported(_))
    ));
    let g = Sparse::from_triplets(2, 3, [(0, 0, 1), (1, 2, 1)]).unwrap();
    assert_eq!(solve_sparse(&g, Algorithm::Auction).unwrap().cost, 2);
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(300))]

    #[test]
    fn solvers_match_the_oracle(
        rows in 1usize..7,
        cols in 1usize..8,
        forbid in prop::sample::select(vec![0u64, 0, 30, 60]),
        seed in any::<u64>(),
        lo in -30i64..0,
        hi in 0i64..30,
    ) {
        let d = small(rows, cols, lo, hi, forbid, seed);
        let expected = brute_force(&d);
        for algo in Algorithm::ALL {
            check(&d, algo, expected);
        }
    }
}
