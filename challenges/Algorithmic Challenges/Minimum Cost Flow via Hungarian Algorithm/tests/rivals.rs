//! The same instances through the two Rust crates this one is benchmarked against.

use mcflow::instances::{geometric, machol_wien, machol_wien_optimum, separable, uniform};
use mcflow::{Algorithm, Dense, solve_dense};
use ndarray::Array2;

fn lapjv_cost(d: &Dense) -> i64 {
    let m = Array2::from_shape_fn((d.rows(), d.cols()), |(i, j)| d.get(i, j) as f64);
    let (row_to_col, _) = lapjv::lapjv(&m).expect("square instance");
    row_to_col
        .iter()
        .enumerate()
        .map(|(i, &j)| d.get(i, j))
        .sum()
}

fn kuhn_munkres_cost(d: &Dense) -> i64 {
    let m = pathfinding::matrix::Matrix::from_fn(d.rows(), d.cols(), |(i, j)| d.get(i, j));
    pathfinding::kuhn_munkres::kuhn_munkres_min(&m).0
}

#[test]
fn rival_crates_agree_with_every_solver() {
    let instances = [
        ("uniform", uniform(80, 80, 1000, 1)),
        ("narrow", uniform(80, 80, 4, 2)),
        ("geometric", geometric(80, 500, 3)),
        ("separable", separable(80, 1000, 5, 4)),
        ("machol-wien", machol_wien(80)),
    ];
    for (name, d) in &instances {
        let want = solve_dense(d, Algorithm::Hungarian).unwrap().cost;
        assert_eq!(lapjv_cost(d), want, "lapjv on {name}");
        assert_eq!(kuhn_munkres_cost(d), want, "pathfinding on {name}");
        for algo in Algorithm::ALL {
            assert_eq!(solve_dense(d, algo).unwrap().cost, want, "{algo} on {name}");
        }
    }
    assert_eq!(kuhn_munkres_cost(&machol_wien(80)), machol_wien_optimum(80));
}
