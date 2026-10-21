//! Criterion statistics for every solver at one size per family. `examples/scale.rs` covers more
//! sizes, peak memory and time-outs; this is the tight confidence interval at a size where every
//! solver finishes in well under a second.

use std::hint::black_box;
use std::time::Duration;

use criterion::{
    BenchmarkGroup, Criterion, criterion_group, criterion_main, measurement::WallTime,
};
use mcflow::flow::{self, FlowAlgorithm};
use mcflow::instances::{dimacs_sparse, random_network, uniform};
use mcflow::{Algorithm, solve_dense, solve_sparse};

fn configure(group: &mut BenchmarkGroup<'_, WallTime>) {
    group.sample_size(10);
    group.warm_up_time(Duration::from_millis(300));
    group.measurement_time(Duration::from_secs(2));
}

fn dense(c: &mut Criterion) {
    let d = uniform(300, 300, 1000, 1);
    let mut group = c.benchmark_group("dense-uniform-300");
    configure(&mut group);
    for algo in Algorithm::ALL {
        group.bench_function(algo.name(), |b| {
            b.iter(|| black_box(solve_dense(&d, algo).unwrap()))
        });
    }
    let m = ndarray::Array2::from_shape_fn((300, 300), |(i, j)| d.get(i, j) as f64);
    group.bench_function("rival-lapjv", |b| {
        b.iter(|| black_box(lapjv::lapjv(&m).unwrap()))
    });
    let pm = pathfinding::matrix::Matrix::from_fn(300, 300, |(i, j)| d.get(i, j));
    group.bench_function("rival-pathfinding", |b| {
        b.iter(|| black_box(pathfinding::kuhn_munkres::kuhn_munkres_min(&pm)))
    });
    group.finish();
}

fn sparse(c: &mut Criterion) {
    let g = dimacs_sparse(4096, 1);
    let mut group = c.benchmark_group("sparse-4096");
    configure(&mut group);
    for algo in [
        Algorithm::Lapmod,
        Algorithm::Auction,
        Algorithm::Ssp,
        Algorithm::CostScaling,
        Algorithm::NetworkSimplex,
    ] {
        group.bench_function(algo.name(), |b| {
            b.iter(|| black_box(solve_sparse(&g, algo).unwrap()))
        });
    }
    group.finish();
}

fn flows(c: &mut Criterion) {
    let net = random_network(5000, 25000, 100, 1000, true, 1).unwrap();
    let mut group = c.benchmark_group("flow-5000-nodes");
    configure(&mut group);
    for algo in FlowAlgorithm::ALL {
        group.bench_function(algo.name(), |b| {
            b.iter(|| black_box(flow::solve(&net, algo).unwrap()))
        });
    }
    group.finish();
}

criterion_group!(benches, dense, sparse, flows);
criterion_main!(benches);
