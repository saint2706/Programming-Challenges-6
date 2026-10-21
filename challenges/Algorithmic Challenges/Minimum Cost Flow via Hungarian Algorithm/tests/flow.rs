//! The three min-cost-flow solvers against an exhaustive oracle and against each other.

use mcflow::flow::{self, Arc, assignment_network};
use mcflow::instances::{self, Rng, random_network};
use mcflow::maxflow::max_flow;
use mcflow::verify::{certify_flow, check_flow, check_potentials};
use mcflow::{Error, Flow, FlowAlgorithm, Network};
use proptest::prelude::*;

/// Cheapest feasible flow by enumerating every flow vector; `None` if none is feasible.
fn brute_force(net: &Network) -> Option<i128> {
    let arcs = net.arcs();
    let mut x = vec![0i64; arcs.len()];
    let mut best: Option<i128> = None;
    loop {
        let mut bal: Vec<i128> = net.supplies().iter().map(|&b| b as i128).collect();
        let mut cost = 0i128;
        for (a, &f) in arcs.iter().zip(&x) {
            bal[a.from as usize] -= f as i128;
            bal[a.to as usize] += f as i128;
            cost += f as i128 * a.cost as i128;
        }
        if bal.iter().all(|&b| b == 0) && best.is_none_or(|b| cost < b) {
            best = Some(cost);
        }
        // Odometer over 0..=cap for every arc.
        let mut k = 0;
        loop {
            if k == arcs.len() {
                return best;
            }
            if x[k] < arcs[k].cap {
                x[k] += 1;
                break;
            }
            x[k] = 0;
            k += 1;
        }
    }
}

fn tiny_network(seed: u64) -> Network {
    let mut rng = Rng::new(seed);
    let nodes = 3 + rng.below(3) as usize;
    let mut net = Network::new(nodes);
    let arcs = 3 + rng.below(5) as usize;
    for _ in 0..arcs {
        let from = rng.below(nodes as u64) as usize;
        let to = (from + 1 + rng.below(nodes as u64 - 1) as usize) % nodes;
        net.add_arc(from, to, rng.range(0, 3), rng.range(-5, 9))
            .unwrap();
    }
    // Supplies: move 0..=3 units from one node to another.
    let (s, t) = (
        rng.below(nodes as u64) as usize,
        rng.below(nodes as u64) as usize,
    );
    if s != t {
        let amount = rng.range(0, 3);
        net.set_supply(s, amount).unwrap();
        net.set_supply(t, -amount).unwrap();
    }
    net
}

fn check(net: &Network, algo: FlowAlgorithm, expected: Option<i128>) {
    let what = format!("{} on {net:?}", algo.name());
    match flow::solve(net, algo) {
        Ok(f) => {
            let want = expected.unwrap_or_else(|| panic!("{what}: solved an infeasible network"));
            assert_eq!(f.cost, want, "{what}");
            check_flow(net, &f).unwrap_or_else(|e| panic!("{what}: {e}"));
            certify_flow(net, &f).unwrap_or_else(|e| panic!("{what}: {e}"));
            if let Some(pi) = &f.potential {
                check_potentials(net, &f, pi).unwrap_or_else(|e| panic!("{what}: {e}"));
            }
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
fn a_hand_checked_network() {
    // 0 -> 1 -> 3 is cheap but narrow, 0 -> 2 -> 3 is dear but wide; send 4 units.
    let mut net = Network::new(4);
    net.add_arc(0, 1, 2, 1).unwrap();
    net.add_arc(1, 3, 2, 1).unwrap();
    net.add_arc(0, 2, 5, 3).unwrap();
    net.add_arc(2, 3, 5, 4).unwrap();
    net.set_supply(0, 4).unwrap();
    net.set_supply(3, -4).unwrap();
    for algo in FlowAlgorithm::ALL {
        let f = flow::solve(&net, algo).unwrap();
        assert_eq!(f.cost, 2 * 2 + 2 * 7, "{algo:?}"); // two units at 2, two at 7
        assert_eq!(f.flow, vec![2, 2, 2, 2], "{algo:?}");
    }
}

#[test]
fn negative_cost_cycles_with_finite_capacity_are_saturated() {
    // 0 -> 1 -> 2 -> 0 costs -1 each: the cycle should carry its full capacity.
    let mut net = Network::new(3);
    net.add_arc(0, 1, 4, -1).unwrap();
    net.add_arc(1, 2, 3, -1).unwrap();
    net.add_arc(2, 0, 5, -1).unwrap();
    for algo in FlowAlgorithm::ALL {
        let f = flow::solve(&net, algo).unwrap();
        assert_eq!(f.cost, -9, "{algo:?}");
        certify_flow(&net, &f).unwrap();
    }
}

#[test]
fn zero_capacity_parallel_and_disconnected_pieces() {
    let mut net = Network::new(5);
    net.add_arc(0, 1, 0, -100).unwrap(); // unusable
    net.add_arc(0, 1, 1, 5).unwrap();
    net.add_arc(0, 1, 1, 2).unwrap(); // parallel and cheaper
    net.add_arc(3, 4, 7, -3).unwrap(); // disconnected island with no supply
    net.set_supply(0, 2).unwrap();
    net.set_supply(1, -2).unwrap();
    for algo in FlowAlgorithm::ALL {
        let f = flow::solve(&net, algo).unwrap();
        assert_eq!(f.cost, 7, "{algo:?}"); // 2 + 5; the island has no supply and no cycle
        assert_eq!(f.flow, vec![0, 1, 1, 0], "{algo:?}");
    }
}

#[test]
fn infeasible_and_unbalanced_networks() {
    let mut net = Network::new(3);
    net.add_arc(0, 1, 1, 1).unwrap();
    net.set_supply(0, 2).unwrap();
    net.set_supply(1, -2).unwrap(); // capacity 1 cannot carry 2
    for algo in FlowAlgorithm::ALL {
        assert_eq!(flow::solve(&net, algo), Err(Error::Infeasible), "{algo:?}");
    }
    let mut net = Network::new(2);
    net.add_arc(0, 1, 5, 1).unwrap();
    net.set_supply(0, 3).unwrap();
    for algo in FlowAlgorithm::ALL {
        assert_eq!(
            flow::solve(&net, algo),
            Err(Error::Unbalanced { sum: 3 }),
            "{algo:?}"
        );
    }
    // Demand with no way in at all.
    let mut net = Network::new(3);
    net.add_arc(1, 2, 5, 1).unwrap();
    net.set_supply(0, 1).unwrap();
    net.set_supply(2, -1).unwrap();
    for algo in FlowAlgorithm::ALL {
        assert_eq!(flow::solve(&net, algo), Err(Error::Infeasible), "{algo:?}");
    }
}

#[test]
fn a_network_with_no_supply_is_zero_cost() {
    let mut net = Network::new(3);
    net.add_arc(0, 1, 3, 4).unwrap();
    net.add_arc(1, 2, 3, 4).unwrap();
    for algo in FlowAlgorithm::ALL {
        let f = flow::solve(&net, algo).unwrap();
        assert_eq!((f.cost, f.flow), (0, vec![0, 0]), "{algo:?}");
    }
}

#[test]
fn construction_limits_are_enforced() {
    let mut net = Network::new(3);
    assert!(matches!(net.add_arc(0, 0, 1, 1), Err(Error::Invalid(_))));
    assert!(matches!(net.add_arc(0, 9, 1, 1), Err(Error::Invalid(_))));
    assert!(matches!(
        net.add_arc(0, 1, -1, 1),
        Err(Error::OutOfRange(_))
    ));
    assert!(matches!(
        net.add_arc(0, 1, (1 << 30) + 1, 1),
        Err(Error::OutOfRange(_))
    ));
    assert!(matches!(
        net.add_arc(0, 1, 1, (1 << 32) + 1),
        Err(Error::OutOfRange(_))
    ));
    assert!(matches!(
        net.set_supply(0, (1 << 30) + 1),
        Err(Error::OutOfRange(_))
    ));
}

#[test]
fn solvers_agree_with_exhaustive_search_on_tiny_networks() {
    for seed in 0..400 {
        let net = tiny_network(seed);
        let expected = brute_force(&net);
        for algo in FlowAlgorithm::ALL {
            check(&net, algo, expected);
        }
    }
}

#[test]
fn solvers_agree_on_larger_random_networks() {
    for (nodes, arcs, negative) in [
        (30, 120, false),
        (30, 120, true),
        (80, 400, true),
        (60, 900, false),
    ] {
        for seed in 0..5 {
            let net = random_network(nodes, arcs, 40, 1000, negative, seed).unwrap();
            let flows: Vec<Flow> = FlowAlgorithm::ALL
                .iter()
                .map(|&a| flow::solve(&net, a).unwrap())
                .collect();
            for (algo, f) in FlowAlgorithm::ALL.iter().zip(&flows) {
                assert_eq!(f.cost, flows[0].cost, "{algo:?} {nodes}x{arcs} seed {seed}");
                certify_flow(&net, f).unwrap_or_else(|e| panic!("{algo:?}: {e}"));
                if let Some(pi) = &f.potential {
                    check_potentials(&net, f, pi).unwrap_or_else(|e| panic!("{algo:?}: {e}"));
                }
            }
        }
    }
}

#[test]
fn an_assignment_through_the_flow_network_costs_what_the_assignment_does() {
    let d = instances::uniform(25, 25, 300, 3);
    let g = d.to_sparse();
    let net = assignment_network(&g).unwrap();
    assert_eq!(net.nodes(), 50); // 25 rows + 25 columns, no dummy for a square matrix
    assert_eq!(net.arcs().len(), 25 * 25);
    let wide = assignment_network(&instances::uniform(5, 8, 9, 1).to_sparse()).unwrap();
    assert_eq!((wide.nodes(), wide.arcs().len()), (14, 5 * 8 + 8)); // plus a dummy supplying 3
    assert_eq!(wide.supplies().iter().sum::<i64>(), 0);
    let want = mcflow::solve_dense(&d, mcflow::Algorithm::Hungarian)
        .unwrap()
        .cost;
    for algo in FlowAlgorithm::ALL {
        assert_eq!(
            flow::solve(&net, algo).unwrap().cost,
            want as i128,
            "{algo:?}"
        );
    }
}

/// Minimum cut by trying every vertex subset that contains `s` and not `t`.
fn min_cut(nodes: usize, arcs: &[Arc], s: usize, t: usize) -> i64 {
    (0u32..1 << nodes)
        .filter(|m| m >> s & 1 == 1 && m >> t & 1 == 0)
        .map(|m| {
            arcs.iter()
                .filter(|a| m >> a.from & 1 == 1 && m >> a.to & 1 == 0)
                .map(|a| a.cap)
                .sum()
        })
        .min()
        .unwrap()
}

#[test]
fn max_flow_equals_min_cut() {
    for seed in 0..200 {
        let mut rng = Rng::new(seed);
        let nodes = 4 + rng.below(5) as usize;
        let arcs: Vec<Arc> = (0..4 + rng.below(14) as usize)
            .map(|_| {
                let from = rng.below(nodes as u64) as usize;
                let to = (from + 1 + rng.below(nodes as u64 - 1) as usize) % nodes;
                Arc {
                    from: from as u32,
                    to: to as u32,
                    cap: rng.range(0, 9),
                    cost: 0,
                }
            })
            .collect();
        assert_eq!(
            max_flow(nodes, &arcs, 0, nodes - 1),
            min_cut(nodes, &arcs, 0, nodes - 1),
            "seed {seed}"
        );
    }
}

#[test]
fn cost_scaling_is_epsilon_optimal_after_every_phase() {
    // The final flow being optimal is a weak test of the method: a relabel that overshoots still
    // lands on the optimum unless two solutions differ by one cost unit across a long cycle. The
    // invariant itself is checked instead: when a phase ends, no residual arc has a reduced cost
    // below -epsilon, and epsilon falls by the scaling factor down to exactly 1.
    let mut nets: Vec<Network> = (0..6)
        .map(|s| random_network(40, 200, 30, 50, s % 2 == 0, s).unwrap())
        .collect();
    nets.push(assignment_network(&instances::uniform(30, 30, 3, 8).to_sparse()).unwrap());
    nets.push(assignment_network(&instances::separable(30, 100, 2, 9).to_sparse()).unwrap());
    for net in &nets {
        let (flow, phases) =
            mcflow::cost_scaling::solve_with_stats(net, mcflow::cost_scaling::Options::default())
                .unwrap();
        assert_eq!(phases.last().unwrap().eps, 1);
        for w in phases.windows(2) {
            assert_eq!(w[1].eps, (w[0].eps / mcflow::cost_scaling::ALPHA).max(1));
        }
        for p in &phases {
            assert!(
                p.worst_slack <= p.eps,
                "phase eps={}: slack {} exceeds epsilon",
                p.eps,
                p.worst_slack
            );
        }
        assert!(phases.iter().map(|p| p.pushes).sum::<u64>() > 0);
        let f = Flow {
            cost: net
                .arcs()
                .iter()
                .zip(&flow)
                .map(|(a, &x)| x as i128 * a.cost as i128)
                .sum(),
            flow,
            potential: None,
        };
        certify_flow(net, &f).unwrap();
    }
}

#[test]
fn network_simplex_reports_its_pivots() {
    let net = random_network(40, 200, 30, 50, true, 4).unwrap();
    let (flow, pi, stats) = mcflow::simplex::solve_with_stats(&net).unwrap();
    assert!(stats.pivots > 0 && stats.degenerate <= stats.pivots);
    let f = Flow {
        cost: net
            .arcs()
            .iter()
            .zip(&flow)
            .map(|(a, &x)| x as i128 * a.cost as i128)
            .sum(),
        flow,
        potential: Some(pi.clone()),
    };
    check_potentials(&net, &f, &pi).unwrap();
}

#[test]
fn a_wrong_flow_is_caught() {
    let mut net = Network::new(3);
    net.add_arc(0, 1, 5, 1).unwrap();
    net.add_arc(1, 2, 5, 1).unwrap();
    net.add_arc(0, 2, 5, 9).unwrap();
    net.set_supply(0, 2).unwrap();
    net.set_supply(2, -2).unwrap();
    let good = flow::solve(&net, FlowAlgorithm::Ssp).unwrap();
    assert_eq!(good.cost, 4);
    // Feasible but not optimal: route through the expensive arc.
    let bad = Flow {
        cost: 18,
        flow: vec![0, 0, 2],
        potential: None,
    };
    check_flow(&net, &bad).unwrap();
    assert!(certify_flow(&net, &bad).is_err());
    // Infeasible: breaks conservation.
    let broken = Flow {
        cost: 1,
        flow: vec![1, 0, 0],
        potential: None,
    };
    assert!(check_flow(&net, &broken).is_err());
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(300))]

    #[test]
    fn solvers_match_exhaustive_search(seed in any::<u64>()) {
        let net = tiny_network(seed);
        let expected = brute_force(&net);
        for algo in FlowAlgorithm::ALL {
            check(&net, algo, expected);
        }
    }
}
