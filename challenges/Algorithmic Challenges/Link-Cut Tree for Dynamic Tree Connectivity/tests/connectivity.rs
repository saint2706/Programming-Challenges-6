//! The offline link-cut-tree solver against rebuilding a union-find for every query.

use lct::connectivity::{Event, EventError, random_events, solve_offline, solve_rebuild};
use proptest::prelude::*;

#[test]
fn validation_errors_name_the_event() {
    use Event::*;
    assert_eq!(
        solve_offline(3, &[Add(0, 3)]),
        Err(EventError::OutOfRange { at: 0 })
    );
    assert_eq!(
        solve_offline(3, &[Add(1, 1)]),
        Err(EventError::SelfLoop { at: 0 })
    );
    assert_eq!(
        solve_offline(3, &[Add(0, 1), Add(1, 0)]),
        Err(EventError::Duplicate { at: 1 })
    );
    assert_eq!(
        solve_offline(3, &[Add(0, 1), Remove(1, 2)]),
        Err(EventError::Missing { at: 1 })
    );
    assert_eq!(
        solve_rebuild(3, &[Remove(0, 1)]),
        Err(EventError::Missing { at: 0 })
    );
}

#[test]
fn a_removed_edge_can_be_added_again() {
    use Event::*;
    let events = [
        Add(0, 1),
        Query(0, 1),
        Remove(0, 1),
        Query(0, 1),
        Add(1, 0),
        Query(0, 1),
    ];
    let got = solve_offline(2, &events).unwrap();
    assert_eq!(got.answers, [true, false, true]);
}

/// A triangle, then losing one edge at a time: the surviving edges must keep everything joined
/// until the last one goes, which is the case a naive forest (no spare edges) gets wrong.
#[test]
fn a_cycle_survives_losing_a_tree_edge() {
    use Event::*;
    let events = [
        Add(0, 1),
        Add(1, 2),
        Add(0, 2),
        Remove(0, 1),
        Query(0, 1),
        Remove(1, 2),
        Query(0, 1),
        Query(0, 2),
    ];
    let got = solve_offline(3, &events).unwrap();
    assert_eq!(got.answers, [true, false, true]);
    assert_eq!(got.answers, solve_rebuild(3, &events).unwrap());
}

#[test]
fn counters_add_up() {
    let events = random_events(40, 60, 4000, 30, 5);
    let adds = events
        .iter()
        .filter(|e| matches!(e, Event::Add(..)))
        .count();
    let queries = events
        .iter()
        .filter(|e| matches!(e, Event::Query(..)))
        .count();
    let got = solve_offline(40, &events).unwrap();
    assert_eq!(got.answers.len(), queries);
    assert_eq!(got.tree_inserts + got.swaps + got.dropped, adds);
    assert!(
        got.swaps > 0 && got.dropped > 0,
        "the workload should exercise both rules"
    );
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(300))]

    #[test]
    fn matches_rebuild(
        n in 2usize..30,
        target in 1usize..60,
        count in 1usize..400,
        query_percent in 5usize..60,
        seed in any::<u64>(),
    ) {
        let events = random_events(n, target, count, query_percent, seed);
        let fast = solve_offline(n, &events).unwrap();
        prop_assert_eq!(fast.answers, solve_rebuild(n, &events).unwrap());
    }
}
