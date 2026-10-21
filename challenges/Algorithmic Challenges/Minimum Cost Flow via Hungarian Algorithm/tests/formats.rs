//! The file formats: round trips, the forms the parser accepts, and what it refuses.

use mcflow::dimacs::{parse_asn, parse_matrix, parse_min, write_asn, write_matrix, write_min};
use mcflow::flow::{self, assignment_network};
use mcflow::instances::{dimacs_sparse, forbid, random_network, uniform};
use mcflow::{Algorithm, Error, FORBIDDEN, FlowAlgorithm, solve_dense};
use proptest::prelude::*;

#[test]
fn matrix_round_trips_with_forbidden_entries() {
    let d = forbid(&uniform(7, 9, 50, 3), 40, 4);
    assert!(d.has_forbidden());
    assert_eq!(parse_matrix(&write_matrix(&d)).unwrap(), d);
}

#[test]
fn matrix_parser_accepts_commas_spaces_comments_and_markers() {
    let d = parse_matrix("# a comment\n1, 2, x\n4 5 inf\n\n7,,9\n").unwrap();
    assert_eq!((d.rows(), d.cols()), (3, 3));
    assert_eq!(d.row(0), &[1, 2, FORBIDDEN]);
    assert_eq!(d.row(1), &[4, 5, FORBIDDEN]);
    assert_eq!(d.row(2), &[7, FORBIDDEN, 9]);
}

#[test]
fn matrix_parser_rejects_garbage() {
    assert!(matches!(parse_matrix("1,2\n3\n"), Err(Error::Invalid(_)))); // ragged
    assert!(matches!(parse_matrix("1,two\n"), Err(Error::Invalid(_))));
    assert!(matches!(
        parse_matrix("99999999999\n"),
        Err(Error::OutOfRange(_))
    ));
}

#[test]
fn dimacs_min_round_trips() {
    let net = random_network(20, 60, 30, 100, true, 5).unwrap();
    assert_eq!(parse_min(&write_min(&net)).unwrap(), net);
}

#[test]
fn dimacs_min_parser_reads_comments_and_reports_the_line() {
    let net = parse_min("c example\np min 3 2\nn 1 4\nn 3 -4\na 1 2 0 5 1\na 2 3 0 5 2\n").unwrap();
    assert_eq!(flow::solve(&net, FlowAlgorithm::Ssp).unwrap().cost, 12);
    let err = parse_min("p min 2 1\na 1 2 0 5\n").unwrap_err();
    assert!(err.to_string().contains("line 2"), "{err}");
    assert!(matches!(
        parse_min("p min 2 1\na 1 2 3 5 1\n"),
        Err(Error::Unsupported(_))
    )); // lower bound
    assert!(matches!(parse_min("a 1 2 0 5 1\n"), Err(Error::Invalid(_)))); // arc before p line
    assert!(matches!(
        parse_min("c only a comment\n"),
        Err(Error::Invalid(_))
    ));
}

#[test]
fn dimacs_assignment_round_trips_and_solves() {
    let g = dimacs_sparse(40, 6);
    let back = parse_asn(&write_asn(&g)).unwrap();
    assert_eq!(back, g);
    let a = mcflow::solve_sparse(&g, Algorithm::Lapmod).unwrap();
    let b = mcflow::solve_sparse(&back, Algorithm::Auction).unwrap();
    assert_eq!(a.cost, b.cost);
}

#[test]
fn dimacs_assignment_rejects_arcs_inside_one_side() {
    let err = parse_asn("p asn 4 1\nn 1\nn 2\na 1 2 5\n").unwrap_err();
    assert!(matches!(err, Error::Invalid(_)), "{err}");
}

#[test]
fn the_reduction_to_flow_keeps_the_optimum() {
    // The CLI's `convert --to min`: the assignment as a min-cost flow file, solved by flow code.
    let d = forbid(&uniform(12, 12, 100, 7), 50, 8);
    let want = solve_dense(&d, Algorithm::Hungarian).unwrap().cost;
    let text = write_min(&assignment_network(&d.to_sparse()).unwrap());
    let net = parse_min(&text).unwrap();
    for algo in FlowAlgorithm::ALL {
        assert_eq!(
            flow::solve(&net, algo).unwrap().cost,
            want as i128,
            "{algo:?}"
        );
    }
}

proptest::proptest! {
    #![proptest_config(proptest::prelude::ProptestConfig::with_cases(2000))]

    /// The parsers read untrusted files: whatever they are given, they must answer with a value or
    /// an error and never panic, allocate without bound, or overflow.
    #[test]
    fn parsers_never_panic_on_token_soup(text in "[pcnalmisx0-9, \n#-]{0,300}") {
        let _ = parse_min(&text);
        let _ = parse_asn(&text);
        let _ = parse_matrix(&text);
    }

    #[test]
    fn parsers_never_panic_on_big_numbers(
        header in prop::sample::select(vec!["p min", "p asn"]),
        a in 0u64..u64::MAX,
        b in 0u64..u64::MAX,
        c in i64::MIN..i64::MAX,
    ) {
        let text = format!("{header} {a} {b}\nn {a} {c}\na {b} {a} 0 {b} {c}\na {a} {b} {c}\n");
        let _ = parse_min(&text);
        let _ = parse_asn(&text);
    }
}
