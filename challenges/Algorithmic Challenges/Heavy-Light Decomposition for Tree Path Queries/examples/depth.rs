use hld::shapes::{Shape, generate};
use hld::{BalancedHld, Layout, PathOps, TopTree, Tree};

fn main() {
    println!(
        "{:<12} {:>9} {:>6} {:>10} {:>9} {:>8}",
        "shape", "n", "log2n", "light_max", "balanced", "toptree"
    );
    for shape in Shape::ALL {
        for n in [1_000usize, 100_000, 1_000_000] {
            let g = generate(shape, n, 7);
            let layout = Layout::new(&Tree::new(g.n, &g.edges).unwrap(), g.root).unwrap();
            let light = (0..n).map(|v| layout.light_edges_to_root(v)).max().unwrap();
            let vals = vec![0i64; n];
            let b = BalancedHld::from_weights(layout.clone(), &vals).max_depth();
            let t = TopTree::from_weights(layout.clone(), &vals).max_depth();
            println!(
                "{:<12} {:>9} {:>6.1} {:>10} {:>9} {:>8}",
                shape.name(),
                n,
                (n as f64).log2(),
                light,
                b,
                t
            );
        }
    }
}
