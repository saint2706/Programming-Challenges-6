//! Maximum bipartite matching (Hopcroft-Karp), used to decide feasibility before the solvers that
//! cannot detect it themselves (the auction would raise prices forever on an infeasible instance).

use crate::matrix::Sparse;

const NONE: usize = usize::MAX;

/// Size of a maximum matching of the bipartite graph whose edges are the entries of `g`.
pub fn max_matching(g: &Sparse) -> usize {
    let (n, m) = (g.rows(), g.cols());
    let mut match_row = vec![NONE; n];
    let mut match_col = vec![NONE; m];
    let mut size = 0usize;

    // A greedy start removes most of the work for the phases below.
    for (i, slot) in match_row.iter_mut().enumerate() {
        if let Some(&j) = g
            .row_cols(i)
            .iter()
            .find(|&&j| match_col[j as usize] == NONE)
        {
            match_col[j as usize] = i;
            *slot = j as usize;
            size += 1;
        }
    }

    let mut dist = vec![0u32; n];
    let mut queue: Vec<usize> = Vec::with_capacity(n);
    let mut next = vec![0usize; n];
    let mut stack: Vec<usize> = Vec::new();
    loop {
        // Layer the rows by alternating-path distance from the free rows.
        queue.clear();
        for i in 0..n {
            if match_row[i] == NONE {
                dist[i] = 0;
                queue.push(i);
            } else {
                dist[i] = u32::MAX;
            }
        }
        let mut reaches_free_col = false;
        let mut head = 0;
        while head < queue.len() {
            let i = queue[head];
            head += 1;
            for &j in g.row_cols(i) {
                let k = match_col[j as usize];
                if k == NONE {
                    reaches_free_col = true;
                } else if dist[k] == u32::MAX {
                    dist[k] = dist[i] + 1;
                    queue.push(k);
                }
            }
        }
        if !reaches_free_col {
            return size;
        }

        // Augment along vertex-disjoint shortest paths with an explicit stack (paths can be long).
        next.fill(0);
        for root in 0..n {
            if match_row[root] != NONE {
                continue;
            }
            stack.clear();
            stack.push(root);
            while let Some(&i) = stack.last() {
                let cols = g.row_cols(i);
                if next[i] == cols.len() {
                    dist[i] = u32::MAX; // dead end in this phase
                    stack.pop();
                    continue;
                }
                let j = cols[next[i]] as usize;
                next[i] += 1;
                let k = match_col[j];
                if k == NONE {
                    for &r in &stack {
                        let jj = g.row_cols(r)[next[r] - 1] as usize;
                        match_row[r] = jj;
                        match_col[jj] = r;
                    }
                    size += 1;
                    break;
                } else if dist[k] == dist[i].wrapping_add(1) {
                    stack.push(k);
                }
            }
        }
    }
}
