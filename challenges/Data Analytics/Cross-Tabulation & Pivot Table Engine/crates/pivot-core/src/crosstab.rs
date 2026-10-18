//! Cross-tabulation proper: a table of counts of two categorical fields, and the statistics that say
//! whether the two are associated: Pearson chi-square (with Yates' correction for 2x2), the
//! likelihood-ratio G, Cramer's V (and its bias-corrected form), standardized residuals, and for a
//! 2x2 table Fisher's exact test and the odds ratio with a confidence interval.
//!
//! With a third *layer* field the table is repeated within each layer and compared with the pooled
//! one: the Mantel-Haenszel common odds ratio, and a flag for Simpson's paradox (the pooled
//! association points one way and the layers, taken together, the other).

use serde::{Deserialize, Serialize};

use crate::encode::{self, Encoded};
use crate::error::{Error, Result};
use crate::filter;
use crate::spec::{Dim, Filter};
use crate::special::{chi2_sf, hypergeom_pmf};
use crate::table::Table;

const Z95: f64 = 1.959_963_984_540_054;

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CrosstabSpec {
    pub row: Dim,
    pub col: Dim,
    #[serde(default)]
    pub layer: Option<Dim>,
    #[serde(default)]
    pub filters: Vec<Filter>,
    #[serde(default = "default_blank")]
    pub blank_label: String,
}

fn default_blank() -> String {
    "(blank)".to_string()
}

#[derive(Debug, Clone, Serialize)]
pub struct Stats {
    pub n: f64,
    pub rows: usize,
    pub cols: usize,
    pub dof: usize,
    pub chi2: f64,
    pub p_value: f64,
    /// 2x2 only: Pearson chi-square with Yates' continuity correction.
    pub chi2_yates: Option<f64>,
    pub p_yates: Option<f64>,
    pub g: f64,
    pub p_g: f64,
    pub cramers_v: f64,
    pub cramers_v_corrected: f64,
    pub contingency_coefficient: f64,
    /// 2x2 only, signed.
    pub phi: Option<f64>,
    pub expected: Vec<Vec<f64>>,
    pub residuals: Vec<Vec<f64>>,
    pub min_expected: f64,
    pub share_expected_below_5: f64,
    pub fisher_p: Option<f64>,
    pub odds_ratio: Option<f64>,
    pub odds_ratio_ci: Option<[f64; 2]>,
}

#[derive(Debug, Clone, Serialize)]
pub struct Layer {
    pub label: String,
    pub counts: Vec<Vec<f64>>,
    pub n: f64,
    /// 2x2: the share of the first column within each row (e.g. the admission rate per gender).
    pub rates: Option<[Option<f64>; 2]>,
    pub odds_ratio: Option<f64>,
    pub stats: Option<Stats>,
}

#[derive(Debug, Clone, Serialize)]
pub struct Simpson {
    pub pooled_odds_ratio: Option<f64>,
    pub pooled_rates: [Option<f64>; 2],
    pub mantel_haenszel_or: Option<f64>,
    pub mantel_haenszel_ci: Option<[f64; 2]>,
    /// Layers whose odds ratio lies on the same side of 1 as the pooled one.
    pub layers_agreeing_with_pooled: usize,
    pub layers_counted: usize,
    /// The pooled and the layer-adjusted association point opposite ways.
    pub reversal: bool,
    pub verdict: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct Crosstab {
    pub row_field: String,
    pub col_field: String,
    pub layer_field: Option<String>,
    pub row_labels: Vec<String>,
    pub col_labels: Vec<String>,
    pub counts: Vec<Vec<f64>>,
    pub row_totals: Vec<f64>,
    pub col_totals: Vec<f64>,
    pub n: f64,
    pub stats: Option<Stats>,
    pub layers: Vec<Layer>,
    pub simpson: Option<Simpson>,
    pub warnings: Vec<String>,
}

fn tabulate(re: &Encoded, ce: &Encoded, rows_in: impl Iterator<Item = usize>) -> Vec<Vec<f64>> {
    let mut counts = vec![vec![0.0; ce.labels.len()]; re.labels.len()];
    for i in rows_in {
        counts[re.codes[i] as usize][ce.codes[i] as usize] += 1.0;
    }
    counts
}

/// Pearson chi-square and friends for a table of counts. Empty rows and columns are ignored.
pub fn stats(counts: &[Vec<f64>]) -> Option<Stats> {
    let keep_rows: Vec<usize> = (0..counts.len())
        .filter(|&i| counts[i].iter().sum::<f64>() > 0.0)
        .collect();
    let n_cols = counts.first().map_or(0, Vec::len);
    let keep_cols: Vec<usize> = (0..n_cols)
        .filter(|&j| counts.iter().map(|r| r[j]).sum::<f64>() > 0.0)
        .collect();
    if keep_rows.len() < 2 || keep_cols.len() < 2 {
        return None;
    }
    let t: Vec<Vec<f64>> = keep_rows
        .iter()
        .map(|&i| keep_cols.iter().map(|&j| counts[i][j]).collect())
        .collect();
    let (r, c) = (t.len(), t[0].len());
    let rt: Vec<f64> = t.iter().map(|row| row.iter().sum()).collect();
    let ct: Vec<f64> = (0..c).map(|j| t.iter().map(|row| row[j]).sum()).collect();
    let n: f64 = rt.iter().sum();

    let mut expected = vec![vec![0.0; c]; r];
    let mut residuals = vec![vec![0.0; c]; r];
    let (mut chi2, mut g, mut yates) = (0.0, 0.0, 0.0);
    for i in 0..r {
        for j in 0..c {
            let e = rt[i] * ct[j] / n;
            let o = t[i][j];
            expected[i][j] = e;
            chi2 += (o - e).powi(2) / e;
            if o > 0.0 {
                g += 2.0 * o * (o / e).ln();
            }
            let d = (o - e).abs();
            yates += (d - d.min(0.5)).powi(2) / e;
            let var = e * (1.0 - rt[i] / n) * (1.0 - ct[j] / n);
            residuals[i][j] = if var > 0.0 { (o - e) / var.sqrt() } else { 0.0 };
        }
    }
    let dof = (r - 1) * (c - 1);
    let k = (r.min(c) - 1) as f64;
    let v = (chi2 / (n * k)).sqrt();
    let phi2_corr = (chi2 / n - dof as f64 / (n - 1.0)).max(0.0);
    let (rc, cc) = (
        r as f64 - (r as f64 - 1.0).powi(2) / (n - 1.0),
        c as f64 - (c as f64 - 1.0).powi(2) / (n - 1.0),
    );
    let v_corr = (phi2_corr / (rc.min(cc) - 1.0)).sqrt();
    let flat: Vec<f64> = expected.iter().flatten().copied().collect();
    let two = r == 2 && c == 2;
    let (a, b, cc2, d) = if two {
        (t[0][0], t[0][1], t[1][0], t[1][1])
    } else {
        (0.0, 0.0, 0.0, 0.0)
    };
    Some(Stats {
        n,
        rows: r,
        cols: c,
        dof,
        chi2,
        p_value: chi2_sf(chi2, dof as f64),
        chi2_yates: two.then_some(yates),
        p_yates: two.then(|| chi2_sf(yates, 1.0)),
        g,
        p_g: chi2_sf(g, dof as f64),
        cramers_v: v,
        cramers_v_corrected: if v_corr.is_finite() { v_corr } else { 0.0 },
        contingency_coefficient: (chi2 / (chi2 + n)).sqrt(),
        phi: two.then(|| (a * d - b * cc2) / (rt[0] * rt[1] * ct[0] * ct[1]).sqrt()),
        expected,
        residuals,
        min_expected: flat.iter().copied().fold(f64::INFINITY, f64::min),
        share_expected_below_5: flat.iter().filter(|&&e| e < 5.0).count() as f64
            / flat.len() as f64,
        fisher_p: two.then(|| fisher_two_sided(a, b, cc2, d)),
        odds_ratio: two.then(|| odds_ratio(a, b, cc2, d).0).flatten(),
        odds_ratio_ci: if two {
            odds_ratio(a, b, cc2, d).1
        } else {
            None
        },
    })
}

/// Odds ratio `ad / bc` and its Woolf 95% interval; a zero cell adds 0.5 to every cell (Haldane).
pub fn odds_ratio(a: f64, b: f64, c: f64, d: f64) -> (Option<f64>, Option<[f64; 2]>) {
    let (a, b, c, d) = if a == 0.0 || b == 0.0 || c == 0.0 || d == 0.0 {
        (a + 0.5, b + 0.5, c + 0.5, d + 0.5)
    } else {
        (a, b, c, d)
    };
    if a + b + c + d == 0.0 {
        return (None, None);
    }
    let or = a * d / (b * c);
    let se = (1.0 / a + 1.0 / b + 1.0 / c + 1.0 / d).sqrt();
    (
        Some(or),
        Some([(or.ln() - Z95 * se).exp(), (or.ln() + Z95 * se).exp()]),
    )
}

/// Fisher's exact test, two-sided: the total probability of every table (with these margins)
/// that is no more likely than the observed one.
pub fn fisher_two_sided(a: f64, b: f64, c: f64, d: f64) -> f64 {
    let (a, b, c, d) = (a as u64, b as u64, c as u64, d as u64);
    let (n, row0, col0) = (a + b + c + d, a + b, a + c);
    let lo = col0.saturating_sub(n - row0);
    let hi = row0.min(col0);
    let observed = hypergeom_pmf(a, n, row0, col0);
    let cutoff = observed * (1.0 + 1e-7);
    (lo..=hi)
        .map(|x| hypergeom_pmf(x, n, row0, col0))
        .filter(|&p| p <= cutoff)
        .sum::<f64>()
        .min(1.0)
}

fn is_2x2(counts: &[Vec<f64>]) -> bool {
    counts.len() == 2 && counts[0].len() == 2
}

fn rates(counts: &[Vec<f64>]) -> [Option<f64>; 2] {
    let rate = |row: &Vec<f64>| {
        let t: f64 = row.iter().sum();
        (t > 0.0).then(|| row[0] / t)
    };
    [rate(&counts[0]), rate(&counts[1])]
}

fn simpson(
    counts: &[Vec<f64>],
    layers: &[Layer],
    row_labels: &[String],
    col_labels: &[String],
) -> Simpson {
    let pooled = odds_ratio(counts[0][0], counts[0][1], counts[1][0], counts[1][1]).0;
    // sums over layers for the Mantel-Haenszel estimate and its Robins-Breslow-Greenland variance
    let (mut r_sum, mut s_sum, mut p_r, mut ps_qr, mut q_s) = (0.0, 0.0, 0.0, 0.0, 0.0);
    let (mut counted, mut agree) = (0, 0);
    for l in layers {
        let (a, b, c, d) = (
            l.counts[0][0],
            l.counts[0][1],
            l.counts[1][0],
            l.counts[1][1],
        );
        let n = a + b + c + d;
        if n == 0.0 {
            continue;
        }
        let (p, q, r, s) = ((a + d) / n, (b + c) / n, a * d / n, b * c / n);
        r_sum += r;
        s_sum += s;
        p_r += p * r;
        ps_qr += p * s + q * r;
        q_s += q * s;
        counted += 1;
        if let (Some(lo), Some(po)) = (l.odds_ratio, pooled)
            && (lo - 1.0) * (po - 1.0) > 0.0
        {
            agree += 1;
        }
    }
    let mh = (s_sum > 0.0 && r_sum > 0.0).then(|| r_sum / s_sum);
    let ci = mh.map(|m| {
        let var = p_r / (2.0 * r_sum * r_sum)
            + ps_qr / (2.0 * r_sum * s_sum)
            + q_s / (2.0 * s_sum * s_sum);
        let se = var.max(0.0).sqrt();
        [(m.ln() - Z95 * se).exp(), (m.ln() + Z95 * se).exp()]
    });
    let reversal = matches!((pooled, mh), (Some(p), Some(m)) if (p - 1.0) * (m - 1.0) < 0.0);
    let verdict = match (pooled, mh) {
        (Some(p), Some(m)) => {
            let dir = |x: f64| if x > 1.0 { "higher" } else { "lower" };
            let (g0, g1, outcome) = (&row_labels[0], &row_labels[1], &col_labels[0]);
            if reversal {
                format!(
                    "Simpson's paradox: pooled, '{g0}' has {} odds of '{outcome}' than '{g1}' (OR {p:.2}), but within the layers it has {} (Mantel-Haenszel OR {m:.2}); {agree} of {counted} layers agree with the pooled direction.",
                    dir(p),
                    dir(m)
                )
            } else {
                format!(
                    "No reversal: pooled OR {p:.2} and layer-adjusted (Mantel-Haenszel) OR {m:.2} point the same way; {agree} of {counted} layers agree with the pooled direction."
                )
            }
        }
        _ => "Too few counts to compare the pooled and layer-adjusted association.".to_string(),
    };
    Simpson {
        pooled_odds_ratio: pooled,
        pooled_rates: rates(counts),
        mantel_haenszel_or: mh,
        mantel_haenszel_ci: ci,
        layers_agreeing_with_pooled: agree,
        layers_counted: counted,
        reversal,
        verdict,
    }
}

pub fn crosstab(table: &Table, spec: &CrosstabSpec) -> Result<Crosstab> {
    let mut keep = filter::mask(table, &spec.filters)?;
    let dims: Vec<&Dim> = [Some(&spec.row), Some(&spec.col), spec.layer.as_ref()]
        .into_iter()
        .flatten()
        .collect();
    for dim in dims.iter().filter(|d| d.hide_blank) {
        let enc = encode::encode(table, dim, &keep, &spec.blank_label)?;
        if let Some(b) = enc.blank {
            for (k, code) in keep.iter_mut().zip(&enc.codes) {
                if *k && *code == b {
                    *k = false;
                }
            }
        }
    }
    let re = encode::encode(table, &spec.row, &keep, &spec.blank_label)?;
    let ce = encode::encode(table, &spec.col, &keep, &spec.blank_label)?;
    let used: Vec<usize> = (0..table.rows).filter(|&i| keep[i]).collect();
    if used.is_empty() {
        return Err(Error::new("no rows match the filters"));
    }
    let counts = tabulate(&re, &ce, used.iter().copied());
    let row_totals: Vec<f64> = counts.iter().map(|r| r.iter().sum()).collect();
    let col_totals: Vec<f64> = (0..ce.labels.len())
        .map(|j| counts.iter().map(|r| r[j]).sum())
        .collect();
    let n: f64 = row_totals.iter().sum();
    let mut warnings = Vec::new();
    let st = stats(&counts);
    if st.is_none() {
        warnings
            .push("each field needs at least two values with data for the statistics".to_string());
    }
    if let Some(s) = &st
        && (s.share_expected_below_5 > 0.2 || s.min_expected < 1.0)
    {
        warnings.push(format!(
            "{:.0}% of cells have an expected count below 5 (smallest {:.2}): the chi-square p-value is unreliable; prefer Fisher's exact test (2x2) or merge categories",
            s.share_expected_below_5 * 100.0,
            s.min_expected
        ));
    }

    let mut layers = Vec::new();
    let mut simp = None;
    let layer_field = spec.layer.as_ref().map(encode::title);
    if let Some(ld) = &spec.layer {
        let le = encode::encode(table, ld, &keep, &spec.blank_label)?;
        for (code, label) in le.labels.iter().enumerate() {
            let rows_in = used
                .iter()
                .copied()
                .filter(|&i| le.codes[i] as usize == code);
            let lc = tabulate(&re, &ce, rows_in);
            let ln: f64 = lc.iter().map(|r| r.iter().sum::<f64>()).sum();
            let two = is_2x2(&lc);
            layers.push(Layer {
                label: label.clone(),
                n: ln,
                rates: two.then(|| rates(&lc)),
                odds_ratio: if two {
                    odds_ratio(lc[0][0], lc[0][1], lc[1][0], lc[1][1]).0
                } else {
                    None
                },
                stats: stats(&lc),
                counts: lc,
            });
        }
        if is_2x2(&counts) {
            simp = Some(simpson(&counts, &layers, &re.labels, &ce.labels));
        } else {
            warnings.push("the Simpson's-paradox check needs a 2x2 table: use two values for the row and the column field".to_string());
        }
    }

    Ok(Crosstab {
        row_field: encode::title(&spec.row),
        col_field: encode::title(&spec.col),
        layer_field,
        row_labels: re.labels,
        col_labels: ce.labels,
        counts,
        row_totals,
        col_totals,
        n,
        stats: st,
        layers,
        simpson: simp,
        warnings,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f64, b: f64, tol: f64) {
        assert!((a - b).abs() <= tol * b.abs().max(1.0), "{a} vs {b}");
    }

    #[test]
    fn two_by_two_matches_hand_computation() {
        // [[10, 20], [30, 40]]: n=100, expected [[12,18],[28,42]]
        let s = stats(&[vec![10.0, 20.0], vec![30.0, 40.0]]).unwrap();
        close(
            s.chi2,
            4.0 / 12.0 + 4.0 / 18.0 + 4.0 / 28.0 + 4.0 / 42.0,
            1e-12,
        );
        assert_eq!((s.dof, s.rows, s.cols), (1, 2, 2));
        close(s.odds_ratio.unwrap(), (10.0 * 40.0) / (20.0 * 30.0), 1e-12);
        close(s.expected[0][0], 12.0, 1e-12);
        close(s.cramers_v, (s.chi2 / 100.0).sqrt(), 1e-12);
        close(
            s.phi.unwrap(),
            (10.0 * 40.0 - 20.0 * 30.0) / (30.0f64 * 70.0 * 40.0 * 60.0).sqrt(),
            1e-12,
        );
        assert!(s.chi2_yates.unwrap() < s.chi2);
        assert!(s.p_yates.unwrap() > s.p_value);
    }

    #[test]
    fn independent_table_has_zero_statistic() {
        let s = stats(&[vec![10.0, 20.0], vec![20.0, 40.0]]).unwrap();
        close(s.chi2, 0.0, 1e-12);
        close(s.p_value, 1.0, 1e-12);
        close(s.cramers_v, 0.0, 1e-9);
        close(s.odds_ratio.unwrap(), 1.0, 1e-12);
        close(s.fisher_p.unwrap(), 1.0, 1e-9);
    }

    #[test]
    fn fisher_matches_the_lady_tasting_tea() {
        // Fisher (1935): 4 of 4 right. Two-sided p = 2/70.
        close(fisher_two_sided(4.0, 0.0, 0.0, 4.0), 2.0 / 70.0, 1e-12);
        close(fisher_two_sided(3.0, 1.0, 1.0, 3.0), 34.0 / 70.0, 1e-12);
    }

    #[test]
    fn larger_tables_use_the_right_degrees_of_freedom() {
        let s = stats(&[
            vec![10.0, 20.0, 30.0],
            vec![20.0, 20.0, 10.0],
            vec![5.0, 5.0, 5.0],
        ])
        .unwrap();
        assert_eq!(s.dof, 4);
        assert!(s.phi.is_none() && s.fisher_p.is_none() && s.chi2_yates.is_none());
        assert_eq!(s.residuals.len(), 3);
        // adjusted residuals sum to a chi-square when squared and added only approximately, but
        // each row's observed total equals its expected total
        let row0: f64 = s.expected[0].iter().sum();
        close(row0, 60.0, 1e-12);
    }

    #[test]
    fn empty_rows_and_columns_are_ignored_and_degenerate_tables_have_no_stats() {
        let s = stats(&[
            vec![10.0, 0.0, 20.0],
            vec![0.0, 0.0, 0.0],
            vec![30.0, 0.0, 40.0],
        ])
        .unwrap();
        assert_eq!((s.rows, s.cols, s.dof), (2, 2, 1));
        assert!(stats(&[vec![5.0, 5.0]]).is_none());
        assert!(stats(&[vec![5.0, 0.0], vec![5.0, 0.0]]).is_none());
    }

    #[test]
    fn zero_cells_get_a_haldane_correction_for_the_odds_ratio() {
        let (or, ci) = odds_ratio(5.0, 0.0, 1.0, 4.0);
        close(or.unwrap(), (5.5 * 4.5) / (0.5 * 1.5), 1e-12);
        assert!(ci.unwrap()[0] > 0.0);
        assert_eq!(odds_ratio(0.0, 0.0, 0.0, 0.0), (Some(1.0), ci_of_zero()));
    }

    fn ci_of_zero() -> Option<[f64; 2]> {
        odds_ratio(0.0, 0.0, 0.0, 0.0).1
    }

    #[test]
    fn small_expected_counts_are_reported() {
        let s = stats(&[vec![1.0, 2.0], vec![3.0, 1.0]]).unwrap();
        assert!(s.share_expected_below_5 > 0.99 && s.min_expected < 5.0);
    }
}
