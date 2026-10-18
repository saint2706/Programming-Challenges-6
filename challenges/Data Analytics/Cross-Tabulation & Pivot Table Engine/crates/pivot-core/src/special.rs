//! The special functions the cross-tab statistics need, from scratch: log-gamma, the regularized
//! incomplete gamma functions (so the chi-square survival function), the normal CDF, and the
//! hypergeometric probabilities behind Fisher's exact test.

use std::f64::consts::{PI, SQRT_2};

const LANCZOS: [f64; 9] = [
    0.999_999_999_999_809_9,
    676.520_368_121_885_1,
    -1_259.139_216_722_402_8,
    771.323_428_777_653_1,
    -176.615_029_162_140_6,
    12.507_343_278_686_905,
    -0.138_571_095_265_720_12,
    9.984_369_578_019_572e-6,
    1.505_632_735_149_311_6e-7,
];

/// ln Γ(x) for x > 0 (Lanczos approximation, g = 7).
pub fn ln_gamma(x: f64) -> f64 {
    if x < 0.5 {
        // reflection: Γ(x) Γ(1-x) = π / sin(πx)
        return (PI / (PI * x).sin()).ln() - ln_gamma(1.0 - x);
    }
    let x = x - 1.0;
    let mut a = LANCZOS[0];
    let t = x + 7.5;
    for (i, c) in LANCZOS.iter().enumerate().skip(1) {
        a += c / (x + i as f64);
    }
    0.5 * (2.0 * PI).ln() + (x + 0.5) * t.ln() - t + a.ln()
}

/// Regularized lower incomplete gamma P(a, x).
pub fn gamma_p(a: f64, x: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    if x < a + 1.0 {
        series_p(a, x)
    } else {
        1.0 - cf_q(a, x)
    }
}

/// Regularized upper incomplete gamma Q(a, x) = 1 - P(a, x).
pub fn gamma_q(a: f64, x: f64) -> f64 {
    if x <= 0.0 {
        return 1.0;
    }
    if x < a + 1.0 {
        1.0 - series_p(a, x)
    } else {
        cf_q(a, x)
    }
}

fn series_p(a: f64, x: f64) -> f64 {
    let mut term = 1.0 / a;
    let mut sum = term;
    let mut ap = a;
    for _ in 0..1000 {
        ap += 1.0;
        term *= x / ap;
        sum += term;
        if term.abs() < sum.abs() * 1e-16 {
            break;
        }
    }
    sum * (-x + a * x.ln() - ln_gamma(a)).exp()
}

/// Continued fraction for Q(a, x), modified Lentz.
fn cf_q(a: f64, x: f64) -> f64 {
    let tiny = 1e-300;
    let mut b = x + 1.0 - a;
    let mut c = 1.0 / tiny;
    let mut d = 1.0 / b;
    let mut h = d;
    for i in 1..1000 {
        let an = -(i as f64) * (i as f64 - a);
        b += 2.0;
        d = an * d + b;
        if d.abs() < tiny {
            d = tiny;
        }
        c = b + an / c;
        if c.abs() < tiny {
            c = tiny;
        }
        d = 1.0 / d;
        let delta = d * c;
        h *= delta;
        if (delta - 1.0).abs() < 1e-16 {
            break;
        }
    }
    (-x + a * x.ln() - ln_gamma(a)).exp() * h
}

/// P(X ≥ x) for X ~ χ² with `df` degrees of freedom.
pub fn chi2_sf(x: f64, df: f64) -> f64 {
    if df <= 0.0 {
        f64::NAN
    } else {
        gamma_q(df / 2.0, x / 2.0)
    }
}

/// Complementary error function.
pub fn erfc(x: f64) -> f64 {
    if x >= 0.0 {
        gamma_q(0.5, x * x)
    } else {
        1.0 + gamma_p(0.5, x * x)
    }
}

/// The standard normal CDF.
pub fn normal_cdf(x: f64) -> f64 {
    0.5 * erfc(-x / SQRT_2)
}

/// ln of the binomial coefficient C(n, k).
pub fn ln_choose(n: u64, k: u64) -> f64 {
    if k > n {
        return f64::NEG_INFINITY;
    }
    ln_gamma(n as f64 + 1.0) - ln_gamma(k as f64 + 1.0) - ln_gamma((n - k) as f64 + 1.0)
}

/// P(X = k) when drawing `draws` from `total` items of which `good` are marked.
pub fn hypergeom_pmf(k: u64, total: u64, good: u64, draws: u64) -> f64 {
    if k > good || k > draws || draws - k > total - good {
        return 0.0;
    }
    (ln_choose(good, k) + ln_choose(total - good, draws - k) - ln_choose(total, draws)).exp()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f64, b: f64, tol: f64) {
        assert!((a - b).abs() <= tol * b.abs().max(1.0), "{a} vs {b}");
    }

    #[test]
    fn log_gamma_known_values() {
        close(ln_gamma(1.0), 0.0, 1e-13);
        close(ln_gamma(5.0), 24f64.ln(), 1e-13);
        close(ln_gamma(0.5), PI.sqrt().ln(), 1e-13);
        close(ln_gamma(10.5), 13.940_625_219_403_763, 1e-12);
        close(ln_gamma(100.0), 359.134_205_369_575_4, 1e-12);
        close(ln_gamma(0.1), 2.252_712_651_734_206, 1e-12);
    }

    #[test]
    fn chi_square_critical_values() {
        close(chi2_sf(3.841_458_820_694_124, 1.0), 0.05, 1e-10);
        close(chi2_sf(5.991_464_547_107_979, 2.0), 0.05, 1e-10);
        close(chi2_sf(9.487_729_036_781_154, 4.0), 0.05, 1e-10);
        close(chi2_sf(18.307_038_053_275_146, 10.0), 0.05, 1e-10);
        close(chi2_sf(6.634_896_601_021_213, 1.0), 0.01, 1e-10);
        assert_eq!(chi2_sf(0.0, 3.0), 1.0);
        assert!(chi2_sf(1000.0, 3.0) < 1e-200);
        assert!(chi2_sf(1.0, 0.0).is_nan());
    }

    #[test]
    fn incomplete_gamma_identities() {
        for &(a, x) in &[
            (0.5, 0.2),
            (2.0, 1.5),
            (7.0, 3.0),
            (7.0, 20.0),
            (30.0, 25.0),
        ] {
            close(gamma_p(a, x) + gamma_q(a, x), 1.0, 1e-13);
        }
        close(gamma_p(1.0, 2.0), 1.0 - (-2f64).exp(), 1e-13); // exponential CDF
    }

    #[test]
    fn normal_cdf_and_erfc() {
        close(normal_cdf(0.0), 0.5, 1e-15);
        close(normal_cdf(1.959_963_984_540_054), 0.975, 1e-12);
        close(normal_cdf(-1.644_853_626_951_472_7), 0.05, 1e-12);
        close(erfc(0.0), 1.0, 1e-15);
        close(erfc(1.0), 0.157_299_207_050_285_13, 1e-12);
        close(erfc(-1.0), 1.842_700_792_949_715, 1e-12);
    }

    #[test]
    fn hypergeometric_sums_to_one() {
        let total: f64 = (0..=10).map(|k| hypergeom_pmf(k, 30, 12, 10)).sum();
        close(total, 1.0, 1e-12);
        close(hypergeom_pmf(2, 5, 3, 2), 0.3, 1e-13); // C(3,2)*C(2,0)/C(5,2)
        assert_eq!(hypergeom_pmf(5, 10, 3, 4), 0.0);
        assert_eq!(ln_choose(3, 5), f64::NEG_INFINITY);
    }
}
