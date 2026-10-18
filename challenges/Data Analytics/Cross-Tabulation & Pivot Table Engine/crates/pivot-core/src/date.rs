//! Calendar arithmetic without a date library: days since 1970-01-01 and back (Howard Hinnant's
//! `days_from_civil` / `civil_from_days`), ISO date and datetime parsing, and the parts a pivot
//! groups by.

pub fn is_leap(year: i64) -> bool {
    (year % 4 == 0 && year % 100 != 0) || year % 400 == 0
}

pub fn days_in_month(year: i64, month: u32) -> u32 {
    match month {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        _ if is_leap(year) => 29,
        _ => 28,
    }
}

/// Days since 1970-01-01 for a proleptic Gregorian date.
pub fn days_from_civil(y: i64, m: u32, d: u32) -> i64 {
    let y = if m <= 2 { y - 1 } else { y };
    let era = y.div_euclid(400);
    let yoe = y - era * 400;
    let mp = (m as i64 + 9) % 12;
    let doy = (153 * mp + 2) / 5 + d as i64 - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146_097 + doe - 719_468
}

/// The inverse of [`days_from_civil`]: `(year, month 1-12, day 1-31)`.
pub fn civil_from_days(z: i64) -> (i64, u32, u32) {
    let z = z + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    (if m <= 2 { y + 1 } else { y }, m, d)
}

/// 0 = Monday ... 6 = Sunday.
pub fn weekday(days: i64) -> u32 {
    (days + 3).rem_euclid(7) as u32
}

pub fn quarter(month: u32) -> u32 {
    (month - 1) / 3 + 1
}

fn digits(s: &[u8]) -> Option<u32> {
    if s.is_empty() || !s.iter().all(u8::is_ascii_digit) {
        return None;
    }
    s.iter().try_fold(0u32, |acc, &b| {
        acc.checked_mul(10)?.checked_add((b - b'0') as u32)
    })
}

/// `YYYY-MM-DD` to days since the epoch; `None` if it is not a real calendar date.
pub fn parse_date(s: &str) -> Option<i64> {
    let b = s.as_bytes();
    if b.len() != 10 || b[4] != b'-' || b[7] != b'-' {
        return None;
    }
    let (y, m, d) = (digits(&b[0..4])?, digits(&b[5..7])?, digits(&b[8..10])?);
    if !(1..=12).contains(&m) || d == 0 || d > days_in_month(y as i64, m) {
        return None;
    }
    Some(days_from_civil(y as i64, m, d))
}

/// `YYYY-MM-DD[ T]HH:MM[:SS[.fraction]][Z]` to seconds since the epoch (UTC, fraction dropped).
pub fn parse_datetime(s: &str) -> Option<i64> {
    let s = s.strip_suffix('Z').unwrap_or(s);
    if s.len() < 16 || !s.is_char_boundary(10) {
        return None;
    }
    let days = parse_date(&s[..10])?;
    let rest = s.as_bytes();
    if rest[10] != b' ' && rest[10] != b'T' {
        return None;
    }
    let time = &s[11..];
    let t = time.as_bytes();
    if t.len() < 5 || t[2] != b':' {
        return None;
    }
    let (h, mi) = (digits(&t[0..2])?, digits(&t[3..5])?);
    let mut sec = 0;
    if t.len() > 5 {
        if t[5] != b':' || t.len() < 8 {
            return None;
        }
        sec = digits(&t[6..8])?;
        if t.len() > 8 && (t[8] != b'.' || digits(&t[9..]).is_none()) {
            return None;
        }
    }
    if h > 23 || mi > 59 || sec > 59 {
        return None;
    }
    Some(days * 86_400 + (h * 3600 + mi * 60 + sec) as i64)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn epoch_and_known_dates() {
        assert_eq!(days_from_civil(1970, 1, 1), 0);
        assert_eq!(days_from_civil(2000, 3, 1), 11_017);
        assert_eq!(days_from_civil(1969, 12, 31), -1);
        assert_eq!(days_from_civil(2024, 2, 29), 19_782);
        assert_eq!(civil_from_days(19_782), (2024, 2, 29));
    }

    #[test]
    fn roundtrip_over_six_centuries() {
        for z in (-200_000..200_000).step_by(37) {
            let (y, m, d) = civil_from_days(z);
            assert_eq!(days_from_civil(y, m, d), z);
            assert!((1..=12).contains(&m) && d >= 1 && d <= days_in_month(y, m));
        }
    }

    #[test]
    fn weekdays() {
        assert_eq!(weekday(0), 3); // Thursday
        assert_eq!(weekday(parse_date("2024-02-29").unwrap()), 3);
        assert_eq!(weekday(parse_date("2019-10-06").unwrap()), 6); // a Sunday
        assert_eq!(weekday(-1), 2);
    }

    #[test]
    fn leap_years_and_month_lengths() {
        assert!(is_leap(2000) && is_leap(2024) && !is_leap(1900) && !is_leap(2023));
        assert_eq!(days_in_month(2024, 2), 29);
        assert_eq!(days_in_month(2023, 2), 28);
        assert_eq!(days_in_month(2023, 4), 30);
    }

    #[test]
    fn date_parsing_rejects_impossible_dates() {
        assert_eq!(parse_date("1970-01-02"), Some(1));
        for bad in [
            "2023-02-29",
            "2023-13-01",
            "2023-00-10",
            "2023-04-31",
            "23-01-01",
            "2023/01/01",
            "2023-1-01",
            "abcd-ef-gh",
            "",
        ] {
            assert_eq!(parse_date(bad), None, "{bad}");
        }
        assert!(parse_date("2024-02-29").is_some());
    }

    #[test]
    fn datetime_parsing() {
        assert_eq!(parse_datetime("1970-01-01 00:00"), Some(0));
        assert_eq!(parse_datetime("1970-01-01T00:01:01Z"), Some(61));
        assert_eq!(
            parse_datetime("1970-01-02 01:00:00.250"),
            Some(86_400 + 3600)
        );
        for bad in [
            "1970-01-01",
            "1970-01-01 25:00",
            "1970-01-01 10:61",
            "1970-01-01 10:00:61",
            "1970-01-01x10:00",
            "1970-01-01 1:00",
            "1970-01-01 10:00:0",
            "1970-01-01 10:00:00.x",
        ] {
            assert_eq!(parse_datetime(bad), None, "{bad}");
        }
    }

    #[test]
    fn quarters() {
        assert_eq!(
            [1, 3, 4, 6, 7, 9, 10, 12].map(quarter),
            [1, 1, 2, 2, 3, 3, 4, 4]
        );
    }
}
