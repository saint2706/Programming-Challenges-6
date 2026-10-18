//! A small RFC 4180 CSV reader. Cells go straight into one text arena per column, so a 1M-row file
//! costs about its own size in memory and typing a column later is a scan over a slice.

use crate::error::{Error, Result};

/// All cells of one column, back to back. Cell `i` is `text[ends[i-1]..ends[i]]`.
#[derive(Debug, Default, Clone)]
pub struct Arena {
    text: String,
    ends: Vec<u32>,
}

impl Arena {
    fn push(&mut self, cell: &str) {
        self.text.push_str(cell);
        self.ends.push(self.text.len() as u32);
    }

    pub fn len(&self) -> usize {
        self.ends.len()
    }

    pub fn is_empty(&self) -> bool {
        self.ends.is_empty()
    }

    pub fn get(&self, i: usize) -> &str {
        let start = if i == 0 { 0 } else { self.ends[i - 1] as usize };
        &self.text[start..self.ends[i] as usize]
    }
}

#[derive(Debug)]
pub struct Parsed {
    pub header: Vec<String>,
    pub columns: Vec<Arena>,
    pub rows: usize,
}

/// Comma, semicolon, tab or pipe: whichever appears most often in the first line (outside quotes).
pub fn detect_delimiter(text: &str) -> u8 {
    let first = text.lines().next().unwrap_or("");
    let mut in_quotes = false;
    let mut counts = [0usize; 4];
    for b in first.bytes() {
        match b {
            b'"' => in_quotes = !in_quotes,
            b',' if !in_quotes => counts[0] += 1,
            b';' if !in_quotes => counts[1] += 1,
            b'\t' if !in_quotes => counts[2] += 1,
            b'|' if !in_quotes => counts[3] += 1,
            _ => {}
        }
    }
    let best = (0..4).max_by_key(|&i| (counts[i], 4 - i)).unwrap_or(0);
    b",;\t|"[best]
}

/// Parse `bytes` as CSV. `trim` strips spaces around every cell.
pub fn parse(bytes: &[u8], delimiter: Option<u8>, trim: bool) -> Result<Parsed> {
    let text = std::str::from_utf8(bytes).map_err(|e| {
        Error::new(format!(
            "the file is not valid UTF-8 (byte {})",
            e.valid_up_to()
        ))
    })?;
    let text = text.strip_prefix('\u{feff}').unwrap_or(text);
    let delim = delimiter.unwrap_or_else(|| detect_delimiter(text));
    let mut records = Records::new(text, delim, trim);

    let header_raw = loop {
        match records.next_record()? {
            Some(r) if r.iter().all(|c| c.is_empty()) => continue,
            Some(r) => break r,
            None => return Err(Error::new("the file has no header row")),
        }
    };
    let header = clean_header(&header_raw);
    let width = header.len();
    let mut columns: Vec<Arena> = (0..width).map(|_| Arena::default()).collect();
    let rows = records.read_body(&mut columns)?;
    Ok(Parsed {
        header,
        columns,
        rows,
    })
}

fn clean_header(raw: &[String]) -> Vec<String> {
    let mut out: Vec<String> = Vec::with_capacity(raw.len());
    for (i, name) in raw.iter().enumerate() {
        let base = if name.is_empty() {
            format!("column_{}", i + 1)
        } else {
            name.clone()
        };
        let mut candidate = base.clone();
        let mut n = 2;
        while out.contains(&candidate) {
            candidate = format!("{base}_{n}");
            n += 1;
        }
        out.push(candidate);
    }
    out
}

struct Records<'a> {
    text: &'a str,
    bytes: &'a [u8],
    pos: usize,
    delim: u8,
    trim: bool,
    line: usize,
    /// The line the current record began on, for error messages.
    start_line: usize,
}

impl<'a> Records<'a> {
    fn new(text: &'a str, delim: u8, trim: bool) -> Self {
        Records {
            text,
            bytes: text.as_bytes(),
            pos: 0,
            delim,
            trim,
            line: 1,
            start_line: 1,
        }
    }

    /// Every remaining record, straight into the column arenas. An unquoted field (nearly all of
    /// them) is a slice of the input and costs no allocation; a quoted one takes the slower path
    /// through `next_cell`. Returns the number of data rows.
    fn read_body(&mut self, columns: &mut [Arena]) -> Result<usize> {
        let width = columns.len();
        let b = self.bytes;
        let mut rows = 0;
        while self.pos < b.len() {
            self.start_line = self.line;
            let mut field = 0;
            let mut blank = false;
            loop {
                let quoted_ahead = {
                    let mut p = self.pos;
                    if self.trim {
                        while p < b.len() && b[p] == b' ' {
                            p += 1;
                        }
                    }
                    p < b.len() && b[p] == b'"'
                };
                let ended;
                if quoted_ahead {
                    let (cell, done) = self.next_cell()?;
                    if field < width {
                        columns[field].push(&cell);
                    }
                    ended = done;
                } else {
                    let start = self.pos;
                    let mut q = start;
                    while q < b.len() && b[q] != self.delim && b[q] != b'\n' && b[q] != b'\r' {
                        q += 1;
                    }
                    let raw = &self.text[start..q];
                    let cell = if self.trim { raw.trim() } else { raw };
                    ended = if q >= b.len() {
                        self.pos = q;
                        true
                    } else if b[q] == self.delim {
                        self.pos = q + 1;
                        false
                    } else {
                        self.pos = q + 1;
                        if b[q] == b'\r' && self.pos < b.len() && b[self.pos] == b'\n' {
                            self.pos += 1;
                        }
                        self.line += 1;
                        true
                    };
                    if field == 0 && ended && cell.is_empty() {
                        blank = true; // an empty line, not a row of one empty value
                    } else if field < width {
                        columns[field].push(cell);
                    }
                }
                field += 1;
                if ended {
                    break;
                }
            }
            if blank {
                continue;
            }
            if field != width {
                return Err(Error::new(format!(
                    "line {}: {} fields, but the header has {}",
                    self.start_line, field, width
                )));
            }
            rows += 1;
        }
        Ok(rows)
    }

    /// The next record, or `None` at the end of the input.
    fn next_record(&mut self) -> Result<Option<Vec<String>>> {
        if self.pos >= self.bytes.len() {
            return Ok(None);
        }
        self.start_line = self.line;
        let mut record = Vec::new();
        loop {
            let (cell, ended) = self.next_cell()?;
            record.push(cell);
            if ended {
                break;
            }
        }
        Ok(Some(record))
    }

    /// One cell, and whether the record ended after it.
    fn next_cell(&mut self) -> Result<(String, bool)> {
        let b = self.bytes;
        let mut cell: Vec<u8> = Vec::new();
        let mut quoted = false;
        // a leading space before an opening quote is allowed when trimming
        let mut p = self.pos;
        if self.trim {
            while p < b.len() && b[p] == b' ' {
                p += 1;
            }
        }
        if p < b.len() && b[p] == b'"' {
            quoted = true;
            p += 1;
            loop {
                if p >= b.len() {
                    return Err(Error::new(format!(
                        "line {}: a quoted field is never closed",
                        self.line
                    )));
                }
                match b[p] {
                    b'"' if p + 1 < b.len() && b[p + 1] == b'"' => {
                        cell.push(b'"');
                        p += 2;
                    }
                    b'"' => {
                        p += 1;
                        break;
                    }
                    b'\n' => {
                        self.line += 1;
                        cell.push(b'\n');
                        p += 1;
                    }
                    c => {
                        cell.push(c);
                        p += 1;
                    }
                }
            }
            // anything between the closing quote and the delimiter is dropped if blank
            while p < b.len() && b[p] == b' ' && self.trim {
                p += 1;
            }
        } else {
            p = self.pos;
            while p < b.len() && b[p] != self.delim && b[p] != b'\n' && b[p] != b'\r' {
                cell.push(b[p]);
                p += 1;
            }
        }
        let ended = if p >= b.len() {
            true
        } else if b[p] == self.delim {
            p += 1;
            false
        } else if b[p] == b'\r' {
            p += 1;
            if p < b.len() && b[p] == b'\n' {
                p += 1;
            }
            self.line += 1;
            true
        } else if b[p] == b'\n' {
            p += 1;
            self.line += 1;
            true
        } else {
            return Err(Error::new(format!(
                "line {}: text after a closing quote",
                self.line
            )));
        };
        self.pos = p;
        let text = String::from_utf8(cell).map_err(|_| Error::new("invalid UTF-8 in a cell"))?;
        let text = if self.trim && !quoted {
            text.trim().to_string()
        } else {
            text
        };
        Ok((text, ended))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rows(p: &Parsed) -> Vec<Vec<String>> {
        (0..p.rows)
            .map(|r| p.columns.iter().map(|c| c.get(r).to_string()).collect())
            .collect()
    }

    #[test]
    fn plain_quoted_and_multiline() {
        let p = parse(
            b"a,b\n1,\"x,y\"\n2,\"line1\nline2\"\n3,\"say \"\"hi\"\"\"\n",
            None,
            true,
        )
        .unwrap();
        assert_eq!(p.header, ["a", "b"]);
        assert_eq!(
            rows(&p),
            [["1", "x,y"], ["2", "line1\nline2"], ["3", "say \"hi\""]]
        );
    }

    #[test]
    fn crlf_cr_bom_and_blank_lines() {
        let p = parse("\u{feff}a,b\r\n1,2\r\n\r\n3,4\r\n".as_bytes(), None, true).unwrap();
        assert_eq!(p.header, ["a", "b"]);
        assert_eq!(rows(&p), [["1", "2"], ["3", "4"]]);
        let p = parse(b"a,b\r1,2\r3,4", None, true).unwrap();
        assert_eq!(p.rows, 2);
    }

    #[test]
    fn trims_unquoted_cells_but_keeps_quoted_spaces() {
        let p = parse(b"a, b\n 1 , \" x \" \n", None, true).unwrap();
        assert_eq!(p.header, ["a", "b"]);
        assert_eq!(rows(&p), [["1", " x "]]);
        let p = parse(b"a,b\n 1 ,  y\n", None, false).unwrap();
        assert_eq!(rows(&p), [[" 1 ", "  y"]]);
    }

    #[test]
    fn delimiter_detection() {
        assert_eq!(detect_delimiter("a;b;c"), b';');
        assert_eq!(detect_delimiter("a\tb\tc"), b'\t');
        assert_eq!(detect_delimiter("\"a,b\";c"), b';');
        assert_eq!(detect_delimiter("single"), b',');
        let p = parse(b"a;b\n1;2\n", None, true).unwrap();
        assert_eq!(rows(&p), [["1", "2"]]);
    }

    #[test]
    fn header_cleanup() {
        let p = parse(b"a,,a,a\n1,2,3,4\n", None, true).unwrap();
        assert_eq!(p.header, ["a", "column_2", "a_2", "a_3"]);
    }

    #[test]
    fn errors_name_the_line() {
        let e = parse(b"a,b\n1,2\n3\n", None, true).unwrap_err();
        assert!(e.to_string().contains("line 3") && e.to_string().contains("1 fields"));
        assert!(
            parse(b"a,b\n1,\"open\n", None, true)
                .unwrap_err()
                .to_string()
                .contains("never closed")
        );
        assert!(
            parse(b"a,b\n1,\"x\"y\n", None, true)
                .unwrap_err()
                .to_string()
                .contains("after a closing quote")
        );
        assert!(parse(b"", None, true).is_err());
        assert!(
            parse(&[0xff, 0xfe], None, true)
                .unwrap_err()
                .to_string()
                .contains("UTF-8")
        );
    }

    #[test]
    fn empty_cells_and_a_final_row_without_newline() {
        let p = parse(b"a,b,c\n1,,3\n,,\n4,5,6", None, true).unwrap();
        assert_eq!(rows(&p), [["1", "", "3"], ["", "", ""], ["4", "5", "6"]]);
    }

    #[test]
    fn header_only_file_has_no_rows() {
        let p = parse(b"a,b\n", None, true).unwrap();
        assert_eq!((p.header.len(), p.rows), (2, 0));
    }
}
