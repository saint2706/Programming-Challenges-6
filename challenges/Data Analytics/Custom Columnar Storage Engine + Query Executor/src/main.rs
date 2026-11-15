//! `colq`: import CSV into `.col` files, inspect them, and run SQL over them.

use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::Instant;

use colstore::format::{self, Table};
use colstore::{Catalog, QueryResult, csv, encoding};

const USAGE: &str = "usage:
  colq import <file.csv> <out.col> [--delim C] [--row-group N]
  colq info <file.col>
  colq query [--table name=file.col ...] [--timing] \"SQL\"
      (without --table, every *.col in the current folder is a table named after its file)";

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match run(&args) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}

fn flag<'a>(args: &'a [String], name: &str) -> Option<&'a str> {
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1))
        .map(String::as_str)
}

fn run(args: &[String]) -> colstore::Result<()> {
    match args.first().map(String::as_str) {
        Some("import") if args.len() >= 3 => {
            let delim = flag(args, "--delim")
                .and_then(|d| d.chars().next())
                .unwrap_or(',');
            let rg = flag(args, "--row-group")
                .map_or(Ok(format::DEFAULT_ROW_GROUP), str::parse)
                .map_err(|_| "bad --row-group")?;
            let text =
                std::fs::read_to_string(&args[1]).map_err(|e| format!("{}: {e}", args[1]))?;
            let t0 = Instant::now();
            let (rows, footer) = csv::import(&text, delim, Path::new(&args[2]), rg)?;
            let size = std::fs::metadata(&args[2])
                .map_err(|e| e.to_string())?
                .len();
            println!(
                "{} rows, {} columns, {} row groups, {size} bytes ({} bytes CSV) in {:.2}s",
                rows,
                footer.names.len(),
                footer.row_groups.len(),
                text.len(),
                t0.elapsed().as_secs_f64()
            );
            Ok(())
        }
        Some("info") if args.len() == 2 => {
            let t = Table::open(Path::new(&args[1]))?;
            println!(
                "{} rows in {} row groups",
                t.rows(),
                t.footer.row_groups.len()
            );
            for (c, (name, ty)) in t.footer.names.iter().zip(&t.footer.types).enumerate() {
                let bytes: u64 = t.footer.row_groups.iter().map(|g| g.chunks[c].len).sum();
                let mut encs: Vec<&str> = t
                    .footer
                    .row_groups
                    .iter()
                    .map(|g| encoding::NAMES[g.chunks[c].encoding as usize])
                    .collect();
                encs.sort_unstable();
                encs.dedup();
                println!(
                    "  {name:<20} {ty:<6?} {bytes:>12} bytes  {}",
                    encs.join(",")
                );
            }
            Ok(())
        }
        Some("query") if args.len() >= 2 => {
            let mut cat = Catalog::new();
            let mut i = 1;
            let mut timing = false;
            let mut sql = None;
            while i < args.len() {
                match args[i].as_str() {
                    "--table" => {
                        let spec = args.get(i + 1).ok_or("--table needs name=path")?;
                        let (n, p) = spec.split_once('=').ok_or("--table needs name=path")?;
                        cat.insert(n.to_lowercase(), PathBuf::from(p));
                        i += 2;
                    }
                    "--timing" => {
                        timing = true;
                        i += 1;
                    }
                    s => {
                        sql = Some(s.to_owned());
                        i += 1;
                    }
                }
            }
            if cat.is_empty() {
                for e in std::fs::read_dir(".").map_err(|e| e.to_string())?.flatten() {
                    let p = e.path();
                    if p.extension().is_some_and(|x| x == "col") {
                        let stem = p.file_stem().unwrap().to_string_lossy().to_lowercase();
                        cat.insert(stem, p);
                    }
                }
            }
            let t0 = Instant::now();
            match colstore::query(&cat, &sql.ok_or("no SQL given")?)? {
                QueryResult::Explain(s) => println!("{s}"),
                QueryResult::Rows(b, stats) => {
                    println!("{}", b.names.join("\t"));
                    for r in 0..b.rows() {
                        let row: Vec<String> =
                            b.columns.iter().map(|c| c.get(r).to_string()).collect();
                        println!("{}", row.join("\t"));
                    }
                    if timing {
                        eprintln!(
                            "{} rows in {:.3}s; scanned {} rows, read {}/{} row groups",
                            b.rows(),
                            t0.elapsed().as_secs_f64(),
                            stats.rows_scanned,
                            stats.row_groups_read,
                            stats.row_groups_total
                        );
                    }
                }
            }
            Ok(())
        }
        _ => Err(USAGE.into()),
    }
}
