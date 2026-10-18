//! The pivot engine behind a C ABI, with JSON in and JSON out.
//!
//! One calling convention serves every host: Python loads the shared library through `ctypes`, and
//! the browser instantiates the same code as a WebAssembly module. A host copies the operation
//! name, a JSON request and (for `load_csv`) the file bytes into memory obtained from [`pt_alloc`],
//! calls [`pt_call`], reads the JSON answer, and releases every buffer with [`pt_free`].
//!
//! Every answer is `{"ok": ...}` or `{"error": "message"}`; nothing here panics across the boundary.
//!
//! | operation  | request                                                  | answer                      |
//! | ---------- | -------------------------------------------------------- | --------------------------- |
//! | `load_csv` | `{"delimiter": ",", "trim": true}` + the CSV as data     | `{table, rows, fields}`     |
//! | `describe` | `{"table": id}`                                          | `{rows, fields}`            |
//! | `pivot`    | `{"table": id, "spec": PivotSpec}`                       | a `PivotResult`             |
//! | `crosstab` | `{"table": id, "spec": CrosstabSpec}`                    | a `Crosstab`                |
//! | `drill`    | `{"table", "spec", "row_path", "col_path", "limit"}`     | a `Drill`                   |
//! | `values`   | `{"table": id, "field": name, "limit": 500}`             | `{total, values}`           |
//! | `drop`     | `{"table": id}`                                          | `{}`                        |
//! | `version`  | `{}`                                                     | `{engine, version}`         |

use std::collections::HashMap;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::{Arc, LazyLock, Mutex};

use pivot_core::crosstab::{CrosstabSpec, crosstab};
use pivot_core::drill::drill;
use pivot_core::encode;
use pivot_core::spec::{Dim, PivotSpec, Sort};
use pivot_core::table::{LoadOptions, Table};
use serde::Deserialize;
use serde_json::{Value, json};

#[derive(Default)]
struct State {
    next: u32,
    tables: HashMap<u32, Arc<Table>>,
}

static STATE: LazyLock<Mutex<State>> = LazyLock::new(|| {
    Mutex::new(State {
        next: 1,
        tables: HashMap::new(),
    })
});

#[derive(Deserialize)]
struct TableRef {
    table: u32,
}

#[derive(Deserialize, Default)]
struct LoadReq {
    #[serde(default)]
    delimiter: Option<String>,
    #[serde(default = "yes")]
    trim: bool,
}

fn yes() -> bool {
    true
}

#[derive(Deserialize)]
struct PivotReq {
    table: u32,
    spec: PivotSpec,
}

#[derive(Deserialize)]
struct CrosstabReq {
    table: u32,
    spec: CrosstabSpec,
}

#[derive(Deserialize)]
struct DrillReq {
    table: u32,
    spec: PivotSpec,
    #[serde(default)]
    row_path: Vec<String>,
    #[serde(default)]
    col_path: Vec<String>,
    #[serde(default = "default_limit")]
    limit: usize,
}

fn default_limit() -> usize {
    100
}

#[derive(Deserialize)]
struct ValuesReq {
    table: u32,
    field: String,
    #[serde(default = "default_values")]
    limit: usize,
}

fn default_values() -> usize {
    500
}

fn get(table: u32) -> Result<Arc<Table>, String> {
    let state = STATE.lock().unwrap_or_else(|e| e.into_inner());
    state
        .tables
        .get(&table)
        .cloned()
        .ok_or_else(|| format!("no table with id {table}: load a file first (or it was dropped)"))
}

fn parse<T: for<'de> Deserialize<'de>>(json: &[u8]) -> Result<T, String> {
    serde_json::from_slice(json).map_err(|e| format!("bad request: {e}"))
}

fn describe(table: &Table) -> Value {
    json!({"rows": table.rows, "fields": table.describe()})
}

fn run(op: &str, json: &[u8], data: &[u8]) -> Result<Value, String> {
    match op {
        "version" => Ok(json!({"engine": "pivot-core", "version": env!("CARGO_PKG_VERSION")})),
        "load_csv" => {
            let req: LoadReq = if json.is_empty() {
                LoadReq::default()
            } else {
                parse(json)?
            };
            let delimiter = match req.delimiter.as_deref() {
                None | Some("") | Some("auto") => None,
                Some(d) if d.len() == 1 && d.is_ascii() => Some(d.as_bytes()[0]),
                Some("\\t") | Some("tab") => Some(b'\t'),
                Some(other) => {
                    return Err(format!(
                        "the delimiter must be one character, got {other:?}"
                    ));
                }
            };
            let table = Table::from_csv(
                data,
                LoadOptions {
                    delimiter,
                    trim: req.trim,
                },
            )
            .map_err(|e| e.to_string())?;
            let mut out = describe(&table);
            let mut state = STATE.lock().unwrap_or_else(|e| e.into_inner());
            let id = state.next;
            state.next += 1;
            state.tables.insert(id, Arc::new(table));
            out["table"] = json!(id);
            Ok(out)
        }
        "describe" => {
            let table = get(parse::<TableRef>(json)?.table)?;
            Ok(describe(&table))
        }
        "pivot" => {
            let req: PivotReq = parse(json)?;
            let table = get(req.table)?;
            let result = pivot_core::pivot::pivot(&table, &req.spec).map_err(|e| e.to_string())?;
            serde_json::to_value(result).map_err(|e| e.to_string())
        }
        "crosstab" => {
            let req: CrosstabReq = parse(json)?;
            let table = get(req.table)?;
            let result = crosstab(&table, &req.spec).map_err(|e| e.to_string())?;
            serde_json::to_value(result).map_err(|e| e.to_string())
        }
        "drill" => {
            let req: DrillReq = parse(json)?;
            let table = get(req.table)?;
            let result = drill(&table, &req.spec, &req.row_path, &req.col_path, req.limit)
                .map_err(|e| e.to_string())?;
            serde_json::to_value(result).map_err(|e| e.to_string())
        }
        "values" => {
            let req: ValuesReq = parse(json)?;
            let table = get(req.table)?;
            let dim = Dim {
                field: req.field,
                group: None,
                sort: Sort::default(),
                top: None,
                hide_blank: false,
            };
            let keep = vec![true; table.rows];
            let enc = encode::encode(&table, &dim, &keep, "(blank)").map_err(|e| e.to_string())?;
            let total = enc.labels.len();
            Ok(
                json!({"total": total, "values": enc.labels.into_iter().take(req.limit).collect::<Vec<_>>()}),
            )
        }
        "drop" => {
            let id = parse::<TableRef>(json)?.table;
            STATE
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .tables
                .remove(&id);
            Ok(json!({}))
        }
        other => Err(format!("unknown operation '{other}'")),
    }
}

/// The whole protocol as a plain Rust function: what `pt_call` does, minus the pointers.
pub fn handle(op: &str, json: &[u8], data: &[u8]) -> Vec<u8> {
    let outcome = catch_unwind(AssertUnwindSafe(|| run(op, json, data)))
        .unwrap_or_else(|_| Err("internal error in the engine".to_string()));
    let answer = match outcome {
        Ok(v) => json!({"ok": v}),
        Err(message) => json!({"error": message}),
    };
    serde_json::to_vec(&answer)
        .unwrap_or_else(|_| br#"{"error":"could not encode the answer"}"#.to_vec())
}

/// A buffer for the host to fill. Release it with [`pt_free`] using the same length.
#[unsafe(no_mangle)]
pub extern "C" fn pt_alloc(len: usize) -> *mut u8 {
    Box::into_raw(vec![0u8; len].into_boxed_slice()) as *mut u8
}

/// Release a buffer from [`pt_alloc`] or the answer of [`pt_call`].
///
/// # Safety
/// `ptr` and `len` must be exactly what an earlier `pt_alloc` or `pt_call` handed out, and the
/// buffer must not be used again.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pt_free(ptr: *mut u8, len: usize) {
    if !ptr.is_null() {
        // SAFETY: the caller promises ptr/len came from a boxed slice of exactly this length.
        drop(unsafe { Box::from_raw(std::ptr::slice_from_raw_parts_mut(ptr, len)) });
    }
}

unsafe fn bytes<'a>(ptr: *const u8, len: usize) -> &'a [u8] {
    if ptr.is_null() || len == 0 {
        &[]
    } else {
        // SAFETY: the caller promises ptr points at len readable bytes.
        unsafe { std::slice::from_raw_parts(ptr, len) }
    }
}

/// Run one operation. Returns a buffer holding the JSON answer and stores its length in `out_len`.
///
/// # Safety
/// Every pointer must reference that many readable bytes (a null pointer with length 0 is an empty
/// input) and `out_len` must be writable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pt_call(
    op: *const u8,
    op_len: usize,
    json: *const u8,
    json_len: usize,
    data: *const u8,
    data_len: usize,
    out_len: *mut usize,
) -> *mut u8 {
    // SAFETY: forwarded from this function's contract.
    let (op, json, data) = unsafe {
        (
            bytes(op, op_len),
            bytes(json, json_len),
            bytes(data, data_len),
        )
    };
    let op = String::from_utf8_lossy(op);
    let answer = handle(&op, json, data).into_boxed_slice();
    if !out_len.is_null() {
        // SAFETY: out_len is writable by contract.
        unsafe { *out_len = answer.len() };
    }
    Box::into_raw(answer) as *mut u8
}

#[cfg(test)]
mod tests {
    use super::*;

    fn call(op: &str, json: &Value, data: &[u8]) -> Value {
        serde_json::from_slice(&handle(op, json.to_string().as_bytes(), data)).unwrap()
    }

    fn load(csv: &str) -> u32 {
        let a = call("load_csv", &json!({}), csv.as_bytes());
        a["ok"]["table"].as_u64().unwrap() as u32
    }

    const CSV: &str = "g,h,x\na,u,1\na,v,2\nb,u,3\nb,v,4\nb,u,\n";

    #[test]
    fn load_describe_pivot_drill_values_drop() {
        let a = call("load_csv", &json!({}), CSV.as_bytes());
        assert_eq!(a["ok"]["rows"], 5);
        assert_eq!(a["ok"]["fields"][2]["kind"], "int");
        let id = a["ok"]["table"].as_u64().unwrap();
        assert_eq!(
            call("describe", &json!({"table": id}), b"")["ok"]["fields"][0]["distinct"],
            2
        );

        let spec =
            json!({"rows": [{"field": "g"}], "cols": [{"field": "h"}], "values": [{"field": "x"}]});
        let p = call("pivot", &json!({"table": id, "spec": spec}), b"");
        assert_eq!(p["ok"]["cells"][1][0][0], 3.0);
        assert_eq!(p["ok"]["rows"][2]["kind"], "total");

        let d = call(
            "drill",
            &json!({"table": id, "spec": spec, "row_path": ["b"], "col_path": ["u"], "limit": 1}),
            b"",
        );
        assert_eq!(d["ok"]["total"], 2);
        assert_eq!(d["ok"]["rows"].as_array().unwrap().len(), 1);
        assert_eq!(d["ok"]["rows"][0], json!(["b", "u", "3"]));

        let v = call("values", &json!({"table": id, "field": "g"}), b"");
        assert_eq!(v["ok"], json!({"total": 2, "values": ["a", "b"]}));

        assert_eq!(call("drop", &json!({"table": id}), b"")["ok"], json!({}));
        assert!(
            call("pivot", &json!({"table": id, "spec": spec}), b"")["error"]
                .as_str()
                .unwrap()
                .contains("no table with id")
        );
    }

    #[test]
    fn crosstab_over_the_boundary() {
        let id = load("a,b\nx,p\nx,p\nx,q\ny,q\ny,q\ny,p\n");
        let spec = json!({"row": {"field": "a"}, "col": {"field": "b"}});
        let c = call("crosstab", &json!({"table": id, "spec": spec}), b"");
        assert_eq!(c["ok"]["counts"], json!([[2.0, 1.0], [1.0, 2.0]]));
        assert!(c["ok"]["stats"]["chi2"].as_f64().unwrap() > 0.0);
    }

    #[test]
    fn errors_are_answers_not_panics() {
        assert!(
            call("nope", &json!({}), b"")["error"]
                .as_str()
                .unwrap()
                .contains("unknown operation")
        );
        assert!(
            call("pivot", &json!({"table": 1}), b"")["error"]
                .as_str()
                .unwrap()
                .contains("bad request")
        );
        assert!(handle("pivot", b"{not json", b"").starts_with(b"{\"error\""));
        assert!(
            call("load_csv", &json!({}), b"a,b\n1\n")["error"]
                .as_str()
                .unwrap()
                .contains("fields")
        );
        assert!(
            call("load_csv", &json!({"delimiter": "ab"}), b"a\n1\n")["error"]
                .as_str()
                .unwrap()
                .contains("one character")
        );
        assert!(
            call("describe", &json!({"table": 99_999}), b"")["error"]
                .as_str()
                .unwrap()
                .contains("no table")
        );
    }

    #[test]
    fn delimiters_and_trim_are_options() {
        let a = call("load_csv", &json!({"delimiter": ";"}), b"a;b\n1;2\n");
        assert_eq!(a["ok"]["fields"].as_array().unwrap().len(), 2);
        let a = call("load_csv", &json!({"delimiter": "tab"}), b"a\tb\n1\t2\n");
        assert_eq!(a["ok"]["fields"].as_array().unwrap().len(), 2);
        let a = call("load_csv", &json!({"trim": false}), b"a,b\n x ,2\n");
        let id = a["ok"]["table"].as_u64().unwrap();
        assert_eq!(
            call("values", &json!({"table": id, "field": "a"}), b"")["ok"]["values"][0],
            " x "
        );
    }

    #[test]
    fn version() {
        assert_eq!(
            call("version", &json!({}), b"")["ok"]["engine"],
            "pivot-core"
        );
    }

    #[test]
    fn the_c_abi_round_trips_and_frees() {
        let (op, data) = (b"load_csv", CSV.as_bytes());
        let mut out_len = 0usize;
        let ptr = unsafe {
            pt_call(
                op.as_ptr(),
                op.len(),
                std::ptr::null(),
                0,
                data.as_ptr(),
                data.len(),
                &mut out_len,
            )
        };
        let answer: Value =
            serde_json::from_slice(unsafe { std::slice::from_raw_parts(ptr, out_len) }).unwrap();
        assert_eq!(answer["ok"]["rows"], 5);
        unsafe { pt_free(ptr, out_len) };

        let buf = pt_alloc(16);
        assert!(!buf.is_null());
        unsafe { pt_free(buf, 16) };
        unsafe { pt_free(std::ptr::null_mut(), 0) };

        // an unknown operation with a null out_len still answers and does not crash
        let ptr = unsafe {
            pt_call(
                b"zzz".as_ptr(),
                3,
                std::ptr::null(),
                0,
                std::ptr::null(),
                0,
                std::ptr::null_mut(),
            )
        };
        unsafe { pt_free(ptr, br#"{"error":"unknown operation 'zzz'"}"#.len()) };
    }
}
