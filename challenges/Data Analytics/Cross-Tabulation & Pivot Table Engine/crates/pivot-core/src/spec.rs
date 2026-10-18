//! The request and result types: what a pivot is asked to do, and what comes back.
//! Both sides are plain JSON, so the engine behaves the same from Rust, Python and the browser.

use serde::{Deserialize, Serialize};
use serde_json::Value as Json;

#[derive(Debug, Clone, Default, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct PivotSpec {
    #[serde(default)]
    pub rows: Vec<Dim>,
    #[serde(default)]
    pub cols: Vec<Dim>,
    #[serde(default)]
    pub values: Vec<Value>,
    #[serde(default)]
    pub filters: Vec<Filter>,
    #[serde(default)]
    pub options: Options,
}

/// A field to lay out along an axis, optionally grouped, sorted and cut to the top N.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Dim {
    pub field: String,
    #[serde(default)]
    pub group: Option<Group>,
    #[serde(default)]
    pub sort: Sort,
    #[serde(default)]
    pub top: Option<Top>,
    /// Drop the "(blank)" item.
    #[serde(default)]
    pub hide_blank: bool,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum Group {
    Year,
    Quarter,
    Month,
    YearMonth,
    YearQuarter,
    Weekday,
    Day,
    Hour,
    /// Equal-width bins `[start + i*width, start + (i+1)*width)`; `start` defaults to 0.
    Bin {
        width: f64,
        #[serde(default)]
        start: Option<f64>,
    },
    /// Bins between the given cut points, plus one below the first and one above the last.
    Edges {
        edges: Vec<f64>,
    },
    /// `n` bins with (about) the same number of rows each.
    Quantile {
        n: usize,
    },
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
#[serde(tag = "by", rename_all = "snake_case", deny_unknown_fields)]
pub enum Sort {
    /// The field's natural order: numbers and dates by value, text A to Z.
    Label {
        #[serde(default)]
        desc: bool,
    },
    /// By a measure's total over the other axis.
    Value {
        measure: String,
        #[serde(default = "yes")]
        desc: bool,
    },
}

impl Default for Sort {
    fn default() -> Self {
        Sort::Label { desc: false }
    }
}

fn yes() -> bool {
    true
}

/// Keep only the `n` items with the largest (or smallest) value of a measure, per parent group.
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct Top {
    pub n: usize,
    pub measure: String,
    #[serde(default = "yes")]
    pub desc: bool,
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Agg {
    Sum,
    Count,
    Distinct,
    Mean,
    Median,
    Min,
    Max,
    Std,
    Var,
    First,
    Last,
    Mode,
    WeightedMean,
}

impl Agg {
    pub fn name(self) -> &'static str {
        match self {
            Agg::Sum => "sum",
            Agg::Count => "count",
            Agg::Distinct => "distinct",
            Agg::Mean => "mean",
            Agg::Median => "median",
            Agg::Min => "min",
            Agg::Max => "max",
            Agg::Std => "std",
            Agg::Var => "var",
            Agg::First => "first",
            Agg::Last => "last",
            Agg::Mode => "mode",
            Agg::WeightedMean => "weighted_mean",
        }
    }

    /// Whether the field must be numeric (count and distinct work on anything).
    pub fn needs_number(self) -> bool {
        !matches!(self, Agg::Count | Agg::Distinct)
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Value {
    /// `None` with `count` counts rows.
    #[serde(default)]
    pub field: Option<String>,
    #[serde(default = "default_agg")]
    pub agg: Agg,
    #[serde(default)]
    pub label: Option<String>,
    /// The weights for `weighted_mean`.
    #[serde(default)]
    pub weight: Option<String>,
    /// A formula over earlier measures; when set, `field` and `agg` are ignored.
    #[serde(default)]
    pub calc: Option<String>,
    #[serde(default)]
    pub show_as: ShowAs,
}

fn default_agg() -> Agg {
    Agg::Sum
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Axis {
    Rows,
    Cols,
}

#[derive(Debug, Clone, Default, Deserialize, Serialize, PartialEq)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum ShowAs {
    #[default]
    Normal,
    PctGrand,
    PctRow,
    PctCol,
    PctParentRow,
    PctParentCol,
    /// Difference from the previous item along the axis.
    Diff {
        along: Axis,
    },
    PctDiff {
        along: Axis,
    },
    Running {
        along: Axis,
    },
    Rank {
        along: Axis,
        #[serde(default = "yes")]
        desc: bool,
    },
    Index,
}

impl ShowAs {
    pub fn is_percent(&self) -> bool {
        matches!(
            self,
            ShowAs::PctGrand
                | ShowAs::PctRow
                | ShowAs::PctCol
                | ShowAs::PctParentRow
                | ShowAs::PctParentCol
                | ShowAs::PctDiff { .. }
        )
    }

    pub fn name(&self) -> &'static str {
        match self {
            ShowAs::Normal => "normal",
            ShowAs::PctGrand => "pct_grand",
            ShowAs::PctRow => "pct_row",
            ShowAs::PctCol => "pct_col",
            ShowAs::PctParentRow => "pct_parent_row",
            ShowAs::PctParentCol => "pct_parent_col",
            ShowAs::Diff { .. } => "diff",
            ShowAs::PctDiff { .. } => "pct_diff",
            ShowAs::Running { .. } => "running",
            ShowAs::Rank { .. } => "rank",
            ShowAs::Index => "index",
        }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct Filter {
    pub field: String,
    pub op: FilterOp,
    #[serde(default)]
    pub value: Option<Json>,
    #[serde(default)]
    pub values: Option<Vec<Json>>,
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum FilterOp {
    Eq,
    Ne,
    In,
    NotIn,
    Gt,
    Ge,
    Lt,
    Le,
    Between,
    Contains,
    IsNull,
    NotNull,
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum SubtotalPosition {
    Top,
    #[default]
    Bottom,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Options {
    /// A grand-total row at the bottom.
    #[serde(default = "yes")]
    pub totals_row: bool,
    /// A grand-total column at the right.
    #[serde(default = "yes")]
    pub totals_col: bool,
    /// A subtotal for every group of an axis with more than one field.
    #[serde(default)]
    pub subtotals: bool,
    #[serde(default)]
    pub subtotal_position: SubtotalPosition,
    /// The most cells (rows x columns) a result may have.
    #[serde(default = "default_max_cells")]
    pub max_cells: usize,
    #[serde(default = "default_blank")]
    pub blank_label: String,
}

fn default_max_cells() -> usize {
    2_000_000
}

fn default_blank() -> String {
    "(blank)".to_string()
}

impl Default for Options {
    fn default() -> Self {
        Options {
            totals_row: true,
            totals_col: true,
            subtotals: false,
            subtotal_position: SubtotalPosition::Bottom,
            max_cells: default_max_cells(),
            blank_label: default_blank(),
        }
    }
}

// ---------------------------------------------------------------------------------- result

#[derive(Debug, Clone, Copy, Serialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum HeaderKind {
    Leaf,
    Subtotal,
    Total,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct Header {
    /// One entry per field of the axis; `None` where the item has no value (under a subtotal or total).
    pub labels: Vec<Option<String>>,
    pub kind: HeaderKind,
    /// How many leading fields are fixed: the number of labels that identify this item.
    pub depth: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct MeasureInfo {
    pub label: String,
    pub agg: String,
    pub show_as: String,
    pub percent: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct PivotResult {
    pub row_fields: Vec<String>,
    pub col_fields: Vec<String>,
    pub measures: Vec<MeasureInfo>,
    pub rows: Vec<Header>,
    pub cols: Vec<Header>,
    /// `cells[row][col][measure]`; `None` is an empty cell.
    pub cells: Vec<Vec<Vec<Option<f64>>>>,
    pub rows_used: usize,
    pub rows_total: usize,
    pub warnings: Vec<String>,
}
