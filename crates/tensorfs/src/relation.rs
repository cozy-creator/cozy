//! The closed TAGGED shape-relation grammar and the ONE bounded checked-integer geometry
//! evaluator. Never an expression language: extending the vocabulary is a format era.

use crate::canon::{as_arr, as_str, Fields, Value};
use crate::dtype::Dtype;
use crate::err::{refuse, Code, Result};
use crate::limits;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Dim {
    /// logical dim at `axis`
    Axis {
        axis: u64,
    },
    /// logical dim ÷ by, remainder forbidden
    Div {
        axis: u64,
        by: u64,
    },
    /// ceil(logical dim ÷ by)
    CeilDiv {
        axis: u64,
        by: u64,
    },
    /// ceil(logical dim ÷ block) × mul — padded blocked arrays
    CeilBlock {
        axis: u64,
        block: u64,
        mul: u64,
    },
    /// total logical element count ÷ by, remainder forbidden
    ElementsDiv {
        by: u64,
    },
    Lit {
        value: u64,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Relation {
    /// storage shape equals the logical shape
    Same,
    Dims(Vec<Dim>),
}

impl Dim {
    fn from_value(v: &Value) -> Result<Dim> {
        let mut f = Fields::new("relation.dim", v)?;
        let t = as_str("relation.dim", "t", f.req("t")?)?.to_string();
        let d = match t.as_str() {
            "axis" => Dim::Axis {
                axis: f.req_uint("axis")?,
            },
            "div" => {
                let axis = f.req_uint("axis")?;
                let by = f.req_uint("by")?;
                let rem = as_str("relation.dim", "rem", f.req("rem")?)?;
                if rem != "forbid" {
                    return refuse(
                        Code::UNKNOWN_RELATION,
                        format!("div.rem {rem:?}: only \"forbid\" is defined"),
                    );
                }
                Dim::Div { axis, by }
            }
            "ceil_div" => Dim::CeilDiv {
                axis: f.req_uint("axis")?,
                by: f.req_uint("by")?,
            },
            "ceil_block" => Dim::CeilBlock {
                axis: f.req_uint("axis")?,
                block: f.req_uint("block")?,
                mul: f.req_uint("mul")?,
            },
            "elements_div" => {
                let by = f.req_uint("by")?;
                let rem = as_str("relation.dim", "rem", f.req("rem")?)?;
                if rem != "forbid" {
                    return refuse(
                        Code::UNKNOWN_RELATION,
                        format!("elements_div.rem {rem:?}: only \"forbid\" is defined"),
                    );
                }
                Dim::ElementsDiv { by }
            }
            "lit" => Dim::Lit {
                value: f.req_uint("value")?,
            },
            other => {
                return refuse(
                    Code::UNKNOWN_RELATION,
                    format!("dim tag {other:?} is outside the closed grammar"),
                )
            }
        };
        f.done()?;
        d.check_constants()?;
        Ok(d)
    }

    fn check_constants(&self) -> Result<()> {
        let zero = |n: u64, what: &str| -> Result<()> {
            if n == 0 {
                return refuse(Code::UNKNOWN_RELATION, format!("{what} must be positive"));
            }
            Ok(())
        };
        match self {
            Dim::Axis { .. } => Ok(()),
            Dim::Div { by, .. } | Dim::CeilDiv { by, .. } => zero(*by, "divisor"),
            Dim::CeilBlock { block, mul, .. } => {
                zero(*block, "block")?;
                zero(*mul, "mul")
            }
            Dim::ElementsDiv { by } => zero(*by, "divisor"),
            Dim::Lit { value } => zero(*value, "literal"),
        }
    }

    fn to_value(&self) -> Value {
        match self {
            Dim::Axis { axis } => Value::obj(vec![
                ("axis", Value::uint(*axis)),
                ("t", Value::str("axis")),
            ]),
            Dim::Div { axis, by } => Value::obj(vec![
                ("axis", Value::uint(*axis)),
                ("by", Value::uint(*by)),
                ("rem", Value::str("forbid")),
                ("t", Value::str("div")),
            ]),
            Dim::CeilDiv { axis, by } => Value::obj(vec![
                ("axis", Value::uint(*axis)),
                ("by", Value::uint(*by)),
                ("t", Value::str("ceil_div")),
            ]),
            Dim::CeilBlock { axis, block, mul } => Value::obj(vec![
                ("axis", Value::uint(*axis)),
                ("block", Value::uint(*block)),
                ("mul", Value::uint(*mul)),
                ("t", Value::str("ceil_block")),
            ]),
            Dim::ElementsDiv { by } => Value::obj(vec![
                ("by", Value::uint(*by)),
                ("rem", Value::str("forbid")),
                ("t", Value::str("elements_div")),
            ]),
            Dim::Lit { value } => Value::obj(vec![
                ("t", Value::str("lit")),
                ("value", Value::uint(*value)),
            ]),
        }
    }

    fn eval(&self, logical: &[u64], elements: u64) -> Result<u64> {
        let dim = |axis: u64| -> Result<u64> {
            match logical.get(axis as usize) {
                Some(d) => Ok(*d),
                None => refuse(
                    Code::SHAPE_MISMATCH,
                    format!(
                        "relation names axis {axis} but the logical shape has rank {}",
                        logical.len()
                    ),
                ),
            }
        };
        let exact = |n: u64, by: u64| -> Result<u64> {
            if !n.is_multiple_of(by) {
                return refuse(
                    Code::DIVISION_REMAINDER,
                    format!("{n} is not divisible by {by}"),
                );
            }
            Ok(n / by)
        };
        let mul = |a: u64, b: u64| -> Result<u64> {
            a.checked_mul(b).ok_or_else(|| crate::err::Refusal {
                code: Code::ARITH_OVERFLOW,
                detail: format!("{a} × {b} overflows"),
            })
        };
        match self {
            Dim::Axis { axis } => dim(*axis),
            Dim::Div { axis, by } => exact(dim(*axis)?, *by),
            Dim::CeilDiv { axis, by } => Ok(dim(*axis)?.div_ceil(*by)),
            Dim::CeilBlock {
                axis,
                block,
                mul: m,
            } => mul(dim(*axis)?.div_ceil(*block), *m),
            Dim::ElementsDiv { by } => exact(elements, *by),
            Dim::Lit { value } => Ok(*value),
        }
    }
}

impl Relation {
    pub fn from_value(v: &Value) -> Result<Relation> {
        let mut f = Fields::new("relation", v)?;
        let t = as_str("relation", "t", f.req("t")?)?.to_string();
        let r = match t.as_str() {
            "same" => Relation::Same,
            "dims" => {
                let dims = as_arr("relation", "dims", f.req("dims")?)?;
                if dims.len() > limits::MAX_RANK || dims.len() > limits::MAX_RELATION_NODES {
                    return refuse(Code::COUNT_CAP, "relation over the node cap");
                }
                Relation::Dims(
                    dims.iter()
                        .map(Dim::from_value)
                        .collect::<Result<Vec<_>>>()?,
                )
            }
            other => {
                return refuse(
                    Code::UNKNOWN_RELATION,
                    format!("relation tag {other:?} is outside the closed grammar"),
                )
            }
        };
        f.done()?;
        Ok(r)
    }

    pub fn to_value(&self) -> Value {
        match self {
            Relation::Same => Value::obj(vec![("t", Value::str("same"))]),
            Relation::Dims(d) => Value::obj(vec![
                ("dims", Value::arr(d.iter().map(Dim::to_value).collect())),
                ("t", Value::str("dims")),
            ]),
        }
    }

    /// The one evaluator: geometry only, never values.
    pub fn eval(&self, logical: &[u64], elements: u64) -> Result<Vec<u64>> {
        match self {
            Relation::Same => Ok(logical.to_vec()),
            Relation::Dims(dims) => dims.iter().map(|d| d.eval(logical, elements)).collect(),
        }
    }

    /// Can this relation be run BACKWARDS at all? A spec property, decided on the relation
    /// alone: every stored dim must name one logical axis and lose nothing (`axis`, `div`),
    /// and between them they must cover the logical axes contiguously from 0. The lossy and
    /// non-positional half of the grammar — `ceil_div`, `ceil_block`, `elements_div`, `lit` —
    /// never inverts, and a spec whose DATA role uses one has put its logical shape somewhere
    /// the border cannot read it back out of.
    pub fn invertible(&self) -> bool {
        let dims = match self {
            Relation::Same => return true,
            Relation::Dims(d) => d,
        };
        let mut seen = vec![false; dims.len()];
        for d in dims {
            let axis = match d {
                Dim::Axis { axis } | Dim::Div { axis, .. } => *axis as usize,
                _ => return false,
            };
            match seen.get_mut(axis) {
                // Two stored dims naming one logical axis: the recovery is over-determined,
                // which is a disagreement waiting to happen and never a default.
                Some(true) | None => return false,
                Some(s) => *s = true,
            }
        }
        seen.into_iter().all(|s| s)
    }

    /// The LOGICAL shape a stored shape came from. `None` when this relation has no inverse
    /// or when the stored shape does not have the relation's own rank — the first is a spec
    /// fact, the second a carrier fact, and the caller separates them.
    ///
    /// This is what makes a PACKED data role possible: `nvfp4-w4a4/1` stores `[out, in/2]`,
    /// so the companion relations must be evaluated against the `[out, in]` that came from,
    /// never against the packed bytes' own shape.
    pub fn invert(&self, stored: &[u64]) -> Result<Option<Vec<u64>>> {
        let dims = match self {
            Relation::Same => return Ok(Some(stored.to_vec())),
            Relation::Dims(d) => d,
        };
        if !self.invertible() || dims.len() != stored.len() {
            return Ok(None);
        }
        let mut logical = vec![0u64; dims.len()];
        for (d, got) in dims.iter().zip(stored) {
            let (axis, value) = match d {
                Dim::Axis { axis } => (*axis, *got),
                Dim::Div { axis, by } => (
                    *axis,
                    got.checked_mul(*by).ok_or_else(|| crate::err::Refusal {
                        code: Code::ARITH_OVERFLOW,
                        detail: format!("{got} × {by} overflows"),
                    })?,
                ),
                _ => return Ok(None),
            };
            logical[axis as usize] = value;
        }
        Ok(Some(logical))
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Carrier {
    SameAsLogical,
    Set(Vec<Dtype>),
}

impl Carrier {
    pub fn from_value(v: &Value) -> Result<Carrier> {
        let mut f = Fields::new("carrier", v)?;
        let t = as_str("carrier", "t", f.req("t")?)?.to_string();
        let c = match t.as_str() {
            "same_as_logical" => Carrier::SameAsLogical,
            "set" => {
                let a = as_arr("carrier", "dtypes", f.req("dtypes")?)?;
                if a.is_empty() {
                    return refuse(Code::MISSING_FIELD, "carrier.dtypes is empty");
                }
                let mut out = Vec::new();
                for x in a {
                    out.push(Dtype::parse(as_str("carrier", "dtypes", x)?)?);
                }
                let sorted: Vec<&str> = out.iter().map(|d| d.name()).collect();
                let mut s2 = sorted.clone();
                s2.sort_unstable();
                s2.dedup();
                if s2 != sorted {
                    return refuse(Code::SORT_ORDER, "carrier.dtypes must be sorted and unique");
                }
                Carrier::Set(out)
            }
            other => {
                return refuse(
                    Code::UNKNOWN_RELATION,
                    format!("carrier tag {other:?} is outside the closed grammar"),
                )
            }
        };
        f.done()?;
        Ok(c)
    }

    pub fn to_value(&self) -> Value {
        match self {
            Carrier::SameAsLogical => Value::obj(vec![("t", Value::str("same_as_logical"))]),
            Carrier::Set(d) => Value::obj(vec![
                (
                    "dtypes",
                    Value::arr(d.iter().map(|x| Value::str(x.name())).collect()),
                ),
                ("t", Value::str("set")),
            ]),
        }
    }

    pub fn admits(&self, logical: Dtype, carrier: Dtype) -> bool {
        match self {
            Carrier::SameAsLogical => carrier == logical,
            Carrier::Set(s) => s.contains(&carrier),
        }
    }
}
