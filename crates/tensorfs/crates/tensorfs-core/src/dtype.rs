//! The closed carrier/logical dtype enum. Every carrier is little-endian (format global).

use crate::err::{refuse, Code, Result};
use crate::limits;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Dtype {
    F64,
    F32,
    F16,
    Bf16,
    F8E4M3FN,
    F8E5M2,
    I64,
    I32,
    I16,
    I8,
    U8,
    Bool,
}

const TABLE: [(&str, Dtype, u64); 12] = [
    ("f64", Dtype::F64, 8),
    ("f32", Dtype::F32, 4),
    ("f16", Dtype::F16, 2),
    ("bf16", Dtype::Bf16, 2),
    ("f8_e4m3fn", Dtype::F8E4M3FN, 1),
    ("f8_e5m2", Dtype::F8E5M2, 1),
    ("i64", Dtype::I64, 8),
    ("i32", Dtype::I32, 4),
    ("i16", Dtype::I16, 2),
    ("i8", Dtype::I8, 1),
    ("u8", Dtype::U8, 1),
    ("bool", Dtype::Bool, 1),
];

impl Dtype {
    pub fn code(self) -> u8 {
        TABLE.iter().position(|(_, d, _)| *d == self).unwrap() as u8
    }
    pub fn from_code(code: u64) -> Result<Dtype> {
        let index = usize::try_from(code).map_err(|_| crate::err::Refusal {
            code: Code::NUMBER_RANGE,
            detail: format!("dtype code {code} is outside this platform's usize range"),
        })?;
        TABLE
            .get(index)
            .map(|(_, d, _)| *d)
            .ok_or_else(|| crate::err::Refusal {
                code: Code::DTYPE_UNKNOWN,
                detail: format!("dtype code {code} is outside 0..{}", TABLE.len()),
            })
    }
    pub fn parse(s: &str) -> Result<Dtype> {
        match TABLE.iter().find(|(n, _, _)| *n == s) {
            Some((_, d, _)) => Ok(*d),
            None => refuse(
                Code::DTYPE_UNKNOWN,
                format!("{s:?} is outside the closed dtype enum"),
            ),
        }
    }
    pub fn name(self) -> &'static str {
        TABLE.iter().find(|(_, d, _)| *d == self).unwrap().0
    }
    pub fn size(self) -> u64 {
        TABLE.iter().find(|(_, d, _)| *d == self).unwrap().2
    }
}

/// Positive dims, rank cap, checked product — the one geometry arithmetic.
pub fn checked_elements(what: &str, shape: &[u64]) -> Result<u64> {
    if shape.len() > limits::MAX_RANK {
        return refuse(
            Code::RANK_CAP,
            format!("{what}: rank {} over cap {}", shape.len(), limits::MAX_RANK),
        );
    }
    let mut n: u64 = 1;
    for d in shape {
        if *d == 0 {
            return refuse(
                Code::ZERO_ELEMENT,
                format!("{what}: zero-element shape {shape:?}"),
            );
        }
        n = match n.checked_mul(*d) {
            Some(v) => v,
            None => {
                return refuse(
                    Code::ARITH_OVERFLOW,
                    format!("{what}: element count overflow in {shape:?}"),
                )
            }
        };
        if n > limits::MAX_ELEMENTS {
            return refuse(
                Code::COUNT_CAP,
                format!("{what}: element count over cap in {shape:?}"),
            );
        }
    }
    Ok(n)
}

pub fn checked_bytes(what: &str, shape: &[u64], dt: Dtype) -> Result<u64> {
    let n = checked_elements(what, shape)?;
    match n.checked_mul(dt.size()) {
        Some(v) => Ok(v),
        None => refuse(Code::ARITH_OVERFLOW, format!("{what}: byte count overflow")),
    }
}
