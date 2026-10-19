use super::encoding;
use crate::format::{Dtype, Encoding};

impl Encoding {
    /// One scale for the tensor, stored as the producer wrote it: `F32`, `BF16` or `F16`.
    pub fn fp8_tensorwise(scale: &Dtype) -> Encoding {
        encoding("fp8.tensorwise", &[("scale_dtype", scale.name().into())])
    }
    /// `axis` 0: a scale per row; 1: per column.
    pub fn fp8_rowwise(axis: u64) -> Encoding {
        encoding("fp8.rowwise", &[("axis", axis.into())])
    }
}
