use super::encoding;
use crate::format::Encoding;

impl Encoding {
    /// One scale for the tensor, its part stored as the producer wrote it (`F32`, `BF16` or `F16`).
    pub fn fp8_tensorwise() -> Encoding {
        encoding("fp8.tensorwise", &[])
    }
    /// `axis` 0: a scale per row; 1: per column.
    pub fn fp8_rowwise(axis: u64) -> Encoding {
        encoding("fp8.rowwise", &[("axis", axis.into())])
    }
}
