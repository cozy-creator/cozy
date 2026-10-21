use super::encoding;
use crate::format::Encoding;

impl Encoding {
    /// One scale for the tensor, its part stored as the producer wrote it (`F32`, `BF16` or `F16`).
    pub fn fp8_tensorwise() -> Encoding {
        encoding("fp8.tensorwise", &[])
    }
    /// A scale per row.
    pub fn fp8_rowwise() -> Encoding {
        encoding("fp8.rowwise", &[])
    }
    /// A scale per column.
    pub fn fp8_columnwise() -> Encoding {
        encoding("fp8.columnwise", &[])
    }
}
