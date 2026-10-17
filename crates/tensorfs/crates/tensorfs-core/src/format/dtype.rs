//! The safetensors 0.8 dtypes, by their safetensors names. Little-endian; sub-byte dtypes pack
//! element 2k in the low bits of a byte (F4) or continuously from bit 0 (F6).

use serde::{Deserialize, Serialize};

#[allow(non_camel_case_types)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub enum Dtype {
    BOOL,
    F4,
    F6_E2M3,
    F6_E3M2,
    U8,
    I8,
    F8_E5M2,
    F8_E4M3,
    F8_E8M0,
    F8_E4M3FNUZ,
    F8_E5M2FNUZ,
    I16,
    U16,
    F16,
    BF16,
    I32,
    U32,
    F32,
    C64,
    F64,
    I64,
    U64,
}

impl Dtype {
    pub const ALL: [Dtype; 22] = [
        Dtype::BOOL,
        Dtype::F4,
        Dtype::F6_E2M3,
        Dtype::F6_E3M2,
        Dtype::U8,
        Dtype::I8,
        Dtype::F8_E5M2,
        Dtype::F8_E4M3,
        Dtype::F8_E8M0,
        Dtype::F8_E4M3FNUZ,
        Dtype::F8_E5M2FNUZ,
        Dtype::I16,
        Dtype::U16,
        Dtype::F16,
        Dtype::BF16,
        Dtype::I32,
        Dtype::U32,
        Dtype::F32,
        Dtype::C64,
        Dtype::F64,
        Dtype::I64,
        Dtype::U64,
    ];

    pub fn bits(self) -> u64 {
        use Dtype::*;
        match self {
            F4 => 4,
            F6_E2M3 | F6_E3M2 => 6,
            BOOL | U8 | I8 | F8_E5M2 | F8_E4M3 | F8_E8M0 | F8_E4M3FNUZ | F8_E5M2FNUZ => 8,
            I16 | U16 | F16 | BF16 => 16,
            I32 | U32 | F32 => 32,
            C64 | F64 | I64 | U64 => 64,
        }
    }

    /// Bytes of a tensor of this dtype and shape; `None` on overflow or when the elements do
    /// not fill whole bytes (safetensors refuses those too).
    pub fn byte_len(self, shape: &[u64]) -> Option<u64> {
        let bits = shape
            .iter()
            .try_fold(self.bits(), |acc, &extent| acc.checked_mul(extent))?;
        (bits % 8 == 0).then_some(bits / 8)
    }
}
