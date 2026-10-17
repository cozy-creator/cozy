//! Dtypes by their safetensors 0.8 names. Little-endian; sub-byte dtypes pack element 2k in
//! the low bits of a byte (F4) or continuously from bit 0 (F6). Any other non-empty name reads
//! as `Other`: its bytes are located and verified, only a typed view or decode refuses.

use std::fmt;

use serde::{Deserialize, Deserializer, Serialize, Serializer};

#[allow(non_camel_case_types)]
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
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
    /// A name this reader does not know: a newer dtype.
    Other(String),
}

/// The known dtypes: name, value, bits per element.
const KNOWN: [(&str, Dtype, u64); 22] = [
    ("BOOL", Dtype::BOOL, 8),
    ("F4", Dtype::F4, 4),
    ("F6_E2M3", Dtype::F6_E2M3, 6),
    ("F6_E3M2", Dtype::F6_E3M2, 6),
    ("U8", Dtype::U8, 8),
    ("I8", Dtype::I8, 8),
    ("F8_E5M2", Dtype::F8_E5M2, 8),
    ("F8_E4M3", Dtype::F8_E4M3, 8),
    ("F8_E8M0", Dtype::F8_E8M0, 8),
    ("F8_E4M3FNUZ", Dtype::F8_E4M3FNUZ, 8),
    ("F8_E5M2FNUZ", Dtype::F8_E5M2FNUZ, 8),
    ("I16", Dtype::I16, 16),
    ("U16", Dtype::U16, 16),
    ("F16", Dtype::F16, 16),
    ("BF16", Dtype::BF16, 16),
    ("I32", Dtype::I32, 32),
    ("U32", Dtype::U32, 32),
    ("F32", Dtype::F32, 32),
    ("C64", Dtype::C64, 64),
    ("F64", Dtype::F64, 64),
    ("I64", Dtype::I64, 64),
    ("U64", Dtype::U64, 64),
];

impl Dtype {
    /// Every known dtype.
    pub fn known() -> impl Iterator<Item = Dtype> {
        KNOWN.into_iter().map(|(_, d, _)| d)
    }

    pub fn parse(name: &str) -> Dtype {
        match KNOWN.iter().find(|(n, _, _)| *n == name) {
            Some((_, d, _)) => d.clone(),
            None => Dtype::Other(name.into()),
        }
    }

    pub fn name(&self) -> &str {
        match self {
            Dtype::Other(name) => name,
            known => {
                KNOWN
                    .iter()
                    .find(|(_, d, _)| d == known)
                    .expect("every variant is listed")
                    .0
            }
        }
    }

    /// Bits per element; `None` for an unknown dtype.
    pub fn bits(&self) -> Option<u64> {
        KNOWN.iter().find(|(_, d, _)| d == self).map(|(_, _, b)| *b)
    }

    /// Bytes of a tensor of this dtype and shape; `None` for an unknown dtype, on overflow, or
    /// when the elements do not fill whole bytes (safetensors refuses those too).
    pub fn byte_len(&self, shape: &[u64]) -> Option<u64> {
        let bits = shape
            .iter()
            .try_fold(self.bits()?, |acc, &extent| acc.checked_mul(extent))?;
        (bits % 8 == 0).then_some(bits / 8)
    }
}

impl fmt::Display for Dtype {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str(self.name())
    }
}

impl Serialize for Dtype {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        s.serialize_str(self.name())
    }
}

impl<'de> Deserialize<'de> for Dtype {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Dtype, D::Error> {
        let name = String::deserialize(d)?;
        if name.is_empty() {
            return Err(serde::de::Error::custom("an empty dtype name"));
        }
        Ok(Dtype::parse(&name))
    }
}
