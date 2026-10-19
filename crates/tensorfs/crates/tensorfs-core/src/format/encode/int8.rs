use super::encoding;
use crate::format::{Dtype, Encoding};

impl Encoding {
    pub fn int8_tensorwise(scale: &Dtype) -> Encoding {
        encoding("int8.tensorwise", &[("scale_dtype", scale.name().into())])
    }
    pub fn int8_rowwise(axis: u64) -> Encoding {
        encoding("int8.rowwise", &[("axis", axis.into())])
    }
    pub fn int8_convrot(rotation_size: u64) -> Encoding {
        let p = [
            ("rotation", "hadamard".into()),
            ("rotation_size", rotation_size.into()),
        ];
        encoding("int8.convrot", &p)
    }
}
