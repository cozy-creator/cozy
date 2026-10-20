use super::encoding;
use crate::format::{Dtype, Encoding};

impl Encoding {
    pub fn int8_tensorwise(scale: &Dtype) -> Encoding {
        encoding("int8.tensorwise", &[("scale_dtype", scale.name().into())])
    }
    pub fn int8_rowwise(axis: u64, scale: &Dtype) -> Encoding {
        let p = [("axis", axis.into()), ("scale_dtype", scale.name().into())];
        encoding("int8.rowwise", &p)
    }
    pub fn int8_convrot(rotation_size: u64, scale: &Dtype) -> Encoding {
        let p = [
            ("rotation", "kron_regular_h4".into()),
            ("rotation_size", rotation_size.into()),
            ("scale_dtype", scale.name().into()),
        ];
        encoding("int8.convrot", &p)
    }
}
