use super::encoding;
use crate::format::Encoding;

impl Encoding {
    pub fn int8_tensorwise() -> Encoding {
        encoding("int8.tensorwise", &[])
    }
    pub fn int8_rowwise(axis: u64) -> Encoding {
        encoding("int8.rowwise", &[("axis", axis.into())])
    }
    pub fn int8_convrot(rotation_size: u64) -> Encoding {
        let p = [
            ("rotation", "kron_regular_h4".into()),
            ("rotation_size", rotation_size.into()),
        ];
        encoding("int8.convrot", &p)
    }
}
