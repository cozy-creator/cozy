use super::encoding;
use crate::format::Encoding;

impl Encoding {
    pub fn int8_tensorwise() -> Encoding {
        encoding("int8.tensorwise", &[])
    }
    pub fn int8_rowwise() -> Encoding {
        encoding("int8.rowwise", &[])
    }
    pub fn int8_columnwise() -> Encoding {
        encoding("int8.columnwise", &[])
    }
    pub fn int8_convrot(rotation_size: u64) -> Encoding {
        let p = [
            ("rotation", "kron_regular_h4".into()),
            ("rotation_size", rotation_size.into()),
        ];
        encoding("int8.convrot", &p)
    }
}
