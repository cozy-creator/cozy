use super::encoding;
use crate::format::{Encoding, ScaleLayout};

impl Encoding {
    pub fn mxfp8(scale_layout: ScaleLayout) -> Encoding {
        let p = [
            ("block_size", 32u64.into()),
            ("scale_layout", scale_layout.name().into()),
        ];
        encoding("mxfp8", &p)
    }
}
