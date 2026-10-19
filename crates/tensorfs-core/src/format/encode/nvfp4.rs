use super::encoding;
use crate::format::{Encoding, ScaleLayout};

/// Which nibble of a byte holds an F4 pair's even element: ModelOpt's low, ComfyUI's high.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NibbleOrder {
    LowFirst,
    HighFirst,
}

/// How nvfp4's per-tensor scales, `scale_2` and `input_scale`, apply: multiplied (ModelOpt,
/// ComfyUI) or divided, being reciprocals (compressed-tensors' `weight_global_scale` and
/// `input_global_scale`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GlobalScale {
    Multiply,
    Divide,
}

impl NibbleOrder {
    pub fn name(self) -> &'static str {
        match self {
            NibbleOrder::LowFirst => "low_first",
            NibbleOrder::HighFirst => "high_first",
        }
    }
}

impl GlobalScale {
    pub fn name(self) -> &'static str {
        match self {
            GlobalScale::Multiply => "multiply",
            GlobalScale::Divide => "divide",
        }
    }
}

impl Encoding {
    pub fn nvfp4(global: GlobalScale, nibbles: NibbleOrder, layout: ScaleLayout) -> Encoding {
        let p = [
            ("block_size", 16u64.into()),
            ("global_scale", global.name().into()),
            ("nibble_order", nibbles.name().into()),
            ("scale_layout", layout.name().into()),
        ];
        encoding("nvfp4", &p)
    }
}
