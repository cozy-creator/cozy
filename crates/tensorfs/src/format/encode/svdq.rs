//! `svdq.nvfp4`'s encoder. `pack` lays out a calibrated producer's arrays (SVDQuant's): nvfp4's
//! (`codes`, `scale`, `scale_2`, `input_scale`, `pre_quant_scale`, required here), then
//! `channel_scale`, `lowrank_down` `[R, K]` and `lowrank_up` `[N, R]` as given.

use std::collections::BTreeMap;

use super::{Encoder, Input, OwnedPart};
use crate::format::decode::{self, PartBytes};
use crate::format::{Error, Result};

pub const SVDQ_NVFP4: Encoder = Encoder {
    scheme: "svdq.nvfp4",
    quantize: None,
    pack: Some(pack),
};

fn pack(input: &Input) -> Result<Vec<OwnedPart>> {
    let fail = |d: String| Error::Encode(format!("svdq.nvfp4: {d}"));
    if input.shape.len() != 2 || !input.arrays.contains_key("pre_quant_scale") {
        return Err(fail(format!("a [N, K] weight with its pre_quant_scale, not {:?}", input.shape)));
    }
    let mut parts = super::nvfp4::pack(input)?;
    for role in ["channel_scale", "lowrank_down", "lowrank_up"] {
        if let Some(a) = input.arrays.get(role) {
            let (dtype, shape, bytes) = (a.dtype.clone(), a.shape.to_vec(), a.bytes.to_vec());
            parts.push(OwnedPart { role: role.into(), dtype, shape, bytes });
        }
    }
    // The decode's own checks: every part's dtype, shape and length, one rank.
    let laid: BTreeMap<&str, PartBytes> = (parts.iter())
        .map(|p| (p.role.as_str(), PartBytes { dtype: &p.dtype, shape: &p.shape, bytes: &p.bytes }))
        .collect();
    decode::planned(input.dtype, input.shape, input.encoding, &laid, true, |_, _| Ok(()))
        .map_err(|e| fail(e.to_string()))?;
    Ok(parts)
}
