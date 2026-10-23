//! bitsandbytes' quantizers, bit for bit as its CPU backend runs them (bitsandbytes 0.50, torch
//! 2.14 on x86 AVX2): `quantize_4bit` (the default backend), its double quant (`absmax.mean()`,
//! then the C++ `quantize_blockwise` over blocks of 256 with bnb's 8-bit dynamic map), and
//! `int8_vectorwise_quant` as `Int8Params` calls it, on the weight cast to F16.
//!
//! bnb's CUDA kernels pick the nearest 4-bit code by a decision tree and the nearest 8-bit code
//! by binary search, where its CPU backend uses midpoints and a 65,536-bin table; the codes may
//! differ near a midpoint. Its `mean()` follows torch's reduction, which splits across threads
//! from 32,768 blocks on.

use super::{Encoder, Input, OwnedPart};
use crate::format::decode::bnb::{FP4_MAP, NESTED, NF4_MAP};
use crate::format::decode::{elem, f16};
use crate::format::{Dtype, Error, Result, Value};

pub const NF4: Encoder = Encoder {
    scheme: "bnb.nf4",
    quantize: Some(|i| four_bit(i, &NF4_MAP)),
    pack: None,
};

pub const FP4: Encoder = Encoder {
    scheme: "bnb.fp4",
    quantize: Some(|i| four_bit(i, &FP4_MAP)),
    pack: None,
};

pub const INT8: Encoder = Encoder {
    scheme: "bnb.int8",
    quantize: Some(int8),
    pack: None,
};

/// torch's `clamp(min=1e-38)` floor on an absmax.
const TINY: f32 = f32::from_bits(0x006c_e3ee);

/// The positive half of bnb's 8-bit dynamic map (`create_dynamic_map()`, the
/// `nested_quant_map` it writes): the map is these negated in reverse, 0, these, then 1.
const DYNAMIC_HALF: [u32; 127] = [
    0x3513a3b7, 0x365a1a94, 0x370205ff, 0x37b24206, 0x38378034, 0x388aefb3, 0x38ba1f4b, 0x3923d70b,
    0x398ce705, 0x39c7e283, 0x3a016f01, 0x3a1eecc1, 0x3a3c6a80, 0x3a59e83e, 0x3a7765ff, 0x3aa7ef9d,
    0x3af1a9fc, 0x3b1db22d, 0x3b428f5c, 0x3b676c8a, 0x3b8624dd, 0x3b989374, 0x3bab020d, 0x3bbd70a4,
    0x3bcfdf3b, 0x3be24dd3, 0x3bf4bc6a, 0x3c039581, 0x3c0ccccd, 0x3c160418, 0x3c1f3b64, 0x3c3ae148,
    0x3c68f5c3, 0x3c8b851f, 0x3ca28f5d, 0x3cb9999a, 0x3cd0a3d7, 0x3ce7ae15, 0x3cfeb852, 0x3d0ae148,
    0x3d166667, 0x3d21eb85, 0x3d2d70a3, 0x3d38f5c3, 0x3d447ae1, 0x3d500000, 0x3d5b851f, 0x3d670a3d,
    0x3d728f5d, 0x3d7e147b, 0x3d84cccd, 0x3d8a8f5d, 0x3d9051eb, 0x3d96147b, 0x3d9bd70a, 0x3da1999a,
    0x3da75c2a, 0x3dad1eb8, 0x3db2e148, 0x3db8a3d7, 0x3dbe6667, 0x3dc428f7, 0x3dc9eb85, 0x3ddb3334,
    0x3df80000, 0x3e0a6666, 0x3e18cccd, 0x3e273333, 0x3e35999a, 0x3e440000, 0x3e526666, 0x3e60cccd,
    0x3e6f3333, 0x3e7d9999, 0x3e860000, 0x3e8d3334, 0x3e946666, 0x3e9b999a, 0x3ea2cccc, 0x3eaa0000,
    0x3eb13333, 0x3eb86666, 0x3ebf999a, 0x3ec6cccc, 0x3ece0000, 0x3ed53333, 0x3edc6666, 0x3ee3999a,
    0x3eeacccc, 0x3ef20000, 0x3ef93332, 0x3f003333, 0x3f03cccc, 0x3f076666, 0x3f0b0000, 0x3f0e999a,
    0x3f123334, 0x3f15cccd, 0x3f196666, 0x3f1d0000, 0x3f20999a, 0x3f243334, 0x3f27cccd, 0x3f2b6666,
    0x3f2f0000, 0x3f32999a, 0x3f363334, 0x3f39cccd, 0x3f3d6666, 0x3f410000, 0x3f44999a, 0x3f483333,
    0x3f4bcccd, 0x3f4f6666, 0x3f530000, 0x3f56999a, 0x3f5a3333, 0x3f5dcccd, 0x3f616666, 0x3f650000,
    0x3f68999a, 0x3f6c3333, 0x3f6fcccd, 0x3f736666, 0x3f770000, 0x3f7a999a, 0x3f7e3333,
];

fn part(role: &str, dtype: Dtype, shape: Vec<u64>, bytes: Vec<u8>) -> OwnedPart {
    OwnedPart {
        role: role.into(),
        dtype,
        shape,
        bytes,
    }
}

fn f32s(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

/// `quantize_4bit`: per block of the flattened tensor its absmax, each value times its
/// reciprocal (the short last block: divided by it), clamped to [-1, 1], then the table entry
/// nearest (the midpoints of the sorted table, a tie going down); two codes to a byte, the first
/// high, an odd count padded with the code of 0.
fn four_bit(input: &Input, map: &[f32; 16]) -> Result<Vec<OwnedPart>> {
    let scheme = &input.encoding.scheme;
    let fail = |d: String| Error::Encode(format!("{scheme}: {d}"));
    let params = &input.encoding.params;
    let block = match params.get("block_size").and_then(Value::as_u64) {
        Some(b @ (32 | 64 | 128 | 256 | 512 | 1024 | 2048 | 4096)) => b as usize,
        _ => return Err(fail("block_size is not 32, 64, ..., 4096".into())),
    };
    let dq = match params.get("double_quant") {
        Some(Value::Bool(dq)) => *dq,
        _ => return Err(fail("double_quant is not a boolean".into())),
    };
    if params.len() != 2 {
        return Err(fail(format!("params {:?}", params.keys())));
    }
    let x = input.values()?;
    let mut order: Vec<usize> = (0..16).collect();
    order.sort_by(|a, b| map[*a].partial_cmp(&map[*b]).expect("a table"));
    let mid: Vec<f32> = order
        .windows(2)
        .map(|p| (map[p[0]] + map[p[1]]) / 2.0)
        .collect();
    let code = |v: f32| order[mid.partition_point(|m| *m < v.clamp(-1.0, 1.0))] as u8;
    let (mut codes, mut absmax) = (Vec::with_capacity(x.len() + 1), Vec::new());
    for b in x.chunks(block) {
        let am = b.iter().fold(0f32, |a, v| a.max(v.abs()));
        let floor = am.max(TINY);
        if b.len() == block {
            let inv = 1.0 / floor;
            codes.extend(b.iter().map(|v| code(v * inv)));
            absmax.push(am);
        } else {
            codes.extend(b.iter().map(|v| code(v / floor)));
            absmax.push(floor);
        }
    }
    if codes.len() % 2 == 1 {
        codes.push(code(0.0));
    }
    let n = x.len() as u64;
    let packed = codes.chunks(2).map(|p| p[0] << 4 | p[1]).collect();
    let stored: Vec<f32> = map.iter().map(|v| v + 0.0).collect(); // bnb stores -0 as +0
    let nb = absmax.len() as u64;
    let mut parts = vec![part("weight", Dtype::U8, vec![n.div_ceil(2), 1], packed)];
    if !dq {
        parts.push(part("absmax", Dtype::F32, vec![nb], f32s(&absmax)));
        parts.push(part("quant_map", Dtype::F32, vec![16], f32s(&stored)));
        return Ok(parts);
    }
    let offset = torch_sum(&absmax) / nb as f32;
    let dynamic: Vec<f32> = (DYNAMIC_HALF.iter().rev().map(|b| -f32::from_bits(*b)))
        .chain([0.0])
        .chain(DYNAMIC_HALF.iter().map(|b| f32::from_bits(*b)))
        .chain([1.0])
        .collect();
    let (codes, nested) = blockwise(
        &absmax.iter().map(|a| a - offset).collect::<Vec<_>>(),
        &dynamic,
    );
    parts.extend([
        part("absmax", Dtype::U8, vec![nb], codes),
        part("quant_map", Dtype::F32, vec![16], f32s(&stored)),
        part(
            "nested_absmax",
            Dtype::F32,
            vec![nested.len() as u64],
            f32s(&nested),
        ),
        part("nested_quant_map", Dtype::F32, vec![256], f32s(&dynamic)),
        part("nested_offset", Dtype::F32, vec![], f32s(&[offset])),
    ]);
    Ok(parts)
}

/// bnb's C++ `quantize_blockwise` on CPU over blocks of 256: each value times its block's
/// reciprocal absmax, clamped, to one of 65,536 bins (`(v + 1) × 0.5 × 65535 + 0.5`, one fused
/// multiply-add, truncated), each bin the code whose midpoints its centre passes.
fn blockwise(x: &[f32], map: &[f32]) -> (Vec<u8>, Vec<f32>) {
    let mid: Vec<f32> = map.windows(2).map(|p| (p[0] + p[1]) * 0.5).collect();
    let bin = |v: f32| {
        let i = ((v.clamp(-1.0, 1.0) + 1.0) * 0.5).mul_add(65535.0, 0.5) as u16;
        let centre = (2 * i as u32) as f32 / 65535.0 - 1.0;
        mid.partition_point(|m| *m < centre) as u8
    };
    let (mut codes, mut absmax) = (Vec::with_capacity(x.len()), Vec::new());
    for b in x.chunks(NESTED) {
        let am = b.iter().fold(0f32, |a, v| a.max(v.abs()));
        absmax.push(am);
        let inv = 1.0 / am;
        codes.extend(b.iter().map(|v| if am == 0.0 { 0 } else { bin(v * inv) }));
    }
    (codes, absmax)
}

/// torch's CPU float sum on one thread (`cascade_sum`, AVX2's 8 lanes): under 8 values one
/// `row_sum`, else the lanes' `row_sum`, then the leftover values and the lanes in order.
fn torch_sum(x: &[f32]) -> f32 {
    if x.len() < 8 {
        return row_sum::<1>(x)[0];
    }
    let whole = x.len() / 8 * 8;
    let lanes = row_sum::<8>(&x[..whole]);
    x[whole..].iter().chain(&lanes).fold(0.0, |a, b| a + b)
}

/// `row_sum`: `x` as vectors of `W`, four interleaved cascades, the leftover vectors into the
/// first, then the four in order.
fn row_sum<const W: usize>(x: &[f32]) -> [f32; W] {
    let at = |i: usize| -> [f32; W] { std::array::from_fn(|l| x[i * W + l]) };
    let (vecs, ilp) = (x.len() / W, x.len() / W / 4);
    let mut p = cascade(ilp, |i, k| at(i * 4 + k));
    for i in ilp * 4..vecs {
        add(&mut p[0], &at(i));
    }
    for k in 1..4 {
        let pk = p[k];
        add(&mut p[0], &pk);
    }
    p[0]
}

/// `multi_row_sum` over four rows: four levels of accumulators, each passed up every
/// `2^power` additions so that like magnitudes add.
fn cascade<const W: usize>(size: usize, at: impl Fn(usize, usize) -> [f32; W]) -> [[f32; W]; 4] {
    let log2 = (usize::BITS - size.saturating_sub(1).leading_zeros()).max(1) as usize;
    let power = 4.max(log2 / 4);
    let (step, mask) = (1 << power, (1 << power) - 1);
    let mut acc = [[[0f32; W]; 4]; 4];
    let row = |i: usize| -> [[f32; W]; 4] { std::array::from_fn(|k| at(i, k)) };
    let mut i = 0;
    while i + step <= size {
        for _ in 0..step {
            add(acc[0].as_flattened_mut(), row(i).as_flattened());
            i += 1;
        }
        for j in 1..4 {
            let lower = std::mem::replace(&mut acc[j - 1], [[0.0; W]; 4]);
            add(acc[j].as_flattened_mut(), lower.as_flattened());
            if i & (mask << (j * power)) != 0 {
                break;
            }
        }
    }
    for i in i..size {
        add(acc[0].as_flattened_mut(), row(i).as_flattened());
    }
    for j in 1..4 {
        let level = acc[j];
        add(acc[0].as_flattened_mut(), level.as_flattened());
    }
    acc[0]
}

/// Lane-wise `a += b`.
fn add(a: &mut [f32], b: &[f32]) {
    a.iter_mut().zip(b).for_each(|(a, b)| *a += b);
}

/// `int8_vectorwise_quant` on the weight cast to F16: per row `q = round(x × ((1 / absmax) × 127))`,
/// ties to even; a zero row is zeros.
fn int8(input: &Input) -> Result<Vec<OwnedPart>> {
    let fail = |d: String| Error::Encode(format!("bnb.int8: {d}"));
    if !input.encoding.params.is_empty() {
        return Err(fail("bnb.int8 has no params".into()));
    }
    let [_, k] = input.shape[..] else {
        return Err(fail(format!("shape {:?} is not [N, K]", input.shape)));
    };
    let x: Vec<f32> = input.values()?.iter().map(|v| elem::f16(f16(*v))).collect();
    if x.iter().any(|v| v.is_infinite()) {
        return Err(fail(
            "a value is beyond F16, which bnb.int8 quantizes in".into(),
        ));
    }
    let (mut q, mut scb) = (Vec::with_capacity(x.len()), Vec::new());
    for row in x.chunks(k.max(1) as usize) {
        let am = row.iter().fold(0f32, |a, v| a.max(v.abs()));
        // PyTorch computes 127.0 / t as reciprocal(t) * 127; match its rounding.
        let s = (1.0 / am) * 127.0;
        q.extend(row.iter().map(|v| match am {
            0.0 => 0,
            _ => (v * s).round_ties_even() as i8 as u8,
        }));
        scb.push(am);
    }
    Ok(vec![
        part("weight", Dtype::I8, input.shape.to_vec(), q),
        part("SCB", Dtype::F32, vec![scb.len() as u64], f32s(&scb)),
        part("weight_format", Dtype::U8, vec![], vec![0]),
    ])
}
