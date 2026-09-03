//! `tfs conform` — the conformance harness (tfs-008).
//!
//! This qualifies an IMPLEMENTATION against a spec's vectors, bit-for-bit, once, at
//! qualification time — never in the serve path. The implementations under qualification
//! live HERE, in the tool, deliberately: `tensorfs-core` does bookkeeping and geometry and
//! never reconstructs values, so a decoder inside it would be exactly the "interpreter in
//! the trust path" cozytensors.md §4 refuses.
//!
//! Every candidate must fail the deliberately-wrong arms before its pass means anything. A
//! decoder that gets nibble order or scale axis wrong produces plausible output, which is
//! the whole reason review-only evidence was ruled too weak.

use std::process::ExitCode;

use tensorfs_core::dtype::Dtype;
use tensorfs_core::registry;
use tensorfs_core::spec::EncodingSpec;
use tensorfs_core::vectors::{Case, EncodingVectors};

use crate::{bail, flag, Flags};

// ---------------------------------------------------------------- carrier math

fn f32_bits(v: f32) -> [u8; 4] {
    v.to_le_bytes()
}

/// bf16 is the top 16 bits of f32. Round-to-nearest-even on the way down, which is what
/// every producer in this ecosystem does.
fn bf16_from_f32(v: f32) -> [u8; 2] {
    let b = v.to_bits();
    if (b & 0x7fff_ffff) > 0x7f80_0000 {
        // NaN: keep it a NaN, do not let rounding turn it into an infinity
        return ((b >> 16) as u16).to_le_bytes();
    }
    let rounded = b + 0x7fff + ((b >> 16) & 1);
    ((rounded >> 16) as u16).to_le_bytes()
}

/// f8_e4m3fn (the "fn" variant: finite-only — no infinities, 0xff/0x7f are NaN).
fn f32_from_e4m3fn(b: u8) -> f32 {
    let sign = if b & 0x80 != 0 { -1.0f32 } else { 1.0 };
    let exp = ((b >> 3) & 0x0f) as i32;
    let man = (b & 0x07) as u32;
    if exp == 0x0f && man == 0x07 {
        return f32::NAN * sign;
    }
    if exp == 0 {
        // subnormal: 2^-6 * (man / 8)
        return sign * (man as f32) * 2f32.powi(-9);
    }
    sign * (1.0 + man as f32 / 8.0) * 2f32.powi(exp - 7)
}

/// e2m1: the 4-bit nvfp4 element. Sixteen values, table-exact.
fn f32_from_e2m1(n: u8) -> f32 {
    const MAG: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
    let v = MAG[(n & 0x07) as usize];
    if n & 0x08 != 0 {
        -v
    } else {
        v
    }
}

/// E8M0: a bare power-of-two exponent byte, bias 127. 0xff is NaN.
fn f32_from_e8m0(b: u8) -> f32 {
    if b == 0xff {
        return f32::NAN;
    }
    2f32.powi(b as i32 - 127)
}

fn f32_from_half(b: [u8; 2]) -> f32 {
    let h = u16::from_le_bytes(b) as u32;
    let (sign, exp, man) = ((h >> 15) & 1, (h >> 10) & 0x1f, h & 0x3ff);
    let bits = match exp {
        0 if man == 0 => sign << 31,
        0 => {
            // subnormal: renormalise into f32's exponent field
            let shift = man.leading_zeros() - 21;
            (sign << 31) | ((127 - 15 - shift) << 23) | ((man << (shift + 1)) & 0x007f_ffff)
        }
        0x1f => (sign << 31) | 0x7f80_0000 | (man << 13),
        e => (sign << 31) | ((e + 127 - 15) << 23) | (man << 13),
    };
    f32::from_bits(bits)
}

/// The value narrowed to the logical dtype and read back. Some producers multiply in the
/// LOGICAL dtype rather than in f32, so where a scale lands is part of the byte contract.
fn logical_round(dt: Dtype, v: f32) -> f32 {
    match dt {
        Dtype::Bf16 => f32::from_bits((u16::from_le_bytes(bf16_from_f32(v)) as u32) << 16),
        Dtype::F16 => f32_from_half(half_from_f32(v)),
        _ => v,
    }
}

fn put_logical(dt: Dtype, v: f32, out: &mut Vec<u8>) {
    match dt {
        Dtype::F32 => out.extend_from_slice(&f32_bits(v)),
        Dtype::Bf16 => out.extend_from_slice(&bf16_from_f32(v)),
        Dtype::F16 => out.extend_from_slice(&half_from_f32(v)),
        _ => out.extend_from_slice(&f32_bits(v)),
    }
}

/// f32 -> f16 with round-to-nearest-EVEN, subnormals included. Found by running: the
/// truncating version passed every bf16 case and failed one f16 nvfp4 case by one ulp,
/// which is exactly the "plausible wrong output" the vectors exist to catch.
fn half_from_f32(v: f32) -> [u8; 2] {
    let x = v.to_bits();
    let sign = (x >> 16) & 0x8000;
    let mag = x & 0x7fff_ffff;
    if mag >= 0x7f80_0000 {
        let h = if mag > 0x7f80_0000 { 0x7e00 } else { 0x7c00 };
        return ((sign | h) as u16).to_le_bytes();
    }
    let exp = ((mag >> 23) as i32) - 127 + 15;
    if exp >= 0x1f {
        return ((sign | 0x7c00) as u16).to_le_bytes();
    }
    if exp <= 0 {
        if exp < -10 {
            return (sign as u16).to_le_bytes();
        }
        let man = (mag & 0x007f_ffff) | 0x0080_0000;
        let shift = (14 - exp) as u32;
        let mut h = man >> shift;
        let rem = man & ((1u32 << shift) - 1);
        let half = 1u32 << (shift - 1);
        if rem > half || (rem == half && (h & 1) == 1) {
            h += 1;
        }
        return ((sign | h) as u16).to_le_bytes();
    }
    let man = mag & 0x007f_ffff;
    let mut h = ((exp as u32) << 10) | (man >> 13);
    let rem = man & 0x1fff;
    // a carry out of the mantissa increments the exponent field, which is the correct
    // representation of the rounded value (and saturates to inf at the top)
    if rem > 0x1000 || (rem == 0x1000 && (h & 1) == 1) {
        h += 1;
    }
    ((sign | h) as u16).to_le_bytes()
}

// ---------------------------------------------------------------- candidates

/// A named implementation under qualification. `wrong: true` marks a deliberately-broken
/// arm that MUST fail — a harness that only ever runs the good one proves nothing.
pub struct Candidate {
    pub name: &'static str,
    pub wrong: bool,
    pub decode: fn(&Case) -> Option<Vec<u8>>,
}

fn role<'a>(c: &'a Case, name: &str) -> Option<&'a tensorfs_core::vectors::RoleBits> {
    c.roles.iter().find(|(n, _)| n == name).map(|(_, r)| r)
}

fn plain(c: &Case) -> Option<Vec<u8>> {
    Some(role(c, "value")?.bits.clone())
}

/// fp8-e4m3 per-output-row: value[i,j] = data[i,j] * scale[i]. ONE decoder serves both
/// physical variants — the scale's rank differs but its bytes do not, which is exactly the
/// claim the two spec digests make about each other.
fn fp8_rowwise(c: &Case) -> Option<Vec<u8>> {
    fp8_inner(c, false, false)
}
fn fp8_wrong_axis(c: &Case) -> Option<Vec<u8>> {
    fp8_inner(c, true, false)
}
fn fp8_no_scale(c: &Case) -> Option<Vec<u8>> {
    fp8_inner(c, false, true)
}

fn fp8_inner(c: &Case, wrong_axis: bool, skip_scale: bool) -> Option<Vec<u8>> {
    let d = role(c, "data")?;
    let s = role(c, "scale")?;
    let cols = *c.logical_shape.get(1)? as usize;
    let rows = *c.logical_shape.first()? as usize;
    let mut out = Vec::new();
    for i in 0..rows {
        for j in 0..cols {
            let q = f32_from_e4m3fn(d.bits[i * cols + j]);
            // the wrong arm indexes the scale by COLUMN — a per-input-channel reading of a
            // per-output-channel scale, which is plausible and silently wrong
            let k = if wrong_axis { j % rows } else { i };
            let sc = if skip_scale {
                1.0
            } else {
                let o = k * 4;
                f32::from_le_bytes(s.bits[o..o + 4].try_into().ok()?)
            };
            put_logical(c.logical_dtype, q * sc, &mut out);
        }
    }
    Some(out)
}

/// ComfyUI `scaled_fp8`, per-TENSOR scalar: value[i,j] = e4m3(data[i,j]) * weight_scale.
/// ONE decoder serves both physical variants, because the second scale is not a weight
/// scale: `input_scale` belongs to the activation quantizer and never touches these bytes.
fn fp8_scaled_scalar(c: &Case) -> Option<Vec<u8>> {
    fp8_scalar_inner(c, false, false, true)
}
/// The reader that treats the two role sets as one loose spec and folds the activation
/// scale into the weight. It gets the +input_scale cases numerically wrong and produces
/// NOTHING for the weight-scale-only ones — which is the "no optional roles" rule, live.
fn fp8_scaled_input_folded(c: &Case) -> Option<Vec<u8>> {
    fp8_scalar_inner(c, true, false, true)
}
fn fp8_scaled_no_scale(c: &Case) -> Option<Vec<u8>> {
    fp8_scalar_inner(c, false, true, true)
}
/// The plausible reading, and the one this harness was written with before the vectors
/// contradicted it: multiply in f32 and round ONCE at the end. ComfyUI narrows the scale
/// into the logical dtype FIRST, so this arm is a full ulp out on 132 of 512 bf16 elements
/// and bit-exact on every f32 one — wrong in exactly the way review can never catch.
fn fp8_scaled_f32_math(c: &Case) -> Option<Vec<u8>> {
    fp8_scalar_inner(c, false, false, false)
}

fn fp8_scalar_inner(c: &Case, fold_input: bool, skip_scale: bool, narrow: bool) -> Option<Vec<u8>> {
    let d = role(c, "data")?;
    let ws = role(c, "weight_scale")?;
    let mut s = if skip_scale {
        1.0
    } else {
        f32::from_le_bytes(ws.bits[0..4].try_into().ok()?)
    };
    if fold_input {
        // absent on the weight-scale-only spec: this arm cannot decode that contract at all
        let i = role(c, "input_scale")?;
        s *= f32::from_le_bytes(i.bits[0..4].try_into().ok()?);
    }
    if narrow {
        s = logical_round(c.logical_dtype, s);
    }
    let mut out = Vec::new();
    for b in &d.bits {
        put_logical(c.logical_dtype, f32_from_e4m3fn(*b) * s, &mut out);
    }
    Some(out)
}

/// MXFP8: value[i,j] = e4m3(data[i,j]) * 2^(E8M0(scale[i, j/32]) - 127)
fn mxfp8(c: &Case) -> Option<Vec<u8>> {
    mxfp8_inner(c, 32)
}
fn mxfp8_wrong_block(c: &Case) -> Option<Vec<u8>> {
    mxfp8_inner(c, 16)
}

fn mxfp8_inner(c: &Case, block: usize) -> Option<Vec<u8>> {
    let d = role(c, "data")?;
    let s = role(c, "scale")?;
    let rows = *c.logical_shape.first()? as usize;
    let cols = *c.logical_shape.get(1)? as usize;
    let blocks = s.shape.get(1).copied().unwrap_or(1) as usize;
    let mut out = Vec::new();
    for i in 0..rows {
        for j in 0..cols {
            let b = (j / block).min(blocks - 1);
            let e = f32_from_e8m0(s.bits[i * blocks + b]);
            put_logical(
                c.logical_dtype,
                f32_from_e4m3fn(d.bits[i * cols + j]) * e,
                &mut out,
            );
        }
    }
    Some(out)
}

/// nvfp4 W4A4: value[i,j] = e2m1(nibble) * e4m3(weight_scale[i, j/16]) * weight_scale_2
fn nvfp4(c: &Case) -> Option<Vec<u8>> {
    nvfp4_inner(c, false, false)
}
fn nvfp4_swapped_nibble(c: &Case) -> Option<Vec<u8>> {
    nvfp4_inner(c, true, false)
}
fn nvfp4_no_global(c: &Case) -> Option<Vec<u8>> {
    nvfp4_inner(c, false, true)
}

fn nvfp4_inner(c: &Case, swap: bool, skip_global: bool) -> Option<Vec<u8>> {
    let w = role(c, "weight")?;
    let ws = role(c, "weight_scale")?;
    let w2 = role(c, "weight_scale_2")?;
    let rows = *c.logical_shape.first()? as usize;
    let cols = *c.logical_shape.get(1)? as usize;
    let blocks = ws.shape.get(1).copied().unwrap_or(1) as usize;
    let g = if skip_global {
        1.0
    } else {
        f32::from_le_bytes(w2.bits[0..4].try_into().ok()?)
    };
    let mut out = Vec::new();
    for i in 0..rows {
        for j in 0..cols {
            let byte = w.bits[i * (cols / 2) + j / 2];
            // element 2k is the LOW nibble; the wrong arm reads them the other way round
            let even = (j % 2 == 0) != swap;
            let n = if even { byte & 0x0f } else { byte >> 4 };
            let bs = f32_from_e4m3fn(ws.bits[i * blocks + (j / 16).min(blocks - 1)]);
            put_logical(c.logical_dtype, f32_from_e2m1(n) * bs * g, &mut out);
        }
    }
    Some(out)
}

pub fn candidates(alias: &str) -> Vec<Candidate> {
    let mk = |name, wrong, decode| Candidate {
        name,
        wrong,
        decode,
    };
    match alias {
        "plain/1" => vec![mk("plain", false, plain as fn(&Case) -> Option<Vec<u8>>)],
        "fp8-rowwise/1" => vec![
            mk(
                "fp8-rowwise",
                false,
                fp8_rowwise as fn(&Case) -> Option<Vec<u8>>,
            ),
            mk("fp8-rowwise:scale-by-column", true, fp8_wrong_axis),
            mk("fp8-rowwise:scale-ignored", true, fp8_no_scale),
        ],
        "fp8-scaled-scalar/1" => vec![
            mk(
                "fp8-scaled-scalar",
                false,
                fp8_scaled_scalar as fn(&Case) -> Option<Vec<u8>>,
            ),
            mk(
                "fp8-scaled-scalar:input-scale-folded",
                true,
                fp8_scaled_input_folded,
            ),
            mk("fp8-scaled-scalar:scale-ignored", true, fp8_scaled_no_scale),
            mk("fp8-scaled-scalar:f32-math", true, fp8_scaled_f32_math),
        ],
        "mxfp8/1" => vec![
            mk("mxfp8", false, mxfp8),
            mk("mxfp8:block-16", true, mxfp8_wrong_block),
        ],
        "nvfp4-w4a4/1" => vec![
            mk("nvfp4", false, nvfp4),
            mk("nvfp4:nibbles-swapped", true, nvfp4_swapped_nibble),
            mk("nvfp4:global-scale-ignored", true, nvfp4_no_global),
        ],
        _ => vec![],
    }
}

// ---------------------------------------------------------------- the harness

fn qualify(alias: &str, spec: &EncodingSpec, v: &EncodingVectors) -> (usize, usize, bool) {
    let cands = candidates(alias);
    if cands.is_empty() {
        println!("  no candidate implementation registered for {alias}");
        return (0, 0, false);
    }
    let (mut good_pass, mut arms_failed_correctly) = (0usize, 0usize);
    let mut ok = true;
    for c in &cands {
        let mut pass = 0usize;
        let mut fail: Vec<String> = Vec::new();
        for case in &v.cases {
            match (c.decode)(case) {
                Some(got) if got == case.expect_logical_bits => pass += 1,
                Some(got) => fail.push(format!(
                    "{}: {} != expected {}",
                    case.name,
                    hexish(&got),
                    hexish(&case.expect_logical_bits)
                )),
                None => fail.push(format!("{}: decoder produced nothing", case.name)),
            }
        }
        let all = v.cases.len();
        if c.wrong {
            // A wrong arm must FAIL. One that passes means the vectors do not discriminate.
            if fail.is_empty() {
                println!(
                    "    {:<28} PASSED — but it is a WRONG arm: the vectors do not discriminate it",
                    c.name
                );
                ok = false;
            } else {
                arms_failed_correctly += 1;
                println!(
                    "    {:<28} refused {}/{} cases (correct — {})",
                    c.name,
                    fail.len(),
                    all,
                    fail.first().map(|s| truncate(s, 60)).unwrap_or_default()
                );
            }
        } else if fail.is_empty() {
            good_pass += 1;
            println!(
                "    {:<28} QUALIFIED {pass}/{all} cases bit-for-bit",
                c.name
            );
        } else {
            ok = false;
            println!("    {:<28} FAILED {}/{all}", c.name, fail.len());
            for f in fail.iter().take(3) {
                println!("        {f}");
            }
        }
    }
    let _ = spec;
    (good_pass, arms_failed_correctly, ok)
}

/// Byte-bounded, CHAR-safe. It used to slice at byte `n` and panicked the moment a refusal
/// line carried a multi-byte character within the first 60 bytes — which the fp8-scaled
/// arms' own elision marker does. A reporting path is not allowed to be the thing that dies.
fn truncate(s: &str, n: usize) -> String {
    match s.char_indices().nth(n) {
        None => s.to_string(),
        Some((i, _)) => format!("{}…", &s[..i]),
    }
}

fn hexish(b: &[u8]) -> String {
    let s: String = b.iter().take(12).map(|x| format!("{x:02x}")).collect();
    if b.len() > 12 {
        format!("{s}… ({} B)", b.len())
    } else {
        s
    }
}

pub fn cmd_conform(which: Option<&str>, _f: &Flags) -> ExitCode {
    let seeds = registry::seeds();
    let mut any = false;
    let mut bad = false;
    let (mut qualified, mut arms) = (0usize, 0usize);
    let mut vectorless: Vec<String> = Vec::new();

    for s in &seeds {
        if let Some(w) = which {
            if s.alias != w && !s.spec.object_id().contains(w) {
                continue;
            }
        }
        any = true;
        println!("{} {}", s.alias, s.spec.object_id());
        match (&s.vectors, s.spec.require_executable()) {
            (Some(v), Ok(r)) => {
                // The reference must be the bytes the spec COMMITS, not a twin beside it.
                let bytes = v.fixture_bytes();
                if tensorfs_core::ids::ObjectRef::of(&bytes) != *r {
                    println!("  VECTOR REFERENCE MISMATCH: the spec names other bytes");
                    bad = true;
                    continue;
                }
                // Geometry first: the ONE evaluator checks every case against the spec.
                match v.check(&s.spec) {
                    Ok(n) => println!("  geometry: {n} cases conform to the spec's roles"),
                    Err(e) => {
                        println!("  geometry REFUSED: {e}");
                        bad = true;
                        continue;
                    }
                }
                let (g, a, ok) = qualify(s.alias, &s.spec, v);
                qualified += g;
                arms += a;
                if !ok {
                    bad = true;
                }
            }
            (_, Err(e)) => {
                println!("  {e}");
                println!(
                    "  remedy: producer-mine raw role bytes + raw expected logical bytes and \
                     bank them as this spec's `vectors` ObjectRef — the spec digest changes, \
                     which is the point (an immutable object gained normative content)."
                );
                vectorless.push(s.spec.object_id());
            }
            (None, Ok(_)) => {
                println!("  spec names vectors the registry does not carry");
                bad = true;
            }
        }
    }
    if !any {
        eprintln!("no seed matches {which:?}");
        return ExitCode::from(2);
    }
    println!();
    println!(
        "{qualified} implementations QUALIFIED bit-for-bit, {arms} deliberately-wrong arms \
         correctly refused, {} vectorless specs refused executable qualification",
        vectorless.len()
    );
    if bad {
        return ExitCode::FAILURE;
    }
    ExitCode::SUCCESS
}

// ---------------------------------------------------------------- mining

/// Turn PRODUCER-MINED rows into the canonical vector document. The rows carry raw bytes
/// and nothing else; canonicalisation happens here, once, in the one writer.
pub fn cmd_mine(rows: &std::path::Path, name: &str, f: &Flags) -> ExitCode {
    let text = match std::fs::read_to_string(rows) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("{}: {e}", rows.display());
            return ExitCode::FAILURE;
        }
    };
    let want = flag(f, "match").unwrap_or("");
    let shape = |s: &str| -> Vec<u64> {
        if s.is_empty() {
            vec![]
        } else {
            s.split(',').filter_map(|d| d.parse().ok()).collect()
        }
    };

    let mut cases: Vec<Case> = Vec::new();
    let mut cur: Option<Case> = None;
    for line in text.lines() {
        let t: Vec<&str> = line.split_whitespace().collect();
        match t.first().copied() {
            Some("case") => {
                if let Some(c) = cur.take() {
                    cases.push(c);
                }
                cur = Some(Case {
                    name: t[1].to_string(),
                    logical_dtype: Dtype::F32,
                    logical_shape: vec![],
                    roles: vec![],
                    expect_logical_bits: vec![],
                });
            }
            Some("logical") => {
                if let Some(c) = cur.as_mut() {
                    c.logical_dtype = ok!(Dtype::parse(t[1]));
                    c.logical_shape = shape(t.get(2).copied().unwrap_or(""));
                }
            }
            Some("expect") => {
                if let Some(c) = cur.as_mut() {
                    c.expect_logical_bits = ok!(tensorfs_core::b64::decode(t[1]));
                }
            }
            Some("role") => {
                if let Some(c) = cur.as_mut() {
                    let (rname, dt) = (t[1].to_string(), ok!(Dtype::parse(t[2])));
                    // a rank-0 role prints an empty shape field, so the payload shifts left
                    let (sh, b64) = if t.len() >= 5 {
                        (shape(t[3]), t[4])
                    } else {
                        (vec![], t[3])
                    };
                    c.roles.push((
                        rname,
                        tensorfs_core::vectors::RoleBits {
                            dtype: dt,
                            shape: sh,
                            bits: ok!(tensorfs_core::b64::decode(b64)),
                        },
                    ));
                }
            }
            _ => {}
        }
    }
    if let Some(c) = cur.take() {
        cases.push(c);
    }
    cases.retain(|c| c.name.contains(want));
    for c in cases.iter_mut() {
        c.roles.sort_by(|a, b| a.0.cmp(&b.0));
    }
    if cases.is_empty() {
        eprintln!("no mined case matches {want:?}");
        return ExitCode::from(2);
    }
    let v = EncodingVectors { cases };
    let bytes = v.fixture_bytes();
    if let Err(e) = EncodingVectors::parse_fixture(&bytes) {
        return bail(e);
    }
    let d: std::path::PathBuf = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../vectors/mined")
        .components()
        .collect();
    if let Err(e) = std::fs::create_dir_all(&d) {
        eprintln!("{}: {e}", d.display());
        return ExitCode::FAILURE;
    }
    let path = d.join(format!("{name}.json"));
    if let Err(e) = std::fs::write(&path, &bytes) {
        eprintln!("{}: {e}", path.display());
        return ExitCode::FAILURE;
    }
    println!("mined {} ({} B)", path.display(), bytes.len());
    println!("  cases     {}", v.cases.len());
    println!(
        "  fixture_id {}",
        tensorfs_core::ids::object_id(&v.fixture_bytes())
    );
    for c in &v.cases {
        println!(
            "    {:<34} logical {} {:?}, roles {:?}",
            c.name,
            c.logical_dtype.name(),
            c.logical_shape,
            c.roles.iter().map(|(n, _)| n.as_str()).collect::<Vec<_>>()
        );
    }
    ExitCode::SUCCESS
}

/// `tfs capability <encoding> <device> [--alias <a>] [--records <path>] [--json]` — the ONE
/// admission
/// question, asked from outside the process.
///
/// `--json` prints one `capability_json` object on stdout and nothing else, and exits
/// SUCCESS on both verdicts: the answer is `admitted`. The column-aligned print
/// keeps its non-zero exit on a refusal and is for people; no program may match on it.
pub fn cmd_capability(encoding: &str, device: &str, f: &Flags) -> ExitCode {
    // The compiled-in records speak only for the reference decoders in this harness, on
    // the CPU they run on. TensorFS holds no accelerator and must never guess about one,
    // so an accelerator answer can only come from an OBSERVED record set handed in here by
    // the component that held the card. Absent one, an accelerator pair refuses -- which is
    // the absence of evidence answering, exactly as intended.
    let mut recs = registry::capability_records();
    if let Some(p) = flag(f, "records") {
        let bytes = match std::fs::read(p) {
            Ok(b) => b,
            Err(e) => {
                eprintln!("{p}: {e}");
                return ExitCode::from(2);
            }
        };
        let observed = match tensorfs_core::capability::CapabilityRecords::parse_observed(&bytes) {
            Ok(o) => o,
            Err(e) => return bail(e),
        };
        recs = recs.with_observed(observed);
    }
    let json = flag(f, "json").is_some();
    if !json {
        println!(
            "capability records: {} ({} qualified pairs)",
            recs.records.len(),
            recs.records.len()
        );
    }
    let enc = match flag(f, "alias") {
        Some(a) => match registry::seeds().iter().find(|s| s.alias == a) {
            Some(s) => s.spec.object_id(),
            None => {
                eprintln!("no seed aliased {a:?}");
                return ExitCode::from(2);
            }
        },
        // A spec id is prefixed BY DEFINITION, and the CLI hands every digest argument over
        // bare (`accept_printed_ids`), so put the one spelling back.
        None if tensorfs_core::ids::hex64("encoding", encoding).is_ok() => {
            format!("sha256:{encoding}")
        }
        None => encoding.to_string(),
    };
    let decision = recs.admit(&enc, device);
    if json {
        let qualified: Vec<&str> = recs
            .records
            .iter()
            .filter(|r| r.encoding == enc)
            .map(|r| r.device.as_str())
            .collect();
        let (record, refusal) = match &decision {
            Ok(r) => (Some(*r), String::new()),
            Err(e) => (None, e.detail.clone()),
        };
        println!(
            "{}",
            String::from_utf8_lossy(&tensorfs_core::machine::capability_json(
                record, &enc, device, &refusal, &qualified
            ))
        );
        return ExitCode::SUCCESS;
    }
    match decision {
        Ok(r) => {
            println!("ADMITTED ({enc}, {device})");
            println!("  implementation {}", r.implementation);
            println!("  qualified against vectors {}", r.vectors.id());
            ExitCode::SUCCESS
        }
        Err(e) => bail(e),
    }
}
