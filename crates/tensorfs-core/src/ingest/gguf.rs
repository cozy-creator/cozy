//! The gguf-v1 PLANNER (tfs-011): what a serving lane would take, derived from a GGUF's
//! metadata region alone. Zero tensor bytes are read here and none need to exist.
//!
//! This is not a second ingest door and it does not widen the first one. `carrier::sniff`
//! still refuses `GGUF_REFUSED` on the normal path, and nothing in this file can produce a
//! candidate, a receipt or a manifest. What it produces is a PLAN — a statement of the two
//! views a lane gets and of exactly which tensors the platform can and cannot describe:
//!
//! 1. the RETAINED CARRIER BLOB — the whole `.gguf` as one opaque CAS object, which is
//!    what an engine that takes a path (`llama-server -m`) streams through
//!    `materialize_blob` (tensorfs.md §12 q3). Its length is stated here; its digest is an
//!    ingest fact and is not invented from a metadata probe.
//! 2. the DECOMPOSED PER-TENSOR VIEW — each tensor's exact byte range inside the data
//!    region, which serves dedup, inspection and derivation. Resynthesis (tfs-016) is not
//!    needed for serving because view 1 is retained whole.
//!
//! **Labels are never evidence.** A filename Q-label, `general.file_type` and
//! `general.architecture` are all producer strings; every fact below comes from the tensor
//! info table. The corpus proves why: `Abiray/MiniMax-H3-GGUF`'s H3 DiT declares
//! `general.architecture = "wan"`, and its `…-Q4_K_M.gguf` carries 250 Q4_K tensors and
//! zero Q5_K/Q6_K — the `_K_M` mixing policy its name states is not the one in the file,
//! while the text encoder shipped under the same label does implement it.
//!
//! **What is NOT built here** (tfs-011's blocked half): no ggml quant family maps to a
//! reviewed `EncodingSpec`, because a reviewed spec needs producer-mined vectors and a lane
//! that names the exact per-tensor encodings it executes, and no lane has selected any. The
//! plan therefore ends in `GGUF_ENCODING_NOT_REVIEWED` naming the families it met. That is
//! the honest terminal state of a planner whose consumers (se-007's llama-server arm,
//! job-008's diffusion producer) have not opened.

use std::collections::BTreeMap;
use std::fs::File;
use std::io::Read;
use std::path::Path;

use crate::canon::{self, as_arr, Fields, Value};
use crate::err::{refuse, Code, Refusal, Result};
use crate::limits;

/// The closed ggml type table: `(id, name, block elements, bytes per block)`. A type id
/// outside it refuses — a foreign enumerant is never guessed at, exactly as the safetensors
/// border's dtype table refuses an unknown spelling.
const GGML_TYPES: [(u32, &str, u64, u64); 31] = [
    (0, "F32", 1, 4),
    (1, "F16", 1, 2),
    (2, "Q4_0", 32, 18),
    (3, "Q4_1", 32, 20),
    (6, "Q5_0", 32, 22),
    (7, "Q5_1", 32, 24),
    (8, "Q8_0", 32, 34),
    (9, "Q8_1", 32, 40),
    (10, "Q2_K", 256, 84),
    (11, "Q3_K", 256, 110),
    (12, "Q4_K", 256, 144),
    (13, "Q5_K", 256, 176),
    (14, "Q6_K", 256, 210),
    (15, "Q8_K", 256, 292),
    (16, "IQ2_XXS", 256, 66),
    (17, "IQ2_XS", 256, 74),
    (18, "IQ3_XXS", 256, 98),
    (19, "IQ1_S", 256, 50),
    (20, "IQ4_NL", 32, 18),
    (21, "IQ3_S", 256, 110),
    (22, "IQ2_S", 256, 82),
    (23, "IQ4_XS", 256, 136),
    (24, "I8", 1, 1),
    (25, "I16", 1, 2),
    (26, "I32", 1, 4),
    (27, "I64", 1, 8),
    (28, "F64", 1, 8),
    (29, "IQ1_M", 256, 56),
    (30, "BF16", 1, 2),
    (31, "TQ1_0", 256, 54),
    (32, "TQ2_0", 256, 66),
    // Ids 4 and 5 are absent on purpose: they were Q4_2/Q4_3 and ggml REMOVED them. A file
    // carrying one refuses like any other unnamed id rather than being read as its
    // successor, which is what "closed table" has to mean to be worth anything.
];

/// The ggml types that ARE ordinary dtypes: one element per block, a direct cozytensors
/// spelling, and `plain/1` describes them. Everything else is a quant family with no
/// reviewed spec.
const PLAIN: [(&str, &str); 4] = [
    ("F32", "f32"),
    ("F16", "f16"),
    ("BF16", "bf16"),
    ("F64", "f64"),
];

const ORIG_SHAPE_PREFIX: &str = "comfy.gguf.orig_shape.";
/// GGUF's own default when `general.alignment` is absent.
const DEFAULT_ALIGNMENT: u64 = 32;
/// A metadata region is a bounded probe by design; this is the cap on one.
const METADATA_CAP: u64 = 64 << 20;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShapeSource {
    /// GGUF declares dimensions fastest-varying first; the logical shape is their reverse.
    DimsReversed,
    /// The producer reshaped the tensor before quantizing and recorded what it was. Without
    /// this key the original shape is NOT in the file — the reshaped dims are internally
    /// consistent and describe a different tensor.
    OrigShapeKv,
}

#[derive(Debug, Clone)]
pub struct GgufTensor {
    pub name: String,
    pub type_id: u32,
    pub type_name: &'static str,
    pub carrier_dims: Vec<u64>,
    pub logical_shape: Vec<u64>,
    pub shape_source: ShapeSource,
    pub elements: u64,
    pub blocks: u64,
    /// Relative to the data region, exactly as the carrier states it.
    pub offset: u64,
    pub nbytes: u64,
    /// The reviewed cozytensors spelling, when there is one. `None` is the blocked half.
    pub plain_dtype: Option<&'static str>,
}

#[derive(Debug, Clone)]
pub struct GgufPlan {
    /// The input was a bounded metadata probe: the plan is derived and byte presence was
    /// deliberately not checked.
    pub probe: bool,
    pub version: u32,
    pub alignment: u64,
    pub architecture: String,
    /// Producer LABELS, carried for the record and never consulted for a decision.
    pub file_type_label: Option<String>,
    pub metadata_bytes: u64,
    pub data_offset: u64,
    pub data_bytes: u64,
    pub orig_shape_kvs: u64,
    pub kv_count: u64,
    pub tensors: Vec<GgufTensor>,
}

impl GgufPlan {
    /// `(type name, tensors, carrier bytes)`, in table order — measured, never the label.
    pub fn type_histogram(&self) -> Vec<(&'static str, u64, u64)> {
        let mut m: BTreeMap<&'static str, (u64, u64)> = BTreeMap::new();
        for t in &self.tensors {
            let e = m.entry(t.type_name).or_insert((0, 0));
            e.0 += 1;
            e.1 += t.nbytes;
        }
        let mut v: Vec<_> = m.into_iter().map(|(k, (n, b))| (k, n, b)).collect();
        v.sort_by_key(|(k, _, _)| GGML_TYPES.iter().position(|(_, n, _, _)| n == k));
        v
    }

    /// The quant families no reviewed EncodingSpec describes, with their tensor counts.
    pub fn unreviewed(&self) -> Vec<(&'static str, u64)> {
        self.type_histogram()
            .into_iter()
            .filter(|(n, _, _)| !PLAIN.iter().any(|(g, _)| g == n))
            .map(|(n, c, _)| (n, c))
            .collect()
    }

    /// The whole carrier a path-consuming engine streams: metadata region plus data region.
    pub fn carrier_blob_bytes(&self) -> u64 {
        self.data_offset + self.data_bytes
    }

    /// The admission verdict this plan states about itself. A plan is not an admission and
    /// never becomes one by being detailed.
    pub fn admission(&self) -> Result<()> {
        match self.unreviewed().first() {
            None => Ok(()),
            Some((name, count)) => refuse(
                Code::GGUF_ENCODING_NOT_REVIEWED,
                format!(
                    "{count} tensors carry ggml {name} and no reviewed EncodingSpec describes \
                     it: a spec is a digest identity and needs producer-mined vectors plus a \
                     lane that names the exact per-tensor encodings it executes. The carrier \
                     facts above are real; admission is not open (tfs-011)"
                ),
            ),
        }
    }
}

// ---------------------------------------------------------------- bounded decode

struct Cursor<'a> {
    b: &'a [u8],
    o: usize,
}

impl<'a> Cursor<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        match self.b.len().checked_sub(self.o).filter(|left| *left >= n) {
            Some(_) => {
                let s = &self.b[self.o..self.o + n];
                self.o += n;
                Ok(s)
            }
            None => refuse(
                Code::GGUF_METADATA_TRUNCATED,
                format!(
                    "metadata region ends at {} and the next field needs {n} more bytes — a \
                     bounded probe that stops mid-table is a shorter probe, never a smaller file",
                    self.b.len()
                ),
            ),
        }
    }
    fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().expect("4")))
    }
    fn u64(&mut self) -> Result<u64> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().expect("8")))
    }
    fn str(&mut self) -> Result<String> {
        let n = self.u64()?;
        if n > limits::DOC_MAX_BYTES as u64 {
            return refuse(
                Code::GGUF_METADATA_TRUNCATED,
                format!("a metadata string claims {n} B — refused before allocating"),
            );
        }
        let s = self.take(n as usize)?;
        match std::str::from_utf8(s) {
            Ok(v) => Ok(v.to_string()),
            Err(e) => refuse(
                Code::GGUF_METADATA_TRUNCATED,
                format!("metadata string is not UTF-8: {e}"),
            ),
        }
    }
    /// One KV value. Only what a PLAN reads is kept: scalars become their decimal spelling,
    /// strings themselves, and arrays their element list. Nothing is interpreted.
    fn value(&mut self, t: u32) -> Result<String> {
        Ok(match t {
            0 => self.take(1)?[0].to_string(),
            1 => (self.take(1)?[0] as i8).to_string(),
            2 => u16::from_le_bytes(self.take(2)?.try_into().expect("2")).to_string(),
            3 => i16::from_le_bytes(self.take(2)?.try_into().expect("2")).to_string(),
            4 => self.u32()?.to_string(),
            5 => (self.u32()? as i32).to_string(),
            6 => f32::from_bits(self.u32()?).to_string(),
            7 => (self.take(1)?[0] != 0).to_string(),
            8 => self.str()?,
            9 => {
                let et = self.u32()?;
                let n = self.u64()?;
                let mut out = Vec::new();
                for _ in 0..n {
                    out.push(self.value(et)?);
                }
                out.join(",")
            }
            10 => self.u64()?.to_string(),
            11 => (self.u64()? as i64).to_string(),
            12 => f64::from_bits(self.u64()?).to_string(),
            other => {
                return refuse(
                    Code::GGUF_TYPE_UNKNOWN,
                    format!("metadata value type {other} is outside GGUF v3's closed table"),
                )
            }
        })
    }
}

fn ggml_type(id: u32) -> Result<(&'static str, u64, u64)> {
    match GGML_TYPES.iter().find(|(i, _, _, _)| *i == id) {
        Some((_, n, blk, sz)) => Ok((n, *blk, *sz)),
        None => refuse(
            Code::GGUF_TYPE_UNKNOWN,
            format!(
                "ggml type id {id} is outside the closed table — an unnamed quant family has \
                 no block geometry, so its byte length is not computable and a plan that \
                 guessed one would be fiction"
            ),
        ),
    }
}

/// Read the metadata region of a GGUF and derive the plan. Nothing below the tensor info
/// table is read, and `data_bytes` is what the table DECLARES, never what is present.
///
/// `probe` states what the input IS. A whole carrier must be at least as long as the plan
/// it declares, and a file that is shorter is a truncated download that would plan exactly
/// like a complete one — so it refuses. `probe = true` asserts the input is a bounded
/// metadata probe (the shape the evidence bank banks, and the shape a pre-download verdict
/// is made from), and then byte presence is not checked and the plan says so.
pub fn plan(path: &Path, probe: bool) -> Result<GgufPlan> {
    let mut f = File::open(path).map_err(|e| Refusal {
        code: Code::IO_FAILED,
        detail: format!("open {}: {e}", path.display()),
    })?;
    let mut buf = Vec::new();
    f.by_ref()
        .take(METADATA_CAP)
        .read_to_end(&mut buf)
        .map_err(|e| Refusal {
            code: Code::IO_FAILED,
            detail: format!("read {}: {e}", path.display()),
        })?;
    let mut c = Cursor { b: &buf, o: 0 };
    if c.take(4)? != b"GGUF" {
        return refuse(
            Code::GGUF_MAGIC_ABSENT,
            "no GGUF magic: this planner classifies on the bytes, never on the extension",
        );
    }
    let version = c.u32()?;
    if version != 3 {
        return refuse(
            Code::GGUF_VERSION_UNSUPPORTED,
            format!("GGUF v{version}: only v3 is decoded, and a version is not a dialect to guess"),
        );
    }
    let tensor_count = c.u64()?;
    let kv_count = c.u64()?;
    if tensor_count > limits::MAX_TENSORS as u64 {
        return refuse(
            Code::COUNT_CAP,
            format!(
                "{tensor_count} tensors declared, cap is {}",
                limits::MAX_TENSORS
            ),
        );
    }

    let mut kv: BTreeMap<String, String> = BTreeMap::new();
    for _ in 0..kv_count {
        let k = c.str()?;
        let t = c.u32()?;
        let v = c.value(t)?;
        if kv.insert(k.clone(), v).is_some() {
            return refuse(
                Code::DUPLICATE_KEY,
                format!("metadata key {k:?} appears twice — the last one would silently win"),
            );
        }
    }
    let architecture = match kv.get("general.architecture") {
        Some(a) => a.clone(),
        None => {
            return refuse(
                Code::GGUF_ARCH_UNDECLARED,
                "no `general.architecture`: an engine selects its graph by that key, and a \
                 carrier that does not state one cannot be planned for any lane",
            )
        }
    };
    // A KV namespaced to a DIFFERENT architecture than the declared one means two graphs
    // are described by one carrier. `general.*` and third-party namespaces are not
    // architecture scopes and do not count.
    let known_ns = [
        "general",
        "tokenizer",
        "split",
        "quantize",
        "comfy",
        "config",
    ];
    for k in kv.keys() {
        let ns = k.split('.').next().unwrap_or("");
        if ns != architecture && !known_ns.contains(&ns) {
            return refuse(
                Code::GGUF_ARCH_MIXED,
                format!(
                    "declared architecture {architecture:?} but key {k:?} is scoped to {ns:?} \
                     — one carrier describing two graphs has no single engine contract"
                ),
            );
        }
    }
    let alignment = match kv.get("general.alignment") {
        Some(a) => a.parse::<u64>().unwrap_or(0),
        None => DEFAULT_ALIGNMENT,
    };
    if alignment == 0 || !alignment.is_power_of_two() {
        return refuse(
            Code::GGUF_GEOMETRY,
            format!("`general.alignment` = {alignment}: not a power of two"),
        );
    }
    let orig: BTreeMap<&str, &str> = kv
        .iter()
        .filter_map(|(k, v)| k.strip_prefix(ORIG_SHAPE_PREFIX).map(|n| (n, v.as_str())))
        .collect();

    let mut tensors = Vec::with_capacity(tensor_count as usize);
    let mut end = 0u64;
    for _ in 0..tensor_count {
        let name = c.str()?;
        let n_dims = c.u32()?;
        if n_dims == 0 || n_dims > 4 {
            return refuse(
                Code::RANK_CAP,
                format!("{name}: GGUF declares {n_dims} dimensions; the format allows 1..=4"),
            );
        }
        let mut dims = Vec::with_capacity(n_dims as usize);
        for _ in 0..n_dims {
            dims.push(c.u64()?);
        }
        let type_id = c.u32()?;
        let offset = c.u64()?;
        let (type_name, blk, size) = ggml_type(type_id)?;

        let mut elements: u64 = 1;
        for d in &dims {
            elements = match elements.checked_mul(*d) {
                Some(v) => v,
                None => {
                    return refuse(
                        Code::ARITH_OVERFLOW,
                        format!("{name}: dimensions {dims:?} overflow u64"),
                    )
                }
            };
        }
        if !elements.is_multiple_of(blk) {
            return refuse(
                Code::DIVISION_REMAINDER,
                format!(
                    "{name}: {elements} elements is not a whole number of {type_name} blocks \
                     ({blk} elements each) — the tensor and its encoding disagree"
                ),
            );
        }
        let blocks = elements / blk;
        let nbytes = match blocks.checked_mul(size) {
            Some(v) => v,
            None => {
                return refuse(
                    Code::ARITH_OVERFLOW,
                    format!("{name}: byte length overflows"),
                )
            }
        };
        if !offset.is_multiple_of(alignment) {
            return refuse(
                Code::GGUF_GEOMETRY,
                format!("{name}: data offset {offset} is not a multiple of alignment {alignment}"),
            );
        }
        if offset < end {
            return refuse(
                Code::CARRIER_OVERLAP,
                format!(
                    "{name}: data offset {offset} is behind the previous tensor's end {end} — \
                     two tensors claiming the same bytes is a lie multiplication catches"
                ),
            );
        }
        // A reshaped-then-quantized tensor's ORIGINAL shape is not in the tensor table; the
        // producer's own KV is the only place it exists.
        let (logical_shape, shape_source) = match orig.get(name.as_str()) {
            Some(v) => {
                let mut s = Vec::new();
                for part in v.split(',') {
                    match part.trim().parse::<u64>() {
                        Ok(n) => s.push(n),
                        Err(_) => {
                            return refuse(
                                Code::WRONG_TYPE,
                                format!("{name}: {ORIG_SHAPE_PREFIX}… is not a list of integers"),
                            )
                        }
                    }
                }
                (s, ShapeSource::OrigShapeKv)
            }
            None => {
                let mut s = dims.clone();
                s.reverse();
                (s, ShapeSource::DimsReversed)
            }
        };
        let mut logical_elements: u64 = 1;
        for d in &logical_shape {
            logical_elements = logical_elements.saturating_mul(*d);
        }
        if logical_elements != elements {
            return refuse(
                Code::SHAPE_MISMATCH,
                format!(
                    "{name}: the recorded original shape {logical_shape:?} holds \
                     {logical_elements} elements and the carrier holds {elements}"
                ),
            );
        }
        end = offset + nbytes;
        tensors.push(GgufTensor {
            name,
            type_id,
            type_name,
            carrier_dims: dims,
            logical_shape,
            shape_source,
            elements,
            blocks,
            offset,
            nbytes,
            plain_dtype: PLAIN.iter().find(|(g, _)| *g == type_name).map(|(_, d)| *d),
        });
    }
    let metadata_bytes = c.o as u64;
    let data_offset = metadata_bytes.next_multiple_of(alignment);
    let declared = data_offset + end;
    if !probe {
        let on_disk = std::fs::metadata(path).map(|m| m.len()).unwrap_or(0);
        if on_disk < declared {
            return refuse(
                Code::GGUF_CARRIER_TRUNCATED,
                format!(
                    "the tensor table declares a {declared} B carrier and the file is \
                     {on_disk} B. A truncated download plans exactly like a complete one, \
                     which is why presence is checked and not assumed; pass the probe flag \
                     if this input is deliberately a bounded metadata probe"
                ),
            );
        }
    }
    Ok(GgufPlan {
        probe,
        version,
        alignment,
        architecture,
        file_type_label: kv.get("general.file_type").cloned(),
        metadata_bytes,
        data_offset,
        data_bytes: end,
        orig_shape_kvs: orig.len() as u64,
        kv_count,
        tensors,
    })
}

/// Check every planned logical shape against a NAMED unquantized twin — the same measured
/// rows the evidence transcriber emits. This is the projection-pairing arm: GGUF states
/// dimensions in reverse, and a producer may reshape a tensor before quantizing it, so a
/// carrier's own table can be internally consistent and still describe a different tensor
/// than the one the lane expects. A disagreement no `orig_shape` KV explains refuses.
pub fn check_twin(plan: &GgufPlan, rows: &[u8]) -> Result<u64> {
    let v = canon::parse(rows, limits::DOC_MAX_BYTES)?;
    let mut f = Fields::new("rows", &v)?;
    let _ = f.req_str("component")?;
    let _ = f.req_str("provenance")?;
    let mut twin: BTreeMap<String, Vec<u64>> = BTreeMap::new();
    for tv in as_arr("rows", "tensors", f.req("tensors")?)? {
        let mut tf = Fields::new("row", tv)?;
        let key = tf.req_str("key")?.to_string();
        let mut shape = Vec::new();
        for d in as_arr("row", "shape", tf.req("shape")?)? {
            match d {
                Value::Int(n) if *n >= 0 => shape.push(*n as u64),
                _ => return refuse(Code::WRONG_TYPE, format!("{key}: shape is not u64s")),
            }
        }
        twin.insert(key, shape);
    }
    let mut matched = 0u64;
    for t in &plan.tensors {
        let want = match twin.get(&t.name) {
            Some(s) => s,
            None => {
                return refuse(
                    Code::MISSING_TENSOR,
                    format!(
                        "{}: the carrier holds it and the named twin does not — the two \
                         artifacts do not have the same tensor schema",
                        t.name
                    ),
                )
            }
        };
        if want != &t.logical_shape {
            return refuse(
                Code::GGUF_SHAPE_UNRECOVERABLE,
                format!(
                    "{}: planned {:?} from {}, twin declares {want:?}. GGUF stores the \
                     reshaped-and-reversed carrier geometry; when the producer's \
                     `{ORIG_SHAPE_PREFIX}` key is absent the original shape is not in the \
                     file at all, and the carrier's own table stays internally consistent \
                     while describing a different tensor",
                    t.name,
                    t.logical_shape,
                    match t.shape_source {
                        ShapeSource::OrigShapeKv => "the orig_shape KV",
                        ShapeSource::DimsReversed => "reversed dims",
                    }
                ),
            );
        }
        matched += 1;
    }
    Ok(matched)
}
