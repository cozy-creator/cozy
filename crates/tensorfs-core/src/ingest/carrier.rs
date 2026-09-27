//! Strict bounded decode of a foreign SAFE carrier. Nothing here executes, imports, or
//! deserializes a publisher object graph: a carrier is a length, a JSON header, and a byte
//! region, and every one of those is checked against arithmetic before a buffer exists.
//!
//! Three properties this file is built to hold:
//!
//! 1. **Refuse before allocation.** The 8-byte length prefix is compared with the cap AND
//!    with the file's real length before `Vec::with_capacity` is ever reached. A header
//!    claiming 2^63 bytes costs one `read_exact` of 8 bytes.
//! 2. **The format's own grammar.** A foreign header is someone else's RFC 8259 JSON, read
//!    by the bounded, duplicate-key-refusing `jcs` reader: any UTF-8, any escape, any number
//!    inside the interoperable range. Only what TensorFS consumes is typed; foreign metadata
//!    it cannot carry is dropped, never a reason to refuse the carrier.
//! 3. **Geometry is arithmetic, not trust.** Every tensor's declared byte run must equal
//!    `checked_prod(shape) x sizeof(dtype)`, must lie inside the file, and must not overlap
//!    its neighbour. A lying header is refused by multiplication, never by a failed read.

use std::collections::BTreeMap;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

use crate::dtype::{checked_bytes, Dtype};
use crate::err::{refuse, Code, Refusal, Result};
use crate::jcs::{self, Json};
use crate::limits;

fn io(what: impl AsRef<str>, e: std::io::Error) -> Refusal {
    Refusal {
        code: Code::IO_FAILED,
        detail: format!("{}: {e}", what.as_ref()),
    }
}

fn kind(v: &Json) -> &'static str {
    match v {
        Json::Null => "null",
        Json::Bool(_) => "bool",
        Json::Num(_) => "number",
        Json::Str(_) => "string",
        Json::Arr(_) => "array",
        Json::Obj(_) => "object",
    }
}

fn object<'a>(what: &str, v: &'a Json) -> Result<&'a [(String, Json)]> {
    match v {
        Json::Obj(m) => Ok(m),
        o => refuse(
            Code::WRONG_TYPE,
            format!("{what}: expected object, got {}", kind(o)),
        ),
    }
}

fn array<'a>(what: &str, v: &'a Json) -> Result<&'a [Json]> {
    match v {
        Json::Arr(a) => Ok(a),
        o => refuse(
            Code::WRONG_TYPE,
            format!("{what}: expected array, got {}", kind(o)),
        ),
    }
}

fn string<'a>(what: &str, v: &'a Json) -> Result<&'a str> {
    match v {
        Json::Str(s) => Ok(s),
        o => refuse(
            Code::WRONG_TYPE,
            format!("{what}: expected string, got {}", kind(o)),
        ),
    }
}

fn uint(what: &str, v: &Json) -> Result<u64> {
    match v {
        Json::Num(n) if *n >= 0.0 && n.fract() == 0.0 => Ok(*n as u64),
        o => refuse(
            Code::WRONG_TYPE,
            format!("{what}: expected non-negative integer, got {}", kind(o)),
        ),
    }
}

fn field<'a>(what: &str, entry: &'a [(String, Json)], name: &str) -> Result<&'a Json> {
    entry
        .iter()
        .find(|(k, _)| k == name)
        .map(|(_, v)| v)
        .ok_or_else(|| Refusal {
            code: Code::MISSING_FIELD,
            detail: format!("{what}: missing {name:?}"),
        })
}

/// Foreign metadata is advisory. Strings and integers are carried as text; anything else is
/// dropped rather than refused.
fn metadata(v: &Json) -> Option<String> {
    match v {
        Json::Str(s) => Some(s.clone()),
        Json::Num(n) if *n >= 0.0 && n.fract() == 0.0 => Some((*n as u64).to_string()),
        _ => None,
    }
}

// ---------------------------------------------------------------- the closed dialect set

/// What a carrier CALLS its element types, mapped into the closed cozytensors enum. A
/// spelling outside this table refuses `DTYPE_UNKNOWN` — a foreign name is never guessed at.
const FOREIGN_DTYPES: [(&str, Dtype); 14] = [
    ("F64", Dtype::F64),
    ("F32", Dtype::F32),
    ("F16", Dtype::F16),
    ("BF16", Dtype::Bf16),
    ("F8_E4M3", Dtype::F8E4M3FN),
    ("F8_E4M3FN", Dtype::F8E4M3FN),
    ("F8_E5M2", Dtype::F8E5M2),
    ("I64", Dtype::I64),
    ("I32", Dtype::I32),
    ("I16", Dtype::I16),
    ("I8", Dtype::I8),
    ("U8", Dtype::U8),
    ("BOOL", Dtype::Bool),
    ("BYTES", Dtype::U8),
];

fn foreign_dtype(key: &str, name: &str) -> Result<Dtype> {
    match FOREIGN_DTYPES.iter().find(|(n, _)| *n == name) {
        Some((_, d)) => Ok(*d),
        None => refuse(
            Code::DTYPE_UNKNOWN,
            format!("{key}: carrier dtype {name:?} is outside the border's closed table"),
        ),
    }
}

// ---------------------------------------------------------------- magic sniff

/// What the first bytes SAY the file is. Observed, never opened: a pickle refusal that had
/// to unpickle to reach its verdict is not a refusal, it is an exploit with a log line.
///
/// Ordered so the strongest evidence wins. The safe carrier has no magic at all (it opens
/// with a little-endian length), so it is the fallthrough — which is why every hostile shape
/// must be named here explicitly rather than inferred from "not one of ours".
///
/// TWO STRENGTHS, TWO CALL SITES. The multi-byte magics (zip, GGUF, compressed) run before
/// anything else — a ≥2-byte collision with a sub-cap little-endian header length is not
/// arithmetically possible (the smallest such prefix reads as a length far over the cap).
/// The SINGLE-BYTE pickle opcodes are 1-in-64 collisions with a legitimate length prefix —
/// the real 66 GB official H3 diffusers DiT opens 0x28 = `(` because its shard-1 header is
/// 0x…28 bytes long, and the pre-parse check refused it as a pickle. They live in
/// `sniff_weak`, called ONLY on the failure paths (length over cap, truncated, or a header
/// that does not parse), where they upgrade the refusal's NAME without ever refusing a
/// carrier the arithmetic accepts.
pub fn sniff(head: &[u8]) -> Result<()> {
    // `PK\x01\x02` is the CENTRAL DIRECTORY header. A whole `.bin` opens with the local
    // header, but a bounded probe of one — the range fetch that reads a torchao pickle's
    // directory without pulling 30 GB, which is exactly what the H3 evidence bank does —
    // starts here. Without this magic such a fragment fell through to the length-prefix
    // arithmetic and refused CARRIER_HEADER_CAP: a refusal, but for the wrong reason
    // (tfs-010 stamp arm, on real banked bytes).
    if head.starts_with(b"PK\x03\x04")
        || head.starts_with(b"PK\x05\x06")
        || head.starts_with(b"PK\x01\x02")
    {
        return refuse(
            Code::ARCHIVE_REFUSED,
            "zip archive (a pickled `.pt`/`.bin` object graph): normal ingest opens no archive \
             and imports no publisher class; the remedy is a safe-carrier export of the \
             same weights",
        );
    }
    if head.starts_with(b"GGUF") {
        return refuse(
            Code::GGUF_REFUSED,
            "GGUF: consumer-gated, never admitted by the normal door (tensorfs.md §6)",
        );
    }
    if head.starts_with(b"\x1f\x8b") || head.starts_with(b"BZh") || head.starts_with(b"\xfd7zXZ") {
        return refuse(
            Code::ARCHIVE_REFUSED,
            "compressed stream: the border decompresses nothing (an unbounded expansion is \
             a resource claim the caller never made)",
        );
    }
    Ok(())
}

/// The weak single-byte pickle evidence — see `sniff`'s header comment. Called only where
/// the length-prefix arithmetic or the header parse has ALREADY refused: it can rename a
/// refusal, never create one.
fn sniff_weak(head: &[u8]) -> Result<()> {
    if head.first() == Some(&0x80) && matches!(head.get(1), Some(2..=5)) {
        return refuse(
            Code::PICKLE_REFUSED,
            "pickle PROTO opcode: refused on the magic, with no opcode interpreted and no \
             publisher class imported",
        );
    }
    if matches!(
        head.first(),
        Some(b'(') | Some(b']') | Some(b'}') | Some(b'c')
    ) {
        return refuse(
            Code::PICKLE_REFUSED,
            "protocol-0/1 pickle opening opcode: refused on the magic, nothing interpreted",
        );
    }
    Ok(())
}

// ---------------------------------------------------------------- decoded shape

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SourceTensor {
    pub key: String,
    pub dtype: Dtype,
    pub shape: Vec<u64>,
    /// Offsets are relative to the data region, exactly as the carrier states them.
    pub begin: u64,
    pub end: u64,
}

impl SourceTensor {
    pub fn nbytes(&self) -> u64 {
        self.end - self.begin
    }
}

/// One shard of a sharded carrier, as the joined view remembers it. `virtual_base` is where
/// this shard's data region begins in the joined VIRTUAL data region (the shards' regions
/// laid end to end in sorted-filename order); tensor offsets in the joined header are
/// virtual, and `resolve` maps them back to (path, absolute file offset) at read time.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Shard {
    pub path: std::path::PathBuf,
    pub data_start: u64,
    pub virtual_base: u64,
    pub virtual_len: u64,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SourceHeader {
    pub tensors: Vec<SourceTensor>,
    /// `__metadata__` — foreign JSON, decoded as flat strings only, never interpreted here.
    pub metadata: Vec<(String, String)>,
    pub header_bytes: u64,
    pub data_start: u64,
    pub file_len: u64,
    /// Declared bytes that no tensor claims. Recorded, not silently absorbed.
    pub gap_bytes: u64,
    /// Empty for a single-file carrier. Non-empty means tensor offsets are VIRTUAL and must
    /// go through `resolve` — never through `data_start` arithmetic on one file.
    pub shards: Vec<Shard>,
}

impl SourceHeader {
    pub fn declared_bytes(&self) -> u64 {
        self.tensors.iter().map(|t| t.nbytes()).sum()
    }
    pub fn get(&self, key: &str) -> Option<&SourceTensor> {
        self.tensors.iter().find(|t| t.key == key)
    }
    /// One O(n) index for callers that look keys up in a loop — a plan executor asking
    /// `get` per op is O(n²) at MAX_TENSORS and this is the constant-time door instead.
    pub fn index(&self) -> std::collections::HashMap<&str, &SourceTensor> {
        self.tensors.iter().map(|t| (t.key.as_str(), t)).collect()
    }
    pub fn meta(&self, key: &str) -> Option<&str> {
        self.metadata
            .iter()
            .find(|(k, _)| k == key)
            .map(|(_, v)| v.as_str())
    }
}

// ---------------------------------------------------------------- the decode

/// Decode one carrier's header from an open file. Reads at most
/// `8 + CARRIER_HEADER_MAX_BYTES` bytes and never touches the data region.
pub fn read_header(path: &Path) -> Result<SourceHeader> {
    let mut f = File::open(path).map_err(|e| io(format!("open {}", path.display()), e))?;
    let file_len = f.metadata().map_err(|e| io("stat", e))?.len();

    let mut head = [0u8; 16];
    let n = read_upto(&mut f, &mut head)?;
    sniff(&head[..n])?;
    if n < 8 {
        sniff_weak(&head[..n])?;
        return refuse(
            Code::CARRIER_TRUNCATED,
            format!("{n} bytes: too short to carry an 8-byte header length"),
        );
    }
    let declared = u64::from_le_bytes(head[0..8].try_into().unwrap());

    // Refuse BEFORE the allocation, against both the cap and the real file. Only when the
    // arithmetic refuses may the weak pickle bytes rename the refusal — a length the cap
    // and the file both accept is a carrier, whatever its low byte spells (the real H3
    // diffusers shard-1 header is 0x…28 B long, and 0x28 is pickle MARK).
    if declared > limits::CARRIER_HEADER_MAX_BYTES as u64 {
        sniff_weak(&head[..n])?;
        return refuse(
            Code::CARRIER_HEADER_CAP,
            format!(
                "header claims {declared} B, cap is {} B — refused before allocating",
                limits::CARRIER_HEADER_MAX_BYTES
            ),
        );
    }
    let data_start = match declared.checked_add(8) {
        Some(v) => v,
        None => return refuse(Code::ARITH_OVERFLOW, "header length + 8 overflows"),
    };
    if data_start > file_len {
        sniff_weak(&head[..n])?;
        return refuse(
            Code::CARRIER_TRUNCATED,
            format!("header claims {declared} B but the file holds {file_len} B in total"),
        );
    }

    f.seek(SeekFrom::Start(8)).map_err(|e| io("seek", e))?;
    let mut buf = vec![0u8; declared as usize];
    f.read_exact(&mut buf)
        .map_err(|e| io(format!("read {declared} header bytes"), e))?;

    parse_header(&buf, data_start, Some(file_len)).map_err(|e| {
        // A sub-cap length whose "header" is not carrier JSON: if the opening byte is a
        // pickle opcode, the honest name is pickle, not a JSON grammar refusal.
        match sniff_weak(&head[..n]) {
            Err(p) => p,
            Ok(()) => e,
        }
    })
}

/// The header bytes alone → the decoded, geometry-checked shape. Separated from IO so a
/// committed header (real evidence with no artifact behind it) decodes through the exact
/// same code.
///
/// `file_len` is `None` for that headerless-evidence case: the data region is then taken to
/// be exactly what the runs declare, which is the only honest reading when there is no file
/// to measure. Every other check is identical.
pub fn parse_header(bytes: &[u8], data_start: u64, file_len: Option<u64>) -> Result<SourceHeader> {
    let v = jcs::parse(bytes, limits::CARRIER_HEADER_MAX_BYTES)?;
    let entries = object("carrier header", &v)?;
    if entries.len() > limits::MAX_TENSORS {
        return refuse(
            Code::COUNT_CAP,
            format!(
                "carrier header holds {} keys, cap {}",
                entries.len(),
                limits::MAX_TENSORS
            ),
        );
    }

    let mut out = SourceHeader {
        header_bytes: data_start - 8,
        data_start,
        file_len: file_len.unwrap_or(data_start),
        ..Default::default()
    };

    for (key, ev) in entries {
        if key == "__metadata__" {
            if let Json::Obj(pairs) = ev {
                for (mk, mv) in pairs {
                    if let Some(value) = metadata(mv) {
                        out.metadata.push((mk.clone(), value));
                    }
                }
            }
            continue;
        }
        if key.is_empty() || key.len() > limits::MAX_KEY_BYTES {
            return refuse(
                Code::KEY_GRAMMAR,
                format!("carrier key length {} out of bounds", key.len()),
            );
        }
        // The one grammar limit that is TensorFS's own: a CozyTensors key is printable ASCII.
        if !key.bytes().all(|b| (0x20..=0x7e).contains(&b)) {
            return refuse(
                Code::KEY_GRAMMAR,
                format!(
                    "carrier key {key:?} is not printable ASCII; a CozyTensors key cannot name it"
                ),
            );
        }
        let f = object("carrier.tensor", ev)?;
        let offsets = array(
            "carrier.tensor.data_offsets",
            field("carrier.tensor", f, "data_offsets")?,
        )?;
        let dtype = foreign_dtype(
            key,
            string("carrier.tensor.dtype", field("carrier.tensor", f, "dtype")?)?,
        )?;
        let shape = array("carrier.tensor.shape", field("carrier.tensor", f, "shape")?)?
            .iter()
            .map(|x| uint("carrier.tensor.shape", x))
            .collect::<Result<Vec<u64>>>()?;

        if offsets.len() != 2 {
            return refuse(
                Code::CARRIER_GEOMETRY,
                format!("{key}: data_offsets holds {} values, not 2", offsets.len()),
            );
        }
        let begin = uint("carrier.tensor.data_offsets", &offsets[0])?;
        let end = uint("carrier.tensor.data_offsets", &offsets[1])?;
        if end < begin {
            return refuse(
                Code::CARRIER_GEOMETRY,
                format!("{key}: data_offsets [{begin}, {end}) runs backwards"),
            );
        }
        // The anchor equation, at the BORDER: what the geometry implies must be what the
        // file claims. This is the check that makes a truncated or doctored file arithmetic.
        let want = checked_bytes(key, &shape, dtype)?;
        if end - begin != want {
            return refuse(
                Code::CARRIER_GEOMETRY,
                format!(
                    "{key}: declared run {} B != {:?} x {} = {want} B",
                    end - begin,
                    shape,
                    dtype.name()
                ),
            );
        }
        out.tensors.push(SourceTensor {
            key: key.clone(),
            dtype,
            shape,
            begin,
            end,
        });
    }

    if out.tensors.is_empty() {
        return refuse(Code::MISSING_FIELD, "carrier declares no tensors");
    }

    let declared_end = out.tensors.iter().map(|t| t.end).max().unwrap_or(0);
    let region = match file_len {
        Some(f) => f - data_start,
        None => declared_end,
    };
    if declared_end > region {
        return refuse(
            Code::CARRIER_TRUNCATED,
            format!(
                "a run ends at {declared_end} but the data region holds {region} B \
                 (file {} B, data starts at {data_start})",
                out.file_len
            ),
        );
    }
    out.file_len = data_start + region;

    // Overlap refuses. Two keys sharing bytes is either corruption or an aliasing trick,
    // and neither has a meaning the border is willing to invent.
    let mut order: Vec<usize> = (0..out.tensors.len()).collect();
    order.sort_by_key(|i| out.tensors[*i].begin);
    let mut cursor = 0u64;
    for i in order {
        let t = &out.tensors[i];
        if t.begin < cursor {
            return refuse(
                Code::CARRIER_OVERLAP,
                format!(
                    "{}: run starts at {} inside the preceding run that ends at {cursor}",
                    t.key, t.begin
                ),
            );
        }
        out.gap_bytes += t.begin - cursor;
        cursor = t.end;
    }
    out.gap_bytes += region - cursor;
    // UNCLAIMED BYTES REFUSE. A safe carrier's data region is exactly the concatenation of
    // its declared runs — every real artifact measured here (SDXL's four components, 6.9 GB,
    // and the shipped H3 DiT header) reports zero unclaimed bytes, and the v1 quarry's own
    // strictest reader demanded the same gapless coverage. Bytes inside the file that no
    // key claims are a channel: they are ingested by nobody, verified by nothing, and
    // preserved by any tool that copies the file whole.
    if out.gap_bytes != 0 {
        return refuse(
            Code::CARRIER_GEOMETRY,
            format!(
                "{} B of the {region} B data region are claimed by no tensor — a safe \
                 carrier's region is exactly its declared runs, with no padding and no gaps",
                out.gap_bytes
            ),
        );
    }
    Ok(out)
}

// ---------------------------------------------------------------- sharded carriers

/// Where a sharded index's `weight_map` values live.
///
/// **A path is an OBJECT's identity, never a layout.** A fetched carrier sits at
/// `<root>/staging/<sha256>`, and that area is FLAT by design (tfs-067): no extension, no
/// directory of its own, and every other carrier of the run beside it under an equally
/// meaningless name. A `weight_map` value resolved against that parent finds nothing, and
/// every shard of every sharded model is unreachable. The shards of a store-resident index
/// are other MEMBERS of the same selection, each at its own hashed path — the selection is
/// the directory.
///
/// So the caller says which world it is in, and the same fact that named the carrier names
/// its shards: a carrier read by MEMBER resolves shards by member, a carrier read by FILE
/// NAME resolves them beside itself. One key, both questions, nothing to drift.
#[derive(Debug, Clone, Copy)]
pub enum Shards<'a> {
    /// Beside the index, by bare filename — a real directory tree, `tfs ingest` over a
    /// checkout. The value grammar is closed to bare sibling names: a traversal inside a
    /// `weight_map` is an exfiltration primitive, not a layout.
    Siblings,
    /// Other members of this same carrier set. Values join onto the index's own member the
    /// way the resolver joined them when it planned the fetch (`providers::join_member`),
    /// and the set is the whole authority: a name that is not a member of it opens nothing,
    /// so traversal is refused by absence rather than by grammar.
    Selection {
        /// The index's own member — the directory a relative value is joined onto.
        index: &'a str,
        /// member -> path over every carrier this plan may read.
        set: &'a BTreeMap<String, PathBuf>,
    },
}

/// The file one `weight_map` value names. The ONLY place a shard reference becomes a path.
fn locate_shard(index_path: &Path, shards: Shards<'_>, name: &str) -> Result<PathBuf> {
    match shards {
        Shards::Siblings => Ok(index_path
            .parent()
            .unwrap_or_else(|| Path::new("."))
            .join(name)),
        Shards::Selection { index, set } => {
            let Some(member) = crate::providers::join_member(index, name) else {
                return refuse(
                    Code::KEY_GRAMMAR,
                    format!("{index}: weight_map names {name:?}, which leaves the source"),
                );
            };
            match set.get(&member) {
                Some(path) => Ok(path.clone()),
                // The selection is short of what its own index requires. Either the record
                // owner narrowed to carriers without expanding this index into the shards it
                // names, or the index does not belong to the revision that was fetched. Name
                // the MEMBER: the path it would have had does not exist to be named.
                None => refuse(
                    Code::MISSING_FIELD,
                    format!(
                        "{index}: weight_map names shard {member:?}, which this carrier set \
                         does not contain"
                    ),
                ),
            }
        }
    }
}

/// Decode a sharded safe carrier through its `*.index.json`: strict bounded index decode,
/// every shard through `read_header` (so every single-file law — geometry, overlap,
/// gaplessness, caps — holds per shard), then EXACT COVER both directions: every
/// `weight_map` key must exist in exactly the shard the map names, and every tensor in
/// every shard must be claimed by the map. The joined header's tensor offsets are virtual
/// (shards' data regions laid end to end in sorted-filename order); the returned identity
/// bytes are the exact index bytes plus each shard's exact 8-byte prefix + header bytes in
/// that same order, so the source digest is content-bound and deterministic.
pub fn read_sharded(index_path: &Path, shards: Shards<'_>) -> Result<(SourceHeader, Vec<u8>)> {
    let index_bytes =
        std::fs::read(index_path).map_err(|e| io(format!("read {}", index_path.display()), e))?;
    let v = jcs::parse(&index_bytes, limits::CARRIER_HEADER_MAX_BYTES)?;
    let top = object("shard index", &v)?;
    let mut weight_map: Option<&[(String, Json)]> = None;
    let mut index_meta: Vec<(String, String)> = Vec::new();
    for (k, ev) in top {
        match (k.as_str(), ev) {
            ("weight_map", _) => weight_map = Some(object("shard index weight_map", ev)?),
            ("metadata", Json::Obj(pairs)) => {
                for (mk, mv) in pairs {
                    if let Some(value) = metadata(mv) {
                        index_meta.push((mk.clone(), value));
                    }
                }
            }
            _ => {}
        }
    }
    let weight_map = match weight_map {
        Some(m) if !m.is_empty() => m,
        _ => {
            return refuse(
                Code::MISSING_FIELD,
                "shard index: empty or missing weight_map",
            )
        }
    };

    // key -> shard filename. A MAP, not a scan-per-tensor: find/dedup/cover below are each
    // per-tensor over MAX_TENSORS=100,000, and a linear probe there is O(n²) at design
    // cardinality.
    let mut by_key: std::collections::HashMap<&str, &str> =
        std::collections::HashMap::with_capacity(weight_map.len());
    let mut names: Vec<&str> = Vec::new();
    for (k, v) in weight_map {
        let name = string("shard index weight_map value", v)?;
        if name.is_empty() {
            return refuse(
                Code::KEY_GRAMMAR,
                format!("shard index: weight_map key {k:?} names no shard"),
            );
        }
        // The grammar belongs to the layout, not to the file. Beside the index the value
        // must be a bare sibling name, because there a traversal is an exfiltration
        // primitive. Inside a selection the carrier set is the whole authority and a name
        // that is not a member of it opens nothing, so the same reference is refused by
        // absence instead -- and relative references stay legal, exactly as the resolver
        // read them when it planned the fetch.
        if matches!(shards, Shards::Siblings)
            && (name.contains('/') || name.contains('\\') || name.contains(".."))
        {
            return refuse(
                Code::KEY_GRAMMAR,
                format!("shard index: shard name {name:?} is not a bare sibling filename"),
            );
        }
        by_key.insert(k.as_str(), name);
        if !names.contains(&name) {
            names.push(name);
        }
    }
    names.sort_unstable();

    // Locate every shard BEFORE opening any of them. An incomplete selection is decided by
    // metadata alone, and the whole point of reading headers first is that such a refusal
    // costs bytes measured in kilobytes -- never a run that streamed 210 GB and then found
    // out.
    let mut located: Vec<(&str, PathBuf)> = Vec::with_capacity(names.len());
    for name in &names {
        located.push((name, locate_shard(index_path, shards, name)?));
    }

    let mut out = SourceHeader::default();
    let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
    let mut raw = (index_bytes.len() as u64).to_le_bytes().to_vec();
    raw.extend_from_slice(&index_bytes);
    let mut virtual_base = 0u64;
    for (name, path) in &located {
        let h = read_header(path)?;
        let mut f = File::open(path).map_err(|e| io(format!("reopen {}", path.display()), e))?;
        let mut hb = vec![0u8; h.header_bytes as usize + 8];
        f.read_exact(&mut hb)
            .map_err(|e| io(format!("reread header of {}", path.display()), e))?;
        raw.extend_from_slice(&hb);

        let region = h.file_len - h.data_start;
        for t in &h.tensors {
            let claimed = by_key.get(t.key.as_str());
            match claimed {
                Some(n) if n == name => {}
                Some(n) => {
                    return refuse(
                        Code::CARRIER_GEOMETRY,
                        format!(
                            "{}: carried by shard {name:?} but the weight_map names {n:?}",
                            t.key
                        ),
                    )
                }
                None => {
                    return refuse(
                        Code::CARRIER_GEOMETRY,
                        format!(
                            "{}: carried by shard {name:?} but claimed by no weight_map key",
                            t.key
                        ),
                    )
                }
            }
            if !seen.insert(t.key.clone()) {
                return refuse(
                    Code::CARRIER_GEOMETRY,
                    format!("{}: appears in more than one shard", t.key),
                );
            }
            out.tensors.push(SourceTensor {
                key: t.key.clone(),
                dtype: t.dtype,
                shape: t.shape.clone(),
                begin: virtual_base + t.begin,
                end: virtual_base + t.end,
            });
        }
        // Advisory metadata that differs across shards is kept as both values, never refused.
        for pair in &h.metadata {
            if !out.metadata.contains(pair) {
                out.metadata.push(pair.clone());
            }
        }
        out.header_bytes += h.header_bytes;
        out.shards.push(Shard {
            path: path.clone(),
            data_start: h.data_start,
            virtual_base,
            virtual_len: region,
        });
        virtual_base += region;
    }

    // The other half of exact cover: every weight_map key was found in some shard.
    let mut missing: Vec<(&str, &str)> = by_key
        .iter()
        .filter(|(k, _)| !seen.contains(**k))
        .map(|(k, n)| (*k, *n))
        .collect();
    missing.sort_unstable();
    if let Some((k, name)) = missing.first() {
        return refuse(
            Code::CARRIER_GEOMETRY,
            format!("{k}: named by the weight_map in shard {name:?} but carried by no shard"),
        );
    }
    for (mk, mv) in index_meta {
        if !out.metadata.iter().any(|(k, _)| *k == mk) {
            out.metadata.push((mk, mv));
        }
    }
    out.data_start = 0;
    out.file_len = virtual_base;
    Ok((out, raw))
}

/// Map one tensor of a header to (file path, absolute begin offset in that file). For a
/// single-file carrier this is `data_start + begin` in the carrier itself; for a sharded
/// one it finds the shard whose virtual window holds the run. A run crossing a shard
/// boundary cannot exist by construction and refuses rather than being read wrong.
pub fn resolve<'a>(
    default_path: &'a Path,
    h: &'a SourceHeader,
    t: &SourceTensor,
) -> Result<(&'a Path, u64)> {
    if h.shards.is_empty() {
        return Ok((default_path, h.data_start + t.begin));
    }
    for s in &h.shards {
        if t.begin >= s.virtual_base && t.end <= s.virtual_base + s.virtual_len {
            return Ok((&s.path, s.data_start + (t.begin - s.virtual_base)));
        }
    }
    refuse(
        Code::CARRIER_GEOMETRY,
        format!(
            "{}: virtual run [{}, {}) lies in no shard's window",
            t.key, t.begin, t.end
        ),
    )
}

fn read_upto(f: &mut File, buf: &mut [u8]) -> Result<usize> {
    let mut n = 0;
    while n < buf.len() {
        match f.read(&mut buf[n..]).map_err(|e| io("read magic", e))? {
            0 => break,
            k => n += k,
        }
    }
    Ok(n)
}

// ---------------------------------------------------------------- bounded region reader

/// A reader over ONE tensor's declared byte run. Bounded by the declaration, positioned
/// once, and never allowed to read past the run — the writer downstream streams from this
/// and therefore can never be handed more bytes than the geometry promised.
pub struct Region {
    f: File,
    left: u64,
}

/// A reader over ONE MEMBER of a fused tensor, taken from the carrier in DESTINATION order,
/// one contiguous run at a time.
///
/// This is what keeps a permutation-class converter inside the border's constant-memory
/// claim. The decomposition is arithmetic (`crate::fill`), so every run's source offset is
/// known before a byte moves — there is no reason to hold the fused tensor. Buffering it
/// would make peak RSS a function of the artifact: the real H3 `attn.qkv_proj.weight` is
/// 231,211,008 B, which is 33x the whole SDXL ingest's measured peak. Here the resident cost
/// is ONE RUN — 1,376,256 B on that same tensor — and the writer downstream cannot be handed
/// more bytes than the geometry promised, exactly as with `Region`.
pub struct RunRegion {
    f: File,
    base: u64,
    runs: std::vec::IntoIter<crate::fill::Run>,
    left: u64,
}

impl RunRegion {
    /// `abs_base` is the tensor's ABSOLUTE begin offset in `path` — the caller resolves it
    /// through `resolve`, so sharded and single-file carriers read through one body.
    pub fn open(path: &Path, abs_base: u64, d: crate::fill::Decomposition) -> Result<RunRegion> {
        let f = File::open(path).map_err(|e| io(format!("open {}", path.display()), e))?;
        Ok(RunRegion {
            f,
            base: abs_base,
            runs: d.runs.into_iter(),
            left: 0,
        })
    }
}

impl Read for RunRegion {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        while self.left == 0 {
            match self.runs.next() {
                None => return Ok(0),
                Some(r) => {
                    self.f.seek(SeekFrom::Start(self.base + r.src_off))?;
                    self.left = r.len;
                }
            }
        }
        let cap = (buf.len() as u64).min(self.left) as usize;
        let n = self.f.read(&mut buf[..cap])?;
        self.left -= n as u64;
        Ok(n)
    }
}

impl Region {
    /// `abs_begin` is the tensor's ABSOLUTE begin offset in `path` (see `resolve`).
    pub fn open(path: &Path, abs_begin: u64, nbytes: u64) -> Result<Region> {
        let mut f = File::open(path).map_err(|e| io(format!("open {}", path.display()), e))?;
        f.seek(SeekFrom::Start(abs_begin))
            .map_err(|e| io("seek to run", e))?;
        Ok(Region { f, left: nbytes })
    }
}

impl Read for Region {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        if self.left == 0 {
            return Ok(0);
        }
        let cap = (buf.len() as u64).min(self.left) as usize;
        let n = self.f.read(&mut buf[..cap])?;
        self.left -= n as u64;
        Ok(n)
    }
}
