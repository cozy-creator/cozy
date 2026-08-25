//! Classification: what IS this carrier, and does a reviewed registry entry authorize it?
//!
//! Three separations this file exists to keep:
//!
//! * **Filenames never authorize.** Nothing here reads a path. A verdict comes from the KEY
//!   SET and the shapes, which are the only things a publisher cannot rename away.
//! * **A registry entry is platform-reviewed DATA, not uploader input.** Entries are banked
//!   documents with the same closed provenance rule as tfs-008's structure fixtures: a
//!   `real` claim without its source digest is an unfalsifiable claim and refuses.
//! * **Surviving ambiguity refuses loudly.** Two entries matching one key set is not a
//!   tiebreak, it is a review defect; the border names both and stops.

use crate::canon::{as_arr, as_obj, as_str, Fields, Value};
use crate::dtype::Dtype;
use crate::err::{refuse, Code, Result};
use crate::header::{topology_digest_of, topology_value};
use crate::ids::{ascii_name, check_format, hex64, object_id, Doc};
use crate::limits;
use crate::structure::{Provenance, REAL, SYNTHETIC};

use super::carrier::SourceHeader;

// ---------------------------------------------------------------- foreign quant markers

/// Suffixes that are ROLES of a neighbouring logical tensor, not tensors of their own.
pub const ROLE_SUFFIXES: [&str; 8] = [
    ".weight_scale",
    ".weight_scale_2",
    ".weight_scale_inv",
    ".scale_weight",
    ".scale_inv",
    ".input_scale",
    ".scale_input",
    ".pre_quant_scale",
];

/// Suffixes that are pure MARKERS: bytes whose only job is to tell a reader what the
/// neighbouring bytes mean. Under an explicit encoding digest they have nothing to say.
pub const MARKER_SUFFIXES: [&str; 2] = [".comfy_quant", ".quantization_metadata"];

/// The bnb reader is KILLED (tensorfs.md §12 q4). These are recognised only so the refusal
/// can name the remedy — no bnb quant state is ever decoded.
pub const BNB_SUFFIXES: [&str; 5] = [
    ".absmax",
    ".quant_map",
    ".quant_state",
    ".nested_absmax",
    ".nested_quant_map",
];

/// Split `<module>.weight_scale` into its MODULE and its role name. The module is what the
/// logical tensor `<module>.weight` is named after — the join is module-to-module, never
/// key-to-key, exactly as the v1 quarry writes it
/// (`scale_key = key[: -len(".weight")] + ".weight_scale"`).
pub fn role_of(key: &str) -> Option<(&str, &str)> {
    ROLE_SUFFIXES
        .iter()
        .find(|s| key.ends_with(**s))
        .map(|s| (&key[..key.len() - s.len()], &s[1..]))
}

/// The module a logical tensor belongs to, if it can carry companion roles at all. Only a
/// `.weight` has siblings; a bias or a norm is a tensor in its own right.
pub fn module_of(key: &str) -> Option<&str> {
    key.strip_suffix(".weight")
}

pub fn is_marker(key: &str) -> bool {
    MARKER_SUFFIXES.iter().any(|s| key.ends_with(*s))
}

pub fn bnb_key(key: &str) -> bool {
    BNB_SUFFIXES.iter().any(|s| key.ends_with(*s))
}

/// The one bnb refusal, stated once so every call site says the same thing.
pub fn refuse_bnb<T>(key: &str) -> Result<T> {
    refuse(
        Code::UNREGISTERED_ENCODING,
        format!(
            "{key}: bitsandbytes quant state. The bnb reader is KILLED (tensorfs.md §12 q4) — \
             the remedy is the fp16/bf16 base checkpoint through this same door, then the \
             quantize lane as a priced job. No bnb state is decoded to reach this verdict."
        ),
    )
}

// ---------------------------------------------------------------- the computed fingerprint

/// The observation, computed from the header alone. `keyset_digest` is deliberately
/// shape-free: it answers "which architecture is this", and stays stable across a family's
/// resolutions and precisions. `topology_digest` is the exact artifact, in the SAME digest
/// space as a banked structure fixture and a real header's projection.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Fingerprint {
    pub keyset_digest: String,
    pub topology_digest: String,
    pub logical_keys: usize,
    pub role_keys: usize,
    pub marker_keys: usize,
    pub dtypes: Vec<(Dtype, usize)>,
}

/// Project one carrier header. `component` is what the CALLER is ingesting this file as —
/// it participates in the topology projection exactly as it will in the final header, so a
/// source fingerprint and a canonical topology are directly comparable.
pub fn fingerprint(component: &str, h: &SourceHeader) -> Result<Fingerprint> {
    let mut logical: Vec<(&str, Dtype, &[u64])> = Vec::new();
    let (mut role_keys, mut marker_keys) = (0usize, 0usize);
    let mut dtypes: Vec<(Dtype, usize)> = Vec::new();

    for t in &h.tensors {
        if bnb_key(&t.key) {
            return refuse_bnb(&t.key);
        }
        match dtypes.iter_mut().find(|(k, _)| *k == t.dtype) {
            Some((_, n)) => *n += 1,
            None => dtypes.push((t.dtype, 1)),
        }
        if is_marker(&t.key) {
            marker_keys += 1;
        } else if role_of(&t.key).is_some() {
            role_keys += 1;
        } else {
            logical.push((t.key.as_str(), t.dtype, t.shape.as_slice()));
        }
    }
    if logical.is_empty() {
        return refuse(
            Code::MISSING_FIELD,
            "every key in this carrier is a role or a marker: there is no logical tensor",
        );
    }
    logical.sort_by(|a, b| a.0.cmp(b.0));
    dtypes.sort();

    let keys = Value::arr(logical.iter().map(|(k, _, _)| Value::str(*k)).collect());
    let topo = topology_value(logical.iter().map(|(k, d, s)| (component, *k, *d, *s)));

    Ok(Fingerprint {
        keyset_digest: object_id(&crate::canon::write(&keys)),
        topology_digest: topology_digest_of(&topo),
        logical_keys: logical.len(),
        role_keys,
        marker_keys,
        dtypes,
    })
}

// ---------------------------------------------------------------- the banked registry

/// One reviewed verdict. The REVIEW happened when this was banked; a hit on it IS profile
/// authorization for everything except a value-moving converter.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Banked {
    pub keyset_digest: String,
    pub dialect: String,
    pub task_family: String,
    pub structural_variant: String,
    pub component: String,
    /// The named reviewed converter this dialect binds to (see `convert::CONVERTERS`).
    pub converter: String,
    pub topology_digest: String,
    pub logical_keys: u64,
    pub provenance: Provenance,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct FingerprintRegistry {
    pub entries: Vec<Banked>,
}

/// What the border concluded. There is no fourth state and no default.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Verdict {
    /// Exactly one reviewed entry matched.
    Authorized(Box<Banked>),
    /// More than one matched: a review defect, refused loudly by name.
    Ambiguous(Vec<String>),
    /// Nothing matched. A filename, a directory, or a config would all be guesses.
    Unregistered,
}

impl FingerprintRegistry {
    pub fn classify(&self, fp: &Fingerprint) -> Verdict {
        let hits: Vec<&Banked> = self
            .entries
            .iter()
            .filter(|e| e.keyset_digest == fp.keyset_digest)
            .collect();
        match hits.len() {
            0 => Verdict::Unregistered,
            1 => Verdict::Authorized(Box::new(hits[0].clone())),
            _ => Verdict::Ambiguous(
                hits.iter()
                    .map(|e| {
                        format!(
                            "{}/{} via {}",
                            e.task_family, e.structural_variant, e.dialect
                        )
                    })
                    .collect(),
            ),
        }
    }

    /// The verdict as an authorization, with the two refusals spelled out at one site.
    pub fn authorize(&self, what: &str, fp: &Fingerprint) -> Result<Banked> {
        match self.classify(fp) {
            Verdict::Authorized(b) => Ok(*b),
            Verdict::Ambiguous(names) => refuse(
                Code::AMBIGUOUS_CLASSIFICATION,
                format!(
                    "{what}: key set {} matches {} reviewed entries {names:?} — surviving \
                     ambiguity refuses; an explicit profile must name exactly one",
                    fp.keyset_digest,
                    names.len()
                ),
            ),
            Verdict::Unregistered => refuse(
                Code::UNREGISTERED_FINGERPRINT,
                format!(
                    "{what}: key set {} ({} logical keys) matches no reviewed registry entry. \
                     A filename never authorizes; bank the fingerprint under review first.",
                    fp.keyset_digest, fp.logical_keys
                ),
            ),
        }
    }

    pub fn merge(&mut self, e: Banked) {
        match self
            .entries
            .iter_mut()
            .find(|x| x.keyset_digest == e.keyset_digest && x.component == e.component)
        {
            Some(slot) => *slot = e,
            None => self.entries.push(e),
        }
        self.entries.sort_by(|a, b| {
            (&a.keyset_digest, &a.component).cmp(&(&b.keyset_digest, &b.component))
        });
    }
}

fn provenance_value(p: &Provenance) -> Value {
    let mut v = vec![
        ("kind", Value::str(p.kind.clone())),
        ("note", Value::str(p.note.clone())),
        ("source", Value::str(p.source.clone())),
    ];
    if let Some(h) = &p.source_sha256 {
        v.push(("source_sha256", Value::str(h.clone())));
    }
    Value::obj(v)
}

fn provenance_from(v: &Value) -> Result<Provenance> {
    let mut pf = Fields::new("Provenance", v)?;
    let kind = pf.req_str("kind")?.to_string();
    let note = pf.req_str("note")?.to_string();
    let source = pf.req_str("source")?.to_string();
    let source_sha256 = match pf.opt("source_sha256") {
        Some(x) => Some(hex64(
            "Provenance.source_sha256",
            as_str("Provenance", "source_sha256", x)?,
        )?),
        None => None,
    };
    pf.done()?;
    match kind.as_str() {
        REAL if source_sha256.is_none() => refuse(
            Code::MISSING_FIELD,
            "provenance.kind \"real\" requires source_sha256: a registry entry claiming to \
             describe a real artifact without naming its bytes is unfalsifiable",
        ),
        REAL | SYNTHETIC => Ok(Provenance {
            kind,
            source,
            source_sha256,
            note,
        }),
        other => refuse(
            Code::UNKNOWN_FIELD,
            format!("provenance.kind {other:?} is outside {{{REAL}, {SYNTHETIC}}}"),
        ),
    }
}

impl Doc for FingerprintRegistry {
    const FORMAT: &'static str = "tensorfs.fingerprint_registry/1";
    const MAX_BYTES: usize = limits::DOC_MAX_BYTES;

    fn from_value(v: &Value) -> Result<Self> {
        let mut f = Fields::new("FingerprintRegistry", v)?;
        check_format(
            "FingerprintRegistry",
            as_str("FingerprintRegistry", "format", f.req("format")?)?,
            Self::FORMAT,
        )?;
        let mut entries = Vec::new();
        for ev in as_arr("FingerprintRegistry", "entries", f.req("entries")?)? {
            let mut e = Fields::new("Banked", ev)?;
            let component = e.req_str("component")?.to_string();
            let converter = e.req_str("converter")?.to_string();
            let dialect = e.req_str("dialect")?.to_string();
            let keyset_digest = crate::ids::prefixed("keyset_digest", e.req_str("keyset_digest")?)?;
            let logical_keys = e.req_uint("logical_keys")?;
            let provenance = provenance_from(e.req("provenance")?)?;
            let structural_variant = e.req_str("structural_variant")?.to_string();
            let task_family = e.req_str("task_family")?.to_string();
            let topology_digest =
                crate::ids::prefixed("topology_digest", e.req_str("topology_digest")?)?;
            e.done()?;
            for (what, s) in [
                ("component", &component),
                ("converter", &converter),
                ("dialect", &dialect),
                ("task_family", &task_family),
                ("structural_variant", &structural_variant),
            ] {
                ascii_name(what, s, limits::MAX_NAME_BYTES)?;
            }
            entries.push(Banked {
                keyset_digest,
                dialect,
                task_family,
                structural_variant,
                component,
                converter,
                topology_digest,
                logical_keys,
                provenance,
            });
        }
        f.done()?;
        let mut seen: Vec<(&str, &str)> = entries
            .iter()
            .map(|e| (e.keyset_digest.as_str(), e.component.as_str()))
            .collect();
        let given = seen.clone();
        seen.sort();
        if seen != given {
            return refuse(
                Code::SORT_ORDER,
                "registry entries must be sorted by (keyset_digest, component)",
            );
        }
        seen.dedup();
        if seen.len() != entries.len() {
            return refuse(
                Code::DUPLICATE_KEY,
                "two entries bank the same (keyset_digest, component)",
            );
        }
        Ok(FingerprintRegistry { entries })
    }

    fn to_value(&self) -> Value {
        Value::obj(vec![
            (
                "entries",
                Value::arr(
                    self.entries
                        .iter()
                        .map(|e| {
                            Value::obj(vec![
                                ("component", Value::str(e.component.clone())),
                                ("converter", Value::str(e.converter.clone())),
                                ("dialect", Value::str(e.dialect.clone())),
                                ("keyset_digest", Value::str(e.keyset_digest.clone())),
                                ("logical_keys", Value::uint(e.logical_keys)),
                                ("provenance", provenance_value(&e.provenance)),
                                (
                                    "structural_variant",
                                    Value::str(e.structural_variant.clone()),
                                ),
                                ("task_family", Value::str(e.task_family.clone())),
                                ("topology_digest", Value::str(e.topology_digest.clone())),
                            ])
                        })
                        .collect(),
                ),
            ),
            ("format", Value::str(Self::FORMAT)),
        ])
    }
}

/// Decoded `__metadata__` facts the border is willing to carry as EVIDENCE STAMPS. Foreign
/// JSON is decoded through this strict list or discarded — it never survives as itself.
pub fn evidence_stamps(h: &SourceHeader) -> Result<Vec<(String, String)>> {
    const KEPT: [&str; 4] = [
        "modelspec.architecture",
        "modelspec.prediction_type",
        "format",
        "scaled_fp8",
    ];
    let mut out = Vec::new();
    for (k, v) in &h.metadata {
        if (k == "quantization_config" || k == "bnb_quantization_config")
            && (v.contains("bitsandbytes") || v.contains("bnb"))
        {
            return refuse_bnb("__metadata__.quantization_config");
        }
        if KEPT.contains(&k.as_str()) {
            if v.len() > limits::MAX_NAME_BYTES {
                return refuse(
                    Code::SIZE_CAP,
                    format!("__metadata__.{k}: stamp value over the byte cap"),
                );
            }
            out.push((k.clone(), v.clone()));
        }
    }
    out.sort();
    Ok(out)
}

/// What a `.comfy_quant` marker actually says, decoded through the SAME bounded reader as
/// everything else. It never survives into the canonical header.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ComfyQuant {
    /// The carrier format the producer named, e.g. `float8_e4m3fn`.
    pub format: String,
    /// Set on the modules whose matmul stays in full precision. MEASURED as the fact that
    /// separates the two role sets: on the real fl2va DiT the 50 modules carrying it are
    /// exactly the `mlp.fc2` ones, and exactly the 50 with no `.input_scale`.
    pub full_precision_matrix_mult: bool,
}

/// Decode one marker payload.
///
/// VOCABULARY CORRECTED 2026-08-25 (decisions #267, tfs-018's pod arm). This decoder
/// required a `quant_algo` field and documented a 64-byte payload. The real payloads on
/// `Comfy-Org/MiniMax-H3@4cc1d817` are `{"format": "float8_e4m3fn"}` (150 markers, 27 B)
/// and `{"format": "float8_e4m3fn", "full_precision_matrix_mult": true}` (50, 63 B) — no
/// `quant_algo` anywhere. The old field was modelled from a description rather than from
/// bytes, and the artifact it was written for is the one it could not read.
pub fn decode_comfy_quant(key: &str, bytes: &[u8]) -> Result<ComfyQuant> {
    if bytes.len() > limits::MAX_DOC_TEXT_BYTES {
        return refuse(
            Code::SIZE_CAP,
            format!("{key}: marker payload {} B over the text cap", bytes.len()),
        );
    }
    // The payload is a NUL-padded JSON blob inside a U8 tensor. Take the text, then the one
    // bounded reader; every field but the two below is discarded by construction.
    let end = bytes.iter().position(|b| *b == 0).unwrap_or(bytes.len());
    let v = crate::canon::parse(&bytes[..end], limits::MAX_DOC_TEXT_BYTES)?;
    let fields = as_obj("comfy_quant", key, &v)?;
    let format = match fields.iter().find(|(k, _)| k == "format") {
        Some((_, x)) => as_str("comfy_quant", "format", x)?.to_string(),
        None => {
            return refuse(
                Code::MISSING_FIELD,
                format!("{key}: marker carries no `format` — nothing decodable into a role"),
            )
        }
    };
    let full_precision_matrix_mult = match fields
        .iter()
        .find(|(k, _)| k == "full_precision_matrix_mult")
    {
        Some((_, crate::canon::Value::Bool(b))) => *b,
        Some((_, o)) => {
            return refuse(
                Code::WRONG_TYPE,
                format!(
                    "{key}: full_precision_matrix_mult is a bool, got {}",
                    o.kind()
                ),
            )
        }
        None => false,
    };
    ascii_name("comfy_quant.format", &format, limits::MAX_NAME_BYTES)?;
    Ok(ComfyQuant {
        format,
        full_precision_matrix_mult,
    })
}
