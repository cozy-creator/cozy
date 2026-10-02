//! Classification: what IS this carrier, and which reviewed converter and profile fit it?
//! The registry is a HINT: a source it does not recognize is stored as it is
//! (`source::AS_IS`), never refused for being unknown.
//!
//! Three separations this file exists to keep:
//!
//! * **Filenames never authorize.** Nothing here reads a path. A verdict comes from the KEY
//!   SET and the shapes, which are the only things a publisher cannot rename away.
//! * **A registry entry is platform-reviewed DATA, not uploader input.** Entries carry a
//!   closed provenance: a `real` claim without its source digest is an unfalsifiable claim
//!   and refuses.
//! * **Surviving ambiguity refuses loudly.** Two entries matching one carrier is not a
//!   tiebreak, it is a review defect; the border names both and stops.
//! * **A key set does not name a component.** MiniMax-H3 ships `transformer` and
//!   `transformer_ref` as 532 identical keys with BYTE-IDENTICAL headers — one cached
//!   header serves both files — so the key set can never decide which one a carrier is.
//!   The component the caller declares participates in the match, and a key set banked
//!   only under other components refuses by naming them.

use super::carrier::SourceHeader;
use crate::canon::{as_arr, as_str, Fields, Value};
use crate::dtype::Dtype;
use crate::err::{refuse, Code, Result};
use crate::header::{tensor_schema_digest_of, tensor_schema_value};
use crate::ids::{ascii_name, hex64, object_id, prefixed};
use crate::limits;

/// Where a banked entry's shapes came from. Closed: `real` or `synthetic`, never absent.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Provenance {
    pub kind: String,
    /// Human-readable name of the source artifact.
    pub source: String,
    /// sha256 of the exact source bytes. REQUIRED when `kind` is `real`.
    pub source_sha256: Option<String>,
    /// What was derived rather than read — the honest part.
    pub note: String,
}

pub const REAL: &str = "real";
pub const SYNTHETIC: &str = "synthetic";

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
/// shape-free: it answers "which carrier dialect is this", and stays stable across a
/// family's resolutions and precisions. `tensor_schema_digest` is the exact artifact, in the
/// same digest space as a real header's projection. Neither is a compatibility claim —
/// that is `fit`'s question, asked of the canonical header, never of a source carrier.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Fingerprint {
    /// What the caller is ingesting this carrier AS. Carried on the observation rather than
    /// passed to `classify` separately, so a fingerprint can never be classified as a
    /// component other than the one its `tensor_schema_digest` was projected under.
    pub component: String,
    pub keyset_digest: String,
    pub tensor_schema_digest: String,
    pub logical_keys: usize,
    pub role_keys: usize,
    pub marker_keys: usize,
    pub dtypes: Vec<(Dtype, usize)>,
}

/// Project one carrier header. `component` is what the CALLER is ingesting this file as —
/// it participates in the tensor schema exactly as it will in the final header, so a source
/// fingerprint and a canonical tensor schema are directly comparable.
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
    let schema = tensor_schema_value(logical.iter().map(|(k, d, s)| (component, *k, *d, *s)));

    Ok(Fingerprint {
        component: component.to_string(),
        keyset_digest: object_id(&crate::canon::write(&keys)),
        tensor_schema_digest: tensor_schema_digest_of(&schema),
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
    pub component: String,
    /// The named reviewed converter this dialect binds to (see `convert::CONVERTERS`).
    pub converter: String,
    pub tensor_schema_digest: String,
    pub logical_keys: u64,
    pub provenance: Provenance,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct FingerprintRegistry {
    pub entries: Vec<Banked>,
    /// Reviewed whole-source recipes. Fingerprints authorize individual component
    /// views; a source profile is the additional review fact that says which exact
    /// components form one artifact, their target encoding, and construction order.
    pub source_profiles: Vec<SourceProfile>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SourceProfile {
    pub name: String,
    /// Whether headers alone may choose this format. False requires the caller
    /// to declare the reviewed format; shapes cannot prove semantic row layout.
    pub auto_select: bool,
    /// Set when a converter's planner, not banked key sets, recognizes the source.
    pub grammar: Option<Grammar>,
    pub components: Vec<SourceProfileComponent>,
}

/// A carrier matches when every key starts with `key_prefix` and the reviewed `converter`
/// plans it: the planner is the key grammar, so any valid subset of its reviewed modules
/// matches without a banked key list. Its keys construct in sorted order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Grammar {
    pub converter: String,
    pub key_prefix: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SourceProfileComponent {
    pub component: String,
    /// Optional reviewed provider/source slot. Required only when carrier bytes
    /// cannot distinguish components (for example H3's identical DiT headers).
    pub source_member: Option<String>,
    /// Optional reviewed provider namespace for structurally interchangeable members.
    pub source_member_prefix: Option<String>,
    pub target_encoding: String,
    pub variants: Vec<SourceProfileVariant>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SourceProfileVariant {
    pub keyset_digest: String,
    pub construction_order: Vec<String>,
}

/// What the border concluded. There is no fifth state and no default.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Verdict {
    /// Exactly one reviewed entry matched (key set AND component).
    Authorized(Box<Banked>),
    /// The key set is reviewed, but never under the component the caller declared. The key
    /// set ALONE cannot decide — it is banked under the components named here — so this is
    /// a typed ambiguity, not an absence: the remedy is a review act, not a rename.
    ComponentAmbiguous {
        /// `<keyset> as <component> via <dialect>` for every entry on this key set.
        banked_as: Vec<String>,
    },
    /// More than one entry banks the same (key set, component): a review defect, refused
    /// loudly by name. Parsed registry data cannot contain this (the parser refuses a
    /// duplicated pair); an in-memory one built by hand can.
    Ambiguous(Vec<String>),
    /// Nothing matched. A filename, a directory, or a config would all be guesses.
    Unregistered,
}

fn name_of(e: &Banked) -> String {
    format!(
        "{} as {} via {}",
        &e.keyset_digest[..19],
        e.component,
        e.dialect
    )
}

impl FingerprintRegistry {
    /// Read the platform's reviewed registry data. This is configuration consumed by one
    /// TensorFS implementation, not a stored/digested product document, so it has no format
    /// field and no ObjectId.
    pub fn parse(bytes: &[u8]) -> Result<Self> {
        Self::from_value(&crate::canon::parse(bytes, limits::DOC_MAX_BYTES)?)
    }

    /// Stable output for `tfs ingest bank`; not an identity-bearing document encoding.
    pub fn to_bytes(&self) -> Vec<u8> {
        crate::canon::write(&self.to_value())
    }

    /// Extend this registry with a newer reviewed one: its entries and profiles replace
    /// these under the same key and add the rest. Every entry keeps its own provenance.
    pub fn extend(&mut self, newer: FingerprintRegistry) {
        for entry in newer.entries {
            self.merge(entry);
        }
        for profile in newer.source_profiles {
            self.merge_source_profile(profile);
        }
    }

    /// Key set and component classify. An entry banked from the exact same tensor schema is
    /// preferred; otherwise any entry for this key set and component serves when they agree
    /// on dialect and converter, and conversion validates the actual shapes and dtypes (an
    /// fp16 upload of a model banked in bf16 classifies).
    pub fn classify(&self, fp: &Fingerprint) -> Verdict {
        let keyset: Vec<&Banked> = self
            .entries
            .iter()
            .filter(|e| e.keyset_digest == fp.keyset_digest)
            .collect();
        if keyset.is_empty() {
            return Verdict::Unregistered;
        }
        let component: Vec<&Banked> = keyset
            .iter()
            .copied()
            .filter(|e| e.component == fp.component)
            .collect();
        if component.is_empty() {
            return Verdict::ComponentAmbiguous {
                banked_as: keyset.iter().map(|e| name_of(e)).collect(),
            };
        }
        let exact: Vec<&Banked> = component
            .iter()
            .copied()
            .filter(|e| {
                e.tensor_schema_digest == fp.tensor_schema_digest
                    && e.logical_keys as usize == fp.logical_keys
            })
            .collect();
        let hits = if exact.is_empty() {
            let mut verdicts = component.clone();
            verdicts.sort_by(|a, b| (&a.dialect, &a.converter).cmp(&(&b.dialect, &b.converter)));
            verdicts.dedup_by(|a, b| (&a.dialect, &a.converter) == (&b.dialect, &b.converter));
            verdicts
        } else {
            exact
        };
        match hits.len() {
            0 => Verdict::Unregistered,
            1 => Verdict::Authorized(Box::new(hits[0].clone())),
            _ => Verdict::Ambiguous(hits.iter().map(|e| name_of(e)).collect()),
        }
    }

    /// Return the exact provider members named by a set of reviewed source profiles.
    ///
    /// This is deliberately a metadata-only query: callers can narrow a provider's
    /// file inventory before any model body is transferred. Prefix-bound or unbound
    /// components cannot name an exact carrier without inspecting the provider
    /// inventory, so they refuse instead of widening the download.
    pub fn exact_source_members(&self, profile_names: &[String]) -> Result<Vec<String>> {
        if profile_names.is_empty() {
            return refuse(Code::MISSING_FIELD, "no source profile requested");
        }
        let mut requested: Vec<&str> = profile_names.iter().map(String::as_str).collect();
        requested.sort();
        if requested.windows(2).any(|window| window[0] == window[1]) {
            return refuse(Code::DUPLICATE_KEY, "source profiles must be unique");
        }

        let mut members = Vec::new();
        for name in requested {
            let Some(profile) = self
                .source_profiles
                .iter()
                .find(|profile| profile.name == name)
            else {
                return refuse(
                    Code::UNREGISTERED_FINGERPRINT,
                    format!("reviewed source profile {name:?} is absent"),
                );
            };
            for component in &profile.components {
                match (&component.source_member, &component.source_member_prefix) {
                    (Some(member), None) => members.push(member.clone()),
                    (None, Some(prefix)) => {
                        return refuse(
                            Code::AMBIGUOUS_CLASSIFICATION,
                            format!(
                                "source profile {name:?} component {:?} names member prefix \
                                 {prefix:?}, not one exact provider member",
                                component.component
                            ),
                        )
                    }
                    (None, None) => {
                        return refuse(
                            Code::MISSING_FIELD,
                            format!(
                                "source profile {name:?} component {:?} names no exact \
                                 provider member",
                                component.component
                            ),
                        )
                    }
                    (Some(_), Some(_)) => {
                        unreachable!("registry parser rejects dual member bindings")
                    }
                }
            }
        }
        members.sort();
        members.dedup();
        if members.is_empty() {
            return refuse(
                Code::MISSING_FIELD,
                "source profiles name no exact provider members",
            );
        }
        Ok(members)
    }

    /// The verdict as an authorization, with the three refusals spelled out at one site.
    pub fn authorize(&self, what: &str, fp: &Fingerprint) -> Result<Banked> {
        match self.classify(fp) {
            Verdict::Authorized(b) => Ok(*b),
            Verdict::ComponentAmbiguous { banked_as } => refuse(
                Code::AMBIGUOUS_CLASSIFICATION,
                format!(
                    "{what}: key set {} is reviewed, but never as component {:?} — it is \
                     banked as {banked_as:?}. A key set does not name a component (H3's \
                     `transformer` and `transformer_ref` are the same 532 keys in \
                     byte-identical headers), so the component is reviewed data: bank this \
                     one, or ingest under a component that is.",
                    fp.keyset_digest, fp.component
                ),
            ),
            Verdict::Ambiguous(names) => refuse(
                Code::AMBIGUOUS_CLASSIFICATION,
                format!(
                    "{what}: (key set {}, component {:?}) matches {} reviewed entries \
                     {names:?} — surviving ambiguity refuses; an explicit profile must name \
                     exactly one",
                    fp.keyset_digest,
                    fp.component,
                    names.len()
                ),
            ),
            Verdict::Unregistered => refuse(
                Code::UNREGISTERED_FINGERPRINT,
                format!(
                    "{what}: key set {} ({} logical keys) matches no reviewed registry entry. \
                     A filename never authorizes: bank the fingerprint under review, or plan \
                     the source as-is.",
                    fp.keyset_digest, fp.logical_keys
                ),
            ),
        }
    }

    pub fn merge(&mut self, e: Banked) {
        match self.entries.iter_mut().find(|x| {
            x.keyset_digest == e.keyset_digest
                && x.component == e.component
                && x.tensor_schema_digest == e.tensor_schema_digest
        }) {
            Some(slot) => *slot = e,
            None => self.entries.push(e),
        }
        self.entries.sort_by(|a, b| {
            (&a.keyset_digest, &a.component, &a.tensor_schema_digest).cmp(&(
                &b.keyset_digest,
                &b.component,
                &b.tensor_schema_digest,
            ))
        });
    }

    pub fn merge_source_profile(&mut self, profile: SourceProfile) {
        match self
            .source_profiles
            .iter_mut()
            .find(|existing| existing.name == profile.name)
        {
            Some(slot) => *slot = profile,
            None => self.source_profiles.push(profile),
        }
        self.source_profiles.sort_by(|a, b| a.name.cmp(&b.name));
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

impl FingerprintRegistry {
    fn from_value(v: &Value) -> Result<Self> {
        let mut f = Fields::new("FingerprintRegistry", v)?;
        let mut entries = Vec::new();
        for ev in as_arr("FingerprintRegistry", "entries", f.req("entries")?)? {
            let mut e = Fields::new("Banked", ev)?;
            let component = e.req_str("component")?.to_string();
            let converter = e.req_str("converter")?.to_string();
            let dialect = e.req_str("dialect")?.to_string();
            let keyset_digest = crate::ids::prefixed("keyset_digest", e.req_str("keyset_digest")?)?;
            let logical_keys = e.req_uint("logical_keys")?;
            let provenance = provenance_from(e.req("provenance")?)?;
            let tensor_schema_digest =
                crate::ids::prefixed("tensor_schema_digest", e.req_str("tensor_schema_digest")?)?;
            e.done()?;
            for (what, s) in [
                ("component", &component),
                ("converter", &converter),
                ("dialect", &dialect),
            ] {
                ascii_name(what, s, limits::MAX_NAME_BYTES)?;
            }
            entries.push(Banked {
                keyset_digest,
                dialect,
                component,
                converter,
                tensor_schema_digest,
                logical_keys,
                provenance,
            });
        }
        // A newer registry's additional top-level sections belong to a newer reader.
        let source_profiles = match f.opt("source_profiles") {
            None => Vec::new(),
            Some(value) => parse_source_profiles(value)?,
        };
        let key = |e: &Banked| {
            (
                e.keyset_digest.clone(),
                e.component.clone(),
                e.tensor_schema_digest.clone(),
            )
        };
        entries.sort_by_key(key);
        entries.dedup();
        if entries
            .windows(2)
            .any(|pair| key(&pair[0]) == key(&pair[1]))
        {
            return refuse(
                Code::DUPLICATE_KEY,
                "two entries bank the same (keyset_digest, component, tensor_schema_digest) \
                 with different verdicts",
            );
        }
        for profile in &source_profiles {
            for component in &profile.components {
                for variant in &component.variants {
                    if !entries.iter().any(|entry| {
                        entry.component == component.component
                            && entry.keyset_digest == variant.keyset_digest
                    }) {
                        return refuse(
                            Code::UNREGISTERED_FINGERPRINT,
                            format!(
                                "source profile {:?} cites unbanked component {:?} / {}",
                                profile.name, component.component, variant.keyset_digest
                            ),
                        );
                    }
                }
            }
        }
        Ok(FingerprintRegistry {
            entries,
            source_profiles,
        })
    }

    fn to_value(&self) -> Value {
        let mut fields = vec![(
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
                                "tensor_schema_digest",
                                Value::str(e.tensor_schema_digest.clone()),
                            ),
                        ])
                    })
                    .collect(),
            ),
        )];
        if !self.source_profiles.is_empty() {
            fields.push((
                "source_profiles",
                Value::arr(
                    self.source_profiles
                        .iter()
                        .map(|profile| {
                            let mut fields = vec![
                                (
                                    "components",
                                    Value::arr(
                                        profile
                                            .components
                                            .iter()
                                            .map(|component| {
                                                let mut fields = vec![
                                                    (
                                                        "component",
                                                        Value::str(component.component.clone()),
                                                    ),
                                                    (
                                                        "target_encoding",
                                                        Value::str(
                                                            component.target_encoding.clone(),
                                                        ),
                                                    ),
                                                    (
                                                        "variants",
                                                        Value::arr(
                                                            component
                                                                .variants
                                                                .iter()
                                                                .map(|variant| {
                                                                    Value::obj(vec![
                                                                        (
                                                                            "construction_order",
                                                                            Value::arr(
                                                                                variant
                                                                                    .construction_order
                                                                                    .iter()
                                                                                    .map(|key| Value::str(key.clone()))
                                                                                    .collect(),
                                                                            ),
                                                                        ),
                                                                        (
                                                                            "keyset_digest",
                                                                            Value::str(variant.keyset_digest.clone()),
                                                                        ),
                                                                    ])
                                                                })
                                                                .collect(),
                                                        ),
                                                    ),
                                                ];
                                                if let Some(member) = &component.source_member {
                                                    fields.push((
                                                        "source_member",
                                                        Value::str(member.clone()),
                                                    ));
                                                }
                                                if let Some(prefix) = &component.source_member_prefix {
                                                    fields.push((
                                                        "source_member_prefix",
                                                        Value::str(prefix.clone()),
                                                    ));
                                                }
                                                Value::obj(fields)
                                            })
                                            .collect(),
                                    ),
                                ),
                                ("name", Value::str(profile.name.clone())),
                            ];
                            if !profile.auto_select {
                                fields.push(("auto_select", Value::Bool(false)));
                            }
                            if let Some(grammar) = &profile.grammar {
                                let converter = Value::str(grammar.converter.clone());
                                let key_prefix = Value::str(grammar.key_prefix.clone());
                                let grammar =
                                    vec![("converter", converter), ("key_prefix", key_prefix)];
                                fields.push(("grammar", Value::obj(grammar)));
                            }
                            Value::obj(fields)
                        })
                        .collect(),
                ),
            ));
        }
        Value::obj(fields)
    }
}

fn parse_source_profiles(value: &Value) -> Result<Vec<SourceProfile>> {
    let mut profiles = Vec::new();
    for value in as_arr("FingerprintRegistry", "source_profiles", value)? {
        let mut fields = Fields::new("SourceProfile", value)?;
        let name = fields.req_str("name")?.to_string();
        let auto_select = match fields.opt("auto_select") {
            None => true,
            Some(Value::Bool(value)) => *value,
            Some(_) => return refuse(Code::WRONG_TYPE, "SourceProfile.auto_select must be bool"),
        };
        ascii_name("source profile", &name, limits::MAX_NAME_BYTES)?;
        let grammar = match fields.opt("grammar") {
            None => None,
            Some(value) => {
                let mut grammar = Fields::new("Grammar", value)?;
                let converter = grammar.req_str("converter")?.to_string();
                let key_prefix = grammar.req_str("key_prefix")?.to_string();
                grammar.done()?;
                Some(Grammar {
                    converter,
                    key_prefix,
                })
            }
        };
        let mut components = Vec::new();
        for value in as_arr("SourceProfile", "components", fields.req("components")?)? {
            let mut component_fields = Fields::new("SourceProfileComponent", value)?;
            let component = component_fields.req_str("component")?.to_string();
            ascii_name(
                "source profile component",
                &component,
                limits::MAX_NAME_BYTES,
            )?;
            let target_encoding = component_fields.req_str("target_encoding")?.to_string();
            ascii_name(
                "source profile target_encoding",
                &target_encoding,
                limits::MAX_NAME_BYTES,
            )?;
            let mut variants = Vec::new();
            for value in as_arr(
                "SourceProfileComponent",
                "variants",
                component_fields.req("variants")?,
            )? {
                let mut variant_fields = Fields::new("SourceProfileVariant", value)?;
                let keyset_digest = prefixed(
                    "source profile keyset_digest",
                    variant_fields.req_str("keyset_digest")?,
                )?;
                let construction_order = as_arr(
                    "SourceProfileVariant",
                    "construction_order",
                    variant_fields.req("construction_order")?,
                )?
                .iter()
                .map(|value| {
                    as_str("SourceProfileVariant", "construction_order", value)
                        .map(ToString::to_string)
                })
                .collect::<Result<Vec<_>>>()?;
                variant_fields.done()?;
                if construction_order.is_empty() {
                    return refuse(
                        Code::CONSTRUCTION_ORDER_REQUIRED,
                        format!(
                            "source profile {name:?} component {component:?} variant \
                             {keyset_digest} has no order"
                        ),
                    );
                }
                let mut unique = construction_order.clone();
                unique.sort();
                unique.dedup();
                if unique.len() != construction_order.len() {
                    return refuse(
                        Code::DUPLICATE_KEY,
                        format!(
                            "source profile {name:?} component {component:?} variant \
                             {keyset_digest} repeats a key"
                        ),
                    );
                }
                variants.push(SourceProfileVariant {
                    keyset_digest,
                    construction_order,
                });
            }
            let source_member = match component_fields.opt("source_member") {
                Some(value) => {
                    let member = as_str("SourceProfileComponent", "source_member", value)?;
                    validate_source_member(member)?;
                    Some(member.to_string())
                }
                None => None,
            };
            let source_member_prefix = match component_fields.opt("source_member_prefix") {
                Some(value) => {
                    let prefix = as_str("SourceProfileComponent", "source_member_prefix", value)?;
                    validate_source_member_prefix(prefix)?;
                    Some(prefix.to_string())
                }
                None => None,
            };
            component_fields.done()?;
            if source_member.is_some() && source_member_prefix.is_some() {
                return refuse(
                    Code::DUPLICATE_KEY,
                    format!(
                        "source profile {name:?} component {component:?} binds both an exact \
                         member and a member prefix"
                    ),
                );
            }
            if variants.is_empty() && grammar.is_none() {
                return refuse(
                    Code::MISSING_FIELD,
                    format!("source profile {name:?} component {component:?} has no variants"),
                );
            }
            let mut keysets: Vec<&str> = variants
                .iter()
                .map(|variant| variant.keyset_digest.as_str())
                .collect();
            let given = keysets.clone();
            keysets.sort();
            if keysets != given {
                return refuse(
                    Code::SORT_ORDER,
                    format!(
                        "source profile {name:?} component {component:?} variants must be \
                         sorted by keyset_digest"
                    ),
                );
            }
            keysets.dedup();
            if keysets.len() != variants.len() {
                return refuse(
                    Code::DUPLICATE_KEY,
                    format!("source profile {name:?} component {component:?} repeats a variant"),
                );
            }
            components.push(SourceProfileComponent {
                component,
                source_member,
                source_member_prefix,
                target_encoding,
                variants,
            });
        }
        fields.done()?;
        if components.is_empty() {
            return refuse(
                Code::MISSING_FIELD,
                format!("source profile {name:?} has no components"),
            );
        }
        let mut component_names: Vec<&str> = components
            .iter()
            .map(|component| component.component.as_str())
            .collect();
        component_names.sort();
        component_names.dedup();
        if component_names.len() != components.len() {
            return refuse(
                Code::DUPLICATE_KEY,
                format!("source profile {name:?} repeats a component"),
            );
        }
        profiles.push(SourceProfile {
            name,
            auto_select,
            grammar,
            components,
        });
    }
    let mut names: Vec<&str> = profiles
        .iter()
        .map(|profile| profile.name.as_str())
        .collect();
    let given = names.clone();
    names.sort();
    if names != given {
        return refuse(Code::SORT_ORDER, "source profiles must be sorted by name");
    }
    names.dedup();
    if names.len() != profiles.len() {
        return refuse(Code::DUPLICATE_KEY, "source profile names must be unique");
    }
    Ok(profiles)
}

pub fn validate_source_member(member: &str) -> Result<()> {
    if member.is_empty()
        || member.len() > 1024
        || !member.is_ascii()
        || member.starts_with('/')
        || member.contains('\\')
        || member
            .split('/')
            .any(|part| part.is_empty() || part == "." || part == "..")
    {
        return refuse(
            Code::KEY_GRAMMAR,
            format!("source member {member:?} is not a safe canonical relative member"),
        );
    }
    Ok(())
}

pub fn validate_source_member_prefix(prefix: &str) -> Result<()> {
    if !prefix.is_empty()
        && (!prefix.ends_with('/') || validate_source_member(&format!("{prefix}member")).is_err())
    {
        return refuse(
            Code::KEY_GRAMMAR,
            format!("source member prefix {prefix:?} is not a safe canonical prefix"),
        );
    }
    Ok(())
}

/// Decoded `__metadata__` facts the border is willing to carry as EVIDENCE STAMPS. Foreign
/// JSON is decoded through this list; a value it cannot carry is discarded.
pub fn evidence_stamps(h: &SourceHeader) -> Result<Vec<(String, String)>> {
    const KEPT: [&str; 5] = [
        "modelspec.architecture",
        "ss_h3_training_mode",
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
        if KEPT.contains(&k.as_str())
            && v.len() <= limits::MAX_NAME_BYTES
            && v.bytes().all(|b| (0x20..=0x7e).contains(&b))
        {
            out.push((k.clone(), v.clone()));
        }
    }
    out.sort();
    Ok(out)
}

/// What a `.comfy_quant` marker actually says. The payload is ComfyUI's JSON, read by the
/// bounded RFC 8259 reader; fields other than the two below are ignored. It never survives
/// into the canonical header.
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
    // The payload is a NUL-padded JSON blob inside a U8 tensor.
    let end = bytes.iter().position(|b| *b == 0).unwrap_or(bytes.len());
    let v = crate::jcs::parse(&bytes[..end], limits::MAX_DOC_TEXT_BYTES)?;
    let crate::jcs::Json::Obj(fields) = &v else {
        return refuse(
            Code::WRONG_TYPE,
            format!("{key}: marker payload is not a JSON object"),
        );
    };
    let format = match fields.iter().find(|(k, _)| k == "format") {
        Some((_, crate::jcs::Json::Str(format))) => format.clone(),
        Some(_) => {
            return refuse(
                Code::WRONG_TYPE,
                format!("{key}: marker `format` is not a string"),
            )
        }
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
        Some((_, crate::jcs::Json::Bool(b))) => *b,
        Some(_) => {
            return refuse(
                Code::WRONG_TYPE,
                format!("{key}: full_precision_matrix_mult is not a bool"),
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

#[cfg(test)]
mod tests {
    use super::*;

    fn registry() -> FingerprintRegistry {
        FingerprintRegistry {
            entries: vec![Banked {
                keyset_digest: object_id(b"keys"),
                dialect: "safetensors.diffusers".into(),
                component: "model".into(),
                converter: "identity/1".into(),
                tensor_schema_digest: object_id(b"tensor-schema"),
                logical_keys: 1,
                provenance: Provenance {
                    kind: SYNTHETIC.into(),
                    source: "unit test".into(),
                    source_sha256: None,
                    note: "plain reviewed data".into(),
                },
            }],
            source_profiles: Vec::new(),
        }
    }

    #[test]
    fn registry_is_plain_data_not_a_document_format() {
        let registry = registry();
        let bytes = registry.to_bytes();
        assert!(!String::from_utf8_lossy(&bytes).contains("format"));
        assert_eq!(FingerprintRegistry::parse(&bytes).unwrap(), registry);

        let newer = String::from_utf8(bytes)
            .unwrap()
            .replacen('{', r#"{"format":"retired","#, 1);
        assert_eq!(
            FingerprintRegistry::parse(newer.as_bytes()).unwrap(),
            registry
        );
    }

    fn source_component(member: Option<&str>, prefix: Option<&str>) -> SourceProfileComponent {
        SourceProfileComponent {
            component: "model".into(),
            source_member: member.map(str::to_string),
            source_member_prefix: prefix.map(str::to_string),
            target_encoding: "bf16".into(),
            variants: vec![SourceProfileVariant {
                keyset_digest: object_id(b"keys"),
                construction_order: vec!["weight".into()],
            }],
        }
    }

    #[test]
    fn exact_source_members_unions_profiles_without_guessing() {
        let mut registry = registry();
        registry.source_profiles = vec![
            SourceProfile {
                auto_select: true,
                grammar: None,
                name: "profile/a".into(),
                components: vec![source_component(Some("a/model.index.json"), None)],
            },
            SourceProfile {
                auto_select: true,
                grammar: None,
                name: "profile/b".into(),
                components: vec![
                    source_component(Some("b/model.safetensors"), None),
                    source_component(Some("a/model.index.json"), None),
                ],
            },
        ];
        assert_eq!(
            registry
                .exact_source_members(&["profile/b".into(), "profile/a".into()])
                .unwrap(),
            vec![
                "a/model.index.json".to_string(),
                "b/model.safetensors".to_string()
            ]
        );
    }

    #[test]
    fn exact_source_members_refuses_missing_or_non_exact_profiles() {
        let mut registry = registry();
        registry.source_profiles = vec![SourceProfile {
            auto_select: true,
            grammar: None,
            name: "profile/prefix".into(),
            components: vec![source_component(None, Some("weights/"))],
        }];
        assert_eq!(
            registry
                .exact_source_members(&["profile/missing".into()])
                .unwrap_err()
                .code,
            Code::UNREGISTERED_FINGERPRINT
        );
        assert_eq!(
            registry
                .exact_source_members(&["profile/prefix".into()])
                .unwrap_err()
                .code,
            Code::AMBIGUOUS_CLASSIFICATION
        );
    }
}
