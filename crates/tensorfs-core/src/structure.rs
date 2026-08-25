//! Structure fixtures (tfs-008): a banked canonical projection plus its `topology_digest`
//! for ONE (task family, structural variant).
//!
//! A fixture is EVIDENCE, so it carries its provenance in the same bytes that carry its
//! claim. `real` names the exact source artifact by digest; `synthetic` says so out loud.
//! There is no third state and no default — a fixture cannot be banked without saying
//! where its shapes came from, which is the only thing that stops a fabrication from
//! reading as evidence (decisions row: the deleted H3 worked example).
//!
//! The projection itself is computed by `header::topology_value`, the SAME function a real
//! header uses, so a fixture's digest is directly comparable with a checkpoint's.

use crate::canon::{as_arr, as_obj, as_str, Fields, Value};
use crate::dtype::Dtype;
use crate::err::{refuse, Code, Result};
use crate::header::{topology_digest_of, topology_value};
use crate::ids::{ascii_name, check_format, hex64, Doc};
use crate::limits;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Logical {
    pub dtype: Dtype,
    pub shape: Vec<u64>,
}

/// Where a fixture's shapes came from. Closed: `real` or `synthetic`, never absent.
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

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StructureFixture {
    pub task_family: String,
    pub structural_variant: String,
    pub provenance: Provenance,
    pub components: Vec<(String, Vec<(String, Logical)>)>,
}

impl StructureFixture {
    pub fn tensors(&self) -> impl Iterator<Item = (&str, &str, Dtype, &[u64])> {
        self.components.iter().flat_map(|(c, ks)| {
            ks.iter()
                .map(move |(k, l)| (c.as_str(), k.as_str(), l.dtype, l.shape.as_slice()))
        })
    }

    pub fn topology_value(&self) -> Value {
        topology_value(self.tensors())
    }

    /// The banked key. Two structures with the same component roles, canonical logical keys
    /// and logical shapes have the same digest, whatever their bytes are encoded as.
    pub fn topology_digest(&self) -> String {
        topology_digest_of(&self.topology_value())
    }

    pub fn stamp(&self) -> String {
        format!("{}/{}", self.task_family, self.structural_variant)
    }

    pub fn count(&self) -> usize {
        self.components.iter().map(|(_, ks)| ks.len()).sum()
    }
}

fn shape_value(s: &[u64]) -> Value {
    Value::arr(s.iter().map(|d| Value::uint(*d)).collect())
}

impl Doc for StructureFixture {
    const FORMAT: &'static str = "tensorfs.structure_fixture/1";
    const MAX_BYTES: usize = limits::DOC_MAX_BYTES;

    fn from_value(v: &Value) -> Result<Self> {
        let mut f = Fields::new("StructureFixture", v)?;
        check_format(
            "StructureFixture",
            as_str("StructureFixture", "format", f.req("format")?)?,
            Self::FORMAT,
        )?;

        let mut components = Vec::new();
        for (name, cv) in as_obj("StructureFixture", "components", f.req("components")?)? {
            ascii_name("StructureFixture.component", name, limits::MAX_NAME_BYTES)?;
            let mut keys = Vec::new();
            for (key, lv) in as_obj("StructureFixture.components", name, cv)? {
                if key.is_empty() || key.len() > limits::MAX_KEY_BYTES {
                    return refuse(
                        Code::KEY_GRAMMAR,
                        format!("{name}/{key}: key length out of bounds"),
                    );
                }
                let mut lf = Fields::new("StructureFixture.logical", lv)?;
                let dtype = Dtype::parse(as_str("logical", "dtype", lf.req("dtype")?)?)?;
                let sa = as_arr("logical", "shape", lf.req("shape")?)?;
                if sa.len() > limits::MAX_RANK {
                    return refuse(Code::RANK_CAP, format!("{name}/{key}: rank over cap"));
                }
                let shape = sa
                    .iter()
                    .map(|x| crate::canon::as_uint("logical", "shape", x))
                    .collect::<Result<Vec<u64>>>()?;
                lf.done()?;
                crate::dtype::checked_elements(key, &shape)?;
                keys.push((key.clone(), Logical { dtype, shape }));
            }
            if keys.is_empty() {
                return refuse(
                    Code::EMPTY_COMPONENTS,
                    format!("component {name:?} has no keys"),
                );
            }
            components.push((name.clone(), keys));
        }

        let mut pf = Fields::new("Provenance", f.req("provenance")?)?;
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

        let structural_variant = f.req_str("structural_variant")?.to_string();
        let task_family = f.req_str("task_family")?.to_string();
        f.done()?;

        ascii_name("task_family", &task_family, limits::MAX_NAME_BYTES)?;
        ascii_name(
            "structural_variant",
            &structural_variant,
            limits::MAX_NAME_BYTES,
        )?;
        if note.len() > limits::MAX_DOC_TEXT_BYTES || source.len() > limits::MAX_NAME_BYTES {
            return refuse(Code::SIZE_CAP, "provenance text over the byte cap");
        }
        match kind.as_str() {
            // A `real` fixture without its source digest is an unfalsifiable claim.
            REAL if source_sha256.is_none() => {
                return refuse(
                    Code::MISSING_FIELD,
                    "provenance.kind \"real\" requires source_sha256: an undigested claim of \
                     realness is exactly the fabrication-that-reads-as-evidence failure",
                )
            }
            REAL | SYNTHETIC => {}
            other => {
                return refuse(
                    Code::UNKNOWN_FIELD,
                    format!("provenance.kind {other:?} is outside {{{REAL}, {SYNTHETIC}}}"),
                )
            }
        }
        if components.is_empty() || components.len() > limits::MAX_COMPONENTS {
            return refuse(Code::EMPTY_COMPONENTS, "component count out of bounds");
        }
        let total: usize = components.iter().map(|(_, k)| k.len()).sum();
        if total > limits::MAX_TENSORS {
            return refuse(Code::COUNT_CAP, "fixture key count over cap");
        }

        Ok(StructureFixture {
            task_family,
            structural_variant,
            provenance: Provenance {
                kind,
                source,
                source_sha256,
                note,
            },
            components,
        })
    }

    fn to_value(&self) -> Value {
        let mut p = vec![
            ("kind", Value::str(self.provenance.kind.clone())),
            ("note", Value::str(self.provenance.note.clone())),
            ("source", Value::str(self.provenance.source.clone())),
        ];
        if let Some(h) = &self.provenance.source_sha256 {
            p.push(("source_sha256", Value::str(h.clone())));
        }
        Value::obj(vec![
            (
                "components",
                Value::map(
                    self.components
                        .iter()
                        .map(|(c, ks)| {
                            (
                                c.clone(),
                                Value::map(
                                    ks.iter()
                                        .map(|(k, l)| {
                                            (
                                                k.clone(),
                                                Value::obj(vec![
                                                    ("dtype", Value::str(l.dtype.name())),
                                                    ("shape", shape_value(&l.shape)),
                                                ]),
                                            )
                                        })
                                        .collect(),
                                ),
                            )
                        })
                        .collect(),
                ),
            ),
            ("format", Value::str(Self::FORMAT)),
            ("provenance", Value::obj(p)),
            (
                "structural_variant",
                Value::str(self.structural_variant.clone()),
            ),
            ("task_family", Value::str(self.task_family.clone())),
        ])
    }
}
