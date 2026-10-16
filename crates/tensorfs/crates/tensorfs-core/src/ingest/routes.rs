//! Routed converters: a single-file checkpoint becomes its target pipeline's canonical
//! checkpoint in the one ingest pass, driven by a generated route table.
//!
//! Each table (`routes/<family>.routes.json`) is derived by `scripts/single-file-routes.py`
//! from diffusers' own single-file converters and the target census on meta, per family spec
//! (`routes/<family>.spec.json`). Per component it lists the constructor's exact key order and
//! how each key's bytes come from one source run; geometry is read from the carrier:
//!   rekey     — the source run streams through under its canonical key
//!   squeeze   — trailing unit axes drop (`[c, c, 1, 1]` conv to `[c, c]` linear)
//!   split k n — row block `k` of `n` equal blocks (OpenCLIP's fused q|k|v), one run
//!   transpose — a matrix becomes `[cols, rows]`
//! A key outside the table refuses; its `drop`/`optional` source keys are not constructed.
//! A carrier's keys belong to the component with the longest matching prefix (Anima's `net.`
//! DiT and its `net.llm_adapter.` conditioner share one file). Floating sources narrow to the
//! converter's lane. Configs and tokenizers come from the converter's pinned reference,
//! fetched by the uploader; nothing model-owned lives here.

use std::collections::HashMap;
use std::sync::OnceLock;

use crate::canon::{self, as_arr, as_str, as_uint, Fields, Value};
use crate::dtype::checked_bytes;
use crate::err::{refuse, Code, Result};
use crate::limits;
use crate::spec::EncodingSpec;

use super::carrier::SourceHeader;
use super::convert::{check_permute_size, Bytes, Converter, Effect, Op, Plan, RolePlan, Xform};

pub const SDXL: &str = "sdxl.single_file/1-diffusers0.40.0-transformers5.16.1";
pub const ANIMA: &str = "anima.single_file/1-diffusers0.40.0-transformers5.16.1";

const TABLES: [(&str, &[u8]); 2] = [
    (SDXL, include_bytes!("routes/sdxl.routes.json")),
    (ANIMA, include_bytes!("routes/anima.routes.json")),
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    Rekey,
    Squeeze,
    Split { block: u64, blocks: u64 },
    Transpose,
}

#[derive(Debug, Clone)]
pub struct Route {
    pub target: String,
    pub source: String,
    pub kind: Kind,
}

#[derive(Debug, Clone)]
pub struct Component {
    pub name: String,
    pub prefix: String,
    /// Source keys every reviewed carrier has and the constructor does not take.
    pub drop: Vec<String>,
    /// Source keys a carrier may or may not have; never constructed.
    pub optional: Vec<String>,
    pub routes: Vec<Route>,
}

#[derive(Debug, Clone)]
pub struct Table {
    pub converter: String,
    pub reference: String,
    pub components: Vec<Component>,
}

/// The generated table a converter plans with, if it is a routed converter.
pub fn table(converter: &str) -> Option<&'static Table> {
    static PARSED: OnceLock<Vec<Table>> = OnceLock::new();
    PARSED
        .get_or_init(|| {
            TABLES
                .iter()
                .map(|(_, bytes)| parse(bytes).expect("a route table is frozen generated data"))
                .collect()
        })
        .iter()
        .find(|t| t.converter == converter)
}

fn strings(what: &str, key: &str, v: &Value) -> Result<Vec<String>> {
    as_arr(what, key, v)?
        .iter()
        .map(|s| as_str(what, key, s).map(str::to_string))
        .collect()
}

pub fn parse(bytes: &[u8]) -> Result<Table> {
    let doc = canon::parse(bytes, limits::DOC_MAX_BYTES)?;
    let mut fields = Fields::new("route table", &doc)?;
    let converter = fields.req_str("converter")?.to_string();
    let reference = fields.req_str("reference")?.to_string();
    let mut components = Vec::new();
    for c in as_arr("route table", "components", fields.req("components")?)? {
        let mut f = Fields::new("route component", c)?;
        let name = f.req_str("component")?.to_string();
        let prefix = f.req_str("prefix")?.to_string();
        let drop = strings("route component", "drop", f.req("drop")?)?;
        let optional = strings("route component", "optional", f.req("optional")?)?;
        let mut routes = Vec::new();
        for row in as_arr("route component", "routes", f.req("routes")?)? {
            let row = as_arr("route", "row", row)?;
            let text = |i: usize| match row.get(i) {
                Some(v) => as_str("route", "row", v).map(str::to_string),
                None => refuse(Code::MISSING_FIELD, "route row is short"),
            };
            let op = if row.len() > 2 {
                text(2)?
            } else {
                String::new()
            };
            let kind = match (op.as_str(), row.len()) {
                ("", 2) => Kind::Rekey,
                ("squeeze", 3) => Kind::Squeeze,
                ("transpose", 3) => Kind::Transpose,
                ("split", 5) => Kind::Split {
                    block: as_uint("route", "block", &row[3])?,
                    blocks: as_uint("route", "blocks", &row[4])?,
                },
                _ => {
                    return refuse(
                        Code::UNKNOWN_FIELD,
                        format!(
                            "{converter}: route {row:?} is an operation no converter implements"
                        ),
                    )
                }
            };
            routes.push(Route {
                target: text(0)?,
                source: text(1)?,
                kind,
            });
        }
        f.done()?;
        components.push(Component {
            name,
            prefix,
            drop,
            optional,
            routes,
        });
    }
    fields.done()?;
    Ok(Table {
        converter,
        reference,
        components,
    })
}

/// Plan one component from its table. Headers in, ops out; no tensor byte is read.
#[allow(clippy::too_many_arguments)]
pub fn plan(
    conv: &Converter,
    table: &Table,
    component: &str,
    file: usize,
    h: &SourceHeader,
    allowed: &[&str],
    plain: &EncodingSpec,
    p: &mut Plan,
) -> Result<()> {
    let Some(lane) = conv.lane else {
        return refuse(
            Code::DTYPE_UNKNOWN,
            format!("{}: a routed converter declares its lane", conv.name),
        );
    };
    if allowed != ["plain/1"] {
        return refuse(
            Code::UNKNOWN_ENCODING,
            format!(
                "{}: the canonical checkpoint is plain/1, not {allowed:?}",
                conv.name
            ),
        );
    }
    let Some(routes) = table.components.iter().find(|c| c.name == component) else {
        return refuse(
            Code::MISSING_TENSOR,
            format!("{} has no reviewed component {component:?}", conv.name),
        );
    };
    let mut source = HashMap::with_capacity(h.tensors.len());
    for t in &h.tensors {
        match t.key.strip_prefix(routes.prefix.as_str()) {
            Some(local) => source.insert(local, t),
            None => {
                return refuse(
                    Code::KEY_GRAMMAR,
                    format!(
                        "{}: outside the {component} prefix {:?}",
                        t.key, routes.prefix
                    ),
                )
            }
        };
    }
    let known: std::collections::HashSet<&str> = routes
        .routes
        .iter()
        .map(|r| r.source.as_str())
        .chain(
            routes
                .drop
                .iter()
                .chain(&routes.optional)
                .map(String::as_str),
        )
        .collect();
    let mut unknown: Vec<&str> = source
        .keys()
        .copied()
        .filter(|k| !known.contains(k))
        .collect();
    let mut missing: Vec<&str> = routes
        .routes
        .iter()
        .map(|r| r.source.as_str())
        .chain(routes.drop.iter().map(String::as_str))
        .filter(|k| !source.contains_key(k))
        .collect();
    if !unknown.is_empty() || !missing.is_empty() {
        unknown.sort();
        missing.sort();
        missing.dedup();
        return refuse(
            Code::MISSING_TENSOR,
            format!(
                "{component}: not the reviewed {} layout: {} unexpected {:?}, {} missing {:?}",
                conv.name,
                unknown.len(),
                &unknown[..unknown.len().min(4)],
                missing.len(),
                &missing[..missing.len().min(4)]
            ),
        );
    }
    p.header_bytes += h.header_bytes;
    for r in &routes.routes {
        let t = source[r.source.as_str()];
        if t.dtype != lane && !super::narrow::narrows(t.dtype, lane) {
            return refuse(
                Code::DTYPE_MISMATCH,
                format!(
                    "{}: no reviewed narrowing from {} to the {} lane",
                    t.key,
                    t.dtype.name(),
                    lane.name()
                ),
            );
        }
        let bad = |why: &str| {
            refuse(
                Code::SHAPE_MISMATCH,
                format!("{}: shape {:?} {why}", t.key, t.shape),
            )
        };
        let stream = Bytes::Stream {
            file,
            key: t.key.clone(),
        };
        let (shape, bytes, effect) = match r.kind {
            Kind::Rekey => (t.shape.clone(), stream, Effect::Rekey),
            Kind::Squeeze => {
                let mut shape = t.shape.clone();
                while shape.last() == Some(&1) {
                    shape.pop();
                }
                if shape.len() == t.shape.len() {
                    return bad("has no trailing unit axis");
                }
                (shape, stream, Effect::Rekey)
            }
            Kind::Split { block, blocks } => {
                let Some(&rows) = t.shape.first() else {
                    return bad("has no row axis");
                };
                if blocks == 0 || block >= blocks || rows % blocks != 0 {
                    return bad(&format!("does not hold row block {block} of {blocks}"));
                }
                check_permute_size(&t.key, t.nbytes())?;
                let mut shape = t.shape.clone();
                shape[0] = rows / blocks;
                let unit_bytes = checked_bytes(&t.key, &shape, t.dtype)?;
                let xform = Xform::QkvSplit {
                    groups: 1,
                    shares: [block, 1, blocks - block - 1],
                    take: 1,
                    unit_bytes,
                };
                (
                    shape,
                    Bytes::Permute {
                        file,
                        key: t.key.clone(),
                        xform,
                    },
                    Effect::Permute,
                )
            }
            Kind::Transpose => match t.shape.as_slice() {
                [rows, cols] => {
                    check_permute_size(&t.key, t.nbytes())?;
                    let xform = Xform::Transpose {
                        rows: *rows,
                        cols: *cols,
                        elem_bytes: t.dtype.size(),
                    };
                    (
                        vec![*cols, *rows],
                        Bytes::Permute {
                            file,
                            key: t.key.clone(),
                            xform,
                        },
                        Effect::Permute,
                    )
                }
                _ => return bad("is not a matrix"),
            },
        };
        p.ops.push(Op {
            component: component.to_string(),
            out_key: r.target.clone(),
            encoding: "plain/1".to_string(),
            encoding_id: plain.object_id(),
            logical_dtype: lane,
            logical_shape: shape.clone(),
            roles: vec![RolePlan {
                role: "value".to_string(),
                dtype: lane,
                shape,
                bytes,
            }],
            effect,
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ingest::convert::{converter, CONVERTERS};

    #[test]
    fn every_routed_converter_has_its_generated_table() {
        for (name, _) in TABLES {
            let conv = converter(name).unwrap();
            let table = table(name).unwrap();
            assert_eq!(conv.reference, Some(table.reference.as_str()), "{name}");
            assert!(conv.lane.is_some(), "{name}");
        }
        assert!(CONVERTERS
            .iter()
            .filter(|c| c.reference.is_some())
            .all(|c| table(c.name).is_some()));
    }

    #[test]
    fn the_sdxl_table_is_the_diffusers_constructor() {
        let table = table(SDXL).unwrap();
        let names: Vec<&str> = table.components.iter().map(|c| c.name.as_str()).collect();
        assert_eq!(names, ["text_encoder", "text_encoder_2", "unet", "vae"]);
        let sizes: Vec<usize> = table.components.iter().map(|c| c.routes.len()).collect();
        assert_eq!(sizes, [196, 517, 1680, 248]);
        let count = |kind: fn(Kind) -> bool| {
            table
                .components
                .iter()
                .flat_map(|c| &c.routes)
                .filter(|r| kind(r.kind))
                .count()
        };
        assert_eq!(
            count(|k| k
                == Kind::Split {
                    block: 0,
                    blocks: 3
                }),
            64
        );
        assert_eq!(count(|k| matches!(k, Kind::Split { blocks: 3, .. })), 192);
        assert_eq!(count(|k| k == Kind::Transpose), 1);
        assert_eq!(count(|k| k == Kind::Squeeze), 8);
    }

    #[test]
    fn the_anima_table_rekeys_every_constructor_key() {
        let table = table(ANIMA).unwrap();
        let shape: Vec<(&str, &str, usize)> = table
            .components
            .iter()
            .map(|c| (c.name.as_str(), c.prefix.as_str(), c.routes.len()))
            .collect();
        assert_eq!(
            shape,
            [
                ("text_conditioner", "net.llm_adapter.", 118),
                ("text_encoder", "model.", 310),
                ("transformer", "net.", 567),
                ("vae", "", 194),
            ]
        );
        assert!(table.components.iter().all(|c| c.drop.is_empty()
            && c.optional.is_empty()
            && c.routes.iter().all(|r| r.kind == Kind::Rekey)));
    }
}
