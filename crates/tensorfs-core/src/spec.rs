//! One nested encoding value inside a CozyTensors header. It is not a standalone document.

use crate::canon::{as_arr, as_obj, as_str, Fields, Value};
use crate::dtype::Dtype;
use crate::err::{refuse, Code, Result};
use crate::ids::{ascii_name, ObjectRef};
use crate::limits;
use crate::relation::{Carrier, Relation};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Role {
    pub carrier: Carrier,
    pub shape: Relation,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EncodingSpec {
    pub logical_dtypes: Vec<Dtype>,
    pub logical_rank: Option<u64>,
    /// EXACT role set, sorted by name. No optional roles: one digest, one byte contract.
    pub roles: Vec<(String, Role)>,
    /// Raw-bit conformance vectors. Mandatory for executable qualification.
    pub vectors: Option<ObjectRef>,
}

impl EncodingSpec {
    pub fn canonical_bytes(&self) -> Vec<u8> {
        crate::cbor::encode_encoding(self).expect("validated encoding value")
    }

    pub fn object_id(&self) -> String {
        crate::ids::object_id(&self.canonical_bytes())
    }

    pub fn object_ref(&self) -> ObjectRef {
        ObjectRef::of(&self.canonical_bytes())
    }

    /// Qualification-fixture seam for the exact nested CBOR value. Production headers
    /// decode this value in place and never store it as an independent document.
    pub fn decode_nested(bytes: &[u8]) -> Result<Self> {
        crate::cbor::decode_encoding(bytes)
    }

    pub fn role(&self, name: &str) -> Option<&Role> {
        self.roles.iter().find(|(n, _)| n == name).map(|(_, r)| r)
    }

    /// A vectorless spec stores and inspects; it may never qualify an implementation.
    pub fn require_executable(&self) -> Result<&ObjectRef> {
        match &self.vectors {
            Some(v) => Ok(v),
            None => refuse(
                Code::VECTORS_REQUIRED,
                format!(
                    "spec {} is vectorless: store/inspect only, never executable qualification",
                    self.object_id()
                ),
            ),
        }
    }

    pub fn admits_logical(&self, dt: Dtype) -> bool {
        self.logical_dtypes.contains(&dt)
    }
}

impl EncodingSpec {
    pub fn from_value(v: &Value) -> Result<Self> {
        let mut f = Fields::new("EncodingSpec", v)?;

        let dts = as_arr("EncodingSpec", "logical_dtypes", f.req("logical_dtypes")?)?;
        if dts.is_empty() {
            return refuse(Code::MISSING_FIELD, "EncodingSpec.logical_dtypes is empty");
        }
        let mut logical_dtypes = Vec::new();
        let mut names = Vec::new();
        for d in dts {
            let n = as_str("EncodingSpec", "logical_dtypes", d)?;
            logical_dtypes.push(Dtype::parse(n)?);
            names.push(n);
        }
        let mut sorted = names.clone();
        sorted.sort_unstable();
        sorted.dedup();
        if sorted != names {
            return refuse(
                Code::SORT_ORDER,
                "EncodingSpec.logical_dtypes must be sorted and unique",
            );
        }

        let logical_rank = match f.opt("logical_rank") {
            Some(x) => Some(crate::canon::as_uint("EncodingSpec", "logical_rank", x)?),
            None => None,
        };

        let rs = as_obj("EncodingSpec", "roles", f.req("roles")?)?;
        if rs.is_empty() || rs.len() > limits::MAX_PARTS {
            return refuse(
                Code::COUNT_CAP,
                format!("EncodingSpec.roles count {} out of bounds", rs.len()),
            );
        }
        let mut roles = Vec::new();
        for (name, rv) in rs {
            ascii_name("EncodingSpec.role", name, limits::MAX_NAME_BYTES)?;
            let mut rf = Fields::new("EncodingSpec.role", rv)?;
            if rf.opt("optional").is_some() {
                return refuse(
                    Code::OPTIONAL_ROLE_FORBIDDEN,
                    format!("role {name:?}: roles are EXACT — a present/absent variation is a different spec digest"),
                );
            }
            let carrier = Carrier::from_value(rf.req("carrier")?)?;
            let shape = Relation::from_value(rf.req("shape")?)?;
            rf.done()?;
            roles.push((name.clone(), Role { carrier, shape }));
        }

        let vectors = match f.opt("vectors") {
            Some(x) => Some(ObjectRef::from_value("EncodingSpec.vectors", x)?),
            None => None,
        };
        f.done()?;

        let uses_dims = roles
            .iter()
            .any(|(_, r)| matches!(r.shape, Relation::Dims(_)));
        match logical_rank {
            Some(r) if r == 0 || r as usize > limits::MAX_RANK => {
                return refuse(Code::RANK_CAP, "EncodingSpec.logical_rank out of bounds")
            }
            None if uses_dims => {
                return refuse(
                    Code::MISSING_FIELD,
                    "EncodingSpec.logical_rank is required whenever a role uses a dims relation",
                )
            }
            _ => {}
        }

        Ok(EncodingSpec {
            logical_dtypes,
            logical_rank,
            roles,
            vectors,
        })
    }

    pub fn to_value(&self) -> Value {
        let mut fields = vec![
            (
                "logical_dtypes",
                Value::arr(
                    self.logical_dtypes
                        .iter()
                        .map(|d| Value::str(d.name()))
                        .collect(),
                ),
            ),
            (
                "roles",
                Value::map(
                    self.roles
                        .iter()
                        .map(|(n, r)| {
                            (
                                n.clone(),
                                Value::obj(vec![
                                    ("carrier", r.carrier.to_value()),
                                    ("shape", r.shape.to_value()),
                                ]),
                            )
                        })
                        .collect(),
                ),
            ),
        ];
        if let Some(r) = self.logical_rank {
            fields.push(("logical_rank", Value::uint(r)));
        }
        if let Some(v) = &self.vectors {
            fields.push(("vectors", v.to_value()));
        }
        Value::obj(fields)
    }
}
