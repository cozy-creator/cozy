//! EncodingSpec conformance vectors: RAW role bytes and RAW expected logical bytes as
//! base64 bit patterns (JSON decimal floats and "NaN" strings are not bit-pattern evidence).

use crate::b64;
use crate::canon::{as_arr, as_obj, as_str, Fields, Value};
use crate::dtype::{checked_bytes, checked_elements, Dtype};
use crate::err::{refuse, Code, Result};
use crate::ids::ObjectRef;
use crate::limits;
use crate::spec::EncodingSpec;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RoleBits {
    pub dtype: Dtype,
    pub shape: Vec<u64>,
    pub bits: Vec<u8>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Case {
    pub name: String,
    pub logical_dtype: Dtype,
    pub logical_shape: Vec<u64>,
    pub roles: Vec<(String, RoleBits)>,
    pub expect_logical_bits: Vec<u8>,
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct EncodingVectors {
    pub cases: Vec<Case>,
}

fn shape_of(what: &str, v: &Value) -> Result<Vec<u64>> {
    as_arr(what, "shape", v)?
        .iter()
        .map(|x| crate::canon::as_uint(what, "shape", x))
        .collect()
}

fn shape_value(s: &[u64]) -> Value {
    Value::arr(s.iter().map(|d| Value::uint(*d)).collect())
}

impl EncodingVectors {
    /// Geometry + byte-length conformance of every case against the spec. Value
    /// reconstruction is an implementation's job; for the identity-carrier case
    /// (one `same_as_logical` role over the same shape) the bits must match exactly.
    pub fn check(&self, spec: &EncodingSpec) -> Result<usize> {
        for c in &self.cases {
            if !spec.admits_logical(c.logical_dtype) {
                return refuse(
                    Code::DTYPE_MISMATCH,
                    format!("{}: logical dtype outside the spec", c.name),
                );
            }
            let elements = checked_elements(&c.name, &c.logical_shape)?;
            let want_roles: Vec<&str> = spec.roles.iter().map(|(n, _)| n.as_str()).collect();
            let got_roles: Vec<&str> = c.roles.iter().map(|(n, _)| n.as_str()).collect();
            if want_roles != got_roles {
                return refuse(
                    Code::ROLE_SET_MISMATCH,
                    format!("{}: roles {got_roles:?} != {want_roles:?}", c.name),
                );
            }
            for (name, rb) in &c.roles {
                let role = spec.role(name).unwrap();
                if !role.carrier.admits(c.logical_dtype, rb.dtype) {
                    return refuse(
                        Code::DTYPE_MISMATCH,
                        format!("{}#{name}: carrier outside the role's set", c.name),
                    );
                }
                let want = role.shape.eval(&c.logical_shape, elements)?;
                if want != rb.shape {
                    return refuse(
                        Code::SHAPE_MISMATCH,
                        format!("{}#{name}: {:?} != {want:?}", c.name, rb.shape),
                    );
                }
                let bytes = checked_bytes(&c.name, &rb.shape, rb.dtype)?;
                if bytes != rb.bits.len() as u64 {
                    return refuse(
                        Code::BYTE_LENGTH_MISMATCH,
                        format!(
                            "{}#{name}: {} raw bytes, geometry wants {bytes}",
                            c.name,
                            rb.bits.len()
                        ),
                    );
                }
            }
            let logical_bytes = checked_bytes(&c.name, &c.logical_shape, c.logical_dtype)?;
            if logical_bytes != c.expect_logical_bits.len() as u64 {
                return refuse(
                    Code::BYTE_LENGTH_MISMATCH,
                    format!(
                        "{}: expected logical bytes {} != {logical_bytes}",
                        c.name,
                        c.expect_logical_bits.len()
                    ),
                );
            }
            if c.roles.len() == 1 {
                let (name, rb) = &c.roles[0];
                let role = spec.role(name).unwrap();
                if matches!(role.carrier, crate::relation::Carrier::SameAsLogical)
                    && matches!(role.shape, crate::relation::Relation::Same)
                    && rb.bits != c.expect_logical_bits
                {
                    return refuse(
                        Code::BYTE_LENGTH_MISMATCH,
                        format!(
                            "{}: identity carrier bits differ from the expected logical bits",
                            c.name
                        ),
                    );
                }
            }
        }
        Ok(self.cases.len())
    }
}

impl EncodingVectors {
    pub fn parse_fixture(bytes: &[u8]) -> Result<Self> {
        let value = crate::canon::parse_canonical(bytes, limits::DOC_MAX_BYTES)?;
        Self::from_value(&value)
    }

    pub fn fixture_bytes(&self) -> Vec<u8> {
        crate::canon::write(&self.to_value())
    }

    pub fn fixture_ref(&self) -> ObjectRef {
        ObjectRef::of(&self.fixture_bytes())
    }

    fn from_value(v: &Value) -> Result<Self> {
        let mut f = Fields::new("EncodingVectors", v)?;
        let mut cases = Vec::new();
        for cv in as_arr("EncodingVectors", "cases", f.req("cases")?)? {
            let mut cf = Fields::new("case", cv)?;
            let expect_logical_bits = b64::decode(cf.req_str("expect_logical_bits")?)?;
            let mut lf = Fields::new("case.logical", cf.req("logical")?)?;
            let logical_dtype = Dtype::parse(as_str("case.logical", "dtype", lf.req("dtype")?)?)?;
            let logical_shape = shape_of("case.logical", lf.req("shape")?)?;
            lf.done()?;
            let name = cf.req_str("name")?.to_string();
            let mut roles = Vec::new();
            for (rn, rv) in as_obj("case", "roles", cf.req("roles")?)? {
                let mut rf = Fields::new("case.role", rv)?;
                let bits = b64::decode(rf.req_str("bits")?)?;
                let dtype = Dtype::parse(as_str("case.role", "dtype", rf.req("dtype")?)?)?;
                let shape = shape_of("case.role", rf.req("shape")?)?;
                rf.done()?;
                roles.push((rn.clone(), RoleBits { dtype, shape, bits }));
            }
            cf.done()?;
            cases.push(Case {
                name,
                logical_dtype,
                logical_shape,
                roles,
                expect_logical_bits,
            });
        }
        f.done()?;
        if cases.is_empty() {
            return refuse(Code::MISSING_FIELD, "EncodingVectors.cases is empty");
        }
        Ok(EncodingVectors { cases })
    }

    fn to_value(&self) -> Value {
        Value::obj(vec![(
            "cases",
            Value::arr(
                self.cases
                    .iter()
                    .map(|c| {
                        Value::obj(vec![
                            (
                                "expect_logical_bits",
                                Value::str(b64::encode(&c.expect_logical_bits)),
                            ),
                            (
                                "logical",
                                Value::obj(vec![
                                    ("dtype", Value::str(c.logical_dtype.name())),
                                    ("shape", shape_value(&c.logical_shape)),
                                ]),
                            ),
                            ("name", Value::str(c.name.clone())),
                            (
                                "roles",
                                Value::map({
                                    c.roles
                                        .iter()
                                        .map(|(n, r)| {
                                            (
                                                n.clone(),
                                                Value::obj(vec![
                                                    ("bits", Value::str(b64::encode(&r.bits))),
                                                    ("dtype", Value::str(r.dtype.name())),
                                                    ("shape", shape_value(&r.shape)),
                                                ]),
                                            )
                                        })
                                        .collect()
                                }),
                            ),
                        ])
                    })
                    .collect(),
            ),
        )])
    }
}
