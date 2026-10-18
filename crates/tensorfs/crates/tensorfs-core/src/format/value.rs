//! Free-form values for metadata and scheme params: no null, integers only, codec-agnostic.

use std::collections::BTreeMap;
use std::fmt;

use serde::de::{self, MapAccess, SeqAccess, Visitor};
use serde::{Deserialize, Deserializer, Serialize, Serializer};

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Value {
    Bool(bool),
    /// Any integer in the i64 or u64 range.
    Int(i128),
    Str(String),
    List(Vec<Value>),
    Map(BTreeMap<String, Value>),
}

impl Value {
    pub fn as_u64(&self) -> Option<u64> {
        match self {
            Value::Int(i) => u64::try_from(*i).ok(),
            _ => None,
        }
    }
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Value::Str(s) => Some(s),
            _ => None,
        }
    }
}

impl From<u64> for Value {
    fn from(v: u64) -> Self {
        Value::Int(v.into())
    }
}
impl From<i64> for Value {
    fn from(v: i64) -> Self {
        Value::Int(v.into())
    }
}
impl From<&str> for Value {
    fn from(v: &str) -> Self {
        Value::Str(v.into())
    }
}
impl From<bool> for Value {
    fn from(v: bool) -> Self {
        Value::Bool(v)
    }
}

impl Serialize for Value {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        match self {
            Value::Bool(b) => s.serialize_bool(*b),
            Value::Int(i) => match (i64::try_from(*i), u64::try_from(*i)) {
                (Ok(v), _) => s.serialize_i64(v),
                (_, Ok(v)) => s.serialize_u64(v),
                _ => Err(serde::ser::Error::custom(format!("{i} is outside i64/u64"))),
            },
            Value::Str(v) => s.serialize_str(v),
            Value::List(v) => s.collect_seq(v),
            Value::Map(v) => s.collect_map(v),
        }
    }
}

impl<'de> Deserialize<'de> for Value {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Value, D::Error> {
        d.deserialize_any(ValueVisitor)
    }
}

struct ValueVisitor;

impl<'de> Visitor<'de> for ValueVisitor {
    type Value = Value;

    fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str("a bool, an integer, a string, a list or a map")
    }
    fn visit_bool<E>(self, v: bool) -> Result<Value, E> {
        Ok(Value::Bool(v))
    }
    fn visit_i64<E>(self, v: i64) -> Result<Value, E> {
        Ok(Value::Int(v.into()))
    }
    fn visit_u64<E>(self, v: u64) -> Result<Value, E> {
        Ok(Value::Int(v.into()))
    }
    fn visit_f64<E: de::Error>(self, v: f64) -> Result<Value, E> {
        Err(E::custom(format!("{v} is not an integer")))
    }
    fn visit_str<E>(self, v: &str) -> Result<Value, E> {
        Ok(Value::Str(v.into()))
    }
    fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<Value, A::Error> {
        let mut out = Vec::new();
        while let Some(v) = seq.next_element()? {
            out.push(v);
        }
        Ok(Value::List(out))
    }
    fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<Value, A::Error> {
        let mut out = BTreeMap::new();
        while let Some((k, v)) = map.next_entry()? {
            out.insert(k, v);
        }
        Ok(Value::Map(out))
    }
}
