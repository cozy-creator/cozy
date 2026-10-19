//! Free-form values for metadata and scheme params (no null, integers only), and config values
//! (which add floats and null). Codec-agnostic.

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

/// Strings quoted, lists `[a, b]`, maps `{k: v}` with bare keys: for reading, not parsing.
impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Value::Bool(b) => write!(f, "{b}"),
            Value::Int(i) => write!(f, "{i}"),
            Value::Str(s) => f.write_str(&serde_json::to_string(s).map_err(|_| fmt::Error)?),
            Value::List(items) => {
                f.write_str("[")?;
                for (i, v) in items.iter().enumerate() {
                    write!(f, "{}{v}", if i > 0 { ", " } else { "" })?;
                }
                f.write_str("]")
            }
            Value::Map(map) => {
                f.write_str("{")?;
                for (i, (k, v)) in map.iter().enumerate() {
                    write!(f, "{}{k}: {v}", if i > 0 { ", " } else { "" })?;
                }
                f.write_str("}")
            }
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

/// A config's value: a [`Value`], or a float or null. Floats and null appear only in configs.
/// Floats are equal when their encodings are (`-0.0 != 0.0`, every NaN equal).
#[derive(Debug, Clone)]
pub enum ConfigValue {
    Null,
    Bool(bool),
    Int(i128),
    Float(f64),
    Str(String),
    List(Vec<ConfigValue>),
    Map(BTreeMap<String, ConfigValue>),
}

/// A structured config: `config.json` as parsed JSON, never a file.
pub type Config = BTreeMap<String, ConfigValue>;

impl PartialEq for ConfigValue {
    fn eq(&self, other: &ConfigValue) -> bool {
        use ConfigValue::*;
        match (self, other) {
            (Null, Null) => true,
            (Bool(a), Bool(b)) => a == b,
            (Int(a), Int(b)) => a == b,
            (Float(a), Float(b)) => a.to_bits() == b.to_bits() || (a.is_nan() && b.is_nan()),
            (Str(a), Str(b)) => a == b,
            (List(a), List(b)) => a == b,
            (Map(a), Map(b)) => a == b,
            _ => false,
        }
    }
}

impl Serialize for ConfigValue {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        match self {
            ConfigValue::Null => s.serialize_unit(),
            ConfigValue::Bool(b) => s.serialize_bool(*b),
            ConfigValue::Int(i) => Value::Int(*i).serialize(s),
            ConfigValue::Float(f) => s.serialize_f64(*f),
            ConfigValue::Str(v) => s.serialize_str(v),
            ConfigValue::List(v) => s.collect_seq(v),
            ConfigValue::Map(v) => s.collect_map(v),
        }
    }
}

impl<'de> Deserialize<'de> for ConfigValue {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<ConfigValue, D::Error> {
        d.deserialize_any(ConfigVisitor)
    }
}

struct ConfigVisitor;

impl<'de> Visitor<'de> for ConfigVisitor {
    type Value = ConfigValue;

    fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str("a config value")
    }
    fn visit_unit<E>(self) -> Result<ConfigValue, E> {
        Ok(ConfigValue::Null)
    }
    fn visit_none<E>(self) -> Result<ConfigValue, E> {
        Ok(ConfigValue::Null)
    }
    fn visit_bool<E>(self, v: bool) -> Result<ConfigValue, E> {
        Ok(ConfigValue::Bool(v))
    }
    fn visit_i64<E>(self, v: i64) -> Result<ConfigValue, E> {
        Ok(ConfigValue::Int(v.into()))
    }
    fn visit_u64<E>(self, v: u64) -> Result<ConfigValue, E> {
        Ok(ConfigValue::Int(v.into()))
    }
    fn visit_f64<E>(self, v: f64) -> Result<ConfigValue, E> {
        Ok(ConfigValue::Float(v))
    }
    fn visit_str<E>(self, v: &str) -> Result<ConfigValue, E> {
        Ok(ConfigValue::Str(v.into()))
    }
    fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<ConfigValue, A::Error> {
        let mut out = Vec::new();
        while let Some(v) = seq.next_element()? {
            out.push(v);
        }
        Ok(ConfigValue::List(out))
    }
    fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<ConfigValue, A::Error> {
        let mut out = BTreeMap::new();
        while let Some((k, v)) = map.next_entry()? {
            out.insert(k, v);
        }
        Ok(ConfigValue::Map(out))
    }
}
