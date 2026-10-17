//! Strict reads and deterministic writes, shared by every v2 document.
//!
//! Read: one JSON object, no duplicate keys, integers only (fitting i64 or u64); the typed
//! layer then ignores unknown fields. Write: sorted keys (byte order), no whitespace, integers
//! only, serde_json's string escaping (the same as Python's `json.dumps(ensure_ascii=False)`).

use std::collections::HashSet;
use std::fmt;

use serde::de::{DeserializeOwned, DeserializeSeed, Deserializer, MapAccess, SeqAccess, Visitor};
use serde::Serialize;
use serde_json::Value;

use super::{check_format, invalid, Error, Result};

/// Parse a document whose `format` tag must be `expected` (any minor).
pub(crate) fn read<T: DeserializeOwned>(bytes: &[u8], expected: &'static str) -> Result<T> {
    let bad = |e: serde_json::Error| Error::Invalid(e.to_string());
    let mut de = serde_json::Deserializer::from_slice(bytes);
    let tag = de.deserialize_map(Root).map_err(bad)?;
    de.end().map_err(bad)?;
    match tag {
        Some(found) => check_format(&found, expected)?,
        None => return invalid(format!("no `format` tag (want {expected})")),
    }
    serde_json::from_slice(bytes).map_err(bad)
}

pub(crate) fn write<T: Serialize>(doc: &T) -> Result<Vec<u8>> {
    let value = serde_json::to_value(doc).map_err(|e| Error::Invalid(e.to_string()))?;
    let mut out = Vec::with_capacity(4096);
    emit(&value, &mut out)?;
    Ok(out)
}

fn emit(v: &Value, out: &mut Vec<u8>) -> Result<()> {
    match v {
        Value::Null => out.extend_from_slice(b"null"),
        Value::Bool(b) => out.extend_from_slice(if *b { b"true" } else { b"false" }),
        Value::Number(n) if n.is_f64() => return invalid(format!("{n} is not an integer")),
        Value::Number(n) => out.extend_from_slice(n.to_string().as_bytes()),
        Value::String(s) => serde_json::to_writer(&mut *out, s).expect("a string serializes"),
        Value::Array(items) => {
            out.push(b'[');
            for (i, item) in items.iter().enumerate() {
                if i > 0 {
                    out.push(b',');
                }
                emit(item, out)?;
            }
            out.push(b']');
        }
        Value::Object(map) => {
            let mut entries: Vec<_> = map.iter().collect();
            entries.sort_unstable_by(|a, b| a.0.as_bytes().cmp(b.0.as_bytes()));
            out.push(b'{');
            for (i, (k, item)) in entries.into_iter().enumerate() {
                if i > 0 {
                    out.push(b',');
                }
                serde_json::to_writer(&mut *out, k).expect("a key serializes");
                out.push(b':');
                emit(item, out)?;
            }
            out.push(b'}');
        }
    }
    Ok(())
}

/// The root object: checked like [`Strict`], and its `format` string returned.
struct Root;

impl<'de> Visitor<'de> for Root {
    type Value = Option<String>;

    fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str("a JSON object")
    }
    fn visit_map<A: MapAccess<'de>>(
        self,
        mut map: A,
    ) -> std::result::Result<Self::Value, A::Error> {
        let (mut seen, mut tag) = (HashSet::new(), None);
        while let Some(key) = map.next_key::<String>()? {
            if !seen.insert(key.clone()) {
                return Err(serde::de::Error::custom(format!("duplicate key {key:?}")));
            }
            if key == "format" {
                tag = map.next_value::<Value>()?.as_str().map(str::to_owned);
            } else {
                map.next_value_seed(Strict)?;
            }
        }
        Ok(tag)
    }
}

/// Walks a value refusing duplicate keys and non-integer numbers, anywhere.
struct Strict;

impl<'de> DeserializeSeed<'de> for Strict {
    type Value = ();
    fn deserialize<D: Deserializer<'de>>(self, d: D) -> std::result::Result<(), D::Error> {
        d.deserialize_any(self)
    }
}

impl<'de> Visitor<'de> for Strict {
    type Value = ();

    fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str("a JSON value")
    }
    fn visit_bool<E>(self, _: bool) -> std::result::Result<(), E> {
        Ok(())
    }
    fn visit_i64<E>(self, _: i64) -> std::result::Result<(), E> {
        Ok(())
    }
    fn visit_u64<E>(self, _: u64) -> std::result::Result<(), E> {
        Ok(())
    }
    fn visit_f64<E: serde::de::Error>(self, v: f64) -> std::result::Result<(), E> {
        Err(E::custom(format!(
            "{v} is not an integer in the i64/u64 range"
        )))
    }
    fn visit_str<E>(self, _: &str) -> std::result::Result<(), E> {
        Ok(())
    }
    fn visit_unit<E>(self) -> std::result::Result<(), E> {
        Ok(())
    }
    fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> std::result::Result<(), A::Error> {
        while seq.next_element_seed(Strict)?.is_some() {}
        Ok(())
    }
    fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> std::result::Result<(), A::Error> {
        let mut seen = HashSet::new();
        while let Some(key) = map.next_key::<String>()? {
            if !seen.insert(key.clone()) {
                return Err(serde::de::Error::custom(format!("duplicate key {key:?}")));
            }
            map.next_value_seed(Strict)?;
        }
        Ok(())
    }
}
