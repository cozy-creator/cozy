//! RFC 8785 (JCS) for the documents tensorfs does not author.
//!
//! A checkpoint's construction configs are `config.json` as diffusers/transformers wrote
//! them: `beta_start: 0.00085`, `trained_betas: null`, free text. The header codec
//! (`canon.rs`, tfs-013) is integer-only, printable-ASCII and null-free on purpose and
//! cannot spell them, so each config is stored as an opaque byte string in the interoperable
//! JCS profile the hub (`internal/canonicaljson`) and cozy-runtime
//! (`cozy_runtime.internal.canonical`) already write. Writing is `serde_json_canonicalizer`;
//! reading is `serde_json` held to the profile: duplicate keys refused, depth capped, every
//! number finite and inside the I-JSON magnitude (|v| <= 2^53-1).

use serde::de::{self, DeserializeSeed, Deserializer, MapAccess, SeqAccess, Visitor};

use crate::err::{refuse, Code, Refusal, Result};

pub const DEPTH_MAX: usize = 32;
/// The I-JSON interoperable magnitude, 2^53 - 1: a value outside it refuses whatever its
/// source spelling, integer or float.
pub const NUMBER_MAX: f64 = 9_007_199_254_740_991.0;

#[derive(Debug, Clone, PartialEq)]
pub enum Json {
    Null,
    Bool(bool),
    Num(f64),
    Str(String),
    Arr(Vec<Json>),
    /// Always sorted by key in UTF-16 code-unit order, unique.
    Obj(Vec<(String, Json)>),
}

impl Json {
    /// The one object constructor: sorts at build time so no caller can hand-order keys.
    pub fn object(pairs: Vec<(String, Json)>) -> Result<Json> {
        let mut pairs = pairs;
        pairs.sort_by(|a, b| a.0.encode_utf16().cmp(b.0.encode_utf16()));
        if let Some(pair) = pairs.windows(2).find(|w| w[0].0 == w[1].0) {
            return refuse(
                Code::DUPLICATE_KEY,
                format!("key {:?} appears twice", pair[0].0),
            );
        }
        Ok(Json::Obj(pairs))
    }
}

pub fn write(v: &Json) -> Vec<u8> {
    serde_json_canonicalizer::to_vec(&value(v)).expect("a profile value always serializes")
}

fn value(v: &Json) -> serde_json::Value {
    use serde_json::Value;
    match v {
        Json::Null => Value::Null,
        Json::Bool(b) => Value::Bool(*b),
        Json::Num(f) => serde_json::Number::from_f64(*f).map_or(Value::Null, Value::Number),
        Json::Str(s) => Value::String(s.clone()),
        Json::Arr(a) => Value::Array(a.iter().map(value).collect()),
        Json::Obj(o) => Value::Object(o.iter().map(|(k, x)| (k.clone(), value(x))).collect()),
    }
}

/// Parse one JSON document (RFC 8259) into the profile: any UTF-8 text, any finite number
/// inside the interoperable magnitude, duplicate keys refused, depth capped.
pub fn parse(bytes: &[u8], max_bytes: usize) -> Result<Json> {
    if bytes.len() > max_bytes {
        return refuse(
            Code::SIZE_CAP,
            format!("{} bytes exceeds cap {}", bytes.len(), max_bytes),
        );
    }
    let mut reader = serde_json::Deserializer::from_slice(bytes);
    let value = Depth(0).deserialize(&mut reader).map_err(refusal)?;
    reader.end().map_err(|e| Refusal {
        code: Code::TRAILING_BYTES,
        detail: e.to_string(),
    })?;
    Ok(value)
}

/// A profile refusal travels through serde as a message prefixed with its code.
fn refusal(error: serde_json::Error) -> Refusal {
    let text = error.to_string();
    for code in [Code::DEPTH_CAP, Code::DUPLICATE_KEY, Code::NUMBER_RANGE] {
        if let Some(detail) = text.strip_prefix(code.as_str()) {
            return Refusal {
                code,
                detail: detail.trim_start_matches(": ").to_string(),
            };
        }
    }
    let code = if text.starts_with("number out of range") {
        Code::NUMBER_RANGE
    } else {
        Code::MALFORMED_JSON
    };
    Refusal { code, detail: text }
}

fn profile<E: de::Error>(code: Code, detail: impl std::fmt::Display) -> E {
    E::custom(format!("{}: {detail}", code.as_str()))
}

fn number<E: de::Error>(v: f64, spelled: impl std::fmt::Display) -> std::result::Result<Json, E> {
    if v.is_finite() && v.abs() <= NUMBER_MAX {
        Ok(Json::Num(v))
    } else {
        Err(profile(
            Code::NUMBER_RANGE,
            format!("{spelled} is outside the interoperable numeric range"),
        ))
    }
}

/// One value at nesting depth `.0`.
struct Depth(usize);

impl<'de> DeserializeSeed<'de> for Depth {
    type Value = Json;
    fn deserialize<D: Deserializer<'de>>(self, d: D) -> std::result::Result<Json, D::Error> {
        if self.0 > DEPTH_MAX {
            return Err(profile(Code::DEPTH_CAP, "nesting deeper than 32"));
        }
        d.deserialize_any(self)
    }
}

impl<'de> Visitor<'de> for Depth {
    type Value = Json;
    fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        f.write_str("a JSON value")
    }
    fn visit_unit<E: de::Error>(self) -> std::result::Result<Json, E> {
        Ok(Json::Null)
    }
    fn visit_bool<E: de::Error>(self, v: bool) -> std::result::Result<Json, E> {
        Ok(Json::Bool(v))
    }
    fn visit_u64<E: de::Error>(self, v: u64) -> std::result::Result<Json, E> {
        if v > NUMBER_MAX as u64 {
            return Err(profile(Code::NUMBER_RANGE, format!("{v} is outside the interoperable numeric range")));
        }
        number(v as f64, v)
    }
    fn visit_i64<E: de::Error>(self, v: i64) -> std::result::Result<Json, E> {
        if v.unsigned_abs() > NUMBER_MAX as u64 {
            return Err(profile(Code::NUMBER_RANGE, format!("{v} is outside the interoperable numeric range")));
        }
        number(v as f64, v)
    }
    fn visit_f64<E: de::Error>(self, v: f64) -> std::result::Result<Json, E> {
        number(v, v)
    }
    fn visit_str<E: de::Error>(self, v: &str) -> std::result::Result<Json, E> {
        Ok(Json::Str(v.to_string()))
    }
    fn visit_string<E: de::Error>(self, v: String) -> std::result::Result<Json, E> {
        Ok(Json::Str(v))
    }
    fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> std::result::Result<Json, A::Error> {
        let mut items = Vec::new();
        while let Some(item) = seq.next_element_seed(Depth(self.0 + 1))? {
            items.push(item);
        }
        Ok(Json::Arr(items))
    }
    fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> std::result::Result<Json, A::Error> {
        let mut pairs = Vec::new();
        while let Some(key) = map.next_key::<String>()? {
            pairs.push((key, map.next_value_seed(Depth(self.0 + 1))?));
        }
        Json::object(pairs).map_err(|r| profile(r.code, r.detail))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn canon(input: &str) -> String {
        String::from_utf8(write(&parse(input.as_bytes(), 1 << 20).unwrap())).unwrap()
    }

    #[test]
    fn numbers_take_the_ecmascript_spelling() {
        assert_eq!(canon("[0,-0,0.0,1.0,1e0,10,1E1]"), "[0,0,0,1,1,10,10]");
        assert_eq!(
            canon("[0.012,0.00085,1e-7,2.5e-5]"),
            "[0.012,0.00085,1e-7,0.000025]"
        );
        assert_eq!(canon("[9007199254740991]"), "[9007199254740991]");
        // Exact ties take the even digit, as ES, V8, Go and Python do; Rust alone rounds up.
        assert_eq!(
            canon("[-630466573175298.25,233257668322821.125,14158759043626.3125,0.5,2.5]"),
            "[-630466573175298.2,233257668322821.12,14158759043626.312,0.5,2.5]"
        );
        assert_eq!(
            parse(b"9007199254740993", 64).unwrap_err().code,
            Code::NUMBER_RANGE
        );
        assert_eq!(parse(b"1e400", 64).unwrap_err().code, Code::NUMBER_RANGE);
    }

    #[test]
    fn keys_sort_by_utf16_code_unit_and_refuse_duplicates() {
        assert_eq!(canon(r#"{"b":1,"a":2,"aa":3}"#), r#"{"a":2,"aa":3,"b":1}"#);
        // U+FF21 is one unit (0xFF21); U+1D400 is a surrogate pair starting 0xD835.
        assert_eq!(
            canon("{\"\u{1D400}\":1,\"\u{FF21}\":2}"),
            "{\"\u{1D400}\":1,\"\u{FF21}\":2}"
        );
        assert_eq!(
            parse(br#"{"a":1,"a":2}"#, 64).unwrap_err().code,
            Code::DUPLICATE_KEY
        );
    }

    #[test]
    fn strings_escape_only_what_jcs_escapes() {
        assert_eq!(
            canon(r#"["\u0041\u00e9\ud83d\ude00\/\u0001\u001f\b\f\n\r\t\"\\"]"#),
            "[\"A\u{e9}\u{1F600}/\\u0001\\u001f\\b\\f\\n\\r\\t\\\"\\\\\"]"
        );
        assert_eq!(canon("[\"\u{7f}\"]"), "[\"\u{7f}\"]");
        assert_eq!(
            parse(b"\"\x01\"", 64).unwrap_err().code,
            Code::MALFORMED_JSON
        );
        assert_eq!(
            parse(b"\"\\ud800\"", 64).unwrap_err().code,
            Code::MALFORMED_JSON
        );
    }

    #[test]
    fn whitespace_null_and_nesting() {
        assert_eq!(
            canon(" { \"x\" : [ null , true , { } , [ ] ] } "),
            r#"{"x":[null,true,{},[]]}"#
        );
        assert_eq!(parse(b"{} x", 64).unwrap_err().code, Code::TRAILING_BYTES);
        let deep = "[".repeat(40) + &"]".repeat(40);
        assert_eq!(
            parse(deep.as_bytes(), 128).unwrap_err().code,
            Code::DEPTH_CAP
        );
        assert_eq!(parse(b"[1,]", 64).unwrap_err().code, Code::MALFORMED_JSON);
    }
}
