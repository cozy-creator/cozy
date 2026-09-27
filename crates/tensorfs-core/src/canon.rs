//! The ONE canonical JSON layer: RFC 8785 (JCS) writer restricted to this schema plus a
//! strict bounded reader. Integer-only numerics, sorted keys, duplicate-key refusal,
//! printable-ASCII strings, hard depth/size caps. STORED bytes are canonical bytes.
//!
//! Divergence from the design text, recorded (decisions row): identity-bearing tensorfs
//! documents are printable-ASCII-only in every field and admit only the `\"` and `\\`
//! escapes. That is strictly stronger than "NFC for non-ASCII" and needs no Unicode
//! tables; admitting non-ASCII paths waits for a normalization table (tfs-002/tfs-015).

use crate::err::{refuse, Code, Refusal, Result};
use crate::limits;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Value {
    Bool(bool),
    Int(i64),
    Str(String),
    Arr(Vec<Value>),
    /// Always sorted by key, unique.
    Obj(Vec<(String, Value)>),
}

impl Value {
    pub fn obj(pairs: Vec<(&str, Value)>) -> Value {
        let mut v: Vec<(String, Value)> =
            pairs.into_iter().map(|(k, x)| (k.to_string(), x)).collect();
        v.sort_by(|a, b| a.0.cmp(&b.0));
        Value::Obj(v)
    }
    /// The dynamic-key form. Every map whose keys come from DATA (component names, tensor
    /// keys, role names) must be built through this: a hand-sorted `Value::Obj` is the one
    /// way a projection can disagree with the document it projects (found live at tfs-015,
    /// where the in-memory header and its parsed twin gave different tensor schema digests).
    pub fn map(pairs: Vec<(String, Value)>) -> Value {
        let mut v = pairs;
        v.sort_by(|a, b| a.0.cmp(&b.0));
        Value::Obj(v)
    }
    /// `obj` plus the fields a parse carried through unread.
    pub fn obj_with(pairs: Vec<(&str, Value)>, extra: &Extra) -> Value {
        let mut v: Vec<(String, Value)> =
            pairs.into_iter().map(|(k, x)| (k.to_string(), x)).collect();
        v.extend(extra.0.iter().cloned());
        v.sort_by(|a, b| a.0.cmp(&b.0));
        Value::Obj(v)
    }
    pub fn str(s: impl Into<String>) -> Value {
        Value::Str(s.into())
    }
    pub fn uint(u: u64) -> Value {
        Value::Int(u as i64)
    }
    pub fn arr(v: Vec<Value>) -> Value {
        Value::Arr(v)
    }
    pub fn kind(&self) -> &'static str {
        match self {
            Value::Bool(_) => "bool",
            Value::Int(_) => "int",
            Value::Str(_) => "string",
            Value::Arr(_) => "array",
            Value::Obj(_) => "object",
        }
    }
}

// ---------------------------------------------------------------- writer

pub fn write(v: &Value) -> Vec<u8> {
    let mut out = Vec::new();
    emit(v, &mut out);
    out
}

fn emit(v: &Value, out: &mut Vec<u8>) {
    match v {
        Value::Bool(b) => out.extend_from_slice(if *b { b"true" } else { b"false" }),
        Value::Int(i) => out.extend_from_slice(i.to_string().as_bytes()),
        Value::Str(s) => emit_str(s, out),
        Value::Arr(a) => {
            out.push(b'[');
            for (i, x) in a.iter().enumerate() {
                if i > 0 {
                    out.push(b',');
                }
                emit(x, out);
            }
            out.push(b']');
        }
        Value::Obj(m) => {
            out.push(b'{');
            for (i, (k, x)) in m.iter().enumerate() {
                if i > 0 {
                    out.push(b',');
                }
                emit_str(k, out);
                out.push(b':');
                emit(x, out);
            }
            out.push(b'}');
        }
    }
}

fn emit_str(s: &str, out: &mut Vec<u8>) {
    out.push(b'"');
    for c in s.bytes() {
        match c {
            b'"' => out.extend_from_slice(b"\\\""),
            b'\\' => out.extend_from_slice(b"\\\\"),
            _ => out.push(c),
        }
    }
    out.push(b'"');
}

/// Projection-only rendering. Never stored, never hashed.
pub fn pretty(v: &Value, indent: usize) -> String {
    let pad = "  ".repeat(indent);
    match v {
        Value::Obj(m) if !m.is_empty() => {
            let body: Vec<String> = m
                .iter()
                .map(|(k, x)| format!("{pad}  \"{k}\": {}", pretty(x, indent + 1)))
                .collect();
            format!("{{\n{}\n{pad}}}", body.join(",\n"))
        }
        Value::Arr(a) if !a.is_empty() => {
            let body: Vec<String> = a
                .iter()
                .map(|x| format!("{pad}  {}", pretty(x, indent + 1)))
                .collect();
            format!("[\n{}\n{pad}]", body.join(",\n"))
        }
        _ => String::from_utf8(write(v)).unwrap(),
    }
}

// ---------------------------------------------------------------- reader

struct P<'a> {
    b: &'a [u8],
    i: usize,
    depth: usize,
    depth_max: usize,
}

pub fn parse(bytes: &[u8], max_bytes: usize) -> Result<Value> {
    parse_with_depth(bytes, max_bytes, limits::DEPTH_MAX)
}

/// Projection-tool reader with an explicit depth. Stored documents always call [`parse`]
/// and retain the frozen schema depth.
pub fn parse_with_depth(bytes: &[u8], max_bytes: usize, depth_max: usize) -> Result<Value> {
    if bytes.len() > max_bytes {
        return refuse(
            Code::SIZE_CAP,
            format!("{} bytes exceeds cap {}", bytes.len(), max_bytes),
        );
    }
    let mut p = P {
        b: bytes,
        i: 0,
        depth: 0,
        depth_max,
    };
    let v = p.value()?;
    p.ws();
    if p.i != p.b.len() {
        return refuse(Code::TRAILING_BYTES, format!("at offset {}", p.i));
    }
    Ok(v)
}

/// Parse AND require the input to be exactly the canonical encoding of what it means.
pub fn parse_canonical(bytes: &[u8], max_bytes: usize) -> Result<Value> {
    let v = parse(bytes, max_bytes)?;
    if write(&v) != bytes {
        return refuse(
            Code::NONCANONICAL_ENCODING,
            "stored bytes are not the canonical encoding of their own content",
        );
    }
    Ok(v)
}

impl<'a> P<'a> {
    fn ws(&mut self) {
        while self.i < self.b.len() && matches!(self.b[self.i], b' ' | b'\t' | b'\n' | b'\r') {
            self.i += 1;
        }
    }
    fn err<T>(&self, code: Code, what: &str) -> Result<T> {
        refuse(code, format!("{what} at offset {}", self.i))
    }
    fn value(&mut self) -> Result<Value> {
        if self.depth > self.depth_max {
            return self.err(
                Code::DEPTH_CAP,
                "nesting deeper than the fixed schema depth",
            );
        }
        self.ws();
        match self.b.get(self.i) {
            None => self.err(Code::MALFORMED_JSON, "unexpected end"),
            Some(b'{') => self.object(),
            Some(b'[') => self.array(),
            Some(b'"') => Ok(Value::Str(self.string()?)),
            Some(b't') | Some(b'f') => self.boolean(),
            Some(c) if *c == b'-' || c.is_ascii_digit() => self.number(),
            Some(b'n') => self.err(Code::WRONG_TYPE, "null is not admitted"),
            _ => self.err(Code::MALFORMED_JSON, "unexpected byte"),
        }
    }
    fn boolean(&mut self) -> Result<Value> {
        for (lit, val) in [(&b"true"[..], true), (&b"false"[..], false)] {
            if self.b[self.i..].starts_with(lit) {
                self.i += lit.len();
                return Ok(Value::Bool(val));
            }
        }
        self.err(Code::MALFORMED_JSON, "bad literal")
    }
    fn number(&mut self) -> Result<Value> {
        let start = self.i;
        if self.b[self.i] == b'-' {
            self.i += 1;
        }
        let ds = self.i;
        while self.i < self.b.len() && self.b[self.i].is_ascii_digit() {
            self.i += 1;
        }
        if self.i == ds {
            return self.err(Code::MALFORMED_JSON, "number without digits");
        }
        if matches!(self.b.get(self.i), Some(b'.') | Some(b'e') | Some(b'E')) {
            return self.err(
                Code::NON_INTEGER_NUMBER,
                "fraction/exponent: the header codec is integer-only",
            );
        }
        let text = std::str::from_utf8(&self.b[start..self.i]).unwrap();
        if (text.len() > 1 && text.starts_with('0')) || text.starts_with("-0") {
            return self.err(Code::NON_INTEGER_NUMBER, "non-canonical integer spelling");
        }
        let n: i64 = text.parse().map_err(|_| Refusal {
            code: Code::NUMBER_RANGE,
            detail: format!("{text} is out of i64 range"),
        })?;
        if !(limits::INT_MIN..=limits::INT_MAX).contains(&n) {
            return self.err(
                Code::NUMBER_RANGE,
                "outside the interoperable integer range",
            );
        }
        Ok(Value::Int(n))
    }
    fn string(&mut self) -> Result<String> {
        self.i += 1; // opening quote
        let mut s = String::new();
        loop {
            let c = match self.b.get(self.i) {
                None => return self.err(Code::MALFORMED_JSON, "unterminated string"),
                Some(c) => *c,
            };
            self.i += 1;
            match c {
                b'"' => return Ok(s),
                b'\\' => {
                    let e = match self.b.get(self.i) {
                        None => return self.err(Code::MALFORMED_JSON, "unterminated escape"),
                        Some(e) => *e,
                    };
                    self.i += 1;
                    match e {
                        b'"' => s.push('"'),
                        b'\\' => s.push('\\'),
                        _ => {
                            return self.err(
                                Code::NONCANONICAL_ENCODING,
                                "only \\\" and \\\\ escapes are canonical here",
                            )
                        }
                    }
                }
                0x20..=0x7e => s.push(c as char),
                _ => {
                    return self.err(
                        Code::NON_ASCII_FIELD,
                        "fields are printable ASCII (see canon.rs divergence note)",
                    )
                }
            }
        }
    }
    fn array(&mut self) -> Result<Value> {
        self.i += 1;
        self.depth += 1;
        let mut out = Vec::new();
        self.ws();
        if self.b.get(self.i) == Some(&b']') {
            self.i += 1;
            self.depth -= 1;
            return Ok(Value::Arr(out));
        }
        loop {
            out.push(self.value()?);
            self.ws();
            match self.b.get(self.i) {
                Some(b',') => self.i += 1,
                Some(b']') => {
                    self.i += 1;
                    self.depth -= 1;
                    return Ok(Value::Arr(out));
                }
                _ => return self.err(Code::MALFORMED_JSON, "expected , or ]"),
            }
        }
    }
    fn object(&mut self) -> Result<Value> {
        self.i += 1;
        self.depth += 1;
        let mut out: Vec<(String, Value)> = Vec::new();
        self.ws();
        if self.b.get(self.i) == Some(&b'}') {
            self.i += 1;
            self.depth -= 1;
            return Ok(Value::Obj(out));
        }
        loop {
            self.ws();
            if self.b.get(self.i) != Some(&b'"') {
                return self.err(Code::MALFORMED_JSON, "expected a key");
            }
            let k = self.string()?;
            if k.len() > limits::MAX_KEY_BYTES {
                return self.err(Code::KEY_GRAMMAR, "key over the byte cap");
            }
            self.ws();
            if self.b.get(self.i) != Some(&b':') {
                return self.err(Code::MALFORMED_JSON, "expected :");
            }
            self.i += 1;
            let v = self.value()?;
            out.push((k, v));
            self.ws();
            match self.b.get(self.i) {
                Some(b',') => self.i += 1,
                Some(b'}') => {
                    self.i += 1;
                    self.depth -= 1;
                    // duplicate-key refusal BEFORE map construction, in n log n
                    out.sort_by(|a, b| a.0.cmp(&b.0));
                    if let Some(w) = out.windows(2).find(|w| w[0].0 == w[1].0) {
                        return refuse(
                            Code::DUPLICATE_KEY,
                            format!("key {:?} appears twice", w[0].0),
                        );
                    }
                    return Ok(Value::Obj(out));
                }
                _ => return self.err(Code::MALFORMED_JSON, "expected , or }"),
            }
        }
    }
}

// ---------------------------------------------------------------- field reader

/// Fields this build does not read. A stored document carries them through unchanged, so a
/// newer writer's additions keep their bytes, identity and meaning across an older reader.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Extra(Vec<(String, Value)>);

impl Extra {
    /// Every `{sha256, length}` object anywhere in the unread fields. Reachability treats
    /// them as referenced: GC never deletes what a newer writer may have named.
    pub fn refs(&self) -> Vec<crate::ids::ObjectRef> {
        fn walk(value: &Value, out: &mut Vec<crate::ids::ObjectRef>) {
            match value {
                Value::Arr(items) => items.iter().for_each(|item| walk(item, out)),
                Value::Obj(pairs) => {
                    let sha256 = pairs.iter().find_map(|(k, v)| match (k.as_str(), v) {
                        ("sha256", Value::Str(s)) => Some(s),
                        _ => None,
                    });
                    let length = pairs.iter().find_map(|(k, v)| match (k.as_str(), v) {
                        ("length", Value::Int(n)) if *n >= 0 => Some(*n as u64),
                        _ => None,
                    });
                    if let (Some(sha256), Some(length)) = (sha256, length) {
                        if crate::ids::hex64("unread reference", sha256).is_ok() {
                            out.push(crate::ids::ObjectRef {
                                sha256: sha256.clone(),
                                length,
                            });
                        }
                    }
                    pairs.iter().for_each(|(_, v)| walk(v, out));
                }
                _ => {}
            }
        }
        let mut out = Vec::new();
        self.0.iter().for_each(|(_, v)| walk(v, &mut out));
        out
    }
}

pub struct Fields<'a> {
    what: &'static str,
    left: Vec<(&'a str, &'a Value)>,
}

impl<'a> Fields<'a> {
    pub fn new(what: &'static str, v: &'a Value) -> Result<Fields<'a>> {
        match v {
            Value::Obj(m) => Ok(Fields {
                what,
                left: m.iter().map(|(k, x)| (k.as_str(), x)).collect(),
            }),
            other => refuse(
                Code::WRONG_TYPE,
                format!("{what}: expected object, got {}", other.kind()),
            ),
        }
    }
    pub fn opt(&mut self, key: &str) -> Option<&'a Value> {
        let pos = self.left.iter().position(|(k, _)| *k == key)?;
        Some(self.left.remove(pos).1)
    }
    pub fn req(&mut self, key: &str) -> Result<&'a Value> {
        match self.opt(key) {
            Some(v) => Ok(v),
            None => refuse(
                Code::MISSING_FIELD,
                format!("{}: missing {key:?}", self.what),
            ),
        }
    }
    pub fn req_str(&mut self, key: &str) -> Result<&'a str> {
        as_str(self.what, key, self.req(key)?)
    }
    pub fn req_uint(&mut self, key: &str) -> Result<u64> {
        as_uint(self.what, key, self.req(key)?)
    }
    /// End a document this build must understand completely: a command, or a record whose
    /// unread field could change what it means.
    pub fn done(self) -> Result<()> {
        match self.left.first() {
            None => Ok(()),
            Some((k, _)) => refuse(
                Code::UNKNOWN_FIELD,
                format!("{}: unknown field {k:?}", self.what),
            ),
        }
    }

    /// End a record whose unread field could keep bytes alive (a GC root): refuse, naming the
    /// newer TensorFS that wrote it, rather than drop what that version may rely on.
    pub fn done_written_by(self, writer: Option<&str>) -> Result<()> {
        let Some((k, _)) = self.left.first() else {
            return Ok(());
        };
        let by = writer.map_or_else(
            || "a newer TensorFS".to_string(),
            |writer| format!("TensorFS {writer}"),
        );
        let upgrade = writer.map_or_else(
            || "upgrade TensorFS".to_string(),
            |writer| format!("upgrade to tensorfs>={writer}"),
        );
        refuse(
            Code::UNKNOWN_FIELD,
            format!(
                "{}: field {k:?} was written by {by}; this is TensorFS {} — {upgrade}. It is \
                 refused rather than dropped because it may keep bytes alive",
                self.what,
                crate::VERSION
            ),
        )
    }

    /// End a document that tolerates additive fields, keeping them for re-emission.
    pub fn rest(self) -> Extra {
        Extra(
            self.left
                .into_iter()
                .map(|(k, v)| (k.to_string(), v.clone()))
                .collect(),
        )
    }
}

pub fn as_str<'a>(what: &str, key: &str, v: &'a Value) -> Result<&'a str> {
    match v {
        Value::Str(s) => Ok(s),
        o => refuse(
            Code::WRONG_TYPE,
            format!("{what}.{key}: expected string, got {}", o.kind()),
        ),
    }
}

pub fn as_uint(what: &str, key: &str, v: &Value) -> Result<u64> {
    match v {
        Value::Int(i) if *i >= 0 => Ok(*i as u64),
        o => refuse(
            Code::WRONG_TYPE,
            format!(
                "{what}.{key}: expected non-negative integer, got {}",
                o.kind()
            ),
        ),
    }
}

pub fn as_arr<'a>(what: &str, key: &str, v: &'a Value) -> Result<&'a Vec<Value>> {
    match v {
        Value::Arr(a) => Ok(a),
        o => refuse(
            Code::WRONG_TYPE,
            format!("{what}.{key}: expected array, got {}", o.kind()),
        ),
    }
}

pub fn as_obj<'a>(what: &str, key: &str, v: &'a Value) -> Result<&'a Vec<(String, Value)>> {
    match v {
        Value::Obj(m) => Ok(m),
        o => refuse(
            Code::WRONG_TYPE,
            format!("{what}.{key}: expected object, got {}", o.kind()),
        ),
    }
}
