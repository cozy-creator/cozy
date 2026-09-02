//! RFC 8785 (JCS) for the documents tensorfs does not author.
//!
//! A checkpoint's construction configs are `config.json` as diffusers/transformers wrote
//! them: `beta_start: 0.00085`, `trained_betas: null`, free text. The header codec
//! (`canon.rs`, tfs-013) is integer-only, printable-ASCII and null-free on purpose and
//! cannot spell them, so each config is stored as an opaque byte string in the interoperable
//! JCS profile the hub (`internal/canonicaljson`) and cozy-runtime
//! (`cozy_runtime.internal.canonical`) already write: keys in UTF-16 code-unit order, no
//! whitespace, ECMAScript shortest round-trip numbers inside the I-JSON magnitude
//! (|v| <= 2^53-1), `\uXXXX` (lowercase) only below U+0020, everything else raw UTF-8.
//! Three writers, one byte string, one digest.

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

// ---------------------------------------------------------------- writer

pub fn write(v: &Json) -> Vec<u8> {
    let mut out = String::new();
    emit(v, &mut out);
    out.into_bytes()
}

fn emit(v: &Json, out: &mut String) {
    match v {
        Json::Null => out.push_str("null"),
        Json::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
        Json::Num(f) => out.push_str(&number_es(*f)),
        Json::Str(s) => emit_str(s, out),
        Json::Arr(a) => {
            out.push('[');
            for (i, x) in a.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                emit(x, out);
            }
            out.push(']');
        }
        Json::Obj(o) => {
            out.push('{');
            for (i, (k, x)) in o.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                emit_str(k, out);
                out.push(':');
                emit(x, out);
            }
            out.push('}');
        }
    }
}

fn emit_str(s: &str, out: &mut String) {
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\u{8}' => out.push_str("\\b"),
            '\u{c}' => out.push_str("\\f"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
}

/// ECMAScript `Number::toString` (ES6 7.1.12.1), RFC 8785's number rule, over the SHORTEST
/// round-tripping digit string — never the value's exact binary expansion. There is no
/// integer fast path on purpose: one rule, one spelling, the same in Go and Python.
fn number_es(v: f64) -> String {
    if v == 0.0 {
        return "0".to_string(); // covers -0: the profile has one zero.
    }
    let (sign, v) = if v < 0.0 { ("-", -v) } else { ("", v) };
    let (digits, e) = shortest_digits(v);
    let k = digits.len() as i32;
    let n = e + k; // value = 0.<digits> x 10^n
    let body = if k <= n && n <= 21 {
        format!("{digits}{}", "0".repeat((n - k) as usize))
    } else if 0 < n && n <= 21 {
        let (int, frac) = digits.split_at(n as usize);
        format!("{int}.{frac}")
    } else if -6 < n && n <= 0 {
        format!("0.{}{digits}", "0".repeat((-n) as usize))
    } else {
        let (first, rest) = digits.split_at(1);
        let mantissa = if rest.is_empty() {
            first.to_string()
        } else {
            format!("{first}.{rest}")
        };
        format!(
            "{mantissa}e{}{}",
            if n > 0 { "+" } else { "-" },
            (n - 1).abs()
        )
    };
    format!("{sign}{body}")
}

/// The shortest round-tripping digits of a positive finite double, as (digits, exponent of
/// the last digit): v = digits x 10^e, no leading or trailing zeros.
///
/// Rust's shortest formatter and ES agree everywhere except an exact tie — a double that
/// sits exactly halfway between two equally short decimals (630466573175298.25 between
/// .2 and .3; every double in [2^49, 2^53) whose fraction is .25 or .75 is one). ES, V8,
/// Go's strconv and Python's repr all take the EVEN digit; Rust takes the upper one. Ties
/// are decided here in exact integer arithmetic, so the three writers spell one string.
fn shortest_digits(v: f64) -> (String, i32) {
    let shortest = format!("{v:e}"); // d[.ddd]e[-]dd
    let (mantissa, exponent) = shortest.split_once('e').expect("LowerExp form");
    let digits = mantissa.replacen('.', "", 1);
    let exponent: i32 = exponent.parse().expect("LowerExp exponent");
    let mut d: u64 = digits.parse().expect("at most 17 digits");
    let mut e = exponent - (digits.len() as i32 - 1);
    if d % 2 == 1 {
        if let Some(even) = even_tie_partner(v, d, e) {
            d = even;
        }
    }
    while d.is_multiple_of(10) {
        d /= 10;
        e += 1;
    }
    (d.to_string(), e)
}

/// The even neighbour of `d` (d-1 or d+1) when `v` is exactly the midpoint between the two
/// and the neighbour round-trips; `None` when `d` is not on a tie.
fn even_tie_partner(v: f64, d: u64, e: i32) -> Option<u64> {
    // v = m x 2^q exactly; ties need a terminating decimal expansion short enough to
    // compete with the shortest digits, which bounds q to a few bits below zero.
    let bits = v.to_bits();
    let biased = ((bits >> 52) & 0x7ff) as i32;
    let fraction = bits & ((1u64 << 52) - 1);
    let (mut m, mut q) = if biased == 0 {
        (fraction, -1074)
    } else {
        (fraction | (1u64 << 52), biased - 1075)
    };
    if q >= 0 {
        return None; // an integer below 2^53 has no shorter competitor
    }
    let reduce = m.trailing_zeros().min((-q) as u32);
    m >>= reduce;
    q += reduce as i32;
    if q >= 0 || -q > 20 || e + (-q) < 0 {
        return None;
    }
    let n = (-q) as u32; // v = exact / 10^n
    let exact = (m as u128).checked_mul(5u128.pow(n))?;
    let scale = 10u128.checked_pow((e + n as i32) as u32)?;
    for c in [d.checked_sub(1)?, d + 1] {
        // midpoint of d and c is (d + c) / 2 x 10^e; tie iff exact x 2 == (d + c) x 10^(n + e)
        let midpoint = (d as u128 + c as u128).checked_mul(scale)?;
        if exact.checked_mul(2)? != midpoint {
            continue;
        }
        let parsed: f64 = format!("{c}e{e}").parse().ok()?;
        if parsed == v {
            return Some(c);
        }
    }
    None
}

// ---------------------------------------------------------------- reader

/// Parse one JSON document (RFC 8259) into the profile: any UTF-8 text, any finite number
/// inside the interoperable magnitude, duplicate keys refused, depth capped.
pub fn parse(bytes: &[u8], max_bytes: usize) -> Result<Json> {
    if bytes.len() > max_bytes {
        return refuse(
            Code::SIZE_CAP,
            format!("{} bytes exceeds cap {}", bytes.len(), max_bytes),
        );
    }
    let text = std::str::from_utf8(bytes).map_err(|e| Refusal {
        code: Code::MALFORMED_JSON,
        detail: format!("not UTF-8: {e}"),
    })?;
    let mut p = P {
        b: text.as_bytes(),
        i: 0,
        depth: 0,
    };
    let v = p.value()?;
    p.ws();
    if p.i != p.b.len() {
        return refuse(Code::TRAILING_BYTES, format!("at offset {}", p.i));
    }
    Ok(v)
}

struct P<'a> {
    b: &'a [u8],
    i: usize,
    depth: usize,
}

impl P<'_> {
    fn ws(&mut self) {
        while self.i < self.b.len() && matches!(self.b[self.i], b' ' | b'\t' | b'\n' | b'\r') {
            self.i += 1;
        }
    }
    fn err<T>(&self, code: Code, what: &str) -> Result<T> {
        refuse(code, format!("{what} at offset {}", self.i))
    }
    fn value(&mut self) -> Result<Json> {
        if self.depth > DEPTH_MAX {
            return self.err(Code::DEPTH_CAP, "nesting deeper than 32");
        }
        self.ws();
        match self.b.get(self.i) {
            None => self.err(Code::MALFORMED_JSON, "unexpected end"),
            Some(b'{') => self.object(),
            Some(b'[') => self.array(),
            Some(b'"') => Ok(Json::Str(self.string()?)),
            Some(b'-') | Some(b'0'..=b'9') => self.number(),
            Some(_) => self.literal(),
        }
    }
    fn literal(&mut self) -> Result<Json> {
        for (lit, val) in [
            (&b"true"[..], Json::Bool(true)),
            (&b"false"[..], Json::Bool(false)),
            (&b"null"[..], Json::Null),
        ] {
            if self.b[self.i..].starts_with(lit) {
                self.i += lit.len();
                return Ok(val);
            }
        }
        self.err(Code::MALFORMED_JSON, "unexpected byte")
    }
    fn digits(&mut self) -> usize {
        let start = self.i;
        while self.i < self.b.len() && self.b[self.i].is_ascii_digit() {
            self.i += 1;
        }
        self.i - start
    }
    fn number(&mut self) -> Result<Json> {
        let start = self.i;
        if self.b[self.i] == b'-' {
            self.i += 1;
        }
        match self.b.get(self.i) {
            Some(b'0') => self.i += 1,
            Some(b'1'..=b'9') => {
                self.digits();
            }
            _ => return self.err(Code::MALFORMED_JSON, "number without digits"),
        }
        if self.b.get(self.i) == Some(&b'.') {
            self.i += 1;
            if self.digits() == 0 {
                return self.err(Code::MALFORMED_JSON, "fraction without digits");
            }
        }
        if matches!(self.b.get(self.i), Some(b'e') | Some(b'E')) {
            self.i += 1;
            if matches!(self.b.get(self.i), Some(b'+') | Some(b'-')) {
                self.i += 1;
            }
            if self.digits() == 0 {
                return self.err(Code::MALFORMED_JSON, "exponent without digits");
            }
        }
        let text = std::str::from_utf8(&self.b[start..self.i]).expect("ASCII number");
        let value: f64 = text.parse().map_err(|_| Refusal {
            code: Code::MALFORMED_JSON,
            detail: format!("number {text}"),
        })?;
        if !value.is_finite() || value.abs() > NUMBER_MAX {
            return self.err(
                Code::NUMBER_RANGE,
                &format!("{text} is outside the interoperable numeric range"),
            );
        }
        Ok(Json::Num(value))
    }
    fn hex4(&mut self) -> Result<u32> {
        let end = self.i + 4;
        let slice = match self.b.get(self.i..end) {
            Some(s) => s,
            None => return self.err(Code::MALFORMED_JSON, "short \\u escape"),
        };
        let text = std::str::from_utf8(slice)
            .ok()
            .filter(|t| t.bytes().all(|c| c.is_ascii_hexdigit()));
        let value = match text.and_then(|t| u32::from_str_radix(t, 16).ok()) {
            Some(v) => v,
            None => return self.err(Code::MALFORMED_JSON, "bad \\u escape"),
        };
        self.i = end;
        Ok(value)
    }
    fn string(&mut self) -> Result<String> {
        self.i += 1; // opening quote
        let mut s = String::new();
        loop {
            let run = self.i;
            while self.i < self.b.len() && !matches!(self.b[self.i], b'"' | b'\\') {
                if self.b[self.i] < 0x20 {
                    return self.err(Code::MALFORMED_JSON, "raw control character");
                }
                self.i += 1;
            }
            // Runs split only at ASCII bytes, so every run is whole UTF-8.
            s.push_str(std::str::from_utf8(&self.b[run..self.i]).expect("validated UTF-8"));
            match self.b.get(self.i) {
                None => return self.err(Code::MALFORMED_JSON, "unterminated string"),
                Some(b'"') => {
                    self.i += 1;
                    return Ok(s);
                }
                Some(_) => self.i += 1, // backslash
            }
            let c = match self.b.get(self.i) {
                None => return self.err(Code::MALFORMED_JSON, "unterminated escape"),
                Some(c) => *c,
            };
            self.i += 1;
            match c {
                b'"' => s.push('"'),
                b'\\' => s.push('\\'),
                b'/' => s.push('/'),
                b'b' => s.push('\u{8}'),
                b'f' => s.push('\u{c}'),
                b'n' => s.push('\n'),
                b'r' => s.push('\r'),
                b't' => s.push('\t'),
                b'u' => {
                    let mut code = self.hex4()?;
                    if (0xD800..0xDC00).contains(&code) {
                        if self.b.get(self.i..self.i + 2) != Some(b"\\u") {
                            return self.err(Code::MALFORMED_JSON, "lone high surrogate");
                        }
                        self.i += 2;
                        let low = self.hex4()?;
                        if !(0xDC00..0xE000).contains(&low) {
                            return self.err(Code::MALFORMED_JSON, "bad low surrogate");
                        }
                        code = 0x10000 + ((code - 0xD800) << 10) + (low - 0xDC00);
                    }
                    match char::from_u32(code) {
                        Some(ch) => s.push(ch),
                        None => return self.err(Code::MALFORMED_JSON, "lone low surrogate"),
                    }
                }
                _ => return self.err(Code::MALFORMED_JSON, "bad escape"),
            }
        }
    }
    fn array(&mut self) -> Result<Json> {
        self.i += 1;
        self.depth += 1;
        let mut items = Vec::new();
        self.ws();
        if self.b.get(self.i) == Some(&b']') {
            self.i += 1;
            self.depth -= 1;
            return Ok(Json::Arr(items));
        }
        loop {
            items.push(self.value()?);
            self.ws();
            match self.b.get(self.i) {
                Some(b',') => self.i += 1,
                Some(b']') => {
                    self.i += 1;
                    self.depth -= 1;
                    return Ok(Json::Arr(items));
                }
                _ => return self.err(Code::MALFORMED_JSON, "expected , or ]"),
            }
        }
    }
    fn object(&mut self) -> Result<Json> {
        self.i += 1;
        self.depth += 1;
        let mut pairs = Vec::new();
        self.ws();
        if self.b.get(self.i) == Some(&b'}') {
            self.i += 1;
            self.depth -= 1;
            return Json::object(pairs);
        }
        loop {
            self.ws();
            if self.b.get(self.i) != Some(&b'"') {
                return self.err(Code::MALFORMED_JSON, "expected string key");
            }
            let key = self.string()?;
            self.ws();
            if self.b.get(self.i) != Some(&b':') {
                return self.err(Code::MALFORMED_JSON, "expected :");
            }
            self.i += 1;
            let value = self.value()?;
            pairs.push((key, value));
            self.ws();
            match self.b.get(self.i) {
                Some(b',') => self.i += 1,
                Some(b'}') => {
                    self.i += 1;
                    self.depth -= 1;
                    return Json::object(pairs);
                }
                _ => return self.err(Code::MALFORMED_JSON, "expected , or }"),
            }
        }
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
