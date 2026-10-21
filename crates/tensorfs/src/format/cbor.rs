//! The header's bytes (spec §9, `conformance/schema/header.cddl`): RFC 8949 core deterministic
//! CBOR, read strictly. Allowed: unsigned and negative integers, byte and text strings, arrays,
//! maps with text keys, `true`, `false`, and inside configs only finite floats (in their
//! shortest exact form) and `null`; every head in its shortest form, definite lengths only, map keys in
//! bytewise order of their encodings. A header has one encoding, so decoding and re-encoding
//! reproduces its bytes. A reader skips, still strictly, any elements after the fields it knows
//! at the end of the root and of every record: later versions append fields.

use std::collections::{BTreeMap, HashSet};

use super::{
    check_format, invalid, Body, ClassRef, Component, ConfigValue, Dtype, Encoding, Error, Header,
    Location, Metadata, ObjectRef, Part, Pipeline, Result, Tensor, Value, CONFIG_FILE, HEADER_FORMAT,
};

/// The deepest nesting a value may use.
const DEPTH: usize = 64;

const UINT: u8 = 0;
const NINT: u8 = 1;
const BYTES: u8 = 2;
const TEXT: u8 = 3;
const ARRAY: u8 = 4;
const MAP: u8 = 5;
const SIMPLE: u8 = 7;
const FALSE: u8 = 0xf4;
const TRUE: u8 = 0xf5;
const NULL: u8 = 0xf6;

struct Writer(Vec<u8>);

impl Writer {
    fn head(&mut self, major: u8, n: u64) {
        let m = major << 5;
        match n {
            0..=23 => self.0.push(m | n as u8),
            24..=0xff => self.0.extend([m | 24, n as u8]),
            0x100..=0xffff => {
                self.0.push(m | 25);
                self.0.extend((n as u16).to_be_bytes());
            }
            0x1_0000..=0xffff_ffff => {
                self.0.push(m | 26);
                self.0.extend((n as u32).to_be_bytes());
            }
            _ => {
                self.0.push(m | 27);
                self.0.extend(n.to_be_bytes());
            }
        }
    }
    fn uint(&mut self, n: u64) {
        self.head(UINT, n)
    }
    fn text(&mut self, s: &str) {
        self.head(TEXT, s.len() as u64);
        self.0.extend(s.as_bytes());
    }
    fn array(&mut self, n: usize) {
        self.head(ARRAY, n as u64)
    }
    fn shape(&mut self, shape: &[u64]) {
        self.array(shape.len());
        shape.iter().for_each(|n| self.uint(*n));
    }
    fn location(&mut self, at: &Location) {
        self.array(3);
        self.uint(at.chunk);
        self.uint(at.offset);
        self.uint(at.length);
    }
    fn class(&mut self, c: &ClassRef) {
        self.array(3);
        [&c.library, &c.version, &c.name].into_iter().for_each(|t| self.text(t));
    }
    fn object(&mut self, r: &ObjectRef) {
        self.array(2);
        self.head(BYTES, 32);
        let hex = r.sha256.as_bytes();
        self.0.extend(hex.chunks(2).map(|d| nibble(d[0]) << 4 | nibble(d[1])));
        self.uint(r.length);
    }
    fn int(&mut self, i: i128) -> Result<()> {
        match (u64::try_from(i), i64::try_from(i)) {
            (Ok(n), _) => self.uint(n),
            (_, Ok(n)) => self.head(NINT, !(n as u64)),
            _ => return invalid(format!("{i} is outside [-2^63, 2^64)")),
        }
        Ok(())
    }
    fn value(&mut self, v: &Value) -> Result<()> {
        match v {
            Value::Bool(b) => self.0.push(if *b { TRUE } else { FALSE }),
            Value::Int(i) => self.int(*i)?,
            Value::Str(s) => self.text(s),
            Value::List(items) => {
                self.array(items.len());
                items.iter().try_for_each(|item| self.value(item))?;
            }
            Value::Map(map) => self.map(map, Writer::value)?,
        }
        Ok(())
    }
    fn config(&mut self, v: &ConfigValue) -> Result<()> {
        match v {
            ConfigValue::Null => self.0.push(NULL),
            ConfigValue::Bool(b) => self.0.push(if *b { TRUE } else { FALSE }),
            ConfigValue::Number(n) => match (n.as_u64(), n.as_i64(), n.as_f64()) {
                (Some(u), _, _) => self.uint(u),
                (_, Some(i), _) => self.int(i.into())?,
                (_, _, Some(f)) => self.float(f),
                _ => return invalid(format!("config number {n}")),
            },
            ConfigValue::String(s) => self.text(s),
            ConfigValue::Array(items) => {
                self.array(items.len());
                items.iter().try_for_each(|item| self.config(item))?;
            }
            ConfigValue::Object(map) => self.map(map, Writer::config)?,
        }
        Ok(())
    }
    /// The shortest of half, single and double that holds a finite `f` exactly.
    fn float(&mut self, f: f64) {
        if let Some(h) = half(f) {
            self.0.push(0xf9);
            self.0.extend(h.to_be_bytes());
        } else if f as f32 as f64 == f {
            self.0.push(0xfa);
            self.0.extend((f as f32).to_bits().to_be_bytes());
        } else {
            self.0.push(0xfb);
            self.0.extend(f.to_bits().to_be_bytes());
        }
    }
    /// Text keys in bytewise order of their encodings: shorter first, then by bytes.
    fn map<'m, V: 'm>(
        &mut self,
        map: impl IntoIterator<Item = (&'m String, &'m V)>,
        mut item: impl FnMut(&mut Self, &V) -> Result<()>,
    ) -> Result<()> {
        let mut entries: Vec<(&String, &V)> = map.into_iter().collect();
        entries.sort_by(|a, b| (a.0.len(), a.0.as_bytes()).cmp(&(b.0.len(), b.0.as_bytes())));
        self.head(MAP, entries.len() as u64);
        for (k, v) in entries {
            self.text(k);
            item(self, v)?;
        }
        Ok(())
    }
}

/// A finite `f` as an IEEE half, when that is exact.
fn half(f: f64) -> Option<u16> {
    let sign = if f.is_sign_negative() { 0x8000 } else { 0 };
    let a = f.abs();
    let e = (a.to_bits() >> 52) as i32 - 1023;
    let (bits, m) = match e {
        16.. => return None,
        -14..=15 => (((e + 15) as u16) << 10, a * 2f64.powi(10 - e) - 1024.0),
        _ => (0, a * 2f64.powi(24)),
    };
    (m.fract() == 0.0 && m < 1024.0).then_some(sign | bits | m as u16)
}

/// An IEEE half's value; NaN for an infinity or NaN.
fn from_half(h: u16) -> f64 {
    let (e, m) = (i32::from(h >> 10 & 31), f64::from(h & 1023));
    let a = match e {
        0 => m * 2f64.powi(-24),
        31 => f64::NAN,
        _ => (1024.0 + m) * 2f64.powi(e - 25),
    };
    if h >> 15 == 1 { -a } else { a }
}

pub(crate) fn write(h: &Header) -> Result<Vec<u8>> {
    let encodings = h.encodings();
    let index = |e: &Encoding| encodings.iter().position(|x| *x == e).unwrap() as u64 + 1;
    let mut w = Writer(Vec::with_capacity(256 * 1024));
    w.array(6);
    w.text(HEADER_FORMAT);
    w.array(h.chunks.len());
    h.chunks.iter().for_each(|c| w.object(c));
    w.array(encodings.len());
    for e in &encodings {
        w.array(2);
        w.text(&e.scheme);
        w.map(&e.params, Writer::value)?;
    }
    match h.pipeline() {
        None => w.array(0),
        Some(p) => {
            w.array(2);
            w.class(&p.class);
            w.map(&p.config, Writer::config)?;
        }
    }
    w.array(h.components().len());
    for c in h.components() {
        w.array(5 + c.config_file.is_some() as usize);
        w.text(&c.name);
        w.class(&c.class);
        w.map(&c.config, Writer::config)?;
        w.array(c.tensors.len());
        for t in &c.tensors {
            w.array(5);
            w.text(&t.key);
            w.text(t.dtype.name());
            w.shape(&t.shape);
            match &t.body {
                Body::Data(at) => {
                    w.uint(0);
                    w.location(at);
                }
                Body::Encoded { encoding, parts } => {
                    w.uint(index(encoding));
                    w.array(parts.len());
                    for p in parts {
                        w.array(4);
                        w.text(&p.role);
                        w.text(p.dtype.name());
                        w.shape(&p.shape);
                        w.location(&p.at());
                    }
                }
            }
        }
        w.map(&c.assets, |w, r| { w.object(r); Ok(()) })?;
        if let Some(file) = &c.config_file {
            w.text(file);
        }
    }
    w.map(&h.metadata, Writer::value)?;
    Ok(w.0)
}

fn hex(bytes: &[u8]) -> String {
    const DIGITS: &[u8; 16] = b"0123456789abcdef";
    let digits = bytes
        .iter()
        .flat_map(|b| [DIGITS[(b >> 4) as usize], DIGITS[(b & 15) as usize]]);
    String::from_utf8(digits.collect()).expect("hex is ASCII")
}

/// A lowercase hex digit's value ([`ObjectRef`] holds only those).
fn nibble(c: u8) -> u8 {
    match c {
        b'0'..=b'9' => c - b'0',
        _ => c - b'a' + 10,
    }
}

struct Reader<'a> {
    b: &'a [u8],
    at: usize,
}

impl<'a> Reader<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        match self.b.get(self.at..).and_then(|rest| rest.get(..n)) {
            Some(s) => {
                self.at += n;
                Ok(s)
            }
            None => invalid(format!("truncated at byte {}", self.at)),
        }
    }

    /// A head: its major type and argument, in shortest form, never indefinite.
    fn head(&mut self) -> Result<(u8, u64)> {
        let at = self.at;
        let first = self.take(1)?[0];
        let (major, info) = (first >> 5, first & 31);
        match first {
            FALSE | TRUE => return Ok((SIMPLE, (first == TRUE) as u64)),
            NULL => return invalid(format!("null outside a config at byte {at}")),
            0xf7 => return invalid(format!("undefined at byte {at}")),
            0xf9..=0xfb => return invalid(format!("a float outside a config at byte {at}")),
            _ if major == SIMPLE => {
                return invalid(format!("simple value {first:#x} at byte {at}"));
            }
            _ if major == 6 => return invalid(format!("a tag at byte {at}")),
            _ => {}
        }
        let mut wide = |n: usize| -> Result<u64> {
            Ok(self.take(n)?.iter().fold(0, |a, b| a << 8 | *b as u64))
        };
        let (n, min) = match info {
            0..=23 => (info as u64, 0),
            24 => (wide(1)?, 24),
            25 => (wide(2)?, 0x100),
            26 => (wide(4)?, 0x1_0000),
            27 => (wide(8)?, 0x1_0000_0000),
            31 => return invalid(format!("an indefinite length at byte {at}")),
            _ => return invalid(format!("reserved head {first:#x} at byte {at}")),
        };
        if n < min {
            return invalid(format!("a non-preferred head at byte {at}"));
        }
        Ok((major, n))
    }

    fn expect(&mut self, major: u8, what: &str) -> Result<u64> {
        let at = self.at;
        match self.head()? {
            (m, n) if m == major => Ok(n),
            _ => invalid(format!("{what}: wrong type at byte {at}")),
        }
    }
    fn uint(&mut self, what: &str) -> Result<u64> {
        self.expect(UINT, what)
    }
    fn text(&mut self, what: &str) -> Result<&'a str> {
        let n = self.expect(TEXT, what)?;
        let at = self.at;
        std::str::from_utf8(self.take(len(n)?)?)
            .or_else(|_| invalid(format!("{what}: invalid UTF-8 at byte {at}")))
    }
    /// `[library, version, name]`, exactly.
    fn class(&mut self, what: &str) -> Result<ClassRef> {
        if self.list(what)? != 3 {
            return invalid(format!("{what}: not [library, version, name]"));
        }
        let mut text = || self.text(what).map(str::to_owned);
        Ok(ClassRef { library: text()?, version: text()?, name: text()? })
    }
    /// A record of at least `n` fields; the count of fields after them, to [`Reader::skip`].
    fn record(&mut self, n: u64, what: &str) -> Result<u64> {
        match self.expect(ARRAY, what)? {
            got if got >= n => Ok(got - n),
            got => invalid(format!("{what}: {got} fields, not at least {n}")),
        }
    }

    /// Passes over `n` well-formed items of any allowed kind.
    fn skip(&mut self, n: u64, depth: usize) -> Result<()> {
        if depth > DEPTH {
            return invalid("items nested too deep");
        }
        for _ in 0..n {
            let at = self.at;
            match self.head()? {
                (BYTES, n) => drop(self.take(len(n)?)?),
                (TEXT, _) => {
                    self.at = at;
                    self.text("a string")?;
                }
                (ARRAY, n) => self.skip(n, depth + 1)?,
                (MAP, n) => {
                    let mut previous: &[u8] = &[];
                    for _ in 0..n {
                        let start = self.at;
                        let key = self.text("a map key")?;
                        if previous >= &self.b[start..self.at] {
                            return invalid(format!("map key {key:?} is out of order or repeated"));
                        }
                        previous = &self.b[start..self.at];
                        self.skip(1, depth + 1)?;
                    }
                }
                (NINT, n) if n > i64::MAX as u64 => {
                    return invalid(format!("an integer below -2^63 at byte {at}"));
                }
                _ => {}
            }
        }
        Ok(())
    }
    fn list(&mut self, what: &str) -> Result<u64> {
        self.expect(ARRAY, what)
    }
    fn shape(&mut self) -> Result<Vec<u64>> {
        let n = self.list("a shape")?;
        (0..n).map(|_| self.uint("a shape extent")).collect()
    }
    /// A dtype name; an unknown one is a newer dtype, read raw.
    fn dtype(&mut self) -> Result<Dtype> {
        match self.text("a dtype")? {
            "" => invalid("an empty dtype"),
            name => Ok(Dtype::parse(name)),
        }
    }
    fn object(&mut self, what: &str) -> Result<ObjectRef> {
        let fields = self.record(2, what)?;
        let sha = self.expect(BYTES, what)?;
        if sha != 32 {
            return invalid(format!("{what}: its sha256 is {sha} bytes"));
        }
        let sha256 = hex(self.take(32)?);
        let length = self.uint(what)?;
        self.skip(fields, 2)?;
        Ok(ObjectRef { sha256, length })
    }
    fn location(&mut self) -> Result<Location> {
        let more = self.record(3, "a location")?;
        let at = Location {
            chunk: self.uint("a chunk index")?,
            offset: self.uint("an offset")?,
            length: self.uint("a length")?,
        };
        self.skip(more, 2)?;
        Ok(at)
    }

    fn value(&mut self, depth: usize) -> Result<Value> {
        if depth > DEPTH {
            return invalid("values nested too deep");
        }
        let at = self.at;
        let (major, n) = self.head()?;
        Ok(match major {
            UINT => Value::Int(n.into()),
            NINT if n <= i64::MAX as u64 => Value::Int(-1 - i128::from(n)),
            NINT => return invalid(format!("an integer below -2^63 at byte {at}")),
            TEXT => {
                self.at = at;
                Value::Str(self.text("a string")?.to_owned())
            }
            ARRAY => Value::List(
                (0..n)
                    .map(|_| self.value(depth + 1))
                    .collect::<Result<_>>()?,
            ),
            MAP => {
                self.at = at;
                Value::Map(self.map(depth, Reader::value)?)
            }
            SIMPLE => Value::Bool(n == 1),
            _ => return invalid(format!("a byte string value at byte {at}")),
        })
    }

    /// A config value: a value, or a finite float in its shortest exact form, or null.
    fn config(&mut self, depth: usize) -> Result<ConfigValue> {
        if depth > DEPTH {
            return invalid("values nested too deep");
        }
        let at = self.at;
        match self.b.get(at).copied() {
            Some(NULL) => {
                self.at += 1;
                return Ok(ConfigValue::Null);
            }
            Some(first @ 0xf9..=0xfb) => {
                let bits = self.take(1 + (1usize << (first - 0xf8)))?[1..]
                    .iter()
                    .fold(0u64, |a, b| a << 8 | *b as u64);
                let f = match first {
                    0xf9 => from_half(bits as u16),
                    0xfa => f32::from_bits(bits as u32) as f64,
                    _ => f64::from_bits(bits),
                };
                let Some(n) = serde_json::Number::from_f64(f) else {
                    return invalid(format!("a float that is not finite at byte {at}"));
                };
                let mut shortest = Writer(Vec::with_capacity(9));
                shortest.float(f);
                if shortest.0 != self.b[at..self.at] {
                    return invalid(format!("a float not in its shortest form at byte {at}"));
                }
                return Ok(ConfigValue::Number(n));
            }
            _ => {}
        }
        let (major, n) = self.head()?;
        Ok(match major {
            UINT => ConfigValue::from(n),
            NINT if n <= i64::MAX as u64 => ConfigValue::from(-1 - n as i64),
            NINT => return invalid(format!("an integer below -2^63 at byte {at}")),
            TEXT => {
                self.at = at;
                ConfigValue::from(self.text("a string")?)
            }
            ARRAY => ConfigValue::Array(
                (0..n)
                    .map(|_| self.config(depth + 1))
                    .collect::<Result<_>>()?,
            ),
            MAP => {
                self.at = at;
                ConfigValue::Object(self.map(depth, Reader::config)?.into_iter().collect())
            }
            SIMPLE => ConfigValue::Bool(n == 1),
            _ => return invalid(format!("a byte string value at byte {at}")),
        })
    }

    /// A map with text keys in bytewise order of their encodings, none repeated.
    fn map<T>(
        &mut self,
        depth: usize,
        mut item: impl FnMut(&mut Self, usize) -> Result<T>,
    ) -> Result<BTreeMap<String, T>> {
        let n = self.expect(MAP, "a map")?;
        let mut out = BTreeMap::new();
        let mut previous: &[u8] = &[];
        for _ in 0..n {
            let start = self.at;
            let key = self.text("a map key")?;
            let encoded = &self.b[start..self.at];
            if previous >= encoded {
                return invalid(format!("map key {key:?} is out of order or repeated"));
            }
            previous = encoded;
            out.insert(key.to_owned(), item(self, depth + 1)?);
        }
        Ok(out)
    }
}

fn len(n: u64) -> Result<usize> {
    usize::try_from(n).or_else(|_| invalid("a length beyond this platform"))
}

/// Room to reserve for `n` items an untrusted count announces.
fn room(n: u64) -> usize {
    n.min(1 << 12) as usize
}

pub(crate) fn read(bytes: &[u8]) -> Result<Header> {
    parse(bytes).map_err(|e| if pre_cut(bytes) { Error::PreCut } else { e })
}

/// A header from before the cut: the JSON layout, or a CBOR layout without a pipeline, a
/// well-formed `cozytensors/1` root of exactly five elements.
fn pre_cut(bytes: &[u8]) -> bool {
    let mut r = Reader { b: bytes, at: 0 };
    let mut old = || -> Result<bool> {
        if r.record(5, "the header")? > 0 {
            return Ok(false);
        }
        check_format(r.text("format")?, HEADER_FORMAT)?;
        r.skip(4, 1)?;
        Ok(r.at == bytes.len())
    };
    bytes.first() == Some(&b'{') || old().unwrap_or(false)
}

fn parse(bytes: &[u8]) -> Result<Header> {
    let mut r = Reader { b: bytes, at: 0 };
    let fields = r.list("the header")?;
    check_format(r.text("format")?, HEADER_FORMAT)?;
    let Some(more) = fields.checked_sub(6) else {
        return invalid(format!("the header: {fields} fields, not at least 6"));
    };
    let n = r.list("chunks")?;
    let mut chunks = Vec::with_capacity(room(n));
    for _ in 0..n {
        chunks.push(r.object("a chunk")?);
    }
    let mut encodings = Vec::new();
    let mut seen = HashSet::new();
    for _ in 0..r.list("encodings")? {
        let fields = r.record(2, "an encoding")?;
        let start = r.at;
        let scheme = r.text("a scheme")?.to_owned();
        let params = r.map(1, Reader::value)?;
        let identity = &bytes[start..r.at];
        r.skip(fields, 2)?;
        if scheme.is_empty() || !seen.insert(identity) {
            return invalid(format!("encoding {scheme:?} is unnamed or repeated"));
        }
        encodings.push(Encoding { scheme, params });
    }
    let pipeline = match r.list("the pipeline")? {
        0 => None,
        1 => return invalid("the pipeline: 1 field, not 0 or at least 2"),
        n => {
            let class = r.class("the pipeline's class")?;
            let config = r.map(1, Reader::config)?.into_iter().collect();
            r.skip(n - 2, 2)?;
            Some(Pipeline { class, config })
        }
    };
    let mut cited = 0;
    let n = r.list("components")?;
    let mut components = Vec::with_capacity(room(n));
    for _ in 0..n {
        let mut fields = r.record(5, "a component")?;
        let name = r.text("a component name")?.to_owned();
        let class = r.class("a component class")?;
        let config = r.map(2, Reader::config)?.into_iter().collect();
        let n = r.list("tensors")?;
        let mut tensors = Vec::with_capacity(room(n));
        for _ in 0..n {
            tensors.push(tensor(&mut r, &encodings, &mut cited)?);
        }
        let assets = r.map(2, |r, _| r.object("an asset"))?;
        // Later fields follow config_file, which is then `config.json` for the default.
        let config_file = match fields {
            0 => None,
            _ => {
                fields -= 1;
                Some(r.text("a config file")?.to_owned())
                    .filter(|f| fields == 0 || f != CONFIG_FILE)
            }
        };
        r.skip(fields, 2)?;
        components.push(Component {
            name,
            class,
            config,
            tensors,
            assets,
            config_file,
        });
    }
    if cited != encodings.len() {
        return invalid(format!(
            "{cited} of {} encodings are cited",
            encodings.len()
        ));
    }
    let metadata: Metadata = r.map(0, Reader::value)?;
    r.skip(more, 1)?;
    if r.at != bytes.len() {
        return invalid(format!("{} bytes after the header", bytes.len() - r.at));
    }
    Header::new(chunks, pipeline, components, metadata)
}

/// One tensor. `cited` counts the encodings cited so far: each is first cited in table order.
fn tensor(r: &mut Reader, encodings: &[Encoding], cited: &mut usize) -> Result<Tensor> {
    let fields = r.record(5, "a tensor")?;
    let key = r.text("a tensor key")?.to_owned();
    let dtype = r.dtype()?;
    let shape = r.shape()?;
    let body = match r.uint("an encoding index")? {
        0 => Body::Data(r.location()?),
        i => {
            let i = usize::try_from(i - 1).unwrap_or(usize::MAX);
            if i > *cited || i >= encodings.len() {
                return invalid(format!(
                    "{key}: encoding {} is out of range or order",
                    i + 1
                ));
            }
            *cited = (*cited).max(i + 1);
            let n = r.list("parts")?;
            let mut parts = Vec::with_capacity(room(n).min(8));
            for _ in 0..n {
                let more = r.record(4, "a part")?;
                let role = r.text("a role")?.to_owned();
                let dtype = r.dtype()?;
                let shape = r.shape()?;
                let at = r.location()?;
                r.skip(more, 4)?;
                let (chunk, offset, length) = (at.chunk, at.offset, at.length);
                parts.push(Part {
                    role,
                    dtype,
                    shape,
                    chunk,
                    offset,
                    length,
                });
            }
            Body::Encoded {
                encoding: encodings[i].clone(),
                parts,
            }
        }
    };
    r.skip(fields, 3)?;
    Ok(Tensor {
        key,
        dtype,
        shape,
        body,
    })
}
