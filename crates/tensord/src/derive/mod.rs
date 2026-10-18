//! The executor's weight channels, served over a connected local stream that TensorD hands it
//! (`weights_writer`'s `output` and `source` requests): a derive channel writes one new
//! checkpoint through the store's [`Writer`]; a source channel reads one granted checkpoint.
//! The executor names no store, path or pin.
//!
//! Frames are [`crate::frame`]'s. Every request is a JSON object with an `op`; every answer is
//! `{"result": …}` or `{"error": <code>, "detail": <text>}`. Checkpoints are named
//! `"sha256:<hex>"`; a role is an encoded tensor's part, `""` (or absent) a plain tensor's bytes.
//!
//! **Derive.** The output request's derivation file is `{"base": ref|null, "sources": [ref],
//! "order": [[component, key]]?, "max_new_bytes": n}` (`ref` = `{"sha256","length"}`). The new
//! checkpoint starts from `base` (every base tensor and file not written again carries over)
//! and may copy or read from `base` and `sources`. It is resumable: the writer is keyed by the
//! output's transaction, so a channel that closes before `commit` loses nothing it journaled.
//! - `has {component, key}` → bool: written already (an interrupted attempt journaled it).
//! - `put {component, key, dtype, shape, length}` (plain), or `put {component, key, dtype,
//!   shape, encoding: {scheme, params?}, parts: [{role, dtype, shape, length}]}` (encoded):
//!   the bytes follow as data frames, exactly `length` per part in order; between them the
//!   executor may send `read` requests, answered at once → null.
//! - `copy {component, key, from, from_component?, from_key?}` → null: a tensor of a granted
//!   checkpoint, renamed when asked; a large part keeps its objects.
//! - `file {path, length}` + the bytes → null: a config, a tokenizer.
//! - `read {from, component, key, role?, offset?, length?}` → `{"length": n}`, then n bytes as
//!   data frames: stored bytes of a granted checkpoint (offset 0 and the rest when absent).
//! - `commit` → `{"manifest": ref}`, and the channel ends.
//!
//! Bytes `put` and `file` carry count against `max_new_bytes`. An output already committed
//! replays: `has` is true, writes are taken and dropped, `commit` answers its manifest. A
//! refused write ends the channel (the writer keeps its journal); a refused read does not.
//!
//! **Source.** `inspect` → `{"manifest": ref, "files": {path: ref}}`; `read {component, key,
//! role?, offset?, length?}` and `file {path}` → `{"length": n}` + bytes. The header is the file
//! `model.cozytensors`, sent as its stored bytes for the executor's TensorFS to parse.
//!
//! Absent objects of a checkpoint held in part are made present through [`Hooks::ensure`]
//! before they are read.
mod adapter;
mod source;
#[cfg(test)]
pub(crate) mod tests;

pub use adapter::{compose, Composed, Selection, GRAPH_FILE};
pub use source::serve_source;

use std::{
    cell::RefCell,
    collections::BTreeSet,
    fmt::Display,
    io::{self, Read, Seek, SeekFrom},
    os::unix::net::UnixStream,
};

use serde::{de::DeserializeOwned, Deserialize};
use serde_json::{json, Value};
use tensorfs_core::format::{Body, Dtype, Encoding, Location, Manifest, ObjectRef};
use tensorfs_core::store2::{self, Checkpoint, Store};
use tensorfs_core::writer::{Data, PartData, Source, Writer};

use crate::frame::{Channel, Frame, CHUNK};
use crate::storage::{self, pin};

/// What the owner decides for a channel.
pub struct Hooks<'a> {
    /// False once the attempt that owns the channel is no longer current.
    pub current: &'a (dyn Fn() -> bool + Sync),
    /// Makes absent objects of a checkpoint held in part present (the machine's Fetcher).
    pub ensure: &'a (dyn Fn(&[ObjectRef]) -> io::Result<()> + Sync),
}

/// An output's derivation file.
#[derive(Clone, Debug, Deserialize)]
pub struct Derivation {
    #[serde(default)]
    pub base: Option<ObjectRef>,
    #[serde(default)]
    pub sources: Vec<ObjectRef>,
    #[serde(default)]
    pub order: Vec<(String, String)>,
    pub max_new_bytes: u64,
}

impl Derivation {
    pub fn parse(raw: &[u8]) -> io::Result<Derivation> {
        serde_json::from_slice(raw).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, format!("derivation: {e}")))
    }

    /// The checkpoints the output may copy and read: its base and its sources.
    pub fn granted(&self) -> impl Iterator<Item = &ObjectRef> {
        self.base.iter().chain(&self.sources)
    }
}

/// A refused request: its code and why.
#[derive(Debug)]
pub(crate) struct Refusal {
    code: String,
    detail: String,
}

pub(crate) type Answer<T> = Result<T, Refusal>;

pub(crate) fn refused(code: &str, detail: impl Display) -> Refusal {
    Refusal { code: code.into(), detail: detail.to_string() }
}

impl From<store2::Error> for Refusal {
    fn from(error: store2::Error) -> Refusal {
        refused(error.code(), error)
    }
}

impl From<io::Error> for Refusal {
    fn from(error: io::Error) -> Refusal {
        refused("io", error)
    }
}

impl From<Refusal> for io::Error {
    fn from(refusal: Refusal) -> io::Error {
        io::Error::other(format!("{}: {}", refusal.code, refusal.detail))
    }
}

pub(crate) fn answer(channel: &mut Channel<UnixStream>, result: Value) -> io::Result<()> {
    channel.send(&json!({ "result": result }))
}

pub(crate) fn refuse(channel: &mut Channel<UnixStream>, refusal: &Refusal) -> io::Result<()> {
    let detail: String = refusal.detail.chars().take(1024).collect();
    channel.send(&json!({ "error": refusal.code, "detail": detail }))
}

pub(crate) fn parse<T: DeserializeOwned>(request: &Value) -> Answer<T> {
    let op = request["op"].as_str().unwrap_or_default();
    T::deserialize(request).map_err(|e| refused("invalid_request", format!("{op}: {e}")))
}

/// The objects of `objects` this store lacks made present through `hooks`.
pub(crate) fn ensure<'a>(store: &Store, objects: impl IntoIterator<Item = &'a ObjectRef>, hooks: &Hooks) -> Answer<()> {
    let mut absent = BTreeSet::new();
    for object in objects {
        if !store.reuse(object)? {
            absent.insert(object.clone());
        }
    }
    if !absent.is_empty() {
        (hooks.ensure)(&absent.into_iter().collect::<Vec<_>>())?;
    }
    Ok(())
}

/// `manifest`'s checkpoint, its manifest and header made present first.
pub(crate) fn open(store: &Store, manifest: &ObjectRef, hooks: &Hooks) -> Answer<Checkpoint> {
    ensure(store, [manifest], hooks)?;
    let parsed = Manifest::from_json(&store.read(manifest)?).map_err(|e| refused("invalid", e))?;
    ensure(store, parsed.header(), hooks)?;
    Ok(store.checkpoint(manifest)?)
}

/// Where a tensor's bytes (`role` empty) or one part's are.
fn location(checkpoint: &Checkpoint, component: &str, key: &str, role: &str) -> Answer<Location> {
    let tensor = checkpoint.tensor(component, key)?;
    match (&tensor.body, tensor.part(role)) {
        (Body::Data(at), _) if role.is_empty() => Ok(*at),
        (_, Some(part)) => Ok(part.at()),
        _ => Err(refused("absent", format!("{component}/{key} has no part {role:?}"))),
    }
}

/// The chunks a tensor's bytes live in.
fn chunks<'c>(checkpoint: &'c Checkpoint, component: &str, key: &str) -> Answer<Vec<&'c ObjectRef>> {
    let header = checkpoint.header();
    let at = checkpoint.tensor(component, key)?.locations();
    Ok(at.iter().flat_map(|(_, at)| header.spans(at)).map(|s| &header.chunks[s.chunk]).collect())
}

#[derive(Deserialize)]
struct ReadIn {
    #[serde(default)]
    from: String,
    component: String,
    key: String,
    #[serde(default)]
    role: String,
    #[serde(default)]
    offset: u64,
    length: Option<u64>,
}

/// One `read`: refused before any byte goes (the channel goes on), else `{"length"}` and the
/// bytes. Only a failure mid-stream ends the channel.
pub(crate) fn serve_read<'c>(
    channel: &mut Channel<UnixStream>,
    store: &Store,
    checkpoint: impl FnOnce(&str) -> Answer<&'c Checkpoint>,
    request: &Value,
    hooks: &Hooks,
) -> io::Result<()> {
    let planned = (|| {
        let r: ReadIn = parse(request)?;
        let checkpoint = checkpoint(&r.from)?;
        let at = location(checkpoint, &r.component, &r.key, &r.role)?;
        let length = r.length.unwrap_or(at.length.saturating_sub(r.offset));
        if r.offset.checked_add(length).is_none_or(|end| end > at.length) {
            return Err(refused("invalid_request", format!("bytes {}+{length} of {}", r.offset, at.length)));
        }
        let header = checkpoint.header();
        let spans: Vec<(ObjectRef, u64, u64)> = header
            .spans(&Location { offset: at.offset + r.offset, length, chunk: at.chunk })
            .into_iter()
            .map(|s| (header.chunks[s.chunk].clone(), s.start, s.length))
            .collect();
        ensure(store, spans.iter().map(|s| &s.0), hooks)?;
        Ok((length, spans))
    })();
    match planned {
        Ok((length, spans)) => {
            answer(channel, json!({ "length": length }))?;
            send(channel, store, &spans)
        }
        Err(refusal) => refuse(channel, &refusal),
    }
}

/// Ranges `(object, offset, length)` as data frames.
fn send(channel: &mut Channel<UnixStream>, store: &Store, ranges: &[(ObjectRef, u64, u64)]) -> io::Result<()> {
    let mut buffer = vec![0; CHUNK];
    for (object, offset, length) in ranges {
        let mut file = store.file(object).map_err(storage::io_error)?;
        file.seek(SeekFrom::Start(*offset))?;
        let mut left = *length;
        while left > 0 {
            let n = left.min(CHUNK as u64) as usize;
            file.read_exact(&mut buffer[..n])?;
            channel.data(&buffer[..n])?;
            left -= n as u64;
        }
    }
    Ok(())
}

/// The committed output of `transaction`, while its pin and manifest are here.
pub fn committed(store: &Store, transaction: &str) -> io::Result<Option<ObjectRef>> {
    let held = store.get_pin(&pin::derived(transaction)).map_err(storage::io_error)?;
    match held.as_ref().and_then(|pin| pin.roots.first()) {
        Some(manifest) if store.reuse(manifest).map_err(storage::io_error)? => Ok(Some(manifest.clone())),
        _ => Ok(None),
    }
}

/// Keeps `transaction`'s committed output as the local model `local/<name>`, durable whatever
/// becomes of its run. Its derived pin stays as the replay cache until it expires.
pub fn adopt(store: &Store, transaction: &str, name: &str) -> io::Result<ObjectRef> {
    let manifest = committed(store, transaction)?
        .ok_or_else(|| io::Error::new(io::ErrorKind::NotFound, format!("{transaction} is not committed")))?;
    storage::set_local(store, name, &manifest)?;
    Ok(manifest)
}

/// Relinquishes an output: what its writer journaled (unless a writer still holds it) and its
/// pin. An adopted output stays in its ref.
pub fn cancel(store: &Store, transaction: &str) -> io::Result<()> {
    let key = pin::derived(transaction);
    store.abandon_writer(&key).map_err(storage::io_error)?;
    store.unpin(&key).map_err(storage::io_error)?;
    Ok(())
}

/// Serves `transaction`'s derive channel until `commit` (its manifest) or the executor closes it
/// (`None`; the writer keeps its journal).
pub fn serve_output(
    store: &Store,
    transaction: &str,
    derivation: &Derivation,
    stream: UnixStream,
    hooks: &Hooks,
) -> io::Result<Option<ObjectRef>> {
    let channel = RefCell::new(Channel(stream));
    output(store, transaction, derivation, &channel, hooks).map_err(|refusal| {
        let _ = refuse(&mut channel.borrow_mut(), &refusal);
        io::Error::from(refusal)
    })
}

/// What a derive channel reads from: the store, its granted checkpoints, the hooks.
struct Context<'a> {
    store: &'a Store,
    granted: Vec<Checkpoint>,
    hooks: &'a Hooks<'a>,
}

impl Context<'_> {
    fn granted(&self, from: &str) -> Answer<&Checkpoint> {
        let sha256 = from.trim_start_matches("sha256:");
        self.granted
            .iter()
            .find(|c| c.id().sha256 == sha256)
            .ok_or_else(|| refused("ungranted", format!("{from:?} is not a granted checkpoint")))
    }

    fn read(&self, channel: &RefCell<Channel<UnixStream>>, request: &Value) -> io::Result<()> {
        serve_read(&mut channel.borrow_mut(), self.store, |from| self.granted(from), request, self.hooks)
    }
}

#[derive(Deserialize)]
struct Name {
    component: String,
    key: String,
}

#[derive(Deserialize)]
struct PartIn {
    #[serde(default)]
    role: String,
    dtype: String,
    shape: Vec<u64>,
    length: u64,
}

#[derive(Deserialize)]
struct Put {
    component: String,
    key: String,
    dtype: String,
    shape: Vec<u64>,
    length: Option<u64>,
    encoding: Option<Encoding>,
    #[serde(default)]
    parts: Vec<PartIn>,
}

impl Put {
    /// Its parts in order: a plain tensor's one, unnamed.
    fn parts(self) -> Answer<(Put, Vec<PartIn>)> {
        let mut put = self;
        let parts = match (&put.encoding, put.length) {
            (None, Some(length)) if put.parts.is_empty() => {
                vec![PartIn { role: String::new(), dtype: put.dtype.clone(), shape: put.shape.clone(), length }]
            }
            (Some(_), None) if !put.parts.is_empty() => std::mem::take(&mut put.parts),
            _ => return Err(refused("invalid_request", "put: a plain tensor has a length, an encoded one its parts")),
        };
        for p in &parts {
            if Dtype::parse(&p.dtype).byte_len(&p.shape).is_some_and(|n| n != p.length) {
                return Err(refused("invalid_request", format!("put {}/{}: {} {:?} is not {} bytes", put.component, put.key, p.dtype, p.shape, p.length)));
            }
        }
        Ok((put, parts))
    }
}

#[derive(Deserialize)]
struct Copy {
    component: String,
    key: String,
    from: String,
    from_component: Option<String>,
    from_key: Option<String>,
}

#[derive(Deserialize)]
struct FileIn {
    path: String,
    length: u64,
}

/// A write's bytes as they come in data frames, `left` of them still to come; `read` requests
/// in between are answered.
struct Incoming<'a> {
    channel: &'a RefCell<Channel<UnixStream>>,
    context: &'a Context<'a>,
    left: u64,
    buffer: Vec<u8>,
    at: usize,
}

impl<'a> Incoming<'a> {
    fn new(channel: &'a RefCell<Channel<UnixStream>>, context: &'a Context<'a>, left: u64) -> Incoming<'a> {
        Incoming { channel, context, left, buffer: vec![], at: 0 }
    }

    fn done(&self) -> bool {
        self.left == 0 && self.at == self.buffer.len()
    }
}

impl Read for Incoming<'_> {
    fn read(&mut self, into: &mut [u8]) -> io::Result<usize> {
        loop {
            if self.at < self.buffer.len() {
                let n = into.len().min(self.buffer.len() - self.at);
                into[..n].copy_from_slice(&self.buffer[self.at..self.at + n]);
                self.at += n;
                return Ok(n);
            }
            if self.left == 0 {
                return Ok(0);
            }
            let frame = self.channel.borrow_mut().receive()?;
            match frame {
                Some(Frame::Bytes(bytes)) if bytes.len() as u64 <= self.left => {
                    self.left -= bytes.len() as u64;
                    (self.buffer, self.at) = (bytes, 0);
                }
                Some(Frame::Bytes(_)) => return Err(io::Error::new(io::ErrorKind::InvalidData, "more bytes than declared")),
                Some(Frame::Json(request)) if request["op"] == "read" => self.context.read(self.channel, &request)?,
                Some(Frame::Json(_)) => {
                    return Err(io::Error::new(io::ErrorKind::InvalidData, "a write's bytes come before the next operation"))
                }
                None => return Err(io::ErrorKind::UnexpectedEof.into()),
            }
        }
    }
}

fn output(
    store: &Store,
    transaction: &str,
    derivation: &Derivation,
    channel: &RefCell<Channel<UnixStream>>,
    hooks: &Hooks,
) -> Answer<Option<ObjectRef>> {
    let granted = derivation.granted().map(|m| open(store, m, hooks)).collect::<Answer<_>>()?;
    let context = Context { store, granted, hooks };
    let done = committed(store, transaction)?;
    let mut writer: Option<Writer> = match done {
        Some(_) => None,
        None => Some(store.writer(derivation.base.as_ref(), Some(&pin::derived(transaction)))?),
    };
    if let Some(writer) = writer.as_mut().filter(|_| !derivation.order.is_empty()) {
        let order: Vec<(&str, &str)> = derivation.order.iter().map(|(c, k)| (c.as_str(), k.as_str())).collect();
        writer.set_order(&order);
    }
    let mut written = 0u64;
    let mut bound = |bytes: u64| -> Answer<()> {
        written = written.saturating_add(bytes);
        match written <= derivation.max_new_bytes {
            true => Ok(()),
            false => Err(refused("bound_exceeded", format!("{written} new bytes exceed the output's {}", derivation.max_new_bytes))),
        }
    };
    loop {
        let frame = channel.borrow_mut().receive()?;
        let request = match frame {
            None => return Ok(None),
            Some(Frame::Json(request)) => request,
            Some(Frame::Bytes(_)) => return Err(refused("invalid_request", "expected an operation")),
        };
        if !(hooks.current)() {
            return Err(refused("writer_closed", "the attempt is no longer current"));
        }
        let result = match request["op"].as_str().unwrap_or_default() {
            "has" => {
                let r: Name = parse(&request)?;
                Value::Bool(writer.as_ref().is_none_or(|w| w.has(&r.component, &r.key)))
            }
            "put" => {
                let (put, parts) = parse::<Put>(&request)?.parts()?;
                bound(parts.iter().map(|p| p.length).sum())?;
                let mut incoming: Vec<Incoming> = parts.iter().map(|p| Incoming::new(channel, &context, p.length)).collect();
                match writer.as_mut() {
                    Some(writer) => {
                        let dtype = Dtype::parse(&put.dtype);
                        let data = match put.encoding {
                            None => Data::Plain(Source::Read(&mut incoming[0])),
                            Some(encoding) => Data::Encoded {
                                encoding,
                                parts: parts
                                    .iter()
                                    .zip(incoming.iter_mut())
                                    .map(|(p, data)| PartData { role: p.role.clone(), dtype: Dtype::parse(&p.dtype), shape: p.shape.clone(), data: Source::Read(data) })
                                    .collect(),
                            },
                        };
                        writer.put(&put.component, &put.key, &dtype, &put.shape, data)?;
                    }
                    None => {
                        for data in incoming.iter_mut() {
                            io::copy(data, &mut io::sink())?;
                        }
                    }
                }
                if !incoming.iter().all(Incoming::done) {
                    return Err(refused("invalid_request", format!("put {}/{}: fewer bytes than declared", put.component, put.key)));
                }
                Value::Null
            }
            "copy" => {
                let r: Copy = parse(&request)?;
                if let Some(writer) = writer.as_mut() {
                    let from = context.granted(&r.from)?;
                    let (component, key) = (r.from_component.as_deref().unwrap_or(&r.component), r.from_key.as_deref().unwrap_or(&r.key));
                    ensure(store, chunks(from, component, key)?, hooks)?;
                    writer.copy_as(&r.component, &r.key, from, component, key)?;
                }
                Value::Null
            }
            "file" => {
                let r: FileIn = parse(&request)?;
                bound(r.length)?;
                let mut bytes = Vec::with_capacity(r.length.min(CHUNK as u64) as usize);
                Incoming::new(channel, &context, r.length).read_to_end(&mut bytes)?;
                if let Some(writer) = writer.as_mut() {
                    writer.file(&r.path, &bytes)?;
                }
                Value::Null
            }
            "read" => {
                context.read(channel, &request)?;
                continue;
            }
            "commit" => {
                let manifest = match (writer.as_mut(), &done) {
                    (Some(writer), _) => writer.commit()?,
                    (None, Some(manifest)) => manifest.clone(),
                    (None, None) => unreachable!("a writer or a committed output"),
                };
                answer(&mut channel.borrow_mut(), json!({ "manifest": manifest }))?;
                return Ok(Some(manifest));
            }
            other => return Err(refused("invalid_request", format!("{other:?} is not a derive operation"))),
        };
        answer(&mut channel.borrow_mut(), result)?;
    }
}
