//! The writer (design §6): tensors and files in, one committed and pinned cozytensors/2
//! checkpoint out.
//!
//! - A part of 16 MiB or more streams straight into objects on the 64 MiB grid. Smaller parts
//!   wait in a spool; `commit` packs them by the chunk rule (`format::layout`).
//! - `copy` takes a tensor from a checkpoint of the same store. A large part already on the grid
//!   keeps its objects, which is what lanes share; small parts are re-packed from where they are.
//! - With a base, every base tensor and file not written again carries over.
//! - Order is data: tensors land in the order the producer `put`s or `copy`s them (a base's
//!   tensors keep the base's order, ones written again keep their place), or as `set_order`
//!   says. Nothing is sorted; the chunk layout follows that order.
//! - `Source::Reuse` and `reuse_file` reference objects already stored elsewhere (another store,
//!   an origin): never read or copied, and not required here.
//! - A writer opened with a key journals finished tensors in `tmp/writers/<key>/` and, reopened
//!   after a crash, skips them (`has`). Its fence is `flock` on an O_CLOEXEC descriptor: a forked
//!   child holds it only until it execs, and a second opener of the key waits.
//! - Objects the writer refers to are kept young while it runs, so GC's grace never takes them.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::fs::{self, File, OpenOptions};
use std::io::{self, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

use cozy_platform::durable::FileId;
use serde::{Deserialize, Serialize};

use crate::err;
use crate::format::layout::{self, GRID, STANDALONE};
use crate::format::{
    self, Body, Component, Dtype, Encoding, Header, Location, Manifest, Metadata, ObjectRef, Part,
    Tensor, Value, HEADER_PATH,
};
use crate::import::{self, Imported, Sink};
use crate::store2::{fail, fault, is_name, Checkpoint, Error, Pin, Result, Store};

const SPOOL: &str = "spool";
const JOURNAL: &str = "journal";
const LOCK: &str = "lock";
/// Keyless writers' directories; no key starts with a dot.
const ANONYMOUS: &str = ".anon-";
/// Padding between packed parts, which never reaches an alignment unit.
const ZEROS: [u8; layout::ALIGN as usize] = [0; layout::ALIGN as usize];

/// What `put` writes: a plain tensor's bytes, or an encoded tensor's parts.
pub enum Data<'a> {
    Plain(Source<'a>),
    Encoded {
        encoding: Encoding,
        parts: Vec<PartData<'a>>,
    },
}

pub struct PartData<'a> {
    pub role: String,
    pub dtype: Dtype,
    pub shape: Vec<u64>,
    pub data: Source<'a>,
}

/// A part's bytes.
pub enum Source<'a> {
    Read(&'a mut dyn Read),
    /// A part of 16 MiB or more already stored as objects on the 64 MiB grid from its first byte:
    /// referenced by digest, never read or copied, and not required in this store.
    Reuse(Vec<ObjectRef>),
}

/// A finished tensor; one journal line. A plain tensor is one part with role `""`; an encoded
/// tensor's parts are in the producer's order.
#[derive(Clone, Debug, Serialize, Deserialize)]
struct Entry {
    component: String,
    key: String,
    dtype: Dtype,
    shape: Vec<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    encoding: Option<Encoding>,
    parts: Vec<(String, Stored)>,
}

impl Entry {
    fn name(&self) -> (String, String) {
        (self.component.clone(), self.key.clone())
    }
    fn stored(&self) -> impl Iterator<Item = &Stored> {
        self.parts.iter().map(|(_, s)| s)
    }
}

/// Tensors in the order they were recorded, with an index by name. Writing a tensor again
/// keeps its place.
#[derive(Default)]
struct Tensors {
    list: Vec<Entry>,
    index: HashMap<(String, String), usize>,
}

impl Tensors {
    fn insert(&mut self, e: Entry) {
        match self.index.get(&e.name()) {
            Some(&i) => self.list[i] = e,
            None => {
                self.index.insert(e.name(), self.list.len());
                self.list.push(e);
            }
        }
    }
    fn contains(&self, name: &(String, String)) -> bool {
        self.index.contains_key(name)
    }
    fn get(&self, name: &(String, String)) -> Option<&Entry> {
        self.index.get(name).map(|&i| &self.list[i])
    }
    fn retain(&mut self, keep: impl FnMut(&Entry) -> bool) {
        self.list.retain(keep);
        self.index = (self.list.iter().enumerate())
            .map(|(i, e)| (e.name(), i))
            .collect();
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct Stored {
    dtype: Dtype,
    shape: Vec<u64>,
    length: u64,
    at: At,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum At {
    /// Its own objects, split on the grid.
    Grid(Vec<ObjectRef>),
    /// At this offset of the spool.
    Spool(u64),
    /// In ranges `(object, offset, length)` of the store's objects.
    Ranges(Vec<(ObjectRef, u64, u64)>),
    /// Objects on the grid that need not be here (`Source::Reuse`).
    Reused(Vec<ObjectRef>),
}

#[derive(Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum Line {
    Tensor(Entry),
    File {
        path: String,
        object: ObjectRef,
        #[serde(default, skip_serializing_if = "std::ops::Not::not")]
        reused: bool,
    },
}

pub struct Writer {
    store: Store,
    dir: PathBuf,
    key: Option<String>,
    /// Held until commit or abandon.
    fence: Option<File>,
    base: Option<Checkpoint>,
    order: HashMap<(String, String), usize>,
    metadata: Metadata,
    tensors: Tensors,
    files: BTreeMap<String, ObjectRef>,
    /// Files given by `reuse_file`, which need not be here.
    reused: std::collections::BTreeSet<String>,
    spool: File,
    spool_len: u64,
    journal: Option<File>,
    /// Journal lines whose bytes are not yet durable.
    pending: Vec<u8>,
    unflushed: u64,
    touched: Instant,
    buffer: Vec<u8>,
    committed: Option<ObjectRef>,
}

impl Store {
    /// A writer of a new checkpoint, starting from `base` (a manifest) when given. With `key`, it
    /// resumes what an interrupted writer of that key journaled, and waits while one holds it.
    pub fn writer(&self, base: Option<&ObjectRef>, key: Option<&str>) -> Result<Writer> {
        let writers = self.root().join("tmp").join("writers");
        let name = match key {
            Some(key) if is_name(key, 200) => key.to_owned(),
            Some(key) => return Err(bad(format!("{key:?} is not a writer key"))),
            None => {
                sweep_anonymous(&writers)?;
                static NEXT: AtomicU64 = AtomicU64::new(0);
                let n = NEXT.fetch_add(1, Ordering::Relaxed);
                format!("{ANONYMOUS}{}-{n}", std::process::id())
            }
        };
        let dir = writers.join(name);
        let fence = fence(&dir)?;
        let base = base.map(|manifest| self.checkpoint(manifest)).transpose()?;
        let spool = open(&dir.join(SPOOL), key.is_none(), false)?;
        let journal = key
            .map(|_| open(&dir.join(JOURNAL), false, true))
            .transpose()?;
        let mut writer = Writer {
            store: self.clone(),
            dir,
            key: key.map(str::to_owned),
            fence: Some(fence),
            base,
            order: HashMap::new(),
            metadata: Metadata::new(),
            tensors: Tensors::default(),
            files: BTreeMap::new(),
            reused: Default::default(),
            spool,
            spool_len: 0,
            journal,
            pending: Vec::new(),
            unflushed: 0,
            touched: Instant::now(),
            buffer: vec![0; 1 << 20],
            committed: None,
        };
        if writer.journal.is_some() {
            writer.resume()?;
        }
        Ok(writer)
    }

    /// Imports `req` into a committed, pinned checkpoint. With `key`, an interrupted import
    /// resumes: the tensors its journal holds are not converted again.
    pub fn import(&self, req: &import::Request, key: Option<&str>) -> err::Result<ObjectRef> {
        let mut writer = self.writer(None, key).map_err(Error::refusal)?;
        import::import(req, &mut writer)?;
        Ok(writer.committed.clone().expect("an import commits"))
    }
}

impl Writer {
    /// Written (or journaled by an interrupted writer of this key); base tensors don't count.
    pub fn has(&self, component: &str, key: &str) -> bool {
        self.tensors
            .contains(&(component.to_owned(), key.to_owned()))
    }

    /// The manifest, once committed.
    pub fn committed(&self) -> Option<&ObjectRef> {
        self.committed.as_ref()
    }

    /// The header order when the producer's calls are not it (a resumed migration): these
    /// tensors first, in this order, then the rest in recorded order. Given before `commit`.
    pub fn set_order(&mut self, tensors: &[(&str, &str)]) {
        self.order = tensors
            .iter()
            .enumerate()
            .map(|(i, (c, k))| (((*c).to_owned(), (*k).to_owned()), i))
            .collect();
    }

    /// A header metadata field; it overrides the base's.
    pub fn set_metadata(&mut self, key: &str, value: Value) {
        self.metadata.insert(key.to_owned(), value);
    }

    /// Writes a tensor, replacing one written before under the same name.
    pub fn put(
        &mut self,
        component: &str,
        key: &str,
        dtype: &Dtype,
        shape: &[u64],
        data: Data,
    ) -> Result<()> {
        self.ready(component, key)?;
        let (encoding, inputs) = match data {
            Data::Plain(data) => {
                let part = PartData {
                    role: String::new(),
                    dtype: dtype.clone(),
                    shape: shape.to_vec(),
                    data,
                };
                (None, vec![part])
            }
            Data::Encoded { encoding, parts } => {
                let mut roles: Vec<&str> = parts.iter().map(|p| p.role.as_str()).collect();
                roles.sort_unstable();
                roles.dedup();
                if encoding.scheme.is_empty()
                    || parts.is_empty()
                    || roles.len() != parts.len()
                    || roles.contains(&"")
                {
                    return Err(bad(format!(
                        "{component}/{key}: an encoding needs a scheme and distinct, named parts"
                    )));
                }
                (Some(encoding), parts)
            }
        };
        let mut parts = Vec::new();
        let mut written = 0;
        for p in inputs {
            let what = match p.role.as_str() {
                "" => format!("{component}/{key}"),
                role => format!("{component}/{key} part {role}"),
            };
            let stored = self.write_part(&what, p.dtype, p.shape, p.data)?;
            written += stored.length;
            parts.push((p.role, stored));
        }
        let entry = Entry {
            component: component.to_owned(),
            key: key.to_owned(),
            dtype: dtype.clone(),
            shape: shape.to_vec(),
            encoding,
            parts,
        };
        self.record(Line::Tensor(entry), written)
    }

    /// Takes `component/key` from `from`, a checkpoint of this store.
    pub fn copy(&mut self, component: &str, key: &str, from: &Checkpoint) -> Result<()> {
        self.ready(component, key)?;
        let entry = self.adopt(component, key, from)?;
        let length = entry.stored().map(|p| p.length).sum();
        self.record(Line::Tensor(entry), length)
    }

    /// A file of the checkpoint's tree: a config, a tokenizer.
    pub fn file(&mut self, path: &str, bytes: &[u8]) -> Result<()> {
        self.live()?;
        file_path(path)?;
        let object = self.store.put_bytes(bytes)?;
        let line = Line::File {
            path: path.to_owned(),
            object,
            reused: false,
        };
        self.record(line, bytes.len() as u64)
    }

    /// A file already stored as `object` elsewhere: referenced, not copied or required here.
    pub fn reuse_file(&mut self, path: &str, object: &ObjectRef) -> Result<()> {
        self.live()?;
        file_path(path)?;
        let line = Line::File {
            path: path.to_owned(),
            object: object.clone(),
            reused: true,
        };
        self.record(line, 0)
    }

    /// Lays out the chunks in header order, writes the header and manifest, and pins the
    /// manifest under the key (or its own digest). The journal and spool go; the writer is spent.
    pub fn commit(&mut self) -> Result<ObjectRef> {
        self.live()?;
        let mut files = self.files.clone();
        let (mut metadata, mut tree) = (Metadata::new(), Metadata::new());
        let base = self.base.take();
        let carried = base.as_ref().map(|b| self.carry(b, &mut files));
        if let Some(b) = &base {
            (metadata, tree) = (b.header().metadata.clone(), b.manifest().metadata.clone());
        }
        self.base = base;
        let mut order = carried.transpose()?.unwrap_or_default();
        let from_base: HashSet<(String, String)> = order.iter().map(Entry::name).collect();
        order.extend(
            (self.tensors.list.iter())
                .filter(|e| !from_base.contains(&e.name()))
                .cloned(),
        );
        metadata.extend(self.metadata.clone());
        self.present()?;
        if !self.order.is_empty() {
            let rank = |e: &Entry| self.order.get(&e.name()).copied().unwrap_or(usize::MAX);
            order.sort_by_key(rank);
        }
        // Components in order of first appearance, each with its tensors in order.
        let mut grouped: Vec<(String, Vec<Entry>)> = Vec::new();
        for e in order {
            match grouped.iter_mut().find(|(c, _)| *c == e.component) {
                Some((_, list)) => list.push(e),
                None => grouped.push((e.component.clone(), vec![e])),
            }
        }

        let list: Vec<(&Entry, &str, &Stored)> = (grouped.iter())
            .flat_map(|(_, entries)| entries)
            .flat_map(|e| e.parts.iter().map(move |(role, s)| (e, role.as_str(), s)))
            .collect();
        let items: Vec<layout::Item> = list
            .iter()
            .map(|(e, role, s)| layout::Item {
                component: &e.component,
                key: &e.key,
                role,
                length: s.length,
            })
            .collect();
        let plan = layout::plan(&items);

        let mut chunks: Vec<Option<ObjectRef>> = vec![None; plan.chunks.len()];
        let mut packs: BTreeMap<u64, Vec<usize>> = BTreeMap::new();
        for (i, (_, _, s)) in list.iter().enumerate() {
            match &s.at {
                At::Grid(objects) | At::Reused(objects) => {
                    for (j, object) in objects.iter().enumerate() {
                        chunks[plan.places[i].chunk as usize + j] = Some(object.clone());
                    }
                }
                _ if s.length > 0 => packs.entry(plan.places[i].chunk).or_default().push(i),
                _ => {}
            }
        }
        for (chunk, mut members) in packs {
            members.sort_by_key(|i| plan.places[*i].offset);
            let mut admission = self.store.admit(None)?;
            let mut end = 0;
            for i in members {
                let (place, stored) = (plan.places[i], list[i].2);
                let pad = &ZEROS[..(place.offset - end) as usize];
                admission.write_all(pad).map_err(fail("write", &self.dir))?;
                let mut src: Box<dyn Read + '_> = match &stored.at {
                    At::Spool(offset) => {
                        let mut spool = &self.spool;
                        let at = self.dir.join(SPOOL);
                        spool
                            .seek(SeekFrom::Start(*offset))
                            .map_err(fail("seek", at))?;
                        Box::new(spool)
                    }
                    At::Ranges(ranges) => Box::new(Ranges::new(&self.store, ranges)),
                    At::Grid(_) | At::Reused(_) => unreachable!("grid parts are standalone"),
                };
                let what = format!("{}/{}", list[i].0.component, list[i].0.key);
                pour(
                    &mut src,
                    &mut admission,
                    stored.length,
                    &mut self.buffer,
                    &what,
                )?;
                end = place.offset + stored.length;
            }
            let object = admission.finish()?;
            debug_assert_eq!(object.length, plan.chunks[chunk as usize]);
            chunks[chunk as usize] = Some(object);
        }

        let mut places = plan.places.iter();
        let components: Vec<Component> = (grouped.iter())
            .map(|(name, entries)| {
                let tensors = (entries.iter())
                    .map(|e| {
                        let mut at = e
                            .parts
                            .iter()
                            .map(|(role, s)| (role, s, *places.next().expect("a place per part")));
                        let body = match &e.encoding {
                            None => Body::Data(at.next().expect("a plain tensor has its data").2),
                            Some(encoding) => Body::Encoded {
                                encoding: encoding.clone(),
                                parts: at
                                    .map(|(role, s, at)| Part {
                                        role: role.clone(),
                                        dtype: s.dtype.clone(),
                                        shape: s.shape.clone(),
                                        chunk: at.chunk,
                                        offset: at.offset,
                                        length: at.length,
                                    })
                                    .collect(),
                            },
                        };
                        Tensor {
                            key: e.key.clone(),
                            dtype: e.dtype.clone(),
                            shape: e.shape.clone(),
                            body,
                        }
                    })
                    .collect();
                Component {
                    name: name.clone(),
                    tensors,
                }
            })
            .collect();
        if !components.is_empty() {
            let chunks = (chunks.into_iter())
                .map(|c| c.expect("every chunk is laid"))
                .collect();
            let header = Header::new(chunks, components, metadata)?;
            let header = self.store.put_bytes(&header.to_cbor()?)?;
            files.insert(HEADER_PATH.to_owned(), header);
        }
        let manifest = Manifest {
            files,
            metadata: tree,
        };
        let id = self.store.put_bytes(&manifest.to_json()?)?;
        let pin = self.key.clone().unwrap_or_else(|| id.sha256.clone());
        self.store.pin(&pin, &Pin::new(vec![id.clone()]))?;
        fault::point("writer.pinned");
        self.close();
        self.committed = Some(id.clone());
        Ok(id)
    }

    /// Ends the writer without a checkpoint: its journal and spool go.
    pub fn abandon(mut self) {
        self.close();
    }

    fn live(&mut self) -> Result<()> {
        if self.fence.is_none() {
            return Err(bad("the writer has committed"));
        }
        self.keep()
    }

    fn ready(&mut self, component: &str, key: &str) -> Result<()> {
        let segment = !component.is_empty()
            && component != "."
            && component != ".."
            && !component.contains(['/', '\\', '\0']);
        if !segment || key.is_empty() {
            return Err(bad(format!("{component:?}/{key:?} cannot name a tensor")));
        }
        self.live()
    }

    /// Makes every object the writer refers to young again, a quarter of the grace after the
    /// last time, so a long write never loses its objects to GC.
    fn keep(&mut self) -> Result<()> {
        if self.touched.elapsed() < self.store.grace() / 4 {
            return Ok(());
        }
        for object in self.referenced() {
            self.store.reuse(&object)?;
        }
        self.touched = Instant::now();
        Ok(())
    }

    fn referenced(&self) -> Vec<ObjectRef> {
        let mut out: Vec<ObjectRef> = self.files.values().cloned().collect();
        for e in &self.tensors.list {
            out.extend(e.stored().flat_map(|p| p.at.objects()));
        }
        if let Some(base) = &self.base {
            out.push(base.id().clone());
            out.extend(base.manifest().files.values().cloned());
            out.extend(base.header().chunks.iter().cloned());
        }
        out
    }

    /// Every object this writer's own tensors and files use is here; a tensor whose objects are
    /// gone is forgotten, to be put again.
    fn present(&mut self) -> Result<()> {
        let mut gone = None;
        'find: for e in &self.tensors.list {
            let here = e.stored().filter(|p| !matches!(p.at, At::Reused(_)));
            for object in here.flat_map(|p| p.at.objects()) {
                if !self.store.reuse(&object)? {
                    gone = Some((e.name(), object));
                    break 'find;
                }
            }
        }
        if let Some((name, object)) = gone {
            self.tensors.retain(|e| e.name() != name);
            return Err(Error::Absent(object.sha256));
        }
        for (path, object) in self.files.clone() {
            if !self.reused.contains(&path) && !self.store.reuse(&object)? {
                self.files.remove(&path);
                return Err(Error::Absent(object.sha256));
            }
        }
        Ok(())
    }

    /// The base's tensors in its order, each one written again in its place, and the base's
    /// files that were not written again.
    fn carry(
        &mut self,
        base: &Checkpoint,
        files: &mut BTreeMap<String, ObjectRef>,
    ) -> Result<Vec<Entry>> {
        let mut out = Vec::new();
        for (component, t) in base.tensors() {
            let name = (component.to_owned(), t.key.clone());
            match self.tensors.get(&name) {
                Some(e) => out.push(e.clone()),
                None => out.push(self.adopt(component, &t.key, base)?),
            }
        }
        for (path, object) in &base.manifest().files {
            if path != HEADER_PATH && !files.contains_key(path) {
                if !self.store.reuse(object)? {
                    return Err(Error::Absent(object.sha256.clone()));
                }
                files.insert(path.clone(), object.clone());
            }
        }
        Ok(out)
    }

    /// `from`'s tensor as an entry of this writer: a large part on the grid keeps its objects,
    /// one another writer laid off the grid is split onto it, a small one is read in place.
    fn adopt(&mut self, component: &str, key: &str, from: &Checkpoint) -> Result<Entry> {
        if from.store().root() != self.store.root() {
            return Err(bad(
                "copy takes tensors from a checkpoint of the same store",
            ));
        }
        let tensor = from.tensor(component, key)?;
        let (header, what) = (from.header(), format!("{component}/{key}"));
        let parts: Vec<(&str, &Dtype, &[u64], Location)> = match &tensor.body {
            Body::Data(at) => vec![("", &tensor.dtype, &tensor.shape, *at)],
            Body::Encoded { parts, .. } => (parts.iter())
                .map(|p| (p.role.as_str(), &p.dtype, p.shape.as_slice(), p.at()))
                .collect(),
        };
        let mut stored = Vec::new();
        for (role, dtype, shape, at) in parts {
            let spans = header.spans(&at);
            let ranges: Vec<(ObjectRef, u64, u64)> = (spans.iter())
                .map(|s| (header.chunks[s.chunk].clone(), s.start, s.length))
                .collect();
            for (object, ..) in &ranges {
                if !self.store.reuse(object)? {
                    return Err(Error::Absent(object.sha256.clone()));
                }
            }
            let last = ranges.len().saturating_sub(1);
            let on_grid = ranges
                .iter()
                .enumerate()
                .all(|(i, (object, start, length))| {
                    *start == 0 && *length == object.length && (i == last || *length == GRID)
                });
            let place = match (at.length >= STANDALONE, on_grid) {
                (true, true) => At::Grid(ranges.into_iter().map(|r| r.0).collect()),
                (true, false) => {
                    let mut src = Ranges::new(&self.store, &ranges);
                    At::Grid(grid(
                        &self.store,
                        &mut self.buffer,
                        &mut src,
                        at.length,
                        &what,
                    )?)
                }
                (false, _) => At::Ranges(ranges),
            };
            let part = Stored {
                dtype: dtype.clone(),
                shape: shape.to_vec(),
                length: at.length,
                at: place,
            };
            stored.push((role.to_owned(), part));
        }
        Ok(Entry {
            component: component.to_owned(),
            key: key.to_owned(),
            dtype: tensor.dtype.clone(),
            shape: tensor.shape.clone(),
            encoding: match &tensor.body {
                Body::Encoded { encoding, .. } => Some(encoding.clone()),
                Body::Data(_) => None,
            },
            parts: stored,
        })
    }

    fn write_part(
        &mut self,
        what: &str,
        dtype: Dtype,
        shape: Vec<u64>,
        data: Source,
    ) -> Result<Stored> {
        let data = match data {
            Source::Read(data) => data,
            Source::Reuse(objects) => return reused(what, dtype, shape, objects),
        };
        // A dtype this build does not know is sized by its bytes.
        let owned: Vec<u8>;
        let mut slice: &[u8];
        let (length, data): (u64, &mut dyn Read) = match dtype.byte_len(&shape) {
            Some(length) => (length, data),
            None if dtype.bits().is_none() => {
                let mut all = Vec::new();
                data.read_to_end(&mut all)
                    .map_err(fail("read", &self.dir))?;
                owned = all;
                slice = &owned;
                (owned.len() as u64, &mut slice)
            }
            None => {
                return Err(bad(format!(
                    "{what}: {dtype} {shape:?} overflows or is not whole bytes"
                )))
            }
        };
        let at = if length >= STANDALONE {
            At::Grid(grid(&self.store, &mut self.buffer, data, length, what)?)
        } else {
            let offset = self.spool_len;
            let path = self.dir.join(SPOOL);
            self.spool
                .seek(SeekFrom::Start(offset))
                .map_err(fail("seek", &path))?;
            pour(data, &mut self.spool, length, &mut self.buffer, what)?;
            self.spool_len += length;
            At::Spool(offset)
        };
        let more = loop {
            match data.read(&mut [0]) {
                Err(e) if e.kind() == io::ErrorKind::Interrupted => continue,
                read => break read.map_err(fail("read", &self.dir))?,
            }
        };
        if more > 0 {
            return Err(bad(format!("{what}: more than its {length} bytes")));
        }
        Ok(Stored {
            dtype,
            shape,
            length,
            at,
        })
    }

    fn record(&mut self, line: Line, bytes: u64) -> Result<()> {
        if self.journal.is_some() {
            serde_json::to_writer(&mut self.pending, &line).expect("a journal line serializes");
            self.pending.push(b'\n');
            self.unflushed += bytes;
        }
        self.apply(line);
        if self.unflushed >= STANDALONE {
            self.flush()?;
        }
        Ok(())
    }

    fn apply(&mut self, line: Line) {
        match line {
            Line::Tensor(e) => self.tensors.insert(e),
            Line::File {
                path,
                object,
                reused,
            } => {
                match reused {
                    true => self.reused.insert(path.clone()),
                    false => self.reused.remove(&path),
                };
                self.files.insert(path, object);
            }
        }
    }

    /// Appends the pending journal lines once the bytes they name are durable: the journal
    /// never names a part a crash could lose.
    fn flush(&mut self) -> Result<()> {
        let Some(journal) = self.journal.as_mut() else {
            return Ok(());
        };
        if self.pending.is_empty() {
            return Ok(());
        }
        let durable = self.store.flushes();
        if durable {
            self.spool
                .sync_data()
                .map_err(fail("sync", self.dir.join(SPOOL)))?;
        }
        let path = self.dir.join(JOURNAL);
        journal
            .write_all(&self.pending)
            .and_then(|()| if durable { journal.sync_data() } else { Ok(()) })
            .map_err(fail("write", path))?;
        self.pending.clear();
        self.unflushed = 0;
        fault::point("writer.journaled");
        Ok(())
    }

    /// Takes back what the journal holds: complete lines whose bytes are still here.
    fn resume(&mut self) -> Result<()> {
        let path = self.dir.join(JOURNAL);
        let journal = self.journal.as_mut().expect("a keyed writer journals");
        let mut bytes = Vec::new();
        journal
            .seek(SeekFrom::Start(0))
            .and_then(|_| journal.read_to_end(&mut bytes))
            .map_err(fail("read", &path))?;
        let mut good = 0;
        let mut lines = Vec::new();
        for line in bytes.split_inclusive(|b| *b == b'\n') {
            match serde_json::from_slice::<Line>(line) {
                Ok(parsed) if line.ends_with(b"\n") => lines.push(parsed),
                _ => break,
            }
            good += line.len();
        }
        if good < bytes.len() {
            journal
                .set_len(good as u64)
                .map_err(fail("truncate", &path))?;
        }
        for line in lines {
            self.apply(line);
        }
        let spooled = (self.spool.metadata())
            .map_err(fail("stat", self.dir.join(SPOOL)))?
            .len();
        let mut end = 0;
        let mut lost = Vec::new();
        for e in &self.tensors.list {
            let mut ok = true;
            for p in e.stored() {
                match &p.at {
                    At::Spool(offset) => ok &= offset + p.length <= spooled,
                    At::Reused(_) => {}
                    other => {
                        for object in other.objects() {
                            ok &= self.store.reuse(&object)?;
                        }
                    }
                }
            }
            match ok {
                true => end = e.stored().fold(end, |end, p| end.max(p.spool_end())),
                false => lost.push(e.name()),
            }
        }
        self.tensors.retain(|e| !lost.contains(&e.name()));
        let paths: Vec<String> = self.files.keys().cloned().collect();
        for path in paths {
            if !self.reused.contains(&path) && !self.store.reuse(&self.files[&path])? {
                self.files.remove(&path);
            }
        }
        self.spool
            .set_len(end)
            .map_err(fail("truncate", self.dir.join(SPOOL)))?;
        self.spool_len = end;
        Ok(())
    }

    /// Removes the journal and spool and lets the fence go.
    fn close(&mut self) {
        for name in [SPOOL, JOURNAL, LOCK] {
            let _ = fs::remove_file(self.dir.join(name));
        }
        let _ = fs::remove_dir(&self.dir);
        self.fence = None;
    }
}

impl Drop for Writer {
    /// An unfinished keyed writer keeps what it journaled; a keyless one leaves nothing.
    fn drop(&mut self) {
        if self.fence.is_none() {
            return;
        }
        match self.key {
            Some(_) => drop(self.flush()),
            None => self.close(),
        }
    }
}

impl Stored {
    fn spool_end(&self) -> u64 {
        match self.at {
            At::Spool(offset) => offset + self.length,
            _ => 0,
        }
    }
}

impl At {
    fn objects(&self) -> Vec<ObjectRef> {
        match self {
            At::Grid(objects) | At::Reused(objects) => objects.clone(),
            At::Spool(_) => vec![],
            At::Ranges(ranges) => ranges.iter().map(|r| r.0.clone()).collect(),
        }
    }
}

/// An import writes through the writer; tensors an interrupted import journaled are skipped.
impl Sink for Writer {
    fn put(
        &mut self,
        component: &str,
        key: &str,
        dtype: &Dtype,
        shape: &[u64],
        data: &mut dyn Read,
    ) -> err::Result<()> {
        if self.has(component, key) {
            return Ok(());
        }
        let data = Data::Plain(Source::Read(data));
        Writer::put(self, component, key, dtype, shape, data).map_err(Error::refusal)
    }

    fn file(&mut self, path: &str, bytes: &[u8]) -> err::Result<()> {
        Writer::file(self, path, bytes).map_err(Error::refusal)
    }

    fn commit(&mut self, imported: &Imported) -> err::Result<()> {
        let list =
            |items: &[String]| Value::List(items.iter().map(|s| Value::Str(s.clone())).collect());
        self.set_metadata("normalized", imported.normalized.into());
        self.set_metadata("converters", list(&imported.converters));
        self.set_metadata("sources", list(&imported.sources));
        Writer::commit(self).map(drop).map_err(Error::refusal)
    }
}

/// Reads ranges of the store's objects, in order.
struct Ranges<'a> {
    store: &'a Store,
    ranges: &'a [(ObjectRef, u64, u64)],
    next: usize,
    current: Option<io::Take<File>>,
}

impl<'a> Ranges<'a> {
    fn new(store: &'a Store, ranges: &'a [(ObjectRef, u64, u64)]) -> Ranges<'a> {
        Ranges {
            store,
            ranges,
            next: 0,
            current: None,
        }
    }
}

impl Read for Ranges<'_> {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        loop {
            if let Some(current) = &mut self.current {
                match current.read(buf)? {
                    0 => self.current = None,
                    n => return Ok(n),
                }
            }
            let Some((object, offset, length)) = self.ranges.get(self.next) else {
                return Ok(0);
            };
            self.next += 1;
            let mut file = self.store.file(object).map_err(io::Error::other)?;
            file.seek(SeekFrom::Start(*offset))?;
            self.current = Some(file.take(*length));
        }
    }
}

/// A [`Source::Reuse`] part: whole objects on the grid that add up to its bytes.
fn reused(what: &str, dtype: Dtype, shape: Vec<u64>, objects: Vec<ObjectRef>) -> Result<Stored> {
    let length: u64 = objects.iter().map(|o| o.length).sum();
    let last = objects.len().saturating_sub(1);
    let on_grid = (objects.iter().enumerate())
        .all(|(i, o)| o.length > 0 && o.length <= GRID && (i == last || o.length == GRID));
    let sized = match dtype.byte_len(&shape) {
        Some(n) => n == length,
        None => dtype.bits().is_none(),
    };
    if length < STANDALONE || !on_grid || !sized {
        return Err(bad(format!(
            "{what}: {} objects of {length} bytes are not a {dtype} {shape:?} part on the grid",
            objects.len()
        )));
    }
    Ok(Stored {
        dtype,
        shape,
        length,
        at: At::Reused(objects),
    })
}

/// `length` bytes of `data` as objects, one per 64 MiB from the first byte.
fn grid(
    store: &Store,
    buffer: &mut [u8],
    data: &mut dyn Read,
    length: u64,
    what: &str,
) -> Result<Vec<ObjectRef>> {
    let mut objects = Vec::new();
    let mut left = length;
    while left > 0 {
        let n = left.min(GRID);
        let mut admission = store.admit(None)?;
        pour(data, &mut admission, n, buffer, what)?;
        objects.push(admission.finish()?);
        left -= n;
        fault::point("writer.grid");
    }
    Ok(objects)
}

/// Moves exactly `n` bytes.
fn pour(
    src: &mut dyn Read,
    dst: &mut dyn Write,
    mut n: u64,
    buf: &mut [u8],
    what: &str,
) -> Result<()> {
    let total = n;
    while n > 0 {
        let want = n.min(buf.len() as u64) as usize;
        let got = match src.read(&mut buf[..want]) {
            Ok(0) => {
                let got = total - n;
                return Err(bad(format!("{what}: {got} of its {total} bytes")));
            }
            Ok(got) => got,
            Err(e) if e.kind() == io::ErrorKind::Interrupted => continue,
            Err(e) => return Err(bad(format!("{what}: reading: {e}"))),
        };
        dst.write_all(&buf[..got]).map_err(fail("write", what))?;
        n -= got as u64;
    }
    Ok(())
}

fn file_path(path: &str) -> Result<()> {
    let plain = |s: &str| !s.is_empty() && s != "." && s != ".." && !s.contains(['\\', '\0']);
    if path == HEADER_PATH || !path.split('/').all(plain) {
        return Err(bad(format!("{path:?} cannot be a file of the checkpoint")));
    }
    Ok(())
}

fn bad(detail: impl Into<String>) -> Error {
    format::Error::Invalid(detail.into()).into()
}

fn open(path: &Path, truncate: bool, append: bool) -> Result<File> {
    OpenOptions::new()
        .read(true)
        .write(!append)
        .append(append)
        .create(true)
        .truncate(truncate)
        .open(path)
        .map_err(fail("open", path))
}

/// The writer's fence: `flock` on `<dir>/lock`, waiting while another holds it. A finished
/// writer removes its directory, so the fence counts only while its path still names the file.
fn fence(dir: &Path) -> Result<File> {
    let path = dir.join(LOCK);
    loop {
        fs::create_dir_all(dir).map_err(fail("create", dir))?;
        let file = match OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(&path)
        {
            Ok(file) => file,
            Err(e) if e.kind() == io::ErrorKind::NotFound => continue,
            Err(e) => return Err(fail("open", &path)(e)),
        };
        file.lock().map_err(fail("lock", &path))?;
        let held = FileId::of(&file).map_err(|e| fail("stat", &path)(e.into()))?;
        match FileId::at(&path).map_err(io::Error::from) {
            Ok(named) if named == held => return Ok(file),
            Ok(_) => {}
            Err(e) if e.kind() == io::ErrorKind::NotFound => {}
            Err(e) => return Err(fail("stat", &path)(e)),
        }
    }
}

/// Removes keyless writers' directories whose writer is gone (its fence is free).
fn sweep_anonymous(writers: &Path) -> Result<()> {
    let entries = match fs::read_dir(writers) {
        Ok(entries) => entries,
        Err(e) if e.kind() == io::ErrorKind::NotFound => return Ok(()),
        Err(e) => return Err(fail("list", writers)(e)),
    };
    for entry in entries.flatten() {
        if !entry.file_name().to_string_lossy().starts_with(ANONYMOUS) {
            continue;
        }
        let dir = entry.path();
        let gone = match File::open(dir.join(LOCK)) {
            Ok(lock) => lock.try_lock().is_ok(),
            Err(e) => e.kind() == io::ErrorKind::NotFound,
        };
        if gone {
            let _ = fs::remove_dir_all(&dir);
        }
    }
    Ok(())
}
