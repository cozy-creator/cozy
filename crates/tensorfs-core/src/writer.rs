//! The writer: one cozytensors out, component by component, into the store or a plain folder.
//!
//! - `component(name, class, config, config_file)` declares a component and hands back its
//!   [`ComponentWriter`]: `put` tensors, `copy_as` them from another checkpoint, add `asset`s.
//!   The header is sealed at `commit`.
//! - A part of 16 MiB or more streams straight into objects on the 64 MiB grid. Smaller parts
//!   wait in a spool; `commit` packs them by the chunk rule (`format::layout`).
//! - `copy_as` takes a tensor from any checkpoint. A large part already on the grid keeps its
//!   objects, which is what variants share; the rest is re-laid from its bytes.
//! - With a base, every base component, tensor and asset not written again carries over.
//! - Order is data: components in the base's order, then in order of declaration; tensors in
//!   the order the producer `put`s or copies them (a base's keep the base's order, ones written
//!   again keep their place), or as `set_order` says. Nothing is sorted.
//! - `Source::Reuse` and `reuse_asset` reference objects already stored elsewhere (another
//!   store, an origin): never read or copied, and not required here.
//! - A keyed writer journals finished tensors and assets and, reopened after a crash, skips them
//!   (`has`); the producer declares its components again. Its fence is `flock` on an O_CLOEXEC
//!   descriptor: a forked child holds it only until it execs, and a second opener waits.
//! - Objects the writer refers to are kept young while it runs, so GC's grace never takes them.
//! - A folder writer writes `chunks/<sha256>` first and `model.<variant>.cozytensors` last, by
//!   rename: a folder without the header is unfinished. Objects its base's store holds are
//!   referenced, not copied (a thin folder).
//! - A streaming writer sends each object to an endpoint as it seals it ([`Stream`]) and, at
//!   commit, the header and a tree of it, which becomes an unreleased checkpoint of its repo.

use std::collections::{BTreeMap, HashMap};
use std::fs::{self, File, OpenOptions};
use std::io::{self, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;

use platform::durable::{self, FileId};
use serde::{Deserialize, Serialize};

use crate::err;
use crate::fetch::Stream;
use crate::format::layout::{self, GRID, STANDALONE};
use crate::format::{
    self, variant_path, Body, Component, Config, Dtype, Encoding, Header, Metadata, ObjectRef,
    Part, Pipeline, Tensor, Value, CHUNKS_DIR, CONFIG_FILE,
};
use crate::import::{self, Imported, Sink};
use crate::sha256::{hex, Sha256};
use crate::store2::{fail, fault, is_name, sync_dir, Admission, Checkpoint, Error, Pin, Result, Store};

const SPOOL: &str = "spool";
const JOURNAL: &str = "journal";
const LOCK: &str = "lock";
/// Keyless writers' directories; no key starts with a dot.
const ANONYMOUS: &str = ".anon-";
/// A folder writer's directory, under `chunks/`.
const WRITING: &str = ".writing";
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
    /// referenced by digest, never read or copied, and not required here.
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

type Name = (String, String);

impl Entry {
    fn name(&self) -> Name {
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
    index: HashMap<Name, usize>,
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
    fn get(&self, name: &Name) -> Option<&Entry> {
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
    /// Objects on the grid that need not be here (`Source::Reuse`).
    Reused(Vec<ObjectRef>),
}

#[derive(Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum Line {
    Tensor(Entry),
    Asset {
        component: String,
        name: String,
        object: ObjectRef,
        #[serde(default, skip_serializing_if = "std::ops::Not::not")]
        reused: bool,
    },
}

/// A component's declaration: what the header says of it besides tensors and assets.
#[derive(Clone, Debug, Default)]
struct Spec {
    name: String,
    class: String,
    config: Config,
    config_file: Option<String>,
}

/// Where the objects and the header go.
#[derive(Clone, Debug)]
enum Dest {
    Store(Store),
    /// `root/chunks/<sha256>` and `root/model.<variant>.cozytensors`; `cache` holds what a thin
    /// folder references.
    Folder {
        root: PathBuf,
        variant: String,
        cache: Option<Store>,
    },
    /// Objects sent as they seal; at commit the header and tree, a checkpoint of repo `name`.
    Endpoint {
        store: Store,
        stream: Arc<Stream>,
        name: String,
        variant: String,
    },
}

pub struct Writer {
    dest: Dest,
    dir: PathBuf,
    key: Option<String>,
    /// Held until commit or abandon.
    fence: Option<File>,
    base: Option<Checkpoint>,
    pipeline: Option<Pipeline>,
    /// Components declared, in order.
    specs: Vec<Spec>,
    /// `set_order`: (component, key) → rank; and components by their first ranked tensor.
    order: HashMap<Name, usize>,
    components_ranked: bool,
    metadata: Metadata,
    tensors: Tensors,
    /// (component, asset name) → (object, given by `reuse_asset`).
    assets: BTreeMap<Name, (ObjectRef, bool)>,
    /// Repo files given to `repo_file` (README, LICENSE).
    repo: BTreeMap<String, ObjectRef>,
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

/// One component of a [`Writer`]: borrowed from it by [`Writer::component`] or [`Writer::get`].
pub struct ComponentWriter<'w> {
    w: &'w mut Writer,
    name: String,
}

impl Store {
    /// A writer of a new cozytensors in this store, starting from `base` when given. With `key`,
    /// it resumes what an interrupted writer of that key journaled, and waits while one holds it;
    /// the committed header is pinned under the key (else its digest).
    pub fn writer(&self, base: Option<&Checkpoint>, key: Option<&str>) -> Result<Writer> {
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
        Writer::open(Dest::Store(self.clone()), writers.join(name), key, base)
    }

    /// A writer of `variant` whose objects go to `stream` as they seal and whose commit adds a
    /// tree of it (with its repo files) to repo `name`'s checkpoints there. The local checkpoint
    /// commits first and remains available when the upload fails.
    pub fn streaming_writer(&self, stream: Arc<Stream>, name: &str, variant: &str, base: Option<&Checkpoint>) -> Result<Writer> {
        let writers = self.root().join("tmp").join("writers");
        sweep_anonymous(&writers)?;
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let dir = writers.join(format!("{ANONYMOUS}{}-s{}", std::process::id(), NEXT.fetch_add(1, Ordering::Relaxed)));
        format::check_file(&variant_path(variant))?;
        let dest = Dest::Endpoint { store: self.clone(), stream, name: name.to_owned(), variant: variant.to_owned() };
        Writer::open(dest, dir, None, base)
    }

    /// Removes the directories of writers no process holds and nothing touched for `older_than`
    /// (`tmp/writers/<key>`), and of imports as idle (`tmp/import/<id>`): the number removed.
    pub fn sweep_writers(&self, older_than: std::time::Duration) -> Result<usize> {
        let cutoff = std::time::SystemTime::now()
            .checked_sub(older_than)
            .unwrap_or(std::time::UNIX_EPOCH);
        let mut removed = 0;
        let tmp = self.root().join("tmp");
        for (area, lock) in [("writers", None), ("import", Some(".lock"))] {
            let dir = tmp.join(area);
            let entries = match fs::read_dir(&dir) {
                Ok(entries) => entries,
                Err(e) if e.kind() == io::ErrorKind::NotFound => continue,
                Err(e) => return Err(fail("list", &dir)(e)),
            };
            for entry in entries.flatten() {
                let path = entry.path();
                if !path.is_dir() || newest(&path) >= cutoff {
                    continue;
                }
                let fence = match lock {
                    None => path.join(LOCK),
                    Some(suffix) => dir.join(format!("{}{suffix}", entry.file_name().to_string_lossy())),
                };
                let held = match File::open(&fence) {
                    Ok(file) => match file.try_lock() {
                        Ok(()) => Some(file),
                        Err(_) => continue,
                    },
                    Err(e) if e.kind() == io::ErrorKind::NotFound => None,
                    Err(_) => continue,
                };
                if fs::remove_dir_all(&path).is_ok() {
                    removed += 1;
                }
                drop(held);
            }
        }
        Ok(removed)
    }

    /// Drops what writer `key` journaled, unless a writer holds it: false while one does.
    pub fn abandon_writer(&self, key: &str) -> Result<bool> {
        if !is_name(key, 200) {
            return Err(bad(format!("{key:?} is not a writer key")));
        }
        let dir = self.root().join("tmp").join("writers").join(key);
        let _held = match File::open(dir.join(LOCK)) {
            Ok(file) => match file.try_lock() {
                Ok(()) => file,
                Err(_) => return Ok(false),
            },
            Err(e) if e.kind() == io::ErrorKind::NotFound => return Ok(true),
            Err(e) => return Err(fail("open", dir.join(LOCK))(e)),
        };
        match fs::remove_dir_all(&dir) {
            Ok(()) => Ok(true),
            Err(e) if e.kind() == io::ErrorKind::NotFound => Ok(true),
            Err(e) => Err(fail("remove", &dir)(e)),
        }
    }

    /// Imports `req` into a committed cozytensors: (its header, the source's repo files). With
    /// `key`, an interrupted import resumes: the tensors its journal holds are not converted
    /// again. The header and repo files stay pinned under the key (else the header's digest).
    pub fn import(
        &self,
        req: &import::Request,
        key: Option<&str>,
    ) -> err::Result<(ObjectRef, BTreeMap<String, ObjectRef>)> {
        let mut writer = self.writer(None, key).map_err(Error::refusal)?;
        import::import(req, &mut writer)?;
        let header = writer.committed.clone().expect("an import commits");
        Ok((header, std::mem::take(&mut writer.repo)))
    }
}

impl Writer {
    /// A writer of `variant` into the plain folder `path` (created when absent), starting from
    /// `base` when given; objects `base`'s store holds are referenced, not copied. With
    /// `resume`, it takes back what an interrupted writer of this variant there journaled.
    pub fn folder(path: &Path, variant: &str, base: Option<&Checkpoint>, resume: bool) -> Result<Writer> {
        let chunks = path.join(CHUNKS_DIR);
        fs::create_dir_all(&chunks).map_err(fail("create", &chunks))?;
        let dir = chunks.join(WRITING).join(variant_path(variant));
        let dest = Dest::Folder {
            root: path.to_path_buf(),
            variant: variant.to_owned(),
            cache: base.and_then(Checkpoint::store).cloned(),
        };
        if !resume {
            let fence = fence(&dir)?;
            for name in [SPOOL, JOURNAL] {
                let _ = fs::remove_file(dir.join(name));
            }
            drop(fence);
        }
        let key = resume.then(|| variant_path(variant));
        Writer::open(dest, dir, key.as_deref(), base)
    }

    fn open(dest: Dest, dir: PathBuf, key: Option<&str>, base: Option<&Checkpoint>) -> Result<Writer> {
        let fence = fence(&dir)?;
        let spool = open(&dir.join(SPOOL), key.is_none(), false)?;
        let journal = key.map(|_| open(&dir.join(JOURNAL), false, true)).transpose()?;
        let mut writer = Writer {
            dest,
            dir,
            key: key.map(str::to_owned),
            fence: Some(fence),
            base: base.cloned(),
            pipeline: None,
            specs: Vec::new(),
            order: HashMap::new(),
            components_ranked: false,
            metadata: Metadata::new(),
            tensors: Tensors::default(),
            assets: BTreeMap::new(),
            repo: BTreeMap::new(),
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

    /// What `model_index.json` held: the pipeline's class (`diffusers.<Class>`) and config.
    pub fn pipeline(&mut self, class: &str, config: Config) {
        self.pipeline = Some(Pipeline {
            class: class.to_owned(),
            config,
        });
    }

    /// Declares component `name` (again: its class and config change, its tensors stay) and
    /// hands it back. `config_file` names the config's source file when it isn't `config.json`.
    pub fn component(
        &mut self,
        name: &str,
        class: &str,
        config: Config,
        config_file: Option<&str>,
    ) -> Result<ComponentWriter<'_>> {
        if !segment(name) {
            return Err(bad(format!("component name {name:?} is not one plain path segment")));
        }
        let spec = Spec {
            name: name.to_owned(),
            class: class.to_owned(),
            config,
            config_file: config_file.filter(|f| *f != CONFIG_FILE).map(str::to_owned),
        };
        match self.specs.iter_mut().find(|s| s.name == name) {
            Some(s) => *s = spec,
            None => self.specs.push(spec),
        }
        Ok(self.get(name))
    }

    /// Component `name`, declared or not: a new one has an unknown class and an empty config
    /// until [`Writer::component`] gives them (or a base has them).
    pub fn get(&mut self, name: &str) -> ComponentWriter<'_> {
        ComponentWriter {
            w: self,
            name: name.to_owned(),
        }
    }

    /// A header metadata field; it overrides the base's.
    pub fn set_metadata(&mut self, key: &str, value: Value) {
        self.metadata.insert(key.to_owned(), value);
    }

    /// The header order when the producer's calls are not it (a resumed migration): these
    /// tensors first, in this order, then the rest in recorded order; components in order of
    /// their first tensor here. Given before `commit`.
    pub fn set_order(&mut self, tensors: &[(&str, &str)]) {
        self.order = (tensors.iter().enumerate())
            .map(|(i, (c, k))| (((*c).to_owned(), (*k).to_owned()), i))
            .collect();
        self.components_ranked = true;
    }

    /// Written (or journaled by an interrupted writer of this key); base tensors don't count.
    pub fn has(&self, component: &str, key: &str) -> bool {
        self.tensors.index.contains_key(&(component.to_owned(), key.to_owned()))
    }

    /// The header, once committed.
    pub fn committed(&self) -> Option<&ObjectRef> {
        self.committed.as_ref()
    }

    /// A repo file (README, LICENSE): never inside the cozytensors. Into the store it is an
    /// object listed by [`Writer::repo_files`] and pinned with the header; into a folder, a
    /// file at its path.
    pub fn repo_file(&mut self, path: &str, bytes: &[u8]) -> Result<()> {
        self.live()?;
        format::check_file(path)?;
        if format::variant_of(path).is_some() {
            return Err(bad(format!("{path:?} names a variant, not a repo file")));
        }
        let object = match &self.dest {
            Dest::Store(store) => store.put_bytes(bytes)?,
            Dest::Folder { root, .. } => {
                let at = root.join(path);
                let parent = at.parent().expect("a file has a directory");
                fs::create_dir_all(parent).map_err(fail("create", parent))?;
                durable::replace(&at, bytes, 0o644).map_err(|e| fail("write", &at)(e.into()))?;
                ObjectRef::of(bytes)
            }
            Dest::Endpoint { store, stream, .. } => {
                let object = store.put_bytes(bytes)?;
                stream.put_stored(store, &object).map_err(sent)?;
                object
            }
        };
        self.repo.insert(path.to_owned(), object);
        Ok(())
    }

    /// The repo files given to [`Writer::repo_file`].
    pub fn repo_files(&self) -> &BTreeMap<String, ObjectRef> {
        &self.repo
    }

    /// Lays out the chunks in header order and writes the header: into the store, pinned under
    /// the key (or its own digest) with the repo files; or into the folder, last. The journal
    /// and spool go; the writer is spent.
    pub fn commit(&mut self) -> Result<ObjectRef> {
        self.live()?;
        let base = self.base.take();
        let built = self.assemble(base.as_ref());
        self.base = base;
        let (specs, grouped, assets) = built?;
        self.present()?;

        let list: Vec<(&Entry, &str, &Stored)> = (grouped.iter())
            .flatten()
            .flat_map(|e| e.parts.iter().map(move |(role, s)| (e, role.as_str(), s)))
            .collect();
        let items: Vec<layout::Item> = (list.iter())
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
                At::Spool(_) if s.length > 0 => packs.entry(plan.places[i].chunk).or_default().push(i),
                At::Spool(_) => {}
            }
        }
        for (chunk, mut members) in packs {
            members.sort_by_key(|i| plan.places[*i].offset);
            let mut admit = self.dest.admit(&self.dir)?;
            let mut end = 0;
            for i in members {
                let (place, stored) = (plan.places[i], list[i].2);
                let pad = &ZEROS[..(place.offset - end) as usize];
                admit.write_all(pad).map_err(fail("write", &self.dir))?;
                let At::Spool(offset) = stored.at else {
                    unreachable!("grid parts are standalone")
                };
                let mut spool = &self.spool;
                spool
                    .seek(SeekFrom::Start(offset))
                    .map_err(fail("seek", self.dir.join(SPOOL)))?;
                let what = format!("{}/{}", list[i].0.component, list[i].0.key);
                pour(&mut spool, &mut admit, stored.length, &mut self.buffer, &what)?;
                end = place.offset + stored.length;
            }
            let object = admit.finish()?;
            debug_assert_eq!(object.length, plan.chunks[chunk as usize]);
            chunks[chunk as usize] = Some(object);
        }

        let mut places = plan.places.iter();
        let components: Vec<Component> = (specs.into_iter().zip(grouped).zip(assets))
            .map(|((spec, entries), assets)| Component {
                tensors: (entries.iter())
                    .map(|e| tensor(e, &mut places))
                    .collect(),
                name: spec.name,
                class: spec.class,
                config: spec.config,
                assets,
                config_file: spec.config_file,
            })
            .collect();
        let chunks = (chunks.into_iter())
            .map(|c| c.expect("every chunk is laid"))
            .collect();
        let mut metadata = (self.base.as_ref()).map_or_else(Metadata::new, |b| b.header().metadata.clone());
        metadata.extend(self.metadata.clone());
        let pipeline = self.pipeline.clone().or_else(|| self.base.as_ref()?.pipeline().cloned());
        let header = Header::new(chunks, pipeline, components, metadata)?;
        if let Dest::Endpoint { store, stream, .. } = &self.dest {
            for object in header.objects() {
                if store.reuse(object)? {
                    stream.put_stored(store, object).map_err(sent)?;
                }
            }
        }
        let header = header.to_cbor()?;
        let id = match &self.dest {
            Dest::Store(store) => {
                let id = store.put_bytes(&header)?;
                let pin = self.key.clone().unwrap_or_else(|| id.sha256.clone());
                let roots = std::iter::once(id.clone()).chain(self.repo.values().cloned()).collect();
                store.pin(&pin, &Pin::new(roots))?;
                fault::point("writer.pinned");
                id
            }
            Dest::Folder { root, variant, .. } => {
                sync_dir(&root.join(CHUNKS_DIR))?;
                let at = root.join(variant_path(variant));
                durable::replace(&at, &header, 0o644).map_err(|e| fail("write", &at)(e.into()))?;
                ObjectRef::of(&header)
            }
            Dest::Endpoint { store, stream, name, variant } => {
                let id = store.put_bytes(&header)?;
                let roots = std::iter::once(id.clone()).chain(self.repo.values().cloned()).collect();
                store.pin(&id.sha256, &Pin::new(roots))?;
                self.committed = Some(id.clone());
                stream.commit(name, variant, header, &self.repo).map_err(sent)?;
                id
            }
        };
        self.close();
        self.committed = Some(id.clone());
        Ok(id)
    }

    /// Ends the writer without a header: its journal and spool go.
    pub fn abandon(mut self) {
        self.close();
    }

    /// The components in header order, each with its tensors in order and its assets.
    #[allow(clippy::type_complexity)]
    fn assemble(
        &mut self,
        base: Option<&Checkpoint>,
    ) -> Result<(Vec<Spec>, Vec<Vec<Entry>>, Vec<BTreeMap<String, ObjectRef>>)> {
        let mut specs: Vec<Spec> = Vec::new();
        let mut grouped: Vec<Vec<Entry>> = Vec::new();
        let mut assets: Vec<BTreeMap<String, ObjectRef>> = Vec::new();
        if let Some(base) = base {
            for c in base.components() {
                let mut entries = Vec::new();
                for t in &c.tensors {
                    match self.tensors.get(&(c.name.clone(), t.key.clone())) {
                        Some(e) => entries.push(e.clone()),
                        None => entries.push(self.adopt(&c.name, &t.key, base)?),
                    }
                }
                for object in c.assets.values() {
                    self.dest.copy_in(&self.dir, base, object)?;
                }
                specs.push(Spec {
                    name: c.name.clone(),
                    class: c.class.clone(),
                    config: c.config.clone(),
                    config_file: c.config_file.clone(),
                });
                grouped.push(entries);
                assets.push(c.assets.clone());
            }
        }
        let named = |specs: &[Spec], name: &str| specs.iter().position(|s| s.name == name);
        for spec in &self.specs {
            match named(&specs, &spec.name) {
                Some(i) => specs[i] = spec.clone(),
                None => {
                    specs.push(spec.clone());
                    grouped.push(Vec::new());
                    assets.push(BTreeMap::new());
                }
            }
        }
        let recorded = (self.tensors.list.iter().map(|e| &e.component))
            .chain(self.assets.keys().map(|(c, _)| c));
        for component in recorded {
            if named(&specs, component).is_none() {
                specs.push(Spec {
                    name: component.clone(),
                    ..Spec::default()
                });
                grouped.push(Vec::new());
                assets.push(BTreeMap::new());
            }
        }
        for e in &self.tensors.list {
            let i = named(&specs, &e.component).expect("every component is listed");
            if !grouped[i].iter().any(|g| g.key == e.key) {
                grouped[i].push(e.clone());
            }
        }
        for ((component, name), (object, _)) in &self.assets {
            let i = named(&specs, component).expect("every component is listed");
            assets[i].insert(name.clone(), object.clone());
        }
        if !self.order.is_empty() {
            let rank = |e: &Entry| self.order.get(&e.name()).copied().unwrap_or(usize::MAX);
            for entries in &mut grouped {
                entries.sort_by_key(rank);
            }
            if self.components_ranked {
                let first = |entries: &Vec<Entry>| entries.iter().map(rank).min().unwrap_or(usize::MAX);
                let mut all: Vec<_> = (specs.into_iter().zip(grouped).zip(assets)).collect();
                all.sort_by_key(|((_, entries), _)| first(entries));
                let ((s, g), a) = all.into_iter().map(|((s, g), a)| ((s, g), a)).unzip();
                (specs, grouped, assets) = (s, g, a);
            }
        }
        Ok((specs, grouped, assets))
    }

    fn live(&mut self) -> Result<()> {
        if self.fence.is_none() {
            return Err(bad("the writer has committed"));
        }
        self.keep().map(drop)
    }

    fn ready(&mut self, component: &str, key: &str) -> Result<()> {
        if !segment(component) || key.is_empty() {
            return Err(bad(format!("{component:?}/{key:?} cannot name a tensor")));
        }
        self.live()
    }

    /// Makes every object the writer refers to young again, a quarter of the grace after the
    /// last time, so a long write never loses its objects to GC; true when it did.
    fn keep(&mut self) -> Result<bool> {
        let Some(store) = self.dest.store() else {
            return Ok(false);
        };
        if self.touched.elapsed() < store.grace() / 4 {
            return Ok(false);
        }
        for object in self.referenced() {
            store.reuse(&object)?;
        }
        self.touched = Instant::now();
        Ok(true)
    }

    /// `length` bytes of `data` as objects, one per 64 MiB from the first byte, keeping what the
    /// writer refers to young between objects.
    fn grid(&mut self, data: &mut dyn Read, length: u64, what: &str) -> Result<Vec<ObjectRef>> {
        let mut buffer = std::mem::take(&mut self.buffer);
        let mut objects = Vec::new();
        let mut left = length;
        let laid = loop {
            if left == 0 {
                break Ok(());
            }
            let n = left.min(GRID);
            let written = (self.dest.admit(&self.dir))
                .and_then(|mut admit| pour(data, &mut admit, n, &mut buffer, what).and_then(|()| admit.finish()));
            match written {
                Ok(object) => objects.push(object),
                Err(e) => break Err(e),
            }
            left -= n;
            fault::point("writer.grid");
            match self.keep() {
                Ok(true) => {
                    let store = self.dest.store().expect("a refresh has a store");
                    if let Err(e) = objects.iter().try_for_each(|o| store.reuse(o).map(drop)) {
                        break Err(e);
                    }
                }
                Ok(false) => {}
                Err(e) => break Err(e),
            }
        };
        self.buffer = buffer;
        laid.map(|()| objects)
    }

    fn referenced(&self) -> Vec<ObjectRef> {
        let mut out: Vec<ObjectRef> = self.assets.values().map(|(o, _)| o.clone()).collect();
        out.extend(self.repo.values().cloned());
        for e in &self.tensors.list {
            out.extend(e.stored().flat_map(|p| p.at.objects()));
        }
        if let Some(base) = &self.base {
            out.push(base.id().clone());
            out.extend(base.header().objects().cloned());
        }
        out
    }

    /// Every object this writer's own tensors and assets use is here; a tensor or asset whose
    /// objects are gone is forgotten, to be put again.
    fn present(&mut self) -> Result<()> {
        let mut gone = None;
        'find: for e in &self.tensors.list {
            let here = e.stored().filter(|p| !matches!(p.at, At::Reused(_)));
            for object in here.flat_map(|p| p.at.objects()) {
                if !self.dest.has(&object)? {
                    gone = Some((e.name(), object));
                    break 'find;
                }
            }
        }
        if let Some((name, object)) = gone {
            self.tensors.retain(|e| e.name() != name);
            return Err(Error::Absent(object.sha256));
        }
        for (name, (object, reused)) in self.assets.clone() {
            if !reused && !self.dest.has(&object)? {
                self.assets.remove(&name);
                return Err(Error::Absent(object.sha256));
            }
        }
        Ok(())
    }

    /// `from`'s tensor as an entry of this writer: a large part on the grid keeps its objects
    /// (copied here when absent), one laid off the grid is split onto it, a small one goes to
    /// the spool.
    fn adopt(&mut self, component: &str, key: &str, from: &Checkpoint) -> Result<Entry> {
        let tensor = from.tensor(component, key)?;
        let (header, what) = (from.header(), format!("{component}/{key}"));
        let parts: Vec<(&str, &Dtype, &[u64], format::Location)> = match &tensor.body {
            Body::Data(at) => vec![("", &tensor.dtype, &tensor.shape, *at)],
            Body::Encoded { parts, .. } => (parts.iter())
                .map(|p| (p.role.as_str(), &p.dtype, p.shape.as_slice(), p.at()))
                .collect(),
        };
        let mut stored = Vec::new();
        for (role, dtype, shape, at) in parts {
            let ranges: Vec<(ObjectRef, u64, u64)> = (header.spans(&at).iter())
                .map(|s| (header.chunks[s.chunk].clone(), s.start, s.length))
                .collect();
            let last = ranges.len().saturating_sub(1);
            let on_grid = (ranges.iter().enumerate())
                .all(|(i, (object, start, length))| {
                    *start == 0 && *length == object.length && (i == last || *length == GRID)
                });
            let mut src = Ranges::new(from, &ranges);
            let place = match (at.length >= STANDALONE, on_grid) {
                (true, true) => {
                    for (object, ..) in &ranges {
                        self.dest.copy_in(&self.dir, from, object)?;
                    }
                    At::Grid(ranges.iter().map(|r| r.0.clone()).collect())
                }
                (true, false) => At::Grid(self.grid(&mut src, at.length, &what)?),
                (false, _) => At::Spool(self.spool_in(&mut src, at.length, &what)?),
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

    /// `length` bytes of `data` at the spool's end: their offset.
    fn spool_in(&mut self, data: &mut dyn Read, length: u64, what: &str) -> Result<u64> {
        let offset = self.spool_len;
        let path = self.dir.join(SPOOL);
        self.spool
            .seek(SeekFrom::Start(offset))
            .map_err(fail("seek", &path))?;
        pour(data, &mut self.spool, length, &mut self.buffer, what)?;
        self.spool_len += length;
        Ok(offset)
    }

    fn write_part(&mut self, what: &str, dtype: Dtype, shape: Vec<u64>, data: Source) -> Result<Stored> {
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
                data.read_to_end(&mut all).map_err(fail("read", &self.dir))?;
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
        let at = match length >= STANDALONE {
            true => At::Grid(self.grid(data, length, what)?),
            false => At::Spool(self.spool_in(data, length, what)?),
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
            Line::Asset {
                component,
                name,
                object,
                reused,
            } => drop(self.assets.insert((component, name), (object, reused))),
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
        let durable = self.dest.flushes();
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
                    At::Grid(objects) => {
                        for object in objects {
                            ok &= self.dest.has(object)?;
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
        for (name, (object, reused)) in self.assets.clone() {
            if !reused && !self.dest.has(&object)? {
                self.assets.remove(&name);
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
        if let Dest::Folder { root, .. } = &self.dest {
            let writing = root.join(CHUNKS_DIR).join(WRITING);
            let _ = fs::remove_dir(&writing);
        }
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

impl ComponentWriter<'_> {
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Writes a tensor, replacing one written before under the same key.
    pub fn put(&mut self, key: &str, dtype: &Dtype, shape: &[u64], data: Data) -> Result<()> {
        let (w, component) = (&mut *self.w, self.name.as_str());
        w.ready(component, key)?;
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
            let stored = w.write_part(&what, p.dtype, p.shape, p.data)?;
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
        w.record(Line::Tensor(entry), written)
    }

    /// Takes `from_component/from_key` from `from` as `key`.
    pub fn copy_as(&mut self, key: &str, from: &Checkpoint, from_component: &str, from_key: &str) -> Result<()> {
        self.w.ready(&self.name, key)?;
        let mut entry = self.w.adopt(from_component, from_key, from)?;
        (entry.component, entry.key) = (self.name.clone(), key.to_owned());
        let length = entry.stored().map(|p| p.length).sum();
        self.w.record(Line::Tensor(entry), length)
    }

    /// Run data in the component's folder (`tokenizer.json`, `vocab.json`), by its path there.
    pub fn asset(&mut self, name: &str, bytes: &[u8]) -> Result<()> {
        self.w.live()?;
        asset_name(&self.name, name)?;
        let object = self.w.dest.put_bytes(&self.w.dir, bytes)?;
        self.add_asset(name, object, false, bytes.len() as u64)
    }

    /// An asset already stored as `object` elsewhere: referenced, not copied or required here.
    pub fn reuse_asset(&mut self, name: &str, object: &ObjectRef) -> Result<()> {
        self.w.live()?;
        asset_name(&self.name, name)?;
        self.add_asset(name, object.clone(), true, 0)
    }

    fn add_asset(&mut self, name: &str, object: ObjectRef, reused: bool, bytes: u64) -> Result<()> {
        let line = Line::Asset {
            component: self.name.clone(),
            name: name.to_owned(),
            object,
            reused,
        };
        self.w.record(line, bytes)
    }

    /// The component's tensor order when the producer's calls are not it: these keys first, in
    /// this order, then the rest in recorded order.
    pub fn set_order(&mut self, keys: &[&str]) {
        let order = &mut self.w.order;
        order.retain(|(c, _), _| *c != self.name);
        for (i, k) in keys.iter().enumerate() {
            order.insert((self.name.clone(), (*k).to_owned()), i);
        }
    }

    /// Written (or journaled by an interrupted writer of this key); base tensors don't count.
    pub fn has(&self, key: &str) -> bool {
        self.w.has(&self.name, key)
    }
}

/// An import writes through the writer; tensors an interrupted import journaled are skipped.
/// Until convert maps files to D10 (D3, which moves this into `import`), every file is a repo
/// file.
impl Sink for Writer {
    fn put(
        &mut self,
        component: &str,
        key: &str,
        dtype: &Dtype,
        shape: &[u64],
        encoding: Option<&Encoding>,
        parts: Vec<import::Part>,
    ) -> err::Result<()> {
        if self.has(component, key) {
            return Ok(());
        }
        let mut parts = parts.into_iter().map(|p| PartData {
            role: p.role.to_owned(),
            dtype: p.dtype.clone(),
            shape: p.shape.to_vec(),
            data: Source::Read(p.data),
        });
        let data = match encoding {
            Some(encoding) => Data::Encoded {
                encoding: encoding.clone(),
                parts: parts.collect(),
            },
            None => Data::Plain(parts.next().expect("a plain tensor's part").data),
        };
        (self.get(component).put(key, dtype, shape, data)).map_err(Error::refusal)
    }

    fn file(&mut self, path: &str, bytes: &[u8]) -> err::Result<()> {
        self.repo_file(path, bytes).map_err(Error::refusal)
    }

    fn commit(&mut self, imported: &Imported) -> err::Result<()> {
        let list =
            |items: &[String]| Value::List(items.iter().map(|s| Value::Str(s.clone())).collect());
        self.set_metadata("normalized", imported.normalized.into());
        self.set_metadata("converters", list(&imported.converters));
        self.set_metadata("sources", list(&imported.sources));
        if !imported.file_metadata.is_empty() {
            let strings = |m: &BTreeMap<String, String>| {
                let pairs = m.iter().map(|(k, v)| (k.clone(), Value::Str(v.clone())));
                Value::Map(pairs.collect())
            };
            let by_component = (imported.file_metadata.iter())
                .map(|(c, m)| (c.clone(), strings(m)))
                .collect();
            self.set_metadata(import::FILE_METADATA, Value::Map(by_component));
        }
        Writer::commit(self).map(drop).map_err(Error::refusal)
    }
}

fn tensor<'a>(e: &Entry, places: &mut impl Iterator<Item = &'a format::Location>) -> Tensor {
    let mut at = (e.parts.iter()).map(|(role, s)| (role, s, *places.next().expect("a place per part")));
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
        }
    }
}

impl Dest {
    fn store(&self) -> Option<&Store> {
        match self {
            Dest::Store(store) => Some(store),
            Dest::Folder { cache, .. } => cache.as_ref(),
            Dest::Endpoint { store, .. } => Some(store),
        }
    }

    fn flushes(&self) -> bool {
        match self {
            Dest::Store(store) => store.flushes(),
            Dest::Folder { .. } => true,
            Dest::Endpoint { store, .. } => store.flushes(),
        }
    }

    /// True when `object` is here (a folder: in its chunks or its cache), then young for GC.
    fn has(&self, object: &ObjectRef) -> Result<bool> {
        match self {
            Dest::Store(store) => store.reuse(object),
            Dest::Folder { root, cache, .. } => {
                let own = root.join(CHUNKS_DIR).join(&object.sha256);
                if fs::metadata(&own).is_ok_and(|m| m.len() == object.length) {
                    return Ok(true);
                }
                cache.as_ref().map_or(Ok(false), |c| c.reuse(object))
            }
            Dest::Endpoint { store, stream, .. } => {
                let held = store.reuse(object)?;
                if held {
                    stream.put_stored(store, object).map_err(sent)?;
                }
                Ok(held)
            }
        }
    }

    fn admit(&self, work: &Path) -> Result<Admit> {
        match self {
            Dest::Store(store) => Ok(Admit::Store(store.admit(None)?)),
            Dest::Folder { root, .. } => {
                static NEXT: AtomicU64 = AtomicU64::new(0);
                let n = NEXT.fetch_add(1, Ordering::Relaxed);
                fs::create_dir_all(work).map_err(fail("create", work))?;
                let temp = work.join(format!("object-{}-{n}", std::process::id()));
                let file = (OpenOptions::new().write(true).create(true).truncate(true).open(&temp))
                    .map_err(fail("create", &temp))?;
                Ok(Admit::Folder(FolderObject {
                    chunks: root.join(CHUNKS_DIR),
                    temp,
                    file: Some(file),
                    hash: Sha256::new(),
                    length: 0,
                }))
            }
            Dest::Endpoint { store, stream, .. } => Ok(Admit::Endpoint(store.clone(), Arc::clone(stream), store.admit(None)?)),
        }
    }

    fn put_bytes(&self, work: &Path, bytes: &[u8]) -> Result<ObjectRef> {
        let mut admit = self.admit(work)?;
        admit.write_all(bytes).map_err(fail("write", work))?;
        admit.finish()
    }

    /// Copies `object` from `from` unless it is here already.
    fn copy_in(&self, work: &Path, from: &Checkpoint, object: &ObjectRef) -> Result<()> {
        if self.has(object)? {
            return Ok(());
        }
        let mut src = from.object(object)?;
        let mut admit = self.admit(work)?;
        io::copy(&mut src, &mut admit).map_err(fail("copy", work))?;
        let got = admit.finish()?;
        if got != *object {
            return Err(Error::Mismatch {
                expected: object.clone(),
                got: format!("{} ({} bytes)", got.sha256, got.length),
            });
        }
        Ok(())
    }
}

/// Bytes on their way to becoming an object of the destination.
enum Admit {
    Store(Admission),
    Folder(FolderObject),
    Endpoint(Store, Arc<Stream>, Admission),
}

/// A send that failed, as the writer's error.
fn sent(refusal: err::Refusal) -> Error {
    Error::Io { op: "send", path: PathBuf::from("the endpoint"), source: io::Error::other(refusal) }
}

/// A temp in the writer's directory, renamed into `chunks/<sha256>` when finished.
struct FolderObject {
    chunks: PathBuf,
    temp: PathBuf,
    file: Option<File>,
    hash: Sha256,
    length: u64,
}

impl Admit {
    fn finish(self) -> Result<ObjectRef> {
        match self {
            Admit::Store(admission) => admission.finish(),
            Admit::Folder(mut o) => {
                let file = o.file.take().expect("a folder object has its file until it finishes");
                file.sync_data().map_err(fail("sync", &o.temp))?;
                drop(file);
                let got = ObjectRef {
                    sha256: hex(&std::mem::take(&mut o.hash).finish()),
                    length: o.length,
                };
                let at = o.chunks.join(&got.sha256);
                fs::rename(&o.temp, &at).map_err(fail("rename into", &at))?;
                Ok(got)
            }
            Admit::Endpoint(store, stream, admission) => {
                let object = admission.finish()?;
                stream.put_stored(&store, &object).map_err(sent)?;
                Ok(object)
            }
        }
    }
}

impl Write for Admit {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        match self {
            Admit::Store(admission) => admission.write(bytes),
            Admit::Folder(o) => {
                let n = o.file.as_mut().expect("unfinished").write(bytes)?;
                o.hash.update(&bytes[..n]);
                o.length += n as u64;
                Ok(n)
            }
            Admit::Endpoint(_, _, admission) => admission.write(bytes),
        }
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

impl Drop for FolderObject {
    fn drop(&mut self) {
        if self.file.take().is_some() {
            let _ = fs::remove_file(&self.temp);
        }
    }
}

/// Reads ranges of a checkpoint's objects, in order.
struct Ranges<'a> {
    from: &'a Checkpoint,
    ranges: &'a [(ObjectRef, u64, u64)],
    next: usize,
    current: Option<io::Take<File>>,
}

impl<'a> Ranges<'a> {
    fn new(from: &'a Checkpoint, ranges: &'a [(ObjectRef, u64, u64)]) -> Ranges<'a> {
        Ranges {
            from,
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
            let mut file = self.from.object(object).map_err(io::Error::other)?;
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

/// Moves exactly `n` bytes.
fn pour(src: &mut dyn Read, dst: &mut dyn Write, mut n: u64, buf: &mut [u8], what: &str) -> Result<()> {
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

fn segment(s: &str) -> bool {
    !s.is_empty() && s != "." && s != ".." && !s.contains(['/', '\\', '\0'])
}

fn asset_name(component: &str, name: &str) -> Result<()> {
    if !segment(component) || !name.split('/').all(segment) {
        return Err(bad(format!("{component:?}/{name:?} cannot name an asset")));
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

/// The newest modification under `path`: when a writer last wrote there.
fn newest(path: &Path) -> std::time::SystemTime {
    let own = fs::metadata(path)
        .and_then(|m| m.modified())
        .unwrap_or(std::time::UNIX_EPOCH);
    match fs::read_dir(path) {
        Ok(entries) => entries
            .flatten()
            .map(|entry| newest(&entry.path()))
            .fold(own, std::cmp::max),
        Err(_) => own,
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
