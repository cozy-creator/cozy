//! Reading a checkpoint: one cozytensors, its header and the chunks and assets it cites, from a
//! store or a plain folder. Bytes come from the chunks by offset; an encoded tensor's logical
//! bytes come through the reference decode.

use std::collections::BTreeMap;
use std::fs::{self, File};
use std::io::{self, Read};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use platform::durable;

use super::retention::{self, Hold};
use super::{fail, invalid, Error, Result, Store};
use crate::format::{
    self, decode, variant_of, variant_path, Body, Component, Header, Location, ObjectRef,
    PartBytes, Pipeline, Tensor, Tree, CHUNKS_DIR,
};
use crate::io::{Copy, Dst, ReadOpts, Src};

/// One cozytensors: its parsed header, and where its chunks and assets are read. Opening one
/// reads the header and nothing more; a chunk is found when it is read, so some may be absent
/// (a pull of some components, a thin folder). While open, it holds what it reads from a store.
#[derive(Clone, Debug)]
pub struct Checkpoint {
    objects: Objects,
    id: ObjectRef,
    header: Arc<Header>,
    header_file: Option<PathBuf>,
    _hold: Option<Arc<Hold>>,
}

#[derive(Clone, Debug)]
enum Objects {
    Store(Store),
    /// A plain folder's `chunks/<sha256>`, else the cache.
    Folder { dir: PathBuf, cache: Option<Store> },
}

impl Store {
    /// The cozytensors whose header is `header`.
    pub fn checkpoint(&self, header: &ObjectRef) -> Result<Checkpoint> {
        let fence = retention::acquire(self)?;
        let parsed = Header::from_cbor(&self.read(header)?)?;
        let ck = Checkpoint::opened(Objects::Store(self.clone()), header.clone(), parsed, None)?;
        drop(fence);
        self.used(&header.sha256, ck.closure())?;
        self.maintain(0)?;
        Ok(ck)
    }

    /// The tree document `tree`.
    pub fn tree(&self, tree: &ObjectRef) -> Result<Tree> {
        Ok(Tree::from_json(&self.read(tree)?)?)
    }

    /// `variant` of `tree` (its only one when `None`).
    pub fn variant(&self, tree: &ObjectRef, variant: Option<&str>) -> Result<Checkpoint> {
        let fence = retention::acquire(self)?;
        let parsed = self.tree(tree)?;
        let header = parsed.variant(variant)?.1;
        let ck = Checkpoint::opened(Objects::Store(self.clone()), header.clone(), Header::from_cbor(&self.read(header)?)?, None)?;
        let files = std::iter::once(tree).chain(parsed.files.values());
        if let Some(hold) = &ck._hold {
            hold.extend(files.clone().map(|o| o.sha256.clone()))?;
        }
        drop(fence);
        self.used(&tree.sha256, files.chain(ck.closure()))?;
        self.maintain(0)?;
        Ok(ck)
    }
}

impl Checkpoint {
    /// Holds the header and what it cites in the store it reads, and the snapshot folder it is in.
    fn opened(objects: Objects, id: ObjectRef, header: Header, header_file: Option<PathBuf>) -> Result<Self> {
        let hold = match &objects {
            Objects::Store(store) | Objects::Folder { cache: Some(store), .. } => {
                let mut held: Vec<String> = std::iter::once(&id).chain(header.objects()).map(|o| o.sha256.clone()).collect();
                if let Objects::Folder { dir, .. } = &objects {
                    let real = fs::canonicalize(dir).ok();
                    let snapshot = real.as_deref().and_then(|r| r.strip_prefix(store.root().join("snapshots")).ok());
                    held.extend(snapshot.and_then(|p| p.iter().next()?.to_str().map(str::to_owned)));
                }
                Some(Hold::new(store, held)?)
            }
            Objects::Folder { cache: None, .. } => None,
        };
        Ok(Self { objects, id, header: Arc::new(header), header_file, _hold: hold })
    }

    /// The header and every object it cites.
    fn closure(&self) -> impl Iterator<Item = &ObjectRef> {
        std::iter::once(&self.id).chain(self.header.objects())
    }

    /// `variant` of the plain folder `dir` (its only one when `None`): its header file, and its
    /// chunks and assets from `dir/chunks/`, else from `cache`. Lengths are checked; digests are
    /// checked where bytes enter a store ([`Store::admit_folder`]).
    pub fn folder(dir: &Path, variant: Option<&str>, cache: Option<&Store>) -> Result<Checkpoint> {
        // No collection runs between reading the header and holding what it names.
        let _fence = cache.map(retention::acquire).transpose()?;
        if dir.is_file() {
            let parent = dir.parent().unwrap_or_else(|| Path::new("."));
            let bytes = fs::read(dir).map_err(fail("read", dir))?;
            return Self::opened(Objects::Folder { dir: parent.to_path_buf(), cache: cache.cloned() }, ObjectRef::of(&bytes), Header::from_cbor(&bytes)?, Some(dir.to_path_buf()));
        }
        let name = match variant {
            Some(v) => variant_path(v),
            None => {
                let mut found = Vec::new();
                for entry in fs::read_dir(dir).map_err(fail("list", dir))? {
                    let name = entry.map_err(fail("list", dir))?.file_name();
                    let name = name.to_string_lossy().into_owned();
                    if variant_of(&name).is_some() {
                        found.push(name);
                    }
                }
                match <[String; 1]>::try_from(found) {
                    Ok([one]) => one,
                    Err(found) => {
                        return Err(invalid(dir, format!("name a variant (has {found:?})")))
                    }
                }
            }
        };
        let path = dir.join(&name);
        let bytes = fs::read(&path).map_err(|e| match e.kind() {
            io::ErrorKind::NotFound => Error::Absent(path.display().to_string()),
            _ => fail("read", &path)(e),
        })?;
        Self::opened(Objects::Folder { dir: dir.to_path_buf(), cache: cache.cloned() }, ObjectRef::of(&bytes), Header::from_cbor(&bytes)?, Some(path))
    }

    /// The header's object: the cozytensors' id.
    pub fn id(&self) -> &ObjectRef {
        &self.id
    }

    pub(crate) fn verification_file(&self, object: &ObjectRef) -> Result<File> {
        if let (Some(path), true) = (&self.header_file, object == &self.id) {
            return File::open(path).map_err(fail("open", path));
        }
        let store = match &self.objects {
            Objects::Store(store) => Some(store),
            Objects::Folder { dir, cache } => {
                if let Some(file) = verification_open(&dir.join(CHUNKS_DIR).join(&object.sha256))? {
                    return Ok(file);
                }
                cache.as_ref()
            }
        };
        if let Some(store) = store {
            let path = store.path(object)?;
            // Preserve a moving GC object's readable copy, but never mask EACCES,
            // ELOOP or another I/O failure by trying a different physical copy.
            for path in [&path, &store.trash(&object.sha256), &path] {
                if let Some(file) = verification_open(path)? {
                    file.lock_shared().map_err(fail("lock", path))?;
                    return Ok(file);
                }
            }
        }
        Err(Error::Absent(object.sha256.clone()))
    }

    /// A plain folder view of this checkpoint, sharing its admitted object files.
    pub fn path(&self) -> Result<PathBuf> {
        self.path_for("")
    }

    pub fn path_for(&self, variant: &str) -> Result<PathBuf> {
        match &self.objects {
            Objects::Folder { dir, .. } => Ok(dir.clone()),
            Objects::Store(store) => {
                let dir = store.root().join("snapshots").join(&self.id.sha256);
                store.used(&self.id.sha256, self.closure())?;
                let mut tree = Tree::default();
                tree.set_variant(variant, self.id.clone());
                store.snapshot(&store.put_tree(&tree)?, &dir, None)?;
                Ok(dir)
            }
        }
    }

    /// Export only a component's config and support files to `dir`, each replaced atomically.
    /// Missing assets refuse before any directory is created; this never downloads weights.
    /// The caller classifies `dir` for the cache.
    pub fn export_component_files(&self, name: &str, dir: &Path) -> Result<PathBuf> {
        let component = self.component(name)?;
        for object in component.assets.values() {
            if !self.holds(object)? {
                return Err(Error::Absent(object.sha256.clone()));
            }
        }
        let config = crate::export::config_json(&component.config);
        let files = std::iter::once((component.config_name(), Ok(config)))
            .chain(component.assets.keys().map(|n| (n.as_str(), self.asset(&component.name, n))));
        for (name, bytes) in files {
            let path = dir.join(name);
            let parent = path.parent().unwrap_or(dir);
            fs::create_dir_all(parent).map_err(fail("create", parent))?;
            durable::replace(&path, &bytes?, 0o644).map_err(|e| fail("write", &path)(e.into()))?;
        }
        Ok(dir.to_path_buf())
    }

    pub fn header(&self) -> &Header {
        &self.header
    }

    pub fn pipeline(&self) -> Option<&Pipeline> {
        self.header.pipeline()
    }

    /// The components, in construction order.
    pub fn components(&self) -> &[Component] {
        self.header.components()
    }

    pub fn component(&self, name: &str) -> Result<&Component> {
        self.header.component(name).ok_or_else(|| {
            let has: Vec<&str> = self.components().iter().map(|c| c.name.as_str()).collect();
            format::Error::Invalid(format!("no component {name:?} (has {has:?})")).into()
        })
    }

    /// Chunk `i`'s file (`i` indexes `Header::chunks`).
    pub fn chunk(&self, i: usize) -> Result<File> {
        match self.header.chunks.get(i) {
            Some(chunk) => self.object(chunk),
            None => Err(format::Error::Invalid(format!("no chunk {i}")).into()),
        }
    }

    /// Where chunk `i` is read: the folder's copy when it has one, else the store's.
    pub fn chunk_path(&self, i: usize) -> PathBuf {
        let sha256 = &self.header.chunks[i].sha256;
        match &self.objects {
            Objects::Store(store) => store.object_path(sha256),
            Objects::Folder { dir, cache } => {
                let own = dir.join(CHUNKS_DIR).join(sha256);
                match (own.exists(), cache) {
                    (false, Some(cache)) => cache.object_path(sha256),
                    _ => own,
                }
            }
        }
    }

    /// Any object of its closure (a chunk, an asset), opened to read.
    pub fn object(&self, object: &ObjectRef) -> Result<File> {
        match &self.objects {
            Objects::Store(store) => store.file(object),
            Objects::Folder { dir, cache } => match folder_file(dir, object)? {
                Some(file) => Ok(file),
                None => match cache {
                    Some(cache) => cache.file(object),
                    None => Err(Error::Absent(object.sha256.clone())),
                },
            },
        }
    }

    /// True when `object` can be read here; a copy of another length cannot (a fetch replaces it).
    pub fn holds(&self, object: &ObjectRef) -> Result<bool> {
        let held = match &self.objects {
            Objects::Store(store) => store.reuse(object),
            Objects::Folder { dir, cache } => folder_file(dir, object).and_then(|own| match own {
                Some(_) => Ok(true),
                None => cache.as_ref().map_or(Ok(false), |c| c.reuse(object)),
            }),
        };
        match held {
            Err(Error::Mismatch { .. }) => Ok(false),
            held => held,
        }
    }

    /// The store it reads (a folder's cache).
    pub(crate) fn store(&self) -> Option<&Store> {
        match &self.objects {
            Objects::Store(store) => Some(store),
            Objects::Folder { cache, .. } => cache.as_ref(),
        }
    }

    /// Asset `name` of `component`.
    pub fn asset(&self, component: &str, name: &str) -> Result<Vec<u8>> {
        let Some(object) = self.component(component)?.assets.get(name) else {
            return Err(format::Error::Invalid(format!("{component} has no asset {name:?}")).into());
        };
        let mut bytes = Vec::with_capacity(object.length as usize);
        (self.object(object)?.read_to_end(&mut bytes)).map_err(fail("read", name))?;
        Ok(bytes)
    }

    /// The chunks and assets of `components` (all when `None`) that cannot be read here.
    pub fn missing(&self, components: Option<&[String]>) -> Result<Vec<ObjectRef>> {
        let mut out = Vec::new();
        for object in wanted(&self.header, components)? {
            if !self.holds(&object)? {
                out.push(object);
            }
        }
        Ok(out)
    }

    /// Every tensor with its component's name, in header (construction) order.
    pub fn tensors(&self) -> impl Iterator<Item = (&str, &Tensor)> {
        self.header.tensors()
    }

    pub fn tensor(&self, component: &str, key: &str) -> Result<&Tensor> {
        self.header
            .tensor(component, key)
            .ok_or_else(|| absent(component, key))
    }

    /// The stored bytes at `at` (a plain tensor's data or one part) into `buf`.
    pub fn bytes_into(&self, at: &Location, buf: &mut [u8]) -> Result<()> {
        if buf.len() as u64 != at.length {
            return Err(format::Error::Invalid(format!(
                "a {} byte buffer for {} bytes",
                buf.len(),
                at.length
            ))
            .into());
        }
        let spans = self.header.spans(at);
        let files = spans.iter().map(|s| self.chunk(s.chunk)).collect::<Result<Vec<_>>>()?;
        let (mut copies, mut filled) = (Vec::with_capacity(spans.len()), 0);
        for (span, file) in spans.iter().zip(&files) {
            let len = span.length as usize;
            let dst = Dst::Host { ptr: buf[filled..filled + len].as_mut_ptr(), len };
            copies.push(Copy { src: Src::File(file, span.start), dst });
            filled += len;
        }
        // SAFETY: the destinations are disjoint ranges of `buf`, borrowed until the read returns.
        #[allow(unsafe_code)]
        let read = unsafe { crate::io::read(&copies, &ReadOpts::default()) };
        read.map(drop).map_err(|e| fail("read", self.chunk_path(spans[e.copy].chunk))(e.source))
    }

    /// A tensor's logical bytes, in its dtype: encoded parts go through the reference decode.
    pub fn read(&self, component: &str, key: &str) -> Result<Vec<u8>> {
        let tensor = self.tensor(component, key)?;
        let raw = |at: &Location| {
            let mut buf = vec![0; at.length as usize];
            self.bytes_into(at, &mut buf).map(|()| buf)
        };
        match &tensor.body {
            Body::Data(at) => raw(at),
            Body::Encoded { encoding, parts } => {
                let bytes = parts
                    .iter()
                    .map(|p| raw(&p.at()))
                    .collect::<Result<Vec<_>>>()?;
                let views: BTreeMap<&str, PartBytes> = parts
                    .iter()
                    .zip(&bytes)
                    .map(|(p, b)| {
                        let view = PartBytes {
                            dtype: &p.dtype,
                            shape: &p.shape,
                            bytes: b,
                        };
                        (p.role.as_str(), view)
                    })
                    .collect();
                Ok(decode(&tensor.dtype, &tensor.shape, encoding, &views)?)
            }
        }
    }
}

fn absent(component: &str, key: &str) -> super::Error {
    format::Error::Invalid(format!("no tensor {component}/{key}")).into()
}

/// A folder's copy of `object`, when it has one of its length.
fn verification_open(path: &Path) -> Result<Option<File>> {
    match File::open(path) {
        Ok(file) => Ok(Some(file)),
        Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(None),
        Err(error) => Err(fail("open", path)(error)),
    }
}

fn folder_file(dir: &Path, object: &ObjectRef) -> Result<Option<File>> {
    let path = dir.join(CHUNKS_DIR).join(&object.sha256);
    let file = match File::open(&path) {
        Ok(file) => file,
        Err(e) if e.kind() == io::ErrorKind::NotFound => return Ok(None),
        Err(e) => return Err(fail("open", &path)(e)),
    };
    let length = file.metadata().map_err(fail("stat", &path))?.len();
    if length != object.length {
        return Err(Error::Mismatch {
            expected: object.clone(),
            got: format!("{length} bytes at {}", path.display()),
        });
    }
    Ok(Some(file))
}

/// The chunks the tensors of `components` (every component when `None`) live in, then their
/// assets; each once.
pub fn wanted(header: &Header, components: Option<&[String]>) -> Result<Vec<ObjectRef>> {
    let picked: Vec<&Component> = match components {
        None => header.components().iter().collect(),
        Some(names) => {
            let mut out = Vec::new();
            for name in names {
                match header.component(name) {
                    Some(c) => out.push(c),
                    None => {
                        let has: Vec<&str> =
                            header.components().iter().map(|c| c.name.as_str()).collect();
                        return Err(Error::Absent(format!("component {name:?} (has {has:?})")));
                    }
                }
            }
            out
        }
    };
    let mut chunks = std::collections::BTreeSet::new();
    let mut assets = Vec::new();
    for c in picked {
        for t in &c.tensors {
            for (_, at) in t.locations() {
                chunks.extend(header.spans(&at).into_iter().map(|s| s.chunk));
            }
        }
        assets.extend(c.assets.values().cloned());
    }
    let mut out: Vec<ObjectRef> = chunks.into_iter().map(|i| header.chunks[i].clone()).collect();
    for a in assets {
        if !out.contains(&a) {
            out.push(a);
        }
    }
    Ok(out)
}
