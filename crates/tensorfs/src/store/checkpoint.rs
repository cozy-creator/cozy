//! Reading a checkpoint: one cozytensors, its header and the chunks and assets it cites, from a
//! store or a plain folder. Bytes come from the chunks by offset; an encoded tensor's logical
//! bytes come through the reference decode.

use std::collections::BTreeMap;
use std::fs::{self, File};
use std::io::{self, Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use super::{fail, invalid, Error, Result, Store};
use crate::format::layout::ALIGN;
use crate::format::{
    self, decode, variant_of, variant_path, Body, Component, Header, Location, ObjectRef,
    PartBytes, Pipeline, Tensor, Tree, CHUNKS_DIR,
};
use crate::sha256::{hex, Sha256};

/// One cozytensors: its parsed header, and where its chunks and assets are read. Opening one
/// reads the header and nothing more; a chunk is found when it is read, so some may be absent
/// (a pull of some components, a thin folder).
#[derive(Clone, Debug)]
pub struct Checkpoint {
    objects: Objects,
    id: ObjectRef,
    header: Arc<Header>,
    header_file: Option<PathBuf>,
}

#[derive(Clone, Debug)]
enum Objects {
    Store(Store),
    /// A plain folder's `chunks/<sha256>`, else the cache.
    Folder { dir: PathBuf, cache: Option<Store> },
}

fn write_view(path: &Path, bytes: &[u8]) -> Result<()> {
    if path.is_file() {
        return Ok(());
    }
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let parent = path.parent().expect("a view file has a parent");
    fs::create_dir_all(parent).map_err(fail("create", parent))?;
    let temp = parent.join(format!(".view-{}-{}", std::process::id(), NEXT.fetch_add(1, Ordering::Relaxed)));
    fs::write(&temp, bytes).map_err(fail("write", &temp))?;
    fs::rename(&temp, path).map_err(fail("rename", path))
}

impl Store {
    /// The cozytensors whose header is `header`.
    pub fn checkpoint(&self, header: &ObjectRef) -> Result<Checkpoint> {
        Ok(Checkpoint {
            objects: Objects::Store(self.clone()),
            header: Arc::new(Header::from_cbor(&self.read(header)?)?),
            id: header.clone(),
            header_file: None,
        })
    }

    /// The tree document `tree`.
    pub fn tree(&self, tree: &ObjectRef) -> Result<Tree> {
        Ok(Tree::from_json(&self.read(tree)?)?)
    }

    /// `variant` of `tree` (its only one when `None`).
    pub fn variant(&self, tree: &ObjectRef, variant: Option<&str>) -> Result<Checkpoint> {
        let tree = self.tree(tree)?;
        self.checkpoint(tree.variant(variant)?.1)
    }
}

impl Checkpoint {
    /// `variant` of the plain folder `dir` (its only one when `None`): its header file, and its
    /// chunks and assets from `dir/chunks/`, else from `cache`. Lengths are checked; digests are
    /// checked where bytes enter a store ([`Store::admit_folder`]).
    pub fn folder(dir: &Path, variant: Option<&str>, cache: Option<&Store>) -> Result<Checkpoint> {
        if dir.is_file() {
            let parent = dir.parent().unwrap_or_else(|| Path::new("."));
            let bytes = fs::read(dir).map_err(fail("read", dir))?;
            return Ok(Checkpoint {
                objects: Objects::Folder { dir: parent.to_path_buf(), cache: cache.cloned() },
                header: Arc::new(Header::from_cbor(&bytes)?),
                id: ObjectRef::of(&bytes),
                header_file: Some(dir.to_path_buf()),
            });
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
        Ok(Checkpoint {
            objects: Objects::Folder {
                dir: dir.to_path_buf(),
                cache: cache.cloned(),
            },
            header: Arc::new(Header::from_cbor(&bytes)?),
            id: ObjectRef::of(&bytes),
            header_file: Some(path),
        })
    }

    /// The header's object: the cozytensors' id.
    pub fn id(&self) -> &ObjectRef {
        &self.id
    }

    /// Hash the actual header file and closure without changing them. Missing or corrupt
    /// objects are returned; other filesystem errors remain errors.
    pub fn verify(&self) -> Result<Vec<ObjectRef>> {
        let mut bad = Vec::new();
        let mut seen = std::collections::HashSet::new();
        let mut buf = vec![0; 1 << 20];
        for object in std::iter::once(&self.id).chain(self.header.objects()) {
            if !seen.insert((&object.sha256, object.length)) { continue; }
            let opened = match (&self.header_file, object == &self.id) {
                (Some(path), true) => File::open(path).map_err(fail("open", path)),
                _ => self.object(object),
            };
            let mut file = match opened {
                Ok(file) => file,
                Err(Error::Absent(_) | Error::Mismatch { .. }) => { bad.push(object.clone()); continue; },
                Err(Error::Io { source, .. }) if source.kind() == io::ErrorKind::NotFound => { bad.push(object.clone()); continue; },
                Err(error) => return Err(error),
            };
            let mut hash = Sha256::new();
            let mut length = 0;
            loop {
                match file.read(&mut buf) {
                    Ok(0) => break,
                    Ok(n) => { length += n as u64; hash.update(&buf[..n]); },
                    Err(error) if error.kind() == io::ErrorKind::Interrupted => continue,
                    Err(error) => return Err(fail("verify", object.id())(error)),
                }
            }
            if length != object.length || hex(&hash.finish()) != object.sha256 { bad.push(object.clone()); }
        }
        Ok(bad)
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
                let mut tree = Tree::default();
                tree.set_variant(variant, self.id.clone());
                store.snapshot(&store.put_tree(&tree)?, &dir, None)?;
                Ok(dir)
            }
        }
    }

    /// The rendered config and assets for a path-only model loader.
    pub fn component_path(&self, name: &str) -> Result<PathBuf> {
        let c = self.component(name)?;
        let root = match &self.objects {
            Objects::Store(store) | Objects::Folder { cache: Some(store), .. } => store.root().to_path_buf(),
            Objects::Folder { dir, .. } => dir.join("chunks"),
        };
        let dir = root.join("views").join(&self.id.sha256).join(name);
        fs::create_dir_all(&dir).map_err(fail("create", &dir))?;
        let config = dir.join(c.config_file.as_deref().unwrap_or("config.json"));
        let bytes = serde_json::to_vec(&c.config).map_err(|e| invalid(&config, e.to_string()))?;
        write_view(&config, &bytes)?;
        for name in c.assets.keys() {
            write_view(&dir.join(name), &self.asset(&c.name, name)?)?;
        }
        Ok(dir)
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

    /// True when `object` can be read here.
    pub fn holds(&self, object: &ObjectRef) -> Result<bool> {
        match &self.objects {
            Objects::Store(store) => store.reuse(object),
            Objects::Folder { dir, cache } => Ok(folder_file(dir, object)?.is_some()
                || cache.as_ref().map(|c| c.reuse(object)).transpose()?.unwrap_or(false)),
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
        let mut filled = 0;
        for span in self.header.spans(at) {
            let mut file = self.chunk(span.chunk)?;
            let piece = &mut buf[filled..filled + span.length as usize];
            file.seek(SeekFrom::Start(span.start))
                .and_then(|_| file.read_exact(piece))
                .map_err(|e| fail("read", self.chunk_path(span.chunk))(e))?;
            filled += span.length as usize;
        }
        Ok(())
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

    /// Reads for `tensors` (all their parts), merged per chunk: every read starts and ends on
    /// [`ALIGN`] in its chunk and lands at an aligned offset of one buffer, so O_DIRECT can fill
    /// it. Reads go in chunk order, which is the writer's execution order.
    pub fn plan(&self, tensors: &[(&str, &str)]) -> Result<ReadPlan> {
        let mut wanted: Vec<(usize, Location, Placed)> = Vec::new();
        for (component, key) in tensors {
            let tensor = self.tensor(component, key)?;
            for (role, at) in tensor.locations() {
                let placed = Placed {
                    component: component.to_string(),
                    key: key.to_string(),
                    role: role.to_string(),
                    pieces: Vec::new(),
                };
                wanted.push((wanted.len(), at, placed));
            }
        }
        // Each part's spans, then all spans in chunk order.
        let mut spans: Vec<(usize, u64, u64, usize)> = Vec::new();
        for (i, at, _) in &wanted {
            spans.extend(
                self.header
                    .spans(at)
                    .into_iter()
                    .map(|s| (s.chunk, s.start, s.length, *i)),
            );
        }
        spans.sort();
        let mut reads: Vec<ChunkRead> = Vec::new();
        let mut bytes = 0;
        for (chunk, start, length, part) in spans {
            let (first, end) = (align_down(start), (start + length).next_multiple_of(ALIGN));
            let merge = reads
                .last()
                .is_some_and(|r| r.chunk == chunk && first <= r.offset + r.length);
            if !merge {
                reads.push(ChunkRead {
                    chunk,
                    object: self.header.chunks[chunk].clone(),
                    offset: first,
                    length: 0,
                    dest: bytes,
                });
            }
            let read = reads.last_mut().expect("a read was pushed");
            read.length = read.length.max(end - read.offset);
            bytes = read.dest + read.length;
            let dest = read.dest + start - read.offset;
            let pieces = &mut wanted[part].2.pieces;
            match pieces.last_mut() {
                Some(last) if last.dest + last.length == dest => last.length += length,
                _ => pieces.push(Piece { dest, length }),
            }
        }
        Ok(ReadPlan {
            reads,
            parts: wanted.into_iter().map(|(_, _, placed)| placed).collect(),
            bytes,
        })
    }
}

fn align_down(n: u64) -> u64 {
    n - n % ALIGN
}

fn absent(component: &str, key: &str) -> super::Error {
    format::Error::Invalid(format!("no tensor {component}/{key}")).into()
}

/// A folder's copy of `object`, when it has one of its length.
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

/// A merged read plan over chunks for a set of tensors.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ReadPlan {
    pub reads: Vec<ChunkRead>,
    /// Each part's place in the buffer, in the order asked (a plain tensor is role `""`).
    pub parts: Vec<Placed>,
    /// The buffer's size.
    pub bytes: u64,
}

/// `length` bytes of chunk `chunk` from `offset`, into the buffer at `dest`; all three are
/// multiples of [`ALIGN`]. The file may end before `offset + length`: the read is then short.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ChunkRead {
    pub chunk: usize,
    pub object: ObjectRef,
    pub offset: u64,
    pub length: u64,
    pub dest: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Placed {
    pub component: String,
    pub key: String,
    pub role: String,
    /// Where the part's bytes land, in order. One piece for a checkpoint written by the chunk
    /// rule; another writer's part spanning unaligned chunk ends takes several.
    pub pieces: Vec<Piece>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Piece {
    pub dest: u64,
    pub length: u64,
}
