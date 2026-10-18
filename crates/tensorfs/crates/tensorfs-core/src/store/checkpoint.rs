//! Reading a checkpoint: its manifest, its header and the chunk files it lives in. Bytes come
//! from the chunks by offset; an encoded tensor's logical bytes come through the reference decode.

use std::collections::BTreeMap;
use std::io::{Read, Seek, SeekFrom};
use std::path::PathBuf;

use super::{fail, Result, Store};
use crate::format::layout::ALIGN;
use crate::format::{self, decode, Body, Header, Location, Manifest, ObjectRef, PartBytes, Tensor};

/// A checkpoint in a store: the parsed manifest and header, and where each chunk lives.
#[derive(Clone, Debug)]
pub struct Checkpoint {
    store: Store,
    id: ObjectRef,
    manifest: Manifest,
    header: Header,
    paths: Vec<PathBuf>,
}

impl Store {
    /// The checkpoint `manifest` names. Its chunks need not all be here (a pull may take some
    /// components only); reading an absent one fails.
    pub fn checkpoint(&self, manifest: &ObjectRef) -> Result<Checkpoint> {
        let parsed = Manifest::from_json(&self.read(manifest)?)?;
        let header = match parsed.header() {
            Some(header) => Header::from_cbor(&self.read(header)?)?,
            None => Header::default(),
        };
        let paths = header
            .chunks
            .iter()
            .map(|chunk| self.path(chunk))
            .collect::<Result<_>>()?;
        Ok(Checkpoint {
            store: self.clone(),
            id: manifest.clone(),
            manifest: parsed,
            header,
            paths,
        })
    }
}

impl Checkpoint {
    /// The manifest's object.
    pub fn id(&self) -> &ObjectRef {
        &self.id
    }

    pub fn manifest(&self) -> &Manifest {
        &self.manifest
    }

    pub fn header(&self) -> &Header {
        &self.header
    }

    pub(crate) fn store(&self) -> &Store {
        &self.store
    }

    /// Chunk `i`'s file, in `Header::chunks` order.
    pub fn chunk_paths(&self) -> &[PathBuf] {
        &self.paths
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

    /// A file of the manifest: a config, a tokenizer.
    pub fn file(&self, path: &str) -> Result<Vec<u8>> {
        match self.manifest.files.get(path) {
            Some(object) => self.store.read(object),
            None => Err(format::Error::Invalid(format!("no file {path:?}")).into()),
        }
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
            let mut file = self.store.file(&self.header.chunks[span.chunk])?;
            let path = &self.paths[span.chunk];
            let piece = &mut buf[filled..filled + span.length as usize];
            file.seek(SeekFrom::Start(span.start))
                .and_then(|_| file.read_exact(piece))
                .map_err(fail("read", path))?;
            filled += span.length as usize;
        }
        Ok(())
    }

    /// A tensor's logical bytes into `buf`: a plain tensor's data, or its parts decoded.
    pub fn read_into(&self, component: &str, key: &str, buf: &mut [u8]) -> Result<()> {
        match &self.tensor(component, key)?.body {
            Body::Data(at) => self.bytes_into(at, buf),
            Body::Encoded { .. } => {
                let bytes = self.read(component, key)?;
                if bytes.len() != buf.len() {
                    return Err(format::Error::Invalid(format!(
                        "a {} byte buffer for {component}/{key}'s {} bytes",
                        buf.len(),
                        bytes.len()
                    ))
                    .into());
                }
                buf.copy_from_slice(&bytes);
                Ok(())
            }
        }
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
