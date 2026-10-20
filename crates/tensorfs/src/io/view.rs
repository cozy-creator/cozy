//! Stored byte descriptions for placement. Header details and chunk opens stay here; callers
//! see them through `Checkpoint::{spans, stored_part, source}`.

#[cfg(unix)]
use std::collections::BTreeMap;
#[cfg(unix)]
use std::fs::File;
use std::sync::Arc;

use super::{ItemSource, ObjectRange};
use crate::failure::{Kind, Result};
use crate::format::Header;
#[cfg(unix)]
use crate::{format::ObjectRef, store::Checkpoint};

#[cfg(unix)]
type Fetch = dyn Fn(&[crate::format::ObjectRef]) -> Result<()> + Send + Sync;

/// A checkpoint's chunks as a placement source (`io::source`, Unix).
#[cfg(unix)]
#[derive(Clone)]
pub(crate) struct View {
    header: Arc<Header>,
    chunks: Arc<BTreeMap<String, (u64, usize)>>,
    stored: Checkpoint,
    fetch: Option<Arc<Fetch>>,
}

/// `len` stored bytes of the part `what` (`component/key#role`), in part order.
#[derive(Debug, Clone)]
pub struct Span {
    pub what: String,
    pub len: u64,
    pub source: ItemSource,
}

pub struct StoredPart {
    pub component: String,
    pub key: String,
    pub role: String,
    pub dtype: String,
    pub shape: Vec<u64>,
}

#[cfg(unix)]
impl From<&Checkpoint> for View {
    fn from(checkpoint: &Checkpoint) -> View {
        View::new(checkpoint.header().clone(), checkpoint.clone())
    }
}

#[cfg(unix)]
impl View {
    fn new(header: Header, stored: Checkpoint) -> Self {
        let chunks = header
            .chunks
            .iter()
            .enumerate()
            .map(|(i, c)| (c.sha256.clone(), (c.length, i)))
            .collect();
        Self {
            header: Arc::new(header),
            chunks: Arc::new(chunks),
            stored,
            fetch: None,
        }
    }

    /// Keep the source context and use its one downloader for the selected chunks.
    pub(crate) fn with_fetch(
        mut self,
        fetch: impl Fn(&[crate::format::ObjectRef]) -> Result<()> + Send + Sync + 'static,
    ) -> Self {
        self.fetch = Some(Arc::new(fetch));
        self
    }

    pub(crate) fn files_for(&self, objects: &[ObjectRef]) -> Result<Vec<(ObjectRef, File)>> {
        let indices = objects
            .iter()
            .map(|object| {
                self.chunks
                    .get(&object.sha256)
                    .filter(|(length, _)| *length == object.length)
                    .map(|(_, index)| *index)
                    .ok_or_else(|| crate::failure::Failure {
                        kind: Kind::Invalid,
                        detail: format!("{}: not a checkpoint chunk", object.sha256),
                    })
            })
            .collect::<Result<Vec<_>>>()?;
        self.files(&indices)
    }

    pub(crate) fn files(&self, chunks: &[usize]) -> Result<Vec<(ObjectRef, File)>> {
        let checkpoint = &self.stored;
        let objects = chunks
            .iter()
            .map(|&index| {
                self.header
                    .chunks
                    .get(index)
                    .cloned()
                    .ok_or_else(|| crate::failure::Failure {
                        kind: Kind::Invalid,
                        detail: format!("no chunk {index}"),
                    })
            })
            .collect::<Result<Vec<_>>>()?;
        if let Some(fetch) = &self.fetch {
            fetch(&objects)?;
        }
        chunks
            .iter()
            .zip(objects)
            .map(|(&index, chunk)| {
                Ok((chunk, checkpoint.chunk(index)?))
            })
            .collect()
    }
}

fn locate<'a>(header: &Header, name: &'a str) -> Option<(usize, usize, Option<&'a str>)> {
    let (component, key) = name.split_once('/')?;
    if let Some((ci, ti)) = header.position(component, key) {
        return Some((ci, ti, None));
    }
    let (key, role) = key.rsplit_once('#')?;
    let (ci, ti) = header.position(component, key)?;
    let tensor = &header.components()[ci].tensors[ti];
    tensor
        .locations()
        .iter()
        .any(|(r, _)| *r == role || (r.is_empty() && role == "value"))
        .then_some((ci, ti, Some(role)))
}

/// The stored spans of the parts `regions` name, in header order, and the chunks they read.
pub(crate) fn plan(header: &Header, regions: &[Vec<String>]) -> Result<(Vec<Span>, Vec<usize>)> {
    let mut wanted = Vec::new();
    for name in regions.iter().flatten() {
        wanted.push(locate(header, name).ok_or_else(|| crate::failure::Failure {
            kind: Kind::Invalid,
            detail: format!("{name:?} is not in the checkpoint"),
        })?);
    }
    wanted.sort_unstable();
    wanted.dedup();
    let (mut items, mut chunks) = (Vec::new(), Vec::new());
    for (ci, ti, only) in wanted {
        let component = &header.components()[ci];
        let tensor = &component.tensors[ti];
        for (role, at) in tensor.locations() {
            let role = if role.is_empty() { "value" } else { role };
            if only.is_some_and(|selected| selected != role) {
                continue;
            }
            let what = format!("{}/{}#{role}", component.name, tensor.key);
            let spans = header.spans(&at);
            if spans.is_empty() {
                items.push(Span {
                    what,
                    len: 0,
                    source: ItemSource::Inline(Arc::from([])),
                });
                continue;
            }
            for span in spans {
                let object = &header.chunks[span.chunk];
                items.push(Span {
                    what: what.clone(),
                    len: span.length,
                    source: ItemSource::Object(ObjectRange {
                        obj: object.clone(),
                        off: span.start,
                        len: span.length,
                    }),
                });
                chunks.push(span.chunk);
            }
        }
    }
    chunks.sort_unstable();
    chunks.dedup();
    Ok((items, chunks))
}

pub(crate) fn part(header: &Header, name: &str) -> Option<StoredPart> {
    let (tensor, role) = name.rsplit_once('#')?;
    let (component, key) = tensor.split_once('/')?;
    let tensor = header.tensor(component, key)?;
    let role = if role == "value" { "" } else { role };
    let (dtype, shape) = match tensor.part(role) {
        Some(part) => (part.dtype.name().to_owned(), part.shape.clone()),
        None if role.is_empty() => (tensor.dtype.name().to_owned(), tensor.shape.clone()),
        None => return None,
    };
    Some(StoredPart {
        component: component.into(),
        key: key.into(),
        role: role.into(),
        dtype,
        shape,
    })
}
