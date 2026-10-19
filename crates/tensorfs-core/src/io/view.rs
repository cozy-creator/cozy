//! Stored byte descriptions for placement callers. Header details and chunk opens stay here.

use std::fs::File;
use std::sync::Arc;

use crate::failure::{fail, Kind, Result};
use crate::format::Header;
use crate::ids::ObjectRef;
use crate::read::{ObjectRange, Order, PlanItem, ReadPlan, Source};
use crate::store2::Checkpoint;

type Fetch = dyn Fn(&[crate::format::ObjectRef]) -> Result<()> + Send + Sync;

#[derive(Clone)]
pub struct View {
    header: Header,
    stored: Option<Checkpoint>,
    fetch: Option<Arc<Fetch>>,
}

pub struct Part {
    pub component: String,
    pub key: String,
    pub role: String,
    pub dtype: String,
    pub shape: Vec<u64>,
}

impl From<&Checkpoint> for View {
    fn from(checkpoint: &Checkpoint) -> View {
        View {
            header: checkpoint.header().clone(),
            stored: Some(checkpoint.clone()),
            fetch: None,
        }
    }
}

impl View {
    pub fn parsed(header: Header) -> View {
        View {
            header,
            stored: None,
            fetch: None,
        }
    }

    /// Keep the source context and use its one downloader for the selected chunks.
    pub fn with_fetch(
        mut self,
        fetch: impl Fn(&[crate::format::ObjectRef]) -> Result<()> + Send + Sync + 'static,
    ) -> Self {
        self.fetch = Some(Arc::new(fetch));
        self
    }

    pub fn has_source(&self) -> bool {
        self.stored.is_some()
    }

    pub fn plan(&self, regions: &[Vec<String>]) -> Result<(ReadPlan, Vec<usize>)> {
        let header = &self.header;
        let mut wanted = Vec::new();
        for name in regions.iter().flatten() {
            wanted.push(locate(header, name).ok_or_else(|| crate::failure::Failure {
                kind: Kind::Invalid,
                detail: format!("{name:?} is not in the checkpoint"),
            })?);
        }
        wanted.sort_unstable();
        wanted.dedup();
        let (mut items, mut chunks, mut dest) = (Vec::new(), Vec::new(), 0);
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
                    items.push(PlanItem {
                        what,
                        dest_off: dest,
                        len: 0,
                        source: Source::Inline(Vec::new()),
                    });
                    continue;
                }
                for span in spans {
                    let object = &header.chunks[span.chunk];
                    let obj = ObjectRef {
                        sha256: object.sha256.clone(),
                        length: object.length,
                    };
                    items.push(PlanItem {
                        what: what.clone(),
                        dest_off: dest,
                        len: span.length,
                        source: Source::Object(ObjectRange {
                            obj,
                            off: span.start,
                            len: span.length,
                        }),
                    });
                    chunks.push(span.chunk);
                    dest += span.length;
                }
            }
        }
        chunks.sort_unstable();
        chunks.dedup();
        Ok((
            ReadPlan {
                io_bytes: dest,
                bytes: dest,
                items,
                inline_parts: 0,
                order: Order::Destination,
            },
            chunks,
        ))
    }

    pub fn files(&self, chunks: &[usize]) -> Result<Vec<(ObjectRef, File)>> {
        let Some(checkpoint) = &self.stored else {
            return fail(Kind::Unsupported, "a parsed header has no chunk source");
        };
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
                Ok((
                    ObjectRef {
                        sha256: chunk.sha256,
                        length: chunk.length,
                    },
                    checkpoint.chunk(index)?,
                ))
            })
            .collect()
    }

    pub fn part(&self, name: &str) -> Option<Part> {
        let (tensor, role) = name.rsplit_once('#')?;
        let (component, key) = tensor.split_once('/')?;
        let tensor = self.header.tensor(component, key)?;
        let role = if role == "value" { "" } else { role };
        let (dtype, shape) = match tensor.part(role) {
            Some(part) => (part.dtype.name().to_owned(), part.shape.clone()),
            None if role.is_empty() => (tensor.dtype.name().to_owned(), tensor.shape.clone()),
            None => return None,
        };
        Some(Part {
            component: component.into(),
            key: key.into(),
            role: role.into(),
            dtype,
            shape,
        })
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
