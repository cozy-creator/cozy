use std::collections::BTreeMap;
use std::fmt;
use std::fs::File;
use std::path::PathBuf;
use std::sync::Arc;

use crate::failure::{fail, Kind, Result};
use crate::format::{self, Body, ObjectRef, PartBytes};
use crate::io::{Dst, ReadOpts, Target};
use crate::store2::{CacheGuard, Checkpoint as Stored, Store};

use super::{remote::Remote, Fetch, Fetched, Ref, Selection};

struct Data {
    raw: Stored,
    named: Option<Ref>,
    tree: Option<ObjectRef>,
    remote: Option<Remote>,
    _extra: CacheGuard,
}

/// An owned source. A cloned component keeps this data alive after its parent drops.
#[derive(Clone)]
pub struct Checkpoint {
    data: Arc<Data>,
    components: Arc<[Component]>,
}

impl fmt::Debug for Checkpoint {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Checkpoint")
            .field("id", self.id())
            .field("named", &self.data.named)
            .finish()
    }
}

impl Checkpoint {
    pub(super) fn local(
        raw: Stored,
        named: Option<Ref>,
        tree: Option<ObjectRef>,
        store: Store,
    ) -> Result<Self> {
        Self::managed(raw, named, tree, None, Vec::new(), store)
    }

    pub(super) fn managed(
        raw: Stored,
        named: Option<Ref>,
        tree: Option<ObjectRef>,
        remote: Option<Remote>,
        extra: Vec<ObjectRef>,
        store: Store,
    ) -> Result<Self> {
        let data = Arc::new(Data {
            raw,
            named,
            tree,
            remote,
            _extra: store.cache_hold(&extra)?,
        });
        let components = data
            .raw
            .components()
            .iter()
            .map(|c| Component {
                data: Arc::clone(&data),
                name: c.name.clone(),
            })
            .collect::<Vec<_>>()
            .into();
        Ok(Self { data, components })
    }

    pub fn id(&self) -> &ObjectRef {
        self.data.raw.id()
    }
    pub fn header(&self) -> &format::Header {
        self.data.raw.header()
    }
    pub fn metadata(&self) -> &format::Metadata {
        &self.header().metadata
    }
    pub fn chunks(&self) -> &[ObjectRef] {
        &self.header().chunks
    }
    pub fn named(&self) -> Option<&Ref> {
        self.data.named.as_ref()
    }
    pub fn pipeline(&self) -> Option<&format::Pipeline> {
        self.data.raw.pipeline()
    }
    pub fn components(&self) -> &[Component] {
        &self.components
    }

    pub fn component(&self, name: &str) -> Result<&Component> {
        self.components
            .iter()
            .find(|c| c.name == name)
            .ok_or_else(|| crate::failure::Failure {
                kind: Kind::NotFound,
                detail: format!("no component {name:?}"),
            })
    }

    /// Newly owned descriptor, valid after every checkpoint/component handle drops.
    pub fn chunk(&self, index: usize) -> Result<File> {
        if index >= self.chunks().len() {
            return fail(Kind::Invalid, format!("no chunk {index}"));
        }
        Ok(self.data.raw.chunk(index)?)
    }

    pub fn missing(&self, selection: &Selection) -> Result<Vec<ObjectRef>> {
        super::selection::objects(&self.data.raw, selection)?
            .into_iter()
            .filter_map(|object| match self.data.raw.holds(&object) {
                Ok(true) => None,
                Ok(false) => Some(Ok(object)),
                Err(error) => Some(Err(error.into())),
            })
            .collect()
    }

    pub async fn fetch(&self, want: &Fetch<'_>) -> Result<Fetched> {
        let objects = super::selection::objects(&self.data.raw, &want.selection)?;
        match &self.data.remote {
            Some(remote) => remote.fetch(&objects, want).await,
            None => {
                let total = objects.iter().map(|o| o.length).sum();
                for object in &objects {
                    if !self.data.raw.holds(object)? {
                        return fail(Kind::NotFound, format!("{} is not present", object.id()));
                    }
                }
                if let Some(report) = want.progress {
                    report(total, total)?;
                }
                Ok(Fetched {
                    local: total,
                    ..Fetched::default()
                })
            }
        }
    }

    pub fn folder(&self) -> Result<PathBuf> {
        let variant = self
            .named()
            .and_then(|r| r.variant.as_deref())
            .unwrap_or("");
        Ok(self.data.raw.path_for(variant)?)
    }

    pub fn tree(&self) -> Option<&ObjectRef> {
        self.data.tree.as_ref()
    }

    pub fn verify(&self) -> Result<Vec<ObjectRef>> {
        Ok(self.data.raw.verify()?)
    }

    /// The same placement/byte-engine view, retaining this source without a cycle.
    /// Its callback checks availability only; Rust callers explicitly await fetch.
    pub fn view(&self) -> crate::io::view::View {
        let data = Arc::clone(&self.data);
        crate::io::view::View::from(&data.raw).with_fetch(move |objects| {
            for object in objects {
                if !data.raw.holds(object)? {
                    return fail(
                        Kind::NotFound,
                        format!(
                            "{} is not present; fetch it asynchronously first",
                            object.id()
                        ),
                    );
                }
            }
            Ok(())
        })
    }
}

#[derive(Clone)]
pub struct Component {
    data: Arc<Data>,
    name: String,
}

impl fmt::Debug for Component {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Component")
            .field("name", &self.name)
            .field("checkpoint", self.data.raw.id())
            .finish()
    }
}

impl Component {
    fn metadata(&self) -> &format::Component {
        self.data
            .raw
            .component(&self.name)
            .expect("component validated against an immutable header")
    }

    pub fn name(&self) -> &str {
        &self.name
    }
    pub fn class(&self) -> &str {
        &self.metadata().class
    }
    pub fn config(&self) -> &format::Config {
        &self.metadata().config
    }
    pub fn config_file(&self) -> &str {
        self.metadata().config_name()
    }
    pub fn tensors(&self) -> &[format::Tensor] {
        &self.metadata().tensors
    }
    pub fn assets(&self) -> &BTreeMap<String, ObjectRef> {
        &self.metadata().assets
    }

    /// Payload bytes must first be made available by async Checkpoint::fetch.
    pub fn read(&self, key: &str) -> Result<Vec<u8>> {
        let tensor = self.data.raw.tensor(&self.name, key)?;
        match &tensor.body {
            Body::Data(at) => {
                let mut bytes = vec![0; at.length as usize];
                self.read_part_into(key, "", 0, &mut bytes)?;
                Ok(bytes)
            }
            Body::Encoded { encoding, parts } => {
                let mut buffers = Vec::with_capacity(parts.len());
                for part in parts {
                    let mut bytes = vec![0; part.at().length as usize];
                    self.read_part_into(key, &part.role, 0, &mut bytes)?;
                    buffers.push(bytes);
                }
                let views = parts
                    .iter()
                    .zip(&buffers)
                    .map(|(part, bytes)| {
                        (
                            part.role.as_str(),
                            PartBytes {
                                dtype: &part.dtype,
                                shape: &part.shape,
                                bytes,
                            },
                        )
                    })
                    .collect();
                Ok(format::decode(
                    &tensor.dtype,
                    &tensor.shape,
                    encoding,
                    &views,
                )?)
            }
        }
    }

    /// This safe borrow is held until the existing synchronous engine returns.
    #[allow(unsafe_code)]
    pub fn read_part_into(&self, key: &str, role: &str, offset: u64, buf: &mut [u8]) -> Result<()> {
        let target = Target {
            component: &self.name,
            key,
            role,
            offset,
            dst: Dst::Host {
                ptr: buf.as_mut_ptr(),
                len: buf.len(),
            },
        };
        // SAFETY: buf is exclusively borrowed and remains alive until read_into
        // completes. The engine validates the part/range; the owned source keeps
        // every file/retention owner alive. No new chunk copy path is introduced.
        unsafe { self.data.raw.read_into(&[target], &ReadOpts::default()) }?;
        Ok(())
    }

    pub fn asset(&self, name: &str) -> Result<Vec<u8>> {
        Ok(self.data.raw.asset(&self.name, name)?)
    }
    pub fn folder(&self) -> Result<PathBuf> {
        Ok(self.data.raw.component_path(&self.name)?)
    }
}
