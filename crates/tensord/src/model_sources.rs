//! Stored metadata and assets for a session's selected models. TensorD supplies each verified
//! header, declared asset and selected component's encoded byte count through TensorFS.
//! `ModelSource` returns sealed read-only descriptors; HostTier supplies CPU weight buffers
//! and object descriptors. Runtime parses the metadata, constructs models and performs GPU
//! transfers/mappings with its own TensorFS plane. Selection remains the access boundary.
use crate::device_executor::{Answer, Frame};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs::File,
    io,
    path::Path,
    sync::Arc,
};
use crate::storage::{self, Store};
use tensorfs_core::store2::Checkpoint;

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct SelectedManifest {
    pub manifest: String,
    pub components: Vec<String>,
}

struct Selection {
    checkpoint: Checkpoint,
    header_bytes: Vec<u8>,
    components: Vec<String>,
    encoded_bytes: u64,
}

pub struct ModelSources {
    selected: BTreeMap<String, Selection>,
}

fn outside() -> io::Error {
    io::Error::new(io::ErrorKind::PermissionDenied, "manifest is outside selection")
}

fn digest(value: &str) -> io::Result<&str> {
    let raw = value.strip_prefix("sha256:").unwrap_or(value);
    if raw.len() != 64
        || !raw
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
    {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "model identity must be SHA-256",
        ));
    }
    Ok(raw)
}

impl ModelSources {
    pub fn open(root: &Path, selections: &[SelectedManifest]) -> io::Result<Self> {
        Self::open_shared(storage::open(root)?, selections)
    }

    /// The machine's store, read for the selected checkpoints only.
    pub fn open_shared(store: Arc<Store>, selections: &[SelectedManifest]) -> io::Result<Self> {
        let mut selected = BTreeMap::new();
        for selection in selections {
            let manifest = digest(&selection.manifest)?.to_owned();
            let checkpoint = storage::checkpoint(&store, &manifest)?;
            let header_bytes = store.read(checkpoint.id()).map_err(storage::io_error)?;
            // No components named: all of them (a `Weights[M]` source, planned by Runtime).
            let components: Vec<String> = match selection.components.is_empty() {
                true => checkpoint.header().components().iter().map(|c| c.name.clone()).collect(),
                false => selection.components.iter().cloned().collect::<BTreeSet<_>>().into_iter().collect(),
            };
            let mut encoded_bytes = 0;
            for component in &components {
                let tensors = checkpoint.header().component(component).ok_or_else(|| {
                    io::Error::new(io::ErrorKind::PermissionDenied, format!("selected component {component} is absent"))
                })?;
                encoded_bytes += (tensors.tensors.iter())
                    .flat_map(|t| t.locations().into_iter().map(|(_, at)| at.length))
                    .sum::<u64>();
            }
            let selection = Selection { checkpoint, header_bytes, components, encoded_bytes };
            if selected.insert(manifest, selection).is_some() {
                return Err(io::Error::new(io::ErrorKind::InvalidInput, "duplicate manifest selection"));
            }
        }
        Ok(Self { selected })
    }

    /// Selected encoded source size, separate from prepared CPU-buffer and GPU residency.
    /// Runtime measures its real device footprint; callers cannot infer a fit from this.
    pub fn selected_facts(&self, manifest: &str) -> io::Result<(Vec<String>, u64, u64)> {
        let selected = self.selected.get(digest(manifest)?).ok_or_else(outside)?;
        Ok((selected.components.clone(), selected.encoded_bytes, selected.checkpoint.id().length))
    }

    pub fn authorized(&self, manifest: &str) -> io::Result<Checkpoint> {
        self.selected.get(digest(manifest)?).map(|s| s.checkpoint.clone()).ok_or_else(outside)
    }

    /// One selected model's header (`name` empty), stored bytes as written, or one of its
    /// components' assets, `<component>/<name>` (tokenizers).
    pub fn source(&self, manifest: &str, name: &str) -> io::Result<Vec<u8>> {
        let selection = self.selected.get(digest(manifest)?).ok_or_else(outside)?;
        if name.is_empty() {
            return Ok(selection.header_bytes.clone());
        }
        let held = name.split_once('/').filter(|(component, asset)| {
            selection.checkpoint.header().component(component).is_some_and(|c| c.assets.contains_key(*asset))
        });
        let Some((component, asset)) = held else {
            return Err(io::Error::new(io::ErrorKind::PermissionDenied, "asset is outside the selected checkpoint"));
        };
        selection.checkpoint.asset(component, asset).map_err(storage::io_error)
    }

    /// Answer a `model_source` request: the bytes in a file nobody can write, with their digest.
    pub fn serve(&self, frame: &Frame) -> io::Result<(Answer, File)> {
        let bytes = self.source(&frame.manifest, &frame.name)?;
        let file = platform::shm::frozen("tensord-weights", &bytes)?;
        let mut answer = Answer::unavailable(frame.seq);
        (answer.ok, answer.held) = (true, true);
        answer.code.clear();
        answer.detail.clear();
        answer.sha256 = format!("{:x}", Sha256::digest(&bytes));
        answer.length = bytes.len() as u64;
        Ok((answer, file))
    }
}
