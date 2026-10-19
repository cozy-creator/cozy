//! The models this machine's store holds, for Status: each local model (`local/<name>` refs)
//! and each Hub checkpoint it pulled (`model-` pins), with what it weighs. A listing is read
//! again only when the set of models changed, not when one was merely used.
use crate::api::v1;
use crate::storage::{self, Store};
use serde::{Deserialize, Serialize};
use std::{
    collections::{BTreeMap, HashMap},
    path::{Path, PathBuf},
    sync::Mutex,
};
use tensorfs_core::format::ObjectRef;

#[derive(Clone, Default, PartialEq, Debug)]
pub struct Listing {
    pub models: Vec<v1::Model>,
    /// Every held model's bytes together, shared content counted once.
    pub bytes: u64,
}

/// What a `model-` pin's owner records: the checkpoint as the Hub named it, and the tree its
/// release names there.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Pulled {
    pub repository: String,
    pub release: String,
    pub lane: String,
    pub tree: Option<ObjectRef>,
}

impl Pulled {
    pub fn owner(&self) -> String {
        serde_json::to_string(self).expect("plain strings")
    }
}

/// Each model's rows: repository, release, lane, header.
type Seen = Vec<(String, String, String, ObjectRef)>;

static LAST: Mutex<Option<(PathBuf, Seen, Listing)>> = Mutex::new(None);

/// The store at `root`'s held models; none while it cannot be read.
pub fn listing(root: &Path) -> Listing {
    if !root.join("format").is_file() {
        return Listing::default();
    }
    let read = Store::open(root).map_err(storage::io_error).and_then(|store| Ok((seen(&store)?, store)));
    let (seen, store) = match read {
        Ok(read) => read,
        Err(error) => {
            tracing::warn!("held models unreadable: {error}");
            return Listing::default();
        }
    };
    let mut last = LAST.lock().unwrap();
    if let Some((at, was, listing)) = last.as_ref() {
        if at == root && *was == seen {
            return listing.clone();
        }
    }
    let listing = list(&store, &seen).unwrap_or_else(|error| {
        tracing::warn!("held models unreadable: {error}");
        Listing::default()
    });
    *last = Some((root.into(), seen, listing.clone()));
    listing
}

fn seen(store: &Store) -> std::io::Result<Seen> {
    let mut seen = Seen::new();
    for repo in store.refs().map_err(storage::io_error)? {
        for release in &repo.releases {
            let tree = store.tree(&release.tree).map_err(storage::io_error)?;
            let version = if repo.name.starts_with("local/") { "" } else { release.version.as_str() };
            for (variant, header) in tree.variants() {
                seen.push((repo.name.clone(), version.into(), variant.into(), header.clone()));
            }
        }
    }
    // Only the pulled checkpoints' pins are read: Status asks every second, and run objects
    // can number thousands.
    let mut ids: Vec<String> = std::fs::read_dir(store.root().join("pins"))?
        .flatten()
        .filter_map(|entry| entry.file_name().to_str()?.strip_suffix(".json").map(str::to_string))
        .filter(|id| id.starts_with("model-"))
        .collect();
    ids.sort();
    for id in ids {
        let Some(pin) = store.get_pin(&id).map_err(storage::io_error)? else {
            continue;
        };
        let Some(header) = pin.roots.first() else {
            continue;
        };
        let pulled: Pulled = pin.owner.as_deref().and_then(|o| serde_json::from_str(o).ok()).unwrap_or_default();
        seen.push((pulled.repository, pulled.release, pulled.lane, header.clone()));
    }
    seen.sort();
    Ok(seen)
}

fn list(store: &Store, seen: &Seen) -> std::io::Result<Listing> {
    let mut closures: BTreeMap<&str, Vec<ObjectRef>> = BTreeMap::new();
    let mut holders: HashMap<ObjectRef, usize> = HashMap::new();
    let mut models: Vec<v1::Model> = vec![];
    for (repository, release, lane, header) in seen {
        let closure = store.closure(header).map_err(storage::io_error)?;
        let held: Vec<ObjectRef> = closure.into_iter().filter(|o| storage::present(store, o)).collect();
        let entry = closures.entry(repository).or_default();
        for object in held {
            if !entry.contains(&object) {
                entry.push(object);
            }
        }
        if models.last().is_none_or(|m| m.repository != *repository) {
            models.push(v1::Model { repository: repository.clone(), ..Default::default() });
        }
        models.last_mut().expect("pushed").checkpoints.push(v1::Checkpoint {
            release: release.clone(),
            lane: lane.clone(),
            manifest: header.id(),
            yanked: false,
        });
    }
    for objects in closures.values() {
        for object in objects {
            *holders.entry(object.clone()).or_default() += 1;
        }
    }
    for model in &mut models {
        let objects = &closures[model.repository.as_str()];
        model.total_bytes = objects.iter().map(|o| o.length).sum();
        model.unique_bytes = objects.iter().filter(|o| holders[*o] == 1).map(|o| o.length).sum();
    }
    Ok(Listing { models, bytes: holders.keys().map(|o| o.length).sum() })
}

/// A store with two models that share one object (an asset): `acme/base` (release 1.0.0, lane
/// fp8), pulled, and the local model `local/mine`. Tests of the listing and of Status build on it.
pub fn fixture(root: &Path) -> std::io::Result<()> {
    use tensorfs_core::writer::{Data, Source};
    let store = storage::open(root)?;
    let model = |key: &str, tensor: &[u8]| -> std::io::Result<ObjectRef> {
        let mut writer = store.writer(None, Some(key)).map_err(storage::io_error)?;
        let mut unet = writer.component("unet", "", Default::default(), None).map_err(storage::io_error)?;
        unet.asset("tokenizer.json", &[1; 1000]).map_err(storage::io_error)?;
        let shape = [tensor.len() as u64];
        unet.put("w", &tensorfs_core::format::Dtype::U8, &shape, Data::Plain(Source::Read(&mut &tensor[..]))).map_err(storage::io_error)?;
        writer.commit().map_err(storage::io_error)
    };
    let acme = model("acme", &[2; 300])?;
    let pulled = Pulled { repository: "acme/base".into(), release: "1.0.0".into(), lane: "fp8".into(), tree: None };
    storage::hold(&store, &storage::pin::model(&acme.sha256), vec![acme], Some(pulled.owner()))?;
    storage::set_local(&store, "mine", &model("mine", &[3; 50])?, &Default::default())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lists_each_model_once_and_reads_again_only_on_change() {
        let root = std::env::temp_dir().join(format!("held-models-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        assert_eq!(listing(&root), Listing::default(), "no store, no models");
        fixture(&root).unwrap();
        let held = listing(&root);
        // Each model's own bytes: its header and its chunk.
        let store = Store::open(&root).unwrap();
        let own: Vec<u64> = held
            .models
            .iter()
            .map(|m| {
                let header = storage::held(&store, &m.checkpoints[0].manifest).unwrap();
                let checkpoint = store.checkpoint(&header).unwrap();
                header.length + checkpoint.header().chunks.iter().map(|c| c.length).sum::<u64>()
            })
            .collect();
        let rows: Vec<_> = held
            .models
            .iter()
            .map(|m| (m.repository.as_str(), m.total_bytes, m.unique_bytes))
            .collect();
        assert_eq!(rows, [("acme/base", 1000 + own[0], own[0]), ("local/mine", 1000 + own[1], own[1])]);
        assert_eq!(held.bytes, 1000 + own[0] + own[1], "the shared asset counts once");
        let release = &held.models[0].checkpoints[0];
        assert_eq!((release.release.as_str(), release.lane.as_str()), ("1.0.0", "fp8"));
        assert!(release.manifest.starts_with("sha256:") && !release.yanked);
        assert_eq!(held.models[1].checkpoints[0].release, "", "a local model has no release");
        // A use answers from the last read; a removed model is seen.
        let acme = storage::held(&store, &release.manifest).unwrap();
        store.touch_pin(&storage::pin::model(&acme.sha256)).unwrap();
        assert_eq!(listing(&root), held);
        store.update_ref("local/mine", |_| None).unwrap();
        assert_eq!(listing(&root).models.len(), 1);
        std::fs::remove_dir_all(&root).unwrap();
    }
}
