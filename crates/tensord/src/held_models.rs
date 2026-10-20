//! The models this machine's store holds, for Status: each local model (`local/<name>`) and
//! each Hub checkpoint it pulled (cached Hub refs), with what it weighs. A listing is read
//! again only when the set of models changed, not when one was merely used.
use crate::api::v1;
use crate::storage::{self, Store};
use std::{
    collections::{BTreeMap, HashMap},
    io,
    path::{Path, PathBuf},
    sync::Mutex,
};
use tensorfs::format::ObjectRef;

#[derive(Clone, Default, PartialEq, Debug)]
pub struct Listing {
    pub models: Vec<v1::Model>,
    /// Every held model's bytes together, shared content counted once.
    pub bytes: u64,
}

/// A Hub checkpoint this machine pulled: `lane` of `tree`, `repository`'s release `release`
/// (empty: the owner's tree by hash). The store's ref of the repository records it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Pulled {
    pub repository: String,
    pub release: String,
    pub lane: String,
    pub tree: ObjectRef,
}

impl Pulled {
    /// Names the pull in the store: `repository`'s ref, a cache TensorFS may expire. The tree
    /// must be here.
    pub fn record(&self, store: &Store) -> io::Result<()> {
        let release = (!self.release.is_empty()).then_some(self.release.as_str());
        let named = store
            .update_ref(&self.repository, |old| Some(tensorfs::fetch::with_tree(old, &self.repository, &self.tree, release)))
            .map_err(storage::io_error)?;
        match named {
            Some(repo) => store.cache_ref(&repo).map_err(storage::io_error),
            None => Ok(()),
        }
    }

    /// The Hub checkpoint `header` is, as this machine pulled it; None for one never pulled.
    pub fn of(store: &Store, header: &ObjectRef) -> io::Result<Option<Pulled>> {
        for repo in store.refs().map_err(storage::io_error)? {
            if repo.name.starts_with("local/") {
                continue;
            }
            let releases = repo.releases.iter().map(|r| (r.version.as_str(), &r.tree));
            for (release, tree) in releases.chain(repo.checkpoints.iter().map(|tree| ("", tree))) {
                let Ok(parsed) = store.tree(tree) else {
                    continue;
                };
                let lane = parsed.variants().find(|(_, h)| *h == header).map(|(lane, _)| lane.to_string());
                if let Some(lane) = lane {
                    let (repository, release) = (repo.name.clone(), release.to_string());
                    return Ok(Some(Pulled { repository, release, lane, tree: tree.clone() }));
                }
            }
        }
        Ok(None)
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
        let local = repo.name.starts_with("local/");
        let releases = repo.releases.iter().map(|r| (r.version.as_str(), &r.tree));
        for (release, tree) in releases.chain(repo.checkpoints.iter().map(|tree| ("", tree))) {
            let Ok(tree) = store.tree(tree) else {
                continue;
            };
            let release = if local { "" } else { release };
            // A Hub tree names every lane its release publishes: the ones pulled are here.
            for (variant, header) in tree.variants().filter(|(_, h)| storage::present(store, h)) {
                seen.push((repo.name.clone(), release.into(), variant.into(), header.clone()));
            }
        }
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
    use tensorfs::writer::{Data, Source};
    let store = storage::open(root)?;
    let model = |key: &str, tensor: &[u8]| -> std::io::Result<ObjectRef> {
        let mut writer = store.writer(None, Some(key)).map_err(storage::io_error)?;
        let mut unet = writer.component("unet", "", Default::default(), None).map_err(storage::io_error)?;
        unet.asset("tokenizer.json", &[1; 1000]).map_err(storage::io_error)?;
        let shape = [tensor.len() as u64];
        unet.put("w", &tensorfs::format::Dtype::U8, &shape, Data::Plain(Source::Read(&mut &tensor[..]))).map_err(storage::io_error)?;
        writer.commit().map_err(storage::io_error)
    };
    let acme = model("acme", &[2; 300])?;
    let mut tree = tensorfs::format::Tree::default();
    tree.set_variant("fp8", acme);
    let tree = store.put_tree(&tree).map_err(storage::io_error)?;
    Pulled { repository: "acme/base".into(), release: "1.0.0".into(), lane: "fp8".into(), tree }.record(&store)?;
    storage::set_local(&store, "mine", &model("mine", &[3; 50])?, &Default::default()).map(drop)
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
        drop(store.checkpoint(&acme).unwrap());
        assert_eq!(listing(&root), held);
        let pulled = Pulled::of(&store, &acme).unwrap().unwrap();
        assert_eq!((pulled.repository.as_str(), pulled.release.as_str(), pulled.lane.as_str()), ("acme/base", "1.0.0", "fp8"));
        store.update_ref("local/mine", |_| None).unwrap();
        assert_eq!(listing(&root).models.len(), 1);
        std::fs::remove_dir_all(&root).unwrap();
    }
}
