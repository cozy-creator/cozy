//! The repo documents: `cozytensors.tree/1`, a revision's file tree, and `cozytensors.repo/1`, a
//! model's releases of trees. Both parse through a `Wire` twin that carries the `format` tag.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use super::doc::{path, segment};
use super::{check_format, invalid, json, variant_of, variant_path, Error, ObjectRef, Result};
use super::{REPO_FORMAT, TREE_FORMAT};

/// The folder that holds a plain folder's chunks and assets: no repo file lives under it.
pub const CHUNKS_DIR: &str = "chunks";

/// `cozytensors.tree/1`: a repo revision as files, as on HF. Each `model.<variant>.cozytensors` is
/// a variant's header; every other file (README, LICENSE, images) is a blob shared by them all.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(try_from = "TreeWire", into = "TreeWire")]
pub struct Tree {
    pub files: BTreeMap<String, ObjectRef>,
}

#[derive(Serialize, Deserialize)]
struct TreeWire {
    format: String,
    files: BTreeMap<String, ObjectRef>,
}

impl TryFrom<TreeWire> for Tree {
    type Error = Error;
    fn try_from(w: TreeWire) -> Result<Self> {
        check_format(&w.format, TREE_FORMAT)?;
        for p in w.files.keys() {
            check_file(p)?;
        }
        Ok(Tree { files: w.files })
    }
}

impl From<Tree> for TreeWire {
    fn from(t: Tree) -> Self {
        TreeWire {
            format: TREE_FORMAT.into(),
            files: t.files,
        }
    }
}

/// A repo file path: plain segments, not under [`CHUNKS_DIR`].
pub(crate) fn check_file(p: &str) -> Result<()> {
    if !path(p) || p.split('/').next() == Some(CHUNKS_DIR) {
        return invalid(format!("file path {p:?} is not a repo file path"));
    }
    Ok(())
}

impl Tree {
    pub fn from_json(bytes: &[u8]) -> Result<Tree> {
        json::read(bytes, TREE_FORMAT)
    }
    pub fn to_json(&self) -> Result<Vec<u8>> {
        json::write(self)
    }

    /// Each variant (`""` for `model.cozytensors`) and its header, by variant name.
    pub fn variants(&self) -> impl Iterator<Item = (&str, &ObjectRef)> {
        (self.files.iter()).filter_map(|(p, r)| Some((variant_of(p)?, r)))
    }

    /// The header of `variant`.
    pub fn header(&self, variant: &str) -> Option<&ObjectRef> {
        self.files.get(&variant_path(variant))
    }

    /// `variant`, or with `None` the tree's only one: (its name, its header).
    pub fn variant(&self, variant: Option<&str>) -> Result<(&str, &ObjectRef)> {
        let found = match variant {
            Some(v) => self.files.get_key_value(&variant_path(v)).map(|(p, r)| (variant_of(p), r)),
            None => match self.variants().collect::<Vec<_>>()[..] {
                [(v, r)] => Some((Some(v), r)),
                _ => None,
            },
        };
        match found {
            Some((Some(v), r)) => Ok((v, r)),
            _ => {
                let has: Vec<&str> = self.variants().map(|(v, _)| v).collect();
                let want = variant.map_or("its only variant".into(), |v| format!("variant {v:?}"));
                invalid(format!("the tree has no {want} (has {has:?})"))
            }
        }
    }

    /// The repo files: everything but the variants' headers.
    pub fn repo_files(&self) -> impl Iterator<Item = (&str, &ObjectRef)> {
        (self.files.iter())
            .filter(|(p, _)| variant_of(p).is_none())
            .map(|(p, r)| (p.as_str(), r))
    }

    /// Sets `variant`'s header, replacing the one it had.
    pub fn set_variant(&mut self, variant: &str, header: ObjectRef) {
        self.files.insert(variant_path(variant), header);
    }
}

/// `refs/<org>/<name>.json`, ASCII-lowercased: repo names are case-insensitive, one identity.
pub fn repo_key(name: &str) -> String {
    format!("refs/{}.json", name.to_ascii_lowercase())
}

/// `cozytensors.repo/1`: a model's releases, each naming a tree, and its unreleased trees. Kept at
/// [`repo_key`]; releases are in publication order.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
#[serde(try_from = "RepoWire", into = "RepoWire")]
pub struct Repo {
    pub name: String,
    pub releases: Vec<Release>,
    /// Trees uploaded that no release names yet. Roots, like release trees.
    pub checkpoints: Vec<ObjectRef>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Release {
    pub version: String,
    pub tree: ObjectRef,
    /// Skipped by an unversioned lookup; an explicit version still resolves it.
    #[serde(default, skip_serializing_if = "is_false")]
    pub yanked: bool,
}

fn is_false(b: &bool) -> bool {
    !b
}

#[derive(Serialize, Deserialize)]
struct RepoWire {
    format: String,
    name: String,
    releases: Vec<Release>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    checkpoints: Vec<ObjectRef>,
}

impl TryFrom<RepoWire> for Repo {
    type Error = Error;
    fn try_from(w: RepoWire) -> Result<Self> {
        check_format(&w.format, REPO_FORMAT)?;
        match w.name.split_once('/') {
            Some((org, name)) if segment(org) && segment(name) => {}
            _ => return invalid(format!("repo name {:?} is not <org>/<name>", w.name)),
        }
        let mut seen = std::collections::HashSet::new();
        if let Some(r) = w.releases.iter().find(|r| r.version.is_empty() || !seen.insert(&r.version)) {
            return invalid(format!("release version {:?} is empty or repeated", r.version));
        }
        Ok(Repo {
            name: w.name,
            releases: w.releases,
            checkpoints: w.checkpoints,
        })
    }
}

impl From<Repo> for RepoWire {
    fn from(r: Repo) -> Self {
        RepoWire {
            format: REPO_FORMAT.into(),
            name: r.name,
            releases: r.releases,
            checkpoints: r.checkpoints,
        }
    }
}

impl Repo {
    pub fn new(name: &str) -> Repo {
        Repo {
            name: name.into(),
            ..Repo::default()
        }
    }
    pub fn from_json(bytes: &[u8]) -> Result<Repo> {
        json::read(bytes, REPO_FORMAT)
    }
    pub fn to_json(&self) -> Result<Vec<u8>> {
        json::write(self)
    }
    /// The repo's key in a store or static origin: [`repo_key`] of its name.
    pub fn key(&self) -> String {
        repo_key(&self.name)
    }
    /// Release `version`, yanked or not; with `None`, the newest release that is not yanked.
    pub fn release(&self, version: Option<&str>) -> Option<&Release> {
        match version {
            Some(v) => self.releases.iter().find(|r| r.version == v),
            None => self.releases.iter().rev().find(|r| !r.yanked),
        }
    }
    /// Release `version` names `tree`: a new release goes last, an existing one keeps its place.
    pub fn set_release(&mut self, version: &str, tree: ObjectRef) {
        match self.releases.iter_mut().find(|r| r.version == version) {
            Some(r) => r.tree = tree,
            None => self.releases.push(Release {
                version: version.into(),
                tree,
                yanked: false,
            }),
        }
    }
    /// Every tree the repo keeps alive: each release's (yanked too) and each checkpoint.
    pub fn roots(&self) -> impl Iterator<Item = &ObjectRef> {
        self.releases.iter().map(|r| &r.tree).chain(&self.checkpoints)
    }
}
