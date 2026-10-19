//! The library's front door, shared by the `tfs` CLI and Python: references, import into a
//! named repo, pull (from Tensorhub by default, or a static origin), and plane regions.
//!
//! A reference is `org/name[@release]` (with a variant, else the release tree's only one) or a
//! digest, `sha256:<hex>` or bare hex, of a tree or a cozytensors header.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::fs;
use std::path::Path;
use std::sync::Arc;

use crate::failure::{fail, Kind, Result};
use crate::fetch::{self, policy, Client, Config, Endpoint, Fetcher, ObjectStore, Origin};
use crate::fetch::{SourcePolicy, TensorhubOrigin, Want, TENSORHUB};
use crate::format::{repo_key, Body, Component, Header, ObjectRef, Release, Repo, Tensor, Tree};
use crate::import::{self, Access};
use crate::store2::Store;

/// The release an import names.
pub const IMPORT_RELEASE: &str = "1";

/// A cozytensors, and the tree, repo, release and variant that name it, when one does.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Named {
    pub header: ObjectRef,
    pub tree: Option<ObjectRef>,
    pub name: Option<String>,
    pub release: Option<String>,
    pub variant: Option<String>,
}

enum Reference<'a> {
    Name(&'a str, Option<&'a str>),
    Digest(&'a str),
}

fn parse(reference: &str) -> Reference<'_> {
    let hex = reference.strip_prefix("sha256:").unwrap_or(reference);
    if hex.len() == 64 && hex.bytes().all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f')) {
        return Reference::Digest(hex);
    }
    match reference.split_once('@') {
        Some((name, release)) => Reference::Name(name, Some(release)),
        None => Reference::Name(reference, None),
    }
}

/// `repo` with release `version` naming `tree`. Releases keep `order`'s order where it has
/// them, so the latest stays last, and take its yanked and revision.
fn with_release(repo: Option<Repo>, name: &str, version: &str, tree: &ObjectRef, order: &[Release]) -> Repo {
    let mut repo = repo.unwrap_or_else(|| Repo::new(name));
    repo.set_release(version, tree.clone());
    if let Some(origin) = order.iter().find(|r| r.version == version) {
        let release = (repo.releases.iter_mut()).find(|r| r.version == version).expect("just set");
        (release.yanked, release.revision) = (origin.yanked, origin.revision);
    }
    let rank = |v: &str| order.iter().position(|r| r.version == v).map_or(0, |p| p + 1);
    repo.releases.sort_by_key(|r| rank(&r.version));
    repo
}

/// What [`Store::import_named`] does besides the sources.
pub struct ImportOptions<'a> {
    /// `org/name`, or a bare name under `local/`; by default the first source's name.
    pub name: Option<&'a str>,
    pub converter: Option<&'a str>,
    pub variant: Option<&'a str>,
    pub progress: &'a (dyn Fn(&str, u64, u64) + Sync),
}

/// Provider credentials from `HF_TOKEN` and `CIVITAI_TOKEN`.
pub fn access() -> Access {
    let bearer = |var: &str| {
        std::env::var(var)
            .ok()
            .filter(|t| !t.trim().is_empty())
            .map_or_else(String::new, |t| format!("bearer {}", t.trim()))
    };
    Access {
        huggingface_token: bearer("HF_TOKEN"),
        civitai_token: bearer("CIVITAI_TOKEN"),
        ..Access::default()
    }
}

/// `local/<the source's own name>`, in ref grammar.
fn default_name(source: &str) -> String {
    let source = source.trim().trim_end_matches('/');
    let stem = if let Some(rest) = source.strip_prefix("hf://") {
        rest.split('/')
            .nth(1)
            .unwrap_or(rest)
            .split('@')
            .next()
            .unwrap_or(rest)
            .to_owned()
    } else if let Some(rest) = source.strip_prefix("civitai://") {
        format!("civitai-{}", rest.replace('/', "-"))
    } else {
        let last = source
            .split(['/', '?'])
            .rfind(|s| !s.is_empty() && !s.contains('='));
        let last = last.unwrap_or("model");
        last.strip_suffix(".safetensors").unwrap_or(last).to_owned()
    };
    let clean: String = stem
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || "._-".contains(c) {
                c
            } else {
                '-'
            }
        })
        .collect();
    format!("local/{}", clean.trim_start_matches('.'))
}

/// The lowercase dtype holding most of the header's bytes (`f32`, `bf16`), or the scheme of
/// most encoded bytes: an import's variant.
fn main_variant(header: &Header) -> String {
    let mut bytes: BTreeMap<String, u64> = BTreeMap::new();
    for (_, t) in header.tensors() {
        let (name, length) = match &t.body {
            Body::Data(at) => (t.dtype.name().to_ascii_lowercase(), at.length),
            Body::Encoded { encoding, parts } => (
                encoding.scheme.clone(),
                parts.iter().map(|p| p.length).sum(),
            ),
        };
        *bytes.entry(name).or_default() += length;
    }
    bytes
        .into_iter()
        .max_by_key(|(_, n)| *n)
        .map_or_else(|| "files".to_owned(), |(name, _)| name)
}

impl Store {
    /// The cozytensors `reference` names here: a name's release (its `variant`), a tree's
    /// variant, or a header.
    pub fn resolve(&self, reference: &str, variant: Option<&str>) -> Result<Named> {
        match parse(reference) {
            Reference::Name(name, release) => {
                let Some(repo) = self.get_ref(name)? else {
                    return fail(Kind::NotFound, format!("no ref {name} in {}", self.root().display()));
                };
                let read = |t: &ObjectRef| self.tree(t).map_err(|e| e.refusal());
                let chosen = fetch::choose(&repo, release, variant, read)?;
                Ok(Named {
                    header: chosen.header,
                    tree: Some(chosen.tree),
                    name: Some(repo.name),
                    release: Some(chosen.version),
                    variant: Some(chosen.variant),
                })
            }
            Reference::Digest(sha256) => {
                let path = self.path(&ObjectRef {
                    sha256: sha256.to_owned(),
                    length: 0,
                })?;
                let length = match fs::metadata(&path) {
                    Ok(m) => m.len(),
                    Err(_) => return fail(Kind::NotFound, format!("no object {sha256} here")),
                };
                let object = ObjectRef {
                    sha256: sha256.to_owned(),
                    length,
                };
                let bytes = self.read(&object)?;
                let (tree, header, variant) = match Tree::from_json(&bytes) {
                    Ok(tree) => {
                        let (v, header) = fetch::variant_of_tree(&tree, variant)?;
                        (Some(object), header, Some(v))
                    }
                    Err(_) => {
                        Header::from_cbor(&bytes)?;
                        (None, object, None)
                    }
                };
                let releases = self.refs()?.into_iter().flat_map(|repo| {
                    let name = repo.name.clone();
                    repo.releases.into_iter().map(move |r| (name.clone(), r))
                });
                for (name, release) in releases {
                    let hit = match &tree {
                        Some(t) => release.tree == *t,
                        None => self.tree(&release.tree).is_ok_and(|t| t.variants().any(|(_, h)| *h == header)),
                    };
                    if hit {
                        let variant = match &variant {
                            Some(v) => v.clone(),
                            None => (self.tree(&release.tree)?.variants())
                                .find(|(_, h)| **h == header)
                                .map(|(v, _)| v.to_owned())
                                .expect("matched"),
                        };
                        return Ok(Named {
                            header,
                            tree: Some(release.tree),
                            name: Some(name),
                            release: Some(release.version),
                            variant: Some(variant),
                        });
                    }
                }
                Ok(Named {
                    header,
                    tree,
                    name: None,
                    release: None,
                    variant,
                })
            }
        }
    }

    /// Imports `sources` into one cozytensors and names it as release [`IMPORT_RELEASE`],
    /// variant its main dtype. Re-importing a name replaces that variant; an interrupted import
    /// resumes.
    pub fn import_named(&self, sources: &[String], opts: &ImportOptions) -> Result<Named> {
        let Some(first) = sources.first() else {
            return fail(Kind::Invalid, "import needs a source");
        };
        let name = match opts.name {
            Some(name) if name.contains('/') => name.to_owned(),
            Some(name) => format!("local/{name}"),
            None => default_name(first),
        };
        // A bad name fails here, before any download.
        self.get_ref(&name)?;
        let access = access();
        let req = import::Request {
            sources,
            converter: opts.converter,
            variant: opts.variant,
            access: &access,
            tmp: &self.root().join("tmp"),
            progress: opts.progress,
            stop: None,
        };
        let key = format!("import.{}", name.replace('/', ".").to_ascii_lowercase());
        let (header, files) = self.import(&req, Some(&key))?;
        let named = self.set_variant(&name, IMPORT_RELEASE, None, &header, &files)?;
        self.unpin(&key)?;
        Ok(named)
    }

    /// Names `header` as `variant` (by default its main dtype) of `name`'s release `version`:
    /// a new tree, the release's old one with that variant set and `files` added, becomes the
    /// release's.
    pub fn set_variant(
        &self,
        name: &str,
        version: &str,
        variant: Option<&str>,
        header: &ObjectRef,
        files: &BTreeMap<String, ObjectRef>,
    ) -> Result<Named> {
        let variant = match variant {
            Some(v) => v.to_owned(),
            None => main_variant(self.checkpoint(header)?.header()),
        };
        let old = self.get_ref(name)?;
        let mut tree = match old.as_ref().and_then(|r| r.release(Some(version))) {
            Some(r) => self.tree(&r.tree)?,
            None => Tree::default(),
        };
        tree.files.extend(files.clone());
        tree.set_variant(&variant, header.clone());
        let at = self.put_tree(&tree)?;
        let repo = self.update_ref(name, |old| Some(with_release(old, name, version, &at, &[])))?;
        Ok(Named {
            header: header.clone(),
            tree: Some(at),
            name: repo.map(|r| r.name),
            release: Some(version.to_owned()),
            variant: Some(variant),
        })
    }

    /// Pulls `reference` (`org/name[@release]`; from a Hub, `org/name@sha256:<hex>` names the
    /// owner's tree by hash) from `from`, its `variant` (else the tree's only one), then names
    /// the tree here as the origin names it.
    pub fn pull(&self, from: PullFrom<'_>, reference: &str, variant: Option<&str>) -> Result<Pulled> {
        match from {
            PullFrom::Origin(origin) => self.pull_static(origin, reference, variant),
            PullFrom::Hub { base, token } => {
                let base = base.unwrap_or(TENSORHUB);
                let client = Client::new(SourcePolicy {
                    any_host: true,
                    allow_local: policy::is_loopback_origin(base),
                    max_redirects: 5,
                    ..SourcePolicy::default()
                });
                let mut hub = TensorhubOrigin::new(&client, base)?;
                if let Some(token) = token {
                    hub = hub.bearer(token);
                }
                self.pull_hub(client, hub, reference, variant)
            }
        }
    }

    /// [`Store::pull`] from the Tensorhub `hub`, read over `client`. A tree asked by hash is
    /// not a release: no ref names it here.
    pub fn pull_hub(&self, client: Client, hub: TensorhubOrigin, reference: &str, variant: Option<&str>) -> Result<Pulled> {
        let Reference::Name(name, release) = parse(reference) else {
            return fail(Kind::Invalid, "a pull names org/name[@release], not a bare digest");
        };
        let store: Arc<dyn ObjectStore> = Arc::new(self.clone());
        let hub = Arc::new(hub);
        let (resolved, pulled) = fetch::runtime()?.block_on(async {
            let resolved = hub.named(name, release, variant).await?;
            let origin: Arc<dyn Origin> = hub.clone();
            let fetcher = Fetcher::new(store, client, None, Config::default());
            let asked = variant.or(Some(resolved.variant.as_str()).filter(|v| !v.is_empty()));
            let pulled = fetcher.pull_tree(&origin, &resolved.tree, asked, &Want::default()).await?;
            fetcher.settled().await;
            Ok::<_, crate::failure::Failure>((resolved, pulled))
        })?;
        let (name, release) = match resolved.release.is_empty() {
            true => (None, None),
            false => {
                let (model, release) = (&resolved.model, &resolved.release);
                let local = self.update_ref(model, |old| Some(with_release(old, model, release, &resolved.tree, &[])))?;
                (local.map(|r| r.name), Some(resolved.release))
            }
        };
        Ok(pulled_as(pulled, name, release))
    }

    fn pull_static(&self, origin: &str, reference: &str, variant: Option<&str>) -> Result<Pulled> {
        let Reference::Name(name, release) = parse(reference) else {
            return fail(Kind::Invalid, "a pull names org/name[@release], not a digest");
        };
        let store: Arc<dyn ObjectStore> = Arc::new(self.clone());
        let (version, repo, pulled) = fetch::runtime()?.block_on(async {
            let (origin, client, repo) = open(origin, name).await?;
            let (version, tree) = fetch::release_of(&repo, release)?;
            let fetcher = Fetcher::new(store, client, None, Config::default());
            let pulled = fetcher.pull_tree(&origin, &tree, variant, &Want::default()).await?;
            fetcher.settled().await;
            Ok::<_, crate::failure::Failure>((version, repo, pulled))
        })?;
        let tree = pulled.tree.clone().expect("a tree was pulled");
        let local = self.update_ref(&repo.name, |old| {
            Some(with_release(old, &repo.name, &version, &tree, &repo.releases))
        })?;
        Ok(pulled_as(pulled, local.map(|r| r.name), Some(version)))
    }
}

fn pulled_as(pulled: fetch::Pulled, name: Option<String>, release: Option<String>) -> Pulled {
    Pulled {
        named: Named {
            header: pulled.header,
            tree: pulled.tree,
            name,
            release,
            variant: Some(pulled.variant),
        },
        objects: pulled.objects,
        fetched: pulled.landed.origin + pulled.landed.volume,
        held: pulled.landed.local,
    }
}

/// Where a pull reads.
#[derive(Debug, Clone, Copy)]
pub enum PullFrom<'a> {
    /// A Tensorhub at `base` (by default [`TENSORHUB`]), anonymous for public release lanes; a
    /// bearer `token` reads the private checkpoints it grants.
    Hub {
        base: Option<&'a str>,
        token: Option<&'a str>,
    },
    /// A static origin: an http(s) URL or a directory in that layout.
    Origin(&'a str),
}

/// What a pull named, and its objects: those fetched (bytes) and those already here.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Pulled {
    pub named: Named,
    pub objects: usize,
    pub fetched: u64,
    pub held: u64,
}

/// The origin `spec` names (a URL or a directory) and its refs document for `name`.
async fn open(spec: &str, name: &str) -> Result<(Arc<dyn Origin>, Client, Repo)> {
    // The origin is the one the caller named; its objects are held to their digests wherever a
    // redirect sends them. Plaintext is for loopback only.
    let client = Client::new(SourcePolicy {
        any_host: true,
        allow_local: policy::is_loopback_origin(spec),
        max_redirects: 5,
        ..SourcePolicy::default()
    });
    let url = match spec.contains("://") {
        true => spec.to_owned(),
        false => std::path::absolute(spec).map_err(|e| io(Path::new(spec), e))?.display().to_string(),
    };
    let endpoint = Endpoint::new(&client, &url, None)?;
    let Some((repo, _)) = endpoint.repo(name).await? else {
        return fail(Kind::NotFound, format!("no ref {name} at {spec}"));
    };
    Ok((Arc::new(endpoint), client, repo))
}

/// The endpoint layout ([`Endpoint`]): `name`'s refs document gains `named`'s release, and
/// `blobs/` every object of its tree's closure here (`named`'s variant whole). Objects are hard-linked when they can be.
pub fn export_origin(store: &Store, named: &Named, dir: &Path) -> Result<Vec<String>> {
    let (Some(tree), Some(name), Some(version)) = (&named.tree, &named.name, &named.release) else {
        return fail(
            Kind::NotFound,
            format!("no release names {}: an origin serves refs, export a named one", named.header.sha256),
        );
    };
    for object in store.closure(&named.header)? {
        if !store.path(&object)?.exists() {
            return fail(Kind::NotFound, format!("object {} is not here", object.sha256));
        }
    }
    let mut written = Vec::new();
    // Other variants of the tree are served as far as they are here.
    for object in store.closure(tree)? {
        let key = object.key();
        let dest = dir.join(&key);
        if fs::metadata(&dest).is_ok_and(|m| m.len() == object.length) {
            continue;
        }
        let src = store.path(&object)?;
        if !src.exists() {
            continue;
        }
        let parent = dest.parent().expect("a blob has a shard");
        fs::create_dir_all(parent).map_err(|e| io(parent, e))?;
        let temp = parent.join(format!(".{}.tmp-{}", object.sha256, std::process::id()));
        let _ = fs::remove_file(&temp);
        if fs::hard_link(&src, &temp).is_err() {
            fs::copy(&src, &temp).map_err(|e| io(&temp, e))?;
        }
        fs::rename(&temp, &dest).map_err(|e| io(&dest, e))?;
        written.push(key);
    }
    let path = dir.join(repo_key(name));
    let old = match fs::read(&path) {
        Ok(bytes) => Some(Repo::from_json(&bytes)?),
        Err(_) => None,
    };
    let order = store.get_ref(name)?.map(|r| r.releases).unwrap_or_default();
    let repo = with_release(old, name, version, tree, &order);
    let parent = path.parent().expect("a ref has a directory");
    fs::create_dir_all(parent).map_err(|e| io(parent, e))?;
    fs::write(&path, repo.to_json()?).map_err(|e| io(&path, e))?;
    written.push(repo_key(name));
    Ok(written)
}

pub(crate) fn io(path: &Path, e: std::io::Error) -> crate::failure::Failure {
    let mut failure = crate::failure::Failure::from(e);
    failure.detail = format!("{}: {}", path.display(), failure.detail);
    failure
}

/// `header`'s tensors grouped into regions for a weight plane, as `<component>/<key>` names.
/// Tensors sharing a chunk stay in one group, its members in header order; groups follow the
/// header order of their first tensor, and neighbouring groups merge while their chunks total
/// at most `max_bytes` (a larger group is a region alone) and they hold the same `boundaries`.
/// A boundary is a `<component>/<key prefix>` module path (`prefix` or `prefix.*`), such as a
/// block: a region never holds tensors of two of them, or of one and of none, unless a shared
/// chunk does. A region never spans components.
///
/// `only`, when not empty, names what a load wants in the same form (blocks, or single
/// tensors): the work and the result cover just those tensors and the ones sharing a chunk with
/// them, never the rest of the checkpoint. A selection of whole boundaries gets the regions the
/// full call gives its tensors.
pub fn regions(header: &Header, max_bytes: u64, boundaries: &[String], only: &[String]) -> Vec<Vec<String>> {
    let split = |paths: &[String]| -> Vec<(String, String)> {
        let pairs = paths.iter().filter_map(|b| b.split_once('/'));
        pairs.map(|(c, p)| (c.to_owned(), p.to_owned())).collect()
    };
    let (boundaries, only) = (split(boundaries), split(only));
    let mut out = Vec::new();
    for c in header.components() {
        let of = |paths: &[(String, String)]| -> Vec<String> {
            paths.iter().filter(|(pc, _)| *pc == c.name).map(|(_, p)| p.clone()).collect()
        };
        let members = if only.is_empty() {
            (0..c.tensors.len()).collect()
        } else {
            let want = of(&only);
            if want.is_empty() {
                continue;
            }
            let picked = (0..c.tensors.len()).filter(|&i| want.iter().any(|p| under(&c.tensors[i].key, p)));
            sharing(header, &c.tensors, picked.collect())
        };
        out.extend(component_regions(header, c, max_bytes, &of(&boundaries), &members));
    }
    out
}

/// `key` is the module path `prefix` or inside it.
fn under(key: &str, prefix: &str) -> bool {
    key == prefix || key.strip_prefix(prefix).is_some_and(|rest| rest.starts_with('.'))
}

/// The chunks (first, last) each location of `t` covers.
fn chunk_ranges<'h>(header: &'h Header, t: &'h Tensor) -> impl Iterator<Item = (usize, usize)> + 'h {
    t.locations().into_iter().filter(|(_, at)| at.length > 0).map(|(_, at)| {
        let (mut chunk, mut end) = (at.chunk as usize, at.offset + at.length);
        while end > header.chunks[chunk].length {
            end -= header.chunks[chunk].length;
            chunk += 1;
        }
        (at.chunk as usize, chunk)
    })
}

/// `picked` (ascending tensor indices) and every tensor sharing a chunk with them, transitively.
fn sharing(header: &Header, tensors: &[Tensor], mut picked: Vec<usize>) -> Vec<usize> {
    let mut chunks: BTreeSet<usize> = BTreeSet::new();
    let mut taken = vec![false; tensors.len()];
    for &i in &picked {
        taken[i] = true;
        chunks.extend(chunk_ranges(header, &tensors[i]).flat_map(|(a, b)| a..=b));
    }
    loop {
        let before = picked.len();
        for (i, t) in tensors.iter().enumerate() {
            if taken[i] {
                continue;
            }
            let ranges: Vec<(usize, usize)> = chunk_ranges(header, t).collect();
            if ranges.iter().any(|&(a, b)| chunks.range(a..=b).next().is_some()) {
                taken[i] = true;
                picked.push(i);
                chunks.extend(ranges.into_iter().flat_map(|(a, b)| a..=b));
            }
        }
        if picked.len() == before {
            picked.sort_unstable();
            return picked;
        }
    }
}

/// [`regions`] of component `c` over `members` (ascending tensor indices, closed under chunk
/// sharing), with `boundaries` the component's key prefixes.
fn component_regions(header: &Header, c: &Component, max_bytes: u64, boundaries: &[String], members: &[usize]) -> Vec<Vec<String>> {
    let name = |i: usize| format!("{}/{}", c.name, c.tensors[i].key);
    // Groups of tensors (by index) and the chunks they use; the group each chunk is in.
    let mut groups: Vec<(Vec<usize>, BTreeSet<usize>)> = Vec::new();
    let mut empty: Vec<String> = Vec::new();
    let mut owner: HashMap<usize, usize> = HashMap::new();
    for &i in members {
        let chunks: BTreeSet<usize> = chunk_ranges(header, &c.tensors[i]).flat_map(|(a, b)| a..=b).collect();
        if chunks.is_empty() {
            empty.push(name(i));
            continue;
        }
        let mut joined: Vec<usize> = chunks.iter().filter_map(|k| owner.get(k).copied()).collect();
        joined.sort_unstable();
        joined.dedup();
        let g = match joined.split_first() {
            None => {
                groups.push((Vec::new(), BTreeSet::new()));
                groups.len() - 1
            }
            Some((&first, rest)) => {
                // Fold later groups into the earliest; their members keep header order.
                for &other in rest.iter().rev() {
                    let (m, used) = std::mem::take(&mut groups[other]);
                    groups[first].0.extend(m);
                    groups[first].1.extend(used);
                }
                groups[first].0.sort_unstable();
                first
            }
        };
        groups[g].0.push(i);
        groups[g].1.extend(chunks);
        for &k in &groups[g].1 {
            owner.insert(k, g);
        }
    }
    // The deepest boundary holding a key, if any.
    let label = |key: &str| boundaries.iter().filter(|p| under(key, p)).max_by_key(|p| p.len()).cloned();
    // Neighbours in the header: no tensor outside `members` lies between them.
    let rank = |i: usize| members.partition_point(|&m| m < i);
    // (names, chunk bytes, the boundaries its tensors are in, its last tensor)
    type Region = (Vec<String>, u64, BTreeSet<Option<String>>, usize);
    let mut regions: Vec<Region> = Vec::new();
    for (m, used) in groups.into_iter().filter(|g| !g.0.is_empty()) {
        let n: u64 = used.iter().map(|k| header.chunks[*k].length).sum();
        let labels: BTreeSet<Option<String>> = m.iter().map(|&i| label(&c.tensors[i].key)).collect();
        let (first, last) = (m[0], m[m.len() - 1]);
        let keys = m.into_iter().map(name);
        match regions.last_mut() {
            Some((r, bytes, l, end))
                if *bytes + n <= max_bytes && *l == labels && (first <= *end || first - *end == rank(first) - rank(*end)) =>
            {
                r.extend(keys);
                *bytes += n;
                *end = (*end).max(last);
            }
            _ => regions.push((keys.collect(), n, labels, last)),
        }
    }
    match regions.first_mut() {
        Some((r, ..)) => r.extend(empty),
        None if !empty.is_empty() => regions.push((empty, 0, BTreeSet::new(), 0)),
        None => {}
    }
    regions.into_iter().map(|(r, ..)| r).collect()
}

#[cfg(test)]
mod tests {
    use super::default_name;

    #[test]
    fn default_names() {
        assert_eq!(
            default_name("hf://acme/tiny-sdxl@abc/unet"),
            "local/tiny-sdxl"
        );
        assert_eq!(default_name("civitai://123/45"), "local/civitai-123-45");
        assert_eq!(default_name("/data/x/model.safetensors"), "local/model");
        assert_eq!(
            default_name("https://e.com/a/B c.safetensors?x=1"),
            "local/B-c"
        );
    }
}
