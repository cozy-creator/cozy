//! A loopback Hub that answers by name (th-245): one package's card, each release's card and install
//! plan (its pyproject and a uv.lock real uv made against PyPI and this Hub's org index), the org
//! index and file doors, and its models as the distribution protocol serves them (xs-054): a static
//! layout TensorFS's own `upload_tree` wrote, `GET /refs/<org>/<name>.json` and `GET`/`HEAD
//! /blobs/…`, each blob GET redirected to another origin (`/o/<sha>` at `localhost`), as the Hub
//! redirects to storage. Anonymous callers see releases (public tags) and their closures; the
//! owner's bearer ([`OWNER`], or those [`admit`] names) sees private tags (`main`) too and uploads
//! (`PUT` of blobs and conditional refs, as TensorFS's `upload_tree` sends them); any other
//! credential is 401.
//! It records every request.
#![allow(dead_code)]
use axum::{
    body::Bytes,
    extract::State,
    http::{HeaderMap, Method, StatusCode, Uri},
    response::{IntoResponse, Response},
};
use base64::Engine as _;
use serde_json::{json, Value};
use std::{
    collections::{BTreeMap, HashSet},
    path::PathBuf,
    sync::{Arc, Mutex},
};
use tensorfs::{
    fetch::{upload_tree, Client, Endpoint, SourcePolicy, MAIN},
    format::{repo_key, ObjectRef, Repo, Tree},
    store::Store,
};

/// The owner's ordinary bearer at this Hub.
pub const OWNER: &str = "names-hub-owner";

/// The owner bearers each Hub (by origin) admits, when not just [`OWNER`].
static ADMITTED: Mutex<BTreeMap<String, Vec<String>>> = Mutex::new(BTreeMap::new());

/// The Hub at `origin` admits exactly `bearers` as the owner's from now on: the rest are revoked.
pub fn admit(origin: &str, bearers: &[&str]) {
    ADMITTED.lock().unwrap().insert(origin.into(), bearers.iter().map(|b| b.to_string()).collect());
}

/// One model: `repository` at `release` in `lane` (no release: the owner's unreleased tree), its
/// cozytensors `header` held in a source store.
pub struct Model {
    pub repository: String,
    pub release: String,
    pub lane: String,
    pub header: ObjectRef,
}

/// The tree each model's release names, put in `store`: one per repository and release, each of
/// its lanes a variant in it.
pub fn trees(store: &Store, models: &[Model]) -> Vec<ObjectRef> {
    let mut trees: BTreeMap<(&str, &str), Tree> = BTreeMap::new();
    for model in models {
        trees.entry((&model.repository, &model.release)).or_default().set_variant(&model.lane, model.header.clone());
    }
    models.iter().map(|m| store.put_tree(&trees[&(m.repository.as_str(), m.release.as_str())]).unwrap()).collect()
}

/// A one-tensor checkpoint in `store`.
pub fn checkpoint(store: &Store, weights: &[u8]) -> ObjectRef {
    tensors(store, &[weights])
}

/// A checkpoint in `store` with one F32 tensor per entry of `weights` (16 MiB or more: its own
/// chunks).
pub fn tensors(store: &Store, weights: &[&[u8]]) -> ObjectRef {
    use tensorfs::{format::Dtype, writer::{Data, Source}};
    let mut writer = store.writer(None, None).unwrap();
    for (at, weights) in weights.iter().enumerate() {
        assert!(!weights.is_empty() && weights.len().is_multiple_of(4));
        let name = if at == 0 { "weight".to_string() } else { format!("weight{at}") };
        let shape = [(weights.len() / 4) as u64];
        writer.get("model").put(&name, &Dtype::F32, &shape, Data::Plain(Source::Read(&mut &weights[..]))).unwrap();
    }
    writer.commit().unwrap()
}

/// One published package release: its interface, its wheel and the wheels of the other
/// packages its lock pins at this Hub (`(wheel, bytes)`, published by the same org).
pub struct Package {
    pub name: String,
    pub release: String,
    pub interface: Value,
    pub wheel: String,
    pub bytes: Vec<u8>,
    pub callees: Vec<(String, Vec<u8>)>,
    /// The release's requirements beyond its callees, locked from PyPI.
    pub requires: Vec<String>,
}

struct Served {
    /// The static layout: `refs/` and `blobs/`.
    layout: PathBuf,
    /// Objects anyone may read: the closures of released trees.
    public: HashSet<String>,
    /// Releases of one package.
    packages: Vec<Package>,
    /// Each release's pyproject and uv.lock, locked once the Hub answers.
    locked: std::sync::OnceLock<Vec<(String, String)>>,
    origin: String,
    heard: Arc<Mutex<Vec<String>>>,
    /// How long each object answer takes, as a slow link's would.
    pace: std::time::Duration,
}

/// What a request was heard as: `METHOD path`, and `+credential` when it carried an admitted
/// bearer (`+refused`: another).
pub type Heard = Arc<Mutex<Vec<String>>>;

pub fn serve(store: &Store, model: Model, packages: impl IntoIterator<Item = Package>) -> (String, Heard) {
    paced(store, model, packages, std::time::Duration::ZERO)
}

/// `serve`, each object answering after `pace`.
pub fn paced(
    store: &Store,
    model: Model,
    packages: impl IntoIterator<Item = Package>,
    pace: std::time::Duration,
) -> (String, Heard) {
    start(store, vec![model], packages, pace)
}

/// Several models, or several lanes of one release.
pub fn models(store: &Store, models: Vec<Model>) -> (String, Heard) {
    start(store, models, None, std::time::Duration::ZERO)
}

/// `models`, each object answering after `pace`.
pub fn paced_models(store: &Store, models: Vec<Model>, pace: std::time::Duration) -> (String, Heard) {
    start(store, models, None, pace)
}

/// Each model's tree uploaded from `store` into a directory endpoint at `layout`, its release a
/// public tag, else the owner's `main`: the objects anyone may read.
fn publish(store: &Store, models: &[Model], layout: &std::path::Path) -> HashSet<String> {
    let trees = trees(store, models);
    let named: BTreeMap<(&str, &str), &ObjectRef> =
        models.iter().zip(&trees).map(|(m, tree)| ((m.repository.as_str(), m.release.as_str()), tree)).collect();
    let mut public = HashSet::new();
    let runtime = tensorfs::fetch::runtime().unwrap();
    std::thread::scope(|scope| {
        scope.spawn(|| {
            let client = Client::new(SourcePolicy::default());
            let endpoint = Endpoint::new(&client, &format!("file://{}", layout.display()), None).unwrap();
            for ((repository, release), tree) in &named {
                let tag = if release.is_empty() { MAIN } else { release };
                runtime.block_on(upload_tree(&endpoint, repository, tree, store, tag)).unwrap();
                if release.is_empty() {
                    continue;
                }
                let key = layout.join(repo_key(repository));
                let mut repo = Repo::from_json(&std::fs::read(&key).unwrap()).unwrap();
                repo.tags.iter_mut().filter(|t| t.name == *release).for_each(|t| t.public = true);
                std::fs::write(&key, repo.to_json().unwrap()).unwrap();
                public.extend(store.closure(tree).unwrap().into_iter().map(|o| o.sha256));
            }
        });
    });
    public
}

fn start(
    store: &Store,
    models: Vec<Model>,
    packages: impl IntoIterator<Item = Package>,
    pace: std::time::Duration,
) -> (String, Heard) {
    let layout = std::env::temp_dir().join(format!("names-hub-{}", uuid::Uuid::new_v4()));
    std::fs::create_dir_all(&layout).unwrap();
    let public = publish(store, &models, &layout);
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    listener.set_nonblocking(true).unwrap();
    let port = listener.local_addr().unwrap().port();
    let heard: Heard = Default::default();
    let served = Arc::new(Served {
        layout,
        public,
        packages: packages.into_iter().collect(),
        locked: Default::default(),
        origin: format!("http://127.0.0.1:{port}"),
        heard: heard.clone(),
        pace,
    });
    let serving = served.clone();
    std::thread::spawn(move || {
        let runtime = tokio::runtime::Builder::new_current_thread().enable_all().build().unwrap();
        runtime.block_on(async move {
            let router = axum::Router::new().fallback(answer).with_state(serving);
            let listener = tokio::net::TcpListener::from_std(listener).unwrap();
            axum::serve(listener, router).await.unwrap()
        })
    });
    let locked = served.packages.iter().map(|package| lock(package, &served.origin)).collect();
    served.locked.set(locked).unwrap();
    served.heard.lock().unwrap().clear();
    (format!("http://127.0.0.1:{port}"), heard)
}

/// `package`'s pyproject and the uv.lock `uv lock` makes of it, as `cozy package publish` binds
/// one: its callees from this Hub's org index, the rest from PyPI.
fn lock(package: &Package, origin: &str) -> (String, String) {
    let (org, _) = package.name.split_once('/').unwrap();
    let mut parts = package.wheel.split('-');
    let (name, version) = (parts.next().unwrap().replace('_', "-"), parts.next().unwrap());
    let callees: Vec<String> = doors(package).skip(1).map(|(distribution, _, _)| distribution).collect();
    let dependencies: Vec<String> = package.requires.iter().chain(&callees).map(|r| format!("{r:?}")).collect();
    let sources: String = callees.iter().map(|c| format!("{c} = {{ index = \"hub\" }}\n")).collect();
    let pyproject = format!(
        "[project]\nname = \"{name}\"\nversion = \"{version}\"\nrequires-python = \">=3.12\"\ndependencies = [{}]\n\n\
         [[tool.uv.index]]\nname = \"hub\"\nurl = \"{origin}/v1/index/{org}/simple\"\nexplicit = true\n\n[tool.uv.sources]\n{sources}",
        dependencies.join(", ")
    );
    let dir = std::env::temp_dir().join(format!("names-hub-lock-{}", uuid::Uuid::new_v4()));
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("pyproject.toml"), &pyproject).unwrap();
    let locked = std::process::Command::new("uv").args(["lock", "--quiet", "--no-config", "--python", "3.12"]).current_dir(&dir).output().unwrap();
    assert!(locked.status.success(), "uv lock: {}", String::from_utf8_lossy(&locked.stderr));
    let lock = std::fs::read_to_string(dir.join("uv.lock")).unwrap();
    let _ = std::fs::remove_dir_all(dir);
    (pyproject, lock)
}

/// Each wheel the release's lock pins at this Hub, as the Hub renders its row (th-245): its
/// distribution, and its file door `/v1/index/<org>/<package>/<release>/<wheel>` under the
/// release that claims it (the release itself, or a callee's own). No other door answers.
fn doors(package: &Package) -> impl Iterator<Item = (String, String, &[u8])> {
    let (org, name) = package.name.split_once('/').unwrap();
    let root = (name.to_string(), package.release.clone(), &package.wheel, &package.bytes);
    let callees = package.callees.iter().map(|(wheel, bytes)| {
        let mut parts = wheel.split('-');
        let (distribution, version) = (parts.next().unwrap().replace('_', "-"), parts.next().unwrap().to_string());
        (distribution, version, wheel, bytes)
    });
    std::iter::once(root).chain(callees).map(move |(distribution, release, wheel, bytes)| {
        let door = format!("/v1/index/{org}/{distribution}/{release}/{wheel}");
        (distribution, door, bytes.as_slice())
    })
}

fn absent() -> Response {
    (StatusCode::NOT_FOUND, r#"{"error":{"code":"not_found","message":"absent"}}"#).into_response()
}

fn json(value: Value) -> Response {
    (StatusCode::OK, serde_json::to_vec(&value).unwrap()).into_response()
}

/// `bytes`, or the range `headers` ask of them.
fn ranged(bytes: &[u8], headers: &HeaderMap) -> Response {
    let asked = headers.get("range").and_then(|v| v.to_str().ok()).and_then(|v| {
        let (first, last) = v.trim().strip_prefix("bytes=")?.split_once('-')?;
        let first: usize = first.parse().ok()?;
        let last = last.parse::<usize>().unwrap_or(bytes.len() - 1).min(bytes.len() - 1);
        Some((first, last))
    });
    match asked {
        Some((first, last)) => (
            StatusCode::PARTIAL_CONTENT,
            [("content-range", format!("bytes {first}-{last}/{}", bytes.len()))],
            bytes[first..=last].to_vec(),
        )
            .into_response(),
        None => (StatusCode::OK, bytes.to_vec()).into_response(),
    }
}

async fn answer(State(served): State<Arc<Served>>, method: Method, uri: Uri, headers: HeaderMap, body: Bytes) -> Response {
    let path = uri.path().to_string();
    let presented = headers.get("authorization").and_then(|v| v.to_str().ok());
    let admitted = ADMITTED.lock().unwrap().get(&served.origin).cloned().unwrap_or_else(|| vec![OWNER.into()]);
    let owner = presented.map(|value| admitted.iter().any(|bearer| value == format!("Bearer {bearer}")));
    let heard = match owner {
        None => format!("{method} {path}"),
        Some(true) => format!("{method} {path} +credential"),
        Some(false) => format!("{method} {path} +refused"),
    };
    served.heard.lock().unwrap().push(heard);
    let owner = match owner {
        Some(false) => {
            let body = r#"{"error":{"code":"unauthorized","message":"the bearer is not valid"}}"#;
            return (StatusCode::UNAUTHORIZED, [("www-authenticate", "Bearer")], body).into_response();
        }
        owner => owner.is_some(),
    };
    let first = served.packages.first();
    let release_path = |p: &Package| format!("/v1/packages/{}/releases/{}", p.name, p.release);
    let release = served.packages.iter().find(|p| path == release_path(p));
    let query = uri.query().unwrap_or_default();
    let planned = served.packages.iter().position(|p| {
        path == format!("/v1/packages/{}/download", p.name) && query.split('&').any(|q| q == format!("release={}", p.release))
    });
    let door = served.packages.iter().flat_map(doors).find(|(_, door, _)| *door == path);
    // The org index's page for one distribution: each file door that serves it.
    let page = path.strip_prefix("/v1/index/").and_then(|rest| rest.split_once("/simple/")).map(|(org, rest)| {
        let distribution = rest.trim_end_matches('/');
        let links: String = served.packages.iter().filter(|p| p.name.starts_with(&format!("{org}/"))).flat_map(doors)
            .filter(|(name, _, _)| name == distribution)
            .map(|(_, door, bytes)| format!("<a href=\"{}{door}#sha256={}\">{}</a>\n", served.origin,
                tensorfs::sha256::hex_digest(bytes), door.rsplit('/').next().unwrap()))
            .collect();
        format!("<!DOCTYPE html><html><body>\n{links}</body></html>\n")
    });
    match (method, path.as_str()) {
        (Method::GET, p) if first.is_some_and(|package| p == format!("/v1/packages/{}", package.name)) => {
            let mut releases: Vec<Value> = served.packages.iter().map(|p| json!({"release": p.release})).collect();
            releases.extend([json!({"release": "0.9.0"}), json!({"release": "9.0.0", "yanked": true})]);
            let (org, name) = first.unwrap().name.split_once('/').unwrap();
            json(json!({"package": {"org": org, "name": name, "created_at": "2026-01-01T00:00:00Z"}, "releases": releases}))
        }
        (Method::GET, _) if release.is_some() => {
            let package = release.unwrap();
            let summary = json!({"release": package.release, "package_interface_digest": "sha256:00",
                "package_interface_length": 0, "created_at": "2026-01-01T00:00:00Z"});
            json(json!({"release": summary, "package_interface": package.interface, "requirements": [],
                "requires_python": ">=3.12", "python_version": "3.12"}))
        }
        // The release's install plan: its exact documents, and its project wheel at a presigned
        // URL on the object host.
        (Method::POST, _) if planned.is_some() => {
            let (package, (pyproject, lock)) = (&served.packages[planned.unwrap()], &served.locked.get().unwrap()[planned.unwrap()]);
            let exact = |bytes: &[u8]| json!({"canonical_bytes": base64::engine::general_purpose::STANDARD.encode(bytes),
                "digest": format!("sha256:{}", tensorfs::sha256::hex_digest(bytes)), "length": bytes.len()});
            let (distribution, door, bytes) = doors(package).next().unwrap();
            let wheel = json!({"digest": format!("sha256:{}", tensorfs::sha256::hex_digest(bytes)),
                "distribution": distribution, "import_roots": [], "kind": "project_wheel", "length": bytes.len(),
                "path": package.wheel, "tags": ["py3-none-any"], "url": format!("{}{door}", served.origin.replace("127.0.0.1", "localhost")),
                "version": package.release});
            json(json!({"python_version": "3.12", "downloads": [wheel], "package_config": exact(b"[package]\n"),
                "package_interface": exact(&serde_json::to_vec(&package.interface).unwrap()), "pyproject": exact(pyproject.as_bytes()),
                "release": package.release, "uv_lock": exact(lock.as_bytes())}))
        }
        (Method::GET, _) if page.is_some() => {
            ([("content-type", "text/html")], page.unwrap()).into_response()
        }
        (Method::GET, _) if door.is_some() => ranged(door.unwrap().2, &headers),
        // The owner's upload: an object once, then its repo document while it is as last read.
        (Method::PUT, p) if owner && (p.starts_with("/blobs/") || p.starts_with("/refs/")) => {
            let file = served.layout.join(&p[1..]);
            if p.starts_with("/refs/") {
                let now = std::fs::read(&file).ok().map(|bytes| Repo::from_json(&bytes).unwrap().to_json().unwrap());
                let etag = now.map(|bytes| format!("\"sha256:{}\"", tensorfs::sha256::hex_digest(&bytes)));
                let asked = |name: &str| headers.get(name).and_then(|v| v.to_str().ok());
                if asked("if-match").is_some_and(|tag| Some(tag) != etag.as_deref()) || (asked("if-none-match").is_some() && etag.is_some()) {
                    return StatusCode::PRECONDITION_FAILED.into_response();
                }
                Repo::from_json(&body).unwrap();
            }
            std::fs::create_dir_all(file.parent().unwrap()).unwrap();
            std::fs::write(&file, &body).unwrap();
            StatusCode::NO_CONTENT.into_response()
        }
        (Method::GET, p) if p.starts_with("/refs/") => {
            let Ok(bytes) = std::fs::read(served.layout.join(&p[1..])) else {
                return absent();
            };
            // Private tags are the owner's.
            let mut repo = Repo::from_json(&bytes).unwrap();
            if !owner {
                repo.tags.retain(|t| t.public);
                if repo.tags.is_empty() {
                    return absent();
                }
            }
            let bytes = repo.to_json().unwrap();
            let etag = format!("\"sha256:{}\"", tensorfs::sha256::hex_digest(&bytes));
            match headers.get("if-none-match").and_then(|v| v.to_str().ok()) == Some(etag.as_str()) {
                true => StatusCode::NOT_MODIFIED.into_response(),
                false => ([("etag", etag)], bytes).into_response(),
            }
        }
        (verb @ (Method::GET | Method::HEAD), p) if p.starts_with("/blobs/") => {
            let sha = p.rsplit('/').next().unwrap_or_default();
            let file = served.layout.join(&p[1..]);
            if !(owner || served.public.contains(sha)) || !file.is_file() {
                return absent();
            }
            // A HEAD answers its length; the server drops the body.
            if verb == Method::HEAD {
                return std::fs::read(&file).unwrap().into_response();
            }
            // Storage is another origin than the Hub's: no credential follows the redirect.
            let storage = served.origin.replace("127.0.0.1", "localhost");
            let location = format!("{storage}/o/{sha}");
            (StatusCode::TEMPORARY_REDIRECT, [("location", location), ("cache-control", "max-age=300".into())]).into_response()
        }
        (Method::GET, p) if p.starts_with("/o/") => {
            let sha = &p[3..];
            let key = ObjectRef { sha256: sha.into(), length: 0 }.key();
            match std::fs::read(served.layout.join(key)) {
                Ok(bytes) => {
                    tokio::time::sleep(served.pace).await;
                    ranged(&bytes, &headers)
                }
                Err(_) => absent(),
            }
        }
        _ => absent(),
    }
}
