//! One-off private proof upload through the existing public session and ordinary owner login.
use hub_client::{Credential, DeviceKey, UserSession};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::{
    collections::BTreeMap,
    fs,
    future::Future,
    path::{Path, PathBuf},
    pin::Pin,
    sync::Arc,
};
use tensorfs::{format::ObjectRef, Client, Config, Endpoint, Open, Ref};

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
const NATIVE: &str = "f7ccd8310e358d39527f591ec144a8bc2ba5c62c";
#[derive(Deserialize)]
struct Plan {
    format: String,
    origin: String,
    repos: Vec<Repo>,
}
#[derive(Deserialize)]
struct Repo {
    source: String,
    source_release: String,
    destination: String,
    tree: ObjectRef,
    folder: PathBuf,
    variants: BTreeMap<String, ObjectRef>,
    files: BTreeMap<String, ObjectRef>,
    objects: BTreeMap<String, ObjectRef>,
}
struct Owner {
    origin: String,
    user: Arc<UserSession>,
    hub: hub_client::Client,
}
// Match async_trait's public ABI without changing the pinned CLI's dependency graph.
impl tensorfs::TokenProvider for Owner {
    fn headers<'a, 'b, 'c, 'd>(
        &'a self,
        request: &'b tensorfs::Request<'c>,
    ) -> Pin<Box<dyn Future<Output = tensorfs::failure::Result<Vec<(String, String)>>> + Send + 'd>>
    where
        'a: 'd,
        'b: 'd,
        'c: 'd,
        Self: 'd,
    {
        Box::pin(async move {
            let url = url::Url::parse(request.url).map_err(|_| denied())?;
            if url.origin().ascii_serialization() != self.origin {
                return Ok(Vec::new());
            }
            let token = self.user.bearer(&self.hub).await.map_err(|_| denied())?;
            Ok(vec![("authorization".into(), format!("Bearer {token}"))])
        })
    }
}
fn denied() -> tensorfs::failure::Failure {
    tensorfs::failure::Failure {
        kind: tensorfs::failure::Kind::Denied,
        detail: "ordinary owner login failed".into(),
    }
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
struct View {
    status: u16,
    etag: Option<String>,
    digest: ObjectRef,
}
async fn view(
    http: &reqwest::Client,
    owner: Option<&Owner>,
    origin: &str,
    name: &str,
) -> Result<View> {
    let mut request = http.get(format!("{origin}/refs/{name}.json"));
    if let Some(owner) = owner {
        request = request.bearer_auth(
            owner
                .user
                .bearer(&owner.hub)
                .await
                .map_err(|_| "owner login failed")?,
        );
    }
    let response = request.send().await.map_err(|_| "refs transport failed")?;
    let status = response.status().as_u16();
    let etag = response
        .headers()
        .get("etag")
        .and_then(|h| h.to_str().ok())
        .map(str::to_owned);
    if response.content_length().is_some_and(|n| n > 16 << 20) {
        return Err("refs response exceeds cap".into());
    }
    let body = response.bytes().await.map_err(|_| "refs body failed")?;
    if body.len() > 16 << 20 {
        return Err("refs response exceeds cap".into());
    }
    Ok(View {
        status,
        etag,
        digest: ObjectRef::of(&body),
    })
}
fn save(path: &Path, value: &Value) -> Result<()> {
    let temp = path.with_extension("json.tmp");
    fs::write(&temp, serde_json::to_vec_pretty(value)?)?;
    fs::rename(temp, path)?;
    Ok(())
}
fn file(path: &Path, at: &ObjectRef) -> Result<()> {
    let meta = fs::symlink_metadata(path)?;
    if !meta.is_file() || meta.len() != at.length || meta.len() > 32 << 20 {
        return Err("metadata type/length/cap mismatch".into());
    }
    if ObjectRef::of(&fs::read(path)?) != *at {
        return Err("metadata digest mismatch".into());
    }
    Ok(())
}
fn validate(repo: &Repo, cache: &Path) -> Result<Value> {
    if !repo.source.starts_with("fidika/")
        || repo.destination
            != format!(
                "fidika/varena-slim-20261020-{}",
                repo.source
                    .strip_prefix("fidika/")
                    .ok_or("source namespace")?
            )
    {
        return Err("only a new named private proof destination is allowed".into());
    }
    let mut pending = vec![repo.folder.clone()];
    let mut paths = BTreeMap::new();
    while let Some(dir) = pending.pop() {
        for entry in fs::read_dir(dir)? {
            let entry = entry?;
            let kind = entry.file_type()?;
            if kind.is_dir() {
                pending.push(entry.path());
                continue;
            }
            if !kind.is_file() {
                return Err("metadata folder contains a link or special file".into());
            }
            let full = entry.path();
            let name = full
                .strip_prefix(&repo.folder)?
                .to_str()
                .ok_or("non-UTF8 path")?
                .to_string();
            let expected = if let Some(sha) = name.strip_prefix("chunks/") {
                repo.objects
                    .get(sha)
                    .ok_or("chunk is not approved metadata/inline adjunct")?
            } else {
                repo.files.get(&name).ok_or("unexpected repo file")?
            };
            file(&full, expected)?;
            if !name.starts_with("chunks/") {
                paths.insert(name, expected.clone());
            }
        }
    }
    if paths != repo.files {
        return Err("repo files differ from staged tree".into());
    }
    let store = tensorfs::store::Store::open(cache).map_err(|_| "metadata store failed")?;
    let (tree, absent) = store
        .admit_folder(&repo.folder)
        .map_err(|_| "thin folder admission failed")?;
    if tree != repo.tree {
        return Err("admission changed the staged tree".into());
    }
    Ok(
        json!({"destination": repo.destination, "tree": tree, "variants": repo.variants,
        "remote_objects_not_downloaded": absent.len()}),
    )
}
async fn run() -> Result<()> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if args.len() < 3 || !matches!(args[0].as_str(), "inspect" | "publish" | "verify") {
        return Err(
            "usage: d10_private_proof inspect PLAN OUTPUT | publish|verify PLAN OUTPUT DEVICE_KEY"
                .into(),
        );
    }
    if option_env!("D10_NATIVE_SOURCE") != Some(NATIVE) {
        return Err("unapproved native source".into());
    }
    let plan_raw = fs::read(&args[1])?;
    let plan: Plan = serde_json::from_slice(&plan_raw)?;
    if plan.format != "cozy.private-d10-proof/1" || plan.origin != "https://tensorhub.com" {
        return Err("unapproved publication plan".into());
    }
    let output = PathBuf::from(&args[2]);
    fs::create_dir_all(&output)?;
    let state_path = output.join("receipt.json");
    let mut state = if state_path.exists() {
        serde_json::from_slice::<Value>(&fs::read(&state_path)?)?
    } else {
        json!({"plan": ObjectRef::of(&plan_raw), "native_source": NATIVE, "repos": {},
            "tensor_chunk_downloads": 0, "public_ref_writes": 0, "credential": "ordinary owner UserSession; memory only"})
    };
    if state["plan"] != json!(ObjectRef::of(&plan_raw)) {
        return Err("output belongs to another plan".into());
    }
    let rows: Vec<_> = plan
        .repos
        .iter()
        .map(|repo| validate(repo, &output.join("admitted")))
        .collect::<Result<_>>()?;
    state["local_validation"] = json!(rows);
    save(&state_path, &state)?;
    if args[0] == "inspect" {
        println!(
            "{}",
            json!({"validated_repos": rows.len(), "tensor_chunk_reads": 0})
        );
        return Ok(());
    }
    let key = DeviceKey::load(Path::new(
        args.get(3).ok_or("ordinary owner device key is required")?,
    ))
    .map_err(|_| "owner key could not be loaded")?;
    if key.hub.trim_end_matches('/') != plan.origin {
        return Err("owner key belongs to another Hub".into());
    }
    let user = Arc::new(UserSession::new(key, None));
    let http = reqwest::Client::builder()
        .redirect(reqwest::redirect::Policy::none())
        .timeout(std::time::Duration::from_secs(90))
        .build()?;
    // Login calls must not re-enter this session's hook while bearer holds its mutex.
    let hub = hub_client::Client::connect(&plan.origin, http.clone(), Credential::Anonymous);
    let owner = Arc::new(Owner {
        origin: plan.origin.clone(),
        user,
        hub,
    });
    let endpoint =
        Endpoint::new(&plan.origin, Some(owner.clone())).map_err(|_| "TensorFS endpoint failed")?;
    if state.get("public_before").is_none() {
        let mut before = BTreeMap::new();
        for repo in &plan.repos {
            let anonymous = view(&http, None, &plan.origin, &repo.source).await?;
            let signed_in = view(&http, Some(&owner), &plan.origin, &repo.source).await?;
            if !matches!(anonymous.status, 200 | 404) || !matches!(signed_in.status, 200 | 404) {
                return Err("public source preflight failed".into());
            }
            before.insert(
                repo.source.clone(),
                json!({"anonymous": anonymous, "owner": signed_in}),
            );
        }
        // Reject any pre-existing destination before the first write. On restart only
        // destinations already recorded as started by this exact plan may exist.
        for repo in &plan.repos {
            let current = view(&http, Some(&owner), &plan.origin, &repo.destination).await?;
            if current.status != 404 {
                return Err("proof destination already exists".into());
            }
        }
        state["public_before"] = json!(before);
        save(&state_path, &state)?;
    }
    if args[0] == "publish" {
        for repo in &plan.repos {
            eprintln!("private proof: {}", repo.destination);
            let current = endpoint
                .repo(&repo.destination)
                .await
                .map_err(|_| "proof destination read failed")?;
            if let Some((doc, _)) = current {
                if state["repos"][&repo.destination]["started"] != true
                    || !doc.releases.is_empty()
                    || doc.checkpoints != vec![repo.tree.clone()]
                {
                    return Err("existing destination is not this plan's private checkpoint".into());
                }
            } else {
                state["repos"][&repo.destination] = json!({"started": true, "source": repo.source, "source_release": repo.source_release});
                save(&state_path, &state)?;
                let mut config = Config::from_env();
                config.endpoint = plan.origin.clone();
                config.token = Some(owner.clone());
                config.cache_dir = output.join("admitted");
                config.mirror = None;
                let client = Client::new(config).map_err(|_| "upload metadata store failed")?;
                let reference = Ref {
                    repo_id: repo.destination.clone(),
                    revision: None,
                    variant: None,
                    endpoint: None,
                };
                let uploaded = client
                    .upload_folder(&reference, &repo.folder, None)
                    .await
                    .map_err(|_| "thin metadata upload failed")?;
                if uploaded != repo.tree {
                    return Err("upload tree identity changed".into());
                }
            }
            state["repos"][&repo.destination]["uploaded"] = json!(true);
            save(&state_path, &state)?;
        }
    }
    // A fresh cache proves digest opens fetched metadata from the endpoint, never the
    // locally admitted folder; Open's lazy=true prevents all weight downloads.
    let mut config = Config::from_env();
    config.endpoint = plan.origin.clone();
    config.token = Some(owner.clone());
    config.cache_dir = output.join(format!("remote-verify-{}", std::process::id()));
    config.mirror = None;
    let client = Client::new(config).map_err(|_| "verification metadata store failed")?;
    for repo in &plan.repos {
        let (doc, _) = endpoint
            .repo(&repo.destination)
            .await
            .map_err(|_| "owner proof read failed")?
            .ok_or("private proof missing")?;
        if !doc.releases.is_empty() || doc.checkpoints != vec![repo.tree.clone()] {
            return Err("proof is not the exact unreleased tree".into());
        }
        let anonymous = view(&http, None, &plan.origin, &repo.destination).await?;
        if anonymous.status != 404 {
            return Err("private proof is anonymously visible".into());
        }
        let reference = format!("{}@sha256:{}", repo.destination, repo.tree.sha256);
        for (variant, header) in &repo.variants {
            let checkpoint = client
                .open(
                    &reference,
                    &Open {
                        variant: Some(variant.clone()),
                        lazy: true,
                        ..Default::default()
                    },
                )
                .await
                .map_err(|_| "remote digest/variant open failed")?;
            if checkpoint.id() != header {
                return Err("remote header differs from staged header".into());
            }
        }
        state["repos"][&repo.destination]["verified"] = json!({"ref": reference, "variants": repo.variants, "anonymous_status": anonymous.status, "released": false});
        save(&state_path, &state)?;
    }
    let mut after = BTreeMap::new();
    for repo in &plan.repos {
        let anonymous = view(&http, None, &plan.origin, &repo.source).await?;
        let signed_in = view(&http, Some(&owner), &plan.origin, &repo.source).await?;
        let next = json!({"anonymous": anonymous, "owner": signed_in});
        if state["public_before"][&repo.source] != next {
            return Err("public source bytes or ETag changed during proof publication".into());
        }
        after.insert(repo.source.clone(), next);
    }
    state["public_after"] = json!(after);
    state["complete"] = json!(true);
    save(&state_path, &state)?;
    println!(
        "{}",
        json!({"private_unreleased_repos": plan.repos.len(), "public_refs_unchanged": true, "tensor_chunk_downloads": 0, "receipt": state_path})
    );
    Ok(())
}
#[tokio::main(flavor = "multi_thread", worker_threads = 2)]
async fn main() {
    if let Err(error) = run().await {
        // All propagated remote/auth failures are mapped to static, credential-free text.
        eprintln!("private proof failed: {error}");
        std::process::exit(1);
    }
}
