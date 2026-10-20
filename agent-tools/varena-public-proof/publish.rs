//! One-off new public test releases through ordinary owner credentials and Endpoint CAS.
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
use tensorfs::{format::{ObjectRef, Repo as RepoDoc}, Client, Config, Open};
use tensorfs::fetch::{Endpoint, Payload, SourcePolicy};

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
    private_repo: String,
    private_tree: ObjectRef,
    proof: ObjectRef,
    tree_path: PathBuf,
    probe_variant: String,
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
impl tensorfs::fetch::TokenProvider for Owner {
    fn headers<'a, 'b, 'c, 'd>(
        &'a self,
        request: &'b tensorfs::fetch::token::Request<'c>,
    ) -> Pin<Box<dyn Future<Output = tensorfs::err::Result<Vec<(String, String)>>> + Send + 'd>>
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
fn denied() -> tensorfs::err::Refusal {
    tensorfs::err::Refusal { code: tensorfs::err::Code::CREDENTIAL_REQUIRED,
        detail: "ordinary owner login failed".into() }
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
                "fidika/varena-slim-public-20261020-{}",
                repo.source
                    .strip_prefix("fidika/")
                    .ok_or("source namespace")?
            )
    {
        return Err("only a new named public test destination is allowed".into());
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
async fn private_tree_status(http: &reqwest::Client, origin: &str, tree: &ObjectRef) -> Result<u16> {
    let sha = &tree.sha256;
    let url = format!("{origin}/blobs/{}/{}/{sha}", &sha[..2], &sha[2..4]);
    Ok(http.head(url).send().await.map_err(|_| "private tree probe failed")?.status().as_u16())
}
async fn snapshot(http: &reqwest::Client, owner: &Owner, plan: &Plan) -> Result<Value> {
    let mut views = BTreeMap::new();
    for repo in &plan.repos {
        let public = view(http, None, &plan.origin, &repo.source).await?;
        let private_owner = view(http, Some(owner), &plan.origin, &repo.private_repo).await?;
        let private_anon = view(http, None, &plan.origin, &repo.private_repo).await?;
        let private_tree = private_tree_status(http, &plan.origin, &repo.private_tree).await?;
        if public.status != 200 || private_owner.status != 200 || private_anon.status != 404 || private_tree != 404 {
            return Err("source public/private boundaries differ from the approved fixture".into());
        }
        views.insert(repo.source.clone(), json!({"public": public, "private_owner": private_owner,
            "private_anonymous": private_anon, "private_tree_status": private_tree}));
    }
    Ok(json!(views))
}
async fn chunk_probe(ck: &tensorfs::Checkpoint, origin: &str) -> Result<Value> {
    // Availability only: HEAD and at most64 stored bytes, never a complete model tensor.
    let mut selected = None;
    for component in ck.components() {
        for tensor in component.tensors().iter().take(16) {
            for (role, location) in tensor.locations().into_iter().take(1) {
                let Some(chunk) = ck.chunks().get(location.chunk as usize) else { continue };
                let length = location.length.min(64).min(chunk.length.saturating_sub(location.offset));
                if length > 0 { selected = Some((component.name().to_string(),tensor.key.clone(),role.to_string(),location.offset,length,chunk.clone())); break; }
            }
            if selected.is_some() { break; }
        }
        if selected.is_some() { break; }
    }
    let (component,key,role,offset,length,chunk) = selected.ok_or("no bounded tensor chunk range")?;
    let sha=&chunk.sha256;
    let url=format!("{origin}/blobs/{}/{}/{sha}",&sha[..2],&sha[2..4]);
    let http=reqwest::Client::builder().redirect(reqwest::redirect::Policy::limited(5))
        .timeout(std::time::Duration::from_secs(90)).build()?;
    let head=http.head(&url).send().await.map_err(|_| "anonymous chunk HEAD failed")?;
    if head.status().as_u16()!=200 || head.content_length()!=Some(chunk.length) { return Err("anonymous chunk HEAD identity/length mismatch".into()); }
    let end=offset+length-1;
    let response=http.get(url).header(reqwest::header::RANGE,format!("bytes={offset}-{end}"))
        .send().await.map_err(|_| "anonymous tiny chunk range failed")?;
    if response.status().as_u16()!=206 || response.content_length()!=Some(length) {
        return Err("chunk endpoint did not honor the tiny bounded range; body not read".into());
    }
    let wanted=format!("bytes {offset}-{end}/{}",chunk.length);
    if response.headers().get(reqwest::header::CONTENT_RANGE).and_then(|h| h.to_str().ok())!=Some(wanted.as_str()) {
        return Err("chunk range has different bounds; body not read".into());
    }
    let bytes=response.bytes().await.map_err(|_| "tiny chunk range body failed")?;
    if bytes.len()!=length as usize { return Err("tiny chunk range length mismatch".into()); }
    Ok(json!({"component":component,"key":key,"role":role,"chunk":chunk,"offset":offset,
        "range_bytes":length,"sample":ObjectRef::of(&bytes),"head_status":200,"range_status":206,
        "full_chunk_downloaded":false,"logical_tensor_decoded":false}))
}
async fn run() -> Result<()> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if args.len() < 3 || !matches!(args[0].as_str(), "inspect" | "publish" | "verify") {
        return Err("usage: d10_public_proof inspect PLAN OUTPUT | publish|verify PLAN OUTPUT DEVICE_KEY".into());
    }
    if option_env!("D10_NATIVE_SOURCE") != Some(NATIVE) { return Err("unapproved native source".into()); }
    let plan_raw = fs::read(&args[1])?;
    let plan: Plan = serde_json::from_slice(&plan_raw)?;
    if plan.format != "cozy.public-d10-proof/1" || plan.origin != "https://tensorhub.com" || plan.repos.len() != 9 {
        return Err("unapproved public test plan".into());
    }
    let output = PathBuf::from(&args[2]); fs::create_dir_all(&output)?;
    let state_path = output.join("receipt.json");
    let mut state: Value = if state_path.exists() { serde_json::from_slice(&fs::read(&state_path)?)? }
        else { json!({"plan":ObjectRef::of(&plan_raw),"native_source":NATIVE,"repos":{},"public_ref_writes":0}) };
    if state["plan"] != json!(ObjectRef::of(&plan_raw)) { return Err("receipt belongs to another plan".into()); }
    let rows = plan.repos.iter().map(|r| validate(r,&output.join("admitted"))).collect::<Result<Vec<_>>>()?;
    state["local_validation"] = json!(rows); save(&state_path,&state)?;
    if args[0] == "inspect" { println!("{}",json!({"validated_repos":rows.len(),"tensor_reads":0})); return Ok(()); }
    let key = DeviceKey::load(Path::new(args.get(3).ok_or("ordinary owner key required")?)).map_err(|_| "owner key unavailable")?;
    if key.hub.trim_end_matches('/') != plan.origin { return Err("key belongs to another Hub".into()); }
    let http = reqwest::Client::builder().redirect(reqwest::redirect::Policy::none()).timeout(std::time::Duration::from_secs(90)).build()?;
    let owner = Arc::new(Owner { origin:plan.origin.clone(), user:Arc::new(UserSession::new(key,None)),
        hub:hub_client::Client::connect(&plan.origin,http.clone(),Credential::Anonymous) });
    let transport = tensorfs::fetch::Client::new(SourcePolicy { any_host:true, max_redirects:5, ..Default::default() });
    let endpoint = Endpoint::new(&transport,&plan.origin,Some(owner.clone())).map_err(|_| "owner endpoint failed")?;
    let before = snapshot(&http,&owner,&plan).await?;
    if state.get("before").is_none() { state["before"]=before; save(&state_path,&state)?; }
    else if state["before"] != before { return Err("original source/private boundaries changed since start".into()); }
    if args[0] == "publish" {
        // Every write is an If-None-Match create of a NEW test repo. No existing ref is replaced.
        for repo in &plan.repos {
            let mut doc = RepoDoc::new(&repo.destination); doc.set_release("0.1.0",repo.tree.clone());
            let encoded = doc.to_json().map_err(|_| "public test document failed")?;
            let current = endpoint.repo(&repo.destination).await.map_err(|_| "test destination read failed")?;
            if let Some((current,_)) = current {
                if state["repos"][&repo.destination]["started"] != true || current != doc {
                    return Err("destination already exists with another document".into());
                }
            } else {
                state["repos"][&repo.destination]=json!({"started":true}); save(&state_path,&state)?;
                for (at,path) in [(&repo.proof,repo.folder.join("VARENA_PROOF.md")),(&repo.tree,repo.tree_path.clone())] {
                    file(&path,at)?;
                    let payload = || fs::File::open(&path).map(Payload::File).map_err(|_| tensorfs::err::Refusal {
                        code:tensorfs::err::Code::IO_FAILED,detail:"test metadata unavailable".into() });
                    endpoint.upload(at,&payload).await.map_err(|_| "test metadata upload failed")?;
                }
                if !endpoint.put_repo_bytes(&repo.destination,encoded,None).await.map_err(|_| "new public test ref failed")? {
                    return Err("new public test ref lost its create precondition".into());
                }
                let count=state["public_ref_writes"].as_u64().unwrap_or(0); state["public_ref_writes"]=json!(count+1);
            }
            state["repos"][&repo.destination]["released"]=json!(true); save(&state_path,&state)?;
            eprintln!("public test release: {}@0.1.0",repo.destination);
        }
    }
    let cold = output.join(format!("anonymous-cold-{}",std::process::id()));
    if cold.exists() { return Err("anonymous verification cache is not fresh".into()); }
    let mut opened=0; let mut downloaded=0u64;
    for repo in &plan.repos {
        let anonymous=view(&http,None,&plan.origin,&repo.destination).await?;
        if anonymous.status!=200 { return Err("named public test release is not anonymously readable".into()); }
        let mut config=Config::from_env(); config.endpoint=plan.origin.clone(); config.token=None; config.mirror=None;
        config.cache_dir=cold.join(repo.destination.replace('/',"__"));
        let client=Client::new(config).map_err(|_| "fresh anonymous native client failed")?;
        let reference=format!("{}@0.1.0",repo.destination); let mut probe=None;
        for (variant,header) in &repo.variants {
            let ck=client.open(&reference,&Open { variant:Some(variant.clone()),lazy:true,..Default::default() }).await
                .map_err(|_| "fresh anonymous named release/variant open failed")?;
            if ck.id()!=header { return Err("public release header differs from source fixture".into()); }
            opened+=1;
            if variant==&repo.probe_variant { let check=chunk_probe(&ck,&plan.origin).await?; downloaded+=check["range_bytes"].as_u64().unwrap(); probe=Some(check); }
        }
        let probe=probe.ok_or("required variant lacks cold chunk proof")?;
        state["repos"][&repo.destination]["verified"]=json!({"ref":reference,"variants":repo.variants,"anonymous":anonymous,"chunk_probe":probe});
        save(&state_path,&state)?;
    }
    let after=snapshot(&http,&owner,&plan).await?;
    if state["before"]!=after { return Err("original public/private refs or private tree denial changed".into()); }
    state["after"]=after; state["complete"]=json!(true); state["anonymous_variant_opens"]=json!(opened);
    state["anonymous_chunk_range_bytes"]=json!(downloaded); save(&state_path,&state)?;
    println!("{}",json!({"public_test_repos":plan.repos.len(),"named_release":"0.1.0","anonymous_variant_opens":opened,
        "anonymous_chunk_range_bytes":downloaded,"original_and_private_refs_unchanged":true,"private_trees_hidden":true,"receipt":state_path}));
    Ok(())
}
#[tokio::main(flavor="multi_thread",worker_threads=2)]
async fn main() { if let Err(error)=run().await { eprintln!("public proof failed: {error}"); std::process::exit(1); } }
