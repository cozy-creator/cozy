//! Code-less uploads against the real development Hub (PostgreSQL + MinIO), using the same
//! capability and TensorFS provider as a request. The fixture owns no mock auth or upload server.
use base64::{engine::general_purpose::STANDARD, Engine as _};
use hub_client::{testing, types, Client, Credential, P256Key, RunCapability};
use std::{path::Path, time::Duration};
use tensord::{hub::{Capability, Catalog, Op, Source}, origin::Publication};
use tensorfs_core::{format::{Config, Dtype, Tree}, store2::Store, writer::{Data, Source as Input}};

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_code_less_upload_resumes_objects_and_commits_only_its_ref() {
    let env = |name| std::env::var(name).ok().filter(|v: &String| !v.is_empty());
    let (Some(origin), Some(reach), Some(ca), Some(log)) =
        (env("TENSORHUB_ORIGIN"), env("TENSORHUB_URL"), env("TENSORHUB_CA"), env("TENSORHUB_LOG")) else {
        eprintln!("skipped: needs the real Tensorhub development harness");
        return;
    };
    let nonce = uuid::Uuid::new_v4().simple().to_string();
    let anonymous = Client::connect(&reach, reqwest::Client::new(), Credential::Anonymous);
    let (device, bearer) = testing::sign_up(&anonymous, &format!("upload-{nonce}@example.test"), Path::new(&log)).await.unwrap();
    let user = Client::connect(&reach, reqwest::Client::new(), Credential::Bearer(bearer.token));
    let account = format!("up{}", &nonce[..12]);
    user.register_account(&account, &types::Empty(Default::default())).await.unwrap();
    let owner = user.get_current_account().await.unwrap().into_inner().user_id;
    let token_endpoint = user.token_endpoint().await.unwrap();
    let model = format!("{account}/model");
    let key = P256Key::generate();
    let signed = device.sign_run_capability(&RunCapability {
        user_id: owner, hub: origin, key_jkt: key.thumbprint(), run: format!("upload-{nonce}"),
        operations: vec![serde_json::json!({"type": "tensorhub_model_upload", "model": model})], lifetime: Duration::from_secs(600),
    });
    let pem = std::fs::read_to_string(ca).unwrap();
    let body = pem.split("-----BEGIN CERTIFICATE-----").nth(1).unwrap().split("-----END CERTIFICATE-----").next().unwrap();
    let ca = STANDARD.decode(body.split_whitespace().collect::<String>()).unwrap();
    let capability = Capability::new(&signed, &token_endpoint, &key.private_jwk()).unwrap();
    let source = Source::new(&reach, Some(ca), vec![], Some(capability)).unwrap();
    let catalog = Catalog::for_op(&source, Op::Publish { model: &model }).unwrap();

    let root = std::env::temp_dir().join(format!("cozy-code-less-{nonce}"));
    let store = Store::open(&root).unwrap();
    let mut writer = store.writer(None, None).unwrap();
    let bytes: Vec<u8> = (0..16_384u32).map(|i| (i.wrapping_mul(97) >> 8) as u8).collect();
    writer.component("model", "conformance.Fixture", Config::default(), None).unwrap()
        .put("w", &Dtype::F32, &[4096], Data::Plain(Input::Read(&mut bytes.as_slice()))).unwrap();
    let header = writer.commit().unwrap();
    let mut tree = Tree::default();
    tree.set_variant("f32", header);
    let tree = store.put_tree(&tree).unwrap();
    let progress = |done, total| assert!(done <= total);
    let stopped = Publication { store: &store, destination: &model, tree: &tree, operation: "first",
        progress: &progress, streams: 2, stop: None, commit: &|| false };
    assert!(catalog.publish(&stopped).unwrap_err().detail.contains("before its ref PUT"));
    // The object closure is uploaded, but the commit was refused: no checkpoint is visible.
    let committed = Publication { operation: "resume", commit: &|| true, ..stopped };
    let uploaded = catalog.publish(&committed).unwrap();
    assert_eq!(uploaded.checkpoint, tree.id());
    assert!(!uploaded.converged, "resuming orphan objects still commits a new checkpoint");
    assert!(catalog.publish(&committed).unwrap().converged);
    assert_eq!(Catalog::for_op(&source, Op::Publish { model: &format!("{account}/other") }).err().unwrap().0,
        "capability_required");
    std::fs::remove_dir_all(root).unwrap();
}
