//! `cozy model publish|yank|delete` against a real Tensorhub (Tensorhub's dev-build `cozy_ref`
//! job: TENSORHUB_ORIGIN behind the private CA TENSORHUB_CA). Each verb edits the model's repo
//! document at the Hub's distribution endpoint as the signed-in owner. Two commits the owner
//! uploads with TensorFS (`draft`, then `main`, which adds an fp8 variant) are released by tag,
//! by commit id and by a variant header; a release is yanked and un-yanked; a tag and then the
//! model are deleted.
mod common;

use common::Home;
use serde_json::Value;
use tensorfs::{
    fetch::{bearer, upload_tree, Client, Endpoint, SourcePolicy},
    format::{Config, Dtype, ObjectRef, Tree},
    store::Store,
    writer::{Data, Source},
};

const UNKNOWN: &str = "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";

/// A one-tensor header of `byte`s.
fn header(store: &Store, byte: u8) -> ObjectRef {
    let data = vec![byte; 4096];
    let mut writer = store.writer(None, None).unwrap();
    let config: Config = serde_json::from_str("{}").unwrap();
    (writer.component("model", Default::default(), config, None).unwrap())
        .put("weight", &Dtype::U8, &[data.len() as u64], Data::Plain(Source::Read(&mut &data[..])))
        .unwrap();
    writer.commit().unwrap()
}

/// The owner's two uploads of `model`: `f32` to tag `draft`, then `f32` plus `fp8` to `main`.
/// Their commits, and the fp8 header.
fn upload(home: &Home, model: &str) -> (String, String, String) {
    let dir = tempfile::tempdir().unwrap();
    let store = Store::open(dir.path()).unwrap();
    let (f32, fp8) = (header(&store, 1), header(&store, 2));
    let mut base = Tree::default();
    base.set_variant("f32", f32);
    let mut more = base.clone();
    more.set_variant("fp8", fp8.clone());
    let (base, more) = (store.put_tree(&base).unwrap(), store.put_tree(&more).unwrap());
    let ca = std::fs::read(std::env::var("TENSORHUB_CA").expect("the Hub's CA")).unwrap();
    let policy = SourcePolicy { any_host: true, allow_local: true, max_redirects: 5, ..SourcePolicy::default() };
    let client = Client::with_roots(policy, &ca).unwrap();
    let url = std::env::var("TENSORHUB_ORIGIN").unwrap();
    let hub = Endpoint::new(&client, &url, Some(bearer(home.token()))).unwrap();
    tensorfs::fetch::runtime().unwrap().block_on(async {
        upload_tree(&hub, model, &base, &store, "draft").await.unwrap();
        upload_tree(&hub, model, &more, &store, "main").await.unwrap();
    });
    (base.id(), more.id(), fp8.id())
}

fn refused(home: &Home, args: &[&str]) -> Value {
    let out = home.cozy(["--json"].iter().chain(args));
    assert_eq!(out.code, 1, "{args:?}: {out:?}");
    out.refusal()
}

#[test]
fn release_verbs_edit_the_repo_document_as_its_owner() {
    let Some(home) = Home::with_hub() else { return };
    let signed_out = refused(&home, &["model", "publish", "lmodel/none", "--release", "1.0.0", "--commit", "main"]);
    assert_eq!(signed_out["code"], "not_logged_in", "{signed_out}");

    let account = home.sign_in("lmodel@example.test");
    let model = format!("{account}/tiny");
    let none = refused(&home, &["model", "publish", &model, "--release", "1.0.0", "--commit", "main"]);
    assert_eq!(none["code"], "model_not_found", "{none}");

    let (base, more, fp8) = upload(&home, &model);
    let publish = |release: &str, commit: &str| home.json(["--json", "model", "publish", &model, "--release", release, "--commit", commit]);
    let released = publish("1.0.0", "draft");
    assert_eq!((released["commit"].as_str(), &released["public"], &released["changed"]), (Some(base.as_str()), &Value::Bool(true), &Value::Bool(true)), "{released}");
    let moved = publish("1.0.0", &more);
    assert_eq!(moved["commit"].as_str(), Some(more.as_str()), "{moved}");
    let by_header = publish("1.1.0", &fp8);
    assert_eq!(by_header["commit"].as_str(), Some(more.as_str()), "{by_header}");
    let unknown = refused(&home, &["model", "publish", &model, "--release", "2.0.0", "--commit", UNKNOWN]);
    assert_eq!(unknown["code"], "commit_not_found", "{unknown}");
    let unversioned = refused(&home, &["model", "publish", &model, "--release", "nightly", "--commit", "main"]);
    assert!(unversioned["message"].as_str().unwrap().contains("PEP 440"), "{unversioned}");

    let yank = |extra: &[&str]| home.json(["--json", "model", "yank", &model, "--release", "1.0.0"].iter().chain(extra));
    let yanked = yank(&[]);
    assert_eq!((&yanked["yanked"], &yanked["changed"]), (&Value::Bool(true), &Value::Bool(true)), "{yanked}");
    assert_eq!(yank(&[])["changed"], false);
    assert_eq!(yank(&["--undo"])["yanked"], false);

    let draft = format!("{model}@draft");
    let deleted = home.json(["--json", "model", "delete", &draft, "--yes"]);
    assert_eq!((deleted["tag"].as_str(), &deleted["removed"]), (Some("draft"), &Value::Bool(true)), "{deleted}");
    let deleted = home.json(["--json", "model", "delete", &model, "--yes"]);
    assert_eq!((deleted["model"].as_str(), &deleted["removed"]), (Some(model.as_str()), &Value::Bool(true)), "{deleted}");
}
