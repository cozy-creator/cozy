//! `cozy model publish|retarget|yank|delete` against a real Tensorhub (Tensorhub's dev-build
//! `cozy_ref` job: TENSORHUB_ORIGIN behind the private CA TENSORHUB_CA). Each verb edits the
//! model's repo document at the Hub's distribution endpoint as the signed-in owner. Two trees
//! the owner uploads with TensorFS, the second adding a variant, are released, retargeted and
//! yanked, and the model is emptied; the Hub's own policy refusals come back.
mod common;

use common::Home;
use serde_json::Value;
use tensorfs::{
    fetch::{bearer, upload_tree, Client, Endpoint, SourcePolicy},
    format::{Config, Dtype, ObjectRef, Tree},
    store::Store,
    writer::{Data, Source},
};

const CHECKPOINT: &str = "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";

/// A one-tensor header of `byte`s.
fn header(store: &Store, byte: u8) -> ObjectRef {
    let data = vec![byte; 4096];
    let mut writer = store.writer(None, None).unwrap();
    let config: Config = serde_json::from_str("{}").unwrap();
    (writer.component("model", "conformance.Fixture", config, None).unwrap())
        .put("weight", &Dtype::U8, &[data.len() as u64], Data::Plain(Source::Read(&mut &data[..])))
        .unwrap();
    writer.commit().unwrap()
}

/// Two checkpoints of `model` the owner uploads: `f32`, and `f32` plus `fp8`.
fn upload(home: &Home, model: &str) -> (String, String) {
    let dir = tempfile::tempdir().unwrap();
    let store = Store::open(dir.path()).unwrap();
    let (f32, fp8) = (header(&store, 1), header(&store, 2));
    let mut base = Tree::default();
    base.set_variant("f32", f32);
    let mut more = base.clone();
    more.set_variant("fp8", fp8);
    let (base, more) = (store.put_tree(&base).unwrap(), store.put_tree(&more).unwrap());
    let ca = std::fs::read(std::env::var("TENSORHUB_CA").expect("the Hub's CA")).unwrap();
    let policy = SourcePolicy { any_host: true, allow_local: true, max_redirects: 5, ..SourcePolicy::default() };
    let client = Client::with_roots(policy, &ca).unwrap();
    let url = std::env::var("TENSORHUB_ORIGIN").unwrap();
    let hub = Endpoint::new(&client, &url, Some(bearer(home.token()))).unwrap();
    tensorfs::fetch::runtime().unwrap().block_on(async {
        for tree in [&base, &more] {
            upload_tree(&hub, model, tree, &store, None).await.unwrap();
        }
    });
    (base.id(), more.id())
}

fn refused(home: &Home, args: &[&str]) -> Value {
    let out = home.cozy(["--json"].iter().chain(args));
    assert_eq!(out.code, 1, "{args:?}: {out:?}");
    out.refusal()
}

#[test]
fn release_verbs_edit_the_repo_document_as_its_owner() {
    let Some(home) = Home::with_hub() else { return };
    let signed_out = refused(&home, &["model", "publish", "lmodel/none", "--release", "1.0.0", "--tree", CHECKPOINT]);
    assert_eq!(signed_out["code"], "not_logged_in", "{signed_out}");

    let account = home.sign_in("lmodel@example.test");
    let model = format!("{account}/tiny");
    let none = refused(&home, &["model", "publish", &model, "--release", "1.0.0", "--tree", CHECKPOINT]);
    assert_eq!(none["code"], "model_not_found", "{none}");

    let (base, more) = upload(&home, &model);
    let unheld = refused(&home, &["model", "publish", &model, "--release", "1.0.0", "--tree", CHECKPOINT]);
    assert_eq!(unheld["code"], "checkpoint_not_found", "{unheld}");
    let release = |verb: &str, tree: &str| home.json(["--json", "model", verb, &model, "--release", "1.0.0", "--tree", tree]);
    let published = release("publish", &base);
    assert_eq!((published["tree"].as_str(), &published["changed"]), (Some(base.as_str()), &Value::Bool(true)), "{published}");
    let moved = release("retarget", &more);
    assert_eq!(moved["tree"].as_str(), Some(more.as_str()), "{moved}");
    // Back to a tree without the fp8 variant: the Hub refuses, and its answer says so.
    let shrunk = refused(&home, &["model", "retarget", &model, "--release", "1.0.0", "--tree", &base]);
    assert_eq!(shrunk["code"], "repo_conflict", "{shrunk}");
    assert!(shrunk["message"].as_str().unwrap().contains("policy"), "{shrunk}");

    let yank = || home.json(["--json", "model", "yank", &model, "--release", "1.0.0"]);
    let yanked = yank();
    assert_eq!((&yanked["yanked"], &yanked["changed"]), (&Value::Bool(true), &Value::Bool(true)), "{yanked}");
    assert_eq!(yank()["changed"], false);

    let deleted = home.json(["--json", "model", "delete", &model, "--yes"]);
    assert_eq!((deleted["model"].as_str(), &deleted["removed"]), (Some(model.as_str()), &Value::Bool(true)), "{deleted}");
}
