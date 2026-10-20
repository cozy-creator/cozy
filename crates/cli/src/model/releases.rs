//! `cozy model publish|retarget|yank|delete`: a model's releases, as its owner. Each verb is one
//! edit of the model's repo document at the Hub's distribution endpoint (TensorFS's
//! `Endpoint::update_repo`): read it, change it, write it under `If-Match` on what was read. A
//! lost race reads it again and re-applies the edit, at most [`ATTEMPTS`] times; an edit the newer
//! document no longer allows is refused. The Hub's policy rules the rest: a yank is never undone,
//! and a retarget only adds variant files to the release's tree.
use std::process::ExitCode;

use base64::Engine as _;
use base64::engine::general_purpose::URL_SAFE_NO_PAD;
use protocol::config::Hub;
use protocol::error::{Error, Kind};
use serde::Serialize;
use tensorfs::format::{ObjectRef, Repo as Doc};
use tensorfs::{Config, Endpoint, Failure};

use super::refs::{ModelRef, Repo, checkpoint, label};
use super::{Delete, Publish, Retarget, Yank};
use crate::Context;
use crate::context::{hub_answer, hub_error};
use crate::output::{self, Record};

/// Writes one edit tries before a document that keeps changing is reported busy.
const ATTEMPTS: u32 = 8;

fn usage(code: &str, message: impl Into<String>) -> Error {
    Error::new(Kind::Usage, code, message)
}

fn conflict(code: &str, message: impl Into<String>) -> Error {
    Error::new(Kind::Conflict, code, message)
}

/// One owner's change to a model's repo document.
enum Edit {
    /// A new release names a tree the model holds.
    Publish { release: String, tree: String },
    /// An existing release names another tree the model holds.
    Retarget { release: String, tree: String },
    Yank { release: String },
    /// One unreleased tree goes.
    Remove { tree: String },
    /// The whole model goes, once every release is yanked.
    Empty,
}

impl Edit {
    /// `doc` as this edit leaves it, or why it no longer applies.
    fn apply(&self, model: &str, doc: Option<Doc>) -> Result<Doc, Error> {
        let mut doc = doc.ok_or_else(|| {
            Error::new(Kind::NotFound, "model_not_found", format!("{model} holds no checkpoint here"))
                .next(format!("cozy model upload <source> {model}"))
        })?;
        let absent = |release: &str| Error::new(Kind::NotFound, "release_not_found", format!("{model} has no release {release}"));
        match self {
            Edit::Publish { release, tree } | Edit::Retarget { release, tree } => {
                match (self, doc.release(Some(release.as_str())).map(|r| r.tree.id())) {
                    (_, Some(named)) if &named == tree => return Ok(doc),
                    (Edit::Publish { .. }, Some(named)) => {
                        return Err(conflict("release_exists", format!("{model}@{release} names {named}"))
                            .next(format!("cozy model retarget {model} --release {release} --tree {tree}")));
                    }
                    (Edit::Retarget { .. }, None) => {
                        return Err(absent(release).next(format!("cozy model publish {model} --release {release} --tree {tree}")));
                    }
                    _ => {}
                }
                let mut roots: Vec<ObjectRef> = doc.roots().cloned().collect();
                roots.sort();
                roots.dedup();
                let Some(named) = roots.iter().find(|r| &r.id() == tree).cloned() else {
                    return Err(Error::new(Kind::NotFound, "checkpoint_not_found", format!("{model} holds no {tree}"))
                        .remedy("upload the checkpoint to this model first"));
                };
                doc.set_release(release, named);
                // A tree a release left stays an unreleased checkpoint; a released one is none.
                roots.retain(|root| !doc.releases.iter().any(|r| &r.tree == root));
                doc.checkpoints = roots;
            }
            Edit::Yank { release } => match doc.releases.iter_mut().find(|r| &r.version == release) {
                Some(r) => r.yanked = true,
                None => return Err(absent(release)),
            },
            Edit::Remove { tree } => {
                if let Some(r) = doc.releases.iter().find(|r| &r.tree.id() == tree) {
                    return Err(conflict("checkpoint_released", format!("{model}@{} names {tree}", r.version))
                        .remedy("only an unreleased checkpoint is removed"));
                }
                doc.checkpoints.retain(|c| &c.id() != tree);
            }
            Edit::Empty => {
                if let Some(r) = doc.releases.iter().find(|r| !r.yanked) {
                    return Err(conflict("release_live", format!("{model}@{} is live", r.version))
                        .remedy("yank every release first")
                        .next(format!("cozy model yank {model} --release {}", r.version)));
                }
                doc = Doc { name: doc.name, ..Doc::default() };
            }
        }
        Ok(doc)
    }
}

/// Writes what `edit` makes of `model`'s document at `endpoint`: the document as written, and
/// whether it changed.
async fn commit(
    endpoint: &Endpoint,
    model: &str,
    mut edit: impl FnMut(Option<Doc>) -> Result<Doc, Error>,
) -> Result<(Doc, bool), Error> {
    let (mut attempts, mut refused, mut changed) = (0, None, false);
    let written = endpoint
        .update_repo(model, |old| {
            attempts += 1;
            let new = match attempts > ATTEMPTS {
                true => Err(conflict("model_busy", format!("{model} changed under each of {ATTEMPTS} writes"))
                    .remedy("run it again")),
                false => edit(old.clone()),
            };
            match new {
                Ok(new) => {
                    changed = old.as_ref() != Some(&new);
                    Ok(Some(new))
                }
                Err(error) => {
                    refused = Some(error);
                    Err(Failure { kind: tensorfs::Kind::Conflict, detail: String::new() })
                }
            }
        })
        .await;
    match (refused, written) {
        (Some(error), _) => Err(error),
        (None, Ok(doc)) => Ok((doc, changed)),
        (None, Err(failure)) => Err(failed(failure)),
    }
}

/// TensorFS's refusal as the CLI's; its detail carries the endpoint's own code and message.
fn failed(failure: Failure) -> Error {
    use tensorfs::Kind as T;
    let kind = match failure.kind {
        T::Conflict => Kind::Conflict,
        T::NotFound => Kind::NotFound,
        T::Denied => Kind::Credential,
        T::Network | T::Io => Kind::Unavailable,
        T::Invalid | T::Format => Kind::Validation,
        _ => Kind::Internal,
    };
    Error::new(kind, format!("repo_{}", failure.kind.code()), failure.detail)
}

/// `hub`'s distribution endpoint as this computer's signed-in user, trusting the private CA the
/// Hub names (a test Hub's).
async fn endpoint(ctx: &Context, hub: &Hub) -> Result<Endpoint, Error> {
    let client = ctx.hub_signed_in(hub)?;
    let hub_client::Credential::User(session) = client.credential() else {
        unreachable!("a signed-in client");
    };
    let ca = hub_answer(client.get_execution_environment().await)?.environment.tensorhub_ca_der_b64url;
    let token = session.bearer(&client.anonymous()).await.map_err(hub_error)?.token;
    let config = Config {
        token: Some(tensorfs::bearer(token)),
        ca_der: ca.and_then(|ca| URL_SAFE_NO_PAD.decode(ca.trim()).ok()),
        ..Config::from_env()
    };
    Endpoint::with_config(&hub.origin.to_string(), &config).map_err(failed)
}

/// A release as written.
#[derive(Serialize)]
struct Changed {
    model: String,
    release: String,
    tree: String,
    yanked: bool,
    /// False when the release already was as asked.
    changed: bool,
}

/// `edit` of `model`'s `release`, and the release as written.
async fn one_release(ctx: &Context, model: &str, release: &str, edit: Edit) -> Result<ExitCode, Error> {
    let model = Repo::hub(model)?.to_string();
    let endpoint = endpoint(ctx, &ctx.hub()?).await?;
    let (doc, changed) = commit(&endpoint, &model, |old| edit.apply(&model, old)).await?;
    let r = doc.release(Some(release)).expect("the edit keeps its release");
    let view = Changed { model, release: r.version.clone(), tree: r.tree.id(), yanked: r.yanked, changed };
    match ctx.json {
        true => output::json(&view),
        false => Record::of(&view).print(ctx),
    }
    Ok(ExitCode::SUCCESS)
}

pub(crate) async fn publish(ctx: &Context, args: Publish) -> Result<ExitCode, Error> {
    let (release, tree) = (label("--release", &args.release)?, checkpoint(&args.tree)?);
    one_release(ctx, &args.model, &release, Edit::Publish { release: release.clone(), tree }).await
}

pub(crate) async fn retarget(ctx: &Context, args: Retarget) -> Result<ExitCode, Error> {
    let (release, tree) = (label("--release", &args.release)?, checkpoint(&args.tree)?);
    one_release(ctx, &args.model, &release, Edit::Retarget { release: release.clone(), tree }).await
}

pub(crate) async fn yank(ctx: &Context, args: Yank) -> Result<ExitCode, Error> {
    let release = label("--release", &args.release)?;
    one_release(ctx, &args.model, &release, Edit::Yank { release: release.clone() }).await
}

#[derive(Serialize)]
struct Deleted {
    model: String,
    /// One unreleased checkpoint; absent when the whole model went.
    #[serde(skip_serializing_if = "Option::is_none")]
    checkpoint: Option<String>,
    /// False when it was already gone.
    removed: bool,
}

/// Never prompts: `--yes` is the confirmation, and without it nothing is sent.
pub(crate) async fn delete(ctx: &Context, args: Delete) -> Result<ExitCode, Error> {
    let (repo, checkpoint) = match ModelRef::parse(&args.model)? {
        ModelRef::Checkpoint { repo, manifest } => (repo, Some(manifest)),
        ModelRef::Hub { repo, release: None, lane: None } => (repo, None),
        _ => return Err(usage("model_ref_invalid", format!("{:?}: delete takes org/name or org/name#sha256:…", args.model))),
    };
    let hub = ctx.hub()?;
    if !args.yes {
        return Err(usage("model_delete_unconfirmed", format!("deleting {} from {} cannot be undone", args.model, hub.origin))
            .remedy("run it again with --yes"));
    }
    let model = repo.to_string();
    let edit = checkpoint.clone().map_or(Edit::Empty, |tree| Edit::Remove { tree });
    let (_, removed) = commit(&endpoint(ctx, &hub).await?, &model, |old| edit.apply(&model, old)).await?;
    let view = Deleted { model, checkpoint, removed };
    match ctx.json {
        true => output::json(&view),
        false => Record::of(&view).note("The Hub's GC reclaims only objects no other model references.").print(ctx),
    }
    Ok(ExitCode::SUCCESS)
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use tensorfs::format::Tree;

    use super::*;

    const MODEL: &str = "fidika/tiny";

    /// A real tree of `variants`, each naming a header of its own name's bytes.
    fn tree(variants: &[&str]) -> ObjectRef {
        let mut tree = Tree::default();
        for variant in variants {
            tree.set_variant(variant, ObjectRef::of(variant.as_bytes()));
        }
        ObjectRef::of(&tree.to_json().unwrap())
    }

    fn at(dir: &Path) -> Endpoint {
        Endpoint::new(&format!("file://{}", dir.display()), None).unwrap()
    }

    async fn edit(endpoint: &Endpoint, edit: Edit) -> Result<(Doc, bool), Error> {
        commit(endpoint, MODEL, |old| edit.apply(MODEL, old)).await
    }

    /// Another writer's edit, landing now through its own endpoint over the same directory.
    fn elsewhere(dir: &Path, change: Edit) -> Result<(Doc, bool), Error> {
        let dir = dir.to_owned();
        let other = std::thread::spawn(move || {
            let runtime = tokio::runtime::Builder::new_current_thread().enable_all().build().unwrap();
            runtime.block_on(edit(&at(&dir), change))
        });
        other.join().unwrap()
    }

    /// A directory endpoint where MODEL holds `checkpoints`, unreleased, as uploads leave it.
    async fn uploaded(dir: &Path, checkpoints: &[&ObjectRef]) -> Endpoint {
        let endpoint = at(dir);
        let checkpoints: Vec<ObjectRef> = checkpoints.iter().map(|&c| c.clone()).collect();
        let doc = Doc { name: MODEL.into(), checkpoints, ..Doc::default() };
        endpoint.update_repo(MODEL, |_| Ok(Some(doc.clone()))).await.unwrap();
        endpoint
    }

    fn publish(release: &str, tree: &ObjectRef) -> Edit {
        Edit::Publish { release: release.into(), tree: tree.id() }
    }

    fn versions(doc: &Doc) -> Vec<(&str, String, bool)> {
        doc.releases.iter().map(|r| (r.version.as_str(), r.tree.id(), r.yanked)).collect()
    }

    #[tokio::test]
    async fn a_tree_is_released_then_retargeted_to_one_that_adds_a_variant() {
        let dir = tempfile::tempdir().unwrap();
        let (base, more) = (tree(&["f32"]), tree(&["f32", "fp8"]));
        let endpoint = uploaded(dir.path(), &[&base, &more]).await;

        let (doc, changed) = edit(&endpoint, publish("1.0.0", &base)).await.unwrap();
        assert!(changed);
        assert_eq!((versions(&doc), doc.checkpoints.clone()), (vec![("1.0.0", base.id(), false)], vec![more.clone()]));
        assert!(!edit(&endpoint, publish("1.0.0", &base)).await.unwrap().1, "the same release again is no change");
        let over = edit(&endpoint, publish("1.0.0", &more)).await.unwrap_err();
        assert_eq!((over.kind, over.code.as_str()), (Kind::Conflict, "release_exists"), "{over:?}");

        let retarget = |tree: &ObjectRef, release: &str| Edit::Retarget { release: release.into(), tree: tree.id() };
        let (doc, changed) = edit(&endpoint, retarget(&more, "1.0.0")).await.unwrap();
        assert!(changed);
        assert_eq!(versions(&doc), [("1.0.0", more.id(), false)]);
        assert_eq!(doc.checkpoints, std::slice::from_ref(&base), "the tree it left stays a checkpoint");
        assert_eq!(endpoint.repo(MODEL).await.unwrap().unwrap().0, doc, "what the endpoint holds");

        let missing = edit(&endpoint, retarget(&base, "2.0.0")).await.unwrap_err();
        assert_eq!((missing.kind, missing.code.as_str()), (Kind::NotFound, "release_not_found"), "{missing:?}");
        let unheld = edit(&endpoint, retarget(&tree(&["int8"]), "1.0.0")).await.unwrap_err();
        assert_eq!((unheld.kind, unheld.code.as_str()), (Kind::NotFound, "checkpoint_not_found"), "{unheld:?}");
        let nothing = commit(&at(&dir.path().join("empty")), MODEL, |old| publish("1.0.0", &base).apply(MODEL, old)).await.unwrap_err();
        assert_eq!(nothing.code, "model_not_found", "{nothing:?}");
    }

    #[tokio::test]
    async fn a_yank_stays_and_a_model_empties_only_once_every_release_is_yanked() {
        let dir = tempfile::tempdir().unwrap();
        let (base, spare) = (tree(&["f32"]), tree(&["bf16"]));
        let endpoint = uploaded(dir.path(), &[&base, &spare]).await;
        edit(&endpoint, publish("1.0.0", &base)).await.unwrap();

        let live = edit(&endpoint, Edit::Empty).await.unwrap_err();
        assert_eq!((live.kind, live.code.as_str()), (Kind::Conflict, "release_live"), "{live:?}");
        let yank = || Edit::Yank { release: "1.0.0".into() };
        let (doc, changed) = edit(&endpoint, yank()).await.unwrap();
        assert_eq!((versions(&doc), changed), (vec![("1.0.0", base.id(), true)], true));
        assert!(!edit(&endpoint, yank()).await.unwrap().1, "a yanked release yanked again is no change");

        let released = edit(&endpoint, Edit::Remove { tree: base.id() }).await.unwrap_err();
        assert_eq!(released.code, "checkpoint_released", "{released:?}");
        let (doc, removed) = edit(&endpoint, Edit::Remove { tree: spare.id() }).await.unwrap();
        assert!(removed && doc.checkpoints.is_empty(), "{doc:?}");
        assert!(!edit(&endpoint, Edit::Remove { tree: spare.id() }).await.unwrap().1, "already gone");

        let (doc, removed) = edit(&endpoint, Edit::Empty).await.unwrap();
        assert!(removed && doc.releases.is_empty() && doc.checkpoints.is_empty(), "{doc:?}");
        assert_eq!(endpoint.repo(MODEL).await.unwrap().unwrap().0, doc);
    }

    #[tokio::test]
    async fn a_concurrent_edit_is_read_again_and_the_edit_applied_to_it() {
        let dir = tempfile::tempdir().unwrap();
        let (base, more) = (tree(&["f32"]), tree(&["f32", "fp8"]));
        let endpoint = uploaded(dir.path(), &[&base, &more]).await;

        // Another writer releases 2.0.0 between this edit's read and its write.
        let mut reads = 0;
        let (doc, changed) = commit(&endpoint, MODEL, |old| {
            reads += 1;
            if reads == 1 {
                elsewhere(dir.path(), publish("2.0.0", &more)).unwrap();
            }
            publish("1.0.0", &base).apply(MODEL, old)
        })
        .await
        .unwrap();
        assert_eq!(reads, 2, "one lost race, one read again");
        assert!(changed);
        assert_eq!(versions(&doc), [("2.0.0", more.id(), false), ("1.0.0", base.id(), false)]);
        assert!(doc.checkpoints.is_empty(), "{doc:?}");
        assert_eq!(endpoint.repo(MODEL).await.unwrap().unwrap().0, doc);

        // Another writer releases 3.0.0 first: this edit no longer applies, and says why.
        let mut reads = 0;
        let refused = commit(&endpoint, MODEL, |old| {
            reads += 1;
            if reads == 1 {
                elsewhere(dir.path(), publish("3.0.0", &more)).unwrap();
            }
            publish("3.0.0", &base).apply(MODEL, old)
        })
        .await
        .unwrap_err();
        assert_eq!((refused.kind, refused.code.as_str()), (Kind::Conflict, "release_exists"), "{refused:?}");
        assert!(refused.message.contains(&more.id()), "{refused:?}");
        let held = endpoint.repo(MODEL).await.unwrap().unwrap().0;
        assert_eq!(held.release(Some("3.0.0")).map(|r| r.tree.id()), Some(more.id()));
    }

    #[tokio::test]
    async fn a_document_that_changes_under_every_write_is_reported_busy() {
        let dir = tempfile::tempdir().unwrap();
        let base = tree(&["f32"]);
        let endpoint = uploaded(dir.path(), &[&base]).await;
        let mut reads = 0;
        let busy = commit(&endpoint, MODEL, |old| {
            reads += 1;
            elsewhere(dir.path(), publish(&format!("0.{reads}"), &base)).unwrap();
            publish("1.0.0", &base).apply(MODEL, old)
        })
        .await
        .unwrap_err();
        assert_eq!((busy.kind, busy.code.as_str(), reads), (Kind::Conflict, "model_busy", ATTEMPTS), "{busy:?}");
        let held = endpoint.repo(MODEL).await.unwrap().unwrap().0;
        assert_eq!((held.releases.len(), held.release(Some("1.0.0"))), (ATTEMPTS as usize, None));
    }
}
