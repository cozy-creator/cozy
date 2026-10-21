//! `cozy model publish|yank|delete`: a model's tags, as its owner. Each verb is one edit of the
//! model's repo document at the Hub's distribution endpoint (TensorFS's `Endpoint::update_repo`):
//! read it, change it, write it under `If-Match` on what was read. A lost race reads it again and
//! re-applies the edit, at most [`ATTEMPTS`] times; an edit the newer document no longer allows is
//! refused. A public tag is a release, named by a PEP 440 version.
use std::process::ExitCode;

use base64::Engine as _;
use base64::engine::general_purpose::URL_SAFE_NO_PAD;
use protocol::config::Hub;
use protocol::error::{Error, Kind};
use serde::Serialize;
use tensorfs::format::{ObjectRef, Repo as Doc};
use tensorfs::{Config, Endpoint, Failure};

use super::refs::{ModelRef, Repo, label};
use super::{Delete, Publish, Yank};
use crate::Context;
use crate::context::{hub_answer, hub_error};
use crate::output::{self, Record};

/// Writes one edit tries before a document that keeps changing is reported busy.
const ATTEMPTS: u32 = 8;

fn usage(code: &str, message: impl Into<String>) -> Error {
    Error::new(Kind::Usage, code, message)
}

/// Where a release goes: where a tag is when the edit is written, or a commit.
enum At {
    Tag(String),
    Commit(ObjectRef),
}

impl At {
    /// `--commit` of `release`: a tag, or a commit id (`sha256:`), resolved once since a commit
    /// is immutable (a variant header's is the commit that cites it); none is the release's own.
    async fn of(endpoint: &Endpoint, model: &str, release: &str, commit: Option<String>) -> Result<At, Error> {
        Ok(match commit {
            Some(id) if id.starts_with("sha256:") => match endpoint.commit(model, &id).await {
                Ok((commit, _)) => At::Commit(commit),
                Err(f) if f.kind == tensorfs::Kind::NotFound => return Err(Error::new(Kind::NotFound, "commit_not_found", f.detail)),
                Err(f) => return Err(failed(f)),
            },
            Some(tag) => At::Tag(tag),
            None => At::Tag(release.to_owned()),
        })
    }
}

/// One owner's change to a model's repo document.
enum Edit {
    /// Tag `release` public at `at`.
    Publish { release: String, at: At },
    Yank { release: String, yanked: bool },
    /// Removes one tag, or with none every tag: the model.
    Delete { tag: Option<String> },
}

impl Edit {
    /// `doc` as this edit leaves it, or why it no longer applies.
    fn apply(&self, model: &str, doc: Option<Doc>) -> Result<Doc, Error> {
        let mut doc = doc.ok_or_else(|| {
            Error::new(Kind::NotFound, "model_not_found", format!("{model} has no commits here"))
                .next(format!("cozy model upload <source> {model}"))
        })?;
        let absent = |tag: &str| Error::new(Kind::NotFound, "tag_not_found", format!("{model} has no tag {tag}"));
        match self {
            Edit::Publish { release, at } => {
                let commit = match at {
                    At::Commit(commit) => commit.clone(),
                    At::Tag(tag) => match doc.tag(tag) {
                        Some(tag) => tag.commit.clone(),
                        None if tag == release => {
                            return Err(absent(tag).remedy("name the commit it releases with --commit <tag|sha256:…>"));
                        }
                        None => return Err(absent(tag)),
                    },
                };
                doc.set_tag(release, commit);
                doc.tag_mut(release).expect("set").public = true;
            }
            Edit::Yank { release, yanked } => match doc.tag_mut(release) {
                Some(tag) => tag.yanked = *yanked,
                None => return Err(absent(release)),
            },
            Edit::Delete { tag: Some(tag) } => doc.tags.retain(|t| &t.name != tag),
            Edit::Delete { tag: None } => doc.tags.clear(),
        }
        Ok(doc)
    }
}

/// Writes what `edit` makes of `model`'s document at `endpoint`: the document as written, and
/// whether it changed.
async fn edit_repo(
    endpoint: &Endpoint,
    model: &str,
    mut edit: impl FnMut(Option<Doc>) -> Result<Doc, Error>,
) -> Result<(Doc, bool), Error> {
    let (mut attempts, mut refused, mut changed) = (0, None, false);
    let written = endpoint
        .update_repo(model, |old| {
            attempts += 1;
            let new = match attempts > ATTEMPTS {
                true => Err(Error::new(Kind::Conflict, "model_busy", format!("{model} changed under each of {ATTEMPTS} writes"))
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

/// A tag as written.
#[derive(Serialize)]
struct Changed {
    model: String,
    tag: String,
    commit: String,
    public: bool,
    yanked: bool,
    /// False when the tag already was as asked.
    changed: bool,
}

/// `edit` of `model`'s tag `release`, and the tag as written.
async fn one_tag(ctx: &Context, endpoint: &Endpoint, model: String, release: &str, edit: Edit) -> Result<ExitCode, Error> {
    let (doc, changed) = edit_repo(endpoint, &model, |old| edit.apply(&model, old)).await?;
    let tag = doc.tag(release).expect("the edit keeps its tag");
    let view = Changed { model, tag: tag.name.clone(), commit: tag.commit.id(), public: tag.public, yanked: tag.yanked, changed };
    match ctx.json {
        true => output::json(&view),
        false => Record::of(&view).print(ctx),
    }
    Ok(ExitCode::SUCCESS)
}

pub(crate) async fn publish(ctx: &Context, args: Publish) -> Result<ExitCode, Error> {
    let (model, release) = (Repo::hub(&args.model)?.to_string(), label("--release", &args.release)?);
    let endpoint = endpoint(ctx, &ctx.hub()?).await?;
    let at = At::of(&endpoint, &model, &release, args.commit).await?;
    one_tag(ctx, &endpoint, model, &release, Edit::Publish { release: release.clone(), at }).await
}

pub(crate) async fn yank(ctx: &Context, args: Yank) -> Result<ExitCode, Error> {
    let (model, release) = (Repo::hub(&args.model)?.to_string(), label("--release", &args.release)?);
    let endpoint = endpoint(ctx, &ctx.hub()?).await?;
    one_tag(ctx, &endpoint, model, &release, Edit::Yank { release: release.clone(), yanked: !args.undo }).await
}

#[derive(Serialize)]
struct Deleted {
    model: String,
    /// One tag; absent when the whole model went.
    #[serde(skip_serializing_if = "Option::is_none")]
    tag: Option<String>,
    /// False when it was already gone.
    removed: bool,
}

/// Never prompts: `--yes` is the confirmation, and without it nothing is sent.
pub(crate) async fn delete(ctx: &Context, args: Delete) -> Result<ExitCode, Error> {
    let (repo, tag) = match ModelRef::parse(&args.model)? {
        ModelRef::Hub { repo, release, lane: None } => (repo, release),
        _ => return Err(usage("model_ref_invalid", format!("{:?}: delete takes org/name or org/name@tag", args.model))),
    };
    let hub = ctx.hub()?;
    if !ctx.yes {
        return Err(usage("model_delete_unconfirmed", format!("deleting {} from {} cannot be undone", args.model, hub.origin))
            .remedy("run it again with --yes"));
    }
    let model = repo.to_string();
    let edit = Edit::Delete { tag: tag.clone() };
    let (_, removed) = edit_repo(&endpoint(ctx, &hub).await?, &model, |old| edit.apply(&model, old)).await?;
    let view = Deleted { model, tag, removed };
    match ctx.json {
        true => output::json(&view),
        false => Record::of(&view).note("The Hub's GC reclaims the commits no tag names.").print(ctx),
    }
    Ok(ExitCode::SUCCESS)
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use tensorfs::format::{Tag, Tree};

    use super::*;

    const MODEL: &str = "fidika/tiny";

    /// A commit of `variants`, each naming a header of its own name's bytes, stored at the
    /// directory endpoint `dir` as plain files.
    fn commit_of(dir: &Path, variants: &[&str]) -> ObjectRef {
        let mut tree = Tree::default();
        for variant in variants {
            tree.set_variant(variant, ObjectRef::of(variant.as_bytes()));
        }
        let bytes = tree.to_json().unwrap();
        let commit = ObjectRef::of(&bytes);
        let path = dir.join(commit.key());
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(path, bytes).unwrap();
        commit
    }

    fn endpoint_at(dir: &Path) -> Endpoint {
        Endpoint::new(&format!("file://{}", dir.display()), None).unwrap()
    }

    async fn edit(endpoint: &Endpoint, edit: Edit) -> Result<(Doc, bool), Error> {
        edit_repo(endpoint, MODEL, |old| edit.apply(MODEL, old)).await
    }

    /// `release` published at what `--commit` names, as `cozy model publish` does.
    async fn publish(endpoint: &Endpoint, release: &str, commit: Option<&str>) -> Result<(Doc, bool), Error> {
        let at = At::of(endpoint, MODEL, release, commit.map(str::to_owned)).await?;
        edit(endpoint, Edit::Publish { release: release.into(), at }).await
    }

    fn release(release: &str, tag: &str) -> Edit {
        Edit::Publish { release: release.into(), at: At::Tag(tag.into()) }
    }

    /// `change` by another writer, landing now through its own endpoint over the same directory.
    fn elsewhere<T: Send + 'static>(dir: &Path, change: impl FnOnce(Endpoint) -> T + Send + 'static) -> T {
        let dir = dir.to_owned();
        std::thread::spawn(move || change(endpoint_at(&dir))).join().unwrap()
    }

    fn block_on<F: std::future::Future>(future: F) -> F::Output {
        tokio::runtime::Builder::new_current_thread().enable_all().build().unwrap().block_on(future)
    }

    /// An upload by another writer: private tag `tag` moves to `commit`.
    fn uploaded(dir: &Path, tag: &str, commit: &ObjectRef) {
        let (tag, commit) = (tag.to_owned(), commit.clone());
        elsewhere(dir, move |endpoint| {
            let upload = endpoint.update_repo(MODEL, |old| {
                let mut doc = old.unwrap_or_else(|| Doc::new(MODEL));
                doc.set_tag(&tag, commit.clone());
                Ok(Some(doc))
            });
            block_on(upload).unwrap();
        });
    }

    fn tags(doc: &Doc) -> Vec<(&str, String, bool, bool)> {
        doc.tags.iter().map(|t| (t.name.as_str(), t.commit.id(), t.public, t.yanked)).collect()
    }

    #[tokio::test]
    async fn a_commit_is_released_by_tag_commit_or_header_and_moves_or_goes_public() {
        let dir = tempfile::tempdir().unwrap();
        let (base, more) = (commit_of(dir.path(), &["f32"]), commit_of(dir.path(), &["f32", "fp8"]));
        uploaded(dir.path(), "main", &base);
        let endpoint = endpoint_at(dir.path());

        let (doc, changed) = publish(&endpoint, "1.0.0", Some("main")).await.unwrap();
        assert!(changed);
        assert_eq!(tags(&doc), [("main", base.id(), false, false), ("1.0.0", base.id(), true, false)]);
        assert!(!publish(&endpoint, "1.0.0", Some(&base.id())).await.unwrap().1, "the same commit again");

        // A commit no tag names, by its id: the release moves there and keeps its place.
        let (doc, _) = publish(&endpoint, "1.0.0", Some(&more.id())).await.unwrap();
        assert_eq!(tags(&doc), [("main", base.id(), false, false), ("1.0.0", more.id(), true, false)]);
        assert_eq!(endpoint.repo(MODEL).await.unwrap().unwrap().0, doc, "what the endpoint holds");
        // A variant header names the commit that cites it.
        let fp8 = ObjectRef::of(b"fp8").id();
        let (doc, _) = publish(&endpoint, "1.1.0", Some(&fp8)).await.unwrap();
        assert_eq!(doc.tag("1.1.0").map(|t| t.commit.id()), Some(more.id()));

        // A private tag goes public where it is.
        uploaded(dir.path(), "2.0.0", &base);
        let (doc, _) = publish(&endpoint, "2.0.0", None).await.unwrap();
        assert_eq!(doc.tag("2.0.0").map(|t| (t.commit.id(), t.public)), Some((base.id(), true)));

        let int8 = ObjectRef::of(b"int8").id();
        for (release, commit, code) in [("3.0.0", None, "tag_not_found"), ("3.0.0", Some("nightly"), "tag_not_found"), ("3.0.0", Some(int8.as_str()), "commit_not_found")] {
            let error = publish(&endpoint, release, commit).await.unwrap_err();
            assert_eq!(error.code, code, "{error:?}");
        }
        let unversioned = publish(&endpoint, "nightly", Some("main")).await.unwrap_err();
        assert!(unversioned.kind == Kind::Validation && unversioned.message.contains("PEP 440"), "{unversioned:?}");
        let nothing = edit(&endpoint_at(&dir.path().join("empty")), release("1.0.0", "main")).await.unwrap_err();
        assert_eq!(nothing.code, "model_not_found", "{nothing:?}");
    }

    #[tokio::test]
    async fn a_release_is_yanked_and_unyanked_and_tags_and_models_are_deleted() {
        let dir = tempfile::tempdir().unwrap();
        uploaded(dir.path(), "main", &commit_of(dir.path(), &["f32"]));
        let endpoint = endpoint_at(dir.path());
        edit(&endpoint, release("1.0.0", "main")).await.unwrap();

        let yank = |yanked| Edit::Yank { release: "1.0.0".into(), yanked };
        let (doc, changed) = edit(&endpoint, yank(true)).await.unwrap();
        assert_eq!((doc.tag("1.0.0").map(|t| t.yanked), changed), (Some(true), true));
        assert!(!edit(&endpoint, yank(true)).await.unwrap().1, "yanked again is no change");
        let (doc, changed) = edit(&endpoint, yank(false)).await.unwrap();
        assert_eq!((doc.tag("1.0.0").map(|t| t.yanked), changed), (Some(false), true));
        let missing = edit(&endpoint, Edit::Yank { release: "9.0.0".into(), yanked: true }).await.unwrap_err();
        assert_eq!(missing.code, "tag_not_found", "{missing:?}");

        let main = || Edit::Delete { tag: Some("main".into()) };
        let (doc, removed) = edit(&endpoint, main()).await.unwrap();
        assert!(removed && doc.tag("main").is_none() && doc.tag("1.0.0").is_some(), "{doc:?}");
        assert!(!edit(&endpoint, main()).await.unwrap().1, "already gone");
        let (doc, removed) = edit(&endpoint, Edit::Delete { tag: None }).await.unwrap();
        assert!(removed && doc.tags.is_empty(), "{doc:?}");
        assert_eq!(endpoint.repo(MODEL).await.unwrap().unwrap().0, doc);
    }

    #[tokio::test]
    async fn a_concurrent_edit_is_read_again_and_the_edit_applied_to_it() {
        let dir = tempfile::tempdir().unwrap();
        let (base, more) = (commit_of(dir.path(), &["f32"]), commit_of(dir.path(), &["f32", "fp8"]));
        uploaded(dir.path(), "main", &base);
        let endpoint = endpoint_at(dir.path());

        // An upload moves main between this edit's read and its write: 1.0.0 goes where main is.
        let mut reads = 0;
        let (doc, changed) = edit_repo(&endpoint, MODEL, |old| {
            reads += 1;
            if reads == 1 {
                uploaded(dir.path(), "main", &more);
            }
            release("1.0.0", "main").apply(MODEL, old)
        })
        .await
        .unwrap();
        assert_eq!((reads, changed), (2, true), "one lost race, one read again");
        assert_eq!(tags(&doc), [("main", more.id(), false, false), ("1.0.0", more.id(), true, false)]);
        assert_eq!(endpoint.repo(MODEL).await.unwrap().unwrap().0, doc);

        // Another writer deletes main first: this edit no longer applies, and says why.
        let mut reads = 0;
        let refused = edit_repo(&endpoint, MODEL, |old| {
            reads += 1;
            if reads == 1 {
                elsewhere(dir.path(), |other| block_on(edit(&other, Edit::Delete { tag: Some("main".into()) }))).unwrap();
            }
            release("2.0.0", "main").apply(MODEL, old)
        })
        .await
        .unwrap_err();
        assert_eq!((refused.kind, refused.code.as_str(), reads), (Kind::NotFound, "tag_not_found", 2), "{refused:?}");
        let held = endpoint.repo(MODEL).await.unwrap().unwrap().0;
        assert_eq!(tags(&held), [("1.0.0", more.id(), true, false)]);
    }

    #[tokio::test]
    async fn a_document_that_changes_under_every_write_is_reported_busy() {
        let dir = tempfile::tempdir().unwrap();
        let base = commit_of(dir.path(), &["f32"]);
        uploaded(dir.path(), "main", &base);
        let endpoint = endpoint_at(dir.path());
        let mut reads = 0;
        let busy = edit_repo(&endpoint, MODEL, |old| {
            reads += 1;
            uploaded(dir.path(), &format!("draft{reads}"), &base);
            release("1.0.0", "main").apply(MODEL, old)
        })
        .await
        .unwrap_err();
        assert_eq!((busy.kind, busy.code.as_str(), reads), (Kind::Conflict, "model_busy", ATTEMPTS), "{busy:?}");
        let held = endpoint.repo(MODEL).await.unwrap().unwrap().0;
        assert_eq!((held.tags.len(), held.tag("1.0.0")), (1 + ATTEMPTS as usize, None::<&Tag>));
    }
}
