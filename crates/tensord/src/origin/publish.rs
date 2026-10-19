//! Code-less uploads use TensorFS's distribution client. The ref PUT is the commit.
use std::{collections::BTreeSet, sync::atomic::{AtomicBool, Ordering}, time::Duration};
use futures_util::{stream, StreamExt};
use tensorfs_core::{err::{refuse, Code, Result}, fetch::{with_tree, Payload}, format::{Header, ObjectRef}, store2::Store};
use crate::{hub::{block, Catalog}, stop::Stop};

pub struct Publication<'a> {
    pub store: &'a Store,
    /// `org/name`, the model repository that takes the checkpoint.
    pub destination: &'a str,
    pub tree: &'a ObjectRef,
    /// Names this publication: the same operation resumes it, a new one starts another.
    pub operation: &'a str,
    /// Bytes the Hub holds of the closure, and its total.
    pub progress: &'a (dyn Fn(u64, u64) + Sync),
    /// PUTs in flight.
    pub streams: usize,
    /// Stops the publication before it finalizes.
    pub stop: Option<&'a Stop>,
    /// Asked once every object is held, just before finalizing, which cannot be taken back:
    /// false stops the publication there, as `stop` does.
    pub commit: &'a (dyn Fn() -> bool + Sync),
}

/// What the Hub committed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Published {
    /// `sha256:<hex>` of the tree, its identity in the repository.
    pub checkpoint: String,
    /// The repository already held it: nothing was pushed or finalized.
    pub converged: bool,
}

impl Catalog {
    pub fn publish(&self, p: &Publication) -> Result<Published> {
        if !self.presents() { return refuse(Code::CREDENTIAL_REQUIRED, "the run's capability does not let it upload"); }
        let endpoint = self.endpoint()?;
        let tree = p.store.tree(p.tree).map_err(|e| e.refusal())?;
        let mut leaves = BTreeSet::new();
        let mut headers = BTreeSet::new();
        for (_, object) in tree.variants() {
            let bytes = p.store.read(object).map_err(|e| e.refusal())?;
            let header = Header::from_cbor(&bytes).map_err(|e| tensorfs_core::err::Refusal { code: Code::UNKNOWN_FORMAT, detail: e.to_string() })?;
            leaves.extend(header.objects().cloned());
            headers.insert(object.clone());
        }
        leaves.extend(tree.repo_files().map(|(_, object)| object.clone()));
        let total = leaves.iter().chain(headers.iter()).map(|o| o.length).sum::<u64>() + p.tree.length;
        let committing = AtomicBool::new(false);
        let work = async {
            let (mut done, mut sent) = (0, 0);
            for stage in [leaves, headers, BTreeSet::from([p.tree.clone()])] {
                let endpoint = &endpoint;
                let mut uploads = stream::iter(stage).map(|object| async move {
                    let payload = || p.store.file(&object).map(Payload::File).map_err(|e| e.refusal());
                    endpoint.upload(&object, &payload).await.map(|sent| (sent, object.length))
                }).buffer_unordered(p.streams.clamp(1, 32));
                while let Some(uploaded) = uploads.next().await {
                    let (new, length) = uploaded?;
                    sent += usize::from(new);
                    done += length;
                    (p.progress)(done, total);
                }
            }
            if !(p.commit)() { return refuse(Code::TRANSFER_FAILED, "the upload was stopped before its ref PUT"); }
            committing.store(true, Ordering::Release);
            endpoint.update_repo(p.destination, |old| Ok(Some(with_tree(old, p.destination, p.tree, None)))).await?;
            Ok(Published { checkpoint: p.tree.id(), converged: sent == 0 })
        };
        block(async {
            let stopped = async {
                loop {
                    if !committing.load(Ordering::Acquire) && p.stop.is_some_and(Stop::is_cancelled) { break; }
                    tokio::time::sleep(Duration::from_millis(100)).await;
                }
            };
            tokio::select! {
                answer = work => answer,
                () = stopped => refuse(Code::TRANSFER_FAILED, "the upload was stopped; already uploaded objects remain resumable"),
            }
        })
    }
}
