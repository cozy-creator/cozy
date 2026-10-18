//! `tfs fetch <root> <ref> --hub <base>`: a checkpoint made whole on the Store's one Fetcher,
//! as TensorD's downloads are. The credential is `--credential-file <path>` or `TFS_CREDENTIAL`
//! (`bearer <token>`, `worker <id> <token>`, or empty for anonymous), never argv.

use std::path::Path;
use std::process::ExitCode;
use std::sync::Arc;
use std::time::Duration;

use tensorfs_core::fetch_v1::{self as fetch, FetchPlan, Fetcher, Plain, Priority};
use tensorfs_core::store::Store;
use tensorfs_core::transport::{self, PullCancellation, SourcePolicy};

use super::{bail, flag, flag_num, flag_on, Flags};

pub fn cmd_fetch(root: &Path, refspec: &str, flags: &Flags) -> ExitCode {
    let Some(hub) = flag(flags, "hub") else {
        eprintln!("--hub <base-url> is required");
        return ExitCode::from(2);
    };
    let spec = match flag(flags, "credential-file") {
        Some(path) => ok!(std::fs::read_to_string(path).map_err(|e| tensorfs_core::err::Refusal {
            code: tensorfs_core::err::Code::IO_FAILED,
            detail: format!("read {path}: {e}"),
        })),
        None => std::env::var("TFS_CREDENTIAL").unwrap_or_default(),
    };
    let credential = ok!(transport::credential_from_spec(&spec, vec![ok!(transport::base_host(hub))]));
    let allowed_hosts = flag(flags, "allow-hosts")
        .map(|hosts| hosts.split(',').map(str::trim).filter(|h| !h.is_empty()).map(str::to_string).collect())
        .unwrap_or_default();
    let policy = SourcePolicy {
        allowed_hosts,
        allow_local: flag_on(flags, "allow-local"),
        max_redirects: flag_num(flags, "max-redirects", 0u32),
        ..Default::default()
    };
    let store = ok!(Store::open(root));
    let streams = flag_num(flags, "streams", transport::PULL_STREAMS);
    let sample = Duration::from_secs_f64(flag_num(flags, "sample-seconds", transport::SAMPLE_SECONDS));
    let fetcher = ok!(Fetcher::new(&store, streams, sample));
    let access = Arc::new(Plain { credential, policy });
    let stop = PullCancellation::default();
    let lane = flag(flags, "lane").unwrap_or("");
    let (origin, closure) = ok!(fetch::resolve(&fetcher, hub, refspec, lane, &*access, &stop));
    let (plan, _) = ok!(FetchPlan::of(&store, "tfs-fetch", &closure.manifest, &closure.objects));
    let (hub_bytes, _) = ok!(fetch::whole(&fetcher, &origin, access, Priority::Now, &stop, &|_, _| {}));
    println!("model        {}@{} lane {}", closure.model, closure.release, closure.lane);
    println!("manifest     {}", closure.manifest.id());
    println!("held         {} object(s), {} B — NOT fetched", plan.held.len(), plan.held_bytes());
    println!("fetched      {} object(s), {hub_bytes} B, {streams} stream(s)", plan.wanted.len());
    println!("complete     every declared object verifiably local");
    ExitCode::SUCCESS
}
