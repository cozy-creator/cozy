//! `cozy model migrate-headers`: dev builds only (feature `migrate`, xs-053 A11; deleted once it
//! has run). Moves models from before D10 onto it, header-only (`tensorfs::migrate::d10`):
//! each release of lanes becomes one tree of `model.<lane>.cozytensors` at its next revision,
//! yanked releases too, and each other old checkpoint a tree of its own; run data a lane lacked
//! comes from its family's pinned upstream. Chunks never move: only the headers, trees and small
//! blobs the Hub lacks are sent, then the repo document is replaced under `If-Match`. Running it
//! again changes nothing. Each model's document as it was is kept in the audit directory: the
//! rollback puts it back in the Hub's storage, since the Hub never takes a release back to lanes.
//! `--local` migrates a TensorFS 0.12 store in place; stop its machine.
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::sync::Arc;

use protocol::error::{Error, Kind};
use serde_json::{json, Value};
use tensorfs::failure::Result as Done;
use tensorfs::fetch::{policy::is_loopback_origin, token::bearer, Client, Endpoint, SourcePolicy};
use tensorfs::migrate::d10::{self, Migrator, UPSTREAMS};
use tensorfs::store::Store;

use crate::context::hub_error;
use crate::{output, Context};

#[derive(clap::Args, Debug)]
pub(crate) struct Args {
    /// Models (org/name) at the Hub.
    #[arg(required_unless_present = "local", conflicts_with = "local")]
    models: Vec<String>,
    /// Instead: a stopped TensorFS 0.12 store, in place. Keep a backup for rollback.
    #[arg(long, value_name = "STORE")]
    local: Option<PathBuf>,
    /// Where each model's report, and its document as it was, are written.
    #[arg(long, value_name = "DIR", default_value = "migrate-headers-audit")]
    audit: PathBuf,
}

fn failed(detail: impl Into<String>) -> Error {
    Error::new(Kind::Conflict, "migration_failed", detail)
}

async fn blocking<T: Send + 'static>(f: impl FnOnce() -> Done<T> + Send + 'static) -> Result<T, Error> {
    let done = tokio::task::spawn_blocking(f).await.map_err(|e| failed(e.to_string()))?;
    done.map_err(|f| failed(f.detail))
}

fn write(path: &Path, doc: &Value) -> Result<(), Error> {
    let io = |e: std::io::Error| Error::new(Kind::Internal, "audit_write_failed", format!("{}: {e}", path.display()));
    std::fs::create_dir_all(path.parent().unwrap_or(Path::new("."))).map_err(io)?;
    std::fs::write(path, serde_json::to_vec_pretty(doc).expect("JSON")).map_err(io)
}

/// The provider credentials the cozy config holds (`huggingface_token`, `civitai_token`, or
/// their env names): pins of gated repos list and read through them.
fn access(ctx: &Context) -> tensorfs::import::Access {
    let bearer = |secret: &Option<protocol::config::Secret>| {
        secret.as_ref().map_or_else(String::new, |s| format!("bearer {}", s.expose().trim()))
    };
    tensorfs::import::Access {
        huggingface_token: bearer(&ctx.config.huggingface_token),
        civitai_token: bearer(&ctx.config.civitai_token),
        ..tensorfs::import::Access::default()
    }
}

pub(crate) async fn exec(ctx: &Context, args: Args) -> Result<ExitCode, Error> {
    let access = access(ctx);
    if let Some(store) = args.local {
        let report = blocking(move || d10::store(&store, &mut Migrator::new(access, UPSTREAMS))).await?;
        write(&args.audit.join("local.json"), &report)?;
        output::json(&report);
        return Ok(ExitCode::SUCCESS);
    }
    let hub = ctx.hub()?;
    let (session, client) = (ctx.hub_session(&hub)?, ctx.hub_signed_in(&hub)?);
    let (mut reports, mut ok) = (vec![], true);
    std::fs::create_dir_all(&args.audit).map_err(|e| failed(format!("{}: {e}", args.audit.display())))?;
    for model in args.models {
        let (token, origin, name) = (session.bearer(&client).await.map_err(hub_error)?.token, hub.origin.to_string(), model.clone());
        let file = |kind: &str| args.audit.join(format!("{}.{kind}json", model.replace('/', "__")));
        let before = file("before.");
        let access = access.clone();
        let done = blocking(move || {
            let local = is_loopback_origin(&origin);
            let http = Client::new(SourcePolicy { any_host: true, allow_local: local, max_redirects: 5, ..Default::default() });
            let endpoint = Arc::new(Endpoint::new(&http, &origin, Some(bearer(token)))?);
            let scratch = tempfile::tempdir()?;
            let store = Store::open(scratch.path())?;
            let done = d10::migrate(&endpoint, &http, &name, &store, &mut Migrator::new(access, UPSTREAMS))?;
            Ok((done.report, done.before))
        });
        let report = match done.await {
            Ok((report, doc)) => {
                if !before.exists() {
                    std::fs::write(&before, doc).map_err(|e| failed(format!("{}: {e}", before.display())))?;
                }
                report
            }
            Err(e) => {
                ok = false;
                json!({"model": model, "error": e.message})
            }
        };
        write(&file(""), &report)?;
        output::say(ctx, &format!("{model}: {}", if report.get("error").is_some() { "failed" } else { "migrated" }));
        reports.push(report);
    }
    let summary = json!({"models": reports, "audit": args.audit});
    match ctx.json {
        true => output::ndjson(&summary),
        false => output::json(&summary),
    }
    Ok(if ok { ExitCode::SUCCESS } else { ExitCode::FAILURE })
}
