//! `tfs ensure <root> <ref> --hub <base>`: make one model resident (proto-063 M2a).
//!
//! STDOUT is JSON lines and nothing else: `ensure.progress` samples, then exactly one
//! `ensure.result` (exit 0) or `ensure.refused` (exit 1). Usage errors exit 2.

use std::io::Write;
use std::path::Path;
use std::process::ExitCode;
use std::sync::Mutex;

use tensorfs_core::canon::{self, Value};
use tensorfs_core::ensure::{self, Ensured, Event, Request};
use tensorfs_core::err::Refusal;
use tensorfs_core::store::Store;
use tensorfs_core::transport;

use crate::{flag, flag_all, flag_num, Flags};

fn read(path: &str) -> Result<Vec<u8>, Refusal> {
    std::fs::read(path).map_err(|e| Refusal {
        code: tensorfs_core::err::Code::IO_FAILED,
        detail: format!("read {path}: {e}"),
    })
}

fn quoted(text: &str) -> String {
    String::from_utf8(canon::write(&Value::str(text))).unwrap()
}

fn say(out: &Mutex<std::io::Stdout>, line: String) {
    let mut out = out.lock().unwrap();
    // A closed reader is not this download's business; the bytes are still wanted.
    let _ = writeln!(out, "{line}").and_then(|()| out.flush());
}

fn progress(event: &Event) -> String {
    format!(
        "{{\"event\":\"ensure.progress\",\"model\":{},\"step\":{},\"phase\":\"{}\",\"bytes_done\":{},\"bytes_total\":{},\"rate\":{:.1},\"attempt\":{}}}",
        quoted(&event.model),
        quoted(&event.step),
        event.phase.as_str(),
        event.bytes_done,
        event.bytes_total,
        event.rate,
        event.attempt,
    )
}

fn result(r: &Ensured) -> String {
    format!(
        "{{\"event\":\"ensure.result\",\"model\":{},\"release\":{},\"lane\":{},\"scope\":{},\"manifest\":\"{}\",\"manifest_length\":{},\"bytes_total\":{},\"bytes_held\":{},\"bytes_fetched\":{},\"bytes_cached\":{},\"cache_written_bytes\":{},\"collected_bytes\":{},\"attempts\":{},\"joined\":{},\"seconds\":{:.3}}}",
        quoted(&r.model),
        quoted(&r.release),
        quoted(&r.lane),
        quoted(&r.scope),
        r.manifest.id(),
        r.manifest.length,
        r.bytes_total,
        r.bytes_held,
        r.bytes_fetched,
        r.bytes_cached,
        r.cache_written_bytes,
        r.collected_bytes,
        r.attempts,
        r.joined,
        r.seconds,
    )
}

fn refused(refusal: &Refusal, model: &str, step: &str) -> String {
    format!(
        "{{\"event\":\"ensure.refused\",\"code\":\"{}\",\"resumable\":{},\"detail\":{},\"model\":{},\"step\":{}}}",
        refusal.code.as_str(),
        ensure::resumable(refusal.code),
        quoted(&refusal.detail),
        quoted(model),
        quoted(step),
    )
}

pub fn cmd_ensure(root: &Path, refspec: &str, flags: &Flags) -> ExitCode {
    let Some(hub) = flag(flags, "hub") else {
        eprintln!("--hub <base-url> is required");
        return ExitCode::from(2);
    };
    let out = Mutex::new(std::io::stdout());
    let model = refspec.split('@').next().unwrap_or_default();
    let step = flag(flags, "step").unwrap_or("");
    let answer = |outcome: Result<Ensured, Refusal>| match outcome {
        Ok(ensured) => {
            say(&out, result(&ensured));
            ExitCode::SUCCESS
        }
        Err(refusal) => {
            say(&out, refused(&refusal, model, step));
            ExitCode::FAILURE
        }
    };
    let setup = || -> Result<_, Refusal> {
        if let Some(path) = flag(flags, "ca-file") {
            transport::trust_roots(&read(path)?)?;
        }
        // The credential reaches the child as a file or TFS_CREDENTIAL, never argv.
        let spec = match flag(flags, "credential-file") {
            Some(path) => String::from_utf8_lossy(&read(path)?).into_owned(),
            None => std::env::var("TFS_CREDENTIAL").unwrap_or_default(),
        };
        let credential =
            transport::credential_from_spec(spec.trim(), vec![transport::base_host(hub)?])?;
        Ok((Store::open(root)?, credential))
    };
    let (store, credential) = match setup() {
        Ok(setup) => setup,
        Err(refusal) => return answer(Err(refusal)),
    };
    let policy = crate::fetch::policy_of(flags);
    let keep: Vec<String> = flag_all(flags, "keep")
        .into_iter()
        .map(str::to_string)
        .collect();
    let on_event = |event: &Event| say(&out, progress(event));
    let mut request = Request::new(&store, hub, refspec, &credential, &policy);
    request.lane = flag(flags, "lane").unwrap_or("");
    request.step = step;
    request.keep = &keep;
    request.streams = flag_num(flags, "streams", transport::PULL_STREAMS);
    request.sample_seconds = flag_num(flags, "sample-seconds", transport::SAMPLE_SECONDS);
    request.on_event = Some(&on_event);
    answer(ensure::ensure(&request))
}

/// `tfs pressure <root> [--keep sha256:<m>]...`: one pressure pass (see `ensure::relieve`),
/// reported as one JSON line.
pub fn cmd_pressure(root: &Path, flags: &Flags) -> ExitCode {
    let keep: Vec<String> = flag_all(flags, "keep")
        .into_iter()
        .map(str::to_string)
        .collect();
    let relief = match Store::open(root).and_then(|store| ensure::relieve(&store, &keep)) {
        Ok(relief) => relief,
        Err(refusal) => {
            println!(
                "{{\"event\":\"pressure.refused\",\"code\":\"{}\",\"detail\":{}}}",
                refusal.code.as_str(),
                quoted(&refusal.detail)
            );
            return ExitCode::FAILURE;
        }
    };
    println!(
        "{{\"event\":\"pressure\",\"pressure\":{},\"collected_bytes\":{},\"capacity_bytes\":{},\"available_bytes\":{},\"unable\":{}}}",
        relief.pressure,
        relief.collected_bytes,
        relief.capacity_bytes,
        relief.available_bytes,
        relief.unable.as_deref().map_or("null".to_string(), quoted),
    );
    ExitCode::SUCCESS
}
