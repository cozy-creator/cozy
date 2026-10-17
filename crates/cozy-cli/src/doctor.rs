//! `cozy doctor`: one machine's driver, GPUs, toolchain and kernels, as its Status reports them.
use std::process::ExitCode;

use cozy_proto::{
    calcifer::Machine,
    error::{Error, Kind},
};

use crate::{
    Context,
    machine::{route, status_fields},
    output,
};

#[derive(clap::Args, Debug)]
pub struct Args {
    /// The machine to check.
    #[arg(long, default_value = "local")]
    on: String,
}

pub async fn exec(ctx: &Context, args: Args) -> Result<ExitCode, Error> {
    let m: Machine = match ctx.calcifer().await?.get(&route(&args.on)?).await {
        Err(e) if e.kind == Kind::NotFound && args.on == "local" => {
            return Err(Error::new(
                Kind::NotFound,
                "no_local_machine",
                "this computer is not a machine; check another with --on",
            )
            .next("cozy doctor --on <machine>")
            .next("cozy machine list"));
        }
        answer => answer?,
    };
    let Some(status) = &m.status else {
        let why = m
            .unreachable
            .as_deref()
            .unwrap_or("it has not reported its status yet");
        return Err(Error::new(
            Kind::Unavailable,
            "machine_unanswered",
            format!("{} cannot be checked: {why}", m.name),
        )
        .next(format!("cozy machine show {}", m.name)));
    };
    if ctx.json {
        output::json(&m);
        return Ok(ExitCode::SUCCESS);
    }
    let mut record = output::Record::new().field("machine", &m.name);
    for (key, value) in status_fields(status) {
        record = record.field(&key, value);
    }
    if let Some(why) = &m.unreachable {
        record = record.note(format!(
            "{} did not answer just now ({why}); this is its last report",
            m.name
        ));
    }
    record
        .note("this machine does not report its driver floor, toolchain or kernel verification yet")
        .print(ctx);
    Ok(ExitCode::SUCCESS)
}
