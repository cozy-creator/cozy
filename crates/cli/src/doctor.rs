//! `cozy doctor`: this computer's tools, then one machine's driver, GPUs, toolchain and kernels,
//! as its Status reports them.
use std::process::ExitCode;

use platform::tools::Side;
use protocol::{
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
    /// The machine to check; without it, this computer's machine when it is one.
    #[arg(long)]
    on: Option<String>,
}

/// `cozy doctor`: this computer's tools (found, or what each one's absence costs and what installs
/// it), then a machine as its Status reports it: `--on`'s, else this computer's when enabled.
pub async fn exec(ctx: &Context, args: Args) -> Result<ExitCode, Error> {
    let tools = crate::tools::rows(ctx, Side::Computer);
    let named = args.on.is_some();
    let on = args.on.unwrap_or_else(|| "local".into());
    let machine: Option<Machine> = match ctx.calcifer().await?.get(&route(&on)?).await {
        Err(e) if e.kind == Kind::NotFound && !named => None,
        answer => Some(answer?),
    };
    if ctx.json {
        output::json(&serde_json::json!({ "tools": tools, "machine": machine }));
        return Ok(ExitCode::SUCCESS);
    }
    let mut table = output::Table::new(ctx, ["THIS COMPUTER", "STATE", "PATH OR INSTALL"]);
    for row in &tools {
        table = table.row(crate::tools::cells(row));
    }
    table.print();
    let Some(m) = machine else {
        output::say(ctx, "this computer is not a machine");
        output::next(ctx, &["cozy machine enable", "cozy doctor --on <machine>"]);
        return Ok(ExitCode::SUCCESS);
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
    if status.toolchain.is_none() {
        record = record
            .note("this machine is older than this cozy: it reports no driver floor, toolchain or kernels");
    }
    record.print(ctx);
    Ok(ExitCode::SUCCESS)
}
