//! `cozy machine end`, `keepalive` and `update`: asks Calcifer makes of one machine.
use std::process::ExitCode;

use cozy_proto::{
    calcifer::{Ended, Keepalive, MachineKind, MachineList, MachineState},
    error::{Error, Kind},
};

use super::{FOLLOW, clock, now_ms, route, usd};
use crate::{Context, output};

/// Ends a rental, returning once its Hub affirms the pod gone, or removes an added machine.
pub async fn end(ctx: &Context, name: &str) -> Result<ExitCode, Error> {
    if name == "local" {
        return Err(Error::new(
            Kind::Validation,
            "machine_local",
            "`local` is this computer's machine; `end` ends rentals and removes added machines",
        )
        .next("cozy machine disable"));
    }
    let calcifer = ctx.calcifer().await?;
    let path = route(name)?;
    let Ended {
        mut machine,
        changed,
    } = calcifer.delete(&path).await?;
    // Follow the row by its rental id: the name is the Hub's and an alias may be what was typed.
    let path = route(machine.rental_id.as_deref().unwrap_or(&machine.name))?;
    if machine.state == MachineState::Ending {
        output::say(ctx, &format!("Shutting down {}…", machine.name));
    }
    while machine.state == MachineState::Ending {
        if machine.reason.as_deref() == Some("release_unconfirmed") {
            return Err(Error::new(
                Kind::Conflict,
                "release_unconfirmed",
                format!(
                    "the Hub has not confirmed {}'s pod destroyed, and its missing record is not proof",
                    machine.name
                ),
            )
            .remedy("Calcifer keeps the record and keeps asking; the pod may still be billing")
            .next(format!("cozy machine show {}", machine.name)));
        }
        tokio::select! {
            () = tokio::time::sleep(FOLLOW) => {}
            _ = tokio::signal::ctrl_c() => {
                return Err(Error::new(
                    Kind::Unavailable,
                    "end_unconfirmed",
                    format!("stopped watching; the release of {} continues in Calcifer", machine.name),
                )
                .next(format!("cozy machine show {}", machine.name)));
            }
        }
        machine = calcifer.get(&path).await?;
    }
    if ctx.json {
        output::json(&Ended { machine, changed });
        return Ok(ExitCode::SUCCESS);
    }
    let said = match (machine.kind, changed) {
        (MachineKind::Added, true) => format!(
            "{} is no longer one of this computer's machines.",
            machine.name
        ),
        (MachineKind::Added, false) => format!("{} was already removed.", machine.name),
        (_, true) => format!(
            "{} shut down. Temporary pod files are gone. Local outputs and uploaded checkpoints remain.",
            machine.name
        ),
        (_, false) => format!("{} is already shut down.", machine.name),
    };
    println!("{said}");
    if let Ok(list) = calcifer.get::<MachineList>("/v1/machines").await {
        let live: Vec<_> = list
            .machines
            .iter()
            .filter(|m| m.state != MachineState::Ended)
            .filter_map(|m| m.rental.as_ref())
            .collect();
        let burn: i64 = live.iter().map(|r| r.usd_micros_per_hour).sum();
        println!("Rentals running: {} · {}/hour now", live.len(), usd(burn));
    }
    output::next(ctx, &["cozy machine list"]);
    Ok(ExitCode::SUCCESS)
}

/// Resets a rental's idle clock once; it answers only after the machine acknowledged.
pub async fn keepalive(ctx: &Context, name: &str) -> Result<ExitCode, Error> {
    let path = format!("{}/keepalive", route(name)?);
    let answer: Keepalive = ctx
        .calcifer()
        .await?
        .post(&path, &serde_json::json!({}), None)
        .await?;
    if ctx.json {
        output::json(&answer);
        return Ok(ExitCode::SUCCESS);
    }
    let ends = match answer.idle_deadline_ms {
        Some(at) => format!("in {} unless work arrives", clock(at - now_ms())),
        None => "after its jobs".into(),
    };
    output::Record::new()
        .field("machine", name)
        .field("ends", ends)
        .print(ctx);
    Ok(ExitCode::SUCCESS)
}
