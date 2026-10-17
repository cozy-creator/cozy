//! `cozy machine end`, `keepalive` and `update`: asks Calcifer makes of one machine.
use std::process::ExitCode;

use cozy_proto::{
    calcifer::{
        self, Ended, Keepalive, Machine, MachineKind, MachineList, MachineState, On, SubmitRun,
    },
    error::{Error, Kind},
    machine as v1,
};
use prost::Message as _;

use super::{FOLLOW, clock, machine, now_ms, route, usd};
use crate::context::hub_answer;
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

/// Updates a machine to its Hub's target software: a numbered run of kind update, which waits for the
/// work the machine accepted and is followed like any run.
pub async fn update(ctx: &Context, name: &str) -> Result<ExitCode, Error> {
    let m = machine(ctx, name).await?;
    let Some(status) = &m.status else {
        return Err(Error::new(
            Kind::Unavailable,
            "machine_unanswered",
            format!("{} has not reported the software it runs", m.name),
        )
        .next(format!("cozy machine show {}", m.name)));
    };
    let hub = match m.hub.as_str() {
        "" => ctx.hub()?,
        origin => ctx.config.resolve(origin)?,
    };
    let target = hub_answer(ctx.hub_anonymous(&hub).get_software().await)?;
    // An unpinned target older than the machine's software is no reason to go back.
    if (status.runtime == target.runtime && status.tensorfs == target.tensorfs)
        || (!target.pinned && newer(&status.runtime, &target.runtime))
    {
        return up_to_date(ctx, &m, &target.runtime);
    }
    let payload = serde_json::json!({"runtime": target.runtime, "tensorfs": target.tensorfs, "agent": "bundled"});
    let spec = v1::RunSpec {
        kind: v1::RunKind::Update as i32,
        payload: serde_json::to_vec(&payload).expect("JSON serializes"),
        ..Default::default()
    };
    let submit = SubmitRun {
        on: On::Machine(m.name.clone()),
        hub: hub.origin,
        kind: calcifer::Kind::Update,
        target: format!(
            "cozy-runtime {}, tensorfs {}",
            target.runtime, target.tensorfs
        ),
        spec: spec.encode_to_vec(),
        inputs: vec![],
        out: None,
        warnings: vec![],
    };
    let submitted = crate::run::submit(ctx, submit, None).await?;
    let run = crate::run::follow(ctx, submitted.run, false).await?;
    Ok(crate::run::exit_code(&run))
}

fn up_to_date(ctx: &Context, m: &Machine, target: &str) -> Result<ExitCode, Error> {
    let runtime = m.status.as_ref().map_or("", |s| s.runtime.as_str());
    if ctx.json {
        output::json(
            &serde_json::json!({"machine": m.name, "changed": false, "runtime": runtime, "target": target}),
        );
    } else {
        output::say(
            ctx,
            &format!(
                "{} runs cozy-runtime {runtime}; the Hub's target is {target}: nothing to update",
                m.name
            ),
        );
    }
    Ok(ExitCode::SUCCESS)
}

/// Whether release `a` is newer than `b`, by their leading dotted numbers.
fn newer(a: &str, b: &str) -> bool {
    let parts = |v: &str| -> Vec<u64> { v.split('.').map_while(|p| p.parse().ok()).collect() };
    parts(a) > parts(b)
}
