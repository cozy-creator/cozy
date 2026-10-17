//! `cozy machine rent`: the catalog, and one paid order Calcifer places and keeps under its key.
use std::{num::NonZeroU64, process::ExitCode};

use cozy_proto::{
    calcifer::{Machine, MachineState, NewMachine, Quote, Rented},
    config::Hub,
    error::{Error, Kind},
};
use hub_client::types::{RentalSku, RentalSkuStatus};

use super::{FOLLOW, list::state_cell, route, usd};
use crate::{Context, context::hub_answer, output};

#[derive(clap::Args, Debug)]
pub struct Rent {
    /// A product from the catalog.
    sku: Option<String>,
    /// GPUs on the machine: any count the catalog lists.
    #[arg(long, requires = "sku", value_parser = clap::value_parser!(u32).range(1..))]
    gpus: Option<u32>,
    /// Disk in GB, at least (the Hub's default is 100).
    #[arg(long, requires = "sku", value_parser = clap::value_parser!(u32).range(1..))]
    disk_gb: Option<u32>,
    /// Return once the Hub accepts the order, without following the boot.
    #[arg(long, requires = "sku")]
    detach: bool,
    /// Resume this paid order instead of placing a new one.
    #[arg(long, requires = "sku")]
    idempotency_key: Option<String>,
}

pub async fn exec(ctx: &Context, rent: Rent) -> Result<ExitCode, Error> {
    let hub = ctx.hub()?;
    let Some(sku) = rent.sku else {
        return catalog(ctx, &hub).await;
    };
    let gpus = rent.gpus.unwrap_or(1);
    admit(ctx, &hub, &sku, gpus).await?;
    let minted = rent.idempotency_key.is_none();
    let key = rent.idempotency_key.unwrap_or_else(mint_key);
    let ask = NewMachine::Rent {
        sku: sku.clone(),
        hub: hub.origin.clone(),
        gpus: rent.gpus,
        disk_gb: rent.disk_gb,
    };
    let rented: Rented = match ctx
        .calcifer()
        .await?
        .post("/v1/machines", &ask, Some(&key))
        .await
    {
        // An answer lost on the way back may still have placed the order: only its key resumes it.
        Err(e) if minted && matches!(e.kind, Kind::Unavailable | Kind::Internal) => {
            return Err(e.remedy(format!(
                "the order may stand; resume it (never a second one) with \
                 `cozy machine rent {sku} --idempotency-key {key}`"
            )));
        }
        answer => answer?,
    };
    output::say(ctx, &price(&sku, &rented.quote));
    if let Some(asked) = rent.disk_gb.filter(|asked| rented.quote.disk_gb < *asked) {
        output::warn(
            ctx,
            &format!(
                "the Hub plans a {} GB disk, below the {asked} GB asked",
                rented.quote.disk_gb
            ),
        );
    }
    if rented.replay {
        output::say(ctx, "this order was placed already; following it");
    }
    let machine = match rent.detach {
        true => rented.machine.clone(),
        false => match follow(ctx, rented.machine.clone()).await? {
            Some(machine) => machine,
            None => return Ok(ExitCode::SUCCESS),
        },
    };
    if ctx.json {
        output::json(&Rented { machine, ..rented });
        return Ok(ExitCode::SUCCESS);
    }
    let q = &rented.quote;
    let mut record = output::Record::new()
        .field("machine", &machine.name)
        .field("state", state_cell(&machine))
        .field("gpus", gpus)
        .field(
            "$/hour",
            usd(q.gpu_usd_micros_per_hour + q.storage_usd_micros_per_hour),
        )
        .note(format!(
            "billing continues until `cozy machine end {}` confirms its release",
            machine.name
        ))
        .note("it ends itself after 15 minutes idle (no queued or running work)");
    if machine.state == MachineState::Booting {
        record = record.note("runs sent to it now wait until it is ready");
    }
    if gpus > 1 && gpus % 2 == 1 {
        record = record.note(
            "an odd GPU count leaves one GPU out of parallel runs, which shard at even counts",
        );
    }
    record.print(ctx);
    output::next(
        ctx,
        &[
            &format!("cozy run <org/package/function> --on {}", machine.name),
            &format!("cozy machine end {}", machine.name),
        ],
    );
    Ok(ExitCode::SUCCESS)
}

/// The price the order is held to: per hour and for the Hub's assumed hours and download.
fn price(sku: &str, q: &Quote) -> String {
    let mut line = format!(
        "{sku} with a {} GB disk: {}/h GPU + {}/h disk",
        q.disk_gb,
        usd(q.gpu_usd_micros_per_hour),
        usd(q.storage_usd_micros_per_hour)
    );
    if let Some(per_gb) = q.bandwidth_usd_micros_per_gb.filter(|p| *p > 0) {
        line += &format!(", traffic up to ${:.3}/GB", per_gb as f64 / 1e6);
    }
    line + &format!(
        "; about {} for {} h and a {} GB download",
        usd(q.expected_total_usd_micros),
        q.expected_hours,
        q.expected_download_gb
    )
}

/// Follows the boot until the machine is ready, saying the Hub's phase as it changes. Ctrl-C stops
/// watching only (`None`): the order goes on in Calcifer.
async fn follow(ctx: &Context, booting: Machine) -> Result<Option<Machine>, Error> {
    let calcifer = ctx.calcifer().await?;
    // A rental id never changes; the Hub may rename the machine while it boots.
    let path = route(booting.rental_id.as_deref().unwrap_or(&booting.name))?;
    let mut said = String::new();
    let mut machine = booting;
    loop {
        match machine.state {
            MachineState::Ready => return Ok(Some(machine)),
            MachineState::Booting => {
                let line = state_cell(&machine);
                if line != said {
                    output::say(ctx, &format!("{}: {line}", machine.name));
                    said = line;
                }
            }
            MachineState::Ending | MachineState::Ended => return Err(boot_failed(&machine)),
        }
        tokio::select! {
            () = tokio::time::sleep(FOLLOW) => {}
            _ = tokio::signal::ctrl_c() => {
                output::say(ctx, &format!(
                    "stopped watching {}: the order continues and bills once it boots; \
                     `cozy machine end {}` ends it", machine.name, machine.name));
                return Ok(None);
            }
        }
        machine = calcifer.get(&path).await?;
    }
}

fn boot_failed(m: &Machine) -> Error {
    let code = m.reason.clone().unwrap_or_else(|| "rental_failed".into());
    let error = Error::new(
        Kind::Unavailable,
        code.clone(),
        format!("rental {} did not come up: {code}", m.name),
    );
    if code == "provider_create_did_not_happen" {
        return error
            .remedy("Please try again later or rent a different GPU.")
            .next("cozy machine rent");
    }
    let sku = m.rental.as_ref().map_or("<sku>", |r| r.sku.as_str());
    error
        .next(format!("cozy machine show {}", m.name))
        .next(format!("cozy machine rent {sku}"))
}

fn mint_key() -> String {
    let mut bytes = [0u8; 8];
    getrandom::fill(&mut bytes).expect("the OS has randomness");
    format!("rent-{}", hex::encode(bytes))
}

/// An explicit ask is honoured or refused, never widened to another card or count; the refusal names
/// which absence it hit.
async fn admit(ctx: &Context, hub: &Hub, sku: &str, gpus: u32) -> Result<(), Error> {
    let count = NonZeroU64::new(u64::from(gpus));
    let status = hub_answer(
        ctx.hub_anonymous(hub)
            .get_rental_sku_status(sku, count, None)
            .await,
    )?;
    if status.known && status.offered {
        return Ok(());
    }
    Err(sku_refusal(sku, gpus, &status))
}

fn sku_refusal(sku: &str, gpus: u32, status: &RentalSkuStatus) -> Error {
    let (ask, retry) = match gpus {
        1 => (sku.to_string(), format!("cozy machine rent {sku}")),
        _ => (
            format!("{sku} ×{gpus}"),
            format!("cozy machine rent {sku} --gpus {gpus}"),
        ),
    };
    if !status.known {
        return Error::new(
            Kind::Validation,
            "sku_unknown",
            format!("the Hub sells no machine named {sku:?}; nothing was rented"),
        )
        .next("cozy machine rent");
    }
    let held = match status.unavailable_reason.as_deref() {
        Some("cooldown") => Some((
            "sku_cooldown",
            match &status.retry_after {
                Some(at) => {
                    format!("{ask} is held back after a recent boot failure; try again after {at}")
                }
                None => {
                    format!("{ask} is held back after a recent boot failure; try again shortly")
                }
            },
        )),
        Some("retry_in_progress") => Some((
            "sku_retry_in_progress",
            format!(
                "{ask} is held back while a retry after a recent boot failure runs; try again once it ends"
            ),
        )),
        _ => None,
    };
    if let Some((code, message)) = held {
        return Error::new(Kind::Capacity, code, message)
            .next(retry)
            .next("cozy machine rent");
    }
    let mut message =
        format!("no provider has {ask} right now; nothing else was rented in its place.");
    if let Some(at) = &status.last_seen_at {
        message += &format!(" It was last offered at {at}.");
    }
    let offered: Vec<String> = status
        .sku
        .iter()
        .flat_map(|s| &s.widths)
        .map(|w| w.accelerator_count.to_string())
        .collect();
    let error = match offered.first() {
        Some(first) => {
            message += &format!(" {sku} is offered now with {} GPU(s).", offered.join(", "));
            Error::new(Kind::Capacity, "sku_out_of_stock", message)
                .next(format!("cozy machine rent {sku} --gpus {first}"))
        }
        None => Error::new(Kind::Capacity, "sku_out_of_stock", message).next(retry),
    };
    error.next("cozy machine rent")
}

/// The catalog, cheapest expected total first: GPU, disk and traffic for the Hub's assumed hours and
/// download.
async fn catalog(ctx: &Context, hub: &Hub) -> Result<ExitCode, Error> {
    let mut skus: Vec<RentalSku> =
        hub_answer(ctx.hub_anonymous(hub).list_rental_skus(None).await)?.0;
    skus.retain(|sku| !sku.widths.is_empty());
    let total = |sku: &RentalSku| sku.widths[0].expected_price.total_usd_micros;
    skus.sort_by(|a, b| total(a).cmp(&total(b)).then_with(|| a.name.cmp(&b.name)));
    if ctx.json {
        output::json(&skus);
        return Ok(ExitCode::SUCCESS);
    }
    let mut table = output::Table::new(
        ctx,
        [
            "NAME", "GPU", "GPUS", "COMPUTE", "VRAM", "$/HOUR", "EXPECTED",
        ],
    );
    for sku in &skus {
        let width = &sku.widths[0];
        let counts: Vec<String> = sku
            .widths
            .iter()
            .map(|w| w.accelerator_count.to_string())
            .collect();
        let compute = match sku.compute_capability.as_str() {
            "" => "—".into(),
            cc => format!("sm_{}", cc.replace('.', "")),
        };
        table = table.row([
            sku.name.clone(),
            sku.accelerator_model.clone(),
            counts.join(", "),
            compute,
            if sku.vram_gb > 0 {
                format!("{} GB", sku.vram_gb)
            } else {
                "—".into()
            },
            usd(width.price_usd_micros_per_hour + width.storage_usd_micros_per_hour),
            usd(width.expected_price.total_usd_micros),
        ]);
    }
    table.print();
    match skus.first() {
        Some(first) => {
            let expected = &first.widths[0].expected_price;
            output::say(
                ctx,
                &format!(
                    "Prices are for the fewest GPUs. EXPECTED is {} h and a {} GB download: GPU, disk and traffic.",
                    expected.hours, expected.download_gb
                ),
            );
            output::next(ctx, &["cozy machine rent <name> [--gpus N]"]);
        }
        None => output::say(ctx, "No machine is for rent on this Hub right now."),
    }
    Ok(ExitCode::SUCCESS)
}
