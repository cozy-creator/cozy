//! `cozy machine rent`: the catalog, and one paid order Calcifer places and keeps under its key.
use std::{num::NonZeroU64, process::ExitCode};

use cozy_proto::{
    config::Hub,
    error::{Error, Kind},
};
use hub_client::types::{RentalSku, RentalSkuStatus};

use super::usd;
use crate::{Context, context::hub_answer, output};

// The order itself (disk, detach, key) goes to Calcifer's rentals once they land.
#[allow(dead_code)]
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
    admit(ctx, &hub, &sku, rent.gpus.unwrap_or(1)).await?;
    Err(crate::not_ported("machine rent <sku>"))
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
