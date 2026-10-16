//! `cozy credits [history|buy]`: the account's prepaid rental credit at the command's Hub, in
//! micros of USD. A Hub that does not meter the account says so, and nothing is bought there.
use std::{process::ExitCode, time::Duration};

use clap::Subcommand;
use cozy_proto::{
    config::Hub,
    error::{Error, Kind},
};
use hub_client::types;
use serde_json::json;

use crate::{Context, context::hub_answer, output};

const CURRENCY: &str = "USD";
/// How often a purchase is read again while its payment lands: a sampling period.
const SAMPLE: Duration = Duration::from_secs(2);

#[derive(clap::Args, Debug)]
pub struct Args {
    #[command(subcommand)]
    command: Option<Command>,
}

#[derive(Subcommand, Debug)]
enum Command {
    /// Credit moves, newest first.
    History {
        /// Continue from this page's cursor.
        #[arg(long)]
        cursor: Option<String>,
    },
    /// Buy credit with a card.
    Buy {
        /// Dollars, such as 10 or 25.50.
        usd: String,
        /// Print the payment page instead of opening a browser.
        #[arg(long)]
        no_browser: bool,
        /// Answer the payment page at once instead of waiting for the payment.
        #[arg(long)]
        no_wait: bool,
    },
}

pub async fn exec(ctx: &Context, args: Args) -> Result<ExitCode, Error> {
    let hub = ctx.hub()?;
    match args.command {
        None => balance(ctx, &hub).await,
        Some(Command::History { cursor }) => history(ctx, &hub, cursor.as_deref()).await,
        Some(Command::Buy {
            usd,
            no_browser,
            no_wait,
        }) => buy(ctx, &hub, &usd, !no_browser, !no_wait).await,
    }
}

async fn balance(ctx: &Context, hub: &Hub) -> Result<ExitCode, Error> {
    let client = ctx.hub_signed_in(hub)?;
    let Some(credit) = hub_answer(client.get_credit().await)?.credit else {
        if ctx.json {
            output::json(&json!({ "hub": hub, "billing": "unmetered" }));
        } else {
            output::Record::new()
                .field("hub", hub)
                .field("billing", "unmetered")
                .note("This Hub does not meter your account: rentals never stop for credit.")
                .print(ctx);
        }
        return Ok(ExitCode::SUCCESS);
    };
    let account = hub_answer(client.get_billing_account().await)?;
    let balance = account.balances.iter().find(|b| b.currency == CURRENCY);
    let amount = |f: fn(&types::BillingBalance) -> &String| balance.map_or(0, |b| micros(f(b)));
    let (total, held, owed) = (
        amount(|b| &b.balance_amount),
        amount(|b| &b.held_amount),
        amount(|b| &b.owed_amount),
    );
    let low = credit.available_usd_micros < credit.warning_usd_micros;
    if ctx.json {
        output::json(&json!({
            "hub": hub, "billing": "metered", "balance_micros": total, "held_micros": held,
            "owed_micros": owed, "available_micros": credit.available_usd_micros,
            "warning_micros": credit.warning_usd_micros, "floor_micros": credit.floor_usd_micros,
            "burn_micros_per_hour": credit.burn_usd_micros_per_hour,
            "depletes_at": credit.depletes_at, "low": low,
        }));
        return Ok(ExitCode::SUCCESS);
    }
    let mut record = output::Record::new()
        .field("hub", hub)
        .field("balance", usd(total))
        .field("held", usd(held))
        .field("available", usd(credit.available_usd_micros));
    if owed != 0 {
        record = record.field("owed", usd(owed));
    }
    if credit.burn_usd_micros_per_hour > 0 {
        record = record.field(
            "burn",
            format!("{}/h", usd(credit.burn_usd_micros_per_hour)),
        );
    }
    if let Some(at) = &credit.depletes_at {
        record = record.field("runs out", minute(at));
    }
    if low {
        record = record.note(format!(
            "Credit is low. Running rentals continue until it passes {}, then they stop, even mid-run.",
            usd(credit.floor_usd_micros)
        ));
    }
    record.print(ctx);
    if low {
        output::next(ctx, &["cozy credits buy <usd>"]);
    }
    Ok(ExitCode::SUCCESS)
}

async fn history(ctx: &Context, hub: &Hub, cursor: Option<&str>) -> Result<ExitCode, Error> {
    let client = ctx.hub_signed_in(hub)?;
    let answer = client.list_balance_transactions(Some(CURRENCY), cursor, Some(50));
    let page = hub_answer(answer.await)?;
    let more = page
        .next_cursor
        .map(|c| format!("cozy credits history --cursor {c}"));
    if ctx.json {
        output::json(&json!({ "hub": hub, "moves": page.data, "more": more }));
        return Ok(ExitCode::SUCCESS);
    }
    let mut table = output::Table::new(ctx, ["WHEN", "TYPE", "AMOUNT", "SOURCE"]);
    for m in &page.data {
        table = table.row([
            minute(&m.created_at),
            m.type_.clone(),
            usd(micros(&m.amount)),
            m.source.clone(),
        ]);
    }
    if let Some(more) = more {
        table = table.note(format!("more: {more}"));
    }
    table.print();
    Ok(ExitCode::SUCCESS)
}

/// Opens an order for `text` dollars of credit and its hosted payment page, then (with `wait`)
/// follows the order until it is paid or ends. Ctrl-C stops waiting; a payment still lands.
async fn buy(
    ctx: &Context,
    hub: &Hub,
    text: &str,
    browser: bool,
    wait: bool,
) -> Result<ExitCode, Error> {
    let amount = parse_usd(text)?;
    let client = ctx.hub_signed_in(hub)?;
    let deposit = hub_answer(client.get_credit_deposit().await)?;
    let (min, max) = (micros(&deposit.min_amount), micros(&deposit.max_amount));
    if !(min..=max).contains(&amount) {
        return Err(Error::new(
            Kind::Validation,
            "credits_amount_out_of_range",
            format!("one purchase is {} to {}", usd(min), usd(max)),
        ));
    }
    let lines = vec![types::BillingOrderLineParams {
        amount: Some(amount.to_string()),
        price_id: Some(deposit.price_id),
        quantity: None,
    }];
    let order = types::BillingCreateOrderRequest {
        lines,
        cart: None,
        expected_total: None,
        pay_url: None,
        payment: None,
        reusable: None,
    };
    let key = idempotency_key()?;
    let order = hub_answer(
        client
            .create_order(Some(&key.parse().map_err(invalid)?), &order)
            .await,
    )?;
    if !order
        .payment_options
        .iter()
        .any(|o| o.accepts.iter().any(|a| a == "redirect"))
    {
        return Err(Error::new(
            Kind::Unavailable,
            "credits_no_hosted_checkout",
            format!("{hub} offers no hosted payment page for credit"),
        ));
    }
    let pay = types::BillingPayOrderRequest {
        expected_total: Some(order.total.clone()),
        payment: Some(types::BillingOrderPaymentParams {
            return_url: Some(format!("{}/credits/return", hub.origin)),
            ..Default::default()
        }),
        reusable: None,
    };
    let key = idempotency_key()?.parse().map_err(invalid)?;
    let paying = hub_answer(client.pay_order(&order.id, &key, &pay).await)?;
    let page = paying
        .payment
        .next_action
        .filter(|next| next.type_ == "redirect_to_url")
        .and_then(|next| next.url)
        .ok_or_else(|| {
            Error::new(
                Kind::Internal,
                "credits_no_payment_page",
                format!("{hub} answered order {} with no payment page", order.id),
            )
        })?;
    let opened = browser && open(&page);
    if !wait {
        if ctx.json {
            output::json(&json!({
                "status": "awaiting payment", "order": order.id, "amount_micros": amount,
                "payment_page": page, "expires_at": order.expires_at,
            }));
        } else {
            output::Record::new()
                .field("status", "awaiting payment")
                .field("amount", usd(amount))
                .field("pay at", &page)
                .print(ctx);
            output::next(ctx, &["cozy credits"]);
        }
        return Ok(ExitCode::SUCCESS);
    }
    output::say(
        ctx,
        &match opened {
            true => {
                format!("Opened the payment page in your browser; if it did not open:\n  {page}")
            }
            false => format!("Pay here:\n  {page}"),
        },
    );
    output::say(
        ctx,
        "Waiting for the payment (Ctrl-C stops waiting; a payment still lands)…",
    );
    loop {
        tokio::select! {
            _ = tokio::signal::ctrl_c() => return Ok(ExitCode::SUCCESS),
            () = tokio::time::sleep(SAMPLE) => {}
        }
        let now = match client.get_order(&order.id).await {
            Ok(now) => now.into_inner(),
            // One unanswered read is not the purchase's end; its own expiry is.
            Err(_) => continue,
        };
        match now.status.as_str() {
            "complete" => return balance(ctx, hub).await,
            "open" | "processing" => {}
            ended => {
                let why = now.payment.last_payment_error.map(|e| e.message);
                return Err(Error::new(
                    Kind::Conflict,
                    format!("credits_order_{ended}"),
                    format!(
                        "order {} {ended}: {}",
                        order.id,
                        why.as_deref().unwrap_or("the payment did not complete")
                    ),
                )
                .next(format!("cozy credits buy {text}")));
            }
        }
    }
}

fn invalid(error: impl std::fmt::Display) -> Error {
    Error::new(Kind::Internal, "idempotency_key", error.to_string())
}

fn idempotency_key() -> Result<String, Error> {
    let mut bytes = [0; 16];
    getrandom::fill(&mut bytes).map_err(invalid)?;
    Ok(hex::encode(bytes))
}

/// Asks this OS to open `url` in the user's browser; false when it cannot.
fn open(url: &str) -> bool {
    let mut command = match std::env::consts::OS {
        "macos" => std::process::Command::new("open"),
        "windows" => {
            let mut command = std::process::Command::new("rundll32");
            command.arg("url.dll,FileProtocolHandler");
            command
        }
        _ => std::process::Command::new("xdg-open"),
    };
    command
        .arg(url)
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .spawn()
        .is_ok()
}

/// An RFC 3339 instant to the minute, as the Hub wrote it: `2026-10-16 22:07Z`.
fn minute(at: &str) -> String {
    match at.get(..16) {
        Some(head) if at.ends_with('Z') => format!("{}Z", head.replacen('T', " ", 1)),
        _ => at.to_owned(),
    }
}

/// A decimal count of micros as the Hub writes it.
fn micros(text: &str) -> i64 {
    text.trim().parse().unwrap_or(0)
}

/// Dollars, such as `10` or `$25.50`, as exact micros.
fn parse_usd(text: &str) -> Result<i64, Error> {
    let bad = || {
        Error::new(
            Kind::Usage,
            "usd_invalid",
            format!("{text:?} is not a dollar amount"),
        )
        .remedy("write dollars, such as 10 or 25.50")
    };
    let raw = text.trim().trim_start_matches('$');
    let (whole, fraction) = raw.split_once('.').unwrap_or((raw, ""));
    if whole.is_empty()
        || fraction.len() > 6
        || !whole
            .chars()
            .chain(fraction.chars())
            .all(|c| c.is_ascii_digit())
    {
        return Err(bad());
    }
    let dollars: i64 = whole.parse().map_err(|_| bad())?;
    let fraction: i64 = format!("{fraction:0<6}").parse().map_err(|_| bad())?;
    dollars
        .checked_mul(1_000_000)
        .and_then(|d| d.checked_add(fraction))
        .ok_or_else(bad)
}

/// Micros as dollars: cents always, finer digits only when present.
fn usd(micros: i64) -> String {
    let sign = if micros < 0 { "-" } else { "" };
    let micros = micros.unsigned_abs();
    let fraction = format!("{:06}", micros % 1_000_000);
    let fraction = fraction.trim_end_matches('0');
    format!("{sign}${}.{fraction:0<2}", micros / 1_000_000)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dollars_are_exact_micros() {
        assert_eq!(parse_usd("25.50").unwrap(), 25_500_000);
        assert_eq!(parse_usd("$10").unwrap(), 10_000_000);
        assert_eq!(parse_usd("0.000001").unwrap(), 1);
        for bad in ["", "1.0000001", "ten", "-1", "1e3"] {
            assert!(parse_usd(bad).is_err(), "{bad}");
        }
        assert_eq!(usd(25_500_000), "$25.50");
        assert_eq!(usd(1), "$0.000001");
        assert_eq!(usd(-3_000_000), "-$3.00");
    }
}
