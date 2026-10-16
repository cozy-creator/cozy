//! `cozy login|logout|whoami`: this computer's device key at each Hub.
use std::process::ExitCode;

use cozy_proto::error::Error;
use hub_client::DeviceKey;
use serde_json::json;

use crate::{
    Context,
    context::{self, hub_answer, hub_error},
    output,
};

#[derive(clap::Args, Debug)]
pub struct Login {}

pub async fn login(_ctx: &Context, _args: Login) -> Result<ExitCode, Error> {
    Err(crate::not_ported("login"))
}

#[derive(clap::Args, Debug)]
pub struct Logout {}

pub async fn logout(_ctx: &Context, _args: Logout) -> Result<ExitCode, Error> {
    Err(crate::not_ported("logout"))
}

#[derive(clap::Args, Debug)]
pub struct Whoami {}

/// Who this computer signs in as at the command's Hub, as that Hub says.
pub async fn whoami(ctx: &Context, _args: Whoami) -> Result<ExitCode, Error> {
    let hub = ctx.hub()?;
    if !ctx.signed_in(&hub) {
        if ctx.json {
            output::json(&json!({ "hub": hub, "logged_in": false }));
        } else {
            output::Record::new()
                .field("hub", &hub)
                .field("login", "not signed in")
                .print(ctx);
            output::next(ctx, &[&context::login(&hub)]);
        }
        return Ok(ExitCode::SUCCESS);
    }
    let key = DeviceKey::load(&hub.device_key(&ctx.home)).map_err(hub_error)?;
    let account = match hub_answer(ctx.hub_signed_in(&hub)?.get_current_account().await) {
        Ok(account) => Some(account.name),
        Err(e) if e.code == "account.name_required" => None,
        Err(e) => return Err(e),
    };
    if ctx.json {
        output::json(&json!({
            "hub": hub, "logged_in": true, "email": key.email, "account": account, "device_key": key.id,
        }));
    } else {
        output::Record::new()
            .field("hub", &hub)
            .field("email", &key.email)
            .field("account", account.as_deref().unwrap_or("none yet"))
            .field("device key", &key.id)
            .print(ctx);
        if account.is_none() {
            output::next(ctx, &[&context::login(&hub)]);
        }
    }
    Ok(ExitCode::SUCCESS)
}
