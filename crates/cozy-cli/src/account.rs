//! `cozy login|logout|whoami`.
use std::process::ExitCode;

use cozy_proto::error::Error;

use crate::Context;

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

pub async fn whoami(_ctx: &Context, _args: Whoami) -> Result<ExitCode, Error> {
    Err(crate::not_ported("whoami"))
}
