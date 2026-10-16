//! `cozy login|logout|whoami`: this computer's device key at each Hub. Login enrolls a key by
//! an emailed code; every later call signs in with it silently. Codes and names are read from
//! stdin, never from argv.
use std::{io::Write as _, process::ExitCode, sync::Arc};

use cozy_proto::{
    config::Hub,
    error::{Error, Kind},
};
use hub_client::{Credential, DeviceKey, Enrollment, UserSession, types};
use serde_json::json;

use crate::{
    Context,
    context::{self, hub_answer, hub_error},
    output,
};

/// The label a Hub shows for this computer's key.
const LABEL: &str = "cozy";

#[derive(clap::Args, Debug)]
pub struct Login {
    /// Your Tensorhub account's email.
    email: String,
}

/// Enrolls this computer's key by the code the Hub emails, then names the account if it has
/// no name yet. A key already accepted for this email is kept.
pub async fn login(ctx: &Context, args: Login) -> Result<ExitCode, Error> {
    let hub = ctx.hub()?;
    let email = args.email.trim();
    if ctx.signed_in(&hub)
        && let Ok(session) = ctx.hub_session(&hub)
        && session.key().email.eq_ignore_ascii_case(email)
        && matches!(standing(ctx, &hub).await, Ok(Some(_)))
    {
        return signed_in(ctx, &hub, session, "already signed in").await;
    }
    let anonymous = ctx.hub_anonymous(&hub);
    let enrollment = Enrollment::begin(&anonymous, email, LABEL, None)
        .await
        .map_err(hub_error)?;
    let code = ask(ctx, &format!("A code was sent to {email}. Code: "), "code")?;
    let (key, bearer) = enrollment
        .finish(&anonymous, &code)
        .await
        .map_err(hub_error)?;
    let keep =
        |e: cozy_platform::Error| Error::new(Kind::Internal, "key_unwritable", e.to_string());
    cozy_platform::durable::private_dir(&ctx.home.keys()).map_err(keep)?;
    key.save(&hub.device_key(&ctx.home)).map_err(hub_error)?;
    let bearer = serde_json::to_vec(&bearer).expect("a bearer serializes");
    cozy_platform::durable::replace(&hub.bearer(&ctx.home), &bearer, 0o600).map_err(keep)?;
    signed_in(ctx, &hub, ctx.hub_session(&hub)?, "signed in").await
}

/// Reports the account `session` signs in as, naming it first when it has no name yet.
async fn signed_in(
    ctx: &Context,
    hub: &Hub,
    session: Arc<UserSession>,
    status: &str,
) -> Result<ExitCode, Error> {
    let user = ctx.hub_signed_in(hub)?;
    let account = match hub_answer(user.get_current_account().await) {
        Err(e) if e.code == "account.name_required" => {
            let name = ask(ctx, "Tensorhub account name: ", "account name")?;
            hub_answer(
                user.register_account(&name, &types::Empty(Default::default()))
                    .await,
            )?
        }
        answer => answer?,
    };
    let key = session.key();
    if ctx.json {
        output::json(&json!({
            "status": status, "hub": hub, "email": key.email, "account": account.name,
            "device_key": key.id,
        }));
    } else {
        output::Record::new()
            .field("status", status)
            .field("hub", hub)
            .field("email", &key.email)
            .field("account", &account.name)
            .print(ctx);
    }
    Ok(ExitCode::SUCCESS)
}

#[derive(clap::Args, Debug)]
pub struct Logout {
    /// Also sign out every other computer of this account (asks for an emailed code).
    #[arg(long)]
    everywhere: bool,
}

/// Erases this computer's key at the Hub, asking the Hub to revoke it first. The key is erased
/// whatever the Hub answers.
pub async fn logout(ctx: &Context, args: Logout) -> Result<ExitCode, Error> {
    let hub = ctx.hub()?;
    if !ctx.signed_in(&hub) {
        return report_logout(ctx, &hub, None, false, vec![]);
    }
    let session = ctx.hub_session(&hub)?;
    let key = session.key();
    if args.everywhere {
        revoke_others(ctx, &hub, key).await?;
    }
    let revoked = hub_answer(ctx.hub_signed_in(&hub)?.revoke_sign_in_key(&key.id).await);
    for path in [hub.device_key(&ctx.home), hub.bearer(&ctx.home)] {
        if let Err(e) = std::fs::remove_file(&path)
            && e.kind() != std::io::ErrorKind::NotFound
        {
            return Err(Error::new(
                Kind::Internal,
                "key_unremovable",
                format!("{}: {e}", path.display()),
            ));
        }
    }
    let mut notes = vec![];
    if let Err(e) = revoked {
        notes.push(format!(
            "{hub} did not confirm revoking this computer's key ({}); it is erased here",
            e.code
        ));
    }
    report_logout(ctx, &hub, Some(&key.email), args.everywhere, notes)
}

/// Revokes every other key of the account, with a token proven by an emailed code.
async fn revoke_others(ctx: &Context, hub: &Hub, key: &DeviceKey) -> Result<(), Error> {
    let anonymous = ctx.hub_anonymous(hub);
    let enrollment = Enrollment::begin(&anonymous, &key.email, LABEL, Some(key))
        .await
        .map_err(hub_error)?;
    let code = ask(
        ctx,
        &format!("A code was sent to {}. Code: ", key.email),
        "code",
    )?;
    let (_, bearer) = enrollment
        .finish(&anonymous, &code)
        .await
        .map_err(hub_error)?;
    let proven = Credential::Bearer(bearer.token);
    let client = hub_client::Client::connect(&hub.origin, ctx.http().clone(), proven);
    hub_answer(client.revoke_other_device_keys().await)
}

fn report_logout(
    ctx: &Context,
    hub: &Hub,
    email: Option<&str>,
    everywhere: bool,
    notes: Vec<String>,
) -> Result<ExitCode, Error> {
    let status = if email.is_some() {
        "signed out"
    } else {
        "not signed in"
    };
    if ctx.json {
        output::json(&json!({
            "status": status, "hub": hub, "email": email, "everywhere": everywhere, "notes": notes,
        }));
        return Ok(ExitCode::SUCCESS);
    }
    let mut record = output::Record::new()
        .field("status", status)
        .field("hub", hub);
    if let Some(email) = email {
        record = record.field("email", email);
    }
    if everywhere {
        record = record.field("other computers", "signed out");
    }
    for note in notes {
        record = record.note(note);
    }
    record.print(ctx);
    Ok(ExitCode::SUCCESS)
}

#[derive(clap::Args, Debug)]
pub struct Whoami {}

/// Who this computer signs in as at the command's Hub, as that Hub says.
pub async fn whoami(ctx: &Context, _args: Whoami) -> Result<ExitCode, Error> {
    let hub = ctx.hub()?;
    let session = match ctx.signed_in(&hub) {
        true => Some(ctx.hub_session(&hub)?),
        false => None,
    };
    let (accepted, account) = match &session {
        Some(_) => match standing(ctx, &hub).await? {
            Some(account) => (true, account),
            None => (false, None),
        },
        None => (false, None),
    };
    let key = session.as_ref().filter(|_| accepted).map(|s| s.key());
    if ctx.json {
        output::json(&json!({
            "hub": hub, "logged_in": accepted, "email": key.map(|k| &k.email),
            "account": account, "device_key": key.map(|k| &k.id),
        }));
        return Ok(ExitCode::SUCCESS);
    }
    let mut record = output::Record::new().field("hub", &hub);
    match key {
        Some(key) => {
            record = record
                .field("email", &key.email)
                .field("account", account.as_deref().unwrap_or("none yet"))
                .field("device key", &key.id);
        }
        None if session.is_some() => {
            record = record.field("login", "this computer's key is no longer accepted");
        }
        None => record = record.field("login", "not signed in"),
    }
    record.print(ctx);
    if !accepted || account.is_none() {
        output::next(ctx, &[&context::login(&hub)]);
    }
    Ok(ExitCode::SUCCESS)
}

/// The account this computer's key signs in as at `hub`: `None` when the Hub no longer accepts
/// the key (revoked from another computer), `Some(None)` when the account has no name yet.
async fn standing(ctx: &Context, hub: &Hub) -> Result<Option<Option<String>>, Error> {
    match hub_answer(ctx.hub_signed_in(hub)?.get_current_account().await) {
        Ok(account) => Ok(Some(Some(account.name))),
        Err(e) if e.code == "account.name_required" => Ok(Some(None)),
        Err(e) if e.kind == Kind::Credential => Ok(None),
        Err(e) => Err(e),
    }
}

/// One line from stdin after a prompt on stderr.
fn ask(_ctx: &Context, prompt: &str, what: &str) -> Result<String, Error> {
    anstream::eprint!("{prompt}");
    let _ = anstream::stderr().flush();
    let mut line = String::new();
    std::io::stdin()
        .read_line(&mut line)
        .map_err(|e| Error::new(Kind::Usage, "input_unreadable", format!("the {what}: {e}")))?;
    let line = line.trim();
    if line.is_empty() {
        return Err(Error::new(
            Kind::Usage,
            "input_missing",
            format!("no {what} was given"),
        ));
    }
    Ok(line.to_owned())
}
