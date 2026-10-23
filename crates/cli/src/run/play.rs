//! `cozy run play N`: a link that plays a run's output in any browser, straight from the machine
//! that made it, local or rented, live as it grows. Calcifer reads the machine's WebRTC listener
//! and signs a capability for that one output; the link carries both in its fragment, which never
//! leaves the browser.
use std::process::ExitCode;

use protocol::{
    calcifer::Play,
    error::{Error, Kind},
};

use crate::{Context, output};

/// The static player page (GitHub Pages' build of `web/player`).
const PLAYER: &str = "https://cozy-creator.github.io/cozy/play/";

pub async fn exec(
    ctx: &Context,
    run: &str,
    output: &str,
    expires: &str,
) -> Result<ExitCode, Error> {
    if output.is_empty() || output.contains(['#', '&', '=', ' ', '\t', '\n']) {
        return Err(Error::new(
            Kind::Usage,
            "play_output_invalid",
            "--output names one output, or name/index",
        ));
    }
    let seconds = lifetime(expires)?;
    let calcifer = ctx.calcifer().await?;
    let run = calcifer.run(run).await?;
    let play: Play = calcifer
        .get_query(
            &format!("/v1/runs/{}/play", run.id),
            &[
                ("output", output.to_owned()),
                ("expires_s", seconds.to_string()),
            ],
        )
        .await?;
    let link = link(&play, run.id.as_str(), output);
    if ctx.json {
        output::json(
            &serde_json::json!({"run": run.number, "output": output, "expires_s": seconds, "link": link}),
        );
    } else {
        output::Record::new()
            .field("run", run.number)
            .field("output", output)
            .field("expires", expires)
            .field("link", &link)
            .print(ctx);
    }
    Ok(ExitCode::SUCCESS)
}

/// The player's link: everything it needs rides the fragment.
fn link(play: &Play, run: &str, output: &str) -> String {
    let fragment = url::form_urlencoded::Serializer::new(String::new())
        .append_pair("v", "1")
        .append_pair("a", &play.address)
        .append_pair("f", &play.fingerprint)
        .append_pair("c", &play.token)
        .append_pair("r", run)
        .append_pair("o", output)
        .finish();
    format!("{PLAYER}#{fragment}")
}

/// `--expires`: a positive count of seconds, minutes, hours or days (`90s`, `30m`, `24h`, `7d`).
fn lifetime(text: &str) -> Result<u64, Error> {
    let invalid = || {
        Error::new(
            Kind::Usage,
            "play_expires_invalid",
            format!("--expires {text:?} is not a duration like 30m, 24h or 7d"),
        )
    };
    let unit = text.chars().last().ok_or_else(invalid)?;
    let count: u64 = text[..text.len() - unit.len_utf8()]
        .parse()
        .map_err(|_| invalid())?;
    let scale = match unit {
        's' => 1,
        'm' => 60,
        'h' => 3600,
        'd' => 86_400,
        _ => return Err(invalid()),
    };
    match count {
        0 => Err(invalid()),
        count => Ok(count * scale),
    }
}
