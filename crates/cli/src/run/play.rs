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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_link_carries_the_machine_its_certificate_and_the_capability() {
        let play = Play {
            address: "203.0.113.7:40123".into(),
            fingerprint: "ab".repeat(32),
            token: "cap.x+y/z".into(),
        };
        let link = link(&play, "run-0102", "video/2");
        let (page, fragment) = link.split_once('#').unwrap();
        assert_eq!(page, PLAYER);
        let pairs: Vec<(String, String)> = url::form_urlencoded::parse(fragment.as_bytes())
            .into_owned()
            .collect();
        let get = |k: &str| {
            pairs
                .iter()
                .find(|(key, _)| key == k)
                .map(|(_, v)| v.as_str())
        };
        assert_eq!(
            (get("v"), get("a"), get("c"), get("r"), get("o")),
            (
                Some("1"),
                Some("203.0.113.7:40123"),
                Some("cap.x+y/z"),
                Some("run-0102"),
                Some("video/2")
            )
        );
        assert_eq!(get("f").unwrap().len(), 64);
    }

    #[test]
    fn expiry_reads_as_a_duration() {
        for (text, seconds) in [("90s", 90), ("30m", 1800), ("24h", 86_400), ("7d", 604_800)] {
            assert_eq!(lifetime(text).unwrap(), seconds, "{text}");
        }
        for text in ["", "0h", "h", "24", "1w", "-1h"] {
            assert_eq!(
                lifetime(text).unwrap_err().code,
                "play_expires_invalid",
                "{text}"
            );
        }
    }
}
