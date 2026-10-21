//! `GET /v1/runs/{id}/play`: how a browser plays one of a run's outputs straight from its
//! machine over WebRTC, with no relay: the address to dial, the certificate its DTLS presents,
//! and a Cozy-Cap for that output alone.
use std::sync::Arc;

use axum::{
    Json,
    extract::{Path, Query, State, rejection::QueryRejection},
};
use ed25519_dalek::{SigningKey, pkcs8::DecodePrivateKey as _};
use protocol::{
    calcifer::Play,
    capability::{self, Grant},
    error::{Error, Kind},
    ids::RunId,
};
use serde::Deserialize;

use crate::{
    api::{Api, Refusal, find},
    store::{State as RunState, now_ms},
};

#[derive(Deserialize)]
pub struct Ask {
    /// `name`, or `name/i` for a list's item `i`.
    output: String,
    #[serde(default = "an_hour")]
    expires_s: u64,
}

fn an_hour() -> u64 {
    3600
}

pub async fn play(
    State(api): State<Arc<Api>>,
    Path(key): Path<String>,
    query: Result<Query<Ask>, QueryRejection>,
) -> Result<Json<Play>, Refusal> {
    let Query(ask) = query?;
    let row = find(&api, &key)?;
    let refuse = |kind, code: &str, message: String| Refusal(Error::new(kind, code, message));
    // A run still going may yet make the output; an ended one never will.
    let (name, index) = match ask.output.rsplit_once('/') {
        Some((name, i)) if i.parse::<u32>().is_ok() => (name, i.parse::<u32>().ok()),
        _ => (ask.output.as_str(), None),
    };
    let made = api
        .calcifer
        .record(&row)?
        .output
        .iter()
        .any(|item| item.name == name && index.is_none_or(|i| item.index == i));
    if row.state == RunState::Settled && !made {
        return Err(refuse(
            Kind::NotFound,
            "output_not_found",
            format!("run {} has no output {:?}", row.number, ask.output),
        ));
    }
    let unavailable = |message| refuse(Kind::Unavailable, "machine_unavailable", message);
    let name = row
        .machine
        .clone()
        .ok_or_else(|| unavailable(format!("run {} has no machine yet", row.number)))?;
    let machine = api.calcifer.store().machine(&name)?;
    let (Some(machine), Some(frame)) = (machine, api.machines.frame(&name)) else {
        return Err(unavailable(format!("{name} has not reported its status")));
    };
    let Some(endpoint) = machine.endpoint else {
        return Err(unavailable(format!("{name} has no address yet")));
    };
    let webrtc = frame.webrtc.unwrap_or_default();
    let Some(address) = webrtc.addresses.first() else {
        let reason = match webrtc.unavailable_reason {
            reason if reason.is_empty() => format!("{name} does not listen for players"),
            reason => reason,
        };
        return Err(refuse(Kind::Unavailable, "webrtc_unavailable", reason));
    };
    let signer = std::fs::read_to_string(&machine.key)
        .ok()
        .and_then(|pem| SigningKey::from_pkcs8_pem(&pem).ok())
        .ok_or_else(|| {
            let message = format!("cannot read the signing key {}", machine.key.display());
            refuse(Kind::Internal, "machine_key_unreadable", message)
        })?;
    Ok(Json(Play {
        address: address.clone(),
        fingerprint: webrtc.fingerprint,
        token: token(
            &signer,
            endpoint.worker_id,
            &row.id,
            ask.output,
            ask.expires_s,
        ),
    }))
}

/// A Cozy-Cap for one output of one run on one machine, for `expires_s` (at least 1) from now.
fn token(
    signer: &SigningKey,
    machine: String,
    run: &RunId,
    output: String,
    expires_s: u64,
) -> String {
    let grant = Grant {
        machine,
        run: run.to_string(),
        outputs: vec![output],
        expires: now_ms() / 1000 + expires_s.max(1) as i64,
        ..Grant::default()
    };
    capability::mint(signer, grant)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The play cap verifies on its machine for that run's one output, and until its expiry.
    #[test]
    fn a_play_cap_admits_one_output_of_one_run_until_it_expires() {
        let signer = SigningKey::from_bytes(&[5; 32]);
        let run = RunId::mint([1; 12]);
        let token = token(&signer, "m".into(), &run, "video/2".into(), 60);
        let keys = [signer.verifying_key()];
        let now = now_ms() / 1000;
        let (grant, _) = capability::verify(&token, "m", &keys, now, "").unwrap();
        assert!(grant.allows(run.as_str(), "video", Some(2)));
        assert!(
            !grant.allows(run.as_str(), "video", Some(1)),
            "that item alone"
        );
        assert!(
            !grant.allows(run.as_str(), "image", None),
            "that output alone"
        );
        assert!(!grant.allows(RunId::mint([2; 12]).as_str(), "video", Some(2)));
        assert!(capability::verify(&token, "other", &keys, now, "").is_err());
        let expired = capability::verify(&token, "m", &keys, now + 61, "");
        assert_eq!(expired.unwrap_err(), capability::Refusal::Expired);
    }
}
