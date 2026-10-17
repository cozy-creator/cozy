//! The Hub's worker API as this machine (hub-client, the worker credential), HTTPS to the
//! granted origin (public roots plus the granted CA): rental authority, the endpoint report and
//! release.
use super::grant::HubGrant;
use base64::{engine::general_purpose::URL_SAFE_NO_PAD, Engine};
use ed25519_dalek::VerifyingKey;
use hub_client::{types, Credential};
use std::{
    io,
    num::NonZeroU64,
    sync::atomic::{AtomicI64, Ordering},
    time::Duration,
};

pub struct Hub {
    grant: HubGrant,
    client: hub_client::Client,
    /// When the Hub last answered anything (0: never in this process).
    contact_ms: AtomicI64,
}

#[derive(Debug)]
pub enum Refusal {
    /// 401/403: the Hub denies this attempt's authority.
    Denied,
    /// Unreachable, slow or any other answer: proves nothing either way.
    Transport(String),
}

impl Hub {
    pub fn new(grant: HubGrant) -> io::Result<Self> {
        let http = crate::hub::http(grant.ca_der.as_deref()).map_err(|e| io::Error::other(e.0))?;
        let credential = Credential::Worker {
            id: grant.worker_id.clone(),
            token: grant.worker_token.clone(),
        };
        Ok(Self {
            client: hub_client::Client::connect(&grant.origin, http, credential),
            grant,
            contact_ms: AtomicI64::new(0),
        })
    }

    pub fn origin(&self) -> &str {
        &self.grant.origin
    }

    /// When the Hub last answered this machine, whatever it said; 0: never in this process.
    pub fn contact_ms(&self) -> i64 {
        self.contact_ms.load(Ordering::Acquire)
    }

    /// An answer, or a refusal the Hub made, is contact; only an unreachable Hub is not.
    fn heard<T>(&self, answer: Result<T, hub_client::Error>) -> Result<T, Refusal> {
        if !matches!(
            answer,
            Err(hub_client::Error::Call(_) | hub_client::Error::Credential(_))
        ) {
            self.contact_ms
                .store(super::lifecycle::now_ms(), Ordering::Release);
        }
        answer.map_err(|error| match error {
            hub_client::Error::Hub {
                status: 401 | 403, ..
            } => Refusal::Denied,
            hub_client::Error::Call(why) => Refusal::Transport(format!("reach Tensorhub: {why}")),
            other => Refusal::Transport(other.to_string()),
        })
    }

    /// The rental's current authorized keys and their lease.
    pub async fn authorized_keys(&self) -> Result<(Vec<VerifyingKey>, Duration), Refusal> {
        let answer = self
            .client
            .get_worker_authorized_keys(&self.grant.worker_id)
            .await;
        let document = self.heard(answer.map(|a| a.into_inner()).map_err(Into::into))?;
        let invalid = || Refusal::Transport("rental authority metadata is invalid".into());
        if document.worker_id != self.grant.worker_id
            || !(1..=3600).contains(&document.lease_seconds)
        {
            return Err(invalid());
        }
        let keys = document
            .authorized_keys
            .iter()
            .map(|spelled| {
                let raw: [u8; 32] = URL_SAFE_NO_PAD.decode(spelled).ok()?.try_into().ok()?;
                VerifyingKey::from_bytes(&raw).ok()
            })
            .collect::<Option<Vec<_>>>()
            .ok_or_else(invalid)?;
        Ok((keys, Duration::from_secs(document.lease_seconds as u64)))
    }

    /// Tells the Hub where this booting machine listens. Ok(false): a Hub that takes no
    /// reports and waits for the provider instead.
    pub async fn report_endpoint(
        &self,
        endpoint: &super::endpoint::Endpoint,
    ) -> Result<bool, Refusal> {
        let report = types::WorkerEndpointReport {
            public_host: endpoint.public_host.clone(),
            ports: endpoint
                .ports
                .iter()
                .filter_map(|(private, public)| {
                    Some((private.clone(), NonZeroU64::new((*public).into())?))
                })
                .collect(),
        };
        let answer = self
            .client
            .report_worker_endpoint(&self.grant.worker_id, &report)
            .await;
        match answer.map(drop).map_err(hub_client::Error::from) {
            Err(hub_client::Error::Hub { status: 404, .. }) => self.heard(Ok(false)),
            answer => self.heard(answer.map(|()| true)),
        }
    }

    /// Ends this rental's allocation; acceptance is the Hub's 204.
    pub async fn release(&self) -> Result<(), Refusal> {
        let request = types::WorkerReleaseRequest {
            worker_id: self.grant.worker_id.clone(),
        };
        let answer = self
            .client
            .release_worker_rental(&self.grant.worker_id, &request)
            .await;
        self.heard(answer.map(drop).map_err(Into::into))
    }
}

impl std::fmt::Display for Refusal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Denied => write!(f, "the Hub denied this rental's authority"),
            Self::Transport(detail) => write!(f, "{detail}"),
        }
    }
}
