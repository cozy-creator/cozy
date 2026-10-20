//! What every verb receives: the global flags, the one loaded config, the terminal's facts and
//! the Hub clients, which share one HTTP client.
use std::{
    fmt::Debug,
    path::PathBuf,
    sync::{Arc, OnceLock},
};

use platform::paths::Home;
use protocol::{
    config::{Config, Hub, Loaded},
    error::{Error, Kind},
};
use hub_client::{Credential, DeviceKey, ResponseValue, UserSession};

use crate::{calcifer_client::Client, output::Term};

#[derive(Debug)]
pub struct Context {
    pub json: bool,
    pub config: Config,
    pub home: Home,
    /// The user's home directory: services and `~` are relative to it.
    pub user_home: PathBuf,
    pub term: Term,
    http: OnceLock<reqwest::Client>,
}

impl Context {
    pub fn new(json: bool, loaded: Loaded, term: Term) -> Self {
        Self {
            json,
            config: loaded.config,
            home: loaded.home,
            user_home: loaded.user_home,
            term,
            http: OnceLock::new(),
        }
    }

    /// Where new work goes: `--tensorhub`, else the current Hub.
    pub fn hub(&self) -> Result<Hub, Error> {
        self.config.selected()
    }

    /// `--tensorhub`'s Hub, which a list narrows to; without it lists span every Hub.
    pub fn narrowed(&self) -> Result<Option<Hub>, Error> {
        self.config.narrowed()
    }

    /// Calcifer, started if none runs.
    pub async fn calcifer(&self) -> Result<Client, Error> {
        Client::connect(self).await
    }

    /// Whether this computer holds a device key for `hub`: signed in, as far as the files say.
    pub fn signed_in(&self, hub: &Hub) -> bool {
        hub.device_key(&self.home).exists()
    }

    /// The one HTTP client this command's Hub calls share.
    pub fn http(&self) -> &reqwest::Client {
        self.http.get_or_init(reqwest::Client::new)
    }

    pub fn hub_anonymous(&self, hub: &Hub) -> hub_client::Client {
        hub_client::Client::connect(&hub.origin, self.http().clone(), Credential::Anonymous)
    }

    /// `hub` as this computer's signed-in user; refused when it has no device key there.
    pub fn hub_signed_in(&self, hub: &Hub) -> Result<hub_client::Client, Error> {
        let credential = Credential::User(self.hub_session(hub)?);
        Ok(hub_client::Client::connect(
            &hub.origin,
            self.http().clone(),
            credential,
        ))
    }

    /// `cozy login` for `hub`, spelled for the next step: `--tensorhub` only for a Hub other than
    /// the current one.
    pub fn login(&self, hub: &Hub) -> String {
        match self.config.current() {
            Ok(current) if current.origin == hub.origin => "cozy login <email>".into(),
            _ => format!("cozy login <email> --tensorhub {}", hub.label()),
        }
    }

    /// This computer's device key at `hub` and the Bearer it mints, kept beside it.
    pub fn hub_session(&self, hub: &Hub) -> Result<Arc<UserSession>, Error> {
        if !self.signed_in(hub) {
            return Err(Error::new(
                Kind::Credential,
                "not_logged_in",
                format!("this computer is not signed in to {hub}"),
            )
            .next(self.login(hub)));
        }
        let key = DeviceKey::load(&hub.device_key(&self.home)).map_err(hub_error)?;
        Ok(Arc::new(UserSession::new(
            key,
            Some(hub.bearer(&self.home)),
        )))
    }

    /// Signed in when this computer is, else anonymous: what public and owned reads both need.
    pub fn hub_client(&self, hub: &Hub) -> Result<hub_client::Client, Error> {
        match self.signed_in(hub) {
            true => self.hub_signed_in(hub),
            false => Ok(self.hub_anonymous(hub)),
        }
    }
}


/// A Hub call's answer, or its refusal as the CLI's own.
pub fn hub_answer<T, E>(
    answer: Result<ResponseValue<T>, progenitor_client::Error<E>>,
) -> Result<T, Error>
where
    E: serde::Serialize + Debug,
{
    answer.map(ResponseValue::into_inner).map_err(hub_error)
}

/// The Hub's refusal with its code, message and remedy verbatim, classed by its HTTP status.
pub fn hub_error(error: impl Into<hub_client::Error>) -> Error {
    use hub_client::Error as Hub;
    match error.into() {
        Hub::Hub {
            status,
            code,
            message,
            remedy,
        } => {
            let kind = match status {
                401 => Kind::Credential,
                403 => Kind::Forbidden,
                404 => Kind::NotFound,
                409 | 410 => Kind::Conflict,
                429 => Kind::Unavailable,
                400..=499 => Kind::Validation,
                _ => Kind::Unavailable,
            };
            let mut error = Error::new(kind, code.clone(), message).cause(code);
            if !remedy.is_empty() {
                error = error.remedy(remedy);
            }
            error
        }
        Hub::Call(message) => Error::new(Kind::Unavailable, "hub_unreachable", message),
        Hub::Credential(message) | Hub::Key(message) => {
            Error::new(Kind::Credential, "hub_credential", message).next("cozy login <email>")
        }
        other => Error::new(Kind::Internal, "hub_answer", other.to_string()),
    }
}
