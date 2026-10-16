//! What every verb receives: the global flags, the one loaded config and the terminal's facts.
use std::path::PathBuf;

use cozy_platform::paths::Home;
use cozy_proto::{
    config::{Config, Hub, Loaded},
    error::Error,
};

use crate::output::Term;

#[derive(Debug)]
pub struct Context {
    pub json: bool,
    pub config: Config,
    pub home: Home,
    /// The user's home directory: services and `~` are relative to it.
    pub user_home: PathBuf,
    pub term: Term,
}

impl Context {
    pub fn new(json: bool, loaded: Loaded, term: Term) -> Self {
        Self {
            json,
            config: loaded.config,
            home: loaded.home,
            user_home: loaded.user_home,
            term,
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

    /// Whether this computer holds a device key for `hub`: signed in, as far as the files say.
    pub fn signed_in(&self, hub: &Hub) -> bool {
        hub.device_key(&self.home).exists()
    }
}
