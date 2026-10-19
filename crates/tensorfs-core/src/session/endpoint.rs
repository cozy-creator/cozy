use std::sync::Arc;

use base64::{engine::general_purpose::STANDARD, Engine};

use crate::failure::{Failure, Result};
use crate::fetch::{self, Client as Transport, SourcePolicy};
use crate::format::Repo;

use super::{Config, TokenProvider};

/// Generic repository/object protocol. Authentication stays with TokenProvider.
#[derive(Clone)]
pub struct Endpoint {
    pub(super) inner: Arc<fetch::Endpoint>,
    pub(super) transport: Transport,
}

impl Endpoint {
    pub fn new(url: &str, token: Option<Arc<dyn TokenProvider>>) -> Result<Self> {
        let mut config = Config::from_env();
        config.token = token;
        Self::with_config(url, &config)
    }

    /// Config supplies explicit local storage, generic credentials and optional CA.
    pub fn with_config(url: &str, config: &Config) -> Result<Self> {
        let local;
        let url = if url == "local" {
            if config.cache_dir.as_os_str().is_empty() {
                return crate::failure::fail(
                    crate::failure::Kind::Invalid,
                    "a local endpoint needs cache_dir or TENSORFS_HOME/HOME",
                );
            }
            local = std::path::absolute(&config.cache_dir)
                .map_err(|e| super::io_failure("local endpoint", e))?;
            local.to_str().ok_or_else(|| Failure {
                kind: crate::failure::Kind::Invalid,
                detail: "local endpoint path is not UTF-8".into(),
            })?
        } else {
            url
        };
        let policy = SourcePolicy {
            any_host: true,
            allow_local: fetch::policy::is_loopback_origin(url),
            max_redirects: 5,
            ..SourcePolicy::default()
        };
        let transport = match config.ca_der.as_deref() {
            Some(der) => Transport::with_roots(
                policy,
                format!(
                    "-----BEGIN CERTIFICATE-----\n{}\n-----END CERTIFICATE-----\n",
                    STANDARD.encode(der)
                )
                .as_bytes(),
            )?,
            None => Transport::new(policy),
        };
        let token = config.token.clone().map(|provider| {
            Arc::new(super::provider::NativeProvider(provider)) as Arc<dyn fetch::TokenProvider>
        });
        let inner = Arc::new(fetch::Endpoint::new(&transport, url, token)?);
        Ok(Self { inner, transport })
    }

    pub async fn repo(&self, repo_id: &str) -> Result<Option<(Repo, String)>> {
        Ok(self.inner.repo(repo_id).await?)
    }

    pub async fn update_repo(
        &self,
        repo_id: &str,
        mut change: impl FnMut(Option<Repo>) -> Result<Option<Repo>>,
    ) -> Result<Repo> {
        let mut callback_error = None;
        let result = self
            .inner
            .update_repo(repo_id, |old| match change(old) {
                Ok(new) => Ok(new),
                Err(error) => {
                    callback_error = Some(error);
                    Err(crate::err::Refusal {
                        code: crate::err::Code::UNKNOWN_FORMAT,
                        detail: "repository callback failed".into(),
                    })
                }
            })
            .await;
        match callback_error {
            Some(error) => Err(error),
            None => Ok(result?),
        }
    }
}
