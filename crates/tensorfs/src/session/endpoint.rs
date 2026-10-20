use std::sync::Arc;

use base64::{engine::general_purpose::STANDARD, Engine};

use crate::failure::{fail, Failure, Kind, Result};
use crate::fetch::{self, Client as Transport, SourcePolicy};
use crate::format::{ObjectRef, Repo, Tree};

use super::provider::from_native;
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
            )
            .map_err(from_native)?,
            None => Transport::new(policy),
        };
        let token = config.token.clone().map(|provider| {
            Arc::new(super::provider::NativeProvider(provider)) as Arc<dyn fetch::TokenProvider>
        });
        let inner = Arc::new(fetch::Endpoint::new(&transport, url, token).map_err(from_native)?);
        Ok(Self { inner, transport })
    }

    pub async fn repo(&self, repo_id: &str) -> Result<Option<(Repo, String)>> {
        self.inner.repo(repo_id).await.map_err(from_native)
    }

    /// `revision` of `repo_id` as a commit, without opening a variant or filling a cache: a tag,
    /// a tag's commit or a header's commit (spec §7), else (`sha256:<hex>`) a commit the endpoint
    /// holds that no tag names. The tag it resolved through, if any.
    pub async fn commit(&self, repo_id: &str, revision: &str) -> Result<(ObjectRef, Option<String>)> {
        let repo = self.repo(repo_id).await?.map_or_else(|| Repo::new(repo_id), |(repo, _)| repo);
        let tree = async |commit: &ObjectRef| -> Result<Tree> { Ok(Tree::from_json(&self.inner.document(commit).await.map_err(from_native)?)?) };
        match fetch::resolve(&repo, Some(revision), tree).await {
            Ok((tag, _)) => return Ok((tag.commit, Some(tag.name))),
            Err(e) if e.kind == Kind::NotFound && revision.starts_with("sha256:") => {}
            Err(e) => return Err(e),
        }
        let object = self.inner.object(revision).await.map_err(from_native)?;
        let Some(object) = object else {
            return fail(Kind::NotFound, format!("{repo_id} has no revision {revision}, and {} holds no such object", self.inner.url()));
        };
        Tree::from_json(&self.inner.document(&object).await.map_err(from_native)?)?;
        Ok((object, None))
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
            None => result.map_err(from_native),
        }
    }
}
