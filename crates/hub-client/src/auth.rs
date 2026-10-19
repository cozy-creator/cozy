//! Who a request speaks as, presented on every generated call: the hooks progenitor's client
//! runs around each request.
use progenitor_client::{ClientHooks, ClientInfo, Error as CallError, OperationInfo};
use reqwest::header::{HeaderName, HeaderValue};
use std::sync::Arc;

use crate::Client;
use crate::device::UserSession;

#[derive(Clone)]
pub enum Credential {
    Anonymous,
    /// A signed-in user: a Bearer from device-key login, minted again when stale or refused.
    User(Arc<UserSession>),
    /// A fixed Bearer: an enrollment's email-proven token, for the calls only it may make.
    Bearer(String),
    /// A rental pod's worker capability, on the worker edge.
    Worker {
        id: String,
        token: String,
    },
}

/// The client's inner state: its credential.
#[derive(Clone)]
pub struct Auth {
    pub(crate) credential: Credential,
}

impl std::fmt::Debug for Auth {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let kind = match &self.credential {
            Credential::Anonymous => "anonymous",
            Credential::User(_) => "user",
            Credential::Bearer(_) => "bearer",
            Credential::Worker { .. } => "worker",
        };
        f.debug_struct("Auth").field("credential", &kind).finish()
    }
}

impl Client {
    /// A client of the Hub at `origin` speaking as `credential`, over `http` (which carries the
    /// TLS roots and timeouts the caller chose).
    pub fn connect(origin: &str, http: reqwest::Client, credential: Credential) -> Self {
        Self::new_with_client(origin.trim_end_matches('/'), http, Auth { credential })
    }

    /// The same Hub, anonymously: for the exchanges that mint a credential.
    pub fn anonymous(&self) -> Self {
        Self::connect(self.baseurl(), self.client().clone(), Credential::Anonymous)
    }

    /// The resource's authorization server, validated before a capability is minted.
    pub async fn token_endpoint(&self) -> Result<String, crate::Error> {
        let resource = self.get_protected_resource_metadata().await?.into_inner();
        let issuer = resource.authorization_servers.first().map(|s| s.trim_end_matches('/'));
        let server = self.get_openid_configuration().await?.into_inner();
        if issuer != Some(server.issuer.trim_end_matches('/'))
            || !server.grant_types_supported.iter().any(|g| g == "urn:ietf:params:oauth:grant-type:jwt-bearer") {
            return Err(crate::Error::Answer("this Hub names no authorization server that takes run capabilities".into()));
        }
        Ok(server.token_endpoint)
    }

    pub fn credential(&self) -> &Credential {
        &self.inner().credential
    }

    /// Attaches the credential.
    async fn present(&self, request: &mut reqwest::Request) -> Result<(), crate::Error> {
        let headers: Vec<(&str, String)> = match &self.inner().credential {
            Credential::Anonymous => vec![],
            Credential::User(session) => {
                // Login is itself a call through this client: boxed, as the future recurses.
                let token = Box::pin(session.bearer(&self.anonymous())).await?;
                vec![("authorization", format!("Bearer {token}"))]
            }
            Credential::Bearer(token) => vec![("authorization", format!("Bearer {token}"))],
            Credential::Worker { id, token } => {
                vec![
                    ("x-cozy-worker-id", id.clone()),
                    ("x-cozy-worker-token", token.clone()),
                ]
            }
        };
        for (name, value) in headers {
            let value = HeaderValue::from_str(&value)
                .map_err(|_| crate::Error::Key(format!("an unusable {name} value")))?;
            request
                .headers_mut()
                .insert(HeaderName::from_static(name), value);
        }
        Ok(())
    }
}

impl ClientHooks<Auth> for Client {
    async fn pre<E>(
        &self,
        request: &mut reqwest::Request,
        _: &OperationInfo,
    ) -> Result<(), CallError<E>> {
        self.present(request)
            .await
            .map_err(|e| CallError::Custom(e.to_string()))
    }

    /// Sends the request; a refused Bearer logs in again and a new DPoP nonce re-proves, each
    /// once. The first answer stands when the second credential cannot be had. A refusal
    /// without the Hub's envelope (a proxy's page, an empty body) keeps its status.
    async fn exec(
        &self,
        request: reqwest::Request,
        info: &OperationInfo,
    ) -> reqwest::Result<reqwest::Response> {
        enveloped(self.execute(request, info).await?).await
    }
}

impl Client {
    async fn execute(
        &self,
        request: reqwest::Request,
        _: &OperationInfo,
    ) -> reqwest::Result<reqwest::Response> {
        let again = request.try_clone();
        let token = presented(&request);
        let response = self.client().execute(request).await?;
        let (Some(mut again), Some(token)) = (again, token) else {
            return Ok(response);
        };
        let retry = match &self.inner().credential {
            Credential::User(session) if response.status() == 401 => {
                session.refused(&token).await;
                true
            }
            _ => false,
        };
        if !retry {
            return Ok(response);
        }
        match self.present(&mut again).await {
            Ok(_) => self.client().execute(again).await,
            Err(error) => {
                tracing::warn!(%error, "presenting the credential again");
                Ok(response)
            }
        }
    }
}

/// An error answer as the Hub's typed envelope: its own when it carries one (or OAuth's
/// `error`), else `http_<status>` with the start of whatever came back.
async fn enveloped(response: reqwest::Response) -> reqwest::Result<reqwest::Response> {
    let status = response.status();
    if !status.is_client_error() && !status.is_server_error() {
        return Ok(response);
    }
    let mut headers = response.headers().clone();
    let body = response.bytes().await?;
    let parsed = serde_json::from_slice::<serde_json::Value>(&body).ok();
    let body = match parsed {
        Some(value) if value["error"].is_string() => body,
        // The Hub's envelope, its fields made whole: a refusal missing one still reads as typed.
        Some(mut value) if value["error"].is_object() => {
            let error = &mut value["error"];
            let whole = ["code", "message", "remedy"]
                .iter()
                .all(|k| error[k].is_string());
            if whole {
                body
            } else {
                for (key, default) in [
                    ("code", format!("http_{}", status.as_u16())),
                    ("message", status.to_string()),
                    ("remedy", String::new()),
                ] {
                    if !error[key].is_string() {
                        error[key] = default.into();
                    }
                }
                headers.remove(reqwest::header::CONTENT_LENGTH);
                serde_json::to_vec(&value).expect("JSON").into()
            }
        }
        _ => {
            let said: String = String::from_utf8_lossy(&body)
                .trim()
                .chars()
                .take(256)
                .collect();
            let message = if said.is_empty() {
                status.to_string()
            } else {
                said
            };
            let code = format!("http_{}", status.as_u16());
            headers.insert(
                reqwest::header::CONTENT_TYPE,
                HeaderValue::from_static("application/json"),
            );
            headers.remove(reqwest::header::CONTENT_LENGTH);
            let envelope =
                serde_json::json!({"error": {"code": code, "message": message, "remedy": ""}});
            serde_json::to_vec(&envelope).expect("JSON").into()
        }
    };
    let mut rebuilt = http::Response::new(body);
    *rebuilt.status_mut() = status;
    *rebuilt.headers_mut() = headers;
    Ok(rebuilt.into())
}

/// The token a request carries, Bearer or DPoP.
fn presented(request: &reqwest::Request) -> Option<String> {
    let value = request.headers().get("authorization")?.to_str().ok()?;
    let (_, token) = value.split_once(' ')?;
    Some(token.to_string())
}
