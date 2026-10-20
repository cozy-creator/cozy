use std::sync::Arc;

use async_trait::async_trait;

use crate::failure::{Failure, Kind, Result};
pub use crate::fetch::token::HeaderMap;

pub struct Request<'a> {
    pub method: &'a str,
    pub url: &'a str,
    pub headers: &'a [(String, String)],
    pub payload: Option<&'a str>,
}

#[async_trait]
pub trait TokenProvider: Send + Sync {
    async fn headers(&self, request: &Request<'_>) -> Result<Vec<(String, String)>>;
    fn retry(&self, _sent: &Request<'_>, _status: u16, _headers: &HeaderMap) -> Result<bool> {
        Ok(false)
    }
}

pub fn bearer(token: impl Into<String>) -> Arc<dyn TokenProvider> {
    Arc::new(Bearer(token.into()))
}

struct Bearer(String);
#[async_trait]
impl TokenProvider for Bearer {
    async fn headers(&self, _request: &Request<'_>) -> Result<Vec<(String, String)>> {
        Ok(vec![("authorization".into(), format!("Bearer {}", self.0))])
    }
}

pub(super) struct NativeProvider(pub Arc<dyn TokenProvider>);

fn request<'a>(native: &crate::fetch::token::Request<'a>) -> Request<'a> {
    Request {
        method: native.method,
        url: native.url,
        headers: native.headers,
        payload: native.payload,
    }
}

fn native_error(error: Failure) -> crate::err::Refusal {
    use crate::err::Code;
    let code = match error.kind {
        Kind::Invalid => Code::KEY_GRAMMAR,
        Kind::Format => Code::UNKNOWN_FORMAT,
        Kind::NotFound => Code::REF_NOT_FOUND,
        Kind::Integrity => Code::OBJECT_CORRUPT,
        Kind::UnknownScheme => Code::UNKNOWN_ENCODING,
        Kind::Decode => Code::ENCODING_MISMATCH,
        Kind::Io => Code::IO_FAILED,
        Kind::Full => Code::CAPACITY_EXHAUSTED,
        Kind::Network => Code::TRANSFER_FAILED,
        Kind::Denied => Code::CREDENTIAL_REQUIRED,
        Kind::Unsupported => Code::ADAPTER_TARGET_UNSUPPORTED,
        Kind::Conflict => Code::REPOSITORY_CONFLICT,
    };
    crate::err::Refusal {
        code,
        detail: error.detail,
    }
}

/// Keep the public diagnostic verbatim across the private transport adapter.
/// The shared legacy conversion supplies the family, but its presentation-only
/// code prefix must not become part of a consumer's original callback detail.
pub(super) fn from_native(error: crate::err::Refusal) -> Failure {
    let detail = error.detail.clone();
    Failure {
        detail,
        ..Failure::from(error)
    }
}

#[async_trait]
impl crate::fetch::TokenProvider for NativeProvider {
    async fn headers(
        &self,
        native: &crate::fetch::token::Request<'_>,
    ) -> crate::err::Result<Vec<(String, String)>> {
        self.0.headers(&request(native)).await.map_err(native_error)
    }
    fn retry(
        &self,
        sent: &crate::fetch::token::Request<'_>,
        status: u16,
        headers: &HeaderMap,
    ) -> crate::err::Result<bool> {
        self.0
            .retry(&request(sent), status, headers)
            .map_err(native_error)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_public_provider_error_roundtrips_without_losing_detail() {
        for kind in Kind::ALL {
            for detail in [
                "",
                "provider's exact diagnostic",
                "CREDENTIAL_REQUIRED: caller text",
            ] {
                let original = Failure {
                    kind,
                    detail: detail.into(),
                };
                let restored = from_native(native_error(original.clone()));
                assert_eq!(restored.kind, original.kind);
                assert_eq!(restored.detail, original.detail);
            }
        }
    }
}
