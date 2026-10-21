//! The operations whose body the Hub keeps byte for byte (`x-raw-body` with a request body):
//! sent exactly as given, under the same hooks as every generated call.
use progenitor_client::{ClientHooks, ClientInfo, OperationInfo, ResponseValue, encode_path};
use reqwest::header::{ACCEPT, CONTENT_TYPE, ETAG, IF_NONE_MATCH};
use sha2::{Digest as _, Sha256};

use crate::{Client, Error, types};

impl Client {
    /// Attaches a cozy-eval assessment report to its candidate checkpoint: `report` is the
    /// report's exact canonical bytes, which the Hub addresses by their sha256.
    pub async fn attach_assessment(
        &self,
        org: &str,
        name: &str,
        checkpoint: &str,
        report: Vec<u8>,
    ) -> Result<types::CheckpointAssessmentAttached, Error> {
        let digest = format!("sha256:{:x}", Sha256::digest(&report));
        let url = format!(
            "{}/v1/models/{}/{}/checkpoints/{}/assessments/{}",
            self.baseurl(),
            encode_path(org),
            encode_path(name),
            encode_path(checkpoint),
            encode_path(&digest),
        );
        let request = self
            .client()
            .put(url)
            .header(ACCEPT, "application/json")
            .header(CONTENT_TYPE, "application/json")
            .header("api-version", Self::api_version())
            .body(report)
            .build();
        self.send(request, "attach_checkpoint_assessment").await
    }

    /// `get_package` unless it still has `etag`: None for unchanged (304), else the package
    /// and its ETag.
    pub async fn package_if_changed(
        &self,
        org: &str,
        name: &str,
        etag: Option<&str>,
    ) -> Result<Option<(types::PackageCard, Option<String>)>, Error> {
        let url = format!(
            "{}/v1/packages/{}/{}",
            self.baseurl(),
            encode_path(org),
            encode_path(name)
        );
        let mut request = (self.client().get(url))
            .header(ACCEPT, "application/json")
            .header("api-version", Self::api_version());
        if let Some(etag) = etag {
            request = request.header(IF_NONE_MATCH, etag);
        }
        let response = self.exchange(request.build(), "get_package").await?;
        if response.status() == reqwest::StatusCode::NOT_MODIFIED {
            return Ok(None);
        }
        let etag = response
            .headers()
            .get(ETAG)
            .and_then(|v| v.to_str().ok())
            .map(str::to_owned);
        let card =
            ResponseValue::<types::PackageCard>::from_response::<types::Error>(response).await?;
        Ok(Some((card.into_inner(), etag)))
    }

    async fn send<T: serde::de::DeserializeOwned>(
        &self,
        request: reqwest::Result<reqwest::Request>,
        operation_id: &'static str,
    ) -> Result<T, Error> {
        let response = self.exchange(request, operation_id).await?;
        Ok(ResponseValue::<T>::from_response::<types::Error>(response)
            .await?
            .into_inner())
    }

    /// `request` under the hooks of every generated call, its refusal read as theirs are.
    async fn exchange(
        &self,
        request: reqwest::Result<reqwest::Request>,
        operation_id: &'static str,
    ) -> Result<reqwest::Response, Error> {
        type Call = progenitor_client::Error<types::Error>;
        let mut request = request.map_err(Call::from)?;
        let info = OperationInfo { operation_id };
        self.pre::<types::Error>(&mut request, &info).await?;
        let response = self.exec(request, &info).await.map_err(Call::from)?;
        match response.status().as_u16() {
            200..=299 | 304 => Ok(response),
            400..=599 => {
                let refusal = ResponseValue::from_response::<types::Error>(response).await?;
                Err(Call::ErrorResponse(refusal).into())
            }
            _ => Err(Call::UnexpectedResponse(response).into()),
        }
    }
}
