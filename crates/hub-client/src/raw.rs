//! The operations whose body the Hub keeps byte for byte (`x-raw-body` with a request body):
//! sent exactly as given, under the same hooks as every generated call.
use progenitor_client::{ClientHooks, ClientInfo, OperationInfo, ResponseValue, encode_path};
use reqwest::header::{ACCEPT, CONTENT_TYPE};
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

    /// Dev builds only (xs-045 U12): every index row of a model the caller owns, its yanked
    /// releases' lanes and its unreleased checkpoints included, with which are pre-cut (`v1`).
    #[cfg(feature = "migrate")]
    pub async fn model_rows(&self, org: &str, name: &str) -> Result<serde_json::Value, Error> {
        let url = format!(
            "{}/v1/models/{}/{}/rows",
            self.baseurl(),
            encode_path(org),
            encode_path(name),
        );
        let request = (self.client().get(url))
            .header(ACCEPT, "application/json")
            .header("api-version", Self::api_version())
            .build();
        self.send(request, "read_model_rows").await
    }

    async fn send<T: serde::de::DeserializeOwned>(
        &self,
        request: reqwest::Result<reqwest::Request>,
        operation_id: &'static str,
    ) -> Result<T, Error> {
        type Call = progenitor_client::Error<types::Error>;
        let mut request = request.map_err(Call::from)?;
        let info = OperationInfo { operation_id };
        self.pre::<types::Error>(&mut request, &info).await?;
        let response = self.exec(request, &info).await.map_err(Call::from)?;
        let answer = match response.status().as_u16() {
            200..=299 => ResponseValue::<T>::from_response::<types::Error>(response).await?,
            400..=599 => {
                let refusal = ResponseValue::from_response::<types::Error>(response).await?;
                return Err(Call::ErrorResponse(refusal).into());
            }
            _ => return Err(Call::UnexpectedResponse(response).into()),
        };
        Ok(answer.into_inner())
    }
}
