//! A call the generator cannot express (a conditional GET), under the same hooks as every
//! generated call.
use progenitor_client::{ClientHooks, ClientInfo, OperationInfo, encode_path};
use reqwest::header::{ACCEPT, ETAG, IF_NONE_MATCH};

use crate::{Client, Error, types};

impl Client {
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
        let card = crate::wire::from_response::<types::PackageCard, types::Error>(response).await?;
        Ok(Some((card.into_inner(), etag)))
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
                let refusal = crate::wire::from_response::<_, types::Error>(response).await?;
                Err(Call::ErrorResponse(refusal).into())
            }
            _ => Err(Call::UnexpectedResponse(response).into()),
        }
    }
}
