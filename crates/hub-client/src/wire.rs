//! How an answer is read: leniently (build.rs gives the generated types their defaults), and a
//! shape it cannot read names the field it is at.
use progenitor_client::{Error, ResponseValue};
use serde::{Deserialize, Deserializer, de::DeserializeOwned};

/// `ResponseValue::from_response`, with a refusal that names its field (`lanes[0].bytes: …`).
pub async fn from_response<T: DeserializeOwned, E>(
    response: reqwest::Response,
) -> Result<ResponseValue<T>, Error<E>> {
    let status = response.status();
    let headers = response.headers().clone();
    let body = response.bytes().await.map_err(Error::ResponseBodyError)?;
    let mut json = serde_json::Deserializer::from_slice(&body);
    let read = serde_path_to_error::deserialize(&mut json)
        .map_err(|e| match e.path().to_string().as_str() {
            "." => e.into_inner().to_string(),
            path => format!("{path}: {}", e.into_inner()),
        })
        .and_then(|inner| json.end().map(|()| inner).map_err(|e| e.to_string()));
    match read {
        Ok(inner) => Ok(ResponseValue::new(inner, status, headers)),
        Err(why) => {
            let why = match status.is_success() {
                true => why,
                false => format!(
                    "{status}: {}",
                    String::from_utf8_lossy(&body[..body.len().min(300)])
                ),
            };
            Err(Error::InvalidResponsePayload(
                body,
                serde::de::Error::custom(why),
            ))
        }
    }
}

/// A `null` list or map is an empty one: Go writes a nil slice as `null`.
pub fn null_is_empty<'de, D: Deserializer<'de>, T: Default + Deserialize<'de>>(
    d: D,
) -> Result<T, D::Error> {
    Option::<T>::deserialize(d).map(Option::unwrap_or_default)
}
