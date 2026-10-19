//! Python providers delegate transport and DPoP to the shared Rust client.

use std::sync::Arc;

use async_trait::async_trait;
use base64::engine::general_purpose::STANDARD;
use base64::Engine;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use tensorfs_core::err::{Code, Refusal, Result};
use tensorfs_core::fetch::{self, Client, Endpoint, JwtBearer, SourcePolicy, TokenProvider};
use tensorfs_core::fetch::token::{HeaderMap, Request};
use tensorfs_core::format::Repo;

use crate::store::{config_from_py, config_to_py, OrRaise};

pub(crate) fn client(url: &str, ca_der: Option<&[u8]>) -> Result<Client> {
    let policy = SourcePolicy { any_host: true, allow_local: fetch::policy::is_loopback_origin(url), max_redirects: 5, ..SourcePolicy::default() };
    match ca_der {
        Some(der) => Client::with_roots(policy, format!("-----BEGIN CERTIFICATE-----\n{}\n-----END CERTIFICATE-----\n", STANDARD.encode(der)).as_bytes()),
        None => Ok(Client::new(policy)),
    }
}

fn failed(_: PyErr) -> Refusal {
    Refusal { code: Code::CREDENTIAL_REQUIRED, detail: "the Python token provider failed".into() }
}

struct PythonProvider(Py<PyAny>);

#[async_trait]
impl TokenProvider for PythonProvider {
    async fn headers(&self, request: &Request<'_>) -> Result<Vec<(String, String)>> {
        Python::attach(|py| {
            let value = self.0.bind(py).call_method1("headers", (request.method, request.url))?;
            if let Ok(dict) = value.cast::<PyDict>() {
                dict.iter().map(|(k, v)| Ok((k.extract()?, v.extract()?))).collect()
            } else { value.extract() }
        }).map_err(failed)
    }

    fn retry(&self, sent: &Request<'_>, status: u16, headers: &HeaderMap) -> bool {
        Python::attach(|py| -> PyResult<bool> {
            let provider = self.0.bind(py);
            if !provider.hasattr("retry")? { return Ok(false); }
            let answer = PyDict::new(py);
            for (name, value) in headers {
                if let Ok(value) = value.to_str() { answer.set_item(name.as_str(), value)?; }
            }
            provider.call_method1("retry", (sent.url, status, answer))?.extract()
        }).unwrap_or(false)
    }
}

pub(crate) fn provider(token: Option<&Bound<'_, PyAny>>) -> PyResult<Option<Arc<dyn TokenProvider>>> {
    Ok(match token {
        None => None,
        Some(token) if token.is_none() => None,
        Some(token) => {
            if let Ok(native) = token.cast::<PyDPoP>() {
                Some(Arc::clone(&native.get().provider))
            } else if let Ok(bearer) = token.extract::<String>() {
                Some(fetch::bearer(bearer))
            } else {
                Some(Arc::new(PythonProvider(token.clone().unbind())))
            }
        }
    })
}

#[pyclass(name = "DPoP", module = "tensorfs", frozen)]
pub struct PyDPoP { provider: Arc<dyn TokenProvider> }

#[pymethods]
impl PyDPoP {
    #[staticmethod]
    #[pyo3(signature = (key, *, token_endpoint, client_id, resource, claims, endpoint = None, ca_der = None))]
    #[allow(clippy::too_many_arguments)]
    fn grant(py: Python<'_>, key: &str, token_endpoint: String, client_id: String, resource: String, claims: &Bound<'_, PyDict>, endpoint: Option<String>, ca_der: Option<Vec<u8>>) -> PyResult<Self> {
        let claims = config_from_py(claims.as_any())?.as_object().cloned().unwrap_or_default();
        let client = client(endpoint.as_deref().unwrap_or(&resource), ca_der.as_deref()).or_raise(py)?;
        let grant = JwtBearer { token_endpoint, client_id, resource, claims };
        let mut dpop = fetch::DPoP::grant(&client, key, grant).or_raise(py)?;
        if let Some(endpoint) = endpoint { dpop = dpop.at(&endpoint); }
        Ok(Self { provider: Arc::new(dpop) })
    }

    #[staticmethod]
    #[pyo3(signature = (key, access_token, resource = None))]
    fn token(py: Python<'_>, key: &str, access_token: &str, resource: Option<String>) -> PyResult<Self> {
        Ok(Self { provider: Arc::new(fetch::DPoP::token(key, access_token, resource.as_deref()).or_raise(py)?) })
    }
}

#[pyclass(name = "Endpoint", module = "tensorfs", frozen)]
pub struct PyEndpoint { endpoint: Arc<Endpoint> }

#[pymethods]
impl PyEndpoint {
    #[new]
    #[pyo3(signature = (url, token = None, *, ca_der = None))]
    fn new(py: Python<'_>, url: &str, token: Option<&Bound<'_, PyAny>>, ca_der: Option<Vec<u8>>) -> PyResult<Self> {
        let client = client(url, ca_der.as_deref()).or_raise(py)?;
        Ok(Self { endpoint: Arc::new(Endpoint::new(&client, url, provider(token)?).or_raise(py)?) })
    }

    fn repo<'py>(&self, py: Python<'py>, repo_id: &str) -> PyResult<Option<(Bound<'py, PyAny>, String)>> {
        let repo = py.detach(|| fetch::runtime()?.block_on(self.endpoint.repo(repo_id))).or_raise(py)?;
        repo.map(|(repo, etag)| {
            let value = serde_json::to_value(repo).expect("Repo is serializable");
            Ok((config_to_py(py, &value)?, etag))
        }).transpose()
    }

    fn update_repo<'py>(&self, py: Python<'py>, repo_id: &str, change: Py<PyAny>) -> PyResult<Bound<'py, PyAny>> {
        let repo = py.detach(|| fetch::runtime()?.block_on(self.endpoint.update_repo(repo_id, |old| {
            Python::attach(|py| -> PyResult<Option<Repo>> {
                let old = serde_json::to_value(old).expect("Repo is serializable");
                let next = change.bind(py).call1((config_to_py(py, &old)?,))?;
                if next.is_none() { return Ok(None); }
                serde_json::from_value(config_from_py(&next)?).map(Some).map_err(|_| pyo3::exceptions::PyValueError::new_err("change returned an invalid repo document"))
            }).map_err(failed)
        }))).or_raise(py)?;
        config_to_py(py, &serde_json::to_value(repo).expect("Repo is serializable"))
    }
}

pub fn install(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyDPoP>()?;
    m.add_class::<PyEndpoint>()
}
