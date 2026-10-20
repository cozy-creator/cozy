//! Python providers delegate transport and DPoP to the shared Rust client.

use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use base64::engine::general_purpose::STANDARD;
use base64::Engine;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use tensorfs::err::{Code, Refusal, Result};
use tensorfs::fetch::{self, Client, Endpoint, JwtBearer, SourcePolicy, TokenProvider};
use tensorfs::fetch::token::{HeaderMap, Request};
use tensorfs::format::Repo;

use crate::store::{config_from_py, config_to_py, OrRaise};

pub(crate) fn client(url: &str, ca_der: Option<&[u8]>) -> Result<Client> {
    let policy = SourcePolicy { any_host: true, allow_local: fetch::policy::is_loopback_origin(url), max_redirects: 5, ..SourcePolicy::default() };
    match ca_der {
        Some(der) => Client::with_roots(policy, format!("-----BEGIN CERTIFICATE-----\n{}\n-----END CERTIFICATE-----\n", STANDARD.encode(der)).as_bytes()),
        None => Ok(Client::new(policy)),
    }
}

fn failed(error: PyErr) -> Refusal {
    Python::attach(|py| {
        if !error.is_instance_of::<crate::errors::Error>(py) {
            return Refusal { code: Code::CREDENTIAL_REQUIRED, detail: "the Python token provider failed".into() };
        }
        let value = error.value(py);
        let family: String = value.getattr("code").and_then(|code| code.extract()).unwrap_or_default();
        let code = match family.as_str() {
            "invalid" => Code::KEY_GRAMMAR,
            "format" => Code::UNKNOWN_FORMAT,
            "not_found" => Code::REF_NOT_FOUND,
            "integrity" => Code::OBJECT_CORRUPT,
            "unknown_scheme" => Code::UNKNOWN_ENCODING,
            "decode" => Code::ENCODING_MISMATCH,
            "io" => Code::IO_FAILED,
            "full" => Code::CAPACITY_EXHAUSTED,
            "network" => Code::TRANSFER_FAILED,
            "unsupported" => Code::ADAPTER_TARGET_UNSUPPORTED,
            "conflict" => Code::REPOSITORY_CONFLICT,
            _ => Code::CREDENTIAL_REQUIRED,
        };
        let detail = value.getattr("detail").and_then(|detail| detail.extract()).unwrap_or_else(|_| error.to_string());
        Refusal { code, detail }
    })
}

struct PythonProvider(Py<PyAny>);

#[async_trait]
impl TokenProvider for PythonProvider {
    async fn headers(&self, request: &Request<'_>) -> Result<Vec<(String, String)>> {
        Python::attach(|py| {
            let provider = self.0.bind(py);
            let headers = PyDict::new(py);
            for (name, value) in request.headers { headers.set_item(name, value)?; }
            let args = (request.method, request.url, headers, request.payload);
            let value = if provider.hasattr("headers")? { provider.call_method1("headers", args)? } else { provider.call1(args)? };
            if let Ok(dict) = value.cast::<PyDict>() {
                dict.iter().map(|(k, v)| Ok((k.extract()?, v.extract()?))).collect()
            } else { value.extract() }
        }).map_err(failed)
    }

    fn retry(&self, sent: &Request<'_>, status: u16, headers: &HeaderMap) -> Result<bool> {
        Python::attach(|py| -> PyResult<bool> {
            let provider = self.0.bind(py);
            if !provider.hasattr("retry")? { return Ok(false); }
            let answer = PyDict::new(py);
            for (name, value) in headers {
                if let Ok(value) = value.to_str() { answer.set_item(name.as_str(), value)?; }
            }
            let request = PyDict::new(py);
            for (name, value) in sent.headers { request.set_item(name, value)?; }
            provider.call_method1("retry", (sent.method, sent.url, request, sent.payload, status, answer))?.extract()
        }).map_err(failed)
    }
}

pub(crate) fn provider(token: Option<&Bound<'_, PyAny>>) -> PyResult<Option<Arc<dyn TokenProvider>>> {
    Ok(match token {
        None => None,
        Some(token) if token.is_none() => None,
        Some(token) => {
            if let Ok(native) = token.cast::<PyDPoP>() {
                Some(native.get().dpop.clone() as Arc<dyn TokenProvider>)
            } else if let Ok(bearer) = token.extract::<String>() {
                Some(fetch::bearer(bearer))
            } else {
                Some(Arc::new(PythonProvider(token.clone().unbind())))
            }
        }
    })
}

#[pyclass(name = "DPoP", module = "tensorfs", frozen)]
pub struct PyDPoP { dpop: Arc<fetch::DPoP> }

#[pymethods]
impl PyDPoP {
    fn issued_token(&self, py: Python<'_>) -> PyResult<Option<String>> {
        py.detach(|| Ok::<_, tensorfs::err::Refusal>(fetch::runtime()?.block_on(self.dpop.issued_token()))).or_raise(py)
    }
    #[staticmethod]
    #[pyo3(signature = (key, *, token_endpoint, client_id, resource, claims, endpoint = None, ca_der = None))]
    #[allow(clippy::too_many_arguments)]
    fn grant(py: Python<'_>, key: &str, token_endpoint: String, client_id: String, resource: String, claims: &Bound<'_, PyDict>, endpoint: Option<String>, ca_der: Option<Vec<u8>>) -> PyResult<Self> {
        let claims = config_from_py(claims.as_any())?.as_object().cloned().unwrap_or_default();
        let client = client(endpoint.as_deref().unwrap_or(&resource), ca_der.as_deref()).or_raise(py)?;
        let grant = JwtBearer { token_endpoint, client_id, resource, claims };
        let mut dpop = fetch::DPoP::grant(&client, key, grant).or_raise(py)?;
        if let Some(endpoint) = endpoint { dpop = dpop.at(&endpoint); }
        Ok(Self { dpop: Arc::new(dpop) })
    }

    #[staticmethod]
    #[pyo3(signature = (key, access_token, resource = None, *, endpoint = None))]
    fn token(py: Python<'_>, key: &str, access_token: &str, resource: Option<String>, endpoint: Option<String>) -> PyResult<Self> {
        let mut dpop = fetch::DPoP::token(key, access_token, resource.as_deref()).or_raise(py)?;
        if let Some(endpoint) = endpoint { dpop = dpop.at(&endpoint); }
        Ok(Self { dpop: Arc::new(dpop) })
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
        let callback_error = Mutex::new(None);
        let result = py.detach(|| fetch::runtime()?.block_on(self.endpoint.update_repo(repo_id, |old| {
            Python::attach(|py| -> PyResult<Option<Repo>> {
                let old = serde_json::to_value(old).expect("Repo is serializable");
                let next = change.bind(py).call1((config_to_py(py, &old)?,))?;
                if next.is_none() { return Ok(None); }
                serde_json::from_value(config_from_py(&next)?).map(Some).map_err(|_| crate::errors::raise(py, tensorfs::failure::Failure {
                    kind: tensorfs::failure::Kind::Invalid, detail: "change returned an invalid repo document".into() }))
            }).map_err(|error| {
                *callback_error.lock().unwrap_or_else(|e| e.into_inner()) = Some(error);
                Refusal { code: Code::UNKNOWN_FORMAT, detail: "the repository change callback failed".into() }
            })
        })));
        if let Some(error) = callback_error.into_inner().unwrap_or_else(|e| e.into_inner()) { return Err(error); }
        let repo = result.or_raise(py)?;
        config_to_py(py, &serde_json::to_value(repo).expect("Repo is serializable"))
    }
}

pub fn install(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyDPoP>()?;
    m.add_class::<PyEndpoint>()
}
