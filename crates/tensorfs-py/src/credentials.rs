//! Python token providers and `Endpoint`, over the public Rust ones.

use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use tensorfs::format::Repo;
use tensorfs::raw::block_on;
use tensorfs::{bearer, Config, Endpoint, Failure, HeaderMap, Kind, Request, Result, TokenProvider};

use crate::store::{config_from_py, config_to_py, OrRaise};

pub(crate) fn config(
    endpoint: Option<String>,
    cache_dir: Option<PathBuf>,
    token: Option<&Bound<'_, PyAny>>,
    ca_der: Option<Vec<u8>>,
) -> PyResult<Config> {
    let mut config = Config::from_env();
    if let Some(endpoint) = endpoint {
        config.endpoint = endpoint;
    }
    if let Some(root) = cache_dir {
        config.cache_dir = root;
    }
    config.token = provider(token)?;
    config.ca_der = ca_der;
    Ok(config)
}

fn failed(error: PyErr) -> Failure {
    Python::attach(|py| {
        if !error.is_instance_of::<crate::errors::Error>(py) {
            return Failure {
                kind: Kind::Denied,
                detail: "the Python token provider failed".into(),
            };
        }
        let value = error.value(py);
        let family: String = value
            .getattr("code")
            .and_then(|code| code.extract())
            .unwrap_or_default();
        let kind = Kind::ALL
            .into_iter()
            .find(|kind| kind.code() == family)
            .unwrap_or(Kind::Denied);
        let detail = value
            .getattr("detail")
            .and_then(|detail| detail.extract())
            .unwrap_or_else(|_| {
                value
                    .str()
                    .map(|text| text.to_string_lossy().into_owned())
                    .unwrap_or_else(|_| "the TensorFS token provider failed".into())
            });
        Failure { kind, detail }
    })
}

struct PythonProvider(Py<PyAny>);

#[async_trait]
impl TokenProvider for PythonProvider {
    async fn headers(&self, request: &Request<'_>) -> Result<Vec<(String, String)>> {
        Python::attach(|py| {
            let provider = self.0.bind(py);
            let headers = PyDict::new(py);
            for (name, value) in request.headers {
                headers.set_item(name, value)?;
            }
            let args = (request.method, request.url, headers, request.payload);
            let value = if provider.hasattr("headers")? {
                provider.call_method1("headers", args)?
            } else {
                provider.call1(args)?
            };
            if let Ok(dict) = value.cast::<PyDict>() {
                dict.iter()
                    .map(|(k, v)| Ok((k.extract()?, v.extract()?)))
                    .collect()
            } else {
                value.extract()
            }
        })
        .map_err(failed)
    }

    fn retry(&self, sent: &Request<'_>, status: u16, headers: &HeaderMap) -> Result<bool> {
        Python::attach(|py| -> PyResult<bool> {
            let provider = self.0.bind(py);
            if !provider.hasattr("retry")? {
                return Ok(false);
            }
            let answer = PyDict::new(py);
            for (name, value) in headers {
                if let Ok(value) = value.to_str() {
                    answer.set_item(name.as_str(), value)?;
                }
            }
            let request = PyDict::new(py);
            for (name, value) in sent.headers {
                request.set_item(name, value)?;
            }
            provider
                .call_method1(
                    "retry",
                    (sent.method, sent.url, request, sent.payload, status, answer),
                )?
                .extract()
        })
        .map_err(failed)
    }
}

pub(crate) fn provider(
    token: Option<&Bound<'_, PyAny>>,
) -> PyResult<Option<Arc<dyn TokenProvider>>> {
    Ok(match token {
        None => None,
        Some(token) if token.is_none() => None,
        Some(token) => {
            if let Ok(token) = token.extract::<String>() {
                Some(bearer(token))
            } else {
                Some(Arc::new(PythonProvider(token.clone().unbind())))
            }
        }
    })
}

#[pyclass(name = "Endpoint", module = "tensorfs", frozen)]
pub struct PyEndpoint {
    endpoint: Endpoint,
}

#[pymethods]
impl PyEndpoint {
    #[new]
    #[pyo3(signature = (url, token = None, *, ca_der = None, cache_dir = None))]
    fn new(
        py: Python<'_>,
        url: &str,
        token: Option<&Bound<'_, PyAny>>,
        ca_der: Option<Vec<u8>>,
        cache_dir: Option<PathBuf>,
    ) -> PyResult<Self> {
        let config = config(Some(url.to_owned()), cache_dir, token, ca_der)?;
        Ok(Self {
            endpoint: Endpoint::with_config(url, &config).or_raise(py)?,
        })
    }

    fn repo<'py>(
        &self,
        py: Python<'py>,
        repo_id: &str,
    ) -> PyResult<Option<(Bound<'py, PyAny>, String)>> {
        let repo = py
            .detach(|| block_on(self.endpoint.repo(repo_id)))
            .or_raise(py)?;
        repo.map(|(repo, etag)| {
            let value = serde_json::to_value(repo).expect("Repo is serializable");
            Ok((config_to_py(py, &value)?, etag))
        })
        .transpose()
    }

    /// `revision` of `repo_id` as a commit: (its id, the tag it resolved through or None).
    fn commit(&self, py: Python<'_>, repo_id: &str, revision: &str) -> PyResult<(String, Option<String>)> {
        let (commit, tag) = py
            .detach(|| block_on(self.endpoint.commit(repo_id, revision)))
            .or_raise(py)?;
        Ok((commit.id(), tag))
    }

    fn update_repo<'py>(
        &self,
        py: Python<'py>,
        repo_id: &str,
        change: Py<PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let callback_error = Mutex::new(None);
        let result = py.detach(|| {
            block_on(self.endpoint.update_repo(repo_id, |old| {
                Python::attach(|py| -> PyResult<Option<Repo>> {
                    let old = serde_json::to_value(old).expect("Repo is serializable");
                    let next = change.bind(py).call1((config_to_py(py, &old)?,))?;
                    if next.is_none() {
                        return Ok(None);
                    }
                    serde_json::from_value(config_from_py(&next)?)
                        .map(Some)
                        .map_err(|_| {
                            crate::errors::raise(
                                py,
                                Failure {
                                    kind: Kind::Invalid,
                                    detail: "change returned an invalid repo document".into(),
                                },
                            )
                        })
                })
                .map_err(|error| {
                    *callback_error.lock().unwrap_or_else(|e| e.into_inner()) = Some(error);
                    Failure {
                        kind: Kind::Invalid,
                        detail: "the repository change callback failed".into(),
                    }
                })
            }))
        });
        if let Some(error) = callback_error
            .into_inner()
            .unwrap_or_else(|e| e.into_inner())
        {
            return Err(error);
        }
        let repo = result.or_raise(py)?;
        config_to_py(
            py,
            &serde_json::to_value(repo).expect("Repo is serializable"),
        )
    }
}

pub fn install(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyEndpoint>()
}
