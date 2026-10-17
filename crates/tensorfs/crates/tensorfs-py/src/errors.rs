//! Typed refusals, in Python, generated from the Rust code list.
//!
//! A refusal that crosses this boundary as a `RuntimeError` carrying a formatted string is
//! not a typed refusal any more — the caller can only regex it, and the code that named the
//! first disagreeing thing is gone. So every `err::Code` gets ONE Python exception class,
//! built at import time from `Code::ALL`. Nothing about the set is written down in Python:
//! adding a code in Rust adds the class, and the fence keeps Python from spelling one.
//!
//! `Refusal` is the base of all of them and carries `.code` and `.detail`, so a caller may
//! catch the exact refusal (`except errors.ShortRead`) or the family (`except
//! errors.Refusal as e: e.code`).

use std::collections::HashMap;
use std::ffi::CString;

use pyo3::prelude::*;
use pyo3::sync::PyOnceLock;
use pyo3::types::{PyDict, PyType};
use tensorfs_core::err::{Code, Refusal};

static BASE: PyOnceLock<Py<PyType>> = PyOnceLock::new();
static CLASSES: PyOnceLock<HashMap<&'static str, Py<PyType>>> = PyOnceLock::new();

/// `SHORT_READ` -> `ShortRead`. The wire name stays the code; this is the Python spelling.
fn camel(code: &str) -> String {
    code.split('_')
        .map(|w| {
            let mut c = w.chars();
            match c.next() {
                Some(f) => f.to_uppercase().collect::<String>() + &c.as_str().to_lowercase(),
                None => String::new(),
            }
        })
        .collect()
}

fn cstr(s: &str) -> CString {
    CString::new(s).expect("code names are ASCII identifiers")
}

pub fn base(py: Python<'_>) -> &Py<PyType> {
    BASE.get_or_init(py, || {
        PyErr::new_type(
            py,
            &cstr("tensorfs.errors.Refusal"),
            Some(&cstr(
                "A TensorFS refusal. `.code` is the frozen refusal code, `.detail` names the \
                 first disagreeing thing with both sides.",
            )),
            None,
            None,
        )
        .expect("the base refusal type is constructible")
    })
}

fn classes(py: Python<'_>) -> &HashMap<&'static str, Py<PyType>> {
    CLASSES.get_or_init(py, || {
        let b = base(py).bind(py).clone();
        let mut m = HashMap::new();
        for c in Code::ALL {
            let name = c.as_str();
            let ty = PyErr::new_type(
                py,
                &cstr(&format!("tensorfs.errors.{}", camel(name))),
                Some(&cstr(&format!("The `{name}` refusal."))),
                Some(&b),
                None,
            )
            .expect("refusal subclass is constructible");
            ty.bind(py).setattr("code", name).expect("class attribute");
            m.insert(name, ty);
        }
        m
    })
}

/// The one conversion. Every refusal leaving the compiled half goes through here.
pub fn to_py(py: Python<'_>, r: &Refusal) -> PyErr {
    let name = r.code.as_str();
    let ty = match classes(py).get(name) {
        Some(t) => t.bind(py).clone(),
        None => return PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(r.to_string()),
    };
    let msg = format!("{name}: {}", r.detail);
    match ty.call1((msg,)) {
        Ok(inst) => {
            let _ = inst.setattr("code", name);
            let _ = inst.setattr("detail", r.detail.as_str());
            PyErr::from_value(inst)
        }
        Err(e) => e,
    }
}

pub trait IntoPy<T> {
    fn or_refuse(self, py: Python<'_>) -> PyResult<T>;
}

impl<T> IntoPy<T> for Result<T, Refusal> {
    fn or_refuse(self, py: Python<'_>) -> PyResult<T> {
        self.map_err(|e| to_py(py, &e))
    }
}

/// Register `Refusal`, every subclass, and the `code -> class` map on the module.
pub fn install(m: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = m.py();
    m.add("Refusal", base(py).bind(py).clone())?;
    let by_code = PyDict::new(py);
    for c in Code::ALL {
        let name = c.as_str();
        let ty = classes(py)[name].bind(py).clone();
        m.add(&*camel(name), ty.clone())?;
        by_code.set_item(name, ty)?;
    }
    m.add("CODES", by_code)?;
    Ok(())
}
