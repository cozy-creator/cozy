//! Every TensorFS failure crosses Python through the same twelve error families.

use pyo3::exceptions::PyException;
use pyo3::prelude::*;
use pyo3::create_exception;
use tensorfs::failure::{Failure, Kind};

create_exception!(
    tensorfs,
    Error,
    PyException,
    "A TensorFS error: `.code` names its family, `.detail` its cause."
);
create_exception!(tensorfs, InvalidError, Error);
create_exception!(tensorfs, FormatError, Error);
create_exception!(tensorfs, NotFoundError, Error);
create_exception!(tensorfs, IntegrityError, Error);
create_exception!(tensorfs, UnknownSchemeError, Error);
create_exception!(tensorfs, DecodeError, Error);
create_exception!(tensorfs, StorageError, Error);
create_exception!(tensorfs, DiskFullError, Error);
create_exception!(tensorfs, NetworkError, Error);
create_exception!(tensorfs, DeniedError, Error);
create_exception!(tensorfs, UnsupportedError, Error);
create_exception!(tensorfs, ConflictError, Error);

fn family(py: Python<'_>, kind: Kind) -> Bound<'_, pyo3::types::PyType> {
    match kind {
        Kind::Invalid => py.get_type::<InvalidError>(),
        Kind::Format => py.get_type::<FormatError>(),
        Kind::NotFound => py.get_type::<NotFoundError>(),
        Kind::Integrity => py.get_type::<IntegrityError>(),
        Kind::UnknownScheme => py.get_type::<UnknownSchemeError>(),
        Kind::Decode => py.get_type::<DecodeError>(),
        Kind::Io => py.get_type::<StorageError>(),
        Kind::Full => py.get_type::<DiskFullError>(),
        Kind::Network => py.get_type::<NetworkError>(),
        Kind::Denied => py.get_type::<DeniedError>(),
        Kind::Unsupported => py.get_type::<UnsupportedError>(),
        Kind::Conflict => py.get_type::<ConflictError>(),
    }
}

/// The one conversion: a failure becomes its family's exception, with `.code` and `.detail`.
pub(crate) fn raise(py: Python<'_>, f: Failure) -> PyErr {
    let err = PyErr::from_type(family(py, f.kind), f.to_string());
    let value = err.value(py);
    let _ = value.setattr("code", f.kind.code());
    let _ = value.setattr("detail", f.detail);
    err
}

pub fn install(m: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = m.py();
    m.add("Error", py.get_type::<Error>())?;
    for kind in Kind::ALL {
        let ty = family(py, kind);
        ty.setattr("code", kind.code())?;
        m.add(ty.name()?, ty)?;
    }
    Ok(())
}
