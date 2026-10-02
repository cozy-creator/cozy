//! Canonical JSON results returned by request-time fit and device qualification.

use crate::canon::{write, Value};
use crate::capability::CapabilityRecord;
use crate::fit::Fit;

pub const FIT_KEYS: &[&str] = &[
    "code",
    "component",
    "custody",
    "detail",
    "device",
    "encoding",
    "ignored",
    "ignored_bytes",
    "key",
    "routes",
    "warning",
];

pub const CAPABILITY_KEYS: &[&str] = &[
    "admitted",
    "code",
    "device",
    "encoding",
    "implementation",
    "note",
    "qualified_devices",
    "reason",
    "vectors",
];

pub const SURFACES: &[(&str, &[&str])] = &[("machine_capability", CAPABILITY_KEYS)];

pub fn fit_json(fit: &Fit, custody: crate::fit::Custody) -> Vec<u8> {
    let (component, key, encoding, device) = match fit {
        Fit::Ok { device, .. } => ("", "", "", device.as_deref().unwrap_or("")),
        Fit::ComponentMissing { component, .. } => (component.as_str(), "", "", ""),
        Fit::TensorMissing { component, key }
        | Fit::ShapeMismatch { component, key, .. }
        | Fit::DtypeMismatch { component, key, .. }
        | Fit::ExtraTensor { component, key } => (component.as_str(), key.as_str(), "", ""),
        Fit::EncodingUnsupported {
            component,
            key,
            encoding,
        } => (component.as_str(), key.as_str(), encoding.as_str(), ""),
        Fit::EncodingUnqualified {
            encoding, device, ..
        } => ("", "", encoding.as_str(), device.as_str()),
    };
    let (routes, ignored, ignored_bytes) = match fit {
        Fit::Ok {
            routes,
            ignored,
            ignored_bytes,
            ..
        } => (routes.clone(), ignored.clone(), *ignored_bytes),
        _ => (crate::fit::Routes::default(), Vec::new(), 0),
    };
    write(&Value::obj(vec![
        ("code", Value::str(fit.code())),
        ("component", Value::str(component)),
        ("custody", Value::str(custody.name())),
        ("detail", Value::str(fit.text())),
        ("device", Value::str(device)),
        ("encoding", Value::str(encoding)),
        (
            "ignored",
            Value::arr(ignored.into_iter().map(Value::str).collect()),
        ),
        ("ignored_bytes", Value::uint(ignored_bytes)),
        ("key", Value::str(key)),
        (
            "routes",
            Value::obj(vec![
                ("decoded_float", Value::uint(routes.decoded_float)),
                ("encoded_gemm", Value::uint(routes.encoded_gemm)),
                ("verbatim", Value::uint(routes.verbatim)),
            ]),
        ),
        ("warning", Value::str(&fit.warning().unwrap_or_default())),
    ]))
}

pub fn capability_json(
    admitted: Option<&CapabilityRecord>,
    encoding: &str,
    device: &str,
    refusal: &str,
    qualified: &[&str],
) -> Vec<u8> {
    let (implementation, vectors, note) = match admitted {
        Some(record) => (
            record.implementation.as_str(),
            record.vectors.id(),
            record.note.as_str(),
        ),
        None => ("", String::new(), ""),
    };
    write(&Value::obj(vec![
        ("admitted", Value::Bool(admitted.is_some())),
        (
            "code",
            Value::str(if admitted.is_some() {
                ""
            } else {
                crate::err::Code::CAPABILITY_UNQUALIFIED.as_str()
            }),
        ),
        ("device", Value::str(device)),
        ("encoding", Value::str(encoding)),
        ("implementation", Value::str(implementation)),
        ("note", Value::str(note)),
        (
            "qualified_devices",
            Value::arr(qualified.iter().map(|value| Value::str(*value)).collect()),
        ),
        ("reason", Value::str(refusal)),
        ("vectors", Value::str(vectors)),
    ]))
}

pub fn document_keys(bytes: &[u8]) -> crate::err::Result<Vec<String>> {
    match crate::canon::parse_canonical(bytes, crate::limits::DOC_MAX_BYTES)? {
        Value::Obj(pairs) => Ok(pairs.into_iter().map(|(key, _)| key).collect()),
        other => crate::err::refuse(
            crate::err::Code::UNKNOWN_FORMAT,
            format!("a machine result is a JSON object, got {}", other.kind()),
        ),
    }
}

pub fn check_surface(kind: &str, bytes: &[u8]) -> crate::err::Result<()> {
    let Some((_, expected)) = SURFACES.iter().find(|(name, _)| *name == kind) else {
        return crate::err::refuse(
            crate::err::Code::UNKNOWN_FORMAT,
            format!("{kind:?} is not a machine result: {SURFACES:?}"),
        );
    };
    let found = document_keys(bytes)?;
    if found
        .iter()
        .map(String::as_str)
        .eq(expected.iter().copied())
    {
        Ok(())
    } else {
        crate::err::refuse(
            crate::err::Code::UNKNOWN_FIELD,
            format!("{kind} keys {found:?} differ from {expected:?}"),
        )
    }
}
