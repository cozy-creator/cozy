//! Request-time package/checkpoint fit.
//!
//! `TensorRequirements` is an in-memory census built from `factory(config)` immediately
//! before a cold model download. It is not a stored document, has no digest, and carries
//! no publication metadata. A row asks for one component/key and exact shape, plus an
//! optional logical dtype constraint.

use crate::capability::CapabilityRecords;
use crate::dtype::Dtype;
use crate::err::{refuse, Code, Result};
use crate::header::{Body, Closure, Header};
use crate::ids::ascii_name;
use crate::limits;

/// How many skipped keys the warning names.
const WARNED_KEYS: usize = 5;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorRequirement {
    pub shape: Vec<u64>,
    pub logical_dtype: Option<Dtype>,
}

/// Transient requirements for one request. Rows are sorted once so fit is deterministic
/// regardless of the package census traversal order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorRequirements {
    components: Vec<(String, Vec<(String, TensorRequirement)>)>,
}

impl TensorRequirements {
    pub fn new(
        rows: impl IntoIterator<Item = (String, String, Vec<u64>, Option<Dtype>)>,
    ) -> Result<Self> {
        let mut rows: Vec<_> = rows.into_iter().collect();
        if rows.is_empty() {
            return refuse(Code::EMPTY_COMPONENTS, "tensor requirements are empty");
        }
        if rows.len() > limits::MAX_TENSORS {
            return refuse(Code::COUNT_CAP, "tensor requirements exceed the tensor cap");
        }
        for (component, key, shape, _) in &rows {
            ascii_name(
                "TensorRequirement.component",
                component,
                limits::MAX_NAME_BYTES,
            )?;
            ascii_name("TensorRequirement.key", key, limits::MAX_KEY_BYTES)?;
            if shape.len() > limits::MAX_RANK {
                return refuse(Code::RANK_CAP, format!("{component}/{key}: rank over cap"));
            }
            crate::dtype::checked_elements(key, shape)?;
        }
        rows.sort_by(|a, b| (&a.0, &a.1).cmp(&(&b.0, &b.1)));
        if let Some(pair) = rows
            .windows(2)
            .find(|pair| pair[0].0 == pair[1].0 && pair[0].1 == pair[1].1)
        {
            return refuse(
                Code::DUPLICATE_KEY,
                format!(
                    "{}/{} appears twice in tensor requirements",
                    pair[0].0, pair[0].1
                ),
            );
        }

        let mut components: Vec<(String, Vec<(String, TensorRequirement)>)> = Vec::new();
        for (component, key, shape, logical_dtype) in rows {
            let requirement = TensorRequirement {
                shape,
                logical_dtype,
            };
            match components.last_mut() {
                Some((held, tensors)) if held == &component => tensors.push((key, requirement)),
                _ => components.push((component, vec![(key, requirement)])),
            }
        }
        if components.len() > limits::MAX_COMPONENTS {
            return refuse(
                Code::COUNT_CAP,
                "tensor requirements exceed the component cap",
            );
        }
        Ok(Self { components })
    }

    pub fn components(&self) -> &[(String, Vec<(String, TensorRequirement)>)] {
        &self.components
    }

    pub fn tensors(&self) -> impl Iterator<Item = (&str, &str, &TensorRequirement)> {
        self.components.iter().flat_map(|(component, tensors)| {
            tensors
                .iter()
                .map(move |(key, requirement)| (component.as_str(), key.as_str(), requirement))
        })
    }

    pub fn count(&self) -> usize {
        self.components
            .iter()
            .map(|(_, tensors)| tensors.len())
            .sum()
    }
}

/// Where the checkpoint came from. Accepted and echoed; it no longer changes the verdict.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Custody {
    Canonical,
    Local,
}

impl Custody {
    pub fn name(self) -> &'static str {
        match self {
            Self::Canonical => "canonical",
            Self::Local => "local",
        }
    }

    pub fn parse(value: &str) -> Result<Self> {
        match value {
            "canonical" => Ok(Self::Canonical),
            "local" => Ok(Self::Local),
            other => refuse(
                Code::UNKNOWN_FIELD,
                format!("custody {other:?} is outside {{canonical, local}}"),
            ),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Routes {
    pub verbatim: u64,
    pub decoded_float: u64,
    pub encoded_gemm: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Fit {
    Ok {
        routes: Routes,
        /// Stored tensors of a constructed component the code does not build, as
        /// `component/key`: skipped, never loaded.
        ignored: Vec<String>,
        /// What those tensors store, in bytes.
        ignored_bytes: u64,
        device: Option<String>,
    },
    ComponentMissing {
        component: String,
        present: Vec<String>,
    },
    TensorMissing {
        component: String,
        key: String,
    },
    ShapeMismatch {
        component: String,
        key: String,
        code: Vec<u64>,
        checkpoint: Vec<u64>,
    },
    DtypeMismatch {
        component: String,
        key: String,
        required: Dtype,
        stored: Dtype,
    },
    /// No longer answered: a stored key the code does not build is skipped (`Ok.ignored`).
    ExtraTensor {
        component: String,
        key: String,
    },
    EncodingUnsupported {
        component: String,
        key: String,
        encoding: String,
    },
    EncodingUnqualified {
        encoding: String,
        device: String,
        qualified_on: Vec<String>,
    },
}

impl Fit {
    pub const CODES: [&'static str; 8] = [
        "ok",
        "component_missing",
        "tensor_missing",
        "shape_mismatch",
        "dtype_mismatch",
        "extra_tensor",
        "encoding_unsupported",
        "encoding_unqualified",
    ];

    pub fn code(&self) -> &'static str {
        match self {
            Self::Ok { .. } => Self::CODES[0],
            Self::ComponentMissing { .. } => Self::CODES[1],
            Self::TensorMissing { .. } => Self::CODES[2],
            Self::ShapeMismatch { .. } => Self::CODES[3],
            Self::DtypeMismatch { .. } => Self::CODES[4],
            Self::ExtraTensor { .. } => Self::CODES[5],
            Self::EncodingUnsupported { .. } => Self::CODES[6],
            Self::EncodingUnqualified { .. } => Self::CODES[7],
        }
    }

    pub fn is_ok(&self) -> bool {
        matches!(self, Self::Ok { .. })
    }

    /// The one line a caller shows when stored tensors were skipped: how many, how many
    /// bytes, and the first few. A truncated model reads "420 stored tensor(s), 9.8 GB".
    pub fn warning(&self) -> Option<String> {
        let Self::Ok {
            ignored,
            ignored_bytes,
            ..
        } = self
        else {
            return None;
        };
        if ignored.is_empty() {
            return None;
        }
        let more = match ignored.len().saturating_sub(WARNED_KEYS) {
            0 => String::new(),
            more => format!(", and {more} more"),
        };
        Some(format!(
            "{} stored tensor(s) the code does not build were skipped, not loaded: {} B ({}{more})",
            ignored.len(),
            ignored_bytes,
            ignored[..ignored.len().min(WARNED_KEYS)].join(", "),
        ))
    }

    pub fn text(&self) -> String {
        match self {
            Self::Ok { routes, device, .. } => {
                let mut text = format!(
                    "fits (routes: verbatim x{}, decoded_float x{}, encoded_gemm x{}",
                    routes.verbatim, routes.decoded_float, routes.encoded_gemm
                );
                if let Some(device) = device {
                    text.push_str(&format!("; encoded leaves qualified on {device}"));
                }
                text.push(')');
                if let Some(warning) = self.warning() {
                    text.push_str(&format!("; {warning}"));
                }
                text
            }
            Self::ComponentMissing { component, present } => format!(
                "the package constructs {component}; the checkpoint has {}",
                present.join(", ")
            ),
            Self::TensorMissing { component, key } => {
                format!("{component}.{key}: required by code, absent from checkpoint")
            }
            Self::ShapeMismatch {
                component,
                key,
                code,
                checkpoint,
            } => format!(
                "{component}.{key}: code {}, checkpoint {}",
                dims(code),
                dims(checkpoint)
            ),
            Self::DtypeMismatch {
                component,
                key,
                required,
                stored,
            } => format!(
                "{component}.{key}: code requires logical dtype {}, checkpoint stores {}",
                required.name(),
                stored.name()
            ),
            Self::ExtraTensor { component, key } => {
                format!("{component}.{key}: stored but not constructed")
            }
            Self::EncodingUnsupported {
                component,
                key,
                encoding,
            } => format!(
                "{component}.{key} is stored {}; this package refuses encoded leaves",
                alias_of(encoding)
            ),
            Self::EncodingUnqualified {
                encoding,
                device,
                qualified_on,
            } => format!(
                "{} has no qualification observation on {device} (qualified on: {})",
                alias_of(encoding),
                if qualified_on.is_empty() {
                    "nothing".to_string()
                } else {
                    qualified_on.join(", ")
                }
            ),
        }
    }
}

fn dims(shape: &[u64]) -> String {
    format!(
        "[{}]",
        shape
            .iter()
            .map(u64::to_string)
            .collect::<Vec<_>>()
            .join(",")
    )
}

fn alias_of(digest: &str) -> String {
    crate::registry::seeds()
        .iter()
        .find(|seed| seed.spec.object_id() == digest)
        .map(|seed| seed.alias.to_string())
        .unwrap_or_else(|| digest.to_string())
}

fn plain_digest() -> String {
    crate::registry::seeds()
        .into_iter()
        .find(|seed| seed.alias == "plain/1")
        .expect("plain/1 is in the registry")
        .spec
        .object_id()
}

/// Compare one transient census with one already-parsed CozyTensors header. No requirement
/// bytes, digest, cache key, or compatibility document is created.
pub fn fit(
    requirements: &TensorRequirements,
    header: &Header,
    custody: Custody,
    encoded_leaves: bool,
    device: Option<&str>,
    observations: Option<&CapabilityRecords>,
) -> Result<Fit> {
    header.validate(&Closure::default())?;
    let plain = plain_digest();
    let _ = custody;
    let mut routes = Routes::default();
    let (mut ignored, mut ignored_bytes) = (Vec::new(), 0u64);
    let mut encoded = Vec::new();

    for (component, required_tensors) in requirements.components() {
        let Some((_, stored_tensors)) = header
            .components
            .iter()
            .find(|(stored_component, _)| stored_component == component)
        else {
            return Ok(Fit::ComponentMissing {
                component: component.clone(),
                present: header
                    .components
                    .iter()
                    .map(|(name, _)| name.clone())
                    .collect(),
            });
        };

        for (key, requirement) in required_tensors {
            let Some((_, stored)) = stored_tensors
                .iter()
                .find(|(stored_key, _)| stored_key == key)
            else {
                return Ok(Fit::TensorMissing {
                    component: component.clone(),
                    key: key.clone(),
                });
            };
            if stored.shape != requirement.shape {
                return Ok(Fit::ShapeMismatch {
                    component: component.clone(),
                    key: key.clone(),
                    code: requirement.shape.clone(),
                    checkpoint: stored.shape.clone(),
                });
            }
            if let Some(required) = requirement.logical_dtype {
                if stored.dtype != required {
                    return Ok(Fit::DtypeMismatch {
                        component: component.clone(),
                        key: key.clone(),
                        required,
                        stored: stored.dtype,
                    });
                }
            }
            if stored.encoding == plain {
                routes.verbatim += 1;
            } else if !encoded_leaves {
                return Ok(Fit::EncodingUnsupported {
                    component: component.clone(),
                    key: key.clone(),
                    encoding: stored.encoding.clone(),
                });
            } else {
                routes.encoded_gemm += 1;
                if !encoded.contains(&stored.encoding) {
                    encoded.push(stored.encoding.clone());
                }
            }
        }

        // A stored key the code does not build is skipped, on every custody: the code may
        // build less than an older checkpoint stores (Qwen's text encoder dropped `lm_head`).
        for (key, stored) in stored_tensors {
            if required_tensors
                .iter()
                .any(|(required_key, _)| required_key == key)
            {
                continue;
            }
            ignored.push(format!("{component}/{key}"));
            for (_, part) in &stored.parts {
                ignored_bytes += match &part.body {
                    Body::Segments(objects) => objects.iter().map(|object| object.length).sum(),
                    Body::Inline(bytes) => bytes.len() as u64,
                };
            }
        }
    }

    if let Some(device) = device {
        let mut records = crate::registry::capability_records();
        if let Some(observed) = observations {
            records = records.with_observed(observed.clone());
        }
        encoded.sort();
        for encoding in encoded {
            if records.admit(&encoding, device).is_ok() {
                continue;
            }
            let mut qualified_on: Vec<String> = records
                .records
                .iter()
                .filter(|record| record.encoding == encoding)
                .map(|record| record.device.clone())
                .collect();
            qualified_on.sort();
            qualified_on.dedup();
            return Ok(Fit::EncodingUnqualified {
                encoding,
                device: device.to_string(),
                qualified_on,
            });
        }
    }

    ignored.sort();
    Ok(Fit::Ok {
        routes,
        ignored,
        ignored_bytes,
        device: device.map(str::to_string),
    })
}
