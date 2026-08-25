//! The hermetic verifier's BOUNDED DOCUMENT INTERFACE — what Tensorhub actually links
//! against, which is nothing.
//!
//! th-002's rule: "the API neither links the decoder nor parses verifier input/output into
//! an alternate semantic model." So the seam is a request document in and a response
//! document out, both canonical, both digest-fenced, carried by an unprivileged process
//! that has no store, no network, no credentials and no writable path but its own output.
//!
//! Three properties the shape buys:
//!
//! * **A partial answer is not an answer.** The response is constructed whole in memory and
//!   renamed into place; a crash leaves no file, and a truncated file fails strict canonical
//!   decode (`VERIFIER_OUTPUT_INCOMPLETE`). Nothing downstream can mint a receipt or a root
//!   from half a document.
//! * **The answer names its question.** `request` is the sha256 of the exact request bytes,
//!   so a response cannot be replayed against a different pair.
//! * **Nothing is fetched.** Every input — the code topology, the header, its closure, the
//!   pin, the registrations — arrives inside the request. The verifier resolves no digest,
//!   opens no store and reaches no network, which is why the grading verdict is decidable
//!   pre-download, before any pod money.

use crate::b64;
use crate::canon::{as_arr, as_str, Fields, Value};
use crate::compat::{CodeTopology, CompatibilityReport, Lane};
use crate::err::{refuse, Code, Result};
use crate::grade::{self, ConversionRegistration, GradeDecision, Registry};
use crate::header::{Closure, Header};
use crate::ids::{check_format, object_id, prefixed, Doc};
use crate::limits;
use crate::spec::EncodingSpec;

fn b64_list(what: &str, v: &Value) -> Result<Vec<Vec<u8>>> {
    let mut out = Vec::new();
    for x in as_arr("VerifyRequest", what, v)? {
        out.push(b64::decode(as_str("VerifyRequest", what, x)?)?);
    }
    Ok(out)
}

/// Inner documents ride as base64 of their EXACT canonical bytes, so their object ids
/// survive the trip. Re-nesting them as values would re-derive the bytes, and re-derived
/// bytes are how a digest quietly stops matching.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VerifyRequest {
    pub code_topology: Vec<u8>,
    pub header: Vec<u8>,
    pub closure: Vec<Vec<u8>>,
    pub registrations: Vec<Vec<u8>>,
    pub lane: Lane,
    pub pin: Vec<String>,
    pub known_markers: Vec<String>,
}

impl Doc for VerifyRequest {
    const FORMAT: &'static str = "tensorfs.verify_request/1";
    const MAX_BYTES: usize = limits::DOC_MAX_BYTES;

    fn from_value(v: &Value) -> Result<Self> {
        let mut f = Fields::new("VerifyRequest", v)?;
        check_format(
            "VerifyRequest",
            as_str("VerifyRequest", "format", f.req("format")?)?,
            Self::FORMAT,
        )?;
        let closure = b64_list("closure", f.req("closure")?)?;
        let code_topology = b64::decode(f.req_str("code_topology")?)?;
        let header = b64::decode(f.req_str("header")?)?;
        let mut known_markers = Vec::new();
        for m in as_arr("VerifyRequest", "known_markers", f.req("known_markers")?)? {
            known_markers.push(as_str("VerifyRequest", "known_markers", m)?.to_string());
        }
        let lane = Lane::parse(f.req_str("lane")?)?;
        let mut pin = Vec::new();
        for p in as_arr("VerifyRequest", "pin", f.req("pin")?)? {
            pin.push(prefixed("pin", as_str("VerifyRequest", "pin", p)?)?);
        }
        let registrations = b64_list("registrations", f.req("registrations")?)?;
        f.done()?;
        Ok(VerifyRequest {
            code_topology,
            header,
            closure,
            registrations,
            lane,
            pin,
            known_markers,
        })
    }

    fn to_value(&self) -> Value {
        Value::obj(vec![
            (
                "closure",
                Value::arr(
                    self.closure
                        .iter()
                        .map(|c| Value::str(b64::encode(c)))
                        .collect(),
                ),
            ),
            (
                "code_topology",
                Value::str(b64::encode(&self.code_topology)),
            ),
            ("format", Value::str(Self::FORMAT)),
            ("header", Value::str(b64::encode(&self.header))),
            (
                "known_markers",
                Value::arr(
                    self.known_markers
                        .iter()
                        .map(|m| Value::str(m.clone()))
                        .collect(),
                ),
            ),
            ("lane", Value::str(self.lane.name())),
            (
                "pin",
                Value::arr(self.pin.iter().map(|p| Value::str(p.clone())).collect()),
            ),
            (
                "registrations",
                Value::arr(
                    self.registrations
                        .iter()
                        .map(|r| Value::str(b64::encode(r)))
                        .collect(),
                ),
            ),
        ])
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VerifyResponse {
    /// sha256 of the exact request bytes. A response is about ONE question.
    pub request: String,
    pub report: Vec<u8>,
    pub decision: Vec<u8>,
    pub verifier_build: String,
}

impl Doc for VerifyResponse {
    const FORMAT: &'static str = "tensorfs.verify_response/1";
    const MAX_BYTES: usize = limits::DOC_MAX_BYTES;

    fn from_value(v: &Value) -> Result<Self> {
        let mut f = Fields::new("VerifyResponse", v)?;
        check_format(
            "VerifyResponse",
            as_str("VerifyResponse", "format", f.req("format")?)?,
            Self::FORMAT,
        )?;
        let decision = b64::decode(f.req_str("decision")?)?;
        let report = b64::decode(f.req_str("report")?)?;
        let request = prefixed("VerifyResponse.request", f.req_str("request")?)?;
        let verifier_build = f.req_str("verifier_build")?.to_string();
        f.done()?;
        Ok(VerifyResponse {
            request,
            report,
            decision,
            verifier_build,
        })
    }

    fn to_value(&self) -> Value {
        Value::obj(vec![
            ("decision", Value::str(b64::encode(&self.decision))),
            ("format", Value::str(Self::FORMAT)),
            ("report", Value::str(b64::encode(&self.report))),
            ("request", Value::str(self.request.clone())),
            ("verifier_build", Value::str(self.verifier_build.clone())),
        ])
    }
}

/// The whole verifier, as one pure function of request BYTES to response BYTES. No I/O
/// lives in here at all, which is what makes "sandbox escape or network access mints
/// nothing" a structural statement rather than a policy: there is nothing to escape to.
pub fn run(request_bytes: &[u8]) -> Result<Vec<u8>> {
    let req = VerifyRequest::parse(request_bytes)?;
    let code = CodeTopology::parse(&req.code_topology)?;
    let header = Header::parse(&req.header)?;
    let mut closure = Closure::default();
    for c in &req.closure {
        closure.insert(EncodingSpec::parse(c)?);
    }
    let mut registry = Registry::default();
    for r in &req.registrations {
        registry
            .registrations
            .push(ConversionRegistration::parse(r)?);
    }
    // `resolve` is check-then-apply-then-recheck: a registration is only allowed to lift a
    // mismatch when the MAPPED topology satisfies outright. The over-constraint guard runs
    // inside it, so a covered cast-only mismatch can never leave here INCOMPATIBLE.
    let (report, g) = grade::resolve(
        &code,
        &header,
        &closure,
        &req.pin,
        req.lane,
        &req.known_markers,
        &registry,
    )?;
    let decision = GradeDecision {
        report: report.object_id(),
        grade: g,
    };
    // Self-check before anything leaves: a document whose own bytes do not re-parse is a
    // document the consumer will reject, and it is far better to learn that here than to
    // hand the hub something unreadable. (Found live: a refusal message carrying an em dash
    // built a `reason` the printable-ASCII canonical layer could not encode.)
    let report_bytes = report.canonical_bytes();
    let decision_bytes = decision.canonical_bytes();
    CompatibilityReport::parse(&report_bytes)?;
    GradeDecision::parse(&decision_bytes)?;
    Ok(VerifyResponse {
        request: object_id(request_bytes),
        report: report_bytes,
        decision: decision_bytes,
        verifier_build: format!("tensorfs-verifier/{}", env!("CARGO_PKG_VERSION")),
    }
    .canonical_bytes())
}

/// The consumer's side of the seam. Anything that is not a complete canonical response
/// about THIS request is `VERIFIER_OUTPUT_INCOMPLETE` — one code covering truncation,
/// garbage, an empty file and a replayed answer, because from the hub's seat they are the
/// same event: no verdict arrived.
pub fn consume(
    request_bytes: &[u8],
    response_bytes: &[u8],
) -> Result<(CompatibilityReport, GradeDecision)> {
    let resp = match VerifyResponse::parse(response_bytes) {
        Ok(r) => r,
        Err(e) => {
            return refuse(
                Code::VERIFIER_OUTPUT_INCOMPLETE,
                format!(
                    "the verifier's {} B output is not a complete canonical response ({}): a \
                     crash, a partial write or a truncated pipe mints nothing",
                    response_bytes.len(),
                    e.code.as_str()
                ),
            )
        }
    };
    let want = object_id(request_bytes);
    if resp.request != want {
        return refuse(
            Code::VERIFIER_OUTPUT_INCOMPLETE,
            format!(
                "the response answers request {} and this is request {want} — a verdict is \
                 about one exact pair",
                resp.request
            ),
        );
    }
    let report = CompatibilityReport::parse(&resp.report)?;
    let decision = GradeDecision::parse(&resp.decision)?;
    if decision.report != report.object_id() {
        return refuse(
            Code::VERIFIER_OUTPUT_INCOMPLETE,
            "the decision names a different report than the one delivered",
        );
    }
    Ok((report, decision))
}
