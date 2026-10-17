//! The machine API as OpenAPI 3.1, generated from the types in [`super`]. The spec is committed
//! at `crates/cozy-proto/openapi/machine.json`; a test fails when it drifts. The functions here
//! only describe the routes: TensorD serves them and the client in [`super::client`] calls them.
#![allow(clippy::needless_raw_string_hashes)]

use utoipa::{
    Modify, OpenApi,
    openapi::security::{ApiKey, ApiKeyValue, SecurityScheme},
};

use super::*;
use crate::error::{Envelope, Error, Kind};

/// The API's OpenAPI document.
#[derive(OpenApi)]
#[openapi(
    info(
        title = "cozy machine API",
        version = "1",
        description = "One machine's API on its pinned-TLS port (design/machine-http-api.md). Every request \
                       carries `authorization: Cozy-Cap <token>`; refusals are the error envelope; unknown \
                       fields are ignored."
    ),
    modifiers(&CozyCap),
    security(("cozy_cap" = [])),
    paths(health, status, keepalive, submit, events, cancel, output, triage, log, head_object, put_object),
    components(schemas(Envelope, Error, Kind))
)]
pub struct MachineApi;

/// The committed spec's text.
pub fn spec() -> String {
    let mut text = MachineApi::openapi()
        .to_pretty_json()
        .expect("the spec serializes");
    text.push('\n');
    text
}

struct CozyCap;

impl Modify for CozyCap {
    fn modify(&self, openapi: &mut utoipa::openapi::OpenApi) {
        let header = ApiKeyValue::with_description(
            "authorization",
            "`Cozy-Cap <token>`: machine scope does everything as its signer; run scope reads one \
             run's events and its granted outputs.",
        );
        openapi
            .components
            .get_or_insert_with(Default::default)
            .add_security_scheme("cozy_cap", SecurityScheme::ApiKey(ApiKey::Header(header)));
    }
}

/// Answers 204 when nothing authorizes the request, 401 when something does: a probe that the
/// listener is up.
#[utoipa::path(get, path = path::HEALTH, security(()), responses(
    (status = 204, description = "Up"),
    (status = 401, description = "The request carried a credential"),
))]
pub fn health() {}

/// The machine's picture. Without a machine-scope cap: identity, version, capabilities and the
/// sealed readiness receipt only (what a Hub reads). With one: all of it, and with `Accept:
/// text/event-stream` an SSE stream of `status` events, each the whole frame, sent again when it
/// changes.
#[utoipa::path(get, path = path::STATUS, security((), ("cozy_cap" = [])), responses(
    (status = 200, description = "The frame, or SSE `event: status` frames", content(
        (StatusFrame = "application/json"),
        (StatusFrame = "text/event-stream"),
    )),
    (status = 401, description = "A credential this machine does not admit", body = Envelope),
))]
pub fn status() {}

/// Starts a rental's idle clock over and answers the frame with the new deadline.
#[utoipa::path(post, path = path::KEEPALIVE, responses(
    (status = 200, description = "The frame", body = StatusFrame),
    (status = 409, description = "This machine has no idle deadline", body = Envelope),
))]
pub fn keepalive() {}

/// Submits a run, idempotent on its id per signer: the same id with the same spec answers its
/// state; with another spec, `run_id_conflict`. An update's activation and a rental's idle
/// release wait for it.
#[utoipa::path(post, path = path::RUNS, request_body = RunRequest, responses(
    (status = 201, description = "Accepted", body = RunState),
    (status = 200, description = "Already accepted", body = RunState),
    (status = 400, description = "An invalid spec", body = Envelope),
    (status = 409, description = "`run_id_conflict`, or a refusal of the run", body = Envelope),
    (status = 503, description = "`machine_updating` or `machine_released`", body = Envelope),
))]
pub fn submit() {}

/// The run's log after `Last-Event-ID` as SSE: first `state` (the snapshot), then `progress`,
/// `product`, `log`, `call`, `gpu` and finally `outcome`, after which the stream ends. Durable
/// events carry `id: <sequence>`; live ones none. Closing it never cancels the run.
#[utoipa::path(get, path = path::EVENTS, params(
    ("id" = String, Path, description = "The run id"),
    ("Last-Event-ID" = Option<u64>, Header, description = "Resume after this sequence"),
    ("after" = Option<u64>, Query, description = "As `Last-Event-ID`"),
), responses(
    (status = 200, description = "SSE: `event: <variant>`, data the whole event", body = RunEvent,
        content_type = "text/event-stream"),
    (status = 404, description = "No such run", body = Envelope),
))]
pub fn events() {}

/// Cancels the run. An id the machine never received is recorded canceled, so a late submit of
/// it answers that.
#[utoipa::path(post, path = path::CANCEL, params(("id" = String, Path, description = "The run id")),
    responses((status = 200, description = "Its state", body = RunState)))]
pub fn cancel() {}

/// An output's current bytes, readable while it grows (each revision is a `product` event).
#[utoipa::path(get, path = path::OUTPUT, params(
    ("id" = String, Path, description = "The run id"),
    ("output" = String, Path, description = "The output's name"),
    ("index" = Option<u32>, Query, description = "A list item's 1-based index; absent for a single output"),
    ("member" = Option<String>, Query, description = "One file of a tree output, by its manifest path"),
    ("Range" = Option<String>, Header, description = "One range, `bytes=first-[last]`"),
    ("If-Match" = Option<String>, Header, description = "`\"<rev>\"`: refuse once the output moved past it"),
), responses(
    (status = 200, description = "All of it", body = Vec<u8>, content_type = "application/octet-stream",
        headers(("ETag" = String, description = "`\"<rev>\"`"))),
    (status = 206, description = "The range", body = Vec<u8>, content_type = "application/octet-stream",
        headers(("ETag" = String, description = "`\"<rev>\"`"),
                ("Content-Range" = String, description = "`bytes first-last/length`"))),
    (status = 404, description = "No such run or output", body = Envelope),
    (status = 412, description = "`output_revision_moved`", body = Envelope),
    (status = 416, description = "`range_unsatisfiable`", body = Envelope),
))]
pub fn output() {}

/// A failed attempt's retained triage bundle.
#[utoipa::path(get, path = path::TRIAGE, params(("id" = String, Path, description = "The run id")),
    responses(
        (status = 200, description = "The bundle", body = Object, content_type = "application/json"),
        (status = 404, description = "None retained", body = Envelope),
    ))]
pub fn triage() {}

/// A log the machine keeps (`tensorfs-transport`, `memory`, `tensord`).
#[utoipa::path(get, path = path::LOG, params(
    ("name" = String, Path, description = "The log's name"),
    ("tail" = Option<u64>, Query, description = "Its newest bytes, from a line start; absent: all"),
), responses(
    (status = 200, description = "The log", body = String, content_type = "text/plain"),
    (status = 404, description = "No such log", body = Envelope),
))]
pub fn log() {}

/// The byte ranges of the object the machine holds for this signer, in `Cozy-Held`. A stored
/// object is held whole, and asking renews it, so a run sent next finds it.
#[utoipa::path(head, path = path::OBJECT, params(
    ("digest" = String, Path, description = "`sha256:<hex>`"),
), responses(
    (status = 200, description = "What is held", headers(
        ("Cozy-Held" = String, description = "`first-last,…` inclusive; empty: nothing"))),
))]
pub fn head_object() {}

/// Lands one part of the object: the body is `Content-Range`'s bytes (without it, the whole
/// object). Parts may go side by side, in any order. Once the held ranges cover the object, the
/// machine verifies its sha256 and stores it for the signer; bytes that differ are dropped
/// (`object_digest_mismatch`).
#[utoipa::path(put, path = path::OBJECT, params(
    ("digest" = String, Path, description = "`sha256:<hex>`"),
    ("Content-Range" = Option<String>, Header, description = "`bytes first-last/length`"),
), request_body(content = Vec<u8>, content_type = "application/octet-stream"), responses(
    (status = 204, description = "Landed", headers(
        ("Cozy-Held" = String, description = "What is held now"))),
    (status = 400, description = "`object_digest_mismatch`, `object_length_exceeded`, an invalid range",
        body = Envelope),
    (status = 507, description = "`machine_disk_full`", body = Envelope),
))]
pub fn put_object() {}

#[cfg(test)]
mod tests {
    /// The committed spec is what the types generate.
    #[test]
    fn the_committed_spec_is_current() {
        let committed = concat!(env!("CARGO_MANIFEST_DIR"), "/openapi/machine.json");
        let current = std::fs::read_to_string(committed).unwrap_or_default();
        assert!(
            current == super::spec(),
            "{committed} drifted from the types; regenerate it with \
             `cargo run -p cozy-proto --example openapi > crates/cozy-proto/openapi/machine.json`"
        );
    }
}
