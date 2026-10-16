//! The one typed refusal every cozy surface renders: Calcifer's HTTP refusals (`{"error": …}`),
//! the CLI's `--json` output and its exit code. A run that ends failed is an answer, not a
//! refusal. `kind` is the class a program branches on;
//! `code` is the stable specific name (`idempotency_conflict`, `machine_not_found`).
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Kind {
    /// A bad invocation: unknown flag, malformed `key=value`, malformed target.
    Usage,
    /// A typed payload, schema or bounds refusal.
    Validation,
    NotFound,
    /// A missing or refused credential; the remedy names it.
    Credential,
    /// The caller may not do this here (a foreign Host/Origin or a non-loopback peer).
    Forbidden,
    /// A concurrent writer, an existing target, an idempotency key reused for another ask.
    Conflict,
    /// Not enough memory, disk or GPUs anywhere the ask may go.
    Capacity,
    /// Calcifer, a machine or the Hub is unreachable.
    Unavailable,
    /// A bug, never a user condition.
    Internal,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq, thiserror::Error)]
#[error("{code}: {message}")]
pub struct Error {
    pub kind: Kind,
    pub code: String,
    pub message: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub remedy: Option<String>,
    /// Commands to try next; at most two.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub next: Vec<String>,
    /// The originating component's own stable code (a machine's or the Hub's), when not ours.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cause: Option<String>,
}

/// The HTTP body of a refusal.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
pub struct Envelope {
    pub error: Error,
}

impl Error {
    pub fn new(kind: Kind, code: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            kind,
            code: code.into(),
            message: message.into(),
            remedy: None,
            next: vec![],
            cause: None,
        }
    }

    pub fn remedy(mut self, remedy: impl Into<String>) -> Self {
        self.remedy = Some(remedy.into());
        self
    }

    pub fn next(mut self, command: impl Into<String>) -> Self {
        if self.next.len() < 2 {
            self.next.push(command.into());
        }
        self
    }

    pub fn cause(mut self, code: impl Into<String>) -> Self {
        self.cause = Some(code.into());
        self
    }

    /// The process exit code: 2 for usage, 1 for everything else.
    pub fn exit_code(&self) -> i32 {
        if self.kind == Kind::Usage { 2 } else { 1 }
    }
}

impl Kind {
    pub fn http_status(self) -> u16 {
        match self {
            Kind::Usage | Kind::Validation => 400,
            Kind::Credential => 401,
            Kind::Forbidden => 403,
            Kind::NotFound => 404,
            Kind::Conflict => 409,
            Kind::Capacity => 507,
            Kind::Unavailable => 503,
            Kind::Internal => 500,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_envelope_round_trips_and_keeps_two_next_steps() {
        let error = Error::new(Kind::Conflict, "idempotency_conflict", "key reused")
            .remedy("use a new key")
            .next("cozy run list")
            .next("cozy run show 3")
            .next("dropped");
        let json = serde_json::to_string(&Envelope {
            error: error.clone(),
        })
        .unwrap();
        assert_eq!(
            json,
            r#"{"error":{"kind":"conflict","code":"idempotency_conflict","message":"key reused","remedy":"use a new key","next":["cozy run list","cozy run show 3"]}}"#
        );
        assert_eq!(
            serde_json::from_str::<Envelope>(&json).unwrap().error,
            error
        );
        assert_eq!(error.exit_code(), 1);
        assert_eq!(Error::new(Kind::Usage, "usage", "bad flag").exit_code(), 2);
        let forbidden = Error::new(Kind::Forbidden, "origin_not_allowed", "foreign page");
        let json = serde_json::to_value(&forbidden).unwrap();
        assert_eq!(json["kind"], "forbidden");
        assert_eq!(serde_json::from_value::<Error>(json).unwrap(), forbidden);
        assert_eq!(forbidden.kind.http_status(), 403);
        assert_eq!(forbidden.exit_code(), 1);
    }
}
