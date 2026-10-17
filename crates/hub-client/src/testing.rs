//! Helpers for tests against a real Tensorhub (the `testing` feature): the CI hub's emailed codes,
//! and a signed-up user.
use std::path::Path;
use std::time::{Duration, Instant};

use crate::{Bearer, Client, DeviceKey, Enrollment, Error};

/// The newest code the hub's development mail sender logged for `email` (its JSON log at `log`).
pub fn emailed_code(log: &Path, email: &str) -> String {
    let started = Instant::now();
    loop {
        let text = std::fs::read_to_string(log).unwrap_or_default();
        let code = text
            .lines()
            .rev()
            .filter_map(|line| serde_json::from_str::<serde_json::Value>(line).ok())
            .filter(|line| line["email"] == email)
            .find_map(|line| line["code"].as_str().map(str::to_string));
        if let Some(code) = code {
            return code;
        }
        assert!(
            started.elapsed() < Duration::from_secs(30),
            "the hub logged no code for {email}"
        );
        std::thread::sleep(Duration::from_millis(200));
    }
}

/// Enrolls a device key for `email`, reading the emailed code from the hub's log.
pub async fn sign_up(
    anonymous: &Client,
    email: &str,
    log: &Path,
) -> Result<(DeviceKey, Bearer), Error> {
    let enrollment = Enrollment::begin(anonymous, email, "test", None).await?;
    enrollment
        .finish(anonymous, &emailed_code(log, email))
        .await
}
