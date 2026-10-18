# Front door

One TLS listener (`api/server.rs`): tokio-rustls offering ALPN `h2, http/1.1`, hyper-util's auto
builder and one axum `Router` (`api/http.rs`) serving the machine API (`MACHINE-API.md`) and
`GET /v1/health` (204, or 401 if an `Authorization` header is sent). HTTP/2 connections are PINGed
every 20 s and dropped after 20 s unanswered; TCP keepalive judges HTTP/1.1 lanes the same way.
The readiness receipt is `GET /v1/status`'s `receipt`, read without a capability.
`machine_api.rs` (`NativeBackend`) implements the backend over the journal, TensorFS store and
installer; the API owns no scheduler, journal or store. Backend calls run in `spawn_blocking`; an
observer's disconnect never cancels a run.

## Identity and configuration (`identity.rs`)

`tensord serve --machine-config <json> --listen <addr>` (both or neither).

`MachineConfig` (relative paths resolve against the config file):
- `worker_id`: printable ASCII, at most 256 bytes.
- `identity_directory`: owned, mode 0700. Holds `identity.json` (P256 leaf, key, worker id,
  0600) under an exclusive `identity.lock`.
- `authorized_keys_file`: `{"keys": [base64url Ed25519]}`, 1..256 keys, not group/world writable.
- `readiness_hmac_key_file`: `{"key_b64url": ...}`, mode 0600, 32..4096 bytes.

Rules: an invalid or mismatched retained identity refuses startup and is never replaced. Each
process gets a fresh boot UUID and keeps the same leaf. Key file changes apply on restart.
The readiness receipt is `{payload, hmac_sha256}`, where the HMAC covers
`cozy.pod-readiness/1\0` + payload. After binding, the service writes `api-ready.json`
(`address`, `worker_id`, `boot_id`, `cert_pem`) to the state root.

## Install (`install.rs`)

`prepare_uploaded(InstallerConfig, UploadedPackage)` materializes verified descriptors into a
private stage and runs `tensord_client.installer install-captured` (see `PACKAGE-BRIDGE.md`).
- No wall-clock kill. Stdout is capped at 8 MiB.
- A failure `{kind: "install_failed", code, detail}` is refused with its code.
- It returns `PreparedGeneration`: a 32-hex identity, the record, interface bytes, and a shared
  `.hold` lock.
