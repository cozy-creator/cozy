# Owner memory reclaim qualification client

Provide a bounded qualification command using the existing recorded rental identity,
TLS pin and machine-v1 Client. The machine-scope signed cap stays inside the pinned HTTPS
client. It sends only `POST /v1/machine/memory/reclaim`; it never looks up Hub state, replaces
the personal daemon, reads environment credentials, emits a token or controls a run.
Older images return an operation-only unsupported error. The Rust maintenance endpoint is
cozy-machine PR46; physical GPU qualification requires actual before/after free-memory receipts.
