package workerprotov1

// WireMinor is this file set's position on the ADDITIVE linear train (03 §1.3).
// It does NOT move for a breaking in-place revision, so it is not a staleness fence.
const WireMinor uint32 = 0

// WireSchemaRev is which SHAPE this is (#530-A1). Diagnostics and ordering only.
const WireSchemaRev uint32 = 4

// SchemaDigest is THE FENCE: sha256 of the canonical `cozy.worker.v1.WireSchema/1` document.
// Carried on Claim/ClaimAck; a mismatch or an absence refuses at the handshake.
const SchemaDigest = "sha256:f22c1ca89c51c5c16e9d7541a3b91e9b38dd8bf423e2b619db7e6eb6cef10469"
