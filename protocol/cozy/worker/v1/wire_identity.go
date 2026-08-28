package workerprotov1

// WireMinor is this file set's position on the ADDITIVE linear train (03 §1.3).
// It does NOT move for a breaking in-place revision, so it is not a staleness fence.
const WireMinor uint32 = 2

// WireSchemaRev is which SHAPE this is (#530-A1). Diagnostics and ordering only.
const WireSchemaRev uint32 = 8

// SchemaDigest is THE FENCE: sha256 of the canonical `cozy.worker.v1.WireSchema/1` document.
// Carried on Claim/ClaimAck; a mismatch or an absence refuses at the handshake.
const SchemaDigest = "sha256:1f1f3d0d36d881000bd713bad55dd5c00414a69c663edd3662f502ea211004e7"
