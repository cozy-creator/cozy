package workerprotov1

// WireMinor is this file set's position on the ADDITIVE linear train (03 §1.3).
// It does NOT move for a breaking in-place revision, so it is not a staleness fence.
const WireMinor uint32 = 0

// WireSchemaRev is which SHAPE this is (#530-A1). Diagnostics and ordering only.
const WireSchemaRev uint32 = 2

// SchemaDigest is THE FENCE: sha256 of the canonical `cozy.worker.v1.WireSchema/1` document.
// Carried on Claim/ClaimAck; a mismatch or an absence refuses at the handshake.
const SchemaDigest = "sha256:1946a5d9c1118632708074f500221eaabc31f1837d230635d44c2b71becf6ce6"
