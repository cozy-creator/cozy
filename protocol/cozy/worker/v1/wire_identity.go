package workerprotov1

// WireMinor is this file set's position on the ADDITIVE linear train (03 §1.3).
// It does NOT move for a breaking in-place revision, so it is not a staleness fence.
const WireMinor uint32 = 2

// WireSchemaRev is which SHAPE this is (#530-A1). Diagnostics and ordering only.
const WireSchemaRev uint32 = 7

// SchemaDigest is THE FENCE: sha256 of the canonical `cozy.worker.v1.WireSchema/1` document.
// Carried on Claim/ClaimAck; a mismatch or an absence refuses at the handshake.
const SchemaDigest = "sha256:d43ccca3e09ed01435eacc3d6cfae90fc03633ceb4ee4b556736743ec78b6cd4"
