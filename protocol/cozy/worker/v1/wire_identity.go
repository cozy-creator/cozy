package workerprotov1

// WireMinor is this file set's position on the ADDITIVE linear train (03 §1.3).
// It does NOT move for a breaking in-place revision, so it is not a staleness fence.
const WireMinor uint32 = 1

// WireSchemaRev is which SHAPE this is (#530-A1). Diagnostics and ordering only.
const WireSchemaRev uint32 = 6

// SchemaDigest is THE FENCE: sha256 of the canonical `cozy.worker.v1.WireSchema/1` document.
// Carried on Claim/ClaimAck; a mismatch or an absence refuses at the handshake.
const SchemaDigest = "sha256:b5cd4430ef6d8b17e4d6d4126cd68ad132e069cae03339deb1476d928e1b3309"
