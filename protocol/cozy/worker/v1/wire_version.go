package workerprotov1

// WireMinor is the current release of cozy.worker.v1. Ordinary additive changes preserve the floor.
// The explicit pre-freeze minor38 hardcut raises it; post-freeze breaks require a new major.
const WireMinor uint32 = 40

// MinCompatibleWireMinor is checked before Claim/preparation side effects.
const MinCompatibleWireMinor uint32 = 38
