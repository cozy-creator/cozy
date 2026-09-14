package workerprotov1

// WireMinor is the current release of cozy.worker.v1. Ordinary additive changes preserve the floor.
// The explicit pre-freeze minor38 hardcut raises it; post-freeze breaks require a new major.
const WireMinor uint32 = 54

// MinCompatibleWireMinor is checked before Claim/preparation side effects.
const MinCompatibleWireMinor uint32 = 44

const AttentionKernelWireMinor uint32 = 49
const RuntimeRevisionWireMinor uint32 = 49
const MachinePublicationAuthorityWireMinor uint32 = 52
const RetainedModelResultsWireMinor uint32 = 53
const PublishedMachineCaptureWireMinor uint32 = 54
