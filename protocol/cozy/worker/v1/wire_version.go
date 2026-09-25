package workerprotov1

// WireMinor is the current release of cozy.worker.v1. Ordinary additive changes preserve the floor.
// The floor is minor60; minor61 execution-lifecycle fields are additive (absent = unknown).
const WireMinor uint32 = 61

// MinCompatibleWireMinor is checked before Claim/preparation side effects.
const MinCompatibleWireMinor uint32 = 60

const AttentionKernelWireMinor uint32 = 49
const RuntimeRevisionWireMinor uint32 = 49
const MachinePublicationAuthorityWireMinor uint32 = 52
const RetainedModelResultsWireMinor uint32 = 53
const PublishedMachineCaptureWireMinor uint32 = 54
const CapturedModelDefaultsWireMinor uint32 = 56
const WorkspaceFencedExecutionWireMinor uint32 = 59

// RentalExecutionAdmissionWireMinor requires typed lookup absence for replay-safe idle admission.
const RentalExecutionAdmissionWireMinor uint32 = 60
const RentalKeepaliveWireMinor uint32 = 60
const RentalIdleTimeoutSeconds int64 = 900

// ExecutionLifecycleWireMinor carries the release package interface on prepare (required for published sets).
const ExecutionLifecycleWireMinor uint32 = 61
const MaxRentalKeepaliveRequestIDBytes = 128
