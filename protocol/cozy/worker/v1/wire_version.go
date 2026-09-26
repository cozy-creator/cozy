package workerprotov1

// WireMinor is the current release of cozy.worker.v1. Ordinary additive changes preserve the floor.
// Minor62 is the coordinated package-installation hard cut. Future additive changes preserve skew.
const WireMinor uint32 = 63

// MinCompatibleWireMinor is checked before Claim/preparation side effects.
const MinCompatibleWireMinor uint32 = 62

const AttentionKernelWireMinor uint32 = 49
const RuntimeRevisionWireMinor uint32 = 49
const MachinePublicationAuthorityWireMinor uint32 = 52
const RetainedModelResultsWireMinor uint32 = 53
const PublishedMachineCaptureWireMinor uint32 = 54
const CapturedModelDefaultsWireMinor uint32 = 56
const ModelDefaultGPUCountWireMinor uint32 = 63
const WorkspaceFencedExecutionWireMinor uint32 = 59

// RentalExecutionAdmissionWireMinor requires typed lookup absence for replay-safe idle admission.
const RentalExecutionAdmissionWireMinor uint32 = 60
const RentalKeepaliveWireMinor uint32 = 60
const RentalIdleTimeoutSeconds int64 = 900

// ExecutionLifecycleWireMinor carries the release package interface on prepare; older senders derive it once.
const ExecutionLifecycleWireMinor uint32 = 61
const MaxRentalKeepaliveRequestIDBytes = 128
