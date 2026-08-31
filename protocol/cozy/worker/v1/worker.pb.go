// cozy.worker.v1 — the RecordOwner<->worker wire contract. Schema only; no behavior.
//
// Authority: tracker-v2/worker-protocol/01-wire-schema.md (schema laws), 02-lifecycle.md
// (behavior), 03-versioning-conformance.md (versioning), rev-2-dynamic-serving.md (THIS
// revision). Issue: tracker/tensorhub/th-024.
//
// ORIENTATION: the WORKER (cozy-runtime's torch-free machine control process) HOSTS this
// service; the RECORDOWNER (the Cozy daemon, a pod's PodRecordOwner, or a
// tensorhub orchestrator-shard via th-007) DIALS it. The orientation is IDENTICAL local and
// remote; only channel establishment differs: a Unix socket (Windows: loopback) co-resident,
// TLS over a private network or authenticated overlay remotely. A worker control port is NEVER
// required to be publicly exposed for topology symmetry.
//
// VERSIONING: the MAJOR is the package path (`cozy.worker.v1`). The MINOR is `wire_minor`, an
// additive linear-train number declared at Claim/ClaimAck — never a negotiation. A future
// breaking change MUST use a new package major (`cozy.worker.v2`); it must not revise v1 in
// place. R7 IS AN AUTHORING RULE ONLY: `reserved` numbers and names are compiler-enforced
// tombstones against reuse; ordinary proto3 decoders do not refuse them on the wire and no
// runtime polices them. This file includes the final pre-release v1 hardcut; all generated
// bindings and fixtures were regenerated together before v1 shipped.
//
// IDENTITY IS CANONICAL BYTES, PROTOBUF IS TRANSPORT: no digest is ever computed over
// protobuf-marshaled bytes. A meaning-fencing digest is the SHA-256 of a DOCUMENT's exact
// canonical bytes (RFC 8785-shaped canonical JSON under tfs-013's writer). Where the fence
// requires byte agreement the canonical bytes travel IN the message as a `bytes` field instead
// of a structured body; the receiver recomputes sha256 over the resident bytes BEFORE parsing a
// single field. DIGEST CLASSES: (a) document identity over canonical JSON bytes; (b) content
// digest over opaque artifact bytes. On the wire a class-(a)/(b) digest is raw 32-byte `bytes`;
// inside a canonical document the same digest is spelled `"sha256:<64 lowercase hex>"`.
// BYTES NAMING LAW: a `bytes` field named `digest`, `*_digest`, or repeated `*_digests` IS a
// 32-byte SHA-256; every other `bytes` field is an opaque payload (base64 in documents).
//
// DOCUMENT VERSIONS. Every current pre-release document is `/1`. A digest-fenced document is NOT
// additively versioned: an unknown key REFUSES, and shape changes hardcut the `/1` definition
// across every writer, reader, and stored byte together. The canonical `format` tag is the
// message's full name plus `/1`; there are no compatibility readers or version aliases.
//
// THE ENVELOPE (every message, fields 1-3): the ownership + boot fence, checked BEFORE any body
// field is read, in this order:
//   1 `record_owner_epoch`        - durable monotonic AUTHORITY generation, minted by whatever
//                                   grants ownership. Older than the accepted claim => dropped.
//   2 `control_stream_generation` - worker-minted, incremented for EVERY accepted control
//                                   stream; a frame from a superseded stream is dropped.
//   3 `worker_boot_id`            - minted at worker-process boot, never reused; a frame
//                                   addressed to a dead boot is dropped.
// FIELD 4 IS RESERVED IN EVERY MESSAGE (#446): the old global `executor_incarnation` is GONE.
// The worker BOOT generation is machine truth; each placement's EXECUTOR generation is placement
// truth (`PlacementStatus.executor_generation`) and rides attempt-scoped facts, not the
// envelope. The attempt fence triple is (request_id, attempt_ordinal, invocation_spec_digest).
//
// PLACEMENTS (#481; renamed from DEPLOYMENTS — tensorhub's "deployment" is an id-less semantic
// tuple and the collision was real). `placement_id` is the RecordOwner-minted routing + journal
// key for one hosted assignment. A placement's fully resolved immutable definition lives directly
// inside PlacementSet. Its identity is `(placement_set_digest, placement_id)`; there is no
// independent PlacementSpec document or digest. The worker never resolves a mutable release id,
// so two workers handed the same set converge to the same bytes or fault typed. LAUNCH ENFORCES
// len(placements) <= 1: a longer set is a typed refusal (FAULT_KIND_PLACEMENT_SET_UNSUPPORTED)
// with the desired state UNAPPLIED.
//
// TWO AXES, NOT ONE ENUM (#473/#482): a placement's convergence is MaterializationState x
// ServingState. A single phase cannot represent staged-on-disk x draining independently, which
// is exactly the state an outgoing spec holds under fallback-retention (#474). Machine
// lifecycle is `WorkerPhase`, pulled out of the placement enum; the old `IntakeState` is retired
// with both of its uses. "prepared"/"warming"/"ready" are retired as state words.
//
// ONE ADMISSION FENCE (#472e/#482/#486c): per-placement `attempt_credits` are DELETED — N
// counters over ONE serialized device advertise N x the real capacity, and the defect is
// arithmetic. Execution capacity is a WORKER property (`admission_generation` +
// `admission_state` + `available_attempt_slots`); dispatchability is a PLACEMENT property (the
// serving axis + dispatchable_binding_digests). Per-placement `readiness_epoch` is DELETED, not
// renamed: one fence, not two to keep consistent.
//
// EVERY OFFER GETS A JOURNALED OUTCOME (#472f/#480b): an AttemptOffer receives either
// AttemptAccepted or a JOURNALED AttemptOutcome(REFUSED) — never silence and never an
// unjournaled decline (there is no AttemptDeclined message to add). A pre-execution refusal
// consumes the attempt_ordinal and zero billed execution budget; the billing fact is the
// STRUCTURAL bit AttemptOutcomeBody.execution_started, never a cause-code allowlist.
//
// ONEOF-LAST RULE (transport determinism only, never identity): every oneof carries the highest
// field numbers in its message.
//
// ENUM VALUE NAMING: every value is prefixed with its enum's SCREAMING_SNAKE type name. proto3
// scopes enum values at PACKAGE level, so the unprefixed spellings in the rev document
// (OPEN/FAILED/DRAINING/OFFLINE) would collide across MaterializationState, ServingState,
// WorkerPhase and AdmissionState and fail to compile. NUMBERS are the normative part.
//
// THINNING ("no writer, no field"): the purge lane is a reservation — private-artifact purge
// (cr-013/tfs-014, post-M9) rides Control as a durable command/acknowledgement pair when it
// lands, holding oneof slot 14 on both frames. `AppliedAdapter` (cr-010) and `DrainPolicy`
// (th-007) stay reservations. The retired desired-state artifact delegation remains absent;
// ArtifactSink host durability and transfers instead use the explicit host-only frames below.

// Code generated by protoc-gen-go. DO NOT EDIT.
// versions:
// 	protoc-gen-go v1.36.11
// 	protoc        v7.35.1
// source: cozy/worker/v1/worker.proto

package workerprotov1

import (
	protoreflect "google.golang.org/protobuf/reflect/protoreflect"
	protoimpl "google.golang.org/protobuf/runtime/protoimpl"
	reflect "reflect"
	sync "sync"
	unsafe "unsafe"
)

const (
	// Verify that this generated code is sufficiently up-to-date.
	_ = protoimpl.EnforceVersion(20 - protoimpl.MinVersion)
	// Verify that runtime/protoimpl is sufficiently up-to-date.
	_ = protoimpl.EnforceVersion(protoimpl.MaxVersion - 20)
)

type LocalDownloadKind int32

const (
	LocalDownloadKind_LOCAL_DOWNLOAD_KIND_UNSPECIFIED      LocalDownloadKind = 0
	LocalDownloadKind_LOCAL_DOWNLOAD_KIND_PROJECT_WHEEL    LocalDownloadKind = 1
	LocalDownloadKind_LOCAL_DOWNLOAD_KIND_DEPENDENCY_WHEEL LocalDownloadKind = 2
	LocalDownloadKind_LOCAL_DOWNLOAD_KIND_MODEL_MANIFEST   LocalDownloadKind = 3
	LocalDownloadKind_LOCAL_DOWNLOAD_KIND_MODEL_OBJECT     LocalDownloadKind = 4
)

// Enum value maps for LocalDownloadKind.
var (
	LocalDownloadKind_name = map[int32]string{
		0: "LOCAL_DOWNLOAD_KIND_UNSPECIFIED",
		1: "LOCAL_DOWNLOAD_KIND_PROJECT_WHEEL",
		2: "LOCAL_DOWNLOAD_KIND_DEPENDENCY_WHEEL",
		3: "LOCAL_DOWNLOAD_KIND_MODEL_MANIFEST",
		4: "LOCAL_DOWNLOAD_KIND_MODEL_OBJECT",
	}
	LocalDownloadKind_value = map[string]int32{
		"LOCAL_DOWNLOAD_KIND_UNSPECIFIED":      0,
		"LOCAL_DOWNLOAD_KIND_PROJECT_WHEEL":    1,
		"LOCAL_DOWNLOAD_KIND_DEPENDENCY_WHEEL": 2,
		"LOCAL_DOWNLOAD_KIND_MODEL_MANIFEST":   3,
		"LOCAL_DOWNLOAD_KIND_MODEL_OBJECT":     4,
	}
)

func (x LocalDownloadKind) Enum() *LocalDownloadKind {
	p := new(LocalDownloadKind)
	*p = x
	return p
}

func (x LocalDownloadKind) String() string {
	return protoimpl.X.EnumStringOf(x.Descriptor(), protoreflect.EnumNumber(x))
}

func (LocalDownloadKind) Descriptor() protoreflect.EnumDescriptor {
	return file_cozy_worker_v1_worker_proto_enumTypes[0].Descriptor()
}

func (LocalDownloadKind) Type() protoreflect.EnumType {
	return &file_cozy_worker_v1_worker_proto_enumTypes[0]
}

func (x LocalDownloadKind) Number() protoreflect.EnumNumber {
	return protoreflect.EnumNumber(x)
}

// Deprecated: Use LocalDownloadKind.Descriptor instead.
func (LocalDownloadKind) EnumDescriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{0}
}

type Posture int32

const (
	Posture_POSTURE_UNSPECIFIED Posture = 0
	Posture_POSTURE_ACCEPTING   Posture = 1
	Posture_POSTURE_DRAINING    Posture = 2
)

// Enum value maps for Posture.
var (
	Posture_name = map[int32]string{
		0: "POSTURE_UNSPECIFIED",
		1: "POSTURE_ACCEPTING",
		2: "POSTURE_DRAINING",
	}
	Posture_value = map[string]int32{
		"POSTURE_UNSPECIFIED": 0,
		"POSTURE_ACCEPTING":   1,
		"POSTURE_DRAINING":    2,
	}
)

func (x Posture) Enum() *Posture {
	p := new(Posture)
	*p = x
	return p
}

func (x Posture) String() string {
	return protoimpl.X.EnumStringOf(x.Descriptor(), protoreflect.EnumNumber(x))
}

func (Posture) Descriptor() protoreflect.EnumDescriptor {
	return file_cozy_worker_v1_worker_proto_enumTypes[1].Descriptor()
}

func (Posture) Type() protoreflect.EnumType {
	return &file_cozy_worker_v1_worker_proto_enumTypes[1]
}

func (x Posture) Number() protoreflect.EnumNumber {
	return protoreflect.EnumNumber(x)
}

// Deprecated: Use Posture.Descriptor instead.
func (Posture) EnumDescriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{1}
}

// Machine lifecycle, pulled out of the placement enum (#482).
type WorkerPhase int32

const (
	WorkerPhase_WORKER_PHASE_UNSPECIFIED WorkerPhase = 0
	WorkerPhase_WORKER_PHASE_BOOTING     WorkerPhase = 1
	WorkerPhase_WORKER_PHASE_ONLINE      WorkerPhase = 2
	WorkerPhase_WORKER_PHASE_DRAINING    WorkerPhase = 3
	WorkerPhase_WORKER_PHASE_FAILED      WorkerPhase = 4
)

// Enum value maps for WorkerPhase.
var (
	WorkerPhase_name = map[int32]string{
		0: "WORKER_PHASE_UNSPECIFIED",
		1: "WORKER_PHASE_BOOTING",
		2: "WORKER_PHASE_ONLINE",
		3: "WORKER_PHASE_DRAINING",
		4: "WORKER_PHASE_FAILED",
	}
	WorkerPhase_value = map[string]int32{
		"WORKER_PHASE_UNSPECIFIED": 0,
		"WORKER_PHASE_BOOTING":     1,
		"WORKER_PHASE_ONLINE":      2,
		"WORKER_PHASE_DRAINING":    3,
		"WORKER_PHASE_FAILED":      4,
	}
)

func (x WorkerPhase) Enum() *WorkerPhase {
	p := new(WorkerPhase)
	*p = x
	return p
}

func (x WorkerPhase) String() string {
	return protoimpl.X.EnumStringOf(x.Descriptor(), protoreflect.EnumNumber(x))
}

func (WorkerPhase) Descriptor() protoreflect.EnumDescriptor {
	return file_cozy_worker_v1_worker_proto_enumTypes[2].Descriptor()
}

func (WorkerPhase) Type() protoreflect.EnumType {
	return &file_cozy_worker_v1_worker_proto_enumTypes[2]
}

func (x WorkerPhase) Number() protoreflect.EnumNumber {
	return protoreflect.EnumNumber(x)
}

// Deprecated: Use WorkerPhase.Descriptor instead.
func (WorkerPhase) EnumDescriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{2}
}

// Axis 1 of a placement's convergence: what is on disk.
type MaterializationState int32

const (
	MaterializationState_MATERIALIZATION_STATE_UNSPECIFIED   MaterializationState = 0
	MaterializationState_MATERIALIZATION_STATE_ABSENT        MaterializationState = 1
	MaterializationState_MATERIALIZATION_STATE_MATERIALIZING MaterializationState = 2
	MaterializationState_MATERIALIZATION_STATE_STAGED        MaterializationState = 3 // asserts the installed receipt digest MATCHED
	MaterializationState_MATERIALIZATION_STATE_FAILED        MaterializationState = 4
)

// Enum value maps for MaterializationState.
var (
	MaterializationState_name = map[int32]string{
		0: "MATERIALIZATION_STATE_UNSPECIFIED",
		1: "MATERIALIZATION_STATE_ABSENT",
		2: "MATERIALIZATION_STATE_MATERIALIZING",
		3: "MATERIALIZATION_STATE_STAGED",
		4: "MATERIALIZATION_STATE_FAILED",
	}
	MaterializationState_value = map[string]int32{
		"MATERIALIZATION_STATE_UNSPECIFIED":   0,
		"MATERIALIZATION_STATE_ABSENT":        1,
		"MATERIALIZATION_STATE_MATERIALIZING": 2,
		"MATERIALIZATION_STATE_STAGED":        3,
		"MATERIALIZATION_STATE_FAILED":        4,
	}
)

func (x MaterializationState) Enum() *MaterializationState {
	p := new(MaterializationState)
	*p = x
	return p
}

func (x MaterializationState) String() string {
	return protoimpl.X.EnumStringOf(x.Descriptor(), protoreflect.EnumNumber(x))
}

func (MaterializationState) Descriptor() protoreflect.EnumDescriptor {
	return file_cozy_worker_v1_worker_proto_enumTypes[3].Descriptor()
}

func (MaterializationState) Type() protoreflect.EnumType {
	return &file_cozy_worker_v1_worker_proto_enumTypes[3]
}

func (x MaterializationState) Number() protoreflect.EnumNumber {
	return protoreflect.EnumNumber(x)
}

// Deprecated: Use MaterializationState.Descriptor instead.
func (MaterializationState) EnumDescriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{3}
}

// Axis 2 of a placement's convergence: what it will take. "prepared"/"warming"/"ready" are
// RETIRED as state words (#482) — "warm" had five meanings across these documents (page cache,
// resident weights, a live process, an accepting placement, a model that has run once) and the
// axes now name exactly one thing each.
type ServingState int32

const (
	ServingState_SERVING_STATE_UNSPECIFIED  ServingState = 0
	ServingState_SERVING_STATE_OFFLINE      ServingState = 1
	ServingState_SERVING_STATE_ACTIVATING   ServingState = 2
	ServingState_SERVING_STATE_DISPATCHABLE ServingState = 3
	ServingState_SERVING_STATE_DRAINING     ServingState = 4
)

// Enum value maps for ServingState.
var (
	ServingState_name = map[int32]string{
		0: "SERVING_STATE_UNSPECIFIED",
		1: "SERVING_STATE_OFFLINE",
		2: "SERVING_STATE_ACTIVATING",
		3: "SERVING_STATE_DISPATCHABLE",
		4: "SERVING_STATE_DRAINING",
	}
	ServingState_value = map[string]int32{
		"SERVING_STATE_UNSPECIFIED":  0,
		"SERVING_STATE_OFFLINE":      1,
		"SERVING_STATE_ACTIVATING":   2,
		"SERVING_STATE_DISPATCHABLE": 3,
		"SERVING_STATE_DRAINING":     4,
	}
)

func (x ServingState) Enum() *ServingState {
	p := new(ServingState)
	*p = x
	return p
}

func (x ServingState) String() string {
	return protoimpl.X.EnumStringOf(x.Descriptor(), protoreflect.EnumNumber(x))
}

func (ServingState) Descriptor() protoreflect.EnumDescriptor {
	return file_cozy_worker_v1_worker_proto_enumTypes[4].Descriptor()
}

func (ServingState) Type() protoreflect.EnumType {
	return &file_cozy_worker_v1_worker_proto_enumTypes[4]
}

func (x ServingState) Number() protoreflect.EnumNumber {
	return protoreflect.EnumNumber(x)
}

// Deprecated: Use ServingState.Descriptor instead.
func (ServingState) EnumDescriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{4}
}

// The worker-level admission fence (§6). CLOSED is STRUCTURAL; OPEN with zero
// available_attempt_slots is TRANSIENT saturation (#486c). Both refuse under
// CAUSE_CODE_NO_CAPACITY, and a RecordOwner backs off differently for each.
type AdmissionState int32

const (
	AdmissionState_ADMISSION_STATE_UNSPECIFIED AdmissionState = 0
	AdmissionState_ADMISSION_STATE_OPEN        AdmissionState = 1
	AdmissionState_ADMISSION_STATE_CLOSED      AdmissionState = 2
)

// Enum value maps for AdmissionState.
var (
	AdmissionState_name = map[int32]string{
		0: "ADMISSION_STATE_UNSPECIFIED",
		1: "ADMISSION_STATE_OPEN",
		2: "ADMISSION_STATE_CLOSED",
	}
	AdmissionState_value = map[string]int32{
		"ADMISSION_STATE_UNSPECIFIED": 0,
		"ADMISSION_STATE_OPEN":        1,
		"ADMISSION_STATE_CLOSED":      2,
	}
)

func (x AdmissionState) Enum() *AdmissionState {
	p := new(AdmissionState)
	*p = x
	return p
}

func (x AdmissionState) String() string {
	return protoimpl.X.EnumStringOf(x.Descriptor(), protoreflect.EnumNumber(x))
}

func (AdmissionState) Descriptor() protoreflect.EnumDescriptor {
	return file_cozy_worker_v1_worker_proto_enumTypes[5].Descriptor()
}

func (AdmissionState) Type() protoreflect.EnumType {
	return &file_cozy_worker_v1_worker_proto_enumTypes[5]
}

func (x AdmissionState) Number() protoreflect.EnumNumber {
	return protoreflect.EnumNumber(x)
}

// Deprecated: Use AdmissionState.Descriptor instead.
func (AdmissionState) EnumDescriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{5}
}

type AttemptKind int32

const (
	AttemptKind_ATTEMPT_KIND_UNSPECIFIED AttemptKind = 0
	AttemptKind_ATTEMPT_KIND_SERVING     AttemptKind = 1
	AttemptKind_ATTEMPT_KIND_JOB         AttemptKind = 2
)

// Enum value maps for AttemptKind.
var (
	AttemptKind_name = map[int32]string{
		0: "ATTEMPT_KIND_UNSPECIFIED",
		1: "ATTEMPT_KIND_SERVING",
		2: "ATTEMPT_KIND_JOB",
	}
	AttemptKind_value = map[string]int32{
		"ATTEMPT_KIND_UNSPECIFIED": 0,
		"ATTEMPT_KIND_SERVING":     1,
		"ATTEMPT_KIND_JOB":         2,
	}
)

func (x AttemptKind) Enum() *AttemptKind {
	p := new(AttemptKind)
	*p = x
	return p
}

func (x AttemptKind) String() string {
	return protoimpl.X.EnumStringOf(x.Descriptor(), protoreflect.EnumNumber(x))
}

func (AttemptKind) Descriptor() protoreflect.EnumDescriptor {
	return file_cozy_worker_v1_worker_proto_enumTypes[6].Descriptor()
}

func (AttemptKind) Type() protoreflect.EnumType {
	return &file_cozy_worker_v1_worker_proto_enumTypes[6]
}

func (x AttemptKind) Number() protoreflect.EnumNumber {
	return protoreflect.EnumNumber(x)
}

// Deprecated: Use AttemptKind.Descriptor instead.
func (AttemptKind) EnumDescriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{6}
}

type AttemptState int32

const (
	AttemptState_ATTEMPT_STATE_UNSPECIFIED         AttemptState = 0
	AttemptState_ATTEMPT_STATE_RUNNING             AttemptState = 1
	AttemptState_ATTEMPT_STATE_OUTCOME_PENDING_ACK AttemptState = 2 // was TERMINAL_PENDING_ACK
	// #507d: the attempt is HELD but its journal record is not durable. The frozen wire could
	// only say this as RUNNING + a LOCAL_SAFETY fault compound — honest, but compound.
	AttemptState_ATTEMPT_STATE_HELD_UNDURABLE AttemptState = 3
)

// Enum value maps for AttemptState.
var (
	AttemptState_name = map[int32]string{
		0: "ATTEMPT_STATE_UNSPECIFIED",
		1: "ATTEMPT_STATE_RUNNING",
		2: "ATTEMPT_STATE_OUTCOME_PENDING_ACK",
		3: "ATTEMPT_STATE_HELD_UNDURABLE",
	}
	AttemptState_value = map[string]int32{
		"ATTEMPT_STATE_UNSPECIFIED":         0,
		"ATTEMPT_STATE_RUNNING":             1,
		"ATTEMPT_STATE_OUTCOME_PENDING_ACK": 2,
		"ATTEMPT_STATE_HELD_UNDURABLE":      3,
	}
)

func (x AttemptState) Enum() *AttemptState {
	p := new(AttemptState)
	*p = x
	return p
}

func (x AttemptState) String() string {
	return protoimpl.X.EnumStringOf(x.Descriptor(), protoreflect.EnumNumber(x))
}

func (AttemptState) Descriptor() protoreflect.EnumDescriptor {
	return file_cozy_worker_v1_worker_proto_enumTypes[7].Descriptor()
}

func (AttemptState) Type() protoreflect.EnumType {
	return &file_cozy_worker_v1_worker_proto_enumTypes[7]
}

func (x AttemptState) Number() protoreflect.EnumNumber {
	return protoreflect.EnumNumber(x)
}

// Deprecated: Use AttemptState.Descriptor instead.
func (AttemptState) EnumDescriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{7}
}

type OutcomeStatus int32

const (
	OutcomeStatus_OUTCOME_STATUS_UNSPECIFIED OutcomeStatus = 0
	OutcomeStatus_OUTCOME_STATUS_SUCCEEDED   OutcomeStatus = 1
	OutcomeStatus_OUTCOME_STATUS_REFUSED     OutcomeStatus = 2
	OutcomeStatus_OUTCOME_STATUS_FAILED      OutcomeStatus = 3
	OutcomeStatus_OUTCOME_STATUS_CANCELED    OutcomeStatus = 4
	OutcomeStatus_OUTCOME_STATUS_ABANDONED   OutcomeStatus = 5
)

// Enum value maps for OutcomeStatus.
var (
	OutcomeStatus_name = map[int32]string{
		0: "OUTCOME_STATUS_UNSPECIFIED",
		1: "OUTCOME_STATUS_SUCCEEDED",
		2: "OUTCOME_STATUS_REFUSED",
		3: "OUTCOME_STATUS_FAILED",
		4: "OUTCOME_STATUS_CANCELED",
		5: "OUTCOME_STATUS_ABANDONED",
	}
	OutcomeStatus_value = map[string]int32{
		"OUTCOME_STATUS_UNSPECIFIED": 0,
		"OUTCOME_STATUS_SUCCEEDED":   1,
		"OUTCOME_STATUS_REFUSED":     2,
		"OUTCOME_STATUS_FAILED":      3,
		"OUTCOME_STATUS_CANCELED":    4,
		"OUTCOME_STATUS_ABANDONED":   5,
	}
)

func (x OutcomeStatus) Enum() *OutcomeStatus {
	p := new(OutcomeStatus)
	*p = x
	return p
}

func (x OutcomeStatus) String() string {
	return protoimpl.X.EnumStringOf(x.Descriptor(), protoreflect.EnumNumber(x))
}

func (OutcomeStatus) Descriptor() protoreflect.EnumDescriptor {
	return file_cozy_worker_v1_worker_proto_enumTypes[8].Descriptor()
}

func (OutcomeStatus) Type() protoreflect.EnumType {
	return &file_cozy_worker_v1_worker_proto_enumTypes[8]
}

func (x OutcomeStatus) Number() protoreflect.EnumNumber {
	return protoreflect.EnumNumber(x)
}

// Deprecated: Use OutcomeStatus.Descriptor instead.
func (OutcomeStatus) EnumDescriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{8}
}

// Retryability is a RecordOwner PROJECTION over (status, cause, origin), never a wire fact. The
// REFUSED projection SPLITS by cause (#480b): author/runtime causes settle refused and are never
// re-dispatched; WORKER pre-execution causes consume the ordinal, leave the budget
// untouched, and are immediately re-dispatchable as attempt_ordinal + 1 elsewhere.
type CauseCode int32

const (
	CauseCode_CAUSE_CODE_UNSPECIFIED CauseCode = 0
	// REFUSED — a judgment about the WORK (origin author/runtime): settle refused
	CauseCode_CAUSE_CODE_INVALID_REQUEST       CauseCode = 1
	CauseCode_CAUSE_CODE_UNSUPPORTED_INPUT     CauseCode = 2
	CauseCode_CAUSE_CODE_LOCAL_SAFETY          CauseCode = 3
	CauseCode_CAUSE_CODE_PROTOCOL              CauseCode = 4
	CauseCode_CAUSE_CODE_CONSTRAINT_INFEASIBLE CauseCode = 5
	// FAILED
	CauseCode_CAUSE_CODE_AUTHOR_EXCEPTION       CauseCode = 6
	CauseCode_CAUSE_CODE_EXECUTOR_FAULT         CauseCode = 7
	CauseCode_CAUSE_CODE_GRANT_EXPIRED          CauseCode = 8
	CauseCode_CAUSE_CODE_ARTIFACT_UNFETCHABLE   CauseCode = 9
	CauseCode_CAUSE_CODE_CAPABILITY_UNAVAILABLE CauseCode = 10
	// CANCELED
	CauseCode_CAUSE_CODE_CLIENT_CANCEL     CauseCode = 11
	CauseCode_CAUSE_CODE_DEADLINE_EXPIRED  CauseCode = 12
	CauseCode_CAUSE_CODE_DRAIN_CANCEL      CauseCode = 13
	CauseCode_CAUSE_CODE_POLICY_CANCEL     CauseCode = 14
	CauseCode_CAUSE_CODE_SUPERSEDED_CANCEL CauseCode = 15
	// ABANDONED
	CauseCode_CAUSE_CODE_EXECUTOR_INVALIDATED CauseCode = 16
	// REFUSED — pre-execution, origin WORKER (rev-2 §6/§7). Zero billed execution budget;
	// execution_started is false; the ordinal is consumed and the next one may go elsewhere now.
	CauseCode_CAUSE_CODE_NO_CAPACITY                CauseCode = 17 // admission CLOSED, or OPEN with no free slot
	CauseCode_CAUSE_CODE_ADMISSION_GENERATION_STALE CauseCode = 18 // the echoed generation is not current
	CauseCode_CAUSE_CODE_UNKNOWN_PLACEMENT          CauseCode = 19
	CauseCode_CAUSE_CODE_PLACEMENT_NOT_DISPATCHABLE CauseCode = 20
)

// Enum value maps for CauseCode.
var (
	CauseCode_name = map[int32]string{
		0:  "CAUSE_CODE_UNSPECIFIED",
		1:  "CAUSE_CODE_INVALID_REQUEST",
		2:  "CAUSE_CODE_UNSUPPORTED_INPUT",
		3:  "CAUSE_CODE_LOCAL_SAFETY",
		4:  "CAUSE_CODE_PROTOCOL",
		5:  "CAUSE_CODE_CONSTRAINT_INFEASIBLE",
		6:  "CAUSE_CODE_AUTHOR_EXCEPTION",
		7:  "CAUSE_CODE_EXECUTOR_FAULT",
		8:  "CAUSE_CODE_GRANT_EXPIRED",
		9:  "CAUSE_CODE_ARTIFACT_UNFETCHABLE",
		10: "CAUSE_CODE_CAPABILITY_UNAVAILABLE",
		11: "CAUSE_CODE_CLIENT_CANCEL",
		12: "CAUSE_CODE_DEADLINE_EXPIRED",
		13: "CAUSE_CODE_DRAIN_CANCEL",
		14: "CAUSE_CODE_POLICY_CANCEL",
		15: "CAUSE_CODE_SUPERSEDED_CANCEL",
		16: "CAUSE_CODE_EXECUTOR_INVALIDATED",
		17: "CAUSE_CODE_NO_CAPACITY",
		18: "CAUSE_CODE_ADMISSION_GENERATION_STALE",
		19: "CAUSE_CODE_UNKNOWN_PLACEMENT",
		20: "CAUSE_CODE_PLACEMENT_NOT_DISPATCHABLE",
	}
	CauseCode_value = map[string]int32{
		"CAUSE_CODE_UNSPECIFIED":                0,
		"CAUSE_CODE_INVALID_REQUEST":            1,
		"CAUSE_CODE_UNSUPPORTED_INPUT":          2,
		"CAUSE_CODE_LOCAL_SAFETY":               3,
		"CAUSE_CODE_PROTOCOL":                   4,
		"CAUSE_CODE_CONSTRAINT_INFEASIBLE":      5,
		"CAUSE_CODE_AUTHOR_EXCEPTION":           6,
		"CAUSE_CODE_EXECUTOR_FAULT":             7,
		"CAUSE_CODE_GRANT_EXPIRED":              8,
		"CAUSE_CODE_ARTIFACT_UNFETCHABLE":       9,
		"CAUSE_CODE_CAPABILITY_UNAVAILABLE":     10,
		"CAUSE_CODE_CLIENT_CANCEL":              11,
		"CAUSE_CODE_DEADLINE_EXPIRED":           12,
		"CAUSE_CODE_DRAIN_CANCEL":               13,
		"CAUSE_CODE_POLICY_CANCEL":              14,
		"CAUSE_CODE_SUPERSEDED_CANCEL":          15,
		"CAUSE_CODE_EXECUTOR_INVALIDATED":       16,
		"CAUSE_CODE_NO_CAPACITY":                17,
		"CAUSE_CODE_ADMISSION_GENERATION_STALE": 18,
		"CAUSE_CODE_UNKNOWN_PLACEMENT":          19,
		"CAUSE_CODE_PLACEMENT_NOT_DISPATCHABLE": 20,
	}
)

func (x CauseCode) Enum() *CauseCode {
	p := new(CauseCode)
	*p = x
	return p
}

func (x CauseCode) String() string {
	return protoimpl.X.EnumStringOf(x.Descriptor(), protoreflect.EnumNumber(x))
}

func (CauseCode) Descriptor() protoreflect.EnumDescriptor {
	return file_cozy_worker_v1_worker_proto_enumTypes[9].Descriptor()
}

func (CauseCode) Type() protoreflect.EnumType {
	return &file_cozy_worker_v1_worker_proto_enumTypes[9]
}

func (x CauseCode) Number() protoreflect.EnumNumber {
	return protoreflect.EnumNumber(x)
}

// Deprecated: Use CauseCode.Descriptor instead.
func (CauseCode) EnumDescriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{9}
}

type CauseOrigin int32

const (
	CauseOrigin_CAUSE_ORIGIN_UNSPECIFIED  CauseOrigin = 0
	CauseOrigin_CAUSE_ORIGIN_AUTHOR       CauseOrigin = 1
	CauseOrigin_CAUSE_ORIGIN_RUNTIME      CauseOrigin = 2
	CauseOrigin_CAUSE_ORIGIN_EXECUTOR     CauseOrigin = 3
	CauseOrigin_CAUSE_ORIGIN_WORKER       CauseOrigin = 4
	CauseOrigin_CAUSE_ORIGIN_INFRA        CauseOrigin = 5
	CauseOrigin_CAUSE_ORIGIN_CLIENT       CauseOrigin = 6
	CauseOrigin_CAUSE_ORIGIN_RECORD_OWNER CauseOrigin = 7
)

// Enum value maps for CauseOrigin.
var (
	CauseOrigin_name = map[int32]string{
		0: "CAUSE_ORIGIN_UNSPECIFIED",
		1: "CAUSE_ORIGIN_AUTHOR",
		2: "CAUSE_ORIGIN_RUNTIME",
		3: "CAUSE_ORIGIN_EXECUTOR",
		4: "CAUSE_ORIGIN_WORKER",
		5: "CAUSE_ORIGIN_INFRA",
		6: "CAUSE_ORIGIN_CLIENT",
		7: "CAUSE_ORIGIN_RECORD_OWNER",
	}
	CauseOrigin_value = map[string]int32{
		"CAUSE_ORIGIN_UNSPECIFIED":  0,
		"CAUSE_ORIGIN_AUTHOR":       1,
		"CAUSE_ORIGIN_RUNTIME":      2,
		"CAUSE_ORIGIN_EXECUTOR":     3,
		"CAUSE_ORIGIN_WORKER":       4,
		"CAUSE_ORIGIN_INFRA":        5,
		"CAUSE_ORIGIN_CLIENT":       6,
		"CAUSE_ORIGIN_RECORD_OWNER": 7,
	}
)

func (x CauseOrigin) Enum() *CauseOrigin {
	p := new(CauseOrigin)
	*p = x
	return p
}

func (x CauseOrigin) String() string {
	return protoimpl.X.EnumStringOf(x.Descriptor(), protoreflect.EnumNumber(x))
}

func (CauseOrigin) Descriptor() protoreflect.EnumDescriptor {
	return file_cozy_worker_v1_worker_proto_enumTypes[10].Descriptor()
}

func (CauseOrigin) Type() protoreflect.EnumType {
	return &file_cozy_worker_v1_worker_proto_enumTypes[10]
}

func (x CauseOrigin) Number() protoreflect.EnumNumber {
	return protoreflect.EnumNumber(x)
}

// Deprecated: Use CauseOrigin.Descriptor instead.
func (CauseOrigin) EnumDescriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{10}
}

type CancelReason int32

const (
	CancelReason_CANCEL_REASON_UNSPECIFIED CancelReason = 0
	CancelReason_CANCEL_REASON_CLIENT      CancelReason = 1
	CancelReason_CANCEL_REASON_DRAIN       CancelReason = 2
	CancelReason_CANCEL_REASON_SUPERSEDED  CancelReason = 3
	CancelReason_CANCEL_REASON_POLICY      CancelReason = 4
	CancelReason_CANCEL_REASON_DEADLINE    CancelReason = 5
)

// Enum value maps for CancelReason.
var (
	CancelReason_name = map[int32]string{
		0: "CANCEL_REASON_UNSPECIFIED",
		1: "CANCEL_REASON_CLIENT",
		2: "CANCEL_REASON_DRAIN",
		3: "CANCEL_REASON_SUPERSEDED",
		4: "CANCEL_REASON_POLICY",
		5: "CANCEL_REASON_DEADLINE",
	}
	CancelReason_value = map[string]int32{
		"CANCEL_REASON_UNSPECIFIED": 0,
		"CANCEL_REASON_CLIENT":      1,
		"CANCEL_REASON_DRAIN":       2,
		"CANCEL_REASON_SUPERSEDED":  3,
		"CANCEL_REASON_POLICY":      4,
		"CANCEL_REASON_DEADLINE":    5,
	}
)

func (x CancelReason) Enum() *CancelReason {
	p := new(CancelReason)
	*p = x
	return p
}

func (x CancelReason) String() string {
	return protoimpl.X.EnumStringOf(x.Descriptor(), protoreflect.EnumNumber(x))
}

func (CancelReason) Descriptor() protoreflect.EnumDescriptor {
	return file_cozy_worker_v1_worker_proto_enumTypes[11].Descriptor()
}

func (CancelReason) Type() protoreflect.EnumType {
	return &file_cozy_worker_v1_worker_proto_enumTypes[11]
}

func (x CancelReason) Number() protoreflect.EnumNumber {
	return protoreflect.EnumNumber(x)
}

// Deprecated: Use CancelReason.Descriptor instead.
func (CancelReason) EnumDescriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{11}
}

type ClaimRejection int32

const (
	ClaimRejection_CLAIM_REJECTION_UNSPECIFIED              ClaimRejection = 0
	ClaimRejection_CLAIM_REJECTION_UNAUTHENTICATED          ClaimRejection = 1 // proof failure; checked before readiness/state
	ClaimRejection_CLAIM_REJECTION_STALE_RECORD_OWNER_EPOCH ClaimRejection = 2 // epoch older than the accepted claim
	ClaimRejection_CLAIM_REJECTION_EPOCH_HELD               ClaimRejection = 3 // equal epoch, different record_owner_id
	ClaimRejection_CLAIM_REJECTION_WORKER_ID_MISMATCH       ClaimRejection = 4
	ClaimRejection_CLAIM_REJECTION_RELEASE_ID_MISMATCH      ClaimRejection = 5
	// The worker cannot yet promise durable ownership: either its journal is unavailable or its
	// post-bind readiness barrier is still closed. In both cases no ownership state changes and an
	// authenticated caller may retry. BootFailure(DISK_SHAPE) remains the boot-fatal form.
	ClaimRejection_CLAIM_REJECTION_UNDURABLE ClaimRejection = 7
)

// Enum value maps for ClaimRejection.
var (
	ClaimRejection_name = map[int32]string{
		0: "CLAIM_REJECTION_UNSPECIFIED",
		1: "CLAIM_REJECTION_UNAUTHENTICATED",
		2: "CLAIM_REJECTION_STALE_RECORD_OWNER_EPOCH",
		3: "CLAIM_REJECTION_EPOCH_HELD",
		4: "CLAIM_REJECTION_WORKER_ID_MISMATCH",
		5: "CLAIM_REJECTION_RELEASE_ID_MISMATCH",
		7: "CLAIM_REJECTION_UNDURABLE",
	}
	ClaimRejection_value = map[string]int32{
		"CLAIM_REJECTION_UNSPECIFIED":              0,
		"CLAIM_REJECTION_UNAUTHENTICATED":          1,
		"CLAIM_REJECTION_STALE_RECORD_OWNER_EPOCH": 2,
		"CLAIM_REJECTION_EPOCH_HELD":               3,
		"CLAIM_REJECTION_WORKER_ID_MISMATCH":       4,
		"CLAIM_REJECTION_RELEASE_ID_MISMATCH":      5,
		"CLAIM_REJECTION_UNDURABLE":                7,
	}
)

func (x ClaimRejection) Enum() *ClaimRejection {
	p := new(ClaimRejection)
	*p = x
	return p
}

func (x ClaimRejection) String() string {
	return protoimpl.X.EnumStringOf(x.Descriptor(), protoreflect.EnumNumber(x))
}

func (ClaimRejection) Descriptor() protoreflect.EnumDescriptor {
	return file_cozy_worker_v1_worker_proto_enumTypes[12].Descriptor()
}

func (ClaimRejection) Type() protoreflect.EnumType {
	return &file_cozy_worker_v1_worker_proto_enumTypes[12]
}

func (x ClaimRejection) Number() protoreflect.EnumNumber {
	return protoreflect.EnumNumber(x)
}

// Deprecated: Use ClaimRejection.Descriptor instead.
func (ClaimRejection) EnumDescriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{12}
}

type FaultKind int32

const (
	FaultKind_FAULT_KIND_UNSPECIFIED         FaultKind = 0
	FaultKind_FAULT_KIND_BINDING_UNAVAILABLE FaultKind = 1
	FaultKind_FAULT_KIND_BINDING_DEGRADED    FaultKind = 2
	FaultKind_FAULT_KIND_HARDWARE_UNSUITABLE FaultKind = 3 // this worker's actual platform cannot install or
	// run the selected package; the previous spec
	// keeps serving
	FaultKind_FAULT_KIND_ARTIFACT_FETCH_FAILED     FaultKind = 4
	FaultKind_FAULT_KIND_CREDENTIAL_UNAPPLIED      FaultKind = 6
	FaultKind_FAULT_KIND_LOCAL_SAFETY_REFUSAL      FaultKind = 7
	FaultKind_FAULT_KIND_EXECUTOR_POISONED         FaultKind = 8
	FaultKind_FAULT_KIND_CONFIG_REFUSED            FaultKind = 9
	FaultKind_FAULT_KIND_UNKNOWN_PLACEMENT         FaultKind = 10
	FaultKind_FAULT_KIND_PLACEMENT_SET_UNSUPPORTED FaultKind = 11 // launch len<=1 breach or an unsupportable set;
	// the desired state is UNAPPLIED
	FaultKind_FAULT_KIND_PLACEMENT_SET_DIGEST_MISMATCH FaultKind = 12 // recompute over
	// placement_set_canonical_bytes disagreed with
	// placement_set_digest; UNAPPLIED, before any
	// field was parsed (§4)
	FaultKind_FAULT_KIND_ARTIFACT_DIGEST_MISMATCH FaultKind = 14 // fetched bytes whose digest is not the named
	// subject's; discarded, never installed
	FaultKind_FAULT_KIND_FALLBACK_PIN_MISSING FaultKind = 15 // #485c: the rollback pin is gone (GC'd or never
)

// Enum value maps for FaultKind.
var (
	FaultKind_name = map[int32]string{
		0:  "FAULT_KIND_UNSPECIFIED",
		1:  "FAULT_KIND_BINDING_UNAVAILABLE",
		2:  "FAULT_KIND_BINDING_DEGRADED",
		3:  "FAULT_KIND_HARDWARE_UNSUITABLE",
		4:  "FAULT_KIND_ARTIFACT_FETCH_FAILED",
		6:  "FAULT_KIND_CREDENTIAL_UNAPPLIED",
		7:  "FAULT_KIND_LOCAL_SAFETY_REFUSAL",
		8:  "FAULT_KIND_EXECUTOR_POISONED",
		9:  "FAULT_KIND_CONFIG_REFUSED",
		10: "FAULT_KIND_UNKNOWN_PLACEMENT",
		11: "FAULT_KIND_PLACEMENT_SET_UNSUPPORTED",
		12: "FAULT_KIND_PLACEMENT_SET_DIGEST_MISMATCH",
		14: "FAULT_KIND_ARTIFACT_DIGEST_MISMATCH",
		15: "FAULT_KIND_FALLBACK_PIN_MISSING",
	}
	FaultKind_value = map[string]int32{
		"FAULT_KIND_UNSPECIFIED":                   0,
		"FAULT_KIND_BINDING_UNAVAILABLE":           1,
		"FAULT_KIND_BINDING_DEGRADED":              2,
		"FAULT_KIND_HARDWARE_UNSUITABLE":           3,
		"FAULT_KIND_ARTIFACT_FETCH_FAILED":         4,
		"FAULT_KIND_CREDENTIAL_UNAPPLIED":          6,
		"FAULT_KIND_LOCAL_SAFETY_REFUSAL":          7,
		"FAULT_KIND_EXECUTOR_POISONED":             8,
		"FAULT_KIND_CONFIG_REFUSED":                9,
		"FAULT_KIND_UNKNOWN_PLACEMENT":             10,
		"FAULT_KIND_PLACEMENT_SET_UNSUPPORTED":     11,
		"FAULT_KIND_PLACEMENT_SET_DIGEST_MISMATCH": 12,
		"FAULT_KIND_ARTIFACT_DIGEST_MISMATCH":      14,
		"FAULT_KIND_FALLBACK_PIN_MISSING":          15,
	}
)

func (x FaultKind) Enum() *FaultKind {
	p := new(FaultKind)
	*p = x
	return p
}

func (x FaultKind) String() string {
	return protoimpl.X.EnumStringOf(x.Descriptor(), protoreflect.EnumNumber(x))
}

func (FaultKind) Descriptor() protoreflect.EnumDescriptor {
	return file_cozy_worker_v1_worker_proto_enumTypes[13].Descriptor()
}

func (FaultKind) Type() protoreflect.EnumType {
	return &file_cozy_worker_v1_worker_proto_enumTypes[13]
}

func (x FaultKind) Number() protoreflect.EnumNumber {
	return protoreflect.EnumNumber(x)
}

// Deprecated: Use FaultKind.Descriptor instead.
func (FaultKind) EnumDescriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{13}
}

type BootFailureReason int32

const (
	BootFailureReason_BOOT_FAILURE_REASON_UNSPECIFIED      BootFailureReason = 0
	BootFailureReason_BOOT_FAILURE_REASON_HARDWARE_VERDICT BootFailureReason = 1
	BootFailureReason_BOOT_FAILURE_REASON_ARTIFACT_FAULT   BootFailureReason = 2 // was IMAGE_FAULT; a native worker has no image
	BootFailureReason_BOOT_FAILURE_REASON_DRIVER_FAULT     BootFailureReason = 3
	BootFailureReason_BOOT_FAILURE_REASON_DISK_SHAPE       BootFailureReason = 4
	BootFailureReason_BOOT_FAILURE_REASON_CONFIG_INVALID   BootFailureReason = 5
	BootFailureReason_BOOT_FAILURE_REASON_OTHER_FATAL      BootFailureReason = 6
)

// Enum value maps for BootFailureReason.
var (
	BootFailureReason_name = map[int32]string{
		0: "BOOT_FAILURE_REASON_UNSPECIFIED",
		1: "BOOT_FAILURE_REASON_HARDWARE_VERDICT",
		2: "BOOT_FAILURE_REASON_ARTIFACT_FAULT",
		3: "BOOT_FAILURE_REASON_DRIVER_FAULT",
		4: "BOOT_FAILURE_REASON_DISK_SHAPE",
		5: "BOOT_FAILURE_REASON_CONFIG_INVALID",
		6: "BOOT_FAILURE_REASON_OTHER_FATAL",
	}
	BootFailureReason_value = map[string]int32{
		"BOOT_FAILURE_REASON_UNSPECIFIED":      0,
		"BOOT_FAILURE_REASON_HARDWARE_VERDICT": 1,
		"BOOT_FAILURE_REASON_ARTIFACT_FAULT":   2,
		"BOOT_FAILURE_REASON_DRIVER_FAULT":     3,
		"BOOT_FAILURE_REASON_DISK_SHAPE":       4,
		"BOOT_FAILURE_REASON_CONFIG_INVALID":   5,
		"BOOT_FAILURE_REASON_OTHER_FATAL":      6,
	}
)

func (x BootFailureReason) Enum() *BootFailureReason {
	p := new(BootFailureReason)
	*p = x
	return p
}

func (x BootFailureReason) String() string {
	return protoimpl.X.EnumStringOf(x.Descriptor(), protoreflect.EnumNumber(x))
}

func (BootFailureReason) Descriptor() protoreflect.EnumDescriptor {
	return file_cozy_worker_v1_worker_proto_enumTypes[14].Descriptor()
}

func (BootFailureReason) Type() protoreflect.EnumType {
	return &file_cozy_worker_v1_worker_proto_enumTypes[14]
}

func (x BootFailureReason) Number() protoreflect.EnumNumber {
	return protoreflect.EnumNumber(x)
}

// Deprecated: Use BootFailureReason.Descriptor instead.
func (BootFailureReason) EnumDescriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{14}
}

type CheckpointOutcome int32

const (
	CheckpointOutcome_CHECKPOINT_OUTCOME_UNSPECIFIED CheckpointOutcome = 0
	CheckpointOutcome_CHECKPOINT_OUTCOME_RECORDED    CheckpointOutcome = 1
	CheckpointOutcome_CHECKPOINT_OUTCOME_REPLAYED    CheckpointOutcome = 2
	CheckpointOutcome_CHECKPOINT_OUTCOME_CONFLICT    CheckpointOutcome = 3
	CheckpointOutcome_CHECKPOINT_OUTCOME_REFUSED     CheckpointOutcome = 4
)

// Enum value maps for CheckpointOutcome.
var (
	CheckpointOutcome_name = map[int32]string{
		0: "CHECKPOINT_OUTCOME_UNSPECIFIED",
		1: "CHECKPOINT_OUTCOME_RECORDED",
		2: "CHECKPOINT_OUTCOME_REPLAYED",
		3: "CHECKPOINT_OUTCOME_CONFLICT",
		4: "CHECKPOINT_OUTCOME_REFUSED",
	}
	CheckpointOutcome_value = map[string]int32{
		"CHECKPOINT_OUTCOME_UNSPECIFIED": 0,
		"CHECKPOINT_OUTCOME_RECORDED":    1,
		"CHECKPOINT_OUTCOME_REPLAYED":    2,
		"CHECKPOINT_OUTCOME_CONFLICT":    3,
		"CHECKPOINT_OUTCOME_REFUSED":     4,
	}
)

func (x CheckpointOutcome) Enum() *CheckpointOutcome {
	p := new(CheckpointOutcome)
	*p = x
	return p
}

func (x CheckpointOutcome) String() string {
	return protoimpl.X.EnumStringOf(x.Descriptor(), protoreflect.EnumNumber(x))
}

func (CheckpointOutcome) Descriptor() protoreflect.EnumDescriptor {
	return file_cozy_worker_v1_worker_proto_enumTypes[15].Descriptor()
}

func (CheckpointOutcome) Type() protoreflect.EnumType {
	return &file_cozy_worker_v1_worker_proto_enumTypes[15]
}

func (x CheckpointOutcome) Number() protoreflect.EnumNumber {
	return protoreflect.EnumNumber(x)
}

// Deprecated: Use CheckpointOutcome.Descriptor instead.
func (CheckpointOutcome) EnumDescriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{15}
}

type CheckpointFaultCode int32

const (
	CheckpointFaultCode_CHECKPOINT_FAULT_CODE_UNSPECIFIED       CheckpointFaultCode = 0
	CheckpointFaultCode_CHECKPOINT_FAULT_CODE_IDENTITY_CONFLICT CheckpointFaultCode = 1
	CheckpointFaultCode_CHECKPOINT_FAULT_CODE_UNKNOWN_ATTEMPT   CheckpointFaultCode = 2
	CheckpointFaultCode_CHECKPOINT_FAULT_CODE_NOT_JOB_MODE      CheckpointFaultCode = 3
	CheckpointFaultCode_CHECKPOINT_FAULT_CODE_STALE_SEQUENCE    CheckpointFaultCode = 4
	CheckpointFaultCode_CHECKPOINT_FAULT_CODE_QUOTA_EXCEEDED    CheckpointFaultCode = 5
)

// Enum value maps for CheckpointFaultCode.
var (
	CheckpointFaultCode_name = map[int32]string{
		0: "CHECKPOINT_FAULT_CODE_UNSPECIFIED",
		1: "CHECKPOINT_FAULT_CODE_IDENTITY_CONFLICT",
		2: "CHECKPOINT_FAULT_CODE_UNKNOWN_ATTEMPT",
		3: "CHECKPOINT_FAULT_CODE_NOT_JOB_MODE",
		4: "CHECKPOINT_FAULT_CODE_STALE_SEQUENCE",
		5: "CHECKPOINT_FAULT_CODE_QUOTA_EXCEEDED",
	}
	CheckpointFaultCode_value = map[string]int32{
		"CHECKPOINT_FAULT_CODE_UNSPECIFIED":       0,
		"CHECKPOINT_FAULT_CODE_IDENTITY_CONFLICT": 1,
		"CHECKPOINT_FAULT_CODE_UNKNOWN_ATTEMPT":   2,
		"CHECKPOINT_FAULT_CODE_NOT_JOB_MODE":      3,
		"CHECKPOINT_FAULT_CODE_STALE_SEQUENCE":    4,
		"CHECKPOINT_FAULT_CODE_QUOTA_EXCEEDED":    5,
	}
)

func (x CheckpointFaultCode) Enum() *CheckpointFaultCode {
	p := new(CheckpointFaultCode)
	*p = x
	return p
}

func (x CheckpointFaultCode) String() string {
	return protoimpl.X.EnumStringOf(x.Descriptor(), protoreflect.EnumNumber(x))
}

func (CheckpointFaultCode) Descriptor() protoreflect.EnumDescriptor {
	return file_cozy_worker_v1_worker_proto_enumTypes[16].Descriptor()
}

func (CheckpointFaultCode) Type() protoreflect.EnumType {
	return &file_cozy_worker_v1_worker_proto_enumTypes[16]
}

func (x CheckpointFaultCode) Number() protoreflect.EnumNumber {
	return protoreflect.EnumNumber(x)
}

// Deprecated: Use CheckpointFaultCode.Descriptor instead.
func (CheckpointFaultCode) EnumDescriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{16}
}

type ArtifactHostStage int32

const (
	ArtifactHostStage_ARTIFACT_HOST_STAGE_UNSPECIFIED ArtifactHostStage = 0
	ArtifactHostStage_ARTIFACT_HOST_STAGE_INTENT      ArtifactHostStage = 1
	ArtifactHostStage_ARTIFACT_HOST_STAGE_RECEIPT     ArtifactHostStage = 2
)

// Enum value maps for ArtifactHostStage.
var (
	ArtifactHostStage_name = map[int32]string{
		0: "ARTIFACT_HOST_STAGE_UNSPECIFIED",
		1: "ARTIFACT_HOST_STAGE_INTENT",
		2: "ARTIFACT_HOST_STAGE_RECEIPT",
	}
	ArtifactHostStage_value = map[string]int32{
		"ARTIFACT_HOST_STAGE_UNSPECIFIED": 0,
		"ARTIFACT_HOST_STAGE_INTENT":      1,
		"ARTIFACT_HOST_STAGE_RECEIPT":     2,
	}
)

func (x ArtifactHostStage) Enum() *ArtifactHostStage {
	p := new(ArtifactHostStage)
	*p = x
	return p
}

func (x ArtifactHostStage) String() string {
	return protoimpl.X.EnumStringOf(x.Descriptor(), protoreflect.EnumNumber(x))
}

func (ArtifactHostStage) Descriptor() protoreflect.EnumDescriptor {
	return file_cozy_worker_v1_worker_proto_enumTypes[17].Descriptor()
}

func (ArtifactHostStage) Type() protoreflect.EnumType {
	return &file_cozy_worker_v1_worker_proto_enumTypes[17]
}

func (x ArtifactHostStage) Number() protoreflect.EnumNumber {
	return protoreflect.EnumNumber(x)
}

// Deprecated: Use ArtifactHostStage.Descriptor instead.
func (ArtifactHostStage) EnumDescriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{17}
}

type ArtifactHostOutcome int32

const (
	ArtifactHostOutcome_ARTIFACT_HOST_OUTCOME_UNSPECIFIED ArtifactHostOutcome = 0
	ArtifactHostOutcome_ARTIFACT_HOST_OUTCOME_RECORDED    ArtifactHostOutcome = 1
	ArtifactHostOutcome_ARTIFACT_HOST_OUTCOME_REPLAYED    ArtifactHostOutcome = 2
	ArtifactHostOutcome_ARTIFACT_HOST_OUTCOME_REFUSED     ArtifactHostOutcome = 3
)

// Enum value maps for ArtifactHostOutcome.
var (
	ArtifactHostOutcome_name = map[int32]string{
		0: "ARTIFACT_HOST_OUTCOME_UNSPECIFIED",
		1: "ARTIFACT_HOST_OUTCOME_RECORDED",
		2: "ARTIFACT_HOST_OUTCOME_REPLAYED",
		3: "ARTIFACT_HOST_OUTCOME_REFUSED",
	}
	ArtifactHostOutcome_value = map[string]int32{
		"ARTIFACT_HOST_OUTCOME_UNSPECIFIED": 0,
		"ARTIFACT_HOST_OUTCOME_RECORDED":    1,
		"ARTIFACT_HOST_OUTCOME_REPLAYED":    2,
		"ARTIFACT_HOST_OUTCOME_REFUSED":     3,
	}
)

func (x ArtifactHostOutcome) Enum() *ArtifactHostOutcome {
	p := new(ArtifactHostOutcome)
	*p = x
	return p
}

func (x ArtifactHostOutcome) String() string {
	return protoimpl.X.EnumStringOf(x.Descriptor(), protoreflect.EnumNumber(x))
}

func (ArtifactHostOutcome) Descriptor() protoreflect.EnumDescriptor {
	return file_cozy_worker_v1_worker_proto_enumTypes[18].Descriptor()
}

func (ArtifactHostOutcome) Type() protoreflect.EnumType {
	return &file_cozy_worker_v1_worker_proto_enumTypes[18]
}

func (x ArtifactHostOutcome) Number() protoreflect.EnumNumber {
	return protoreflect.EnumNumber(x)
}

// Deprecated: Use ArtifactHostOutcome.Descriptor instead.
func (ArtifactHostOutcome) EnumDescriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{18}
}

type ArtifactHostRefusal int32

const (
	ArtifactHostRefusal_ARTIFACT_HOST_REFUSAL_UNSPECIFIED       ArtifactHostRefusal = 0
	ArtifactHostRefusal_ARTIFACT_HOST_REFUSAL_UNKNOWN_ATTEMPT   ArtifactHostRefusal = 1
	ArtifactHostRefusal_ARTIFACT_HOST_REFUSAL_INTENT_CONFLICT   ArtifactHostRefusal = 2
	ArtifactHostRefusal_ARTIFACT_HOST_REFUSAL_STALE_WRITER      ArtifactHostRefusal = 3
	ArtifactHostRefusal_ARTIFACT_HOST_REFUSAL_RECEIPT_CONFLICT  ArtifactHostRefusal = 4
	ArtifactHostRefusal_ARTIFACT_HOST_REFUSAL_INVENTORY_INVALID ArtifactHostRefusal = 5
)

// Enum value maps for ArtifactHostRefusal.
var (
	ArtifactHostRefusal_name = map[int32]string{
		0: "ARTIFACT_HOST_REFUSAL_UNSPECIFIED",
		1: "ARTIFACT_HOST_REFUSAL_UNKNOWN_ATTEMPT",
		2: "ARTIFACT_HOST_REFUSAL_INTENT_CONFLICT",
		3: "ARTIFACT_HOST_REFUSAL_STALE_WRITER",
		4: "ARTIFACT_HOST_REFUSAL_RECEIPT_CONFLICT",
		5: "ARTIFACT_HOST_REFUSAL_INVENTORY_INVALID",
	}
	ArtifactHostRefusal_value = map[string]int32{
		"ARTIFACT_HOST_REFUSAL_UNSPECIFIED":       0,
		"ARTIFACT_HOST_REFUSAL_UNKNOWN_ATTEMPT":   1,
		"ARTIFACT_HOST_REFUSAL_INTENT_CONFLICT":   2,
		"ARTIFACT_HOST_REFUSAL_STALE_WRITER":      3,
		"ARTIFACT_HOST_REFUSAL_RECEIPT_CONFLICT":  4,
		"ARTIFACT_HOST_REFUSAL_INVENTORY_INVALID": 5,
	}
)

func (x ArtifactHostRefusal) Enum() *ArtifactHostRefusal {
	p := new(ArtifactHostRefusal)
	*p = x
	return p
}

func (x ArtifactHostRefusal) String() string {
	return protoimpl.X.EnumStringOf(x.Descriptor(), protoreflect.EnumNumber(x))
}

func (ArtifactHostRefusal) Descriptor() protoreflect.EnumDescriptor {
	return file_cozy_worker_v1_worker_proto_enumTypes[19].Descriptor()
}

func (ArtifactHostRefusal) Type() protoreflect.EnumType {
	return &file_cozy_worker_v1_worker_proto_enumTypes[19]
}

func (x ArtifactHostRefusal) Number() protoreflect.EnumNumber {
	return protoreflect.EnumNumber(x)
}

// Deprecated: Use ArtifactHostRefusal.Descriptor instead.
func (ArtifactHostRefusal) EnumDescriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{19}
}

type ArtifactTransactionState int32

const (
	ArtifactTransactionState_ARTIFACT_TRANSACTION_STATE_UNSPECIFIED ArtifactTransactionState = 0
	ArtifactTransactionState_ARTIFACT_TRANSACTION_STATE_INTENT      ArtifactTransactionState = 1
	ArtifactTransactionState_ARTIFACT_TRANSACTION_STATE_RECEIPT     ArtifactTransactionState = 2
)

// Enum value maps for ArtifactTransactionState.
var (
	ArtifactTransactionState_name = map[int32]string{
		0: "ARTIFACT_TRANSACTION_STATE_UNSPECIFIED",
		1: "ARTIFACT_TRANSACTION_STATE_INTENT",
		2: "ARTIFACT_TRANSACTION_STATE_RECEIPT",
	}
	ArtifactTransactionState_value = map[string]int32{
		"ARTIFACT_TRANSACTION_STATE_UNSPECIFIED": 0,
		"ARTIFACT_TRANSACTION_STATE_INTENT":      1,
		"ARTIFACT_TRANSACTION_STATE_RECEIPT":     2,
	}
)

func (x ArtifactTransactionState) Enum() *ArtifactTransactionState {
	p := new(ArtifactTransactionState)
	*p = x
	return p
}

func (x ArtifactTransactionState) String() string {
	return protoimpl.X.EnumStringOf(x.Descriptor(), protoreflect.EnumNumber(x))
}

func (ArtifactTransactionState) Descriptor() protoreflect.EnumDescriptor {
	return file_cozy_worker_v1_worker_proto_enumTypes[20].Descriptor()
}

func (ArtifactTransactionState) Type() protoreflect.EnumType {
	return &file_cozy_worker_v1_worker_proto_enumTypes[20]
}

func (x ArtifactTransactionState) Number() protoreflect.EnumNumber {
	return protoreflect.EnumNumber(x)
}

// Deprecated: Use ArtifactTransactionState.Descriptor instead.
func (ArtifactTransactionState) EnumDescriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{20}
}

type ArtifactReadOutcome int32

const (
	ArtifactReadOutcome_ARTIFACT_READ_OUTCOME_UNSPECIFIED ArtifactReadOutcome = 0
	ArtifactReadOutcome_ARTIFACT_READ_OUTCOME_DATA        ArtifactReadOutcome = 1
	ArtifactReadOutcome_ARTIFACT_READ_OUTCOME_REFUSED     ArtifactReadOutcome = 2
)

// Enum value maps for ArtifactReadOutcome.
var (
	ArtifactReadOutcome_name = map[int32]string{
		0: "ARTIFACT_READ_OUTCOME_UNSPECIFIED",
		1: "ARTIFACT_READ_OUTCOME_DATA",
		2: "ARTIFACT_READ_OUTCOME_REFUSED",
	}
	ArtifactReadOutcome_value = map[string]int32{
		"ARTIFACT_READ_OUTCOME_UNSPECIFIED": 0,
		"ARTIFACT_READ_OUTCOME_DATA":        1,
		"ARTIFACT_READ_OUTCOME_REFUSED":     2,
	}
)

func (x ArtifactReadOutcome) Enum() *ArtifactReadOutcome {
	p := new(ArtifactReadOutcome)
	*p = x
	return p
}

func (x ArtifactReadOutcome) String() string {
	return protoimpl.X.EnumStringOf(x.Descriptor(), protoreflect.EnumNumber(x))
}

func (ArtifactReadOutcome) Descriptor() protoreflect.EnumDescriptor {
	return file_cozy_worker_v1_worker_proto_enumTypes[21].Descriptor()
}

func (ArtifactReadOutcome) Type() protoreflect.EnumType {
	return &file_cozy_worker_v1_worker_proto_enumTypes[21]
}

func (x ArtifactReadOutcome) Number() protoreflect.EnumNumber {
	return protoreflect.EnumNumber(x)
}

// Deprecated: Use ArtifactReadOutcome.Descriptor instead.
func (ArtifactReadOutcome) EnumDescriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{21}
}

type ArtifactReadRefusal int32

const (
	ArtifactReadRefusal_ARTIFACT_READ_REFUSAL_UNSPECIFIED         ArtifactReadRefusal = 0
	ArtifactReadRefusal_ARTIFACT_READ_REFUSAL_UNKNOWN_TRANSACTION ArtifactReadRefusal = 1
	ArtifactReadRefusal_ARTIFACT_READ_REFUSAL_STALE_WRITER        ArtifactReadRefusal = 2
	ArtifactReadRefusal_ARTIFACT_READ_REFUSAL_UNKNOWN_OBJECT      ArtifactReadRefusal = 3
	ArtifactReadRefusal_ARTIFACT_READ_REFUSAL_SOURCE_REF_MISMATCH ArtifactReadRefusal = 4
	ArtifactReadRefusal_ARTIFACT_READ_REFUSAL_RANGE_INVALID       ArtifactReadRefusal = 5
	ArtifactReadRefusal_ARTIFACT_READ_REFUSAL_SOURCE_UNAVAILABLE  ArtifactReadRefusal = 6
)

// Enum value maps for ArtifactReadRefusal.
var (
	ArtifactReadRefusal_name = map[int32]string{
		0: "ARTIFACT_READ_REFUSAL_UNSPECIFIED",
		1: "ARTIFACT_READ_REFUSAL_UNKNOWN_TRANSACTION",
		2: "ARTIFACT_READ_REFUSAL_STALE_WRITER",
		3: "ARTIFACT_READ_REFUSAL_UNKNOWN_OBJECT",
		4: "ARTIFACT_READ_REFUSAL_SOURCE_REF_MISMATCH",
		5: "ARTIFACT_READ_REFUSAL_RANGE_INVALID",
		6: "ARTIFACT_READ_REFUSAL_SOURCE_UNAVAILABLE",
	}
	ArtifactReadRefusal_value = map[string]int32{
		"ARTIFACT_READ_REFUSAL_UNSPECIFIED":         0,
		"ARTIFACT_READ_REFUSAL_UNKNOWN_TRANSACTION": 1,
		"ARTIFACT_READ_REFUSAL_STALE_WRITER":        2,
		"ARTIFACT_READ_REFUSAL_UNKNOWN_OBJECT":      3,
		"ARTIFACT_READ_REFUSAL_SOURCE_REF_MISMATCH": 4,
		"ARTIFACT_READ_REFUSAL_RANGE_INVALID":       5,
		"ARTIFACT_READ_REFUSAL_SOURCE_UNAVAILABLE":  6,
	}
)

func (x ArtifactReadRefusal) Enum() *ArtifactReadRefusal {
	p := new(ArtifactReadRefusal)
	*p = x
	return p
}

func (x ArtifactReadRefusal) String() string {
	return protoimpl.X.EnumStringOf(x.Descriptor(), protoreflect.EnumNumber(x))
}

func (ArtifactReadRefusal) Descriptor() protoreflect.EnumDescriptor {
	return file_cozy_worker_v1_worker_proto_enumTypes[22].Descriptor()
}

func (ArtifactReadRefusal) Type() protoreflect.EnumType {
	return &file_cozy_worker_v1_worker_proto_enumTypes[22]
}

func (x ArtifactReadRefusal) Number() protoreflect.EnumNumber {
	return protoreflect.EnumNumber(x)
}

// Deprecated: Use ArtifactReadRefusal.Descriptor instead.
func (ArtifactReadRefusal) EnumDescriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{22}
}

type ArtifactTransferState int32

const (
	ArtifactTransferState_ARTIFACT_TRANSFER_STATE_UNSPECIFIED     ArtifactTransferState = 0
	ArtifactTransferState_ARTIFACT_TRANSFER_STATE_ACCEPTED        ArtifactTransferState = 1
	ArtifactTransferState_ARTIFACT_TRANSFER_STATE_UPLOADING       ArtifactTransferState = 2
	ArtifactTransferState_ARTIFACT_TRANSFER_STATE_UPLOADED        ArtifactTransferState = 3
	ArtifactTransferState_ARTIFACT_TRANSFER_STATE_ALREADY_PRESENT ArtifactTransferState = 4
	ArtifactTransferState_ARTIFACT_TRANSFER_STATE_HELD            ArtifactTransferState = 5
	ArtifactTransferState_ARTIFACT_TRANSFER_STATE_FAILED          ArtifactTransferState = 6
)

// Enum value maps for ArtifactTransferState.
var (
	ArtifactTransferState_name = map[int32]string{
		0: "ARTIFACT_TRANSFER_STATE_UNSPECIFIED",
		1: "ARTIFACT_TRANSFER_STATE_ACCEPTED",
		2: "ARTIFACT_TRANSFER_STATE_UPLOADING",
		3: "ARTIFACT_TRANSFER_STATE_UPLOADED",
		4: "ARTIFACT_TRANSFER_STATE_ALREADY_PRESENT",
		5: "ARTIFACT_TRANSFER_STATE_HELD",
		6: "ARTIFACT_TRANSFER_STATE_FAILED",
	}
	ArtifactTransferState_value = map[string]int32{
		"ARTIFACT_TRANSFER_STATE_UNSPECIFIED":     0,
		"ARTIFACT_TRANSFER_STATE_ACCEPTED":        1,
		"ARTIFACT_TRANSFER_STATE_UPLOADING":       2,
		"ARTIFACT_TRANSFER_STATE_UPLOADED":        3,
		"ARTIFACT_TRANSFER_STATE_ALREADY_PRESENT": 4,
		"ARTIFACT_TRANSFER_STATE_HELD":            5,
		"ARTIFACT_TRANSFER_STATE_FAILED":          6,
	}
)

func (x ArtifactTransferState) Enum() *ArtifactTransferState {
	p := new(ArtifactTransferState)
	*p = x
	return p
}

func (x ArtifactTransferState) String() string {
	return protoimpl.X.EnumStringOf(x.Descriptor(), protoreflect.EnumNumber(x))
}

func (ArtifactTransferState) Descriptor() protoreflect.EnumDescriptor {
	return file_cozy_worker_v1_worker_proto_enumTypes[23].Descriptor()
}

func (ArtifactTransferState) Type() protoreflect.EnumType {
	return &file_cozy_worker_v1_worker_proto_enumTypes[23]
}

func (x ArtifactTransferState) Number() protoreflect.EnumNumber {
	return protoreflect.EnumNumber(x)
}

// Deprecated: Use ArtifactTransferState.Descriptor instead.
func (ArtifactTransferState) EnumDescriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{23}
}

type ModelSourceProvider int32

const (
	ModelSourceProvider_MODEL_SOURCE_PROVIDER_UNSPECIFIED  ModelSourceProvider = 0
	ModelSourceProvider_MODEL_SOURCE_PROVIDER_HUGGING_FACE ModelSourceProvider = 1
	ModelSourceProvider_MODEL_SOURCE_PROVIDER_CIVITAI      ModelSourceProvider = 2
)

// Enum value maps for ModelSourceProvider.
var (
	ModelSourceProvider_name = map[int32]string{
		0: "MODEL_SOURCE_PROVIDER_UNSPECIFIED",
		1: "MODEL_SOURCE_PROVIDER_HUGGING_FACE",
		2: "MODEL_SOURCE_PROVIDER_CIVITAI",
	}
	ModelSourceProvider_value = map[string]int32{
		"MODEL_SOURCE_PROVIDER_UNSPECIFIED":  0,
		"MODEL_SOURCE_PROVIDER_HUGGING_FACE": 1,
		"MODEL_SOURCE_PROVIDER_CIVITAI":      2,
	}
)

func (x ModelSourceProvider) Enum() *ModelSourceProvider {
	p := new(ModelSourceProvider)
	*p = x
	return p
}

func (x ModelSourceProvider) String() string {
	return protoimpl.X.EnumStringOf(x.Descriptor(), protoreflect.EnumNumber(x))
}

func (ModelSourceProvider) Descriptor() protoreflect.EnumDescriptor {
	return file_cozy_worker_v1_worker_proto_enumTypes[24].Descriptor()
}

func (ModelSourceProvider) Type() protoreflect.EnumType {
	return &file_cozy_worker_v1_worker_proto_enumTypes[24]
}

func (x ModelSourceProvider) Number() protoreflect.EnumNumber {
	return protoreflect.EnumNumber(x)
}

// Deprecated: Use ModelSourceProvider.Descriptor instead.
func (ModelSourceProvider) EnumDescriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{24}
}

type ModelSourceFileState int32

const (
	ModelSourceFileState_MODEL_SOURCE_FILE_STATE_UNSPECIFIED ModelSourceFileState = 0
	ModelSourceFileState_MODEL_SOURCE_FILE_STATE_ACCEPTED    ModelSourceFileState = 1
	ModelSourceFileState_MODEL_SOURCE_FILE_STATE_DOWNLOADING ModelSourceFileState = 2
	ModelSourceFileState_MODEL_SOURCE_FILE_STATE_VERIFIED    ModelSourceFileState = 3
	ModelSourceFileState_MODEL_SOURCE_FILE_STATE_FAILED      ModelSourceFileState = 4
)

// Enum value maps for ModelSourceFileState.
var (
	ModelSourceFileState_name = map[int32]string{
		0: "MODEL_SOURCE_FILE_STATE_UNSPECIFIED",
		1: "MODEL_SOURCE_FILE_STATE_ACCEPTED",
		2: "MODEL_SOURCE_FILE_STATE_DOWNLOADING",
		3: "MODEL_SOURCE_FILE_STATE_VERIFIED",
		4: "MODEL_SOURCE_FILE_STATE_FAILED",
	}
	ModelSourceFileState_value = map[string]int32{
		"MODEL_SOURCE_FILE_STATE_UNSPECIFIED": 0,
		"MODEL_SOURCE_FILE_STATE_ACCEPTED":    1,
		"MODEL_SOURCE_FILE_STATE_DOWNLOADING": 2,
		"MODEL_SOURCE_FILE_STATE_VERIFIED":    3,
		"MODEL_SOURCE_FILE_STATE_FAILED":      4,
	}
)

func (x ModelSourceFileState) Enum() *ModelSourceFileState {
	p := new(ModelSourceFileState)
	*p = x
	return p
}

func (x ModelSourceFileState) String() string {
	return protoimpl.X.EnumStringOf(x.Descriptor(), protoreflect.EnumNumber(x))
}

func (ModelSourceFileState) Descriptor() protoreflect.EnumDescriptor {
	return file_cozy_worker_v1_worker_proto_enumTypes[25].Descriptor()
}

func (ModelSourceFileState) Type() protoreflect.EnumType {
	return &file_cozy_worker_v1_worker_proto_enumTypes[25]
}

func (x ModelSourceFileState) Number() protoreflect.EnumNumber {
	return protoreflect.EnumNumber(x)
}

// Deprecated: Use ModelSourceFileState.Descriptor instead.
func (ModelSourceFileState) EnumDescriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{25}
}

type ModelSourcePrepareOutcome int32

const (
	ModelSourcePrepareOutcome_MODEL_SOURCE_PREPARE_OUTCOME_UNSPECIFIED ModelSourcePrepareOutcome = 0
	ModelSourcePrepareOutcome_MODEL_SOURCE_PREPARE_OUTCOME_PREPARED    ModelSourcePrepareOutcome = 1
	ModelSourcePrepareOutcome_MODEL_SOURCE_PREPARE_OUTCOME_REPLAYED    ModelSourcePrepareOutcome = 2
	ModelSourcePrepareOutcome_MODEL_SOURCE_PREPARE_OUTCOME_REFUSED     ModelSourcePrepareOutcome = 3
)

// Enum value maps for ModelSourcePrepareOutcome.
var (
	ModelSourcePrepareOutcome_name = map[int32]string{
		0: "MODEL_SOURCE_PREPARE_OUTCOME_UNSPECIFIED",
		1: "MODEL_SOURCE_PREPARE_OUTCOME_PREPARED",
		2: "MODEL_SOURCE_PREPARE_OUTCOME_REPLAYED",
		3: "MODEL_SOURCE_PREPARE_OUTCOME_REFUSED",
	}
	ModelSourcePrepareOutcome_value = map[string]int32{
		"MODEL_SOURCE_PREPARE_OUTCOME_UNSPECIFIED": 0,
		"MODEL_SOURCE_PREPARE_OUTCOME_PREPARED":    1,
		"MODEL_SOURCE_PREPARE_OUTCOME_REPLAYED":    2,
		"MODEL_SOURCE_PREPARE_OUTCOME_REFUSED":     3,
	}
)

func (x ModelSourcePrepareOutcome) Enum() *ModelSourcePrepareOutcome {
	p := new(ModelSourcePrepareOutcome)
	*p = x
	return p
}

func (x ModelSourcePrepareOutcome) String() string {
	return protoimpl.X.EnumStringOf(x.Descriptor(), protoreflect.EnumNumber(x))
}

func (ModelSourcePrepareOutcome) Descriptor() protoreflect.EnumDescriptor {
	return file_cozy_worker_v1_worker_proto_enumTypes[26].Descriptor()
}

func (ModelSourcePrepareOutcome) Type() protoreflect.EnumType {
	return &file_cozy_worker_v1_worker_proto_enumTypes[26]
}

func (x ModelSourcePrepareOutcome) Number() protoreflect.EnumNumber {
	return protoreflect.EnumNumber(x)
}

// Deprecated: Use ModelSourcePrepareOutcome.Descriptor instead.
func (ModelSourcePrepareOutcome) EnumDescriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{26}
}

type ArtifactFinalizeDisposition int32

const (
	ArtifactFinalizeDisposition_ARTIFACT_FINALIZE_DISPOSITION_UNSPECIFIED         ArtifactFinalizeDisposition = 0
	ArtifactFinalizeDisposition_ARTIFACT_FINALIZE_DISPOSITION_ADOPT               ArtifactFinalizeDisposition = 1
	ArtifactFinalizeDisposition_ARTIFACT_FINALIZE_DISPOSITION_ABANDON             ArtifactFinalizeDisposition = 2
	ArtifactFinalizeDisposition_ARTIFACT_FINALIZE_DISPOSITION_ABANDON_UNCOMMITTED ArtifactFinalizeDisposition = 3
)

// Enum value maps for ArtifactFinalizeDisposition.
var (
	ArtifactFinalizeDisposition_name = map[int32]string{
		0: "ARTIFACT_FINALIZE_DISPOSITION_UNSPECIFIED",
		1: "ARTIFACT_FINALIZE_DISPOSITION_ADOPT",
		2: "ARTIFACT_FINALIZE_DISPOSITION_ABANDON",
		3: "ARTIFACT_FINALIZE_DISPOSITION_ABANDON_UNCOMMITTED",
	}
	ArtifactFinalizeDisposition_value = map[string]int32{
		"ARTIFACT_FINALIZE_DISPOSITION_UNSPECIFIED":         0,
		"ARTIFACT_FINALIZE_DISPOSITION_ADOPT":               1,
		"ARTIFACT_FINALIZE_DISPOSITION_ABANDON":             2,
		"ARTIFACT_FINALIZE_DISPOSITION_ABANDON_UNCOMMITTED": 3,
	}
)

func (x ArtifactFinalizeDisposition) Enum() *ArtifactFinalizeDisposition {
	p := new(ArtifactFinalizeDisposition)
	*p = x
	return p
}

func (x ArtifactFinalizeDisposition) String() string {
	return protoimpl.X.EnumStringOf(x.Descriptor(), protoreflect.EnumNumber(x))
}

func (ArtifactFinalizeDisposition) Descriptor() protoreflect.EnumDescriptor {
	return file_cozy_worker_v1_worker_proto_enumTypes[27].Descriptor()
}

func (ArtifactFinalizeDisposition) Type() protoreflect.EnumType {
	return &file_cozy_worker_v1_worker_proto_enumTypes[27]
}

func (x ArtifactFinalizeDisposition) Number() protoreflect.EnumNumber {
	return protoreflect.EnumNumber(x)
}

// Deprecated: Use ArtifactFinalizeDisposition.Descriptor instead.
func (ArtifactFinalizeDisposition) EnumDescriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{27}
}

type ArtifactFinalizeOutcome int32

const (
	ArtifactFinalizeOutcome_ARTIFACT_FINALIZE_OUTCOME_UNSPECIFIED ArtifactFinalizeOutcome = 0
	ArtifactFinalizeOutcome_ARTIFACT_FINALIZE_OUTCOME_ADOPTED     ArtifactFinalizeOutcome = 1
	ArtifactFinalizeOutcome_ARTIFACT_FINALIZE_OUTCOME_ABANDONED   ArtifactFinalizeOutcome = 2
)

// Enum value maps for ArtifactFinalizeOutcome.
var (
	ArtifactFinalizeOutcome_name = map[int32]string{
		0: "ARTIFACT_FINALIZE_OUTCOME_UNSPECIFIED",
		1: "ARTIFACT_FINALIZE_OUTCOME_ADOPTED",
		2: "ARTIFACT_FINALIZE_OUTCOME_ABANDONED",
	}
	ArtifactFinalizeOutcome_value = map[string]int32{
		"ARTIFACT_FINALIZE_OUTCOME_UNSPECIFIED": 0,
		"ARTIFACT_FINALIZE_OUTCOME_ADOPTED":     1,
		"ARTIFACT_FINALIZE_OUTCOME_ABANDONED":   2,
	}
)

func (x ArtifactFinalizeOutcome) Enum() *ArtifactFinalizeOutcome {
	p := new(ArtifactFinalizeOutcome)
	*p = x
	return p
}

func (x ArtifactFinalizeOutcome) String() string {
	return protoimpl.X.EnumStringOf(x.Descriptor(), protoreflect.EnumNumber(x))
}

func (ArtifactFinalizeOutcome) Descriptor() protoreflect.EnumDescriptor {
	return file_cozy_worker_v1_worker_proto_enumTypes[28].Descriptor()
}

func (ArtifactFinalizeOutcome) Type() protoreflect.EnumType {
	return &file_cozy_worker_v1_worker_proto_enumTypes[28]
}

func (x ArtifactFinalizeOutcome) Number() protoreflect.EnumNumber {
	return protoreflect.EnumNumber(x)
}

// Deprecated: Use ArtifactFinalizeOutcome.Descriptor instead.
func (ArtifactFinalizeOutcome) EnumDescriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{28}
}

type PreparePackageSetRequest struct {
	state              protoimpl.MessageState `protogen:"open.v1"`
	DownloadDelegation []byte                 `protobuf:"bytes,1,opt,name=download_delegation,json=downloadDelegation,proto3" json:"download_delegation,omitempty"` // exact canonical DownloadDelegation/1 bytes
	Files              []*LocalDownloadFile   `protobuf:"bytes,2,rep,name=files,proto3" json:"files,omitempty"`
	EnvironmentRoot    string                 `protobuf:"bytes,3,opt,name=environment_root,json=environmentRoot,proto3" json:"environment_root,omitempty"` // absolute supervisor-owned overlay root
	unknownFields      protoimpl.UnknownFields
	sizeCache          protoimpl.SizeCache
}

func (x *PreparePackageSetRequest) Reset() {
	*x = PreparePackageSetRequest{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[0]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *PreparePackageSetRequest) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*PreparePackageSetRequest) ProtoMessage() {}

func (x *PreparePackageSetRequest) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[0]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use PreparePackageSetRequest.ProtoReflect.Descriptor instead.
func (*PreparePackageSetRequest) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{0}
}

func (x *PreparePackageSetRequest) GetDownloadDelegation() []byte {
	if x != nil {
		return x.DownloadDelegation
	}
	return nil
}

func (x *PreparePackageSetRequest) GetFiles() []*LocalDownloadFile {
	if x != nil {
		return x.Files
	}
	return nil
}

func (x *PreparePackageSetRequest) GetEnvironmentRoot() string {
	if x != nil {
		return x.EnvironmentRoot
	}
	return ""
}

type LocalDownloadFile struct {
	state         protoimpl.MessageState `protogen:"open.v1"`
	Digest        []byte                 `protobuf:"bytes,1,opt,name=digest,proto3" json:"digest,omitempty"`
	Filename      string                 `protobuf:"bytes,2,opt,name=filename,proto3" json:"filename,omitempty"`
	Kind          LocalDownloadKind      `protobuf:"varint,3,opt,name=kind,proto3,enum=cozy.worker.v1.LocalDownloadKind" json:"kind,omitempty"`
	Length        uint64                 `protobuf:"varint,4,opt,name=length,proto3" json:"length,omitempty"`
	Path          string                 `protobuf:"bytes,5,opt,name=path,proto3" json:"path,omitempty"` // absolute verified supervisor-owned file
	unknownFields protoimpl.UnknownFields
	sizeCache     protoimpl.SizeCache
}

func (x *LocalDownloadFile) Reset() {
	*x = LocalDownloadFile{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[1]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *LocalDownloadFile) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*LocalDownloadFile) ProtoMessage() {}

func (x *LocalDownloadFile) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[1]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use LocalDownloadFile.ProtoReflect.Descriptor instead.
func (*LocalDownloadFile) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{1}
}

func (x *LocalDownloadFile) GetDigest() []byte {
	if x != nil {
		return x.Digest
	}
	return nil
}

func (x *LocalDownloadFile) GetFilename() string {
	if x != nil {
		return x.Filename
	}
	return ""
}

func (x *LocalDownloadFile) GetKind() LocalDownloadKind {
	if x != nil {
		return x.Kind
	}
	return LocalDownloadKind_LOCAL_DOWNLOAD_KIND_UNSPECIFIED
}

func (x *LocalDownloadFile) GetLength() uint64 {
	if x != nil {
		return x.Length
	}
	return 0
}

func (x *LocalDownloadFile) GetPath() string {
	if x != nil {
		return x.Path
	}
	return ""
}

type PreparePackageSetResult struct {
	state         protoimpl.MessageState `protogen:"open.v1"`
	PlacementSet  *DesiredPlacementSet   `protobuf:"bytes,1,opt,name=placement_set,json=placementSet,proto3" json:"placement_set,omitempty"`
	unknownFields protoimpl.UnknownFields
	sizeCache     protoimpl.SizeCache
}

func (x *PreparePackageSetResult) Reset() {
	*x = PreparePackageSetResult{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[2]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *PreparePackageSetResult) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*PreparePackageSetResult) ProtoMessage() {}

func (x *PreparePackageSetResult) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[2]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use PreparePackageSetResult.ProtoReflect.Descriptor instead.
func (*PreparePackageSetResult) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{2}
}

func (x *PreparePackageSetResult) GetPlacementSet() *DesiredPlacementSet {
	if x != nil {
		return x.PlacementSet
	}
	return nil
}

type LocalModelSourceFile struct {
	state         protoimpl.MessageState `protogen:"open.v1"`
	Member        string                 `protobuf:"bytes,1,opt,name=member,proto3" json:"member,omitempty"`                     // safe relative provider member, <= MaxModelSourceMemberBytes
	ObjectId      string                 `protobuf:"bytes,2,opt,name=object_id,json=objectId,proto3" json:"object_id,omitempty"` // lowercase sha256:<64 hex>
	Length        uint64                 `protobuf:"varint,3,opt,name=length,proto3" json:"length,omitempty"`
	Path          string                 `protobuf:"bytes,4,opt,name=path,proto3" json:"path,omitempty"` // absolute verified supervisor-owned file; loopback RPC only
	unknownFields protoimpl.UnknownFields
	sizeCache     protoimpl.SizeCache
}

func (x *LocalModelSourceFile) Reset() {
	*x = LocalModelSourceFile{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[3]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *LocalModelSourceFile) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*LocalModelSourceFile) ProtoMessage() {}

func (x *LocalModelSourceFile) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[3]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use LocalModelSourceFile.ProtoReflect.Descriptor instead.
func (*LocalModelSourceFile) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{3}
}

func (x *LocalModelSourceFile) GetMember() string {
	if x != nil {
		return x.Member
	}
	return ""
}

func (x *LocalModelSourceFile) GetObjectId() string {
	if x != nil {
		return x.ObjectId
	}
	return ""
}

func (x *LocalModelSourceFile) GetLength() uint64 {
	if x != nil {
		return x.Length
	}
	return 0
}

func (x *LocalModelSourceFile) GetPath() string {
	if x != nil {
		return x.Path
	}
	return ""
}

type ModelSourceProfile struct {
	state         protoimpl.MessageState `protogen:"open.v1"`
	Slot          string                 `protobuf:"bytes,1,opt,name=slot,proto3" json:"slot,omitempty"`
	Profile       string                 `protobuf:"bytes,2,opt,name=profile,proto3" json:"profile,omitempty"`
	unknownFields protoimpl.UnknownFields
	sizeCache     protoimpl.SizeCache
}

func (x *ModelSourceProfile) Reset() {
	*x = ModelSourceProfile{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[4]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *ModelSourceProfile) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*ModelSourceProfile) ProtoMessage() {}

func (x *ModelSourceProfile) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[4]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use ModelSourceProfile.ProtoReflect.Descriptor instead.
func (*ModelSourceProfile) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{4}
}

func (x *ModelSourceProfile) GetSlot() string {
	if x != nil {
		return x.Slot
	}
	return ""
}

func (x *ModelSourceProfile) GetProfile() string {
	if x != nil {
		return x.Profile
	}
	return ""
}

type PreparedModelSource struct {
	state         protoimpl.MessageState `protogen:"open.v1"`
	Slot          string                 `protobuf:"bytes,1,opt,name=slot,proto3" json:"slot,omitempty"`
	Profile       string                 `protobuf:"bytes,2,opt,name=profile,proto3" json:"profile,omitempty"`
	Manifest      *Ref                   `protobuf:"bytes,3,opt,name=manifest,proto3" json:"manifest,omitempty"`
	unknownFields protoimpl.UnknownFields
	sizeCache     protoimpl.SizeCache
}

func (x *PreparedModelSource) Reset() {
	*x = PreparedModelSource{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[5]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *PreparedModelSource) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*PreparedModelSource) ProtoMessage() {}

func (x *PreparedModelSource) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[5]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use PreparedModelSource.ProtoReflect.Descriptor instead.
func (*PreparedModelSource) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{5}
}

func (x *PreparedModelSource) GetSlot() string {
	if x != nil {
		return x.Slot
	}
	return ""
}

func (x *PreparedModelSource) GetProfile() string {
	if x != nil {
		return x.Profile
	}
	return ""
}

func (x *PreparedModelSource) GetManifest() *Ref {
	if x != nil {
		return x.Manifest
	}
	return nil
}

type PrepareModelSourceRequest struct {
	state                 protoimpl.MessageState  `protogen:"open.v1"`
	OperationId           string                  `protobuf:"bytes,1,opt,name=operation_id,json=operationId,proto3" json:"operation_id,omitempty"`
	SourceSelectionDigest []byte                  `protobuf:"bytes,2,opt,name=source_selection_digest,json=sourceSelectionDigest,proto3" json:"source_selection_digest,omitempty"`
	Profiles              []*ModelSourceProfile   `protobuf:"bytes,3,rep,name=profiles,proto3" json:"profiles,omitempty"`                                      // sorted unique by slot; <= MaxModelSourceProfiles
	Files                 []*LocalModelSourceFile `protobuf:"bytes,4,rep,name=files,proto3" json:"files,omitempty"`                                            // sorted unique by member; <= MaxModelSourceFiles
	SourceUri             string                  `protobuf:"bytes,5,opt,name=source_uri,json=sourceUri,proto3" json:"source_uri,omitempty"`                   // credential-free pinned hf:// or civitai:// provenance
	DeclaredLicense       string                  `protobuf:"bytes,6,opt,name=declared_license,json=declaredLicense,proto3" json:"declared_license,omitempty"` // bounded printable provenance; empty means undeclared
	unknownFields         protoimpl.UnknownFields
	sizeCache             protoimpl.SizeCache
}

func (x *PrepareModelSourceRequest) Reset() {
	*x = PrepareModelSourceRequest{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[6]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *PrepareModelSourceRequest) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*PrepareModelSourceRequest) ProtoMessage() {}

func (x *PrepareModelSourceRequest) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[6]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use PrepareModelSourceRequest.ProtoReflect.Descriptor instead.
func (*PrepareModelSourceRequest) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{6}
}

func (x *PrepareModelSourceRequest) GetOperationId() string {
	if x != nil {
		return x.OperationId
	}
	return ""
}

func (x *PrepareModelSourceRequest) GetSourceSelectionDigest() []byte {
	if x != nil {
		return x.SourceSelectionDigest
	}
	return nil
}

func (x *PrepareModelSourceRequest) GetProfiles() []*ModelSourceProfile {
	if x != nil {
		return x.Profiles
	}
	return nil
}

func (x *PrepareModelSourceRequest) GetFiles() []*LocalModelSourceFile {
	if x != nil {
		return x.Files
	}
	return nil
}

func (x *PrepareModelSourceRequest) GetSourceUri() string {
	if x != nil {
		return x.SourceUri
	}
	return ""
}

func (x *PrepareModelSourceRequest) GetDeclaredLicense() string {
	if x != nil {
		return x.DeclaredLicense
	}
	return ""
}

type PrepareModelSourceResult struct {
	state         protoimpl.MessageState    `protogen:"open.v1"`
	Outcome       ModelSourcePrepareOutcome `protobuf:"varint,1,opt,name=outcome,proto3,enum=cozy.worker.v1.ModelSourcePrepareOutcome" json:"outcome,omitempty"`
	Sources       []*PreparedModelSource    `protobuf:"bytes,2,rep,name=sources,proto3" json:"sources,omitempty"` // sorted unique by slot; exact profile set on success
	SafeCode      string                    `protobuf:"bytes,3,opt,name=safe_code,json=safeCode,proto3" json:"safe_code,omitempty"`
	SafeDetail    string                    `protobuf:"bytes,4,opt,name=safe_detail,json=safeDetail,proto3" json:"safe_detail,omitempty"`
	unknownFields protoimpl.UnknownFields
	sizeCache     protoimpl.SizeCache
}

func (x *PrepareModelSourceResult) Reset() {
	*x = PrepareModelSourceResult{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[7]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *PrepareModelSourceResult) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*PrepareModelSourceResult) ProtoMessage() {}

func (x *PrepareModelSourceResult) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[7]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use PrepareModelSourceResult.ProtoReflect.Descriptor instead.
func (*PrepareModelSourceResult) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{7}
}

func (x *PrepareModelSourceResult) GetOutcome() ModelSourcePrepareOutcome {
	if x != nil {
		return x.Outcome
	}
	return ModelSourcePrepareOutcome_MODEL_SOURCE_PREPARE_OUTCOME_UNSPECIFIED
}

func (x *PrepareModelSourceResult) GetSources() []*PreparedModelSource {
	if x != nil {
		return x.Sources
	}
	return nil
}

func (x *PrepareModelSourceResult) GetSafeCode() string {
	if x != nil {
		return x.SafeCode
	}
	return ""
}

func (x *PrepareModelSourceResult) GetSafeDetail() string {
	if x != nil {
		return x.SafeDetail
	}
	return ""
}

type RecordOwnerFrame struct {
	state protoimpl.MessageState `protogen:"open.v1"`
	// Types that are valid to be assigned to Msg:
	//
	//	*RecordOwnerFrame_Claim
	//	*RecordOwnerFrame_DesiredState
	//	*RecordOwnerFrame_AttemptOffer
	//	*RecordOwnerFrame_CancelAttempt
	//	*RecordOwnerFrame_OutcomeAck
	//	*RecordOwnerFrame_CheckpointReceipt
	//	*RecordOwnerFrame_SnapshotAck
	//	*RecordOwnerFrame_ArtifactFinalizeRequest
	//	*RecordOwnerFrame_ArtifactHostAck
	//	*RecordOwnerFrame_ArtifactTransferRequest
	//	*RecordOwnerFrame_ArtifactReadRequest
	//	*RecordOwnerFrame_ModelSourceFileRequest
	//	*RecordOwnerFrame_ModelSourcePrepareRequest
	Msg           isRecordOwnerFrame_Msg `protobuf_oneof:"msg"`
	unknownFields protoimpl.UnknownFields
	sizeCache     protoimpl.SizeCache
}

func (x *RecordOwnerFrame) Reset() {
	*x = RecordOwnerFrame{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[8]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *RecordOwnerFrame) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*RecordOwnerFrame) ProtoMessage() {}

func (x *RecordOwnerFrame) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[8]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use RecordOwnerFrame.ProtoReflect.Descriptor instead.
func (*RecordOwnerFrame) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{8}
}

func (x *RecordOwnerFrame) GetMsg() isRecordOwnerFrame_Msg {
	if x != nil {
		return x.Msg
	}
	return nil
}

func (x *RecordOwnerFrame) GetClaim() *Claim {
	if x != nil {
		if x, ok := x.Msg.(*RecordOwnerFrame_Claim); ok {
			return x.Claim
		}
	}
	return nil
}

func (x *RecordOwnerFrame) GetDesiredState() *DesiredWorkerState {
	if x != nil {
		if x, ok := x.Msg.(*RecordOwnerFrame_DesiredState); ok {
			return x.DesiredState
		}
	}
	return nil
}

func (x *RecordOwnerFrame) GetAttemptOffer() *AttemptOffer {
	if x != nil {
		if x, ok := x.Msg.(*RecordOwnerFrame_AttemptOffer); ok {
			return x.AttemptOffer
		}
	}
	return nil
}

func (x *RecordOwnerFrame) GetCancelAttempt() *CancelAttempt {
	if x != nil {
		if x, ok := x.Msg.(*RecordOwnerFrame_CancelAttempt); ok {
			return x.CancelAttempt
		}
	}
	return nil
}

func (x *RecordOwnerFrame) GetOutcomeAck() *AttemptOutcomeAck {
	if x != nil {
		if x, ok := x.Msg.(*RecordOwnerFrame_OutcomeAck); ok {
			return x.OutcomeAck
		}
	}
	return nil
}

func (x *RecordOwnerFrame) GetCheckpointReceipt() *JobCheckpointReceipt {
	if x != nil {
		if x, ok := x.Msg.(*RecordOwnerFrame_CheckpointReceipt); ok {
			return x.CheckpointReceipt
		}
	}
	return nil
}

func (x *RecordOwnerFrame) GetSnapshotAck() *SnapshotAck {
	if x != nil {
		if x, ok := x.Msg.(*RecordOwnerFrame_SnapshotAck); ok {
			return x.SnapshotAck
		}
	}
	return nil
}

func (x *RecordOwnerFrame) GetArtifactFinalizeRequest() *ArtifactFinalizeRequest {
	if x != nil {
		if x, ok := x.Msg.(*RecordOwnerFrame_ArtifactFinalizeRequest); ok {
			return x.ArtifactFinalizeRequest
		}
	}
	return nil
}

func (x *RecordOwnerFrame) GetArtifactHostAck() *ArtifactHostAck {
	if x != nil {
		if x, ok := x.Msg.(*RecordOwnerFrame_ArtifactHostAck); ok {
			return x.ArtifactHostAck
		}
	}
	return nil
}

func (x *RecordOwnerFrame) GetArtifactTransferRequest() *ArtifactTransferRequest {
	if x != nil {
		if x, ok := x.Msg.(*RecordOwnerFrame_ArtifactTransferRequest); ok {
			return x.ArtifactTransferRequest
		}
	}
	return nil
}

func (x *RecordOwnerFrame) GetArtifactReadRequest() *ArtifactReadRequest {
	if x != nil {
		if x, ok := x.Msg.(*RecordOwnerFrame_ArtifactReadRequest); ok {
			return x.ArtifactReadRequest
		}
	}
	return nil
}

func (x *RecordOwnerFrame) GetModelSourceFileRequest() *ModelSourceFileRequest {
	if x != nil {
		if x, ok := x.Msg.(*RecordOwnerFrame_ModelSourceFileRequest); ok {
			return x.ModelSourceFileRequest
		}
	}
	return nil
}

func (x *RecordOwnerFrame) GetModelSourcePrepareRequest() *ModelSourcePrepareRequest {
	if x != nil {
		if x, ok := x.Msg.(*RecordOwnerFrame_ModelSourcePrepareRequest); ok {
			return x.ModelSourcePrepareRequest
		}
	}
	return nil
}

type isRecordOwnerFrame_Msg interface {
	isRecordOwnerFrame_Msg()
}

type RecordOwnerFrame_Claim struct {
	Claim *Claim `protobuf:"bytes,5,opt,name=claim,proto3,oneof"`
}

type RecordOwnerFrame_DesiredState struct {
	DesiredState *DesiredWorkerState `protobuf:"bytes,6,opt,name=desired_state,json=desiredState,proto3,oneof"`
}

type RecordOwnerFrame_AttemptOffer struct {
	AttemptOffer *AttemptOffer `protobuf:"bytes,7,opt,name=attempt_offer,json=attemptOffer,proto3,oneof"`
}

type RecordOwnerFrame_CancelAttempt struct {
	CancelAttempt *CancelAttempt `protobuf:"bytes,8,opt,name=cancel_attempt,json=cancelAttempt,proto3,oneof"`
}

type RecordOwnerFrame_OutcomeAck struct {
	OutcomeAck *AttemptOutcomeAck `protobuf:"bytes,9,opt,name=outcome_ack,json=outcomeAck,proto3,oneof"`
}

type RecordOwnerFrame_CheckpointReceipt struct {
	CheckpointReceipt *JobCheckpointReceipt `protobuf:"bytes,10,opt,name=checkpoint_receipt,json=checkpointReceipt,proto3,oneof"`
}

type RecordOwnerFrame_SnapshotAck struct {
	SnapshotAck *SnapshotAck `protobuf:"bytes,11,opt,name=snapshot_ack,json=snapshotAck,proto3,oneof"`
}

type RecordOwnerFrame_ArtifactFinalizeRequest struct {
	ArtifactFinalizeRequest *ArtifactFinalizeRequest `protobuf:"bytes,17,opt,name=artifact_finalize_request,json=artifactFinalizeRequest,proto3,oneof"`
}

type RecordOwnerFrame_ArtifactHostAck struct {
	// pod-supervisor -> Runtime injection. An external RecordOwner MUST NOT author this frame.
	ArtifactHostAck *ArtifactHostAck `protobuf:"bytes,19,opt,name=artifact_host_ack,json=artifactHostAck,proto3,oneof"`
}

type RecordOwnerFrame_ArtifactTransferRequest struct {
	// External RecordOwner -> pod-supervisor only. It MUST NOT be forwarded to Runtime.
	ArtifactTransferRequest *ArtifactTransferRequest `protobuf:"bytes,20,opt,name=artifact_transfer_request,json=artifactTransferRequest,proto3,oneof"`
}

type RecordOwnerFrame_ArtifactReadRequest struct {
	// pod-supervisor -> Runtime injection. An external RecordOwner MUST NOT author it.
	ArtifactReadRequest *ArtifactReadRequest `protobuf:"bytes,21,opt,name=artifact_read_request,json=artifactReadRequest,proto3,oneof"`
}

type RecordOwnerFrame_ModelSourceFileRequest struct {
	ModelSourceFileRequest *ModelSourceFileRequest `protobuf:"bytes,22,opt,name=model_source_file_request,json=modelSourceFileRequest,proto3,oneof"` // external owner -> supervisor only
}

type RecordOwnerFrame_ModelSourcePrepareRequest struct {
	ModelSourcePrepareRequest *ModelSourcePrepareRequest `protobuf:"bytes,23,opt,name=model_source_prepare_request,json=modelSourcePrepareRequest,proto3,oneof"` // owner -> supervisor only
}

func (*RecordOwnerFrame_Claim) isRecordOwnerFrame_Msg() {}

func (*RecordOwnerFrame_DesiredState) isRecordOwnerFrame_Msg() {}

func (*RecordOwnerFrame_AttemptOffer) isRecordOwnerFrame_Msg() {}

func (*RecordOwnerFrame_CancelAttempt) isRecordOwnerFrame_Msg() {}

func (*RecordOwnerFrame_OutcomeAck) isRecordOwnerFrame_Msg() {}

func (*RecordOwnerFrame_CheckpointReceipt) isRecordOwnerFrame_Msg() {}

func (*RecordOwnerFrame_SnapshotAck) isRecordOwnerFrame_Msg() {}

func (*RecordOwnerFrame_ArtifactFinalizeRequest) isRecordOwnerFrame_Msg() {}

func (*RecordOwnerFrame_ArtifactHostAck) isRecordOwnerFrame_Msg() {}

func (*RecordOwnerFrame_ArtifactTransferRequest) isRecordOwnerFrame_Msg() {}

func (*RecordOwnerFrame_ArtifactReadRequest) isRecordOwnerFrame_Msg() {}

func (*RecordOwnerFrame_ModelSourceFileRequest) isRecordOwnerFrame_Msg() {}

func (*RecordOwnerFrame_ModelSourcePrepareRequest) isRecordOwnerFrame_Msg() {}

type WorkerFrame struct {
	state protoimpl.MessageState `protogen:"open.v1"`
	// Types that are valid to be assigned to Msg:
	//
	//	*WorkerFrame_ClaimAck
	//	*WorkerFrame_ObservedState
	//	*WorkerFrame_AttemptAccepted
	//	*WorkerFrame_AttemptOutcome
	//	*WorkerFrame_BootFailure
	//	*WorkerFrame_CheckpointRequest
	//	*WorkerFrame_CheckpointAck
	//	*WorkerFrame_Snapshot
	//	*WorkerFrame_ArtifactFinalizeResult
	//	*WorkerFrame_ArtifactIntent
	//	*WorkerFrame_ArtifactReceipt
	//	*WorkerFrame_ArtifactReadResult
	//	*WorkerFrame_ArtifactTransferStatus
	//	*WorkerFrame_ModelSourceFileStatus
	//	*WorkerFrame_ModelSourcePrepared
	Msg           isWorkerFrame_Msg `protobuf_oneof:"msg"`
	unknownFields protoimpl.UnknownFields
	sizeCache     protoimpl.SizeCache
}

func (x *WorkerFrame) Reset() {
	*x = WorkerFrame{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[9]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *WorkerFrame) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*WorkerFrame) ProtoMessage() {}

func (x *WorkerFrame) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[9]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use WorkerFrame.ProtoReflect.Descriptor instead.
func (*WorkerFrame) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{9}
}

func (x *WorkerFrame) GetMsg() isWorkerFrame_Msg {
	if x != nil {
		return x.Msg
	}
	return nil
}

func (x *WorkerFrame) GetClaimAck() *ClaimAck {
	if x != nil {
		if x, ok := x.Msg.(*WorkerFrame_ClaimAck); ok {
			return x.ClaimAck
		}
	}
	return nil
}

func (x *WorkerFrame) GetObservedState() *ObservedWorkerState {
	if x != nil {
		if x, ok := x.Msg.(*WorkerFrame_ObservedState); ok {
			return x.ObservedState
		}
	}
	return nil
}

func (x *WorkerFrame) GetAttemptAccepted() *AttemptAccepted {
	if x != nil {
		if x, ok := x.Msg.(*WorkerFrame_AttemptAccepted); ok {
			return x.AttemptAccepted
		}
	}
	return nil
}

func (x *WorkerFrame) GetAttemptOutcome() *AttemptOutcome {
	if x != nil {
		if x, ok := x.Msg.(*WorkerFrame_AttemptOutcome); ok {
			return x.AttemptOutcome
		}
	}
	return nil
}

func (x *WorkerFrame) GetBootFailure() *BootFailure {
	if x != nil {
		if x, ok := x.Msg.(*WorkerFrame_BootFailure); ok {
			return x.BootFailure
		}
	}
	return nil
}

func (x *WorkerFrame) GetCheckpointRequest() *JobCheckpointRequest {
	if x != nil {
		if x, ok := x.Msg.(*WorkerFrame_CheckpointRequest); ok {
			return x.CheckpointRequest
		}
	}
	return nil
}

func (x *WorkerFrame) GetCheckpointAck() *JobCheckpointAck {
	if x != nil {
		if x, ok := x.Msg.(*WorkerFrame_CheckpointAck); ok {
			return x.CheckpointAck
		}
	}
	return nil
}

func (x *WorkerFrame) GetSnapshot() *WorkerSnapshot {
	if x != nil {
		if x, ok := x.Msg.(*WorkerFrame_Snapshot); ok {
			return x.Snapshot
		}
	}
	return nil
}

func (x *WorkerFrame) GetArtifactFinalizeResult() *ArtifactFinalizeResult {
	if x != nil {
		if x, ok := x.Msg.(*WorkerFrame_ArtifactFinalizeResult); ok {
			return x.ArtifactFinalizeResult
		}
	}
	return nil
}

func (x *WorkerFrame) GetArtifactIntent() *ArtifactIntentFrame {
	if x != nil {
		if x, ok := x.Msg.(*WorkerFrame_ArtifactIntent); ok {
			return x.ArtifactIntent
		}
	}
	return nil
}

func (x *WorkerFrame) GetArtifactReceipt() *ArtifactReceiptFrame {
	if x != nil {
		if x, ok := x.Msg.(*WorkerFrame_ArtifactReceipt); ok {
			return x.ArtifactReceipt
		}
	}
	return nil
}

func (x *WorkerFrame) GetArtifactReadResult() *ArtifactReadResult {
	if x != nil {
		if x, ok := x.Msg.(*WorkerFrame_ArtifactReadResult); ok {
			return x.ArtifactReadResult
		}
	}
	return nil
}

func (x *WorkerFrame) GetArtifactTransferStatus() *ArtifactTransferStatus {
	if x != nil {
		if x, ok := x.Msg.(*WorkerFrame_ArtifactTransferStatus); ok {
			return x.ArtifactTransferStatus
		}
	}
	return nil
}

func (x *WorkerFrame) GetModelSourceFileStatus() *ModelSourceFileStatus {
	if x != nil {
		if x, ok := x.Msg.(*WorkerFrame_ModelSourceFileStatus); ok {
			return x.ModelSourceFileStatus
		}
	}
	return nil
}

func (x *WorkerFrame) GetModelSourcePrepared() *ModelSourcePrepared {
	if x != nil {
		if x, ok := x.Msg.(*WorkerFrame_ModelSourcePrepared); ok {
			return x.ModelSourcePrepared
		}
	}
	return nil
}

type isWorkerFrame_Msg interface {
	isWorkerFrame_Msg()
}

type WorkerFrame_ClaimAck struct {
	ClaimAck *ClaimAck `protobuf:"bytes,5,opt,name=claim_ack,json=claimAck,proto3,oneof"`
}

type WorkerFrame_ObservedState struct {
	ObservedState *ObservedWorkerState `protobuf:"bytes,6,opt,name=observed_state,json=observedState,proto3,oneof"`
}

type WorkerFrame_AttemptAccepted struct {
	AttemptAccepted *AttemptAccepted `protobuf:"bytes,7,opt,name=attempt_accepted,json=attemptAccepted,proto3,oneof"`
}

type WorkerFrame_AttemptOutcome struct {
	AttemptOutcome *AttemptOutcome `protobuf:"bytes,8,opt,name=attempt_outcome,json=attemptOutcome,proto3,oneof"`
}

type WorkerFrame_BootFailure struct {
	// Valid as the stream's ONLY substantive reply: the boot-fatal verdict INSTEAD OF ClaimAck.
	BootFailure *BootFailure `protobuf:"bytes,9,opt,name=boot_failure,json=bootFailure,proto3,oneof"`
}

type WorkerFrame_CheckpointRequest struct {
	CheckpointRequest *JobCheckpointRequest `protobuf:"bytes,10,opt,name=checkpoint_request,json=checkpointRequest,proto3,oneof"`
}

type WorkerFrame_CheckpointAck struct {
	CheckpointAck *JobCheckpointAck `protobuf:"bytes,11,opt,name=checkpoint_ack,json=checkpointAck,proto3,oneof"`
}

type WorkerFrame_Snapshot struct {
	Snapshot *WorkerSnapshot `protobuf:"bytes,12,opt,name=snapshot,proto3,oneof"`
}

type WorkerFrame_ArtifactFinalizeResult struct {
	ArtifactFinalizeResult *ArtifactFinalizeResult `protobuf:"bytes,16,opt,name=artifact_finalize_result,json=artifactFinalizeResult,proto3,oneof"`
}

type WorkerFrame_ArtifactIntent struct {
	// Runtime -> pod-supervisor; forwarded only after the supervisor's durable host act.
	ArtifactIntent *ArtifactIntentFrame `protobuf:"bytes,17,opt,name=artifact_intent,json=artifactIntent,proto3,oneof"`
}

type WorkerFrame_ArtifactReceipt struct {
	ArtifactReceipt *ArtifactReceiptFrame `protobuf:"bytes,18,opt,name=artifact_receipt,json=artifactReceipt,proto3,oneof"`
}

type WorkerFrame_ArtifactReadResult struct {
	// Runtime -> pod-supervisor only. The supervisor consumes it instead of forwarding bytes.
	ArtifactReadResult *ArtifactReadResult `protobuf:"bytes,19,opt,name=artifact_read_result,json=artifactReadResult,proto3,oneof"`
}

type WorkerFrame_ArtifactTransferStatus struct {
	// pod-supervisor -> external RecordOwner only. Runtime MUST NOT author it.
	ArtifactTransferStatus *ArtifactTransferStatus `protobuf:"bytes,20,opt,name=artifact_transfer_status,json=artifactTransferStatus,proto3,oneof"`
}

type WorkerFrame_ModelSourceFileStatus struct {
	ModelSourceFileStatus *ModelSourceFileStatus `protobuf:"bytes,21,opt,name=model_source_file_status,json=modelSourceFileStatus,proto3,oneof"` // supervisor -> external owner only
}

type WorkerFrame_ModelSourcePrepared struct {
	ModelSourcePrepared *ModelSourcePrepared `protobuf:"bytes,22,opt,name=model_source_prepared,json=modelSourcePrepared,proto3,oneof"` // supervisor -> external owner only
}

func (*WorkerFrame_ClaimAck) isWorkerFrame_Msg() {}

func (*WorkerFrame_ObservedState) isWorkerFrame_Msg() {}

func (*WorkerFrame_AttemptAccepted) isWorkerFrame_Msg() {}

func (*WorkerFrame_AttemptOutcome) isWorkerFrame_Msg() {}

func (*WorkerFrame_BootFailure) isWorkerFrame_Msg() {}

func (*WorkerFrame_CheckpointRequest) isWorkerFrame_Msg() {}

func (*WorkerFrame_CheckpointAck) isWorkerFrame_Msg() {}

func (*WorkerFrame_Snapshot) isWorkerFrame_Msg() {}

func (*WorkerFrame_ArtifactFinalizeResult) isWorkerFrame_Msg() {}

func (*WorkerFrame_ArtifactIntent) isWorkerFrame_Msg() {}

func (*WorkerFrame_ArtifactReceipt) isWorkerFrame_Msg() {}

func (*WorkerFrame_ArtifactReadResult) isWorkerFrame_Msg() {}

func (*WorkerFrame_ArtifactTransferStatus) isWorkerFrame_Msg() {}

func (*WorkerFrame_ModelSourceFileStatus) isWorkerFrame_Msg() {}

func (*WorkerFrame_ModelSourcePrepared) isWorkerFrame_Msg() {}

type Claim struct {
	state                   protoimpl.MessageState `protogen:"open.v1"`
	RecordOwnerEpoch        uint64                 `protobuf:"varint,1,opt,name=record_owner_epoch,json=recordOwnerEpoch,proto3" json:"record_owner_epoch,omitempty"`                      // envelope; the claimed authority generation
	ControlStreamGeneration uint64                 `protobuf:"varint,2,opt,name=control_stream_generation,json=controlStreamGeneration,proto3" json:"control_stream_generation,omitempty"` // 0 on Claim (none minted yet)
	WorkerBootId            string                 `protobuf:"bytes,3,opt,name=worker_boot_id,json=workerBootId,proto3" json:"worker_boot_id,omitempty"`                                   // empty on a fresh dial; set when re-dialing a known boot
	RecordOwnerId           string                 `protobuf:"bytes,5,opt,name=record_owner_id,json=recordOwnerId,proto3" json:"record_owner_id,omitempty"`                                // stable identity of the claiming RecordOwner
	WorkerId                string                 `protobuf:"bytes,6,opt,name=worker_id,json=workerId,proto3" json:"worker_id,omitempty"`                                                 // the worker identity the RecordOwner expects to be claiming
	WireMinor               uint32                 `protobuf:"varint,7,opt,name=wire_minor,json=wireMinor,proto3" json:"wire_minor,omitempty"`                                             // highest minor the RECORDOWNER implements
	Proof                   []byte                 `protobuf:"bytes,8,opt,name=proof,proto3" json:"proof,omitempty"`                                                                       // 64-byte Ed25519 signature over canonical ClaimProof/1 on
	unknownFields           protoimpl.UnknownFields
	sizeCache               protoimpl.SizeCache
}

func (x *Claim) Reset() {
	*x = Claim{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[10]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *Claim) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*Claim) ProtoMessage() {}

func (x *Claim) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[10]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use Claim.ProtoReflect.Descriptor instead.
func (*Claim) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{10}
}

func (x *Claim) GetRecordOwnerEpoch() uint64 {
	if x != nil {
		return x.RecordOwnerEpoch
	}
	return 0
}

func (x *Claim) GetControlStreamGeneration() uint64 {
	if x != nil {
		return x.ControlStreamGeneration
	}
	return 0
}

func (x *Claim) GetWorkerBootId() string {
	if x != nil {
		return x.WorkerBootId
	}
	return ""
}

func (x *Claim) GetRecordOwnerId() string {
	if x != nil {
		return x.RecordOwnerId
	}
	return ""
}

func (x *Claim) GetWorkerId() string {
	if x != nil {
		return x.WorkerId
	}
	return ""
}

func (x *Claim) GetWireMinor() uint32 {
	if x != nil {
		return x.WireMinor
	}
	return 0
}

func (x *Claim) GetProof() []byte {
	if x != nil {
		return x.Proof
	}
	return nil
}

// DOCUMENT SHAPE (not a wire message): canonical form `cozy.worker.v1.ClaimProof/1`.
// The Cozy daemon signs these exact reconstructed bytes with the rental key before its first
// Claim. There is no challenge, timestamp, jti, or application request-signature grammar: epoch,
// boot, worker, and exact TLS leaf bind replay to this worker lifetime.
type ClaimProof struct {
	state                      protoimpl.MessageState `protogen:"open.v1"`
	RecordOwnerEpoch           uint64                 `protobuf:"varint,1,opt,name=record_owner_epoch,json=recordOwnerEpoch,proto3" json:"record_owner_epoch,omitempty"`
	WorkerBootId               string                 `protobuf:"bytes,2,opt,name=worker_boot_id,json=workerBootId,proto3" json:"worker_boot_id,omitempty"`
	WorkerId                   string                 `protobuf:"bytes,3,opt,name=worker_id,json=workerId,proto3" json:"worker_id,omitempty"`
	WorkerTlsCertificateDigest []byte                 `protobuf:"bytes,4,opt,name=worker_tls_certificate_digest,json=workerTlsCertificateDigest,proto3" json:"worker_tls_certificate_digest,omitempty"`
	unknownFields              protoimpl.UnknownFields
	sizeCache                  protoimpl.SizeCache
}

func (x *ClaimProof) Reset() {
	*x = ClaimProof{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[11]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *ClaimProof) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*ClaimProof) ProtoMessage() {}

func (x *ClaimProof) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[11]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use ClaimProof.ProtoReflect.Descriptor instead.
func (*ClaimProof) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{11}
}

func (x *ClaimProof) GetRecordOwnerEpoch() uint64 {
	if x != nil {
		return x.RecordOwnerEpoch
	}
	return 0
}

func (x *ClaimProof) GetWorkerBootId() string {
	if x != nil {
		return x.WorkerBootId
	}
	return ""
}

func (x *ClaimProof) GetWorkerId() string {
	if x != nil {
		return x.WorkerId
	}
	return ""
}

func (x *ClaimProof) GetWorkerTlsCertificateDigest() []byte {
	if x != nil {
		return x.WorkerTlsCertificateDigest
	}
	return nil
}

// Claim proof is evaluated before readiness or ownership mutation. A bad proof answers
// UNAUTHENTICATED even before ready. A valid proof while the worker's post-bind readiness barrier
// is closed answers UNDURABLE without minting a control_stream_generation. On acceptance the
// worker fences the previous stream (if any), increments
// control_stream_generation, and answers with its identity + one WorkerSnapshot. DISPATCH STAYS
// CLOSED until the RecordOwner's SnapshotAck (02 §6) — admission_state reports CLOSED until then.
type ClaimAck struct {
	state                   protoimpl.MessageState `protogen:"open.v1"`
	RecordOwnerEpoch        uint64                 `protobuf:"varint,1,opt,name=record_owner_epoch,json=recordOwnerEpoch,proto3" json:"record_owner_epoch,omitempty"`                      // echo of the accepted epoch
	ControlStreamGeneration uint64                 `protobuf:"varint,2,opt,name=control_stream_generation,json=controlStreamGeneration,proto3" json:"control_stream_generation,omitempty"` // newly minted for this stream
	WorkerBootId            string                 `protobuf:"bytes,3,opt,name=worker_boot_id,json=workerBootId,proto3" json:"worker_boot_id,omitempty"`                                   // this boot's identity
	Accepted                bool                   `protobuf:"varint,5,opt,name=accepted,proto3" json:"accepted,omitempty"`                                                                // false => `rejection` set; stream then closes
	Rejection               ClaimRejection         `protobuf:"varint,6,opt,name=rejection,proto3,enum=cozy.worker.v1.ClaimRejection" json:"rejection,omitempty"`
	WireMinor               uint32                 `protobuf:"varint,7,opt,name=wire_minor,json=wireMinor,proto3" json:"wire_minor,omitempty"` // highest minor the WORKER implements
	WorkerId                string                 `protobuf:"bytes,8,opt,name=worker_id,json=workerId,proto3" json:"worker_id,omitempty"`     // CREDENTIAL subject; cross-checked vs the worker's server
	// identity (02 §4)
	WorkerInstanceId string `protobuf:"bytes,9,opt,name=worker_instance_id,json=workerInstanceId,proto3" json:"worker_instance_id,omitempty"` // ONE provisioned instance lifetime; outcome-replay
	// authorization keys on this, never worker_id
	WorkerReleaseId      string `protobuf:"bytes,10,opt,name=worker_release_id,json=workerReleaseId,proto3" json:"worker_release_id,omitempty"`
	ControlRuntimeDigest string `protobuf:"bytes,11,opt,name=control_runtime_digest,json=controlRuntimeDigest,proto3" json:"control_runtime_digest,omitempty"` // class (b), `sha256:<hex>`: exact image-owned control
	// Runtime wheel bytes measured from the installed wheelhouse.
	// It is NOT the OCI image identity: an image cannot bake its
	// own manifest digest without a self-reference. The provisioner
	// independently verifies the selected immutable OCI identity;
	// a RecordOwner needs BOTH facts to admit the machine.
	GitCommit     string           `protobuf:"bytes,12,opt,name=git_commit,json=gitCommit,proto3" json:"git_commit,omitempty"`
	Resources     *WorkerResources `protobuf:"bytes,13,opt,name=resources,proto3" json:"resources,omitempty"` // torch-free worker statics; never re-sent mid-stream.
	unknownFields protoimpl.UnknownFields
	sizeCache     protoimpl.SizeCache
}

func (x *ClaimAck) Reset() {
	*x = ClaimAck{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[12]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *ClaimAck) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*ClaimAck) ProtoMessage() {}

func (x *ClaimAck) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[12]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use ClaimAck.ProtoReflect.Descriptor instead.
func (*ClaimAck) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{12}
}

func (x *ClaimAck) GetRecordOwnerEpoch() uint64 {
	if x != nil {
		return x.RecordOwnerEpoch
	}
	return 0
}

func (x *ClaimAck) GetControlStreamGeneration() uint64 {
	if x != nil {
		return x.ControlStreamGeneration
	}
	return 0
}

func (x *ClaimAck) GetWorkerBootId() string {
	if x != nil {
		return x.WorkerBootId
	}
	return ""
}

func (x *ClaimAck) GetAccepted() bool {
	if x != nil {
		return x.Accepted
	}
	return false
}

func (x *ClaimAck) GetRejection() ClaimRejection {
	if x != nil {
		return x.Rejection
	}
	return ClaimRejection_CLAIM_REJECTION_UNSPECIFIED
}

func (x *ClaimAck) GetWireMinor() uint32 {
	if x != nil {
		return x.WireMinor
	}
	return 0
}

func (x *ClaimAck) GetWorkerId() string {
	if x != nil {
		return x.WorkerId
	}
	return ""
}

func (x *ClaimAck) GetWorkerInstanceId() string {
	if x != nil {
		return x.WorkerInstanceId
	}
	return ""
}

func (x *ClaimAck) GetWorkerReleaseId() string {
	if x != nil {
		return x.WorkerReleaseId
	}
	return ""
}

func (x *ClaimAck) GetControlRuntimeDigest() string {
	if x != nil {
		return x.ControlRuntimeDigest
	}
	return ""
}

func (x *ClaimAck) GetGitCommit() string {
	if x != nil {
		return x.GitCommit
	}
	return ""
}

func (x *ClaimAck) GetResources() *WorkerResources {
	if x != nil {
		return x.Resources
	}
	return nil
}

// Authenticated boot-fatal delivery: the worker accepted a stream but cannot serve.
type BootFailure struct {
	state                   protoimpl.MessageState `protogen:"open.v1"`
	RecordOwnerEpoch        uint64                 `protobuf:"varint,1,opt,name=record_owner_epoch,json=recordOwnerEpoch,proto3" json:"record_owner_epoch,omitempty"`
	ControlStreamGeneration uint64                 `protobuf:"varint,2,opt,name=control_stream_generation,json=controlStreamGeneration,proto3" json:"control_stream_generation,omitempty"`
	WorkerBootId            string                 `protobuf:"bytes,3,opt,name=worker_boot_id,json=workerBootId,proto3" json:"worker_boot_id,omitempty"`
	WorkerId                string                 `protobuf:"bytes,5,opt,name=worker_id,json=workerId,proto3" json:"worker_id,omitempty"`
	WorkerInstanceId        string                 `protobuf:"bytes,6,opt,name=worker_instance_id,json=workerInstanceId,proto3" json:"worker_instance_id,omitempty"`
	Reason                  BootFailureReason      `protobuf:"varint,7,opt,name=reason,proto3,enum=cozy.worker.v1.BootFailureReason" json:"reason,omitempty"`
	Detail                  string                 `protobuf:"bytes,8,opt,name=detail,proto3" json:"detail,omitempty"`                                                            // bounded <= 1024 bytes, sanitized
	Resources               *WorkerResources       `protobuf:"bytes,9,opt,name=resources,proto3" json:"resources,omitempty"`                                                      // whatever was measurable
	ControlRuntimeDigest    string                 `protobuf:"bytes,10,opt,name=control_runtime_digest,json=controlRuntimeDigest,proto3" json:"control_runtime_digest,omitempty"` // same measured control Runtime provenance as ClaimAck 11
	WorkerReleaseId         string                 `protobuf:"bytes,11,opt,name=worker_release_id,json=workerReleaseId,proto3" json:"worker_release_id,omitempty"`
	unknownFields           protoimpl.UnknownFields
	sizeCache               protoimpl.SizeCache
}

func (x *BootFailure) Reset() {
	*x = BootFailure{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[13]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *BootFailure) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*BootFailure) ProtoMessage() {}

func (x *BootFailure) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[13]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use BootFailure.ProtoReflect.Descriptor instead.
func (*BootFailure) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{13}
}

func (x *BootFailure) GetRecordOwnerEpoch() uint64 {
	if x != nil {
		return x.RecordOwnerEpoch
	}
	return 0
}

func (x *BootFailure) GetControlStreamGeneration() uint64 {
	if x != nil {
		return x.ControlStreamGeneration
	}
	return 0
}

func (x *BootFailure) GetWorkerBootId() string {
	if x != nil {
		return x.WorkerBootId
	}
	return ""
}

func (x *BootFailure) GetWorkerId() string {
	if x != nil {
		return x.WorkerId
	}
	return ""
}

func (x *BootFailure) GetWorkerInstanceId() string {
	if x != nil {
		return x.WorkerInstanceId
	}
	return ""
}

func (x *BootFailure) GetReason() BootFailureReason {
	if x != nil {
		return x.Reason
	}
	return BootFailureReason_BOOT_FAILURE_REASON_UNSPECIFIED
}

func (x *BootFailure) GetDetail() string {
	if x != nil {
		return x.Detail
	}
	return ""
}

func (x *BootFailure) GetResources() *WorkerResources {
	if x != nil {
		return x.Resources
	}
	return nil
}

func (x *BootFailure) GetControlRuntimeDigest() string {
	if x != nil {
		return x.ControlRuntimeDigest
	}
	return ""
}

func (x *BootFailure) GetWorkerReleaseId() string {
	if x != nil {
		return x.WorkerReleaseId
	}
	return ""
}

type WorkerSnapshot struct {
	state                              protoimpl.MessageState `protogen:"open.v1"`
	RecordOwnerEpoch                   uint64                 `protobuf:"varint,1,opt,name=record_owner_epoch,json=recordOwnerEpoch,proto3" json:"record_owner_epoch,omitempty"`
	ControlStreamGeneration            uint64                 `protobuf:"varint,2,opt,name=control_stream_generation,json=controlStreamGeneration,proto3" json:"control_stream_generation,omitempty"`
	WorkerBootId                       string                 `protobuf:"bytes,3,opt,name=worker_boot_id,json=workerBootId,proto3" json:"worker_boot_id,omitempty"`
	SnapshotId                         string                 `protobuf:"bytes,5,opt,name=snapshot_id,json=snapshotId,proto3" json:"snapshot_id,omitempty"`                                                                               // worker-minted; names this exact snapshot
	SnapshotDigest                     []byte                 `protobuf:"bytes,6,opt,name=snapshot_digest,json=snapshotDigest,proto3" json:"snapshot_digest,omitempty"`                                                                   // class (a): sha256 over EXACTLY the bytes in 7
	SnapshotCanonicalBytes             []byte                 `protobuf:"bytes,7,opt,name=snapshot_canonical_bytes,json=snapshotCanonicalBytes,proto3" json:"snapshot_canonical_bytes,omitempty"`                                         // the WorkerSnapshotBody DOCUMENT, canonical JSON bytes
	AcceptedPlacementSetCanonicalBytes []byte                 `protobuf:"bytes,8,opt,name=accepted_placement_set_canonical_bytes,json=acceptedPlacementSetCanonicalBytes,proto3" json:"accepted_placement_set_canonical_bytes,omitempty"` // the EXACT accepted set bytes as journaled
	unknownFields                      protoimpl.UnknownFields
	sizeCache                          protoimpl.SizeCache
}

func (x *WorkerSnapshot) Reset() {
	*x = WorkerSnapshot{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[14]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *WorkerSnapshot) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*WorkerSnapshot) ProtoMessage() {}

func (x *WorkerSnapshot) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[14]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use WorkerSnapshot.ProtoReflect.Descriptor instead.
func (*WorkerSnapshot) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{14}
}

func (x *WorkerSnapshot) GetRecordOwnerEpoch() uint64 {
	if x != nil {
		return x.RecordOwnerEpoch
	}
	return 0
}

func (x *WorkerSnapshot) GetControlStreamGeneration() uint64 {
	if x != nil {
		return x.ControlStreamGeneration
	}
	return 0
}

func (x *WorkerSnapshot) GetWorkerBootId() string {
	if x != nil {
		return x.WorkerBootId
	}
	return ""
}

func (x *WorkerSnapshot) GetSnapshotId() string {
	if x != nil {
		return x.SnapshotId
	}
	return ""
}

func (x *WorkerSnapshot) GetSnapshotDigest() []byte {
	if x != nil {
		return x.SnapshotDigest
	}
	return nil
}

func (x *WorkerSnapshot) GetSnapshotCanonicalBytes() []byte {
	if x != nil {
		return x.SnapshotCanonicalBytes
	}
	return nil
}

func (x *WorkerSnapshot) GetAcceptedPlacementSetCanonicalBytes() []byte {
	if x != nil {
		return x.AcceptedPlacementSetCanonicalBytes
	}
	return nil
}

// DOCUMENT SHAPE (not a wire message): canonical form `cozy.worker.v1.WorkerSnapshotBody/1`, the
// subject of WorkerSnapshot.snapshot_digest.
//
// NAMING (implementation resolution, rev §5): the rev document calls both the wire message and
// the document `WorkerSnapshot`, which one proto package cannot express. The document takes the
// `Body` suffix, exactly as AttemptOutcome/AttemptOutcomeBody already does, so the wire message
// keeps the name the rev's proto sketch and WorkerFrame slot 12 give it.
//
// BOUNDEDNESS IS STRUCTURAL, NOT A PROMISE: available_attempt_slots counts BOTH running attempts
// and outcomes awaiting ack, so a RecordOwner that stops acking runs itself out of admission
// rather than growing the worker's held set.
type WorkerSnapshotBody struct {
	state                        protoimpl.MessageState `protogen:"open.v1"`
	AcceptedDesiredStateRevision uint64                 `protobuf:"varint,1,opt,name=accepted_desired_state_revision,json=acceptedDesiredStateRevision,proto3" json:"accepted_desired_state_revision,omitempty"` // durably ACCEPTED intent
	AcceptedPlacementSetDigest   []byte                 `protobuf:"bytes,2,opt,name=accepted_placement_set_digest,json=acceptedPlacementSetDigest,proto3" json:"accepted_placement_set_digest,omitempty"`        // class (a); equals sha256 of the bytes the
	// enclosing WorkerSnapshot carries in field 8
	JournalHighwater      uint64                       `protobuf:"varint,3,opt,name=journal_highwater,json=journalHighwater,proto3" json:"journal_highwater,omitempty"`                  // the journal position this snapshot reflects
	WorkerPhase           WorkerPhase                  `protobuf:"varint,4,opt,name=worker_phase,json=workerPhase,proto3,enum=cozy.worker.v1.WorkerPhase" json:"worker_phase,omitempty"` // machine lifecycle
	Placements            []*PlacementStatus           `protobuf:"bytes,5,rep,name=placements,proto3" json:"placements,omitempty"`                                                       // sorted by placement_id
	ConvergedRevision     uint64                       `protobuf:"varint,6,opt,name=converged_revision,json=convergedRevision,proto3" json:"converged_revision,omitempty"`               // advances ONLY when observed satisfies accepted
	AdmissionGeneration   uint64                       `protobuf:"varint,7,opt,name=admission_generation,json=admissionGeneration,proto3" json:"admission_generation,omitempty"`
	AdmissionState        AdmissionState               `protobuf:"varint,8,opt,name=admission_state,json=admissionState,proto3,enum=cozy.worker.v1.AdmissionState" json:"admission_state,omitempty"`
	AvailableAttemptSlots uint32                       `protobuf:"varint,9,opt,name=available_attempt_slots,json=availableAttemptSlots,proto3" json:"available_attempt_slots,omitempty"`
	HeldAttempts          []*HeldAttempt               `protobuf:"bytes,10,rep,name=held_attempts,json=heldAttempts,proto3" json:"held_attempts,omitempty"`                         // running attempts AND outcomes pending ack
	ArtifactTransactions  []*ArtifactTransactionStatus `protobuf:"bytes,11,rep,name=artifact_transactions,json=artifactTransactions,proto3" json:"artifact_transactions,omitempty"` // sorted compact supervisor-
	unknownFields         protoimpl.UnknownFields
	sizeCache             protoimpl.SizeCache
}

func (x *WorkerSnapshotBody) Reset() {
	*x = WorkerSnapshotBody{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[15]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *WorkerSnapshotBody) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*WorkerSnapshotBody) ProtoMessage() {}

func (x *WorkerSnapshotBody) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[15]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use WorkerSnapshotBody.ProtoReflect.Descriptor instead.
func (*WorkerSnapshotBody) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{15}
}

func (x *WorkerSnapshotBody) GetAcceptedDesiredStateRevision() uint64 {
	if x != nil {
		return x.AcceptedDesiredStateRevision
	}
	return 0
}

func (x *WorkerSnapshotBody) GetAcceptedPlacementSetDigest() []byte {
	if x != nil {
		return x.AcceptedPlacementSetDigest
	}
	return nil
}

func (x *WorkerSnapshotBody) GetJournalHighwater() uint64 {
	if x != nil {
		return x.JournalHighwater
	}
	return 0
}

func (x *WorkerSnapshotBody) GetWorkerPhase() WorkerPhase {
	if x != nil {
		return x.WorkerPhase
	}
	return WorkerPhase_WORKER_PHASE_UNSPECIFIED
}

func (x *WorkerSnapshotBody) GetPlacements() []*PlacementStatus {
	if x != nil {
		return x.Placements
	}
	return nil
}

func (x *WorkerSnapshotBody) GetConvergedRevision() uint64 {
	if x != nil {
		return x.ConvergedRevision
	}
	return 0
}

func (x *WorkerSnapshotBody) GetAdmissionGeneration() uint64 {
	if x != nil {
		return x.AdmissionGeneration
	}
	return 0
}

func (x *WorkerSnapshotBody) GetAdmissionState() AdmissionState {
	if x != nil {
		return x.AdmissionState
	}
	return AdmissionState_ADMISSION_STATE_UNSPECIFIED
}

func (x *WorkerSnapshotBody) GetAvailableAttemptSlots() uint32 {
	if x != nil {
		return x.AvailableAttemptSlots
	}
	return 0
}

func (x *WorkerSnapshotBody) GetHeldAttempts() []*HeldAttempt {
	if x != nil {
		return x.HeldAttempts
	}
	return nil
}

func (x *WorkerSnapshotBody) GetArtifactTransactions() []*ArtifactTransactionStatus {
	if x != nil {
		return x.ArtifactTransactions
	}
	return nil
}

type SnapshotAck struct {
	state                   protoimpl.MessageState `protogen:"open.v1"`
	RecordOwnerEpoch        uint64                 `protobuf:"varint,1,opt,name=record_owner_epoch,json=recordOwnerEpoch,proto3" json:"record_owner_epoch,omitempty"`
	ControlStreamGeneration uint64                 `protobuf:"varint,2,opt,name=control_stream_generation,json=controlStreamGeneration,proto3" json:"control_stream_generation,omitempty"`
	WorkerBootId            string                 `protobuf:"bytes,3,opt,name=worker_boot_id,json=workerBootId,proto3" json:"worker_boot_id,omitempty"`
	SnapshotId              string                 `protobuf:"bytes,5,opt,name=snapshot_id,json=snapshotId,proto3" json:"snapshot_id,omitempty"`             // echo of the exact snapshot durably reconciled
	SnapshotDigest          []byte                 `protobuf:"bytes,6,opt,name=snapshot_digest,json=snapshotDigest,proto3" json:"snapshot_digest,omitempty"` // echo; COMPARED, never recomputed by the worker. An ack
	unknownFields           protoimpl.UnknownFields
	sizeCache               protoimpl.SizeCache
}

func (x *SnapshotAck) Reset() {
	*x = SnapshotAck{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[16]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *SnapshotAck) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*SnapshotAck) ProtoMessage() {}

func (x *SnapshotAck) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[16]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use SnapshotAck.ProtoReflect.Descriptor instead.
func (*SnapshotAck) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{16}
}

func (x *SnapshotAck) GetRecordOwnerEpoch() uint64 {
	if x != nil {
		return x.RecordOwnerEpoch
	}
	return 0
}

func (x *SnapshotAck) GetControlStreamGeneration() uint64 {
	if x != nil {
		return x.ControlStreamGeneration
	}
	return 0
}

func (x *SnapshotAck) GetWorkerBootId() string {
	if x != nil {
		return x.WorkerBootId
	}
	return ""
}

func (x *SnapshotAck) GetSnapshotId() string {
	if x != nil {
		return x.SnapshotId
	}
	return ""
}

func (x *SnapshotAck) GetSnapshotDigest() []byte {
	if x != nil {
		return x.SnapshotDigest
	}
	return nil
}

type DesiredWorkerState struct {
	state                   protoimpl.MessageState `protogen:"open.v1"`
	RecordOwnerEpoch        uint64                 `protobuf:"varint,1,opt,name=record_owner_epoch,json=recordOwnerEpoch,proto3" json:"record_owner_epoch,omitempty"`
	ControlStreamGeneration uint64                 `protobuf:"varint,2,opt,name=control_stream_generation,json=controlStreamGeneration,proto3" json:"control_stream_generation,omitempty"`
	WorkerBootId            string                 `protobuf:"bytes,3,opt,name=worker_boot_id,json=workerBootId,proto3" json:"worker_boot_id,omitempty"`
	Revision                uint64                 `protobuf:"varint,5,opt,name=revision,proto3" json:"revision,omitempty"` // RecordOwner-owned, monotonic; older ignored, equal
	// idempotent, newer applies; equal revision + different
	// bytes is a protocol error (02 §3 — now byte-checkable
	// rather than claim-checkable)
	Posture      Posture `protobuf:"varint,6,opt,name=posture,proto3,enum=cozy.worker.v1.Posture" json:"posture,omitempty"`
	WireMinor    uint32  `protobuf:"varint,7,opt,name=wire_minor,json=wireMinor,proto3" json:"wire_minor,omitempty"`            // the minor the RecordOwner speaks on this stream; echoed
	DrainGraceMs uint64  `protobuf:"varint,8,opt,name=drain_grace_ms,json=drainGraceMs,proto3" json:"drain_grace_ms,omitempty"` // draining posture only: grace before forceful; 0 = the
	// worker's runtime-constant default
	//
	// Types that are valid to be assigned to Mode:
	//
	//	*DesiredWorkerState_Job
	//	*DesiredWorkerState_PlacementSet
	//	*DesiredWorkerState_PackageSet
	Mode          isDesiredWorkerState_Mode `protobuf_oneof:"mode"`
	unknownFields protoimpl.UnknownFields
	sizeCache     protoimpl.SizeCache
}

func (x *DesiredWorkerState) Reset() {
	*x = DesiredWorkerState{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[17]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *DesiredWorkerState) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*DesiredWorkerState) ProtoMessage() {}

func (x *DesiredWorkerState) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[17]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use DesiredWorkerState.ProtoReflect.Descriptor instead.
func (*DesiredWorkerState) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{17}
}

func (x *DesiredWorkerState) GetRecordOwnerEpoch() uint64 {
	if x != nil {
		return x.RecordOwnerEpoch
	}
	return 0
}

func (x *DesiredWorkerState) GetControlStreamGeneration() uint64 {
	if x != nil {
		return x.ControlStreamGeneration
	}
	return 0
}

func (x *DesiredWorkerState) GetWorkerBootId() string {
	if x != nil {
		return x.WorkerBootId
	}
	return ""
}

func (x *DesiredWorkerState) GetRevision() uint64 {
	if x != nil {
		return x.Revision
	}
	return 0
}

func (x *DesiredWorkerState) GetPosture() Posture {
	if x != nil {
		return x.Posture
	}
	return Posture_POSTURE_UNSPECIFIED
}

func (x *DesiredWorkerState) GetWireMinor() uint32 {
	if x != nil {
		return x.WireMinor
	}
	return 0
}

func (x *DesiredWorkerState) GetDrainGraceMs() uint64 {
	if x != nil {
		return x.DrainGraceMs
	}
	return 0
}

func (x *DesiredWorkerState) GetMode() isDesiredWorkerState_Mode {
	if x != nil {
		return x.Mode
	}
	return nil
}

func (x *DesiredWorkerState) GetJob() *JobDirective {
	if x != nil {
		if x, ok := x.Mode.(*DesiredWorkerState_Job); ok {
			return x.Job
		}
	}
	return nil
}

func (x *DesiredWorkerState) GetPlacementSet() *DesiredPlacementSet {
	if x != nil {
		if x, ok := x.Mode.(*DesiredWorkerState_PlacementSet); ok {
			return x.PlacementSet
		}
	}
	return nil
}

func (x *DesiredWorkerState) GetPackageSet() *DesiredPackageSet {
	if x != nil {
		if x, ok := x.Mode.(*DesiredWorkerState_PackageSet); ok {
			return x.PackageSet
		}
	}
	return nil
}

type isDesiredWorkerState_Mode interface {
	isDesiredWorkerState_Mode()
}

type DesiredWorkerState_Job struct {
	Job *JobDirective `protobuf:"bytes,13,opt,name=job,proto3,oneof"`
}

type DesiredWorkerState_PlacementSet struct {
	PlacementSet *DesiredPlacementSet `protobuf:"bytes,15,opt,name=placement_set,json=placementSet,proto3,oneof"`
}

type DesiredWorkerState_PackageSet struct {
	// Private rentals send only Creator's logical exact package/model set. The pod host
	// verifies and resolves it; Creator never reconstructs a qualified Linux/CUDA PlacementSet.
	PackageSet *DesiredPackageSet `protobuf:"bytes,16,opt,name=package_set,json=packageSet,proto3,oneof"`
}

func (*DesiredWorkerState_Job) isDesiredWorkerState_Mode() {}

func (*DesiredWorkerState_PlacementSet) isDesiredWorkerState_Mode() {}

func (*DesiredWorkerState_PackageSet) isDesiredWorkerState_Mode() {}

type DesiredPackageSet struct {
	state                       protoimpl.MessageState `protogen:"open.v1"`
	DownloadDelegation          []byte                 `protobuf:"bytes,1,opt,name=download_delegation,json=downloadDelegation,proto3" json:"download_delegation,omitempty"`                              // exact canonical DownloadDelegation/1 bytes
	DownloadDelegationSignature []byte                 `protobuf:"bytes,2,opt,name=download_delegation_signature,json=downloadDelegationSignature,proto3" json:"download_delegation_signature,omitempty"` // 64-byte Ed25519 signature by the rental Creator key
	unknownFields               protoimpl.UnknownFields
	sizeCache                   protoimpl.SizeCache
}

func (x *DesiredPackageSet) Reset() {
	*x = DesiredPackageSet{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[18]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *DesiredPackageSet) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*DesiredPackageSet) ProtoMessage() {}

func (x *DesiredPackageSet) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[18]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use DesiredPackageSet.ProtoReflect.Descriptor instead.
func (*DesiredPackageSet) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{18}
}

func (x *DesiredPackageSet) GetDownloadDelegation() []byte {
	if x != nil {
		return x.DownloadDelegation
	}
	return nil
}

func (x *DesiredPackageSet) GetDownloadDelegationSignature() []byte {
	if x != nil {
		return x.DownloadDelegationSignature
	}
	return nil
}

// The serving-mode desired state, GENUINELY content-addressed (§4; was DeploymentSetDirective).
// The one-digester ECHO is replaced by a byte fence in the exact shape InvocationSpec already
// uses, and the duplicate structured representation is DELETED — there is no second place for
// the set to disagree with itself.
//
// ORDER OF OPERATIONS, NON-NEGOTIABLE: recompute sha256(placement_set_canonical_bytes) and
// compare BEFORE parsing a single field; on match, journal the EXACT accepted bytes and their
// digest (never a re-serialization, never a structured projection); only then parse under
// unknown-field refusal. Recovery, status attribution, and the snapshot all quote the journaled
// bytes. A mismatch is FAULT_KIND_PLACEMENT_SET_DIGEST_MISMATCH, UNAPPLIED, with
// accepted_desired_state_revision NOT advancing and the previous set still serving.
type DesiredPlacementSet struct {
	state                      protoimpl.MessageState `protogen:"open.v1"`
	PlacementSetDigest         []byte                 `protobuf:"bytes,1,opt,name=placement_set_digest,json=placementSetDigest,proto3" json:"placement_set_digest,omitempty"`                           // class (a): sha256 over EXACTLY the bytes in 3
	PlacementSetCanonicalBytes []byte                 `protobuf:"bytes,3,opt,name=placement_set_canonical_bytes,json=placementSetCanonicalBytes,proto3" json:"placement_set_canonical_bytes,omitempty"` // the PlacementSet DOCUMENT, canonical JSON bytes
	unknownFields              protoimpl.UnknownFields
	sizeCache                  protoimpl.SizeCache
}

func (x *DesiredPlacementSet) Reset() {
	*x = DesiredPlacementSet{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[19]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *DesiredPlacementSet) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*DesiredPlacementSet) ProtoMessage() {}

func (x *DesiredPlacementSet) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[19]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use DesiredPlacementSet.ProtoReflect.Descriptor instead.
func (*DesiredPlacementSet) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{19}
}

func (x *DesiredPlacementSet) GetPlacementSetDigest() []byte {
	if x != nil {
		return x.PlacementSetDigest
	}
	return nil
}

func (x *DesiredPlacementSet) GetPlacementSetCanonicalBytes() []byte {
	if x != nil {
		return x.PlacementSetCanonicalBytes
	}
	return nil
}

// DOCUMENT SHAPE (not a wire message): canonical form `cozy.worker.v1.PlacementSet/1`.
// Placements sorted by placement_id. LAUNCH ENFORCES len(placements) <= 1 (header note); #475
// adds that when the clamp lifts, multi-placement serving launches CO-FITTING ONLY.
type PlacementSet struct {
	state         protoimpl.MessageState `protogen:"open.v1"`
	Placements    []*Placement           `protobuf:"bytes,1,rep,name=placements,proto3" json:"placements,omitempty"`
	unknownFields protoimpl.UnknownFields
	sizeCache     protoimpl.SizeCache
}

func (x *PlacementSet) Reset() {
	*x = PlacementSet{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[20]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *PlacementSet) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*PlacementSet) ProtoMessage() {}

func (x *PlacementSet) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[20]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use PlacementSet.ProtoReflect.Descriptor instead.
func (*PlacementSet) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{20}
}

func (x *PlacementSet) GetPlacements() []*Placement {
	if x != nil {
		return x.Placements
	}
	return nil
}

// DOCUMENT SHAPE: canonical `cozy.worker.v1.DownloadDelegation/1`. Creator signs the exact
// canonical bytes. Replay is intentionally valid until expiry; Tensorhub keeps no nonce state.
type DownloadDelegation struct {
	state                      protoimpl.MessageState `protogen:"open.v1"`
	ExpiresAtUnix              uint64                 `protobuf:"varint,1,opt,name=expires_at_unix,json=expiresAtUnix,proto3" json:"expires_at_unix,omitempty"`
	Models                     []*DownloadModelRef    `protobuf:"bytes,2,rep,name=models,proto3" json:"models,omitempty"`     // sorted unique by (package,slot,model,release,manifest)
	Packages                   []*DownloadPackageRef  `protobuf:"bytes,3,rep,name=packages,proto3" json:"packages,omitempty"` // sorted unique by (package,release)
	RentalId                   string                 `protobuf:"bytes,4,opt,name=rental_id,json=rentalId,proto3" json:"rental_id,omitempty"`
	WorkerBootId               string                 `protobuf:"bytes,5,opt,name=worker_boot_id,json=workerBootId,proto3" json:"worker_boot_id,omitempty"`
	WorkerId                   string                 `protobuf:"bytes,6,opt,name=worker_id,json=workerId,proto3" json:"worker_id,omitempty"`
	WorkerTlsCertificateDigest []byte                 `protobuf:"bytes,7,opt,name=worker_tls_certificate_digest,json=workerTlsCertificateDigest,proto3" json:"worker_tls_certificate_digest,omitempty"` // sha256 of exact readiness-pinned leaf DER
	unknownFields              protoimpl.UnknownFields
	sizeCache                  protoimpl.SizeCache
}

func (x *DownloadDelegation) Reset() {
	*x = DownloadDelegation{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[21]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *DownloadDelegation) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*DownloadDelegation) ProtoMessage() {}

func (x *DownloadDelegation) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[21]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use DownloadDelegation.ProtoReflect.Descriptor instead.
func (*DownloadDelegation) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{21}
}

func (x *DownloadDelegation) GetExpiresAtUnix() uint64 {
	if x != nil {
		return x.ExpiresAtUnix
	}
	return 0
}

func (x *DownloadDelegation) GetModels() []*DownloadModelRef {
	if x != nil {
		return x.Models
	}
	return nil
}

func (x *DownloadDelegation) GetPackages() []*DownloadPackageRef {
	if x != nil {
		return x.Packages
	}
	return nil
}

func (x *DownloadDelegation) GetRentalId() string {
	if x != nil {
		return x.RentalId
	}
	return ""
}

func (x *DownloadDelegation) GetWorkerBootId() string {
	if x != nil {
		return x.WorkerBootId
	}
	return ""
}

func (x *DownloadDelegation) GetWorkerId() string {
	if x != nil {
		return x.WorkerId
	}
	return ""
}

func (x *DownloadDelegation) GetWorkerTlsCertificateDigest() []byte {
	if x != nil {
		return x.WorkerTlsCertificateDigest
	}
	return nil
}

type DownloadModelRef struct {
	state         protoimpl.MessageState `protogen:"open.v1"`
	Manifest      string                 `protobuf:"bytes,1,opt,name=manifest,proto3" json:"manifest,omitempty"` // sha256:<64 lowercase hex>
	Model         string                 `protobuf:"bytes,2,opt,name=model,proto3" json:"model,omitempty"`       // org/name
	Release       string                 `protobuf:"bytes,3,opt,name=release,proto3" json:"release,omitempty"`   // immutable author release
	Package       string                 `protobuf:"bytes,4,opt,name=package,proto3" json:"package,omitempty"`   // one package named by this delegation
	Slot          string                 `protobuf:"bytes,5,opt,name=slot,proto3" json:"slot,omitempty"`         // exact descriptor model-slot path; unique per package
	unknownFields protoimpl.UnknownFields
	sizeCache     protoimpl.SizeCache
}

func (x *DownloadModelRef) Reset() {
	*x = DownloadModelRef{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[22]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *DownloadModelRef) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*DownloadModelRef) ProtoMessage() {}

func (x *DownloadModelRef) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[22]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use DownloadModelRef.ProtoReflect.Descriptor instead.
func (*DownloadModelRef) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{22}
}

func (x *DownloadModelRef) GetManifest() string {
	if x != nil {
		return x.Manifest
	}
	return ""
}

func (x *DownloadModelRef) GetModel() string {
	if x != nil {
		return x.Model
	}
	return ""
}

func (x *DownloadModelRef) GetRelease() string {
	if x != nil {
		return x.Release
	}
	return ""
}

func (x *DownloadModelRef) GetPackage() string {
	if x != nil {
		return x.Package
	}
	return ""
}

func (x *DownloadModelRef) GetSlot() string {
	if x != nil {
		return x.Slot
	}
	return ""
}

type DownloadPackageRef struct {
	state         protoimpl.MessageState `protogen:"open.v1"`
	Package       string                 `protobuf:"bytes,1,opt,name=package,proto3" json:"package,omitempty"`                                  // org/name
	Release       string                 `protobuf:"bytes,2,opt,name=release,proto3" json:"release,omitempty"`                                  // exact semver release
	ReleaseDigest string                 `protobuf:"bytes,3,opt,name=release_digest,json=releaseDigest,proto3" json:"release_digest,omitempty"` // exact immutable PackageRelease/1 digest
	unknownFields protoimpl.UnknownFields
	sizeCache     protoimpl.SizeCache
}

func (x *DownloadPackageRef) Reset() {
	*x = DownloadPackageRef{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[23]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *DownloadPackageRef) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*DownloadPackageRef) ProtoMessage() {}

func (x *DownloadPackageRef) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[23]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use DownloadPackageRef.ProtoReflect.Descriptor instead.
func (*DownloadPackageRef) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{23}
}

func (x *DownloadPackageRef) GetPackage() string {
	if x != nil {
		return x.Package
	}
	return ""
}

func (x *DownloadPackageRef) GetRelease() string {
	if x != nil {
		return x.Release
	}
	return ""
}

func (x *DownloadPackageRef) GetReleaseDigest() string {
	if x != nil {
		return x.ReleaseDigest
	}
	return ""
}

type Placement struct {
	state       protoimpl.MessageState `protogen:"open.v1"`
	PlacementId string                 `protobuf:"bytes,1,opt,name=placement_id,json=placementId,proto3" json:"placement_id,omitempty"` // RecordOwner-minted routing + journal key; NEVER part of
	// invocation identity (InvocationSpec digests exclude it)
	//
	// Types that are valid to be assigned to PackageMode:
	//
	//	*Placement_Package
	//	*Placement_Development
	PackageMode       isPlacement_PackageMode `protobuf_oneof:"package_mode"`
	EnvironmentDigest []byte                  `protobuf:"bytes,3,opt,name=environment_digest,json=environmentDigest,proto3" json:"environment_digest,omitempty"` // class (a): exact selected Environment identity
	PackageDescriptor *Ref                    `protobuf:"bytes,5,opt,name=package_descriptor,json=packageDescriptor,proto3" json:"package_descriptor,omitempty"` // exact PackageDescriptor bytes
	BindingsDigest    []byte                  `protobuf:"bytes,6,opt,name=bindings_digest,json=bindingsDigest,proto3" json:"bindings_digest,omitempty"`          // class (a): sha256(canonical JSON of exactly
	// {entrypoints:<field 8>,models:<field 7>})
	Models        []*Model      `protobuf:"bytes,7,rep,name=models,proto3" json:"models,omitempty"`            // sorted unique by id; [] for a weightless package
	Entrypoints   []*Entrypoint `protobuf:"bytes,8,rep,name=entrypoints,proto3" json:"entrypoints,omitempty"`  // sorted unique by name
	Environment   *Environment  `protobuf:"bytes,10,opt,name=environment,proto3" json:"environment,omitempty"` // exact installed overlay wheels; the worker reads its
	unknownFields protoimpl.UnknownFields
	sizeCache     protoimpl.SizeCache
}

func (x *Placement) Reset() {
	*x = Placement{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[24]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *Placement) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*Placement) ProtoMessage() {}

func (x *Placement) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[24]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use Placement.ProtoReflect.Descriptor instead.
func (*Placement) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{24}
}

func (x *Placement) GetPlacementId() string {
	if x != nil {
		return x.PlacementId
	}
	return ""
}

func (x *Placement) GetPackageMode() isPlacement_PackageMode {
	if x != nil {
		return x.PackageMode
	}
	return nil
}

func (x *Placement) GetPackage() *PackageSelection {
	if x != nil {
		if x, ok := x.PackageMode.(*Placement_Package); ok {
			return x.Package
		}
	}
	return nil
}

func (x *Placement) GetDevelopment() *DevelopmentPackage {
	if x != nil {
		if x, ok := x.PackageMode.(*Placement_Development); ok {
			return x.Development
		}
	}
	return nil
}

func (x *Placement) GetEnvironmentDigest() []byte {
	if x != nil {
		return x.EnvironmentDigest
	}
	return nil
}

func (x *Placement) GetPackageDescriptor() *Ref {
	if x != nil {
		return x.PackageDescriptor
	}
	return nil
}

func (x *Placement) GetBindingsDigest() []byte {
	if x != nil {
		return x.BindingsDigest
	}
	return nil
}

func (x *Placement) GetModels() []*Model {
	if x != nil {
		return x.Models
	}
	return nil
}

func (x *Placement) GetEntrypoints() []*Entrypoint {
	if x != nil {
		return x.Entrypoints
	}
	return nil
}

func (x *Placement) GetEnvironment() *Environment {
	if x != nil {
		return x.Environment
	}
	return nil
}

type isPlacement_PackageMode interface {
	isPlacement_PackageMode()
}

type Placement_Package struct {
	Package *PackageSelection `protobuf:"bytes,2,opt,name=package,proto3,oneof"` // installed published release
}

type Placement_Development struct {
	Development *DevelopmentPackage `protobuf:"bytes,11,opt,name=development,proto3,oneof"` // explicit source-checkout execution
}

func (*Placement_Package) isPlacement_PackageMode() {}

func (*Placement_Development) isPlacement_PackageMode() {}

// Exact immutable bytes. Nested placement facts use this one reference shape instead of generic
// artifact subjects, role strings, or digest-only pointers.
type Ref struct {
	state         protoimpl.MessageState `protogen:"open.v1"`
	Digest        []byte                 `protobuf:"bytes,1,opt,name=digest,proto3" json:"digest,omitempty"`
	Length        uint64                 `protobuf:"varint,2,opt,name=length,proto3" json:"length,omitempty"`
	unknownFields protoimpl.UnknownFields
	sizeCache     protoimpl.SizeCache
}

func (x *Ref) Reset() {
	*x = Ref{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[25]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *Ref) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*Ref) ProtoMessage() {}

func (x *Ref) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[25]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use Ref.ProtoReflect.Descriptor instead.
func (*Ref) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{25}
}

func (x *Ref) GetDigest() []byte {
	if x != nil {
		return x.Digest
	}
	return nil
}

func (x *Ref) GetLength() uint64 {
	if x != nil {
		return x.Length
	}
	return 0
}

// One exact wheel. PackageSelection owns the project wheel; Environment owns selected non-base
// overlay wheels. The worker reads packages already present from its image-owned base inventory.
type WheelFact struct {
	state         protoimpl.MessageState `protogen:"open.v1"`
	Ref           *Ref                   `protobuf:"bytes,1,opt,name=ref,proto3" json:"ref,omitempty"`
	Distribution  string                 `protobuf:"bytes,2,opt,name=distribution,proto3" json:"distribution,omitempty"` // normalized lowercase distribution name
	Version       string                 `protobuf:"bytes,3,opt,name=version,proto3" json:"version,omitempty"`
	Filename      string                 `protobuf:"bytes,4,opt,name=filename,proto3" json:"filename,omitempty"`
	ImportRoots   []string               `protobuf:"bytes,5,rep,name=import_roots,json=importRoots,proto3" json:"import_roots,omitempty"` // sorted unique
	Tags          []string               `protobuf:"bytes,6,rep,name=tags,proto3" json:"tags,omitempty"`                                  // sorted unique
	unknownFields protoimpl.UnknownFields
	sizeCache     protoimpl.SizeCache
}

func (x *WheelFact) Reset() {
	*x = WheelFact{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[26]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *WheelFact) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*WheelFact) ProtoMessage() {}

func (x *WheelFact) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[26]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use WheelFact.ProtoReflect.Descriptor instead.
func (*WheelFact) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{26}
}

func (x *WheelFact) GetRef() *Ref {
	if x != nil {
		return x.Ref
	}
	return nil
}

func (x *WheelFact) GetDistribution() string {
	if x != nil {
		return x.Distribution
	}
	return ""
}

func (x *WheelFact) GetVersion() string {
	if x != nil {
		return x.Version
	}
	return ""
}

func (x *WheelFact) GetFilename() string {
	if x != nil {
		return x.Filename
	}
	return ""
}

func (x *WheelFact) GetImportRoots() []string {
	if x != nil {
		return x.ImportRoots
	}
	return nil
}

func (x *WheelFact) GetTags() []string {
	if x != nil {
		return x.Tags
	}
	return nil
}

// The directly executable package identity. release_digest is provenance for the immutable
// release.json that Tensorhub committed; it is deliberately digest-only because no worker fetch,
// parse, or second package authority remains.
type PackageSelection struct {
	state         protoimpl.MessageState `protogen:"open.v1"`
	Package       string                 `protobuf:"bytes,1,opt,name=package,proto3" json:"package,omitempty"`                                  // exact org/name package identity
	Release       string                 `protobuf:"bytes,2,opt,name=release,proto3" json:"release,omitempty"`                                  // exact semantic release
	ReleaseDigest []byte                 `protobuf:"bytes,3,opt,name=release_digest,json=releaseDigest,proto3" json:"release_digest,omitempty"` // sha256 of release.json; provenance, never an artifact Ref
	ProjectWheel  *WheelFact             `protobuf:"bytes,4,opt,name=project_wheel,json=projectWheel,proto3" json:"project_wheel,omitempty"`
	unknownFields protoimpl.UnknownFields
	sizeCache     protoimpl.SizeCache
}

func (x *PackageSelection) Reset() {
	*x = PackageSelection{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[27]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *PackageSelection) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*PackageSelection) ProtoMessage() {}

func (x *PackageSelection) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[27]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use PackageSelection.ProtoReflect.Descriptor instead.
func (*PackageSelection) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{27}
}

func (x *PackageSelection) GetPackage() string {
	if x != nil {
		return x.Package
	}
	return ""
}

func (x *PackageSelection) GetRelease() string {
	if x != nil {
		return x.Release
	}
	return ""
}

func (x *PackageSelection) GetReleaseDigest() []byte {
	if x != nil {
		return x.ReleaseDigest
	}
	return nil
}

func (x *PackageSelection) GetProjectWheel() *WheelFact {
	if x != nil {
		return x.ProjectWheel
	}
	return nil
}

// Local same-user source execution is not a partially installed package. The source digest is
// the daemon-measured checkout identity; its path is a launch-local fact and never enters the
// protocol. A development Placement omits the published-only environment field.
type DevelopmentPackage struct {
	state         protoimpl.MessageState `protogen:"open.v1"`
	Package       string                 `protobuf:"bytes,1,opt,name=package,proto3" json:"package,omitempty"`                               // exact org/name package identity
	Release       string                 `protobuf:"bytes,2,opt,name=release,proto3" json:"release,omitempty"`                               // exact semantic release
	SourceDigest  []byte                 `protobuf:"bytes,3,opt,name=source_digest,json=sourceDigest,proto3" json:"source_digest,omitempty"` // sha256 of the daemon's exact source snapshot
	unknownFields protoimpl.UnknownFields
	sizeCache     protoimpl.SizeCache
}

func (x *DevelopmentPackage) Reset() {
	*x = DevelopmentPackage{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[28]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *DevelopmentPackage) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*DevelopmentPackage) ProtoMessage() {}

func (x *DevelopmentPackage) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[28]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use DevelopmentPackage.ProtoReflect.Descriptor instead.
func (*DevelopmentPackage) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{28}
}

func (x *DevelopmentPackage) GetPackage() string {
	if x != nil {
		return x.Package
	}
	return ""
}

func (x *DevelopmentPackage) GetRelease() string {
	if x != nil {
		return x.Release
	}
	return ""
}

func (x *DevelopmentPackage) GetSourceDigest() []byte {
	if x != nil {
		return x.SourceDigest
	}
	return nil
}

type Environment struct {
	state         protoimpl.MessageState `protogen:"open.v1"`
	Wheels        []*WheelFact           `protobuf:"bytes,2,rep,name=wheels,proto3" json:"wheels,omitempty"` // sorted unique by normalized distribution; [] is normal
	unknownFields protoimpl.UnknownFields
	sizeCache     protoimpl.SizeCache
}

func (x *Environment) Reset() {
	*x = Environment{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[29]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *Environment) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*Environment) ProtoMessage() {}

func (x *Environment) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[29]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use Environment.ProtoReflect.Descriptor instead.
func (*Environment) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{29}
}

func (x *Environment) GetWheels() []*WheelFact {
	if x != nil {
		return x.Wheels
	}
	return nil
}

type Model struct {
	state         protoimpl.MessageState `protogen:"open.v1"`
	Id            string                 `protobuf:"bytes,1,opt,name=id,proto3" json:"id,omitempty"`             // placement-local stable id
	Repo          string                 `protobuf:"bytes,2,opt,name=repo,proto3" json:"repo,omitempty"`         // exact org/name repository identity
	Version       string                 `protobuf:"bytes,3,opt,name=version,proto3" json:"version,omitempty"`   // exact model release
	Lane          string                 `protobuf:"bytes,4,opt,name=lane,proto3" json:"lane,omitempty"`         // exact selected lane
	Manifest      *Ref                   `protobuf:"bytes,5,opt,name=manifest,proto3" json:"manifest,omitempty"` // exact TensorFS Manifest; no redundant header reference
	unknownFields protoimpl.UnknownFields
	sizeCache     protoimpl.SizeCache
}

func (x *Model) Reset() {
	*x = Model{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[30]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *Model) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*Model) ProtoMessage() {}

func (x *Model) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[30]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use Model.ProtoReflect.Descriptor instead.
func (*Model) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{30}
}

func (x *Model) GetId() string {
	if x != nil {
		return x.Id
	}
	return ""
}

func (x *Model) GetRepo() string {
	if x != nil {
		return x.Repo
	}
	return ""
}

func (x *Model) GetVersion() string {
	if x != nil {
		return x.Version
	}
	return ""
}

func (x *Model) GetLane() string {
	if x != nil {
		return x.Lane
	}
	return ""
}

func (x *Model) GetManifest() *Ref {
	if x != nil {
		return x.Manifest
	}
	return nil
}

type Entrypoint struct {
	state                   protoimpl.MessageState `protogen:"open.v1"`
	Name                    string                 `protobuf:"bytes,1,opt,name=name,proto3" json:"name,omitempty"`
	EntrypointBindingDigest []byte                 `protobuf:"bytes,2,opt,name=entrypoint_binding_digest,json=entrypointBindingDigest,proto3" json:"entrypoint_binding_digest,omitempty"` // sha256(canonical JSON of exactly
	// {name:<field 1>,slots:<field 3>})
	Slots         []*Slot `protobuf:"bytes,3,rep,name=slots,proto3" json:"slots,omitempty"` // sorted unique by slot; [] for a weightless entrypoint
	unknownFields protoimpl.UnknownFields
	sizeCache     protoimpl.SizeCache
}

func (x *Entrypoint) Reset() {
	*x = Entrypoint{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[31]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *Entrypoint) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*Entrypoint) ProtoMessage() {}

func (x *Entrypoint) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[31]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use Entrypoint.ProtoReflect.Descriptor instead.
func (*Entrypoint) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{31}
}

func (x *Entrypoint) GetName() string {
	if x != nil {
		return x.Name
	}
	return ""
}

func (x *Entrypoint) GetEntrypointBindingDigest() []byte {
	if x != nil {
		return x.EntrypointBindingDigest
	}
	return nil
}

func (x *Entrypoint) GetSlots() []*Slot {
	if x != nil {
		return x.Slots
	}
	return nil
}

type Slot struct {
	state                     protoimpl.MessageState `protogen:"open.v1"`
	Slot                      string                 `protobuf:"bytes,1,opt,name=slot,proto3" json:"slot,omitempty"`
	ReferenceModelId          string                 `protobuf:"bytes,2,opt,name=reference_model_id,json=referenceModelId,proto3" json:"reference_model_id,omitempty"` // must name one Placement.models id
	Config                    *Config                `protobuf:"bytes,3,opt,name=config,proto3" json:"config,omitempty"`
	Components                []*Component           `protobuf:"bytes,4,rep,name=components,proto3" json:"components,omitempty"`                                                                  // ORDERED MCC destination sequence; never sorted
	ModelConstructionContract *Ref                   `protobuf:"bytes,5,opt,name=model_construction_contract,json=modelConstructionContract,proto3" json:"model_construction_contract,omitempty"` // required for qualified publication; omitted when
	// DevelopmentPackage derives it from source + local model
	Stamps        []*Stamp `protobuf:"bytes,6,rep,name=stamps,proto3" json:"stamps,omitempty"` // sorted unique by (component,key)
	unknownFields protoimpl.UnknownFields
	sizeCache     protoimpl.SizeCache
}

func (x *Slot) Reset() {
	*x = Slot{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[32]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *Slot) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*Slot) ProtoMessage() {}

func (x *Slot) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[32]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use Slot.ProtoReflect.Descriptor instead.
func (*Slot) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{32}
}

func (x *Slot) GetSlot() string {
	if x != nil {
		return x.Slot
	}
	return ""
}

func (x *Slot) GetReferenceModelId() string {
	if x != nil {
		return x.ReferenceModelId
	}
	return ""
}

func (x *Slot) GetConfig() *Config {
	if x != nil {
		return x.Config
	}
	return nil
}

func (x *Slot) GetComponents() []*Component {
	if x != nil {
		return x.Components
	}
	return nil
}

func (x *Slot) GetModelConstructionContract() *Ref {
	if x != nil {
		return x.ModelConstructionContract
	}
	return nil
}

func (x *Slot) GetStamps() []*Stamp {
	if x != nil {
		return x.Stamps
	}
	return nil
}

type Config struct {
	state         protoimpl.MessageState `protogen:"open.v1"`
	Document      *Ref                   `protobuf:"bytes,1,opt,name=document,proto3" json:"document,omitempty"`
	Assets        []*Asset               `protobuf:"bytes,2,rep,name=assets,proto3" json:"assets,omitempty"` // sorted unique by name
	unknownFields protoimpl.UnknownFields
	sizeCache     protoimpl.SizeCache
}

func (x *Config) Reset() {
	*x = Config{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[33]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *Config) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*Config) ProtoMessage() {}

func (x *Config) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[33]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use Config.ProtoReflect.Descriptor instead.
func (*Config) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{33}
}

func (x *Config) GetDocument() *Ref {
	if x != nil {
		return x.Document
	}
	return nil
}

func (x *Config) GetAssets() []*Asset {
	if x != nil {
		return x.Assets
	}
	return nil
}

type Asset struct {
	state         protoimpl.MessageState `protogen:"open.v1"`
	Name          string                 `protobuf:"bytes,1,opt,name=name,proto3" json:"name,omitempty"`
	ModelId       string                 `protobuf:"bytes,2,opt,name=model_id,json=modelId,proto3" json:"model_id,omitempty"` // required; must name one Placement.models id
	Ref           *Ref                   `protobuf:"bytes,3,opt,name=ref,proto3" json:"ref,omitempty"`
	unknownFields protoimpl.UnknownFields
	sizeCache     protoimpl.SizeCache
}

func (x *Asset) Reset() {
	*x = Asset{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[34]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *Asset) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*Asset) ProtoMessage() {}

func (x *Asset) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[34]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use Asset.ProtoReflect.Descriptor instead.
func (*Asset) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{34}
}

func (x *Asset) GetName() string {
	if x != nil {
		return x.Name
	}
	return ""
}

func (x *Asset) GetModelId() string {
	if x != nil {
		return x.ModelId
	}
	return ""
}

func (x *Asset) GetRef() *Ref {
	if x != nil {
		return x.Ref
	}
	return nil
}

type Component struct {
	state         protoimpl.MessageState `protogen:"open.v1"`
	Component     string                 `protobuf:"bytes,1,opt,name=component,proto3" json:"component,omitempty"`
	ModelId       string                 `protobuf:"bytes,2,opt,name=model_id,json=modelId,proto3" json:"model_id,omitempty"` // must name one Placement.models id
	unknownFields protoimpl.UnknownFields
	sizeCache     protoimpl.SizeCache
}

func (x *Component) Reset() {
	*x = Component{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[35]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *Component) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*Component) ProtoMessage() {}

func (x *Component) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[35]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use Component.ProtoReflect.Descriptor instead.
func (*Component) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{35}
}

func (x *Component) GetComponent() string {
	if x != nil {
		return x.Component
	}
	return ""
}

func (x *Component) GetModelId() string {
	if x != nil {
		return x.ModelId
	}
	return ""
}

type Stamp struct {
	state         protoimpl.MessageState `protogen:"open.v1"`
	Component     string                 `protobuf:"bytes,1,opt,name=component,proto3" json:"component,omitempty"` // must name one Slot.components component
	Key           string                 `protobuf:"bytes,2,opt,name=key,proto3" json:"key,omitempty"`
	Values        []string               `protobuf:"bytes,3,rep,name=values,proto3" json:"values,omitempty"` // sorted unique, nonempty
	unknownFields protoimpl.UnknownFields
	sizeCache     protoimpl.SizeCache
}

func (x *Stamp) Reset() {
	*x = Stamp{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[36]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *Stamp) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*Stamp) ProtoMessage() {}

func (x *Stamp) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[36]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use Stamp.ProtoReflect.Descriptor instead.
func (*Stamp) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{36}
}

func (x *Stamp) GetComponent() string {
	if x != nil {
		return x.Component
	}
	return ""
}

func (x *Stamp) GetKey() string {
	if x != nil {
		return x.Key
	}
	return ""
}

func (x *Stamp) GetValues() []string {
	if x != nil {
		return x.Values
	}
	return nil
}

type JobDirective struct {
	state               protoimpl.MessageState `protogen:"open.v1"`
	BuildId             string                 `protobuf:"bytes,1,opt,name=build_id,json=buildId,proto3" json:"build_id,omitempty"`
	JobDescriptorId     string                 `protobuf:"bytes,2,opt,name=job_descriptor_id,json=jobDescriptorId,proto3" json:"job_descriptor_id,omitempty"`
	ResourceCaps        *ResourceCaps          `protobuf:"bytes,3,opt,name=resource_caps,json=resourceCaps,proto3" json:"resource_caps,omitempty"`
	PublicationContract *PublicationContract   `protobuf:"bytes,4,opt,name=publication_contract,json=publicationContract,proto3" json:"publication_contract,omitempty"`
	ReclaimOnTerminal   bool                   `protobuf:"varint,5,opt,name=reclaim_on_terminal,json=reclaimOnTerminal,proto3" json:"reclaim_on_terminal,omitempty"`
	DeviceCount         uint32                 `protobuf:"varint,6,opt,name=device_count,json=deviceCount,proto3" json:"device_count,omitempty"`
	unknownFields       protoimpl.UnknownFields
	sizeCache           protoimpl.SizeCache
}

func (x *JobDirective) Reset() {
	*x = JobDirective{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[37]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *JobDirective) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*JobDirective) ProtoMessage() {}

func (x *JobDirective) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[37]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use JobDirective.ProtoReflect.Descriptor instead.
func (*JobDirective) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{37}
}

func (x *JobDirective) GetBuildId() string {
	if x != nil {
		return x.BuildId
	}
	return ""
}

func (x *JobDirective) GetJobDescriptorId() string {
	if x != nil {
		return x.JobDescriptorId
	}
	return ""
}

func (x *JobDirective) GetResourceCaps() *ResourceCaps {
	if x != nil {
		return x.ResourceCaps
	}
	return nil
}

func (x *JobDirective) GetPublicationContract() *PublicationContract {
	if x != nil {
		return x.PublicationContract
	}
	return nil
}

func (x *JobDirective) GetReclaimOnTerminal() bool {
	if x != nil {
		return x.ReclaimOnTerminal
	}
	return false
}

func (x *JobDirective) GetDeviceCount() uint32 {
	if x != nil {
		return x.DeviceCount
	}
	return 0
}

// CONVERGENCE FACTS ON THE WIRE (§8; was Report). `applied_revision` is RETIRED as dishonest —
// it advanced on ACCEPTANCE, so a RecordOwner reading it learned only that its own message
// arrived. Acceptance and convergence are now two separate, checkable facts:
// converged_revision < accepted_desired_state_revision is the normal, visible state of a
// convergence in progress or a latched failure — never an error, always readable.
type ObservedWorkerState struct {
	state                   protoimpl.MessageState `protogen:"open.v1"`
	RecordOwnerEpoch        uint64                 `protobuf:"varint,1,opt,name=record_owner_epoch,json=recordOwnerEpoch,proto3" json:"record_owner_epoch,omitempty"`
	ControlStreamGeneration uint64                 `protobuf:"varint,2,opt,name=control_stream_generation,json=controlStreamGeneration,proto3" json:"control_stream_generation,omitempty"`
	WorkerBootId            string                 `protobuf:"bytes,3,opt,name=worker_boot_id,json=workerBootId,proto3" json:"worker_boot_id,omitempty"`
	AppliedWireMinor        uint32                 `protobuf:"varint,8,opt,name=applied_wire_minor,json=appliedWireMinor,proto3" json:"applied_wire_minor,omitempty"` // stale echo is a visible skew fact, never a refusal
	HeldAttempts            []*HeldAttempt         `protobuf:"bytes,9,rep,name=held_attempts,json=heldAttempts,proto3" json:"held_attempts,omitempty"`                // running attempts AND outcomes pending ack
	Faults                  []*Fault               `protobuf:"bytes,10,rep,name=faults,proto3" json:"faults,omitempty"`                                               // MACHINE-scope faults; placement faults ride their status
	Activity                []*ActivityEvent       `protobuf:"bytes,11,rep,name=activity,proto3" json:"activity,omitempty"`                                           // the non-attempt activity lane (durable)
	JobCapacity             *JobCapacity           `protobuf:"bytes,13,opt,name=job_capacity,json=jobCapacity,proto3" json:"job_capacity,omitempty"`                  // job mode only
	Placements              []*PlacementStatus     `protobuf:"bytes,15,rep,name=placements,proto3" json:"placements,omitempty"`                                       // serving mode only; sorted by placement_id
	// The ONE worker-level admission fence (§6). The RecordOwner dispatches against the generation
	// it last saw; the worker admits only if the echoed generation is CURRENT, admission_state is
	// OPEN, and a slot is free. A stale generation refuses DETERMINISTICALLY — same input, same
	// verdict, no race window.
	AdmissionGeneration uint64 `protobuf:"varint,16,opt,name=admission_generation,json=admissionGeneration,proto3" json:"admission_generation,omitempty"` // bumps whenever admission MEANING changes (executor
	// respawn, window resize, phase change, cutover)
	AdmissionState AdmissionState `protobuf:"varint,17,opt,name=admission_state,json=admissionState,proto3,enum=cozy.worker.v1.AdmissionState" json:"admission_state,omitempty"` // CLOSED is STRUCTURAL (pre-snapshot-barrier,
	// draining, mid-cutover); OPEN with zero slots is TRANSIENT
	// saturation — a RecordOwner backs off differently for each
	AvailableAttemptSlots        uint32 `protobuf:"varint,18,opt,name=available_attempt_slots,json=availableAttemptSlots,proto3" json:"available_attempt_slots,omitempty"`                        // free seats in the ONE shared acceptance window,
	AcceptedDesiredStateRevision uint64 `protobuf:"varint,19,opt,name=accepted_desired_state_revision,json=acceptedDesiredStateRevision,proto3" json:"accepted_desired_state_revision,omitempty"` // durably ACCEPTED intent (advances on
	// acceptance — which is all it ever honestly meant)
	AcceptedPlacementSetDigest []byte `protobuf:"bytes,20,opt,name=accepted_placement_set_digest,json=acceptedPlacementSetDigest,proto3" json:"accepted_placement_set_digest,omitempty"` // class (a): the accepted set's digest
	ConvergedRevision          uint64 `protobuf:"varint,21,opt,name=converged_revision,json=convergedRevision,proto3" json:"converged_revision,omitempty"`                               // advances ONLY when observed satisfies accepted. It may
	// NEVER advance past a placement whose serving state is not
	// DISPATCHABLE for a desired spec: the field is a derived
	// fact, and a worker that advances it while observed state
	// disagrees is in breach, not merely optimistic.
	WorkerPhase   WorkerPhase `protobuf:"varint,22,opt,name=worker_phase,json=workerPhase,proto3,enum=cozy.worker.v1.WorkerPhase" json:"worker_phase,omitempty"` // machine lifecycle, out of the placement enum
	unknownFields protoimpl.UnknownFields
	sizeCache     protoimpl.SizeCache
}

func (x *ObservedWorkerState) Reset() {
	*x = ObservedWorkerState{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[38]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *ObservedWorkerState) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*ObservedWorkerState) ProtoMessage() {}

func (x *ObservedWorkerState) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[38]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use ObservedWorkerState.ProtoReflect.Descriptor instead.
func (*ObservedWorkerState) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{38}
}

func (x *ObservedWorkerState) GetRecordOwnerEpoch() uint64 {
	if x != nil {
		return x.RecordOwnerEpoch
	}
	return 0
}

func (x *ObservedWorkerState) GetControlStreamGeneration() uint64 {
	if x != nil {
		return x.ControlStreamGeneration
	}
	return 0
}

func (x *ObservedWorkerState) GetWorkerBootId() string {
	if x != nil {
		return x.WorkerBootId
	}
	return ""
}

func (x *ObservedWorkerState) GetAppliedWireMinor() uint32 {
	if x != nil {
		return x.AppliedWireMinor
	}
	return 0
}

func (x *ObservedWorkerState) GetHeldAttempts() []*HeldAttempt {
	if x != nil {
		return x.HeldAttempts
	}
	return nil
}

func (x *ObservedWorkerState) GetFaults() []*Fault {
	if x != nil {
		return x.Faults
	}
	return nil
}

func (x *ObservedWorkerState) GetActivity() []*ActivityEvent {
	if x != nil {
		return x.Activity
	}
	return nil
}

func (x *ObservedWorkerState) GetJobCapacity() *JobCapacity {
	if x != nil {
		return x.JobCapacity
	}
	return nil
}

func (x *ObservedWorkerState) GetPlacements() []*PlacementStatus {
	if x != nil {
		return x.Placements
	}
	return nil
}

func (x *ObservedWorkerState) GetAdmissionGeneration() uint64 {
	if x != nil {
		return x.AdmissionGeneration
	}
	return 0
}

func (x *ObservedWorkerState) GetAdmissionState() AdmissionState {
	if x != nil {
		return x.AdmissionState
	}
	return AdmissionState_ADMISSION_STATE_UNSPECIFIED
}

func (x *ObservedWorkerState) GetAvailableAttemptSlots() uint32 {
	if x != nil {
		return x.AvailableAttemptSlots
	}
	return 0
}

func (x *ObservedWorkerState) GetAcceptedDesiredStateRevision() uint64 {
	if x != nil {
		return x.AcceptedDesiredStateRevision
	}
	return 0
}

func (x *ObservedWorkerState) GetAcceptedPlacementSetDigest() []byte {
	if x != nil {
		return x.AcceptedPlacementSetDigest
	}
	return nil
}

func (x *ObservedWorkerState) GetConvergedRevision() uint64 {
	if x != nil {
		return x.ConvergedRevision
	}
	return 0
}

func (x *ObservedWorkerState) GetWorkerPhase() WorkerPhase {
	if x != nil {
		return x.WorkerPhase
	}
	return WorkerPhase_WORKER_PHASE_UNSPECIFIED
}

// PER-PLACEMENT TRUTH, on TWO AXES (§8, #473/#482). The frozen wire's single IntakeState cannot
// express "staged on disk but offline" — the exact state an outgoing spec holds under
// fallback-retention — and a single phase cannot represent staged x draining independently.
//
// The axes make #474's fallback-retention expressible: outgoing sits at (STAGED, OFFLINE),
// restorable without a refetch, while incoming sits at (STAGED, ACTIVATING). An activation
// failure restores outgoing to (STAGED, DISPATCHABLE) and LATCHES the failure for that desired
// revision — the worker does not evict-and-retry forever, and converged_revision stays behind
// with the fault typed on the placement.
//
// REPLACEMENT REQUIRES A LIVE FALLBACK (#485c): a placement whose rollback pin is gone — GC'd
// under a bytes/age cap, or never materialized — CANNOT accept a new spec until it
// re-materializes one. Fallback capability is a PRECONDITION of replacement, not a best-effort
// nicety; a cap that evicts the pin thereby PAUSES replacement, recorded as
// FAULT_KIND_FALLBACK_PIN_MISSING, and forcing past it is an explicit administrative act.
type PlacementStatus struct {
	state                        protoimpl.MessageState    `protogen:"open.v1"`
	PlacementId                  string                    `protobuf:"bytes,1,opt,name=placement_id,json=placementId,proto3" json:"placement_id,omitempty"`
	ExecutorGeneration           uint64                    `protobuf:"varint,4,opt,name=executor_generation,json=executorGeneration,proto3" json:"executor_generation,omitempty"` // THIS placement's executor fence; bumps on its respawn
	DispatchableBindingDigests   [][]byte                  `protobuf:"bytes,6,rep,name=dispatchable_binding_digests,json=dispatchableBindingDigests,proto3" json:"dispatchable_binding_digests,omitempty"`
	MaterializableBindingDigests [][]byte                  `protobuf:"bytes,7,rep,name=materializable_binding_digests,json=materializableBindingDigests,proto3" json:"materializable_binding_digests,omitempty"` // DISJOINT from dispatchable
	Faults                       []*Fault                  `protobuf:"bytes,8,rep,name=faults,proto3" json:"faults,omitempty"`                                                                                   // placement-scoped
	Accelerator                  *AcceleratorQualification `protobuf:"bytes,9,opt,name=accelerator,proto3" json:"accelerator,omitempty"`                                                                         // absent = present-but-unqualified
	PlacementSetDigest           []byte                    `protobuf:"bytes,11,opt,name=placement_set_digest,json=placementSetDigest,proto3" json:"placement_set_digest,omitempty"`                              // class (a): parent desired-set identity. Together with
	// placement_id, identifies what this placement holds.
	Materialization MaterializationState `protobuf:"varint,12,opt,name=materialization,proto3,enum=cozy.worker.v1.MaterializationState" json:"materialization,omitempty"` // STAGED asserts the
	// exact overlay installed and imported successfully; it is
	// unspeakable otherwise
	Serving                            ServingState `protobuf:"varint,13,opt,name=serving,proto3,enum=cozy.worker.v1.ServingState" json:"serving,omitempty"`
	RetainedFallbackPlacementSetDigest []byte       `protobuf:"bytes,14,opt,name=retained_fallback_placement_set_digest,json=retainedFallbackPlacementSetDigest,proto3" json:"retained_fallback_placement_set_digest,omitempty"` // class (a): predecessor set kept for restore
	// (#474/#485c). Empty means replacement is PAUSED.
	Acquisition *PlacementAcquisitionObservation `protobuf:"bytes,15,opt,name=acquisition,proto3" json:"acquisition,omitempty"` // OBSERVATION ONLY for this
	// placement_set_digest. Never identity, readiness,
	// convergence, or admission authority; no acceptance gate
	// may read it. Terminal values remain visible after
	// STAGED/DISPATCHABLE until the next acquisition for this
	// placement starts or the placement leaves observed state.
	PackageRevisionDigest string `protobuf:"bytes,16,opt,name=package_revision_digest,json=packageRevisionDigest,proto3" json:"package_revision_digest,omitempty"` // exact InvocationSpec package_revision_digest
	EnvironmentDigest     string `protobuf:"bytes,17,opt,name=environment_digest,json=environmentDigest,proto3" json:"environment_digest,omitempty"`               // exact InvocationSpec environment_digest
	ConfigDigest          string `protobuf:"bytes,18,opt,name=config_digest,json=configDigest,proto3" json:"config_digest,omitempty"`                              // exact InvocationSpec config_digest; empty config is
	unknownFields         protoimpl.UnknownFields
	sizeCache             protoimpl.SizeCache
}

func (x *PlacementStatus) Reset() {
	*x = PlacementStatus{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[39]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *PlacementStatus) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*PlacementStatus) ProtoMessage() {}

func (x *PlacementStatus) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[39]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use PlacementStatus.ProtoReflect.Descriptor instead.
func (*PlacementStatus) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{39}
}

func (x *PlacementStatus) GetPlacementId() string {
	if x != nil {
		return x.PlacementId
	}
	return ""
}

func (x *PlacementStatus) GetExecutorGeneration() uint64 {
	if x != nil {
		return x.ExecutorGeneration
	}
	return 0
}

func (x *PlacementStatus) GetDispatchableBindingDigests() [][]byte {
	if x != nil {
		return x.DispatchableBindingDigests
	}
	return nil
}

func (x *PlacementStatus) GetMaterializableBindingDigests() [][]byte {
	if x != nil {
		return x.MaterializableBindingDigests
	}
	return nil
}

func (x *PlacementStatus) GetFaults() []*Fault {
	if x != nil {
		return x.Faults
	}
	return nil
}

func (x *PlacementStatus) GetAccelerator() *AcceleratorQualification {
	if x != nil {
		return x.Accelerator
	}
	return nil
}

func (x *PlacementStatus) GetPlacementSetDigest() []byte {
	if x != nil {
		return x.PlacementSetDigest
	}
	return nil
}

func (x *PlacementStatus) GetMaterialization() MaterializationState {
	if x != nil {
		return x.Materialization
	}
	return MaterializationState_MATERIALIZATION_STATE_UNSPECIFIED
}

func (x *PlacementStatus) GetServing() ServingState {
	if x != nil {
		return x.Serving
	}
	return ServingState_SERVING_STATE_UNSPECIFIED
}

func (x *PlacementStatus) GetRetainedFallbackPlacementSetDigest() []byte {
	if x != nil {
		return x.RetainedFallbackPlacementSetDigest
	}
	return nil
}

func (x *PlacementStatus) GetAcquisition() *PlacementAcquisitionObservation {
	if x != nil {
		return x.Acquisition
	}
	return nil
}

func (x *PlacementStatus) GetPackageRevisionDigest() string {
	if x != nil {
		return x.PackageRevisionDigest
	}
	return ""
}

func (x *PlacementStatus) GetEnvironmentDigest() string {
	if x != nil {
		return x.EnvironmentDigest
	}
	return ""
}

func (x *PlacementStatus) GetConfigDigest() string {
	if x != nil {
		return x.ConfigDigest
	}
	return ""
}

// Cozy-visible proof of concurrent package/model acquisition and cache reuse. Both legs use
// one process-relative monotonic clock and therefore one origin; they reset with the worker
// process and MUST NOT be compared across worker_boot_id. A leg is absent before it starts. Once
// present its start is non-zero, counters are cumulative and never decrease, and end is zero
// while in progress or >= start when terminal. The two terminal intervals prove overlap exactly.
//
// Scope is deliberately disjoint. `package` counts the desired package overlay bytes needed
// above the base worker image (including its project/custom wheels), excluding model bytes.
// `model` counts the object bytes reached from Placement.models[].manifest. At terminal,
// downloaded_bytes + reused_bytes is the exact reached
// content length for that leg. Downloaded bytes were newly fetched into verified local holdings;
// reused bytes were already verified and held locally. Partial observations carry progress so
// far; terminal counters persist as described on PlacementStatus.
type PlacementAcquisitionObservation struct {
	state         protoimpl.MessageState     `protogen:"open.v1"`
	Package       *AcquisitionLegObservation `protobuf:"bytes,1,opt,name=package,proto3" json:"package,omitempty"`
	Model         *AcquisitionLegObservation `protobuf:"bytes,2,opt,name=model,proto3" json:"model,omitempty"`
	unknownFields protoimpl.UnknownFields
	sizeCache     protoimpl.SizeCache
}

func (x *PlacementAcquisitionObservation) Reset() {
	*x = PlacementAcquisitionObservation{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[40]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *PlacementAcquisitionObservation) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*PlacementAcquisitionObservation) ProtoMessage() {}

func (x *PlacementAcquisitionObservation) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[40]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use PlacementAcquisitionObservation.ProtoReflect.Descriptor instead.
func (*PlacementAcquisitionObservation) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{40}
}

func (x *PlacementAcquisitionObservation) GetPackage() *AcquisitionLegObservation {
	if x != nil {
		return x.Package
	}
	return nil
}

func (x *PlacementAcquisitionObservation) GetModel() *AcquisitionLegObservation {
	if x != nil {
		return x.Model
	}
	return nil
}

type AcquisitionLegObservation struct {
	state              protoimpl.MessageState `protogen:"open.v1"`
	StartedMonotonicNs uint64                 `protobuf:"varint,1,opt,name=started_monotonic_ns,json=startedMonotonicNs,proto3" json:"started_monotonic_ns,omitempty"` // non-zero; one process-relative clock for both legs
	EndedMonotonicNs   uint64                 `protobuf:"varint,2,opt,name=ended_monotonic_ns,json=endedMonotonicNs,proto3" json:"ended_monotonic_ns,omitempty"`       // 0 while in progress; terminal iff >= start
	DownloadedBytes    uint64                 `protobuf:"varint,3,opt,name=downloaded_bytes,json=downloadedBytes,proto3" json:"downloaded_bytes,omitempty"`            // cumulative unique content bytes newly fetched this leg
	ReusedBytes        uint64                 `protobuf:"varint,4,opt,name=reused_bytes,json=reusedBytes,proto3" json:"reused_bytes,omitempty"`                        // cumulative verified bytes already held before this leg
	unknownFields      protoimpl.UnknownFields
	sizeCache          protoimpl.SizeCache
}

func (x *AcquisitionLegObservation) Reset() {
	*x = AcquisitionLegObservation{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[41]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *AcquisitionLegObservation) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*AcquisitionLegObservation) ProtoMessage() {}

func (x *AcquisitionLegObservation) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[41]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use AcquisitionLegObservation.ProtoReflect.Descriptor instead.
func (*AcquisitionLegObservation) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{41}
}

func (x *AcquisitionLegObservation) GetStartedMonotonicNs() uint64 {
	if x != nil {
		return x.StartedMonotonicNs
	}
	return 0
}

func (x *AcquisitionLegObservation) GetEndedMonotonicNs() uint64 {
	if x != nil {
		return x.EndedMonotonicNs
	}
	return 0
}

func (x *AcquisitionLegObservation) GetDownloadedBytes() uint64 {
	if x != nil {
		return x.DownloadedBytes
	}
	return 0
}

func (x *AcquisitionLegObservation) GetReusedBytes() uint64 {
	if x != nil {
		return x.ReusedBytes
	}
	return 0
}

// Executor-probed capability facts (#430: MEASURED, never inferred from chip generation or OS
// version). Vocabulary IS the AcceleratorFacts record's (#422) — one facts schema, no second
// wire vocabulary. The torch-free worker NEVER fabricates these (it cannot measure them); until
// an executor qualifies the device, the accelerator is PRESENT-BUT-UNQUALIFIED and this message
// is absent from PlacementStatus.
type AcceleratorQualification struct {
	state        protoimpl.MessageState `protogen:"open.v1"`
	Qualified    bool                   `protobuf:"varint,1,opt,name=qualified,proto3" json:"qualified,omitempty"`
	Capabilities []string               `protobuf:"bytes,2,rep,name=capabilities,proto3" json:"capabilities,omitempty"` // sorted facts vocabulary (bf16, fp8_compute,
	// pinned_h2d_streams, non_blocking_safe,
	// attention_upcast_required); presence = measured true
	RecommendedMaxBytes uint64   `protobuf:"varint,3,opt,name=recommended_max_bytes,json=recommendedMaxBytes,proto3" json:"recommended_max_bytes,omitempty"` // the working-set envelope the ledger prices against
	PinnedCapacityBytes uint64   `protobuf:"varint,4,opt,name=pinned_capacity_bytes,json=pinnedCapacityBytes,proto3" json:"pinned_capacity_bytes,omitempty"`
	H2DGbpsMeasured     uint32   `protobuf:"varint,5,opt,name=h2d_gbps_measured,json=h2dGbpsMeasured,proto3" json:"h2d_gbps_measured,omitempty"`
	D2HGbpsMeasured     uint32   `protobuf:"varint,6,opt,name=d2h_gbps_measured,json=d2hGbpsMeasured,proto3" json:"d2h_gbps_measured,omitempty"`
	TorchVersion        string   `protobuf:"bytes,7,opt,name=torch_version,json=torchVersion,proto3" json:"torch_version,omitempty"` // TELEMETRY ONLY
	Unreadable          []string `protobuf:"bytes,8,rep,name=unreadable,proto3" json:"unreadable,omitempty"`                         // tri-state honesty
	unknownFields       protoimpl.UnknownFields
	sizeCache           protoimpl.SizeCache
}

func (x *AcceleratorQualification) Reset() {
	*x = AcceleratorQualification{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[42]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *AcceleratorQualification) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*AcceleratorQualification) ProtoMessage() {}

func (x *AcceleratorQualification) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[42]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use AcceleratorQualification.ProtoReflect.Descriptor instead.
func (*AcceleratorQualification) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{42}
}

func (x *AcceleratorQualification) GetQualified() bool {
	if x != nil {
		return x.Qualified
	}
	return false
}

func (x *AcceleratorQualification) GetCapabilities() []string {
	if x != nil {
		return x.Capabilities
	}
	return nil
}

func (x *AcceleratorQualification) GetRecommendedMaxBytes() uint64 {
	if x != nil {
		return x.RecommendedMaxBytes
	}
	return 0
}

func (x *AcceleratorQualification) GetPinnedCapacityBytes() uint64 {
	if x != nil {
		return x.PinnedCapacityBytes
	}
	return 0
}

func (x *AcceleratorQualification) GetH2DGbpsMeasured() uint32 {
	if x != nil {
		return x.H2DGbpsMeasured
	}
	return 0
}

func (x *AcceleratorQualification) GetD2HGbpsMeasured() uint32 {
	if x != nil {
		return x.D2HGbpsMeasured
	}
	return 0
}

func (x *AcceleratorQualification) GetTorchVersion() string {
	if x != nil {
		return x.TorchVersion
	}
	return ""
}

func (x *AcceleratorQualification) GetUnreadable() []string {
	if x != nil {
		return x.Unreadable
	}
	return nil
}

type ActivityEvent struct {
	state         protoimpl.MessageState `protogen:"open.v1"`
	Seq           uint64                 `protobuf:"varint,1,opt,name=seq,proto3" json:"seq,omitempty"`
	Kind          string                 `protobuf:"bytes,2,opt,name=kind,proto3" json:"kind,omitempty"`
	Step          string                 `protobuf:"bytes,3,opt,name=step,proto3" json:"step,omitempty"` // bounded <= 256 bytes
	AtUnixMs      uint64                 `protobuf:"varint,4,opt,name=at_unix_ms,json=atUnixMs,proto3" json:"at_unix_ms,omitempty"`
	unknownFields protoimpl.UnknownFields
	sizeCache     protoimpl.SizeCache
}

func (x *ActivityEvent) Reset() {
	*x = ActivityEvent{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[43]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *ActivityEvent) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*ActivityEvent) ProtoMessage() {}

func (x *ActivityEvent) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[43]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use ActivityEvent.ProtoReflect.Descriptor instead.
func (*ActivityEvent) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{43}
}

func (x *ActivityEvent) GetSeq() uint64 {
	if x != nil {
		return x.Seq
	}
	return 0
}

func (x *ActivityEvent) GetKind() string {
	if x != nil {
		return x.Kind
	}
	return ""
}

func (x *ActivityEvent) GetStep() string {
	if x != nil {
		return x.Step
	}
	return ""
}

func (x *ActivityEvent) GetAtUnixMs() uint64 {
	if x != nil {
		return x.AtUnixMs
	}
	return 0
}

// Was StartAttempt (#481): receiving an OFFER begins VALIDATION, not execution, so refusing one
// is an ordinary outcome rather than an exception. EVERY offer receives either AttemptAccepted or
// a JOURNALED AttemptOutcome(REFUSED) — never silence.
type AttemptOffer struct {
	state                   protoimpl.MessageState `protogen:"open.v1"`
	RecordOwnerEpoch        uint64                 `protobuf:"varint,1,opt,name=record_owner_epoch,json=recordOwnerEpoch,proto3" json:"record_owner_epoch,omitempty"`
	ControlStreamGeneration uint64                 `protobuf:"varint,2,opt,name=control_stream_generation,json=controlStreamGeneration,proto3" json:"control_stream_generation,omitempty"`
	WorkerBootId            string                 `protobuf:"bytes,3,opt,name=worker_boot_id,json=workerBootId,proto3" json:"worker_boot_id,omitempty"`
	RequestId               string                 `protobuf:"bytes,5,opt,name=request_id,json=requestId,proto3" json:"request_id,omitempty"`                 // fence (a)
	AttemptOrdinal          uint64                 `protobuf:"varint,6,opt,name=attempt_ordinal,json=attemptOrdinal,proto3" json:"attempt_ordinal,omitempty"` // fence (b); RecordOwner-bumped only. EVERY offer consumes
	// its ordinal — what a pre-execution refusal does not
	// consume is BUDGET (#480b; 02 §6's "only a proven decline
	// is non-consuming" is retired with the decline).
	InvocationSpecDigest []byte `protobuf:"bytes,7,opt,name=invocation_spec_digest,json=invocationSpecDigest,proto3" json:"invocation_spec_digest,omitempty"` // fence (c), class (a): sha256 over EXACTLY the bytes in 9;
	// the receiver RECOMPUTES and refuses on mismatch
	Grant                        *DeliveryGrant `protobuf:"bytes,8,opt,name=grant,proto3" json:"grant,omitempty"`                                                                                       // expiring, refreshable; OUTSIDE the digest
	InvocationSpecCanonicalBytes []byte         `protobuf:"bytes,9,opt,name=invocation_spec_canonical_bytes,json=invocationSpecCanonicalBytes,proto3" json:"invocation_spec_canonical_bytes,omitempty"` // the InvocationSpec DOCUMENT, canonical JSON
	// bytes; immutable for the attempt; parsed under
	// unknown-field REFUSAL
	PlacementId string `protobuf:"bytes,10,opt,name=placement_id,json=placementId,proto3" json:"placement_id,omitempty"` // the target placement (serving mode; REQUIRED there, empty
	// in job mode); unknown id refuses
	// FAULT_KIND_UNKNOWN_PLACEMENT as a JOURNALED REFUSED
	// outcome
	AdmissionGeneration uint64 `protobuf:"varint,12,opt,name=admission_generation,json=admissionGeneration,proto3" json:"admission_generation,omitempty"` // the generation the RecordOwner OBSERVED when it
	unknownFields       protoimpl.UnknownFields
	sizeCache           protoimpl.SizeCache
}

func (x *AttemptOffer) Reset() {
	*x = AttemptOffer{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[44]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *AttemptOffer) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*AttemptOffer) ProtoMessage() {}

func (x *AttemptOffer) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[44]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use AttemptOffer.ProtoReflect.Descriptor instead.
func (*AttemptOffer) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{44}
}

func (x *AttemptOffer) GetRecordOwnerEpoch() uint64 {
	if x != nil {
		return x.RecordOwnerEpoch
	}
	return 0
}

func (x *AttemptOffer) GetControlStreamGeneration() uint64 {
	if x != nil {
		return x.ControlStreamGeneration
	}
	return 0
}

func (x *AttemptOffer) GetWorkerBootId() string {
	if x != nil {
		return x.WorkerBootId
	}
	return ""
}

func (x *AttemptOffer) GetRequestId() string {
	if x != nil {
		return x.RequestId
	}
	return ""
}

func (x *AttemptOffer) GetAttemptOrdinal() uint64 {
	if x != nil {
		return x.AttemptOrdinal
	}
	return 0
}

func (x *AttemptOffer) GetInvocationSpecDigest() []byte {
	if x != nil {
		return x.InvocationSpecDigest
	}
	return nil
}

func (x *AttemptOffer) GetGrant() *DeliveryGrant {
	if x != nil {
		return x.Grant
	}
	return nil
}

func (x *AttemptOffer) GetInvocationSpecCanonicalBytes() []byte {
	if x != nil {
		return x.InvocationSpecCanonicalBytes
	}
	return nil
}

func (x *AttemptOffer) GetPlacementId() string {
	if x != nil {
		return x.PlacementId
	}
	return ""
}

func (x *AttemptOffer) GetAdmissionGeneration() uint64 {
	if x != nil {
		return x.AdmissionGeneration
	}
	return 0
}

type AttemptAccepted struct {
	state                   protoimpl.MessageState `protogen:"open.v1"`
	RecordOwnerEpoch        uint64                 `protobuf:"varint,1,opt,name=record_owner_epoch,json=recordOwnerEpoch,proto3" json:"record_owner_epoch,omitempty"`
	ControlStreamGeneration uint64                 `protobuf:"varint,2,opt,name=control_stream_generation,json=controlStreamGeneration,proto3" json:"control_stream_generation,omitempty"`
	WorkerBootId            string                 `protobuf:"bytes,3,opt,name=worker_boot_id,json=workerBootId,proto3" json:"worker_boot_id,omitempty"`
	RequestId               string                 `protobuf:"bytes,5,opt,name=request_id,json=requestId,proto3" json:"request_id,omitempty"`
	AttemptOrdinal          uint64                 `protobuf:"varint,6,opt,name=attempt_ordinal,json=attemptOrdinal,proto3" json:"attempt_ordinal,omitempty"`
	InvocationSpecDigest    []byte                 `protobuf:"bytes,7,opt,name=invocation_spec_digest,json=invocationSpecDigest,proto3" json:"invocation_spec_digest,omitempty"`          // echo
	PlanDigest              []byte                 `protobuf:"bytes,8,opt,name=plan_digest,json=planDigest,proto3" json:"plan_digest,omitempty"`                                          // class (a): AttemptPlan (cr-007), fixed at acceptance
	ModelConstructionDigest []byte                 `protobuf:"bytes,9,opt,name=model_construction_digest,json=modelConstructionDigest,proto3" json:"model_construction_digest,omitempty"` // class (a): ModelConstructionContract (cr-004)
	Plan                    *AttemptPlanSummary    `protobuf:"bytes,10,opt,name=plan,proto3" json:"plan,omitempty"`                                                                       // the CLOSED observable projection of the chosen plan
	PlacementId             string                 `protobuf:"bytes,11,opt,name=placement_id,json=placementId,proto3" json:"placement_id,omitempty"`                                      // echo
	ExecutorGeneration      uint64                 `protobuf:"varint,12,opt,name=executor_generation,json=executorGeneration,proto3" json:"executor_generation,omitempty"`                // the placement executor generation the attempt runs under,
	unknownFields           protoimpl.UnknownFields
	sizeCache               protoimpl.SizeCache
}

func (x *AttemptAccepted) Reset() {
	*x = AttemptAccepted{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[45]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *AttemptAccepted) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*AttemptAccepted) ProtoMessage() {}

func (x *AttemptAccepted) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[45]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use AttemptAccepted.ProtoReflect.Descriptor instead.
func (*AttemptAccepted) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{45}
}

func (x *AttemptAccepted) GetRecordOwnerEpoch() uint64 {
	if x != nil {
		return x.RecordOwnerEpoch
	}
	return 0
}

func (x *AttemptAccepted) GetControlStreamGeneration() uint64 {
	if x != nil {
		return x.ControlStreamGeneration
	}
	return 0
}

func (x *AttemptAccepted) GetWorkerBootId() string {
	if x != nil {
		return x.WorkerBootId
	}
	return ""
}

func (x *AttemptAccepted) GetRequestId() string {
	if x != nil {
		return x.RequestId
	}
	return ""
}

func (x *AttemptAccepted) GetAttemptOrdinal() uint64 {
	if x != nil {
		return x.AttemptOrdinal
	}
	return 0
}

func (x *AttemptAccepted) GetInvocationSpecDigest() []byte {
	if x != nil {
		return x.InvocationSpecDigest
	}
	return nil
}

func (x *AttemptAccepted) GetPlanDigest() []byte {
	if x != nil {
		return x.PlanDigest
	}
	return nil
}

func (x *AttemptAccepted) GetModelConstructionDigest() []byte {
	if x != nil {
		return x.ModelConstructionDigest
	}
	return nil
}

func (x *AttemptAccepted) GetPlan() *AttemptPlanSummary {
	if x != nil {
		return x.Plan
	}
	return nil
}

func (x *AttemptAccepted) GetPlacementId() string {
	if x != nil {
		return x.PlacementId
	}
	return ""
}

func (x *AttemptAccepted) GetExecutorGeneration() uint64 {
	if x != nil {
		return x.ExecutorGeneration
	}
	return 0
}

type AttemptPlanSummary struct {
	state           protoimpl.MessageState `protogen:"open.v1"`
	Delivery        string                 `protobuf:"bytes,1,opt,name=delivery,proto3" json:"delivery,omitempty"`               // native | float
	Materialization string                 `protobuf:"bytes,2,opt,name=materialization,proto3" json:"materialization,omitempty"` // aot_decode | staged_decode | jit_decode
	ComputeDtype    string                 `protobuf:"bytes,3,opt,name=compute_dtype,json=computeDtype,proto3" json:"compute_dtype,omitempty"`
	Placement       string                 `protobuf:"bytes,4,opt,name=placement,proto3" json:"placement,omitempty"` // TENSOR RESIDENCY, unrelated to placement_id:
	// all_resident | component_staged | host_offload | streamed
	ReservedDeviceMemoryBytes uint64 `protobuf:"varint,5,opt,name=reserved_device_memory_bytes,json=reservedDeviceMemoryBytes,proto3" json:"reserved_device_memory_bytes,omitempty"`
	ReservedHostBytes         uint64 `protobuf:"varint,6,opt,name=reserved_host_bytes,json=reservedHostBytes,proto3" json:"reserved_host_bytes,omitempty"`
	DecisionExplanation       string `protobuf:"bytes,7,opt,name=decision_explanation,json=decisionExplanation,proto3" json:"decision_explanation,omitempty"` // bounded <= 512 bytes diagnostic prose; not parseable
	unknownFields             protoimpl.UnknownFields
	sizeCache                 protoimpl.SizeCache
}

func (x *AttemptPlanSummary) Reset() {
	*x = AttemptPlanSummary{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[46]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *AttemptPlanSummary) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*AttemptPlanSummary) ProtoMessage() {}

func (x *AttemptPlanSummary) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[46]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use AttemptPlanSummary.ProtoReflect.Descriptor instead.
func (*AttemptPlanSummary) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{46}
}

func (x *AttemptPlanSummary) GetDelivery() string {
	if x != nil {
		return x.Delivery
	}
	return ""
}

func (x *AttemptPlanSummary) GetMaterialization() string {
	if x != nil {
		return x.Materialization
	}
	return ""
}

func (x *AttemptPlanSummary) GetComputeDtype() string {
	if x != nil {
		return x.ComputeDtype
	}
	return ""
}

func (x *AttemptPlanSummary) GetPlacement() string {
	if x != nil {
		return x.Placement
	}
	return ""
}

func (x *AttemptPlanSummary) GetReservedDeviceMemoryBytes() uint64 {
	if x != nil {
		return x.ReservedDeviceMemoryBytes
	}
	return 0
}

func (x *AttemptPlanSummary) GetReservedHostBytes() uint64 {
	if x != nil {
		return x.ReservedHostBytes
	}
	return 0
}

func (x *AttemptPlanSummary) GetDecisionExplanation() string {
	if x != nil {
		return x.DecisionExplanation
	}
	return ""
}

// No placement_id here: the fence triple resolves against the journaled acceptance, which
// records the placement.
type CancelAttempt struct {
	state                   protoimpl.MessageState `protogen:"open.v1"`
	RecordOwnerEpoch        uint64                 `protobuf:"varint,1,opt,name=record_owner_epoch,json=recordOwnerEpoch,proto3" json:"record_owner_epoch,omitempty"`
	ControlStreamGeneration uint64                 `protobuf:"varint,2,opt,name=control_stream_generation,json=controlStreamGeneration,proto3" json:"control_stream_generation,omitempty"`
	WorkerBootId            string                 `protobuf:"bytes,3,opt,name=worker_boot_id,json=workerBootId,proto3" json:"worker_boot_id,omitempty"`
	RequestId               string                 `protobuf:"bytes,5,opt,name=request_id,json=requestId,proto3" json:"request_id,omitempty"`
	AttemptOrdinal          uint64                 `protobuf:"varint,6,opt,name=attempt_ordinal,json=attemptOrdinal,proto3" json:"attempt_ordinal,omitempty"`
	Reason                  CancelReason           `protobuf:"varint,7,opt,name=reason,proto3,enum=cozy.worker.v1.CancelReason" json:"reason,omitempty"`
	GraceMs                 uint64                 `protobuf:"varint,8,opt,name=grace_ms,json=graceMs,proto3" json:"grace_ms,omitempty"`                                         // operator override; 0 = worker default. Never author-set.
	InvocationSpecDigest    []byte                 `protobuf:"bytes,9,opt,name=invocation_spec_digest,json=invocationSpecDigest,proto3" json:"invocation_spec_digest,omitempty"` // full-triple fence; stale/mismatched cancel is DROPPED
	unknownFields           protoimpl.UnknownFields
	sizeCache               protoimpl.SizeCache
}

func (x *CancelAttempt) Reset() {
	*x = CancelAttempt{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[47]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *CancelAttempt) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*CancelAttempt) ProtoMessage() {}

func (x *CancelAttempt) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[47]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use CancelAttempt.ProtoReflect.Descriptor instead.
func (*CancelAttempt) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{47}
}

func (x *CancelAttempt) GetRecordOwnerEpoch() uint64 {
	if x != nil {
		return x.RecordOwnerEpoch
	}
	return 0
}

func (x *CancelAttempt) GetControlStreamGeneration() uint64 {
	if x != nil {
		return x.ControlStreamGeneration
	}
	return 0
}

func (x *CancelAttempt) GetWorkerBootId() string {
	if x != nil {
		return x.WorkerBootId
	}
	return ""
}

func (x *CancelAttempt) GetRequestId() string {
	if x != nil {
		return x.RequestId
	}
	return ""
}

func (x *CancelAttempt) GetAttemptOrdinal() uint64 {
	if x != nil {
		return x.AttemptOrdinal
	}
	return 0
}

func (x *CancelAttempt) GetReason() CancelReason {
	if x != nil {
		return x.Reason
	}
	return CancelReason_CANCEL_REASON_UNSPECIFIED
}

func (x *CancelAttempt) GetGraceMs() uint64 {
	if x != nil {
		return x.GraceMs
	}
	return 0
}

func (x *CancelAttempt) GetInvocationSpecDigest() []byte {
	if x != nil {
		return x.InvocationSpecDigest
	}
	return nil
}

// The envelope around a journaled AttemptOutcomeBody document — the ATTEMPT OUTCOME (was
// AttemptTerminal; "terminal" reads as a device and the thing is an outcome, #481). One
// immutable outcome per attempt; the customer-visible settlement is the RecordOwner's
// RequestClosure and never crosses this wire. Body travels as canonical bytes; routing copies
// must agree with it. journal-before-send applies unchanged, INCLUDING to refusals: the outcome
// is persisted before it is sent and replays until AttemptOutcomeAck.
type AttemptOutcome struct {
	state                   protoimpl.MessageState `protogen:"open.v1"`
	RecordOwnerEpoch        uint64                 `protobuf:"varint,1,opt,name=record_owner_epoch,json=recordOwnerEpoch,proto3" json:"record_owner_epoch,omitempty"`
	ControlStreamGeneration uint64                 `protobuf:"varint,2,opt,name=control_stream_generation,json=controlStreamGeneration,proto3" json:"control_stream_generation,omitempty"`
	WorkerBootId            string                 `protobuf:"bytes,3,opt,name=worker_boot_id,json=workerBootId,proto3" json:"worker_boot_id,omitempty"`
	RequestId               string                 `protobuf:"bytes,5,opt,name=request_id,json=requestId,proto3" json:"request_id,omitempty"`                                        // routing copy; must agree with the document
	AttemptOrdinal          uint64                 `protobuf:"varint,6,opt,name=attempt_ordinal,json=attemptOrdinal,proto3" json:"attempt_ordinal,omitempty"`                        // routing copy
	InvocationSpecDigest    []byte                 `protobuf:"bytes,7,opt,name=invocation_spec_digest,json=invocationSpecDigest,proto3" json:"invocation_spec_digest,omitempty"`     // routing copy
	OutcomeId               string                 `protobuf:"bytes,8,opt,name=outcome_id,json=outcomeId,proto3" json:"outcome_id,omitempty"`                                        // WORKER-minted observation id; stable across replays
	OutcomeDigest           []byte                 `protobuf:"bytes,9,opt,name=outcome_digest,json=outcomeDigest,proto3" json:"outcome_digest,omitempty"`                            // class (a): sha256 over EXACTLY the bytes in 10
	OutcomeCanonicalBytes   []byte                 `protobuf:"bytes,10,opt,name=outcome_canonical_bytes,json=outcomeCanonicalBytes,proto3" json:"outcome_canonical_bytes,omitempty"` // the AttemptOutcomeBody DOCUMENT, canonical JSON bytes
	PlacementId             string                 `protobuf:"bytes,11,opt,name=placement_id,json=placementId,proto3" json:"placement_id,omitempty"`                                 // routing
	unknownFields           protoimpl.UnknownFields
	sizeCache               protoimpl.SizeCache
}

func (x *AttemptOutcome) Reset() {
	*x = AttemptOutcome{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[48]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *AttemptOutcome) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*AttemptOutcome) ProtoMessage() {}

func (x *AttemptOutcome) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[48]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use AttemptOutcome.ProtoReflect.Descriptor instead.
func (*AttemptOutcome) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{48}
}

func (x *AttemptOutcome) GetRecordOwnerEpoch() uint64 {
	if x != nil {
		return x.RecordOwnerEpoch
	}
	return 0
}

func (x *AttemptOutcome) GetControlStreamGeneration() uint64 {
	if x != nil {
		return x.ControlStreamGeneration
	}
	return 0
}

func (x *AttemptOutcome) GetWorkerBootId() string {
	if x != nil {
		return x.WorkerBootId
	}
	return ""
}

func (x *AttemptOutcome) GetRequestId() string {
	if x != nil {
		return x.RequestId
	}
	return ""
}

func (x *AttemptOutcome) GetAttemptOrdinal() uint64 {
	if x != nil {
		return x.AttemptOrdinal
	}
	return 0
}

func (x *AttemptOutcome) GetInvocationSpecDigest() []byte {
	if x != nil {
		return x.InvocationSpecDigest
	}
	return nil
}

func (x *AttemptOutcome) GetOutcomeId() string {
	if x != nil {
		return x.OutcomeId
	}
	return ""
}

func (x *AttemptOutcome) GetOutcomeDigest() []byte {
	if x != nil {
		return x.OutcomeDigest
	}
	return nil
}

func (x *AttemptOutcome) GetOutcomeCanonicalBytes() []byte {
	if x != nil {
		return x.OutcomeCanonicalBytes
	}
	return nil
}

func (x *AttemptOutcome) GetPlacementId() string {
	if x != nil {
		return x.PlacementId
	}
	return ""
}

// DOCUMENT SHAPE (not a wire message): canonical form
// `cozy.worker.v1.AttemptOutcomeBody/1` (#480c/#481/th-049).
type AttemptOutcomeBody struct {
	state                protoimpl.MessageState `protogen:"open.v1"`
	RequestId            string                 `protobuf:"bytes,1,opt,name=request_id,json=requestId,proto3" json:"request_id,omitempty"`
	AttemptOrdinal       uint64                 `protobuf:"varint,2,opt,name=attempt_ordinal,json=attemptOrdinal,proto3" json:"attempt_ordinal,omitempty"`
	InvocationSpecDigest string                 `protobuf:"bytes,3,opt,name=invocation_spec_digest,json=invocationSpecDigest,proto3" json:"invocation_spec_digest,omitempty"` // sha256:<hex> spelling inside the document
	Status               OutcomeStatus          `protobuf:"varint,4,opt,name=status,proto3,enum=cozy.worker.v1.OutcomeStatus" json:"status,omitempty"`                        // closed NEUTRAL vocabulary; retryability is a RecordOwner
	// PROJECTION over (status, cause, origin), unrepresentable
	// as a worker wire fact
	OutputManifest   *OutputManifest  `protobuf:"bytes,5,opt,name=output_manifest,json=outputManifest,proto3" json:"output_manifest,omitempty"`
	Metrics          *AttemptMetrics  `protobuf:"bytes,6,opt,name=metrics,proto3" json:"metrics,omitempty"`
	TriageBundle     *TriageBundleRef `protobuf:"bytes,7,opt,name=triage_bundle,json=triageBundle,proto3" json:"triage_bundle,omitempty"`               // observation only
	SafeMessage      string           `protobuf:"bytes,8,opt,name=safe_message,json=safeMessage,proto3" json:"safe_message,omitempty"`                  // bounded <= 4096 bytes, sanitized
	Cause            *OutcomeCause    `protobuf:"bytes,9,opt,name=cause,proto3" json:"cause,omitempty"`                                                 // REQUIRED
	Result           *ResultEnvelope  `protobuf:"bytes,10,opt,name=result,proto3" json:"result,omitempty"`                                              // the typed function result (success outcomes)
	ExecutionStarted bool             `protobuf:"varint,11,opt,name=execution_started,json=executionStarted,proto3" json:"execution_started,omitempty"` // THE BILLING FACT, structural rather than derived from a
	// cause-code allowlist (#480c): false => the author never
	// ran and zero execution budget was billed, so the
	// RecordOwner may immediately mint attempt_ordinal + 1 and
	// dispatch it elsewhere with no cooldown and no budget draw
	ArtifactReceipts []*ArtifactReceiptRef `protobuf:"bytes,12,rep,name=artifact_receipts,json=artifactReceipts,proto3" json:"artifact_receipts,omitempty"` // committed job outputs only; sorted by slot
	unknownFields    protoimpl.UnknownFields
	sizeCache        protoimpl.SizeCache
}

func (x *AttemptOutcomeBody) Reset() {
	*x = AttemptOutcomeBody{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[49]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *AttemptOutcomeBody) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*AttemptOutcomeBody) ProtoMessage() {}

func (x *AttemptOutcomeBody) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[49]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use AttemptOutcomeBody.ProtoReflect.Descriptor instead.
func (*AttemptOutcomeBody) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{49}
}

func (x *AttemptOutcomeBody) GetRequestId() string {
	if x != nil {
		return x.RequestId
	}
	return ""
}

func (x *AttemptOutcomeBody) GetAttemptOrdinal() uint64 {
	if x != nil {
		return x.AttemptOrdinal
	}
	return 0
}

func (x *AttemptOutcomeBody) GetInvocationSpecDigest() string {
	if x != nil {
		return x.InvocationSpecDigest
	}
	return ""
}

func (x *AttemptOutcomeBody) GetStatus() OutcomeStatus {
	if x != nil {
		return x.Status
	}
	return OutcomeStatus_OUTCOME_STATUS_UNSPECIFIED
}

func (x *AttemptOutcomeBody) GetOutputManifest() *OutputManifest {
	if x != nil {
		return x.OutputManifest
	}
	return nil
}

func (x *AttemptOutcomeBody) GetMetrics() *AttemptMetrics {
	if x != nil {
		return x.Metrics
	}
	return nil
}

func (x *AttemptOutcomeBody) GetTriageBundle() *TriageBundleRef {
	if x != nil {
		return x.TriageBundle
	}
	return nil
}

func (x *AttemptOutcomeBody) GetSafeMessage() string {
	if x != nil {
		return x.SafeMessage
	}
	return ""
}

func (x *AttemptOutcomeBody) GetCause() *OutcomeCause {
	if x != nil {
		return x.Cause
	}
	return nil
}

func (x *AttemptOutcomeBody) GetResult() *ResultEnvelope {
	if x != nil {
		return x.Result
	}
	return nil
}

func (x *AttemptOutcomeBody) GetExecutionStarted() bool {
	if x != nil {
		return x.ExecutionStarted
	}
	return false
}

func (x *AttemptOutcomeBody) GetArtifactReceipts() []*ArtifactReceiptRef {
	if x != nil {
		return x.ArtifactReceipts
	}
	return nil
}

// Carried exact bytes, not a structured duplicate. The receiver hashes 2, compares 1, then opens
// the canonical ArtifactReceipt document. This reference is nested in AttemptOutcomeBody/1 and
// therefore has no format tag of its own.
type ArtifactReceiptRef struct {
	state                         protoimpl.MessageState `protogen:"open.v1"`
	ArtifactReceiptDigest         []byte                 `protobuf:"bytes,1,opt,name=artifact_receipt_digest,json=artifactReceiptDigest,proto3" json:"artifact_receipt_digest,omitempty"`
	ArtifactReceiptCanonicalBytes []byte                 `protobuf:"bytes,2,opt,name=artifact_receipt_canonical_bytes,json=artifactReceiptCanonicalBytes,proto3" json:"artifact_receipt_canonical_bytes,omitempty"`
	unknownFields                 protoimpl.UnknownFields
	sizeCache                     protoimpl.SizeCache
}

func (x *ArtifactReceiptRef) Reset() {
	*x = ArtifactReceiptRef{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[50]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *ArtifactReceiptRef) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*ArtifactReceiptRef) ProtoMessage() {}

func (x *ArtifactReceiptRef) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[50]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use ArtifactReceiptRef.ProtoReflect.Descriptor instead.
func (*ArtifactReceiptRef) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{50}
}

func (x *ArtifactReceiptRef) GetArtifactReceiptDigest() []byte {
	if x != nil {
		return x.ArtifactReceiptDigest
	}
	return nil
}

func (x *ArtifactReceiptRef) GetArtifactReceiptCanonicalBytes() []byte {
	if x != nil {
		return x.ArtifactReceiptCanonicalBytes
	}
	return nil
}

// DOCUMENT SHAPE: canonical `cozy.worker.v1.ArtifactReceipt/1`. TensorFS receipt bytes remain
// opaque to this protocol package; Runtime proves their digest before authoring this document.
type ArtifactReceipt struct {
	state                         protoimpl.MessageState `protogen:"open.v1"`
	OwnerAuthorityScope           string                 `protobuf:"bytes,1,opt,name=owner_authority_scope,json=ownerAuthorityScope,proto3" json:"owner_authority_scope,omitempty"`
	RequestId                     string                 `protobuf:"bytes,2,opt,name=request_id,json=requestId,proto3" json:"request_id,omitempty"`
	InvocationSpecDigest          string                 `protobuf:"bytes,3,opt,name=invocation_spec_digest,json=invocationSpecDigest,proto3" json:"invocation_spec_digest,omitempty"`
	OutputSlot                    string                 `protobuf:"bytes,4,opt,name=output_slot,json=outputSlot,proto3" json:"output_slot,omitempty"`
	ArtifactTransactionId         string                 `protobuf:"bytes,5,opt,name=artifact_transaction_id,json=artifactTransactionId,proto3" json:"artifact_transaction_id,omitempty"`
	TensorfsReceiptDigest         string                 `protobuf:"bytes,6,opt,name=tensorfs_receipt_digest,json=tensorfsReceiptDigest,proto3" json:"tensorfs_receipt_digest,omitempty"`
	TensorfsReceiptCanonicalBytes []byte                 `protobuf:"bytes,7,opt,name=tensorfs_receipt_canonical_bytes,json=tensorfsReceiptCanonicalBytes,proto3" json:"tensorfs_receipt_canonical_bytes,omitempty"`
	unknownFields                 protoimpl.UnknownFields
	sizeCache                     protoimpl.SizeCache
}

func (x *ArtifactReceipt) Reset() {
	*x = ArtifactReceipt{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[51]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *ArtifactReceipt) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*ArtifactReceipt) ProtoMessage() {}

func (x *ArtifactReceipt) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[51]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use ArtifactReceipt.ProtoReflect.Descriptor instead.
func (*ArtifactReceipt) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{51}
}

func (x *ArtifactReceipt) GetOwnerAuthorityScope() string {
	if x != nil {
		return x.OwnerAuthorityScope
	}
	return ""
}

func (x *ArtifactReceipt) GetRequestId() string {
	if x != nil {
		return x.RequestId
	}
	return ""
}

func (x *ArtifactReceipt) GetInvocationSpecDigest() string {
	if x != nil {
		return x.InvocationSpecDigest
	}
	return ""
}

func (x *ArtifactReceipt) GetOutputSlot() string {
	if x != nil {
		return x.OutputSlot
	}
	return ""
}

func (x *ArtifactReceipt) GetArtifactTransactionId() string {
	if x != nil {
		return x.ArtifactTransactionId
	}
	return ""
}

func (x *ArtifactReceipt) GetTensorfsReceiptDigest() string {
	if x != nil {
		return x.TensorfsReceiptDigest
	}
	return ""
}

func (x *ArtifactReceipt) GetTensorfsReceiptCanonicalBytes() []byte {
	if x != nil {
		return x.TensorfsReceiptCanonicalBytes
	}
	return nil
}

// One produced TensorFS ObjectRef plus an opaque transaction-scoped reader token. source_ref uses
// `<namespace>:<token>` printable ASCII with no slash, backslash, URI, or filesystem semantics.
type ArtifactObjectSource struct {
	state         protoimpl.MessageState `protogen:"open.v1"`
	ObjectId      string                 `protobuf:"bytes,1,opt,name=object_id,json=objectId,proto3" json:"object_id,omitempty"`
	Length        uint64                 `protobuf:"varint,2,opt,name=length,proto3" json:"length,omitempty"`
	SourceRef     string                 `protobuf:"bytes,3,opt,name=source_ref,json=sourceRef,proto3" json:"source_ref,omitempty"`
	unknownFields protoimpl.UnknownFields
	sizeCache     protoimpl.SizeCache
}

func (x *ArtifactObjectSource) Reset() {
	*x = ArtifactObjectSource{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[52]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *ArtifactObjectSource) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*ArtifactObjectSource) ProtoMessage() {}

func (x *ArtifactObjectSource) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[52]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use ArtifactObjectSource.ProtoReflect.Descriptor instead.
func (*ArtifactObjectSource) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{52}
}

func (x *ArtifactObjectSource) GetObjectId() string {
	if x != nil {
		return x.ObjectId
	}
	return ""
}

func (x *ArtifactObjectSource) GetLength() uint64 {
	if x != nil {
		return x.Length
	}
	return 0
}

func (x *ArtifactObjectSource) GetSourceRef() string {
	if x != nil {
		return x.SourceRef
	}
	return ""
}

// Runtime asks its authenticated host to commit semantic intent before the first TensorFS write.
// requested_writer_generation == 0 means first open or replacement-child rejoin. Once ACKed, an
// identical retry from the same child echoes the generation and receives the same ACK without
// fencing itself. The attempt ordinal is routing only and does not enter transaction identity.
// The TensorFS declaration is carried directly: it is not wrapped in a second Worker Protocol
// document. Its digest and exact bytes are the durable conflict/replay fence.
type ArtifactIntentFrame struct {
	state                             protoimpl.MessageState `protogen:"open.v1"`
	RecordOwnerEpoch                  uint64                 `protobuf:"varint,1,opt,name=record_owner_epoch,json=recordOwnerEpoch,proto3" json:"record_owner_epoch,omitempty"`
	ControlStreamGeneration           uint64                 `protobuf:"varint,2,opt,name=control_stream_generation,json=controlStreamGeneration,proto3" json:"control_stream_generation,omitempty"`
	WorkerBootId                      string                 `protobuf:"bytes,3,opt,name=worker_boot_id,json=workerBootId,proto3" json:"worker_boot_id,omitempty"`
	RequestId                         string                 `protobuf:"bytes,5,opt,name=request_id,json=requestId,proto3" json:"request_id,omitempty"`
	AttemptOrdinal                    uint64                 `protobuf:"varint,6,opt,name=attempt_ordinal,json=attemptOrdinal,proto3" json:"attempt_ordinal,omitempty"`
	InvocationSpecDigest              []byte                 `protobuf:"bytes,7,opt,name=invocation_spec_digest,json=invocationSpecDigest,proto3" json:"invocation_spec_digest,omitempty"`
	OutputSlot                        string                 `protobuf:"bytes,8,opt,name=output_slot,json=outputSlot,proto3" json:"output_slot,omitempty"`
	TensorfsDeclarationDigest         []byte                 `protobuf:"bytes,9,opt,name=tensorfs_declaration_digest,json=tensorfsDeclarationDigest,proto3" json:"tensorfs_declaration_digest,omitempty"`
	RequestedWriterGeneration         uint64                 `protobuf:"varint,10,opt,name=requested_writer_generation,json=requestedWriterGeneration,proto3" json:"requested_writer_generation,omitempty"`
	TensorfsDeclarationCanonicalBytes []byte                 `protobuf:"bytes,11,opt,name=tensorfs_declaration_canonical_bytes,json=tensorfsDeclarationCanonicalBytes,proto3" json:"tensorfs_declaration_canonical_bytes,omitempty"`
	unknownFields                     protoimpl.UnknownFields
	sizeCache                         protoimpl.SizeCache
}

func (x *ArtifactIntentFrame) Reset() {
	*x = ArtifactIntentFrame{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[53]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *ArtifactIntentFrame) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*ArtifactIntentFrame) ProtoMessage() {}

func (x *ArtifactIntentFrame) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[53]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use ArtifactIntentFrame.ProtoReflect.Descriptor instead.
func (*ArtifactIntentFrame) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{53}
}

func (x *ArtifactIntentFrame) GetRecordOwnerEpoch() uint64 {
	if x != nil {
		return x.RecordOwnerEpoch
	}
	return 0
}

func (x *ArtifactIntentFrame) GetControlStreamGeneration() uint64 {
	if x != nil {
		return x.ControlStreamGeneration
	}
	return 0
}

func (x *ArtifactIntentFrame) GetWorkerBootId() string {
	if x != nil {
		return x.WorkerBootId
	}
	return ""
}

func (x *ArtifactIntentFrame) GetRequestId() string {
	if x != nil {
		return x.RequestId
	}
	return ""
}

func (x *ArtifactIntentFrame) GetAttemptOrdinal() uint64 {
	if x != nil {
		return x.AttemptOrdinal
	}
	return 0
}

func (x *ArtifactIntentFrame) GetInvocationSpecDigest() []byte {
	if x != nil {
		return x.InvocationSpecDigest
	}
	return nil
}

func (x *ArtifactIntentFrame) GetOutputSlot() string {
	if x != nil {
		return x.OutputSlot
	}
	return ""
}

func (x *ArtifactIntentFrame) GetTensorfsDeclarationDigest() []byte {
	if x != nil {
		return x.TensorfsDeclarationDigest
	}
	return nil
}

func (x *ArtifactIntentFrame) GetRequestedWriterGeneration() uint64 {
	if x != nil {
		return x.RequestedWriterGeneration
	}
	return 0
}

func (x *ArtifactIntentFrame) GetTensorfsDeclarationCanonicalBytes() []byte {
	if x != nil {
		return x.TensorfsDeclarationCanonicalBytes
	}
	return nil
}

// One ACK type covers intent and receipt durability. The supervisor injects it into Runtime only
// after its one SQLite transaction commits. A replacement child receives the same transaction id
// and a newly fenced writer generation; if a receipt already exists it is replayed here and no
// TensorFS write occurs. REFUSED carries no usable generation or transaction authority.
type ArtifactHostAck struct {
	state                     protoimpl.MessageState `protogen:"open.v1"`
	RecordOwnerEpoch          uint64                 `protobuf:"varint,1,opt,name=record_owner_epoch,json=recordOwnerEpoch,proto3" json:"record_owner_epoch,omitempty"`
	ControlStreamGeneration   uint64                 `protobuf:"varint,2,opt,name=control_stream_generation,json=controlStreamGeneration,proto3" json:"control_stream_generation,omitempty"`
	WorkerBootId              string                 `protobuf:"bytes,3,opt,name=worker_boot_id,json=workerBootId,proto3" json:"worker_boot_id,omitempty"`
	RequestId                 string                 `protobuf:"bytes,5,opt,name=request_id,json=requestId,proto3" json:"request_id,omitempty"`
	AttemptOrdinal            uint64                 `protobuf:"varint,6,opt,name=attempt_ordinal,json=attemptOrdinal,proto3" json:"attempt_ordinal,omitempty"`
	InvocationSpecDigest      []byte                 `protobuf:"bytes,7,opt,name=invocation_spec_digest,json=invocationSpecDigest,proto3" json:"invocation_spec_digest,omitempty"`
	OutputSlot                string                 `protobuf:"bytes,8,opt,name=output_slot,json=outputSlot,proto3" json:"output_slot,omitempty"`
	ArtifactTransactionId     string                 `protobuf:"bytes,9,opt,name=artifact_transaction_id,json=artifactTransactionId,proto3" json:"artifact_transaction_id,omitempty"`
	WriterGeneration          uint64                 `protobuf:"varint,10,opt,name=writer_generation,json=writerGeneration,proto3" json:"writer_generation,omitempty"`
	TensorfsDeclarationDigest []byte                 `protobuf:"bytes,11,opt,name=tensorfs_declaration_digest,json=tensorfsDeclarationDigest,proto3" json:"tensorfs_declaration_digest,omitempty"`
	Stage                     ArtifactHostStage      `protobuf:"varint,12,opt,name=stage,proto3,enum=cozy.worker.v1.ArtifactHostStage" json:"stage,omitempty"`
	Outcome                   ArtifactHostOutcome    `protobuf:"varint,13,opt,name=outcome,proto3,enum=cozy.worker.v1.ArtifactHostOutcome" json:"outcome,omitempty"`
	Refusal                   ArtifactHostRefusal    `protobuf:"varint,14,opt,name=refusal,proto3,enum=cozy.worker.v1.ArtifactHostRefusal" json:"refusal,omitempty"`
	ArtifactReceipt           *ArtifactReceiptRef    `protobuf:"bytes,15,opt,name=artifact_receipt,json=artifactReceipt,proto3" json:"artifact_receipt,omitempty"`
	Manifest                  *Ref                   `protobuf:"bytes,16,opt,name=manifest,proto3" json:"manifest,omitempty"` // present exactly with artifact_receipt; the produced Manifest identity
	unknownFields             protoimpl.UnknownFields
	sizeCache                 protoimpl.SizeCache
}

func (x *ArtifactHostAck) Reset() {
	*x = ArtifactHostAck{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[54]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *ArtifactHostAck) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*ArtifactHostAck) ProtoMessage() {}

func (x *ArtifactHostAck) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[54]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use ArtifactHostAck.ProtoReflect.Descriptor instead.
func (*ArtifactHostAck) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{54}
}

func (x *ArtifactHostAck) GetRecordOwnerEpoch() uint64 {
	if x != nil {
		return x.RecordOwnerEpoch
	}
	return 0
}

func (x *ArtifactHostAck) GetControlStreamGeneration() uint64 {
	if x != nil {
		return x.ControlStreamGeneration
	}
	return 0
}

func (x *ArtifactHostAck) GetWorkerBootId() string {
	if x != nil {
		return x.WorkerBootId
	}
	return ""
}

func (x *ArtifactHostAck) GetRequestId() string {
	if x != nil {
		return x.RequestId
	}
	return ""
}

func (x *ArtifactHostAck) GetAttemptOrdinal() uint64 {
	if x != nil {
		return x.AttemptOrdinal
	}
	return 0
}

func (x *ArtifactHostAck) GetInvocationSpecDigest() []byte {
	if x != nil {
		return x.InvocationSpecDigest
	}
	return nil
}

func (x *ArtifactHostAck) GetOutputSlot() string {
	if x != nil {
		return x.OutputSlot
	}
	return ""
}

func (x *ArtifactHostAck) GetArtifactTransactionId() string {
	if x != nil {
		return x.ArtifactTransactionId
	}
	return ""
}

func (x *ArtifactHostAck) GetWriterGeneration() uint64 {
	if x != nil {
		return x.WriterGeneration
	}
	return 0
}

func (x *ArtifactHostAck) GetTensorfsDeclarationDigest() []byte {
	if x != nil {
		return x.TensorfsDeclarationDigest
	}
	return nil
}

func (x *ArtifactHostAck) GetStage() ArtifactHostStage {
	if x != nil {
		return x.Stage
	}
	return ArtifactHostStage_ARTIFACT_HOST_STAGE_UNSPECIFIED
}

func (x *ArtifactHostAck) GetOutcome() ArtifactHostOutcome {
	if x != nil {
		return x.Outcome
	}
	return ArtifactHostOutcome_ARTIFACT_HOST_OUTCOME_UNSPECIFIED
}

func (x *ArtifactHostAck) GetRefusal() ArtifactHostRefusal {
	if x != nil {
		return x.Refusal
	}
	return ArtifactHostRefusal_ARTIFACT_HOST_REFUSAL_UNSPECIFIED
}

func (x *ArtifactHostAck) GetArtifactReceipt() *ArtifactReceiptRef {
	if x != nil {
		return x.ArtifactReceipt
	}
	return nil
}

func (x *ArtifactHostAck) GetManifest() *Ref {
	if x != nil {
		return x.Manifest
	}
	return nil
}

// Runtime sends the exact committed receipt and exact local source inventory together. The
// supervisor validates all routing copies and writer_generation, commits both before ACK, and then
// may forward this frame to the external RecordOwner. A stale child cannot publish a late receipt.
type ArtifactReceiptFrame struct {
	state                     protoimpl.MessageState  `protogen:"open.v1"`
	RecordOwnerEpoch          uint64                  `protobuf:"varint,1,opt,name=record_owner_epoch,json=recordOwnerEpoch,proto3" json:"record_owner_epoch,omitempty"`
	ControlStreamGeneration   uint64                  `protobuf:"varint,2,opt,name=control_stream_generation,json=controlStreamGeneration,proto3" json:"control_stream_generation,omitempty"`
	WorkerBootId              string                  `protobuf:"bytes,3,opt,name=worker_boot_id,json=workerBootId,proto3" json:"worker_boot_id,omitempty"`
	RequestId                 string                  `protobuf:"bytes,5,opt,name=request_id,json=requestId,proto3" json:"request_id,omitempty"`
	AttemptOrdinal            uint64                  `protobuf:"varint,6,opt,name=attempt_ordinal,json=attemptOrdinal,proto3" json:"attempt_ordinal,omitempty"`
	InvocationSpecDigest      []byte                  `protobuf:"bytes,7,opt,name=invocation_spec_digest,json=invocationSpecDigest,proto3" json:"invocation_spec_digest,omitempty"`
	OutputSlot                string                  `protobuf:"bytes,8,opt,name=output_slot,json=outputSlot,proto3" json:"output_slot,omitempty"`
	ArtifactTransactionId     string                  `protobuf:"bytes,9,opt,name=artifact_transaction_id,json=artifactTransactionId,proto3" json:"artifact_transaction_id,omitempty"`
	WriterGeneration          uint64                  `protobuf:"varint,10,opt,name=writer_generation,json=writerGeneration,proto3" json:"writer_generation,omitempty"`
	TensorfsDeclarationDigest []byte                  `protobuf:"bytes,11,opt,name=tensorfs_declaration_digest,json=tensorfsDeclarationDigest,proto3" json:"tensorfs_declaration_digest,omitempty"`
	ArtifactReceipt           *ArtifactReceiptRef     `protobuf:"bytes,12,opt,name=artifact_receipt,json=artifactReceipt,proto3" json:"artifact_receipt,omitempty"`
	Objects                   []*ArtifactObjectSource `protobuf:"bytes,13,rep,name=objects,proto3" json:"objects,omitempty"` // sorted uniquely by object_id; committed in the
	// same host-ledger transaction as artifact_receipt
	Manifest      *Ref `protobuf:"bytes,14,opt,name=manifest,proto3" json:"manifest,omitempty"` // exact produced TensorFS Manifest; it is also one row in objects
	unknownFields protoimpl.UnknownFields
	sizeCache     protoimpl.SizeCache
}

func (x *ArtifactReceiptFrame) Reset() {
	*x = ArtifactReceiptFrame{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[55]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *ArtifactReceiptFrame) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*ArtifactReceiptFrame) ProtoMessage() {}

func (x *ArtifactReceiptFrame) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[55]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use ArtifactReceiptFrame.ProtoReflect.Descriptor instead.
func (*ArtifactReceiptFrame) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{55}
}

func (x *ArtifactReceiptFrame) GetRecordOwnerEpoch() uint64 {
	if x != nil {
		return x.RecordOwnerEpoch
	}
	return 0
}

func (x *ArtifactReceiptFrame) GetControlStreamGeneration() uint64 {
	if x != nil {
		return x.ControlStreamGeneration
	}
	return 0
}

func (x *ArtifactReceiptFrame) GetWorkerBootId() string {
	if x != nil {
		return x.WorkerBootId
	}
	return ""
}

func (x *ArtifactReceiptFrame) GetRequestId() string {
	if x != nil {
		return x.RequestId
	}
	return ""
}

func (x *ArtifactReceiptFrame) GetAttemptOrdinal() uint64 {
	if x != nil {
		return x.AttemptOrdinal
	}
	return 0
}

func (x *ArtifactReceiptFrame) GetInvocationSpecDigest() []byte {
	if x != nil {
		return x.InvocationSpecDigest
	}
	return nil
}

func (x *ArtifactReceiptFrame) GetOutputSlot() string {
	if x != nil {
		return x.OutputSlot
	}
	return ""
}

func (x *ArtifactReceiptFrame) GetArtifactTransactionId() string {
	if x != nil {
		return x.ArtifactTransactionId
	}
	return ""
}

func (x *ArtifactReceiptFrame) GetWriterGeneration() uint64 {
	if x != nil {
		return x.WriterGeneration
	}
	return 0
}

func (x *ArtifactReceiptFrame) GetTensorfsDeclarationDigest() []byte {
	if x != nil {
		return x.TensorfsDeclarationDigest
	}
	return nil
}

func (x *ArtifactReceiptFrame) GetArtifactReceipt() *ArtifactReceiptRef {
	if x != nil {
		return x.ArtifactReceipt
	}
	return nil
}

func (x *ArtifactReceiptFrame) GetObjects() []*ArtifactObjectSource {
	if x != nil {
		return x.Objects
	}
	return nil
}

func (x *ArtifactReceiptFrame) GetManifest() *Ref {
	if x != nil {
		return x.Manifest
	}
	return nil
}

// Compact supervisor-ledger state included in WorkerSnapshotBody/1. A stateless replacement
// Runtime does not author these rows; pod-supervisor injects them before digesting the external
// snapshot. It MUST NOT put declarations, receipt bytes, or object inventory here. After the
// snapshot ACK, the supervisor replays each RECEIPT transaction's exact ArtifactReceiptFrame.
type ArtifactTransactionStatus struct {
	state                     protoimpl.MessageState   `protogen:"open.v1"`
	ArtifactTransactionId     string                   `protobuf:"bytes,1,opt,name=artifact_transaction_id,json=artifactTransactionId,proto3" json:"artifact_transaction_id,omitempty"`
	RequestId                 string                   `protobuf:"bytes,2,opt,name=request_id,json=requestId,proto3" json:"request_id,omitempty"`
	AttemptOrdinal            uint64                   `protobuf:"varint,3,opt,name=attempt_ordinal,json=attemptOrdinal,proto3" json:"attempt_ordinal,omitempty"`
	InvocationSpecDigest      string                   `protobuf:"bytes,4,opt,name=invocation_spec_digest,json=invocationSpecDigest,proto3" json:"invocation_spec_digest,omitempty"`
	OutputSlot                string                   `protobuf:"bytes,5,opt,name=output_slot,json=outputSlot,proto3" json:"output_slot,omitempty"`
	WriterGeneration          uint64                   `protobuf:"varint,6,opt,name=writer_generation,json=writerGeneration,proto3" json:"writer_generation,omitempty"`
	State                     ArtifactTransactionState `protobuf:"varint,7,opt,name=state,proto3,enum=cozy.worker.v1.ArtifactTransactionState" json:"state,omitempty"`
	TensorfsDeclarationDigest []byte                   `protobuf:"bytes,8,opt,name=tensorfs_declaration_digest,json=tensorfsDeclarationDigest,proto3" json:"tensorfs_declaration_digest,omitempty"`
	ArtifactReceiptDigest     []byte                   `protobuf:"bytes,9,opt,name=artifact_receipt_digest,json=artifactReceiptDigest,proto3" json:"artifact_receipt_digest,omitempty"` // absent in INTENT; exact replay fence in RECEIPT
	unknownFields             protoimpl.UnknownFields
	sizeCache                 protoimpl.SizeCache
}

func (x *ArtifactTransactionStatus) Reset() {
	*x = ArtifactTransactionStatus{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[56]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *ArtifactTransactionStatus) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*ArtifactTransactionStatus) ProtoMessage() {}

func (x *ArtifactTransactionStatus) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[56]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use ArtifactTransactionStatus.ProtoReflect.Descriptor instead.
func (*ArtifactTransactionStatus) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{56}
}

func (x *ArtifactTransactionStatus) GetArtifactTransactionId() string {
	if x != nil {
		return x.ArtifactTransactionId
	}
	return ""
}

func (x *ArtifactTransactionStatus) GetRequestId() string {
	if x != nil {
		return x.RequestId
	}
	return ""
}

func (x *ArtifactTransactionStatus) GetAttemptOrdinal() uint64 {
	if x != nil {
		return x.AttemptOrdinal
	}
	return 0
}

func (x *ArtifactTransactionStatus) GetInvocationSpecDigest() string {
	if x != nil {
		return x.InvocationSpecDigest
	}
	return ""
}

func (x *ArtifactTransactionStatus) GetOutputSlot() string {
	if x != nil {
		return x.OutputSlot
	}
	return ""
}

func (x *ArtifactTransactionStatus) GetWriterGeneration() uint64 {
	if x != nil {
		return x.WriterGeneration
	}
	return 0
}

func (x *ArtifactTransactionStatus) GetState() ArtifactTransactionState {
	if x != nil {
		return x.State
	}
	return ArtifactTransactionState_ARTIFACT_TRANSACTION_STATE_UNSPECIFIED
}

func (x *ArtifactTransactionStatus) GetTensorfsDeclarationDigest() []byte {
	if x != nil {
		return x.TensorfsDeclarationDigest
	}
	return nil
}

func (x *ArtifactTransactionStatus) GetArtifactReceiptDigest() []byte {
	if x != nil {
		return x.ArtifactReceiptDigest
	}
	return nil
}

// The supervisor reads produced objects without receiving a path or giving Runtime a network
// destination. It requests one exact bounded range from the opaque source_ref in the committed
// inventory. External RecordOwners cannot author this host-child frame.
type ArtifactReadRequest struct {
	state                   protoimpl.MessageState `protogen:"open.v1"`
	RecordOwnerEpoch        uint64                 `protobuf:"varint,1,opt,name=record_owner_epoch,json=recordOwnerEpoch,proto3" json:"record_owner_epoch,omitempty"`
	ControlStreamGeneration uint64                 `protobuf:"varint,2,opt,name=control_stream_generation,json=controlStreamGeneration,proto3" json:"control_stream_generation,omitempty"`
	WorkerBootId            string                 `protobuf:"bytes,3,opt,name=worker_boot_id,json=workerBootId,proto3" json:"worker_boot_id,omitempty"`
	ArtifactTransactionId   string                 `protobuf:"bytes,5,opt,name=artifact_transaction_id,json=artifactTransactionId,proto3" json:"artifact_transaction_id,omitempty"`
	WriterGeneration        uint64                 `protobuf:"varint,6,opt,name=writer_generation,json=writerGeneration,proto3" json:"writer_generation,omitempty"`
	ObjectId                string                 `protobuf:"bytes,7,opt,name=object_id,json=objectId,proto3" json:"object_id,omitempty"`
	SourceRef               string                 `protobuf:"bytes,8,opt,name=source_ref,json=sourceRef,proto3" json:"source_ref,omitempty"`
	ReadId                  string                 `protobuf:"bytes,9,opt,name=read_id,json=readId,proto3" json:"read_id,omitempty"`
	Offset                  uint64                 `protobuf:"varint,10,opt,name=offset,proto3" json:"offset,omitempty"`
	MaxBytes                uint32                 `protobuf:"varint,11,opt,name=max_bytes,json=maxBytes,proto3" json:"max_bytes,omitempty"`
	unknownFields           protoimpl.UnknownFields
	sizeCache               protoimpl.SizeCache
}

func (x *ArtifactReadRequest) Reset() {
	*x = ArtifactReadRequest{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[57]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *ArtifactReadRequest) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*ArtifactReadRequest) ProtoMessage() {}

func (x *ArtifactReadRequest) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[57]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use ArtifactReadRequest.ProtoReflect.Descriptor instead.
func (*ArtifactReadRequest) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{57}
}

func (x *ArtifactReadRequest) GetRecordOwnerEpoch() uint64 {
	if x != nil {
		return x.RecordOwnerEpoch
	}
	return 0
}

func (x *ArtifactReadRequest) GetControlStreamGeneration() uint64 {
	if x != nil {
		return x.ControlStreamGeneration
	}
	return 0
}

func (x *ArtifactReadRequest) GetWorkerBootId() string {
	if x != nil {
		return x.WorkerBootId
	}
	return ""
}

func (x *ArtifactReadRequest) GetArtifactTransactionId() string {
	if x != nil {
		return x.ArtifactTransactionId
	}
	return ""
}

func (x *ArtifactReadRequest) GetWriterGeneration() uint64 {
	if x != nil {
		return x.WriterGeneration
	}
	return 0
}

func (x *ArtifactReadRequest) GetObjectId() string {
	if x != nil {
		return x.ObjectId
	}
	return ""
}

func (x *ArtifactReadRequest) GetSourceRef() string {
	if x != nil {
		return x.SourceRef
	}
	return ""
}

func (x *ArtifactReadRequest) GetReadId() string {
	if x != nil {
		return x.ReadId
	}
	return ""
}

func (x *ArtifactReadRequest) GetOffset() uint64 {
	if x != nil {
		return x.Offset
	}
	return 0
}

func (x *ArtifactReadRequest) GetMaxBytes() uint32 {
	if x != nil {
		return x.MaxBytes
	}
	return 0
}

type ArtifactReadResult struct {
	state                   protoimpl.MessageState `protogen:"open.v1"`
	RecordOwnerEpoch        uint64                 `protobuf:"varint,1,opt,name=record_owner_epoch,json=recordOwnerEpoch,proto3" json:"record_owner_epoch,omitempty"`
	ControlStreamGeneration uint64                 `protobuf:"varint,2,opt,name=control_stream_generation,json=controlStreamGeneration,proto3" json:"control_stream_generation,omitempty"`
	WorkerBootId            string                 `protobuf:"bytes,3,opt,name=worker_boot_id,json=workerBootId,proto3" json:"worker_boot_id,omitempty"`
	ArtifactTransactionId   string                 `protobuf:"bytes,5,opt,name=artifact_transaction_id,json=artifactTransactionId,proto3" json:"artifact_transaction_id,omitempty"`
	WriterGeneration        uint64                 `protobuf:"varint,6,opt,name=writer_generation,json=writerGeneration,proto3" json:"writer_generation,omitempty"`
	ObjectId                string                 `protobuf:"bytes,7,opt,name=object_id,json=objectId,proto3" json:"object_id,omitempty"`
	SourceRef               string                 `protobuf:"bytes,8,opt,name=source_ref,json=sourceRef,proto3" json:"source_ref,omitempty"`
	ReadId                  string                 `protobuf:"bytes,9,opt,name=read_id,json=readId,proto3" json:"read_id,omitempty"`
	Offset                  uint64                 `protobuf:"varint,10,opt,name=offset,proto3" json:"offset,omitempty"`
	Outcome                 ArtifactReadOutcome    `protobuf:"varint,11,opt,name=outcome,proto3,enum=cozy.worker.v1.ArtifactReadOutcome" json:"outcome,omitempty"`
	Data                    []byte                 `protobuf:"bytes,12,opt,name=data,proto3" json:"data,omitempty"`
	DataDigest              []byte                 `protobuf:"bytes,13,opt,name=data_digest,json=dataDigest,proto3" json:"data_digest,omitempty"`
	Refusal                 ArtifactReadRefusal    `protobuf:"varint,14,opt,name=refusal,proto3,enum=cozy.worker.v1.ArtifactReadRefusal" json:"refusal,omitempty"`
	SafeDetail              string                 `protobuf:"bytes,15,opt,name=safe_detail,json=safeDetail,proto3" json:"safe_detail,omitempty"`
	unknownFields           protoimpl.UnknownFields
	sizeCache               protoimpl.SizeCache
}

func (x *ArtifactReadResult) Reset() {
	*x = ArtifactReadResult{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[58]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *ArtifactReadResult) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*ArtifactReadResult) ProtoMessage() {}

func (x *ArtifactReadResult) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[58]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use ArtifactReadResult.ProtoReflect.Descriptor instead.
func (*ArtifactReadResult) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{58}
}

func (x *ArtifactReadResult) GetRecordOwnerEpoch() uint64 {
	if x != nil {
		return x.RecordOwnerEpoch
	}
	return 0
}

func (x *ArtifactReadResult) GetControlStreamGeneration() uint64 {
	if x != nil {
		return x.ControlStreamGeneration
	}
	return 0
}

func (x *ArtifactReadResult) GetWorkerBootId() string {
	if x != nil {
		return x.WorkerBootId
	}
	return ""
}

func (x *ArtifactReadResult) GetArtifactTransactionId() string {
	if x != nil {
		return x.ArtifactTransactionId
	}
	return ""
}

func (x *ArtifactReadResult) GetWriterGeneration() uint64 {
	if x != nil {
		return x.WriterGeneration
	}
	return 0
}

func (x *ArtifactReadResult) GetObjectId() string {
	if x != nil {
		return x.ObjectId
	}
	return ""
}

func (x *ArtifactReadResult) GetSourceRef() string {
	if x != nil {
		return x.SourceRef
	}
	return ""
}

func (x *ArtifactReadResult) GetReadId() string {
	if x != nil {
		return x.ReadId
	}
	return ""
}

func (x *ArtifactReadResult) GetOffset() uint64 {
	if x != nil {
		return x.Offset
	}
	return 0
}

func (x *ArtifactReadResult) GetOutcome() ArtifactReadOutcome {
	if x != nil {
		return x.Outcome
	}
	return ArtifactReadOutcome_ARTIFACT_READ_OUTCOME_UNSPECIFIED
}

func (x *ArtifactReadResult) GetData() []byte {
	if x != nil {
		return x.Data
	}
	return nil
}

func (x *ArtifactReadResult) GetDataDigest() []byte {
	if x != nil {
		return x.DataDigest
	}
	return nil
}

func (x *ArtifactReadResult) GetRefusal() ArtifactReadRefusal {
	if x != nil {
		return x.Refusal
	}
	return ArtifactReadRefusal_ARTIFACT_READ_REFUSAL_UNSPECIFIED
}

func (x *ArtifactReadResult) GetSafeDetail() string {
	if x != nil {
		return x.SafeDetail
	}
	return ""
}

type ArtifactObjectRef struct {
	state         protoimpl.MessageState `protogen:"open.v1"`
	ObjectId      string                 `protobuf:"bytes,1,opt,name=object_id,json=objectId,proto3" json:"object_id,omitempty"`
	Length        uint64                 `protobuf:"varint,2,opt,name=length,proto3" json:"length,omitempty"`
	unknownFields protoimpl.UnknownFields
	sizeCache     protoimpl.SizeCache
}

func (x *ArtifactObjectRef) Reset() {
	*x = ArtifactObjectRef{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[59]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *ArtifactObjectRef) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*ArtifactObjectRef) ProtoMessage() {}

func (x *ArtifactObjectRef) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[59]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use ArtifactObjectRef.ProtoReflect.Descriptor instead.
func (*ArtifactObjectRef) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{59}
}

func (x *ArtifactObjectRef) GetObjectId() string {
	if x != nil {
		return x.ObjectId
	}
	return ""
}

func (x *ArtifactObjectRef) GetLength() uint64 {
	if x != nil {
		return x.Length
	}
	return 0
}

type ArtifactUploadHeader struct {
	state         protoimpl.MessageState `protogen:"open.v1"`
	Name          string                 `protobuf:"bytes,1,opt,name=name,proto3" json:"name,omitempty"`
	Value         string                 `protobuf:"bytes,2,opt,name=value,proto3" json:"value,omitempty"`
	unknownFields protoimpl.UnknownFields
	sizeCache     protoimpl.SizeCache
}

func (x *ArtifactUploadHeader) Reset() {
	*x = ArtifactUploadHeader{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[60]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *ArtifactUploadHeader) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*ArtifactUploadHeader) ProtoMessage() {}

func (x *ArtifactUploadHeader) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[60]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use ArtifactUploadHeader.ProtoReflect.Descriptor instead.
func (*ArtifactUploadHeader) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{60}
}

func (x *ArtifactUploadHeader) GetName() string {
	if x != nil {
		return x.Name
	}
	return ""
}

func (x *ArtifactUploadHeader) GetValue() string {
	if x != nil {
		return x.Value
	}
	return ""
}

// Host-only th-059 upload capability. url includes the scoped query credential and is memory-only;
// headers are sorted unique lowercase names and every one is signed. This message is consumed by
// pod-supervisor and MUST NOT be forwarded to Runtime, an executor, a log, or the host ledger.
type ArtifactUploadGrant struct {
	state           protoimpl.MessageState  `protogen:"open.v1"`
	ObjectId        string                  `protobuf:"bytes,1,opt,name=object_id,json=objectId,proto3" json:"object_id,omitempty"`
	Length          uint64                  `protobuf:"varint,2,opt,name=length,proto3" json:"length,omitempty"`
	Url             string                  `protobuf:"bytes,3,opt,name=url,proto3" json:"url,omitempty"`
	RequiredHeaders []*ArtifactUploadHeader `protobuf:"bytes,4,rep,name=required_headers,json=requiredHeaders,proto3" json:"required_headers,omitempty"`
	ExpiresAtUnix   uint64                  `protobuf:"varint,5,opt,name=expires_at_unix,json=expiresAtUnix,proto3" json:"expires_at_unix,omitempty"`
	unknownFields   protoimpl.UnknownFields
	sizeCache       protoimpl.SizeCache
}

func (x *ArtifactUploadGrant) Reset() {
	*x = ArtifactUploadGrant{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[61]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *ArtifactUploadGrant) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*ArtifactUploadGrant) ProtoMessage() {}

func (x *ArtifactUploadGrant) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[61]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use ArtifactUploadGrant.ProtoReflect.Descriptor instead.
func (*ArtifactUploadGrant) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{61}
}

func (x *ArtifactUploadGrant) GetObjectId() string {
	if x != nil {
		return x.ObjectId
	}
	return ""
}

func (x *ArtifactUploadGrant) GetLength() uint64 {
	if x != nil {
		return x.Length
	}
	return 0
}

func (x *ArtifactUploadGrant) GetUrl() string {
	if x != nil {
		return x.Url
	}
	return ""
}

func (x *ArtifactUploadGrant) GetRequiredHeaders() []*ArtifactUploadHeader {
	if x != nil {
		return x.RequiredHeaders
	}
	return nil
}

func (x *ArtifactUploadGrant) GetExpiresAtUnix() uint64 {
	if x != nil {
		return x.ExpiresAtUnix
	}
	return 0
}

// Authenticated RecordOwner -> supervisor request for ONE exact object. Exact replay of one
// operation/revision compares every field; a higher grant_revision refreshes only that object's
// capability without changing artifact identity. RecordOwners pipeline several requests instead
// of putting thousands of presigned URLs in one protobuf.
type ArtifactTransferRequest struct {
	state                   protoimpl.MessageState `protogen:"open.v1"`
	RecordOwnerEpoch        uint64                 `protobuf:"varint,1,opt,name=record_owner_epoch,json=recordOwnerEpoch,proto3" json:"record_owner_epoch,omitempty"`
	ControlStreamGeneration uint64                 `protobuf:"varint,2,opt,name=control_stream_generation,json=controlStreamGeneration,proto3" json:"control_stream_generation,omitempty"`
	WorkerBootId            string                 `protobuf:"bytes,3,opt,name=worker_boot_id,json=workerBootId,proto3" json:"worker_boot_id,omitempty"`
	RequestId               string                 `protobuf:"bytes,5,opt,name=request_id,json=requestId,proto3" json:"request_id,omitempty"`
	AttemptOrdinal          uint64                 `protobuf:"varint,6,opt,name=attempt_ordinal,json=attemptOrdinal,proto3" json:"attempt_ordinal,omitempty"`
	InvocationSpecDigest    []byte                 `protobuf:"bytes,7,opt,name=invocation_spec_digest,json=invocationSpecDigest,proto3" json:"invocation_spec_digest,omitempty"`
	OutputSlot              string                 `protobuf:"bytes,8,opt,name=output_slot,json=outputSlot,proto3" json:"output_slot,omitempty"`
	ArtifactTransactionId   string                 `protobuf:"bytes,9,opt,name=artifact_transaction_id,json=artifactTransactionId,proto3" json:"artifact_transaction_id,omitempty"`
	ArtifactReceiptDigest   []byte                 `protobuf:"bytes,10,opt,name=artifact_receipt_digest,json=artifactReceiptDigest,proto3" json:"artifact_receipt_digest,omitempty"`
	OperationId             string                 `protobuf:"bytes,11,opt,name=operation_id,json=operationId,proto3" json:"operation_id,omitempty"`
	GrantRevision           uint64                 `protobuf:"varint,12,opt,name=grant_revision,json=grantRevision,proto3" json:"grant_revision,omitempty"`
	// Types that are valid to be assigned to Decision:
	//
	//	*ArtifactTransferRequest_UploadGrant
	//	*ArtifactTransferRequest_Held
	Decision      isArtifactTransferRequest_Decision `protobuf_oneof:"decision"`
	unknownFields protoimpl.UnknownFields
	sizeCache     protoimpl.SizeCache
}

func (x *ArtifactTransferRequest) Reset() {
	*x = ArtifactTransferRequest{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[62]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *ArtifactTransferRequest) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*ArtifactTransferRequest) ProtoMessage() {}

func (x *ArtifactTransferRequest) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[62]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use ArtifactTransferRequest.ProtoReflect.Descriptor instead.
func (*ArtifactTransferRequest) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{62}
}

func (x *ArtifactTransferRequest) GetRecordOwnerEpoch() uint64 {
	if x != nil {
		return x.RecordOwnerEpoch
	}
	return 0
}

func (x *ArtifactTransferRequest) GetControlStreamGeneration() uint64 {
	if x != nil {
		return x.ControlStreamGeneration
	}
	return 0
}

func (x *ArtifactTransferRequest) GetWorkerBootId() string {
	if x != nil {
		return x.WorkerBootId
	}
	return ""
}

func (x *ArtifactTransferRequest) GetRequestId() string {
	if x != nil {
		return x.RequestId
	}
	return ""
}

func (x *ArtifactTransferRequest) GetAttemptOrdinal() uint64 {
	if x != nil {
		return x.AttemptOrdinal
	}
	return 0
}

func (x *ArtifactTransferRequest) GetInvocationSpecDigest() []byte {
	if x != nil {
		return x.InvocationSpecDigest
	}
	return nil
}

func (x *ArtifactTransferRequest) GetOutputSlot() string {
	if x != nil {
		return x.OutputSlot
	}
	return ""
}

func (x *ArtifactTransferRequest) GetArtifactTransactionId() string {
	if x != nil {
		return x.ArtifactTransactionId
	}
	return ""
}

func (x *ArtifactTransferRequest) GetArtifactReceiptDigest() []byte {
	if x != nil {
		return x.ArtifactReceiptDigest
	}
	return nil
}

func (x *ArtifactTransferRequest) GetOperationId() string {
	if x != nil {
		return x.OperationId
	}
	return ""
}

func (x *ArtifactTransferRequest) GetGrantRevision() uint64 {
	if x != nil {
		return x.GrantRevision
	}
	return 0
}

func (x *ArtifactTransferRequest) GetDecision() isArtifactTransferRequest_Decision {
	if x != nil {
		return x.Decision
	}
	return nil
}

func (x *ArtifactTransferRequest) GetUploadGrant() *ArtifactUploadGrant {
	if x != nil {
		if x, ok := x.Decision.(*ArtifactTransferRequest_UploadGrant); ok {
			return x.UploadGrant
		}
	}
	return nil
}

func (x *ArtifactTransferRequest) GetHeld() *ArtifactObjectRef {
	if x != nil {
		if x, ok := x.Decision.(*ArtifactTransferRequest_Held); ok {
			return x.Held
		}
	}
	return nil
}

type isArtifactTransferRequest_Decision interface {
	isArtifactTransferRequest_Decision()
}

type ArtifactTransferRequest_UploadGrant struct {
	UploadGrant *ArtifactUploadGrant `protobuf:"bytes,13,opt,name=upload_grant,json=uploadGrant,proto3,oneof"`
}

type ArtifactTransferRequest_Held struct {
	Held *ArtifactObjectRef `protobuf:"bytes,14,opt,name=held,proto3,oneof"`
}

func (*ArtifactTransferRequest_UploadGrant) isArtifactTransferRequest_Decision() {}

func (*ArtifactTransferRequest_Held) isArtifactTransferRequest_Decision() {}

// Durable supervisor -> RecordOwner observation. Intermediate observations are monotonic by
// update_sequence; terminal observations replay after reconnect or exact TransferRequest replay.
type ArtifactTransferStatus struct {
	state                   protoimpl.MessageState `protogen:"open.v1"`
	RecordOwnerEpoch        uint64                 `protobuf:"varint,1,opt,name=record_owner_epoch,json=recordOwnerEpoch,proto3" json:"record_owner_epoch,omitempty"`
	ControlStreamGeneration uint64                 `protobuf:"varint,2,opt,name=control_stream_generation,json=controlStreamGeneration,proto3" json:"control_stream_generation,omitempty"`
	WorkerBootId            string                 `protobuf:"bytes,3,opt,name=worker_boot_id,json=workerBootId,proto3" json:"worker_boot_id,omitempty"`
	RequestId               string                 `protobuf:"bytes,5,opt,name=request_id,json=requestId,proto3" json:"request_id,omitempty"`
	AttemptOrdinal          uint64                 `protobuf:"varint,6,opt,name=attempt_ordinal,json=attemptOrdinal,proto3" json:"attempt_ordinal,omitempty"`
	InvocationSpecDigest    []byte                 `protobuf:"bytes,7,opt,name=invocation_spec_digest,json=invocationSpecDigest,proto3" json:"invocation_spec_digest,omitempty"`
	OutputSlot              string                 `protobuf:"bytes,8,opt,name=output_slot,json=outputSlot,proto3" json:"output_slot,omitempty"`
	ArtifactTransactionId   string                 `protobuf:"bytes,9,opt,name=artifact_transaction_id,json=artifactTransactionId,proto3" json:"artifact_transaction_id,omitempty"`
	OperationId             string                 `protobuf:"bytes,10,opt,name=operation_id,json=operationId,proto3" json:"operation_id,omitempty"`
	GrantRevision           uint64                 `protobuf:"varint,11,opt,name=grant_revision,json=grantRevision,proto3" json:"grant_revision,omitempty"`
	ObjectId                string                 `protobuf:"bytes,12,opt,name=object_id,json=objectId,proto3" json:"object_id,omitempty"`
	Length                  uint64                 `protobuf:"varint,13,opt,name=length,proto3" json:"length,omitempty"`
	UpdateSequence          uint64                 `protobuf:"varint,14,opt,name=update_sequence,json=updateSequence,proto3" json:"update_sequence,omitempty"`
	State                   ArtifactTransferState  `protobuf:"varint,15,opt,name=state,proto3,enum=cozy.worker.v1.ArtifactTransferState" json:"state,omitempty"`
	TransferredBytes        uint64                 `protobuf:"varint,16,opt,name=transferred_bytes,json=transferredBytes,proto3" json:"transferred_bytes,omitempty"`
	HttpStatus              uint32                 `protobuf:"varint,17,opt,name=http_status,json=httpStatus,proto3" json:"http_status,omitempty"`
	Etag                    string                 `protobuf:"bytes,18,opt,name=etag,proto3" json:"etag,omitempty"`
	ChecksumSha256          string                 `protobuf:"bytes,19,opt,name=checksum_sha256,json=checksumSha256,proto3" json:"checksum_sha256,omitempty"`
	Attempts                uint32                 `protobuf:"varint,20,opt,name=attempts,proto3" json:"attempts,omitempty"`
	SafeCode                string                 `protobuf:"bytes,21,opt,name=safe_code,json=safeCode,proto3" json:"safe_code,omitempty"`
	SafeDetail              string                 `protobuf:"bytes,22,opt,name=safe_detail,json=safeDetail,proto3" json:"safe_detail,omitempty"`
	unknownFields           protoimpl.UnknownFields
	sizeCache               protoimpl.SizeCache
}

func (x *ArtifactTransferStatus) Reset() {
	*x = ArtifactTransferStatus{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[63]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *ArtifactTransferStatus) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*ArtifactTransferStatus) ProtoMessage() {}

func (x *ArtifactTransferStatus) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[63]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use ArtifactTransferStatus.ProtoReflect.Descriptor instead.
func (*ArtifactTransferStatus) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{63}
}

func (x *ArtifactTransferStatus) GetRecordOwnerEpoch() uint64 {
	if x != nil {
		return x.RecordOwnerEpoch
	}
	return 0
}

func (x *ArtifactTransferStatus) GetControlStreamGeneration() uint64 {
	if x != nil {
		return x.ControlStreamGeneration
	}
	return 0
}

func (x *ArtifactTransferStatus) GetWorkerBootId() string {
	if x != nil {
		return x.WorkerBootId
	}
	return ""
}

func (x *ArtifactTransferStatus) GetRequestId() string {
	if x != nil {
		return x.RequestId
	}
	return ""
}

func (x *ArtifactTransferStatus) GetAttemptOrdinal() uint64 {
	if x != nil {
		return x.AttemptOrdinal
	}
	return 0
}

func (x *ArtifactTransferStatus) GetInvocationSpecDigest() []byte {
	if x != nil {
		return x.InvocationSpecDigest
	}
	return nil
}

func (x *ArtifactTransferStatus) GetOutputSlot() string {
	if x != nil {
		return x.OutputSlot
	}
	return ""
}

func (x *ArtifactTransferStatus) GetArtifactTransactionId() string {
	if x != nil {
		return x.ArtifactTransactionId
	}
	return ""
}

func (x *ArtifactTransferStatus) GetOperationId() string {
	if x != nil {
		return x.OperationId
	}
	return ""
}

func (x *ArtifactTransferStatus) GetGrantRevision() uint64 {
	if x != nil {
		return x.GrantRevision
	}
	return 0
}

func (x *ArtifactTransferStatus) GetObjectId() string {
	if x != nil {
		return x.ObjectId
	}
	return ""
}

func (x *ArtifactTransferStatus) GetLength() uint64 {
	if x != nil {
		return x.Length
	}
	return 0
}

func (x *ArtifactTransferStatus) GetUpdateSequence() uint64 {
	if x != nil {
		return x.UpdateSequence
	}
	return 0
}

func (x *ArtifactTransferStatus) GetState() ArtifactTransferState {
	if x != nil {
		return x.State
	}
	return ArtifactTransferState_ARTIFACT_TRANSFER_STATE_UNSPECIFIED
}

func (x *ArtifactTransferStatus) GetTransferredBytes() uint64 {
	if x != nil {
		return x.TransferredBytes
	}
	return 0
}

func (x *ArtifactTransferStatus) GetHttpStatus() uint32 {
	if x != nil {
		return x.HttpStatus
	}
	return 0
}

func (x *ArtifactTransferStatus) GetEtag() string {
	if x != nil {
		return x.Etag
	}
	return ""
}

func (x *ArtifactTransferStatus) GetChecksumSha256() string {
	if x != nil {
		return x.ChecksumSha256
	}
	return ""
}

func (x *ArtifactTransferStatus) GetAttempts() uint32 {
	if x != nil {
		return x.Attempts
	}
	return 0
}

func (x *ArtifactTransferStatus) GetSafeCode() string {
	if x != nil {
		return x.SafeCode
	}
	return ""
}

func (x *ArtifactTransferStatus) GetSafeDetail() string {
	if x != nil {
		return x.SafeDetail
	}
	return ""
}

// Creator resolves and pins provider metadata. One selected file capability at a time reaches
// pod-supervisor; Runtime later performs TensorFS's bounded profile/header proof over the verified
// local files. URL is memory-only; the ledger
// stores only stable identity/revision and measured progress. A higher capability_revision may
// refresh access but cannot change member/ObjectRef/source selection. Broad account tokens are not
// representable.
type ModelSourceFileRequest struct {
	state                   protoimpl.MessageState `protogen:"open.v1"`
	RecordOwnerEpoch        uint64                 `protobuf:"varint,1,opt,name=record_owner_epoch,json=recordOwnerEpoch,proto3" json:"record_owner_epoch,omitempty"`
	ControlStreamGeneration uint64                 `protobuf:"varint,2,opt,name=control_stream_generation,json=controlStreamGeneration,proto3" json:"control_stream_generation,omitempty"`
	WorkerBootId            string                 `protobuf:"bytes,3,opt,name=worker_boot_id,json=workerBootId,proto3" json:"worker_boot_id,omitempty"`
	OperationId             string                 `protobuf:"bytes,5,opt,name=operation_id,json=operationId,proto3" json:"operation_id,omitempty"`
	SourceSelectionDigest   []byte                 `protobuf:"bytes,6,opt,name=source_selection_digest,json=sourceSelectionDigest,proto3" json:"source_selection_digest,omitempty"`
	Member                  string                 `protobuf:"bytes,7,opt,name=member,proto3" json:"member,omitempty"`
	ObjectId                string                 `protobuf:"bytes,8,opt,name=object_id,json=objectId,proto3" json:"object_id,omitempty"` // lowercase sha256:<64 hex>
	Length                  uint64                 `protobuf:"varint,9,opt,name=length,proto3" json:"length,omitempty"`
	Provider                ModelSourceProvider    `protobuf:"varint,10,opt,name=provider,proto3,enum=cozy.worker.v1.ModelSourceProvider" json:"provider,omitempty"`
	Url                     string                 `protobuf:"bytes,11,opt,name=url,proto3" json:"url,omitempty"`                                             // memory-only access; never echoed in status/snapshot/loopback
	ExpiresAtUnix           uint64                 `protobuf:"varint,12,opt,name=expires_at_unix,json=expiresAtUnix,proto3" json:"expires_at_unix,omitempty"` // zero only for an immutable public URL
	CapabilityRevision      uint64                 `protobuf:"varint,13,opt,name=capability_revision,json=capabilityRevision,proto3" json:"capability_revision,omitempty"`
	unknownFields           protoimpl.UnknownFields
	sizeCache               protoimpl.SizeCache
}

func (x *ModelSourceFileRequest) Reset() {
	*x = ModelSourceFileRequest{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[64]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *ModelSourceFileRequest) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*ModelSourceFileRequest) ProtoMessage() {}

func (x *ModelSourceFileRequest) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[64]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use ModelSourceFileRequest.ProtoReflect.Descriptor instead.
func (*ModelSourceFileRequest) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{64}
}

func (x *ModelSourceFileRequest) GetRecordOwnerEpoch() uint64 {
	if x != nil {
		return x.RecordOwnerEpoch
	}
	return 0
}

func (x *ModelSourceFileRequest) GetControlStreamGeneration() uint64 {
	if x != nil {
		return x.ControlStreamGeneration
	}
	return 0
}

func (x *ModelSourceFileRequest) GetWorkerBootId() string {
	if x != nil {
		return x.WorkerBootId
	}
	return ""
}

func (x *ModelSourceFileRequest) GetOperationId() string {
	if x != nil {
		return x.OperationId
	}
	return ""
}

func (x *ModelSourceFileRequest) GetSourceSelectionDigest() []byte {
	if x != nil {
		return x.SourceSelectionDigest
	}
	return nil
}

func (x *ModelSourceFileRequest) GetMember() string {
	if x != nil {
		return x.Member
	}
	return ""
}

func (x *ModelSourceFileRequest) GetObjectId() string {
	if x != nil {
		return x.ObjectId
	}
	return ""
}

func (x *ModelSourceFileRequest) GetLength() uint64 {
	if x != nil {
		return x.Length
	}
	return 0
}

func (x *ModelSourceFileRequest) GetProvider() ModelSourceProvider {
	if x != nil {
		return x.Provider
	}
	return ModelSourceProvider_MODEL_SOURCE_PROVIDER_UNSPECIFIED
}

func (x *ModelSourceFileRequest) GetUrl() string {
	if x != nil {
		return x.Url
	}
	return ""
}

func (x *ModelSourceFileRequest) GetExpiresAtUnix() uint64 {
	if x != nil {
		return x.ExpiresAtUnix
	}
	return 0
}

func (x *ModelSourceFileRequest) GetCapabilityRevision() uint64 {
	if x != nil {
		return x.CapabilityRevision
	}
	return 0
}

type ModelSourceFileStatus struct {
	state                   protoimpl.MessageState `protogen:"open.v1"`
	RecordOwnerEpoch        uint64                 `protobuf:"varint,1,opt,name=record_owner_epoch,json=recordOwnerEpoch,proto3" json:"record_owner_epoch,omitempty"`
	ControlStreamGeneration uint64                 `protobuf:"varint,2,opt,name=control_stream_generation,json=controlStreamGeneration,proto3" json:"control_stream_generation,omitempty"`
	WorkerBootId            string                 `protobuf:"bytes,3,opt,name=worker_boot_id,json=workerBootId,proto3" json:"worker_boot_id,omitempty"`
	OperationId             string                 `protobuf:"bytes,5,opt,name=operation_id,json=operationId,proto3" json:"operation_id,omitempty"`
	SourceSelectionDigest   []byte                 `protobuf:"bytes,6,opt,name=source_selection_digest,json=sourceSelectionDigest,proto3" json:"source_selection_digest,omitempty"`
	Member                  string                 `protobuf:"bytes,7,opt,name=member,proto3" json:"member,omitempty"`
	ObjectId                string                 `protobuf:"bytes,8,opt,name=object_id,json=objectId,proto3" json:"object_id,omitempty"`
	Length                  uint64                 `protobuf:"varint,9,opt,name=length,proto3" json:"length,omitempty"`
	CapabilityRevision      uint64                 `protobuf:"varint,10,opt,name=capability_revision,json=capabilityRevision,proto3" json:"capability_revision,omitempty"`
	State                   ModelSourceFileState   `protobuf:"varint,11,opt,name=state,proto3,enum=cozy.worker.v1.ModelSourceFileState" json:"state,omitempty"`
	TransferredBytes        uint64                 `protobuf:"varint,12,opt,name=transferred_bytes,json=transferredBytes,proto3" json:"transferred_bytes,omitempty"`
	Attempts                uint32                 `protobuf:"varint,13,opt,name=attempts,proto3" json:"attempts,omitempty"`
	SafeCode                string                 `protobuf:"bytes,14,opt,name=safe_code,json=safeCode,proto3" json:"safe_code,omitempty"`
	SafeDetail              string                 `protobuf:"bytes,15,opt,name=safe_detail,json=safeDetail,proto3" json:"safe_detail,omitempty"`
	unknownFields           protoimpl.UnknownFields
	sizeCache               protoimpl.SizeCache
}

func (x *ModelSourceFileStatus) Reset() {
	*x = ModelSourceFileStatus{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[65]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *ModelSourceFileStatus) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*ModelSourceFileStatus) ProtoMessage() {}

func (x *ModelSourceFileStatus) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[65]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use ModelSourceFileStatus.ProtoReflect.Descriptor instead.
func (*ModelSourceFileStatus) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{65}
}

func (x *ModelSourceFileStatus) GetRecordOwnerEpoch() uint64 {
	if x != nil {
		return x.RecordOwnerEpoch
	}
	return 0
}

func (x *ModelSourceFileStatus) GetControlStreamGeneration() uint64 {
	if x != nil {
		return x.ControlStreamGeneration
	}
	return 0
}

func (x *ModelSourceFileStatus) GetWorkerBootId() string {
	if x != nil {
		return x.WorkerBootId
	}
	return ""
}

func (x *ModelSourceFileStatus) GetOperationId() string {
	if x != nil {
		return x.OperationId
	}
	return ""
}

func (x *ModelSourceFileStatus) GetSourceSelectionDigest() []byte {
	if x != nil {
		return x.SourceSelectionDigest
	}
	return nil
}

func (x *ModelSourceFileStatus) GetMember() string {
	if x != nil {
		return x.Member
	}
	return ""
}

func (x *ModelSourceFileStatus) GetObjectId() string {
	if x != nil {
		return x.ObjectId
	}
	return ""
}

func (x *ModelSourceFileStatus) GetLength() uint64 {
	if x != nil {
		return x.Length
	}
	return 0
}

func (x *ModelSourceFileStatus) GetCapabilityRevision() uint64 {
	if x != nil {
		return x.CapabilityRevision
	}
	return 0
}

func (x *ModelSourceFileStatus) GetState() ModelSourceFileState {
	if x != nil {
		return x.State
	}
	return ModelSourceFileState_MODEL_SOURCE_FILE_STATE_UNSPECIFIED
}

func (x *ModelSourceFileStatus) GetTransferredBytes() uint64 {
	if x != nil {
		return x.TransferredBytes
	}
	return 0
}

func (x *ModelSourceFileStatus) GetAttempts() uint32 {
	if x != nil {
		return x.Attempts
	}
	return 0
}

func (x *ModelSourceFileStatus) GetSafeCode() string {
	if x != nil {
		return x.SafeCode
	}
	return ""
}

func (x *ModelSourceFileStatus) GetSafeDetail() string {
	if x != nil {
		return x.SafeDetail
	}
	return ""
}

// Once every selected body is verified, Runtime re-proves each descriptor-reviewed profile
// against the downloaded headers and emits exact TensorFS Manifest refs. Pinned source URI and
// declared license are credential-free provenance, not authorization.
type ModelSourcePrepareRequest struct {
	state                   protoimpl.MessageState `protogen:"open.v1"`
	RecordOwnerEpoch        uint64                 `protobuf:"varint,1,opt,name=record_owner_epoch,json=recordOwnerEpoch,proto3" json:"record_owner_epoch,omitempty"`
	ControlStreamGeneration uint64                 `protobuf:"varint,2,opt,name=control_stream_generation,json=controlStreamGeneration,proto3" json:"control_stream_generation,omitempty"`
	WorkerBootId            string                 `protobuf:"bytes,3,opt,name=worker_boot_id,json=workerBootId,proto3" json:"worker_boot_id,omitempty"`
	OperationId             string                 `protobuf:"bytes,5,opt,name=operation_id,json=operationId,proto3" json:"operation_id,omitempty"`
	SourceSelectionDigest   []byte                 `protobuf:"bytes,6,opt,name=source_selection_digest,json=sourceSelectionDigest,proto3" json:"source_selection_digest,omitempty"`
	Profiles                []*ModelSourceProfile  `protobuf:"bytes,7,rep,name=profiles,proto3" json:"profiles,omitempty"`                                      // sorted unique by slot
	SourceUri               string                 `protobuf:"bytes,8,opt,name=source_uri,json=sourceUri,proto3" json:"source_uri,omitempty"`                   // credential-free pinned provenance
	DeclaredLicense         string                 `protobuf:"bytes,9,opt,name=declared_license,json=declaredLicense,proto3" json:"declared_license,omitempty"` // bounded printable provenance; empty means undeclared
	unknownFields           protoimpl.UnknownFields
	sizeCache               protoimpl.SizeCache
}

func (x *ModelSourcePrepareRequest) Reset() {
	*x = ModelSourcePrepareRequest{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[66]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *ModelSourcePrepareRequest) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*ModelSourcePrepareRequest) ProtoMessage() {}

func (x *ModelSourcePrepareRequest) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[66]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use ModelSourcePrepareRequest.ProtoReflect.Descriptor instead.
func (*ModelSourcePrepareRequest) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{66}
}

func (x *ModelSourcePrepareRequest) GetRecordOwnerEpoch() uint64 {
	if x != nil {
		return x.RecordOwnerEpoch
	}
	return 0
}

func (x *ModelSourcePrepareRequest) GetControlStreamGeneration() uint64 {
	if x != nil {
		return x.ControlStreamGeneration
	}
	return 0
}

func (x *ModelSourcePrepareRequest) GetWorkerBootId() string {
	if x != nil {
		return x.WorkerBootId
	}
	return ""
}

func (x *ModelSourcePrepareRequest) GetOperationId() string {
	if x != nil {
		return x.OperationId
	}
	return ""
}

func (x *ModelSourcePrepareRequest) GetSourceSelectionDigest() []byte {
	if x != nil {
		return x.SourceSelectionDigest
	}
	return nil
}

func (x *ModelSourcePrepareRequest) GetProfiles() []*ModelSourceProfile {
	if x != nil {
		return x.Profiles
	}
	return nil
}

func (x *ModelSourcePrepareRequest) GetSourceUri() string {
	if x != nil {
		return x.SourceUri
	}
	return ""
}

func (x *ModelSourcePrepareRequest) GetDeclaredLicense() string {
	if x != nil {
		return x.DeclaredLicense
	}
	return ""
}

type ModelSourcePrepared struct {
	state                   protoimpl.MessageState    `protogen:"open.v1"`
	RecordOwnerEpoch        uint64                    `protobuf:"varint,1,opt,name=record_owner_epoch,json=recordOwnerEpoch,proto3" json:"record_owner_epoch,omitempty"`
	ControlStreamGeneration uint64                    `protobuf:"varint,2,opt,name=control_stream_generation,json=controlStreamGeneration,proto3" json:"control_stream_generation,omitempty"`
	WorkerBootId            string                    `protobuf:"bytes,3,opt,name=worker_boot_id,json=workerBootId,proto3" json:"worker_boot_id,omitempty"`
	OperationId             string                    `protobuf:"bytes,5,opt,name=operation_id,json=operationId,proto3" json:"operation_id,omitempty"`
	SourceSelectionDigest   []byte                    `protobuf:"bytes,6,opt,name=source_selection_digest,json=sourceSelectionDigest,proto3" json:"source_selection_digest,omitempty"`
	Outcome                 ModelSourcePrepareOutcome `protobuf:"varint,7,opt,name=outcome,proto3,enum=cozy.worker.v1.ModelSourcePrepareOutcome" json:"outcome,omitempty"`
	Sources                 []*PreparedModelSource    `protobuf:"bytes,8,rep,name=sources,proto3" json:"sources,omitempty"` // sorted unique by slot; exact requested set on success
	SafeCode                string                    `protobuf:"bytes,9,opt,name=safe_code,json=safeCode,proto3" json:"safe_code,omitempty"`
	SafeDetail              string                    `protobuf:"bytes,10,opt,name=safe_detail,json=safeDetail,proto3" json:"safe_detail,omitempty"`
	unknownFields           protoimpl.UnknownFields
	sizeCache               protoimpl.SizeCache
}

func (x *ModelSourcePrepared) Reset() {
	*x = ModelSourcePrepared{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[67]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *ModelSourcePrepared) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*ModelSourcePrepared) ProtoMessage() {}

func (x *ModelSourcePrepared) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[67]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use ModelSourcePrepared.ProtoReflect.Descriptor instead.
func (*ModelSourcePrepared) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{67}
}

func (x *ModelSourcePrepared) GetRecordOwnerEpoch() uint64 {
	if x != nil {
		return x.RecordOwnerEpoch
	}
	return 0
}

func (x *ModelSourcePrepared) GetControlStreamGeneration() uint64 {
	if x != nil {
		return x.ControlStreamGeneration
	}
	return 0
}

func (x *ModelSourcePrepared) GetWorkerBootId() string {
	if x != nil {
		return x.WorkerBootId
	}
	return ""
}

func (x *ModelSourcePrepared) GetOperationId() string {
	if x != nil {
		return x.OperationId
	}
	return ""
}

func (x *ModelSourcePrepared) GetSourceSelectionDigest() []byte {
	if x != nil {
		return x.SourceSelectionDigest
	}
	return nil
}

func (x *ModelSourcePrepared) GetOutcome() ModelSourcePrepareOutcome {
	if x != nil {
		return x.Outcome
	}
	return ModelSourcePrepareOutcome_MODEL_SOURCE_PREPARE_OUTCOME_UNSPECIFIED
}

func (x *ModelSourcePrepared) GetSources() []*PreparedModelSource {
	if x != nil {
		return x.Sources
	}
	return nil
}

func (x *ModelSourcePrepared) GetSafeCode() string {
	if x != nil {
		return x.SafeCode
	}
	return ""
}

func (x *ModelSourcePrepared) GetSafeDetail() string {
	if x != nil {
		return x.SafeDetail
	}
	return ""
}

// Durable owner->worker request. This is a typed protobuf frame, not a canonical document. The
// record owner journals the decision fields before send; Runtime journals the result fields
// before reply. Replay keys on the request tuple and compares the typed fields.
type ArtifactFinalizeRequest struct {
	state                   protoimpl.MessageState      `protogen:"open.v1"`
	RecordOwnerEpoch        uint64                      `protobuf:"varint,1,opt,name=record_owner_epoch,json=recordOwnerEpoch,proto3" json:"record_owner_epoch,omitempty"`
	ControlStreamGeneration uint64                      `protobuf:"varint,2,opt,name=control_stream_generation,json=controlStreamGeneration,proto3" json:"control_stream_generation,omitempty"`
	WorkerBootId            string                      `protobuf:"bytes,3,opt,name=worker_boot_id,json=workerBootId,proto3" json:"worker_boot_id,omitempty"`
	RequestId               string                      `protobuf:"bytes,5,opt,name=request_id,json=requestId,proto3" json:"request_id,omitempty"`
	InvocationSpecDigest    []byte                      `protobuf:"bytes,6,opt,name=invocation_spec_digest,json=invocationSpecDigest,proto3" json:"invocation_spec_digest,omitempty"`
	OutputSlot              string                      `protobuf:"bytes,7,opt,name=output_slot,json=outputSlot,proto3" json:"output_slot,omitempty"`
	Disposition             ArtifactFinalizeDisposition `protobuf:"varint,8,opt,name=disposition,proto3,enum=cozy.worker.v1.ArtifactFinalizeDisposition" json:"disposition,omitempty"`
	ArtifactReceiptDigest   []byte                      `protobuf:"bytes,9,opt,name=artifact_receipt_digest,json=artifactReceiptDigest,proto3" json:"artifact_receipt_digest,omitempty"`
	ScratchRootId           string                      `protobuf:"bytes,10,opt,name=scratch_root_id,json=scratchRootId,proto3" json:"scratch_root_id,omitempty"`
	OwnerAuthorityScope     string                      `protobuf:"bytes,11,opt,name=owner_authority_scope,json=ownerAuthorityScope,proto3" json:"owner_authority_scope,omitempty"`
	unknownFields           protoimpl.UnknownFields
	sizeCache               protoimpl.SizeCache
}

func (x *ArtifactFinalizeRequest) Reset() {
	*x = ArtifactFinalizeRequest{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[68]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *ArtifactFinalizeRequest) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*ArtifactFinalizeRequest) ProtoMessage() {}

func (x *ArtifactFinalizeRequest) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[68]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use ArtifactFinalizeRequest.ProtoReflect.Descriptor instead.
func (*ArtifactFinalizeRequest) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{68}
}

func (x *ArtifactFinalizeRequest) GetRecordOwnerEpoch() uint64 {
	if x != nil {
		return x.RecordOwnerEpoch
	}
	return 0
}

func (x *ArtifactFinalizeRequest) GetControlStreamGeneration() uint64 {
	if x != nil {
		return x.ControlStreamGeneration
	}
	return 0
}

func (x *ArtifactFinalizeRequest) GetWorkerBootId() string {
	if x != nil {
		return x.WorkerBootId
	}
	return ""
}

func (x *ArtifactFinalizeRequest) GetRequestId() string {
	if x != nil {
		return x.RequestId
	}
	return ""
}

func (x *ArtifactFinalizeRequest) GetInvocationSpecDigest() []byte {
	if x != nil {
		return x.InvocationSpecDigest
	}
	return nil
}

func (x *ArtifactFinalizeRequest) GetOutputSlot() string {
	if x != nil {
		return x.OutputSlot
	}
	return ""
}

func (x *ArtifactFinalizeRequest) GetDisposition() ArtifactFinalizeDisposition {
	if x != nil {
		return x.Disposition
	}
	return ArtifactFinalizeDisposition_ARTIFACT_FINALIZE_DISPOSITION_UNSPECIFIED
}

func (x *ArtifactFinalizeRequest) GetArtifactReceiptDigest() []byte {
	if x != nil {
		return x.ArtifactReceiptDigest
	}
	return nil
}

func (x *ArtifactFinalizeRequest) GetScratchRootId() string {
	if x != nil {
		return x.ScratchRootId
	}
	return ""
}

func (x *ArtifactFinalizeRequest) GetOwnerAuthorityScope() string {
	if x != nil {
		return x.OwnerAuthorityScope
	}
	return ""
}

// Durable worker->owner result. A commit-first ABANDON_UNCOMMITTED race carries the released
// receipt as evidence; other outcomes may omit it. No extra acknowledgement exists.
type ArtifactFinalizeResult struct {
	state                   protoimpl.MessageState  `protogen:"open.v1"`
	RecordOwnerEpoch        uint64                  `protobuf:"varint,1,opt,name=record_owner_epoch,json=recordOwnerEpoch,proto3" json:"record_owner_epoch,omitempty"`
	ControlStreamGeneration uint64                  `protobuf:"varint,2,opt,name=control_stream_generation,json=controlStreamGeneration,proto3" json:"control_stream_generation,omitempty"`
	WorkerBootId            string                  `protobuf:"bytes,3,opt,name=worker_boot_id,json=workerBootId,proto3" json:"worker_boot_id,omitempty"`
	RequestId               string                  `protobuf:"bytes,5,opt,name=request_id,json=requestId,proto3" json:"request_id,omitempty"`
	InvocationSpecDigest    []byte                  `protobuf:"bytes,6,opt,name=invocation_spec_digest,json=invocationSpecDigest,proto3" json:"invocation_spec_digest,omitempty"`
	OutputSlot              string                  `protobuf:"bytes,7,opt,name=output_slot,json=outputSlot,proto3" json:"output_slot,omitempty"`
	Outcome                 ArtifactFinalizeOutcome `protobuf:"varint,8,opt,name=outcome,proto3,enum=cozy.worker.v1.ArtifactFinalizeOutcome" json:"outcome,omitempty"`
	ArtifactReceipt         *ArtifactReceiptRef     `protobuf:"bytes,9,opt,name=artifact_receipt,json=artifactReceipt,proto3" json:"artifact_receipt,omitempty"`
	OwnerAuthorityScope     string                  `protobuf:"bytes,10,opt,name=owner_authority_scope,json=ownerAuthorityScope,proto3" json:"owner_authority_scope,omitempty"`
	unknownFields           protoimpl.UnknownFields
	sizeCache               protoimpl.SizeCache
}

func (x *ArtifactFinalizeResult) Reset() {
	*x = ArtifactFinalizeResult{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[69]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *ArtifactFinalizeResult) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*ArtifactFinalizeResult) ProtoMessage() {}

func (x *ArtifactFinalizeResult) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[69]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use ArtifactFinalizeResult.ProtoReflect.Descriptor instead.
func (*ArtifactFinalizeResult) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{69}
}

func (x *ArtifactFinalizeResult) GetRecordOwnerEpoch() uint64 {
	if x != nil {
		return x.RecordOwnerEpoch
	}
	return 0
}

func (x *ArtifactFinalizeResult) GetControlStreamGeneration() uint64 {
	if x != nil {
		return x.ControlStreamGeneration
	}
	return 0
}

func (x *ArtifactFinalizeResult) GetWorkerBootId() string {
	if x != nil {
		return x.WorkerBootId
	}
	return ""
}

func (x *ArtifactFinalizeResult) GetRequestId() string {
	if x != nil {
		return x.RequestId
	}
	return ""
}

func (x *ArtifactFinalizeResult) GetInvocationSpecDigest() []byte {
	if x != nil {
		return x.InvocationSpecDigest
	}
	return nil
}

func (x *ArtifactFinalizeResult) GetOutputSlot() string {
	if x != nil {
		return x.OutputSlot
	}
	return ""
}

func (x *ArtifactFinalizeResult) GetOutcome() ArtifactFinalizeOutcome {
	if x != nil {
		return x.Outcome
	}
	return ArtifactFinalizeOutcome_ARTIFACT_FINALIZE_OUTCOME_UNSPECIFIED
}

func (x *ArtifactFinalizeResult) GetArtifactReceipt() *ArtifactReceiptRef {
	if x != nil {
		return x.ArtifactReceipt
	}
	return nil
}

func (x *ArtifactFinalizeResult) GetOwnerAuthorityScope() string {
	if x != nil {
		return x.OwnerAuthorityScope
	}
	return ""
}

type ResultEnvelope struct {
	state protoimpl.MessageState `protogen:"open.v1"`
	// lands with cr-010 (post-launch), keyed by
	// resolved_adapter_plan_id
	ResultSchemaDigest []byte           `protobuf:"bytes,1,opt,name=result_schema_digest,json=resultSchemaDigest,proto3" json:"result_schema_digest,omitempty"`
	InlineResult       []byte           `protobuf:"bytes,2,opt,name=inline_result,json=inlineResult,proto3" json:"inline_result,omitempty"` // canonical typed result bytes, <= 4 MiB; XOR with 3
	ResultBlob         *OutputEntry     `protobuf:"bytes,3,opt,name=result_blob,json=resultBlob,proto3" json:"result_blob,omitempty"`       // by immutable blob receipt when large; XOR with 2
	Adjustments        []*AdjustmentRow `protobuf:"bytes,4,rep,name=adjustments,proto3" json:"adjustments,omitempty"`
	CheckpointRef      string           `protobuf:"bytes,5,opt,name=checkpoint_ref,json=checkpointRef,proto3" json:"checkpoint_ref,omitempty"`
	unknownFields      protoimpl.UnknownFields
	sizeCache          protoimpl.SizeCache
}

func (x *ResultEnvelope) Reset() {
	*x = ResultEnvelope{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[70]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *ResultEnvelope) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*ResultEnvelope) ProtoMessage() {}

func (x *ResultEnvelope) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[70]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use ResultEnvelope.ProtoReflect.Descriptor instead.
func (*ResultEnvelope) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{70}
}

func (x *ResultEnvelope) GetResultSchemaDigest() []byte {
	if x != nil {
		return x.ResultSchemaDigest
	}
	return nil
}

func (x *ResultEnvelope) GetInlineResult() []byte {
	if x != nil {
		return x.InlineResult
	}
	return nil
}

func (x *ResultEnvelope) GetResultBlob() *OutputEntry {
	if x != nil {
		return x.ResultBlob
	}
	return nil
}

func (x *ResultEnvelope) GetAdjustments() []*AdjustmentRow {
	if x != nil {
		return x.Adjustments
	}
	return nil
}

func (x *ResultEnvelope) GetCheckpointRef() string {
	if x != nil {
		return x.CheckpointRef
	}
	return ""
}

type AdjustmentRow struct {
	state         protoimpl.MessageState `protogen:"open.v1"`
	Field         string                 `protobuf:"bytes,1,opt,name=field,proto3" json:"field,omitempty"`
	Requested     string                 `protobuf:"bytes,2,opt,name=requested,proto3" json:"requested,omitempty"`
	Applied       string                 `protobuf:"bytes,3,opt,name=applied,proto3" json:"applied,omitempty"`
	Reason        string                 `protobuf:"bytes,4,opt,name=reason,proto3" json:"reason,omitempty"` // adjusted | clamp | warn, plus bounded detail
	unknownFields protoimpl.UnknownFields
	sizeCache     protoimpl.SizeCache
}

func (x *AdjustmentRow) Reset() {
	*x = AdjustmentRow{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[71]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *AdjustmentRow) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*AdjustmentRow) ProtoMessage() {}

func (x *AdjustmentRow) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[71]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use AdjustmentRow.ProtoReflect.Descriptor instead.
func (*AdjustmentRow) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{71}
}

func (x *AdjustmentRow) GetField() string {
	if x != nil {
		return x.Field
	}
	return ""
}

func (x *AdjustmentRow) GetRequested() string {
	if x != nil {
		return x.Requested
	}
	return ""
}

func (x *AdjustmentRow) GetApplied() string {
	if x != nil {
		return x.Applied
	}
	return ""
}

func (x *AdjustmentRow) GetReason() string {
	if x != nil {
		return x.Reason
	}
	return ""
}

type OutcomeCause struct {
	state         protoimpl.MessageState `protogen:"open.v1"`
	Code          CauseCode              `protobuf:"varint,1,opt,name=code,proto3,enum=cozy.worker.v1.CauseCode" json:"code,omitempty"`
	Origin        CauseOrigin            `protobuf:"varint,2,opt,name=origin,proto3,enum=cozy.worker.v1.CauseOrigin" json:"origin,omitempty"`
	Detail        string                 `protobuf:"bytes,3,opt,name=detail,proto3" json:"detail,omitempty"` // bounded <= 1024 bytes, sanitized
	Shortfall     *ResourceShortfall     `protobuf:"bytes,4,opt,name=shortfall,proto3" json:"shortfall,omitempty"`
	unknownFields protoimpl.UnknownFields
	sizeCache     protoimpl.SizeCache
}

func (x *OutcomeCause) Reset() {
	*x = OutcomeCause{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[72]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *OutcomeCause) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*OutcomeCause) ProtoMessage() {}

func (x *OutcomeCause) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[72]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use OutcomeCause.ProtoReflect.Descriptor instead.
func (*OutcomeCause) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{72}
}

func (x *OutcomeCause) GetCode() CauseCode {
	if x != nil {
		return x.Code
	}
	return CauseCode_CAUSE_CODE_UNSPECIFIED
}

func (x *OutcomeCause) GetOrigin() CauseOrigin {
	if x != nil {
		return x.Origin
	}
	return CauseOrigin_CAUSE_ORIGIN_UNSPECIFIED
}

func (x *OutcomeCause) GetDetail() string {
	if x != nil {
		return x.Detail
	}
	return ""
}

func (x *OutcomeCause) GetShortfall() *ResourceShortfall {
	if x != nil {
		return x.Shortfall
	}
	return nil
}

type ResourceShortfall struct {
	state          protoimpl.MessageState `protogen:"open.v1"`
	Resource       string                 `protobuf:"bytes,1,opt,name=resource,proto3" json:"resource,omitempty"` // device_memory | host_ram | pinned | disk | ...
	Scope          string                 `protobuf:"bytes,2,opt,name=scope,proto3" json:"scope,omitempty"`
	NeededBytes    uint64                 `protobuf:"varint,3,opt,name=needed_bytes,json=neededBytes,proto3" json:"needed_bytes,omitempty"`
	AvailableBytes uint64                 `protobuf:"varint,4,opt,name=available_bytes,json=availableBytes,proto3" json:"available_bytes,omitempty"`
	RequestShape   string                 `protobuf:"bytes,5,opt,name=request_shape,json=requestShape,proto3" json:"request_shape,omitempty"`
	EvidenceClass  string                 `protobuf:"bytes,6,opt,name=evidence_class,json=evidenceClass,proto3" json:"evidence_class,omitempty"` // measured | fitted | prior
	unknownFields  protoimpl.UnknownFields
	sizeCache      protoimpl.SizeCache
}

func (x *ResourceShortfall) Reset() {
	*x = ResourceShortfall{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[73]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *ResourceShortfall) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*ResourceShortfall) ProtoMessage() {}

func (x *ResourceShortfall) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[73]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use ResourceShortfall.ProtoReflect.Descriptor instead.
func (*ResourceShortfall) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{73}
}

func (x *ResourceShortfall) GetResource() string {
	if x != nil {
		return x.Resource
	}
	return ""
}

func (x *ResourceShortfall) GetScope() string {
	if x != nil {
		return x.Scope
	}
	return ""
}

func (x *ResourceShortfall) GetNeededBytes() uint64 {
	if x != nil {
		return x.NeededBytes
	}
	return 0
}

func (x *ResourceShortfall) GetAvailableBytes() uint64 {
	if x != nil {
		return x.AvailableBytes
	}
	return 0
}

func (x *ResourceShortfall) GetRequestShape() string {
	if x != nil {
		return x.RequestShape
	}
	return ""
}

func (x *ResourceShortfall) GetEvidenceClass() string {
	if x != nil {
		return x.EvidenceClass
	}
	return ""
}

type AttemptOutcomeAck struct {
	state                   protoimpl.MessageState `protogen:"open.v1"`
	RecordOwnerEpoch        uint64                 `protobuf:"varint,1,opt,name=record_owner_epoch,json=recordOwnerEpoch,proto3" json:"record_owner_epoch,omitempty"`
	ControlStreamGeneration uint64                 `protobuf:"varint,2,opt,name=control_stream_generation,json=controlStreamGeneration,proto3" json:"control_stream_generation,omitempty"`
	WorkerBootId            string                 `protobuf:"bytes,3,opt,name=worker_boot_id,json=workerBootId,proto3" json:"worker_boot_id,omitempty"`
	RequestId               string                 `protobuf:"bytes,5,opt,name=request_id,json=requestId,proto3" json:"request_id,omitempty"`
	AttemptOrdinal          uint64                 `protobuf:"varint,6,opt,name=attempt_ordinal,json=attemptOrdinal,proto3" json:"attempt_ordinal,omitempty"`
	InvocationSpecDigest    []byte                 `protobuf:"bytes,7,opt,name=invocation_spec_digest,json=invocationSpecDigest,proto3" json:"invocation_spec_digest,omitempty"`
	OutcomeId               string                 `protobuf:"bytes,8,opt,name=outcome_id,json=outcomeId,proto3" json:"outcome_id,omitempty"`             // echo; a mismatched ack is NOT an ack — replay continues
	OutcomeDigest           []byte                 `protobuf:"bytes,9,opt,name=outcome_digest,json=outcomeDigest,proto3" json:"outcome_digest,omitempty"` // echo; compared, never recomputed
	unknownFields           protoimpl.UnknownFields
	sizeCache               protoimpl.SizeCache
}

func (x *AttemptOutcomeAck) Reset() {
	*x = AttemptOutcomeAck{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[74]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *AttemptOutcomeAck) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*AttemptOutcomeAck) ProtoMessage() {}

func (x *AttemptOutcomeAck) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[74]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use AttemptOutcomeAck.ProtoReflect.Descriptor instead.
func (*AttemptOutcomeAck) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{74}
}

func (x *AttemptOutcomeAck) GetRecordOwnerEpoch() uint64 {
	if x != nil {
		return x.RecordOwnerEpoch
	}
	return 0
}

func (x *AttemptOutcomeAck) GetControlStreamGeneration() uint64 {
	if x != nil {
		return x.ControlStreamGeneration
	}
	return 0
}

func (x *AttemptOutcomeAck) GetWorkerBootId() string {
	if x != nil {
		return x.WorkerBootId
	}
	return ""
}

func (x *AttemptOutcomeAck) GetRequestId() string {
	if x != nil {
		return x.RequestId
	}
	return ""
}

func (x *AttemptOutcomeAck) GetAttemptOrdinal() uint64 {
	if x != nil {
		return x.AttemptOrdinal
	}
	return 0
}

func (x *AttemptOutcomeAck) GetInvocationSpecDigest() []byte {
	if x != nil {
		return x.InvocationSpecDigest
	}
	return nil
}

func (x *AttemptOutcomeAck) GetOutcomeId() string {
	if x != nil {
		return x.OutcomeId
	}
	return ""
}

func (x *AttemptOutcomeAck) GetOutcomeDigest() []byte {
	if x != nil {
		return x.OutcomeDigest
	}
	return nil
}

type JobCheckpointRequest struct {
	state                   protoimpl.MessageState `protogen:"open.v1"`
	RecordOwnerEpoch        uint64                 `protobuf:"varint,1,opt,name=record_owner_epoch,json=recordOwnerEpoch,proto3" json:"record_owner_epoch,omitempty"`
	ControlStreamGeneration uint64                 `protobuf:"varint,2,opt,name=control_stream_generation,json=controlStreamGeneration,proto3" json:"control_stream_generation,omitempty"`
	WorkerBootId            string                 `protobuf:"bytes,3,opt,name=worker_boot_id,json=workerBootId,proto3" json:"worker_boot_id,omitempty"`
	RequestId               string                 `protobuf:"bytes,5,opt,name=request_id,json=requestId,proto3" json:"request_id,omitempty"`
	AttemptOrdinal          uint64                 `protobuf:"varint,6,opt,name=attempt_ordinal,json=attemptOrdinal,proto3" json:"attempt_ordinal,omitempty"`
	OperationKey            string                 `protobuf:"bytes,7,opt,name=operation_key,json=operationKey,proto3" json:"operation_key,omitempty"`    // worker-minted, attempt-scoped idempotency key
	LogicalKey              string                 `protobuf:"bytes,8,opt,name=logical_key,json=logicalKey,proto3" json:"logical_key,omitempty"`          // author-stable checkpoint slot
	ContentDigest           []byte                 `protobuf:"bytes,9,opt,name=content_digest,json=contentDigest,proto3" json:"content_digest,omitempty"` // class (b): the checkpoint artifact's bytes
	Seq                     uint64                 `protobuf:"varint,10,opt,name=seq,proto3" json:"seq,omitempty"`                                        // ordering only, never identity
	Artifact                *OutputEntry           `protobuf:"bytes,11,opt,name=artifact,proto3" json:"artifact,omitempty"`
	unknownFields           protoimpl.UnknownFields
	sizeCache               protoimpl.SizeCache
}

func (x *JobCheckpointRequest) Reset() {
	*x = JobCheckpointRequest{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[75]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *JobCheckpointRequest) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*JobCheckpointRequest) ProtoMessage() {}

func (x *JobCheckpointRequest) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[75]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use JobCheckpointRequest.ProtoReflect.Descriptor instead.
func (*JobCheckpointRequest) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{75}
}

func (x *JobCheckpointRequest) GetRecordOwnerEpoch() uint64 {
	if x != nil {
		return x.RecordOwnerEpoch
	}
	return 0
}

func (x *JobCheckpointRequest) GetControlStreamGeneration() uint64 {
	if x != nil {
		return x.ControlStreamGeneration
	}
	return 0
}

func (x *JobCheckpointRequest) GetWorkerBootId() string {
	if x != nil {
		return x.WorkerBootId
	}
	return ""
}

func (x *JobCheckpointRequest) GetRequestId() string {
	if x != nil {
		return x.RequestId
	}
	return ""
}

func (x *JobCheckpointRequest) GetAttemptOrdinal() uint64 {
	if x != nil {
		return x.AttemptOrdinal
	}
	return 0
}

func (x *JobCheckpointRequest) GetOperationKey() string {
	if x != nil {
		return x.OperationKey
	}
	return ""
}

func (x *JobCheckpointRequest) GetLogicalKey() string {
	if x != nil {
		return x.LogicalKey
	}
	return ""
}

func (x *JobCheckpointRequest) GetContentDigest() []byte {
	if x != nil {
		return x.ContentDigest
	}
	return nil
}

func (x *JobCheckpointRequest) GetSeq() uint64 {
	if x != nil {
		return x.Seq
	}
	return 0
}

func (x *JobCheckpointRequest) GetArtifact() *OutputEntry {
	if x != nil {
		return x.Artifact
	}
	return nil
}

type JobCheckpointReceipt struct {
	state                   protoimpl.MessageState `protogen:"open.v1"`
	RecordOwnerEpoch        uint64                 `protobuf:"varint,1,opt,name=record_owner_epoch,json=recordOwnerEpoch,proto3" json:"record_owner_epoch,omitempty"`
	ControlStreamGeneration uint64                 `protobuf:"varint,2,opt,name=control_stream_generation,json=controlStreamGeneration,proto3" json:"control_stream_generation,omitempty"`
	WorkerBootId            string                 `protobuf:"bytes,3,opt,name=worker_boot_id,json=workerBootId,proto3" json:"worker_boot_id,omitempty"`
	RequestId               string                 `protobuf:"bytes,5,opt,name=request_id,json=requestId,proto3" json:"request_id,omitempty"`
	AttemptOrdinal          uint64                 `protobuf:"varint,6,opt,name=attempt_ordinal,json=attemptOrdinal,proto3" json:"attempt_ordinal,omitempty"`
	OperationKey            string                 `protobuf:"bytes,7,opt,name=operation_key,json=operationKey,proto3" json:"operation_key,omitempty"`    // echo
	LogicalKey              string                 `protobuf:"bytes,8,opt,name=logical_key,json=logicalKey,proto3" json:"logical_key,omitempty"`          // echo
	ContentDigest           []byte                 `protobuf:"bytes,9,opt,name=content_digest,json=contentDigest,proto3" json:"content_digest,omitempty"` // echo; mismatch vs recorded identity is a conflict
	ReceiptId               string                 `protobuf:"bytes,10,opt,name=receipt_id,json=receiptId,proto3" json:"receipt_id,omitempty"`            // RecordOwner-minted durable record id
	Outcome                 CheckpointOutcome      `protobuf:"varint,11,opt,name=outcome,proto3,enum=cozy.worker.v1.CheckpointOutcome" json:"outcome,omitempty"`
	Fault                   *CheckpointFault       `protobuf:"bytes,12,opt,name=fault,proto3" json:"fault,omitempty"` // set iff CONFLICT or REFUSED
	unknownFields           protoimpl.UnknownFields
	sizeCache               protoimpl.SizeCache
}

func (x *JobCheckpointReceipt) Reset() {
	*x = JobCheckpointReceipt{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[76]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *JobCheckpointReceipt) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*JobCheckpointReceipt) ProtoMessage() {}

func (x *JobCheckpointReceipt) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[76]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use JobCheckpointReceipt.ProtoReflect.Descriptor instead.
func (*JobCheckpointReceipt) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{76}
}

func (x *JobCheckpointReceipt) GetRecordOwnerEpoch() uint64 {
	if x != nil {
		return x.RecordOwnerEpoch
	}
	return 0
}

func (x *JobCheckpointReceipt) GetControlStreamGeneration() uint64 {
	if x != nil {
		return x.ControlStreamGeneration
	}
	return 0
}

func (x *JobCheckpointReceipt) GetWorkerBootId() string {
	if x != nil {
		return x.WorkerBootId
	}
	return ""
}

func (x *JobCheckpointReceipt) GetRequestId() string {
	if x != nil {
		return x.RequestId
	}
	return ""
}

func (x *JobCheckpointReceipt) GetAttemptOrdinal() uint64 {
	if x != nil {
		return x.AttemptOrdinal
	}
	return 0
}

func (x *JobCheckpointReceipt) GetOperationKey() string {
	if x != nil {
		return x.OperationKey
	}
	return ""
}

func (x *JobCheckpointReceipt) GetLogicalKey() string {
	if x != nil {
		return x.LogicalKey
	}
	return ""
}

func (x *JobCheckpointReceipt) GetContentDigest() []byte {
	if x != nil {
		return x.ContentDigest
	}
	return nil
}

func (x *JobCheckpointReceipt) GetReceiptId() string {
	if x != nil {
		return x.ReceiptId
	}
	return ""
}

func (x *JobCheckpointReceipt) GetOutcome() CheckpointOutcome {
	if x != nil {
		return x.Outcome
	}
	return CheckpointOutcome_CHECKPOINT_OUTCOME_UNSPECIFIED
}

func (x *JobCheckpointReceipt) GetFault() *CheckpointFault {
	if x != nil {
		return x.Fault
	}
	return nil
}

type CheckpointFault struct {
	state                 protoimpl.MessageState `protogen:"open.v1"`
	Code                  CheckpointFaultCode    `protobuf:"varint,1,opt,name=code,proto3,enum=cozy.worker.v1.CheckpointFaultCode" json:"code,omitempty"`
	Detail                string                 `protobuf:"bytes,2,opt,name=detail,proto3" json:"detail,omitempty"` // bounded <= 1024 bytes
	RecordedContentDigest []byte                 `protobuf:"bytes,3,opt,name=recorded_content_digest,json=recordedContentDigest,proto3" json:"recorded_content_digest,omitempty"`
	unknownFields         protoimpl.UnknownFields
	sizeCache             protoimpl.SizeCache
}

func (x *CheckpointFault) Reset() {
	*x = CheckpointFault{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[77]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *CheckpointFault) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*CheckpointFault) ProtoMessage() {}

func (x *CheckpointFault) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[77]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use CheckpointFault.ProtoReflect.Descriptor instead.
func (*CheckpointFault) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{77}
}

func (x *CheckpointFault) GetCode() CheckpointFaultCode {
	if x != nil {
		return x.Code
	}
	return CheckpointFaultCode_CHECKPOINT_FAULT_CODE_UNSPECIFIED
}

func (x *CheckpointFault) GetDetail() string {
	if x != nil {
		return x.Detail
	}
	return ""
}

func (x *CheckpointFault) GetRecordedContentDigest() []byte {
	if x != nil {
		return x.RecordedContentDigest
	}
	return nil
}

type JobCheckpointAck struct {
	state                   protoimpl.MessageState `protogen:"open.v1"`
	RecordOwnerEpoch        uint64                 `protobuf:"varint,1,opt,name=record_owner_epoch,json=recordOwnerEpoch,proto3" json:"record_owner_epoch,omitempty"`
	ControlStreamGeneration uint64                 `protobuf:"varint,2,opt,name=control_stream_generation,json=controlStreamGeneration,proto3" json:"control_stream_generation,omitempty"`
	WorkerBootId            string                 `protobuf:"bytes,3,opt,name=worker_boot_id,json=workerBootId,proto3" json:"worker_boot_id,omitempty"`
	RequestId               string                 `protobuf:"bytes,5,opt,name=request_id,json=requestId,proto3" json:"request_id,omitempty"`
	AttemptOrdinal          uint64                 `protobuf:"varint,6,opt,name=attempt_ordinal,json=attemptOrdinal,proto3" json:"attempt_ordinal,omitempty"`
	OperationKey            string                 `protobuf:"bytes,7,opt,name=operation_key,json=operationKey,proto3" json:"operation_key,omitempty"`
	ReceiptId               string                 `protobuf:"bytes,8,opt,name=receipt_id,json=receiptId,proto3" json:"receipt_id,omitempty"` // echo; a mismatched ack is not an ack
	ContentDigest           []byte                 `protobuf:"bytes,9,opt,name=content_digest,json=contentDigest,proto3" json:"content_digest,omitempty"`
	unknownFields           protoimpl.UnknownFields
	sizeCache               protoimpl.SizeCache
}

func (x *JobCheckpointAck) Reset() {
	*x = JobCheckpointAck{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[78]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *JobCheckpointAck) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*JobCheckpointAck) ProtoMessage() {}

func (x *JobCheckpointAck) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[78]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use JobCheckpointAck.ProtoReflect.Descriptor instead.
func (*JobCheckpointAck) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{78}
}

func (x *JobCheckpointAck) GetRecordOwnerEpoch() uint64 {
	if x != nil {
		return x.RecordOwnerEpoch
	}
	return 0
}

func (x *JobCheckpointAck) GetControlStreamGeneration() uint64 {
	if x != nil {
		return x.ControlStreamGeneration
	}
	return 0
}

func (x *JobCheckpointAck) GetWorkerBootId() string {
	if x != nil {
		return x.WorkerBootId
	}
	return ""
}

func (x *JobCheckpointAck) GetRequestId() string {
	if x != nil {
		return x.RequestId
	}
	return ""
}

func (x *JobCheckpointAck) GetAttemptOrdinal() uint64 {
	if x != nil {
		return x.AttemptOrdinal
	}
	return 0
}

func (x *JobCheckpointAck) GetOperationKey() string {
	if x != nil {
		return x.OperationKey
	}
	return ""
}

func (x *JobCheckpointAck) GetReceiptId() string {
	if x != nil {
		return x.ReceiptId
	}
	return ""
}

func (x *JobCheckpointAck) GetContentDigest() []byte {
	if x != nil {
		return x.ContentDigest
	}
	return nil
}

type ProgressOpen struct {
	state                   protoimpl.MessageState `protogen:"open.v1"`
	RecordOwnerEpoch        uint64                 `protobuf:"varint,1,opt,name=record_owner_epoch,json=recordOwnerEpoch,proto3" json:"record_owner_epoch,omitempty"`
	ControlStreamGeneration uint64                 `protobuf:"varint,2,opt,name=control_stream_generation,json=controlStreamGeneration,proto3" json:"control_stream_generation,omitempty"` // binds the watch to the fenced control stream
	WorkerBootId            string                 `protobuf:"bytes,3,opt,name=worker_boot_id,json=workerBootId,proto3" json:"worker_boot_id,omitempty"`
	RequestId               string                 `protobuf:"bytes,5,opt,name=request_id,json=requestId,proto3" json:"request_id,omitempty"` // empty = all attempts on this stream
	unknownFields           protoimpl.UnknownFields
	sizeCache               protoimpl.SizeCache
}

func (x *ProgressOpen) Reset() {
	*x = ProgressOpen{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[79]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *ProgressOpen) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*ProgressOpen) ProtoMessage() {}

func (x *ProgressOpen) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[79]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use ProgressOpen.ProtoReflect.Descriptor instead.
func (*ProgressOpen) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{79}
}

func (x *ProgressOpen) GetRecordOwnerEpoch() uint64 {
	if x != nil {
		return x.RecordOwnerEpoch
	}
	return 0
}

func (x *ProgressOpen) GetControlStreamGeneration() uint64 {
	if x != nil {
		return x.ControlStreamGeneration
	}
	return 0
}

func (x *ProgressOpen) GetWorkerBootId() string {
	if x != nil {
		return x.WorkerBootId
	}
	return ""
}

func (x *ProgressOpen) GetRequestId() string {
	if x != nil {
		return x.RequestId
	}
	return ""
}

type AttemptProgress struct {
	state                   protoimpl.MessageState `protogen:"open.v1"`
	RecordOwnerEpoch        uint64                 `protobuf:"varint,1,opt,name=record_owner_epoch,json=recordOwnerEpoch,proto3" json:"record_owner_epoch,omitempty"`
	ControlStreamGeneration uint64                 `protobuf:"varint,2,opt,name=control_stream_generation,json=controlStreamGeneration,proto3" json:"control_stream_generation,omitempty"`
	WorkerBootId            string                 `protobuf:"bytes,3,opt,name=worker_boot_id,json=workerBootId,proto3" json:"worker_boot_id,omitempty"`
	RequestId               string                 `protobuf:"bytes,5,opt,name=request_id,json=requestId,proto3" json:"request_id,omitempty"`
	AttemptOrdinal          uint64                 `protobuf:"varint,6,opt,name=attempt_ordinal,json=attemptOrdinal,proto3" json:"attempt_ordinal,omitempty"`
	Seq                     uint64                 `protobuf:"varint,7,opt,name=seq,proto3" json:"seq,omitempty"` // starts at 1, strictly increasing per (request_id,
	// attempt_ordinal); gaps are VISIBLE and the only loss
	// allowed
	ContentType   string `protobuf:"bytes,8,opt,name=content_type,json=contentType,proto3" json:"content_type,omitempty"`
	Data          []byte `protobuf:"bytes,9,opt,name=data,proto3" json:"data,omitempty"`                                   // bounded <= 65536 bytes; NEVER an outcome
	PlacementId   string `protobuf:"bytes,10,opt,name=placement_id,json=placementId,proto3" json:"placement_id,omitempty"` // routing
	unknownFields protoimpl.UnknownFields
	sizeCache     protoimpl.SizeCache
}

func (x *AttemptProgress) Reset() {
	*x = AttemptProgress{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[80]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *AttemptProgress) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*AttemptProgress) ProtoMessage() {}

func (x *AttemptProgress) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[80]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use AttemptProgress.ProtoReflect.Descriptor instead.
func (*AttemptProgress) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{80}
}

func (x *AttemptProgress) GetRecordOwnerEpoch() uint64 {
	if x != nil {
		return x.RecordOwnerEpoch
	}
	return 0
}

func (x *AttemptProgress) GetControlStreamGeneration() uint64 {
	if x != nil {
		return x.ControlStreamGeneration
	}
	return 0
}

func (x *AttemptProgress) GetWorkerBootId() string {
	if x != nil {
		return x.WorkerBootId
	}
	return ""
}

func (x *AttemptProgress) GetRequestId() string {
	if x != nil {
		return x.RequestId
	}
	return ""
}

func (x *AttemptProgress) GetAttemptOrdinal() uint64 {
	if x != nil {
		return x.AttemptOrdinal
	}
	return 0
}

func (x *AttemptProgress) GetSeq() uint64 {
	if x != nil {
		return x.Seq
	}
	return 0
}

func (x *AttemptProgress) GetContentType() string {
	if x != nil {
		return x.ContentType
	}
	return ""
}

func (x *AttemptProgress) GetData() []byte {
	if x != nil {
		return x.Data
	}
	return nil
}

func (x *AttemptProgress) GetPlacementId() string {
	if x != nil {
		return x.PlacementId
	}
	return ""
}

// DOCUMENT SHAPE (not a wire message): the subject of `invocation_spec_digest`. Canonical form
// `cozy.worker.v1.InvocationSpec/1`. The key set is CLOSED (01 §3 law 4): no human
// model/adapter ref is spellable; unknown keys refuse on read and are unwritable on author.
// placement_id is DELIBERATELY absent: the same invocation is the same work wherever it routes.
type InvocationSpec struct {
	state                 protoimpl.MessageState `protogen:"open.v1"`
	PackageRevisionDigest string                 `protobuf:"bytes,1,opt,name=package_revision_digest,json=packageRevisionDigest,proto3" json:"package_revision_digest,omitempty"` // published release.json or development source digest
	EnvironmentDigest     string                 `protobuf:"bytes,2,opt,name=environment_digest,json=environmentDigest,proto3" json:"environment_digest,omitempty"`               // class (a), sha256:<hex>: exact selected Environment
	ConfigDigest          string                 `protobuf:"bytes,3,opt,name=config_digest,json=configDigest,proto3" json:"config_digest,omitempty"`                              // class (a): the evaluated-config document (cr-003)
	PayloadDigest         string                 `protobuf:"bytes,4,opt,name=payload_digest,json=payloadDigest,proto3" json:"payload_digest,omitempty"`                           // class (a): the canonical typed-argument document
	Inputs                []*InputBinding        `protobuf:"bytes,5,rep,name=inputs,proto3" json:"inputs,omitempty"`                                                              // ORDERED input identities — INSIDE the digest
	Outputs               []*OutputBinding       `protobuf:"bytes,6,rep,name=outputs,proto3" json:"outputs,omitempty"`                                                            // output ids/kinds/limits — INSIDE the digest
	DeadlineUnixMs        uint64                 `protobuf:"varint,7,opt,name=deadline_unix_ms,json=deadlineUnixMs,proto3" json:"deadline_unix_ms,omitempty"`                     // absolute attempt deadline; worker-enforced; 0 = none
	// Types that are valid to be assigned to Spec:
	//
	//	*InvocationSpec_Serving
	//	*InvocationSpec_Job
	Spec          isInvocationSpec_Spec `protobuf_oneof:"spec"`
	unknownFields protoimpl.UnknownFields
	sizeCache     protoimpl.SizeCache
}

func (x *InvocationSpec) Reset() {
	*x = InvocationSpec{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[81]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *InvocationSpec) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*InvocationSpec) ProtoMessage() {}

func (x *InvocationSpec) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[81]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use InvocationSpec.ProtoReflect.Descriptor instead.
func (*InvocationSpec) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{81}
}

func (x *InvocationSpec) GetPackageRevisionDigest() string {
	if x != nil {
		return x.PackageRevisionDigest
	}
	return ""
}

func (x *InvocationSpec) GetEnvironmentDigest() string {
	if x != nil {
		return x.EnvironmentDigest
	}
	return ""
}

func (x *InvocationSpec) GetConfigDigest() string {
	if x != nil {
		return x.ConfigDigest
	}
	return ""
}

func (x *InvocationSpec) GetPayloadDigest() string {
	if x != nil {
		return x.PayloadDigest
	}
	return ""
}

func (x *InvocationSpec) GetInputs() []*InputBinding {
	if x != nil {
		return x.Inputs
	}
	return nil
}

func (x *InvocationSpec) GetOutputs() []*OutputBinding {
	if x != nil {
		return x.Outputs
	}
	return nil
}

func (x *InvocationSpec) GetDeadlineUnixMs() uint64 {
	if x != nil {
		return x.DeadlineUnixMs
	}
	return 0
}

func (x *InvocationSpec) GetSpec() isInvocationSpec_Spec {
	if x != nil {
		return x.Spec
	}
	return nil
}

func (x *InvocationSpec) GetServing() *ServingInvocationSpec {
	if x != nil {
		if x, ok := x.Spec.(*InvocationSpec_Serving); ok {
			return x.Serving
		}
	}
	return nil
}

func (x *InvocationSpec) GetJob() *JobInvocationSpec {
	if x != nil {
		if x, ok := x.Spec.(*InvocationSpec_Job); ok {
			return x.Job
		}
	}
	return nil
}

type isInvocationSpec_Spec interface {
	isInvocationSpec_Spec()
}

type InvocationSpec_Serving struct {
	Serving *ServingInvocationSpec `protobuf:"bytes,8,opt,name=serving,proto3,oneof"`
}

type InvocationSpec_Job struct {
	Job *JobInvocationSpec `protobuf:"bytes,9,opt,name=job,proto3,oneof"`
}

func (*InvocationSpec_Serving) isInvocationSpec_Spec() {}

func (*InvocationSpec_Job) isInvocationSpec_Spec() {}

type InputBinding struct {
	state         protoimpl.MessageState `protogen:"open.v1"`
	InputId       string                 `protobuf:"bytes,1,opt,name=input_id,json=inputId,proto3" json:"input_id,omitempty"` // the payload field path, list index included
	Digest        string                 `protobuf:"bytes,2,opt,name=digest,proto3" json:"digest,omitempty"`                  // class (b), sha256:<hex>: the input asset's content
	Length        uint64                 `protobuf:"varint,3,opt,name=length,proto3" json:"length,omitempty"`
	KindMime      string                 `protobuf:"bytes,4,opt,name=kind_mime,json=kindMime,proto3" json:"kind_mime,omitempty"`
	Order         uint32                 `protobuf:"varint,5,opt,name=order,proto3" json:"order,omitempty"`
	unknownFields protoimpl.UnknownFields
	sizeCache     protoimpl.SizeCache
}

func (x *InputBinding) Reset() {
	*x = InputBinding{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[82]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *InputBinding) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*InputBinding) ProtoMessage() {}

func (x *InputBinding) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[82]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use InputBinding.ProtoReflect.Descriptor instead.
func (*InputBinding) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{82}
}

func (x *InputBinding) GetInputId() string {
	if x != nil {
		return x.InputId
	}
	return ""
}

func (x *InputBinding) GetDigest() string {
	if x != nil {
		return x.Digest
	}
	return ""
}

func (x *InputBinding) GetLength() uint64 {
	if x != nil {
		return x.Length
	}
	return 0
}

func (x *InputBinding) GetKindMime() string {
	if x != nil {
		return x.KindMime
	}
	return ""
}

func (x *InputBinding) GetOrder() uint32 {
	if x != nil {
		return x.Order
	}
	return 0
}

type OutputBinding struct {
	state         protoimpl.MessageState `protogen:"open.v1"`
	OutputId      string                 `protobuf:"bytes,1,opt,name=output_id,json=outputId,proto3" json:"output_id,omitempty"`
	MimeType      string                 `protobuf:"bytes,2,opt,name=mime_type,json=mimeType,proto3" json:"mime_type,omitempty"`
	MaxBytes      uint64                 `protobuf:"varint,3,opt,name=max_bytes,json=maxBytes,proto3" json:"max_bytes,omitempty"`
	unknownFields protoimpl.UnknownFields
	sizeCache     protoimpl.SizeCache
}

func (x *OutputBinding) Reset() {
	*x = OutputBinding{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[83]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *OutputBinding) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*OutputBinding) ProtoMessage() {}

func (x *OutputBinding) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[83]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use OutputBinding.ProtoReflect.Descriptor instead.
func (*OutputBinding) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{83}
}

func (x *OutputBinding) GetOutputId() string {
	if x != nil {
		return x.OutputId
	}
	return ""
}

func (x *OutputBinding) GetMimeType() string {
	if x != nil {
		return x.MimeType
	}
	return ""
}

func (x *OutputBinding) GetMaxBytes() uint64 {
	if x != nil {
		return x.MaxBytes
	}
	return 0
}

type ServingInvocationSpec struct {
	state                   protoimpl.MessageState `protogen:"open.v1"`
	EntrypointBindingDigest string                 `protobuf:"bytes,1,opt,name=entrypoint_binding_digest,json=entrypointBindingDigest,proto3" json:"entrypoint_binding_digest,omitempty"` // class (a): selected entrypoint binding identity
	AttemptBindingId        string                 `protobuf:"bytes,2,opt,name=attempt_binding_id,json=attemptBindingId,proto3" json:"attempt_binding_id,omitempty"`                      // equal to field 1 when no adapters
	unknownFields           protoimpl.UnknownFields
	sizeCache               protoimpl.SizeCache
}

func (x *ServingInvocationSpec) Reset() {
	*x = ServingInvocationSpec{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[84]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *ServingInvocationSpec) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*ServingInvocationSpec) ProtoMessage() {}

func (x *ServingInvocationSpec) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[84]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use ServingInvocationSpec.ProtoReflect.Descriptor instead.
func (*ServingInvocationSpec) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{84}
}

func (x *ServingInvocationSpec) GetEntrypointBindingDigest() string {
	if x != nil {
		return x.EntrypointBindingDigest
	}
	return ""
}

func (x *ServingInvocationSpec) GetAttemptBindingId() string {
	if x != nil {
		return x.AttemptBindingId
	}
	return ""
}

type JobInvocationSpec struct {
	state               protoimpl.MessageState `protogen:"open.v1"`
	BuildId             string                 `protobuf:"bytes,1,opt,name=build_id,json=buildId,proto3" json:"build_id,omitempty"`
	JobDescriptorId     string                 `protobuf:"bytes,2,opt,name=job_descriptor_id,json=jobDescriptorId,proto3" json:"job_descriptor_id,omitempty"`
	PublicationContract *PublicationContract   `protobuf:"bytes,3,opt,name=publication_contract,json=publicationContract,proto3" json:"publication_contract,omitempty"`
	unknownFields       protoimpl.UnknownFields
	sizeCache           protoimpl.SizeCache
}

func (x *JobInvocationSpec) Reset() {
	*x = JobInvocationSpec{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[85]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *JobInvocationSpec) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*JobInvocationSpec) ProtoMessage() {}

func (x *JobInvocationSpec) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[85]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use JobInvocationSpec.ProtoReflect.Descriptor instead.
func (*JobInvocationSpec) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{85}
}

func (x *JobInvocationSpec) GetBuildId() string {
	if x != nil {
		return x.BuildId
	}
	return ""
}

func (x *JobInvocationSpec) GetJobDescriptorId() string {
	if x != nil {
		return x.JobDescriptorId
	}
	return ""
}

func (x *JobInvocationSpec) GetPublicationContract() *PublicationContract {
	if x != nil {
		return x.PublicationContract
	}
	return nil
}

// Refreshable ACCESS, strictly PER-ATTEMPT: urls, token, expiry — and the invocation it serves.
// Nothing else. Pre-attempt placement acquisition does not ride this per-attempt grant.
type DeliveryGrant struct {
	state                protoimpl.MessageState    `protogen:"open.v1"`
	InvocationSpecDigest []byte                    `protobuf:"bytes,1,opt,name=invocation_spec_digest,json=invocationSpecDigest,proto3" json:"invocation_spec_digest,omitempty"` // the subject this grant serves; revalidated on refresh
	Credential           *DeliveryAccessCredential `protobuf:"bytes,2,opt,name=credential,proto3" json:"credential,omitempty"`                                                   // per-attempt scoped capability token
	FileBaseUrl          string                    `protobuf:"bytes,3,opt,name=file_base_url,json=fileBaseUrl,proto3" json:"file_base_url,omitempty"`
	ExpiresAtUnix        uint64                    `protobuf:"varint,4,opt,name=expires_at_unix,json=expiresAtUnix,proto3" json:"expires_at_unix,omitempty"`
	Inputs               []*InputAccess            `protobuf:"bytes,5,rep,name=inputs,proto3" json:"inputs,omitempty"`   // sorted by input_id; ids must match the spec's set
	Outputs              []*OutputAccess           `protobuf:"bytes,6,rep,name=outputs,proto3" json:"outputs,omitempty"` // sorted by output_id
	unknownFields        protoimpl.UnknownFields
	sizeCache            protoimpl.SizeCache
}

func (x *DeliveryGrant) Reset() {
	*x = DeliveryGrant{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[86]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *DeliveryGrant) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*DeliveryGrant) ProtoMessage() {}

func (x *DeliveryGrant) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[86]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use DeliveryGrant.ProtoReflect.Descriptor instead.
func (*DeliveryGrant) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{86}
}

func (x *DeliveryGrant) GetInvocationSpecDigest() []byte {
	if x != nil {
		return x.InvocationSpecDigest
	}
	return nil
}

func (x *DeliveryGrant) GetCredential() *DeliveryAccessCredential {
	if x != nil {
		return x.Credential
	}
	return nil
}

func (x *DeliveryGrant) GetFileBaseUrl() string {
	if x != nil {
		return x.FileBaseUrl
	}
	return ""
}

func (x *DeliveryGrant) GetExpiresAtUnix() uint64 {
	if x != nil {
		return x.ExpiresAtUnix
	}
	return 0
}

func (x *DeliveryGrant) GetInputs() []*InputAccess {
	if x != nil {
		return x.Inputs
	}
	return nil
}

func (x *DeliveryGrant) GetOutputs() []*OutputAccess {
	if x != nil {
		return x.Outputs
	}
	return nil
}

type InputAccess struct {
	state         protoimpl.MessageState `protogen:"open.v1"`
	InputId       string                 `protobuf:"bytes,1,opt,name=input_id,json=inputId,proto3" json:"input_id,omitempty"`
	Url           string                 `protobuf:"bytes,2,opt,name=url,proto3" json:"url,omitempty"` // presigned download URL
	unknownFields protoimpl.UnknownFields
	sizeCache     protoimpl.SizeCache
}

func (x *InputAccess) Reset() {
	*x = InputAccess{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[87]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *InputAccess) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*InputAccess) ProtoMessage() {}

func (x *InputAccess) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[87]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use InputAccess.ProtoReflect.Descriptor instead.
func (*InputAccess) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{87}
}

func (x *InputAccess) GetInputId() string {
	if x != nil {
		return x.InputId
	}
	return ""
}

func (x *InputAccess) GetUrl() string {
	if x != nil {
		return x.Url
	}
	return ""
}

type OutputAccess struct {
	state         protoimpl.MessageState `protogen:"open.v1"`
	OutputId      string                 `protobuf:"bytes,1,opt,name=output_id,json=outputId,proto3" json:"output_id,omitempty"`
	Url           string                 `protobuf:"bytes,2,opt,name=url,proto3" json:"url,omitempty"` // presigned upload URL
	unknownFields protoimpl.UnknownFields
	sizeCache     protoimpl.SizeCache
}

func (x *OutputAccess) Reset() {
	*x = OutputAccess{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[88]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *OutputAccess) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*OutputAccess) ProtoMessage() {}

func (x *OutputAccess) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[88]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use OutputAccess.ProtoReflect.Descriptor instead.
func (*OutputAccess) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{88}
}

func (x *OutputAccess) GetOutputId() string {
	if x != nil {
		return x.OutputId
	}
	return ""
}

func (x *OutputAccess) GetUrl() string {
	if x != nil {
		return x.Url
	}
	return ""
}

type DeliveryAccessCredential struct {
	state         protoimpl.MessageState `protogen:"open.v1"`
	Issuer        string                 `protobuf:"bytes,1,opt,name=issuer,proto3" json:"issuer,omitempty"`
	KeyId         string                 `protobuf:"bytes,2,opt,name=key_id,json=keyId,proto3" json:"key_id,omitempty"`
	Epoch         uint64                 `protobuf:"varint,3,opt,name=epoch,proto3" json:"epoch,omitempty"`
	ExpiresAtUnix uint64                 `protobuf:"varint,4,opt,name=expires_at_unix,json=expiresAtUnix,proto3" json:"expires_at_unix,omitempty"`
	Token         []byte                 `protobuf:"bytes,5,opt,name=token,proto3" json:"token,omitempty"` // STRUCTURALLY SECRET; never in a digest, a log, or an
	unknownFields protoimpl.UnknownFields
	sizeCache     protoimpl.SizeCache
}

func (x *DeliveryAccessCredential) Reset() {
	*x = DeliveryAccessCredential{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[89]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *DeliveryAccessCredential) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*DeliveryAccessCredential) ProtoMessage() {}

func (x *DeliveryAccessCredential) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[89]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use DeliveryAccessCredential.ProtoReflect.Descriptor instead.
func (*DeliveryAccessCredential) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{89}
}

func (x *DeliveryAccessCredential) GetIssuer() string {
	if x != nil {
		return x.Issuer
	}
	return ""
}

func (x *DeliveryAccessCredential) GetKeyId() string {
	if x != nil {
		return x.KeyId
	}
	return ""
}

func (x *DeliveryAccessCredential) GetEpoch() uint64 {
	if x != nil {
		return x.Epoch
	}
	return 0
}

func (x *DeliveryAccessCredential) GetExpiresAtUnix() uint64 {
	if x != nil {
		return x.ExpiresAtUnix
	}
	return 0
}

func (x *DeliveryAccessCredential) GetToken() []byte {
	if x != nil {
		return x.Token
	}
	return nil
}

type ResourceCaps struct {
	state                protoimpl.MessageState `protogen:"open.v1"`
	DeviceRequired       bool                   `protobuf:"varint,1,opt,name=device_required,json=deviceRequired,proto3" json:"device_required,omitempty"`
	MaxDeviceMemoryBytes uint64                 `protobuf:"varint,2,opt,name=max_device_memory_bytes,json=maxDeviceMemoryBytes,proto3" json:"max_device_memory_bytes,omitempty"`
	MaxRssBytes          uint64                 `protobuf:"varint,3,opt,name=max_rss_bytes,json=maxRssBytes,proto3" json:"max_rss_bytes,omitempty"`
	MaxDiskBytes         uint64                 `protobuf:"varint,4,opt,name=max_disk_bytes,json=maxDiskBytes,proto3" json:"max_disk_bytes,omitempty"`
	unknownFields        protoimpl.UnknownFields
	sizeCache            protoimpl.SizeCache
}

func (x *ResourceCaps) Reset() {
	*x = ResourceCaps{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[90]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *ResourceCaps) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*ResourceCaps) ProtoMessage() {}

func (x *ResourceCaps) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[90]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use ResourceCaps.ProtoReflect.Descriptor instead.
func (*ResourceCaps) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{90}
}

func (x *ResourceCaps) GetDeviceRequired() bool {
	if x != nil {
		return x.DeviceRequired
	}
	return false
}

func (x *ResourceCaps) GetMaxDeviceMemoryBytes() uint64 {
	if x != nil {
		return x.MaxDeviceMemoryBytes
	}
	return 0
}

func (x *ResourceCaps) GetMaxRssBytes() uint64 {
	if x != nil {
		return x.MaxRssBytes
	}
	return 0
}

func (x *ResourceCaps) GetMaxDiskBytes() uint64 {
	if x != nil {
		return x.MaxDiskBytes
	}
	return 0
}

type PublicationContract struct {
	state         protoimpl.MessageState `protogen:"open.v1"`
	Outputs       []*OutputBinding       `protobuf:"bytes,1,rep,name=outputs,proto3" json:"outputs,omitempty"` // sorted by output_id
	GrantId       string                 `protobuf:"bytes,2,opt,name=grant_id,json=grantId,proto3" json:"grant_id,omitempty"`
	unknownFields protoimpl.UnknownFields
	sizeCache     protoimpl.SizeCache
}

func (x *PublicationContract) Reset() {
	*x = PublicationContract{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[91]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *PublicationContract) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*PublicationContract) ProtoMessage() {}

func (x *PublicationContract) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[91]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use PublicationContract.ProtoReflect.Descriptor instead.
func (*PublicationContract) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{91}
}

func (x *PublicationContract) GetOutputs() []*OutputBinding {
	if x != nil {
		return x.Outputs
	}
	return nil
}

func (x *PublicationContract) GetGrantId() string {
	if x != nil {
		return x.GrantId
	}
	return ""
}

// Torch-free worker-measurable STATICS (NVML / sysctl / OS facts), sent once at ClaimAck and
// never re-sent. Accelerator-neutral (#446): backend + memory_model use the AcceleratorFacts
// vocabulary (#422/#423). The worker never fabricates capability facts — a discovered
// accelerator is PRESENT-BUT-UNQUALIFIED until a placement executor qualifies it
// (PlacementStatus.accelerator). torch_version is gone from here: a torch-free process cannot
// honestly report one.
type WorkerResources struct {
	state                  protoimpl.MessageState `protogen:"open.v1"`
	Platform               string                 `protobuf:"bytes,1,opt,name=platform,proto3" json:"platform,omitempty"`                          // os/arch, e.g. linux/amd64, windows/amd64, darwin/arm64
	Backend                string                 `protobuf:"bytes,2,opt,name=backend,proto3" json:"backend,omitempty"`                            // cuda | mps; "" = no accelerator discovered
	MemoryModel            string                 `protobuf:"bytes,3,opt,name=memory_model,json=memoryModel,proto3" json:"memory_model,omitempty"` // discrete | unified — the ledger's PRICING MODEL selector
	DeviceCount            uint32                 `protobuf:"varint,4,opt,name=device_count,json=deviceCount,proto3" json:"device_count,omitempty"`
	DeviceName             string                 `protobuf:"bytes,5,opt,name=device_name,json=deviceName,proto3" json:"device_name,omitempty"`
	DeviceMemoryTotalBytes uint64                 `protobuf:"varint,6,opt,name=device_memory_total_bytes,json=deviceMemoryTotalBytes,proto3" json:"device_memory_total_bytes,omitempty"` // per device
	DriverVersion          string                 `protobuf:"bytes,7,opt,name=driver_version,json=driverVersion,proto3" json:"driver_version,omitempty"`
	BackendVersion         string                 `protobuf:"bytes,8,opt,name=backend_version,json=backendVersion,proto3" json:"backend_version,omitempty"` // CUDA toolkit version; macOS/Metal version on mps
	HostRamTotalBytes      uint64                 `protobuf:"varint,9,opt,name=host_ram_total_bytes,json=hostRamTotalBytes,proto3" json:"host_ram_total_bytes,omitempty"`
	VcpuCount              uint32                 `protobuf:"varint,10,opt,name=vcpu_count,json=vcpuCount,proto3" json:"vcpu_count,omitempty"`
	DiskTotalBytes         uint64                 `protobuf:"varint,11,opt,name=disk_total_bytes,json=diskTotalBytes,proto3" json:"disk_total_bytes,omitempty"`
	PowerCapWatts          uint32                 `protobuf:"varint,12,opt,name=power_cap_watts,json=powerCapWatts,proto3" json:"power_cap_watts,omitempty"`
	PartitionProfile       string                 `protobuf:"bytes,13,opt,name=partition_profile,json=partitionProfile,proto3" json:"partition_profile,omitempty"` // device partition (NVIDIA MIG on cuda); empty = whole
	// device
	Interconnect  string   `protobuf:"bytes,14,opt,name=interconnect,proto3" json:"interconnect,omitempty"`
	PeerAccess    bool     `protobuf:"varint,15,opt,name=peer_access,json=peerAccess,proto3" json:"peer_access,omitempty"`
	Unreadable    []string `protobuf:"bytes,16,rep,name=unreadable,proto3" json:"unreadable,omitempty"` // tri-state honesty: unreadable is distinct from zero
	unknownFields protoimpl.UnknownFields
	sizeCache     protoimpl.SizeCache
}

func (x *WorkerResources) Reset() {
	*x = WorkerResources{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[92]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *WorkerResources) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*WorkerResources) ProtoMessage() {}

func (x *WorkerResources) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[92]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use WorkerResources.ProtoReflect.Descriptor instead.
func (*WorkerResources) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{92}
}

func (x *WorkerResources) GetPlatform() string {
	if x != nil {
		return x.Platform
	}
	return ""
}

func (x *WorkerResources) GetBackend() string {
	if x != nil {
		return x.Backend
	}
	return ""
}

func (x *WorkerResources) GetMemoryModel() string {
	if x != nil {
		return x.MemoryModel
	}
	return ""
}

func (x *WorkerResources) GetDeviceCount() uint32 {
	if x != nil {
		return x.DeviceCount
	}
	return 0
}

func (x *WorkerResources) GetDeviceName() string {
	if x != nil {
		return x.DeviceName
	}
	return ""
}

func (x *WorkerResources) GetDeviceMemoryTotalBytes() uint64 {
	if x != nil {
		return x.DeviceMemoryTotalBytes
	}
	return 0
}

func (x *WorkerResources) GetDriverVersion() string {
	if x != nil {
		return x.DriverVersion
	}
	return ""
}

func (x *WorkerResources) GetBackendVersion() string {
	if x != nil {
		return x.BackendVersion
	}
	return ""
}

func (x *WorkerResources) GetHostRamTotalBytes() uint64 {
	if x != nil {
		return x.HostRamTotalBytes
	}
	return 0
}

func (x *WorkerResources) GetVcpuCount() uint32 {
	if x != nil {
		return x.VcpuCount
	}
	return 0
}

func (x *WorkerResources) GetDiskTotalBytes() uint64 {
	if x != nil {
		return x.DiskTotalBytes
	}
	return 0
}

func (x *WorkerResources) GetPowerCapWatts() uint32 {
	if x != nil {
		return x.PowerCapWatts
	}
	return 0
}

func (x *WorkerResources) GetPartitionProfile() string {
	if x != nil {
		return x.PartitionProfile
	}
	return ""
}

func (x *WorkerResources) GetInterconnect() string {
	if x != nil {
		return x.Interconnect
	}
	return ""
}

func (x *WorkerResources) GetPeerAccess() bool {
	if x != nil {
		return x.PeerAccess
	}
	return false
}

func (x *WorkerResources) GetUnreadable() []string {
	if x != nil {
		return x.Unreadable
	}
	return nil
}

type JobCapacity struct {
	state protoimpl.MessageState `protogen:"open.v1"`
	// #446); jobs_available IS the credit
	JobsInFlight  uint32 `protobuf:"varint,1,opt,name=jobs_in_flight,json=jobsInFlight,proto3" json:"jobs_in_flight,omitempty"`
	JobsAvailable uint32 `protobuf:"varint,2,opt,name=jobs_available,json=jobsAvailable,proto3" json:"jobs_available,omitempty"`
	FreeDiskBytes uint64 `protobuf:"varint,4,opt,name=free_disk_bytes,json=freeDiskBytes,proto3" json:"free_disk_bytes,omitempty"`
	unknownFields protoimpl.UnknownFields
	sizeCache     protoimpl.SizeCache
}

func (x *JobCapacity) Reset() {
	*x = JobCapacity{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[93]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *JobCapacity) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*JobCapacity) ProtoMessage() {}

func (x *JobCapacity) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[93]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use JobCapacity.ProtoReflect.Descriptor instead.
func (*JobCapacity) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{93}
}

func (x *JobCapacity) GetJobsInFlight() uint32 {
	if x != nil {
		return x.JobsInFlight
	}
	return 0
}

func (x *JobCapacity) GetJobsAvailable() uint32 {
	if x != nil {
		return x.JobsAvailable
	}
	return 0
}

func (x *JobCapacity) GetFreeDiskBytes() uint64 {
	if x != nil {
		return x.FreeDiskBytes
	}
	return 0
}

// Was ActiveAttempt (#481): the set holds unacked OUTCOMES too, and those are not active.
type HeldAttempt struct {
	state                protoimpl.MessageState `protogen:"open.v1"`
	RequestId            string                 `protobuf:"bytes,1,opt,name=request_id,json=requestId,proto3" json:"request_id,omitempty"`
	AttemptOrdinal       uint64                 `protobuf:"varint,2,opt,name=attempt_ordinal,json=attemptOrdinal,proto3" json:"attempt_ordinal,omitempty"`
	Kind                 AttemptKind            `protobuf:"varint,3,opt,name=kind,proto3,enum=cozy.worker.v1.AttemptKind" json:"kind,omitempty"`
	State                AttemptState           `protobuf:"varint,4,opt,name=state,proto3,enum=cozy.worker.v1.AttemptState" json:"state,omitempty"`
	InvocationSpecDigest []byte                 `protobuf:"bytes,5,opt,name=invocation_spec_digest,json=invocationSpecDigest,proto3" json:"invocation_spec_digest,omitempty"`
	PlacementId          string                 `protobuf:"bytes,6,opt,name=placement_id,json=placementId,proto3" json:"placement_id,omitempty"`                       // the executing placement; empty in job mode
	ExecutorGeneration   uint64                 `protobuf:"varint,7,opt,name=executor_generation,json=executorGeneration,proto3" json:"executor_generation,omitempty"` // the generation it runs/ran under
	OutcomeId            string                 `protobuf:"bytes,8,opt,name=outcome_id,json=outcomeId,proto3" json:"outcome_id,omitempty"`                             // set iff state is OUTCOME_PENDING_ACK
	OutcomeDigest        []byte                 `protobuf:"bytes,9,opt,name=outcome_digest,json=outcomeDigest,proto3" json:"outcome_digest,omitempty"`                 // class (a); set iff state is OUTCOME_PENDING_ACK
	unknownFields        protoimpl.UnknownFields
	sizeCache            protoimpl.SizeCache
}

func (x *HeldAttempt) Reset() {
	*x = HeldAttempt{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[94]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *HeldAttempt) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*HeldAttempt) ProtoMessage() {}

func (x *HeldAttempt) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[94]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use HeldAttempt.ProtoReflect.Descriptor instead.
func (*HeldAttempt) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{94}
}

func (x *HeldAttempt) GetRequestId() string {
	if x != nil {
		return x.RequestId
	}
	return ""
}

func (x *HeldAttempt) GetAttemptOrdinal() uint64 {
	if x != nil {
		return x.AttemptOrdinal
	}
	return 0
}

func (x *HeldAttempt) GetKind() AttemptKind {
	if x != nil {
		return x.Kind
	}
	return AttemptKind_ATTEMPT_KIND_UNSPECIFIED
}

func (x *HeldAttempt) GetState() AttemptState {
	if x != nil {
		return x.State
	}
	return AttemptState_ATTEMPT_STATE_UNSPECIFIED
}

func (x *HeldAttempt) GetInvocationSpecDigest() []byte {
	if x != nil {
		return x.InvocationSpecDigest
	}
	return nil
}

func (x *HeldAttempt) GetPlacementId() string {
	if x != nil {
		return x.PlacementId
	}
	return ""
}

func (x *HeldAttempt) GetExecutorGeneration() uint64 {
	if x != nil {
		return x.ExecutorGeneration
	}
	return 0
}

func (x *HeldAttempt) GetOutcomeId() string {
	if x != nil {
		return x.OutcomeId
	}
	return ""
}

func (x *HeldAttempt) GetOutcomeDigest() []byte {
	if x != nil {
		return x.OutcomeDigest
	}
	return nil
}

type Fault struct {
	state         protoimpl.MessageState `protogen:"open.v1"`
	Kind          FaultKind              `protobuf:"varint,1,opt,name=kind,proto3,enum=cozy.worker.v1.FaultKind" json:"kind,omitempty"`
	Subject       string                 `protobuf:"bytes,2,opt,name=subject,proto3" json:"subject,omitempty"` // the placement_id, subject digest, or machine facet
	Reason        string                 `protobuf:"bytes,3,opt,name=reason,proto3" json:"reason,omitempty"`   // bounded <= 1024 bytes
	Detail        string                 `protobuf:"bytes,4,opt,name=detail,proto3" json:"detail,omitempty"`   // bounded <= 1024 bytes
	unknownFields protoimpl.UnknownFields
	sizeCache     protoimpl.SizeCache
}

func (x *Fault) Reset() {
	*x = Fault{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[95]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *Fault) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*Fault) ProtoMessage() {}

func (x *Fault) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[95]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use Fault.ProtoReflect.Descriptor instead.
func (*Fault) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{95}
}

func (x *Fault) GetKind() FaultKind {
	if x != nil {
		return x.Kind
	}
	return FaultKind_FAULT_KIND_UNSPECIFIED
}

func (x *Fault) GetSubject() string {
	if x != nil {
		return x.Subject
	}
	return ""
}

func (x *Fault) GetReason() string {
	if x != nil {
		return x.Reason
	}
	return ""
}

func (x *Fault) GetDetail() string {
	if x != nil {
		return x.Detail
	}
	return ""
}

type OutputManifest struct {
	state                    protoimpl.MessageState `protogen:"open.v1"`
	PublicationReceiptDigest string                 `protobuf:"bytes,1,opt,name=publication_receipt_digest,json=publicationReceiptDigest,proto3" json:"publication_receipt_digest,omitempty"` // class (a), sha256:<hex>: the PublicationReceipt's
	// canonical digest (#420). Was `manifest_id` — the frozen
	// proto's own comment said what it carries and the field
	// name claimed otherwise (#481/#486).
	Outputs       []*OutputEntry `protobuf:"bytes,3,rep,name=outputs,proto3" json:"outputs,omitempty"` // sorted by output_id
	unknownFields protoimpl.UnknownFields
	sizeCache     protoimpl.SizeCache
}

func (x *OutputManifest) Reset() {
	*x = OutputManifest{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[96]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *OutputManifest) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*OutputManifest) ProtoMessage() {}

func (x *OutputManifest) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[96]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use OutputManifest.ProtoReflect.Descriptor instead.
func (*OutputManifest) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{96}
}

func (x *OutputManifest) GetPublicationReceiptDigest() string {
	if x != nil {
		return x.PublicationReceiptDigest
	}
	return ""
}

func (x *OutputManifest) GetOutputs() []*OutputEntry {
	if x != nil {
		return x.Outputs
	}
	return nil
}

type OutputEntry struct {
	state         protoimpl.MessageState `protogen:"open.v1"`
	OutputId      string                 `protobuf:"bytes,1,opt,name=output_id,json=outputId,proto3" json:"output_id,omitempty"`
	Digest        []byte                 `protobuf:"bytes,2,opt,name=digest,proto3" json:"digest,omitempty"` // class (b)
	Length        uint64                 `protobuf:"varint,3,opt,name=length,proto3" json:"length,omitempty"`
	MimeType      string                 `protobuf:"bytes,4,opt,name=mime_type,json=mimeType,proto3" json:"mime_type,omitempty"`
	unknownFields protoimpl.UnknownFields
	sizeCache     protoimpl.SizeCache
}

func (x *OutputEntry) Reset() {
	*x = OutputEntry{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[97]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *OutputEntry) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*OutputEntry) ProtoMessage() {}

func (x *OutputEntry) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[97]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use OutputEntry.ProtoReflect.Descriptor instead.
func (*OutputEntry) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{97}
}

func (x *OutputEntry) GetOutputId() string {
	if x != nil {
		return x.OutputId
	}
	return ""
}

func (x *OutputEntry) GetDigest() []byte {
	if x != nil {
		return x.Digest
	}
	return nil
}

func (x *OutputEntry) GetLength() uint64 {
	if x != nil {
		return x.Length
	}
	return 0
}

func (x *OutputEntry) GetMimeType() string {
	if x != nil {
		return x.MimeType
	}
	return ""
}

type AttemptMetrics struct {
	state                 protoimpl.MessageState `protogen:"open.v1"`
	RuntimeMs             uint64                 `protobuf:"varint,1,opt,name=runtime_ms,json=runtimeMs,proto3" json:"runtime_ms,omitempty"`
	QueueMs               uint64                 `protobuf:"varint,2,opt,name=queue_ms,json=queueMs,proto3" json:"queue_ms,omitempty"`
	PeakDeviceMemoryBytes uint64                 `protobuf:"varint,3,opt,name=peak_device_memory_bytes,json=peakDeviceMemoryBytes,proto3" json:"peak_device_memory_bytes,omitempty"`
	RssAtEndBytes         uint64                 `protobuf:"varint,4,opt,name=rss_at_end_bytes,json=rssAtEndBytes,proto3" json:"rss_at_end_bytes,omitempty"`
	OutputCount           uint32                 `protobuf:"varint,5,opt,name=output_count,json=outputCount,proto3" json:"output_count,omitempty"`
	InputTokens           uint64                 `protobuf:"varint,6,opt,name=input_tokens,json=inputTokens,proto3" json:"input_tokens,omitempty"` // analytics at launch, never settlement input
	OutputTokens          uint64                 `protobuf:"varint,7,opt,name=output_tokens,json=outputTokens,proto3" json:"output_tokens,omitempty"`
	DeviceLeaseMs         uint64                 `protobuf:"varint,8,opt,name=device_lease_ms,json=deviceLeaseMs,proto3" json:"device_lease_ms,omitempty"` // worker-attested
	DeviceCount           uint32                 `protobuf:"varint,9,opt,name=device_count,json=deviceCount,proto3" json:"device_count,omitempty"`
	HandlerMs             uint64                 `protobuf:"varint,10,opt,name=handler_ms,json=handlerMs,proto3" json:"handler_ms,omitempty"`
	FinalizationMs        uint64                 `protobuf:"varint,11,opt,name=finalization_ms,json=finalizationMs,proto3" json:"finalization_ms,omitempty"`
	UnverifiedFields      []string               `protobuf:"bytes,12,rep,name=unverified_fields,json=unverifiedFields,proto3" json:"unverified_fields,omitempty"`
	unknownFields         protoimpl.UnknownFields
	sizeCache             protoimpl.SizeCache
}

func (x *AttemptMetrics) Reset() {
	*x = AttemptMetrics{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[98]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *AttemptMetrics) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*AttemptMetrics) ProtoMessage() {}

func (x *AttemptMetrics) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[98]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use AttemptMetrics.ProtoReflect.Descriptor instead.
func (*AttemptMetrics) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{98}
}

func (x *AttemptMetrics) GetRuntimeMs() uint64 {
	if x != nil {
		return x.RuntimeMs
	}
	return 0
}

func (x *AttemptMetrics) GetQueueMs() uint64 {
	if x != nil {
		return x.QueueMs
	}
	return 0
}

func (x *AttemptMetrics) GetPeakDeviceMemoryBytes() uint64 {
	if x != nil {
		return x.PeakDeviceMemoryBytes
	}
	return 0
}

func (x *AttemptMetrics) GetRssAtEndBytes() uint64 {
	if x != nil {
		return x.RssAtEndBytes
	}
	return 0
}

func (x *AttemptMetrics) GetOutputCount() uint32 {
	if x != nil {
		return x.OutputCount
	}
	return 0
}

func (x *AttemptMetrics) GetInputTokens() uint64 {
	if x != nil {
		return x.InputTokens
	}
	return 0
}

func (x *AttemptMetrics) GetOutputTokens() uint64 {
	if x != nil {
		return x.OutputTokens
	}
	return 0
}

func (x *AttemptMetrics) GetDeviceLeaseMs() uint64 {
	if x != nil {
		return x.DeviceLeaseMs
	}
	return 0
}

func (x *AttemptMetrics) GetDeviceCount() uint32 {
	if x != nil {
		return x.DeviceCount
	}
	return 0
}

func (x *AttemptMetrics) GetHandlerMs() uint64 {
	if x != nil {
		return x.HandlerMs
	}
	return 0
}

func (x *AttemptMetrics) GetFinalizationMs() uint64 {
	if x != nil {
		return x.FinalizationMs
	}
	return 0
}

func (x *AttemptMetrics) GetUnverifiedFields() []string {
	if x != nil {
		return x.UnverifiedFields
	}
	return nil
}

type TriageBundleRef struct {
	state              protoimpl.MessageState `protogen:"open.v1"`
	SubjectId          string                 `protobuf:"bytes,1,opt,name=subject_id,json=subjectId,proto3" json:"subject_id,omitempty"`
	WriteReceiptDigest []byte                 `protobuf:"bytes,4,opt,name=write_receipt_digest,json=writeReceiptDigest,proto3" json:"write_receipt_digest,omitempty"` // class (a)
	Length             uint64                 `protobuf:"varint,3,opt,name=length,proto3" json:"length,omitempty"`                                                    // bounded <= 1048576 bytes
	unknownFields      protoimpl.UnknownFields
	sizeCache          protoimpl.SizeCache
}

func (x *TriageBundleRef) Reset() {
	*x = TriageBundleRef{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[99]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *TriageBundleRef) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*TriageBundleRef) ProtoMessage() {}

func (x *TriageBundleRef) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[99]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use TriageBundleRef.ProtoReflect.Descriptor instead.
func (*TriageBundleRef) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{99}
}

func (x *TriageBundleRef) GetSubjectId() string {
	if x != nil {
		return x.SubjectId
	}
	return ""
}

func (x *TriageBundleRef) GetWriteReceiptDigest() []byte {
	if x != nil {
		return x.WriteReceiptDigest
	}
	return nil
}

func (x *TriageBundleRef) GetLength() uint64 {
	if x != nil {
		return x.Length
	}
	return 0
}

var File_cozy_worker_v1_worker_proto protoreflect.FileDescriptor

const file_cozy_worker_v1_worker_proto_rawDesc = "" +
	"\n" +
	"\x1bcozy/worker/v1/worker.proto\x12\x0ecozy.worker.v1\"\xaf\x01\n" +
	"\x18PreparePackageSetRequest\x12/\n" +
	"\x13download_delegation\x18\x01 \x01(\fR\x12downloadDelegation\x127\n" +
	"\x05files\x18\x02 \x03(\v2!.cozy.worker.v1.LocalDownloadFileR\x05files\x12)\n" +
	"\x10environment_root\x18\x03 \x01(\tR\x0fenvironmentRoot\"\xaa\x01\n" +
	"\x11LocalDownloadFile\x12\x16\n" +
	"\x06digest\x18\x01 \x01(\fR\x06digest\x12\x1a\n" +
	"\bfilename\x18\x02 \x01(\tR\bfilename\x125\n" +
	"\x04kind\x18\x03 \x01(\x0e2!.cozy.worker.v1.LocalDownloadKindR\x04kind\x12\x16\n" +
	"\x06length\x18\x04 \x01(\x04R\x06length\x12\x12\n" +
	"\x04path\x18\x05 \x01(\tR\x04path\"c\n" +
	"\x17PreparePackageSetResult\x12H\n" +
	"\rplacement_set\x18\x01 \x01(\v2#.cozy.worker.v1.DesiredPlacementSetR\fplacementSet\"w\n" +
	"\x14LocalModelSourceFile\x12\x16\n" +
	"\x06member\x18\x01 \x01(\tR\x06member\x12\x1b\n" +
	"\tobject_id\x18\x02 \x01(\tR\bobjectId\x12\x16\n" +
	"\x06length\x18\x03 \x01(\x04R\x06length\x12\x12\n" +
	"\x04path\x18\x04 \x01(\tR\x04path\"B\n" +
	"\x12ModelSourceProfile\x12\x12\n" +
	"\x04slot\x18\x01 \x01(\tR\x04slot\x12\x18\n" +
	"\aprofile\x18\x02 \x01(\tR\aprofile\"t\n" +
	"\x13PreparedModelSource\x12\x12\n" +
	"\x04slot\x18\x01 \x01(\tR\x04slot\x12\x18\n" +
	"\aprofile\x18\x02 \x01(\tR\aprofile\x12/\n" +
	"\bmanifest\x18\x03 \x01(\v2\x13.cozy.worker.v1.RefR\bmanifest\"\xbc\x02\n" +
	"\x19PrepareModelSourceRequest\x12!\n" +
	"\foperation_id\x18\x01 \x01(\tR\voperationId\x126\n" +
	"\x17source_selection_digest\x18\x02 \x01(\fR\x15sourceSelectionDigest\x12>\n" +
	"\bprofiles\x18\x03 \x03(\v2\".cozy.worker.v1.ModelSourceProfileR\bprofiles\x12:\n" +
	"\x05files\x18\x04 \x03(\v2$.cozy.worker.v1.LocalModelSourceFileR\x05files\x12\x1d\n" +
	"\n" +
	"source_uri\x18\x05 \x01(\tR\tsourceUri\x12)\n" +
	"\x10declared_license\x18\x06 \x01(\tR\x0fdeclaredLicense\"\xdc\x01\n" +
	"\x18PrepareModelSourceResult\x12C\n" +
	"\aoutcome\x18\x01 \x01(\x0e2).cozy.worker.v1.ModelSourcePrepareOutcomeR\aoutcome\x12=\n" +
	"\asources\x18\x02 \x03(\v2#.cozy.worker.v1.PreparedModelSourceR\asources\x12\x1b\n" +
	"\tsafe_code\x18\x03 \x01(\tR\bsafeCode\x12\x1f\n" +
	"\vsafe_detail\x18\x04 \x01(\tR\n" +
	"safeDetail\"\x85\t\n" +
	"\x10RecordOwnerFrame\x12-\n" +
	"\x05claim\x18\x05 \x01(\v2\x15.cozy.worker.v1.ClaimH\x00R\x05claim\x12I\n" +
	"\rdesired_state\x18\x06 \x01(\v2\".cozy.worker.v1.DesiredWorkerStateH\x00R\fdesiredState\x12C\n" +
	"\rattempt_offer\x18\a \x01(\v2\x1c.cozy.worker.v1.AttemptOfferH\x00R\fattemptOffer\x12F\n" +
	"\x0ecancel_attempt\x18\b \x01(\v2\x1d.cozy.worker.v1.CancelAttemptH\x00R\rcancelAttempt\x12D\n" +
	"\voutcome_ack\x18\t \x01(\v2!.cozy.worker.v1.AttemptOutcomeAckH\x00R\n" +
	"outcomeAck\x12U\n" +
	"\x12checkpoint_receipt\x18\n" +
	" \x01(\v2$.cozy.worker.v1.JobCheckpointReceiptH\x00R\x11checkpointReceipt\x12@\n" +
	"\fsnapshot_ack\x18\v \x01(\v2\x1b.cozy.worker.v1.SnapshotAckH\x00R\vsnapshotAck\x12e\n" +
	"\x19artifact_finalize_request\x18\x11 \x01(\v2'.cozy.worker.v1.ArtifactFinalizeRequestH\x00R\x17artifactFinalizeRequest\x12M\n" +
	"\x11artifact_host_ack\x18\x13 \x01(\v2\x1f.cozy.worker.v1.ArtifactHostAckH\x00R\x0fartifactHostAck\x12e\n" +
	"\x19artifact_transfer_request\x18\x14 \x01(\v2'.cozy.worker.v1.ArtifactTransferRequestH\x00R\x17artifactTransferRequest\x12Y\n" +
	"\x15artifact_read_request\x18\x15 \x01(\v2#.cozy.worker.v1.ArtifactReadRequestH\x00R\x13artifactReadRequest\x12c\n" +
	"\x19model_source_file_request\x18\x16 \x01(\v2&.cozy.worker.v1.ModelSourceFileRequestH\x00R\x16modelSourceFileRequest\x12l\n" +
	"\x1cmodel_source_prepare_request\x18\x17 \x01(\v2).cozy.worker.v1.ModelSourcePrepareRequestH\x00R\x19modelSourcePrepareRequestB\x05\n" +
	"\x03msgJ\x04\b\x0e\x10\x0fJ\x04\b\x10\x10\x11J\x04\b\x12\x10\x13R\x15artifact_grant_updateR\x10ensure_artifacts\"\xe8\t\n" +
	"\vWorkerFrame\x127\n" +
	"\tclaim_ack\x18\x05 \x01(\v2\x18.cozy.worker.v1.ClaimAckH\x00R\bclaimAck\x12L\n" +
	"\x0eobserved_state\x18\x06 \x01(\v2#.cozy.worker.v1.ObservedWorkerStateH\x00R\robservedState\x12L\n" +
	"\x10attempt_accepted\x18\a \x01(\v2\x1f.cozy.worker.v1.AttemptAcceptedH\x00R\x0fattemptAccepted\x12I\n" +
	"\x0fattempt_outcome\x18\b \x01(\v2\x1e.cozy.worker.v1.AttemptOutcomeH\x00R\x0eattemptOutcome\x12@\n" +
	"\fboot_failure\x18\t \x01(\v2\x1b.cozy.worker.v1.BootFailureH\x00R\vbootFailure\x12U\n" +
	"\x12checkpoint_request\x18\n" +
	" \x01(\v2$.cozy.worker.v1.JobCheckpointRequestH\x00R\x11checkpointRequest\x12I\n" +
	"\x0echeckpoint_ack\x18\v \x01(\v2 .cozy.worker.v1.JobCheckpointAckH\x00R\rcheckpointAck\x12<\n" +
	"\bsnapshot\x18\f \x01(\v2\x1e.cozy.worker.v1.WorkerSnapshotH\x00R\bsnapshot\x12b\n" +
	"\x18artifact_finalize_result\x18\x10 \x01(\v2&.cozy.worker.v1.ArtifactFinalizeResultH\x00R\x16artifactFinalizeResult\x12N\n" +
	"\x0fartifact_intent\x18\x11 \x01(\v2#.cozy.worker.v1.ArtifactIntentFrameH\x00R\x0eartifactIntent\x12Q\n" +
	"\x10artifact_receipt\x18\x12 \x01(\v2$.cozy.worker.v1.ArtifactReceiptFrameH\x00R\x0fartifactReceipt\x12V\n" +
	"\x14artifact_read_result\x18\x13 \x01(\v2\".cozy.worker.v1.ArtifactReadResultH\x00R\x12artifactReadResult\x12b\n" +
	"\x18artifact_transfer_status\x18\x14 \x01(\v2&.cozy.worker.v1.ArtifactTransferStatusH\x00R\x16artifactTransferStatus\x12`\n" +
	"\x18model_source_file_status\x18\x15 \x01(\v2%.cozy.worker.v1.ModelSourceFileStatusH\x00R\x15modelSourceFileStatus\x12Y\n" +
	"\x15model_source_prepared\x18\x16 \x01(\v2#.cozy.worker.v1.ModelSourcePreparedH\x00R\x13modelSourcePreparedB\x05\n" +
	"\x03msgJ\x04\b\r\x10\x0eJ\x04\b\x0e\x10\x0fJ\x04\b\x0f\x10\x10\"\xb1\x02\n" +
	"\x05Claim\x12,\n" +
	"\x12record_owner_epoch\x18\x01 \x01(\x04R\x10recordOwnerEpoch\x12:\n" +
	"\x19control_stream_generation\x18\x02 \x01(\x04R\x17controlStreamGeneration\x12$\n" +
	"\x0eworker_boot_id\x18\x03 \x01(\tR\fworkerBootId\x12&\n" +
	"\x0frecord_owner_id\x18\x05 \x01(\tR\rrecordOwnerId\x12\x1b\n" +
	"\tworker_id\x18\x06 \x01(\tR\bworkerId\x12\x1d\n" +
	"\n" +
	"wire_minor\x18\a \x01(\rR\twireMinor\x12\x14\n" +
	"\x05proof\x18\b \x01(\fR\x05proofJ\x04\b\x04\x10\x05J\x04\b\t\x10\n" +
	"R\x12wire_schema_digest\"\xc0\x01\n" +
	"\n" +
	"ClaimProof\x12,\n" +
	"\x12record_owner_epoch\x18\x01 \x01(\x04R\x10recordOwnerEpoch\x12$\n" +
	"\x0eworker_boot_id\x18\x02 \x01(\tR\fworkerBootId\x12\x1b\n" +
	"\tworker_id\x18\x03 \x01(\tR\bworkerId\x12A\n" +
	"\x1dworker_tls_certificate_digest\x18\x04 \x01(\fR\x1aworkerTlsCertificateDigest\"\xbe\x04\n" +
	"\bClaimAck\x12,\n" +
	"\x12record_owner_epoch\x18\x01 \x01(\x04R\x10recordOwnerEpoch\x12:\n" +
	"\x19control_stream_generation\x18\x02 \x01(\x04R\x17controlStreamGeneration\x12$\n" +
	"\x0eworker_boot_id\x18\x03 \x01(\tR\fworkerBootId\x12\x1a\n" +
	"\baccepted\x18\x05 \x01(\bR\baccepted\x12<\n" +
	"\trejection\x18\x06 \x01(\x0e2\x1e.cozy.worker.v1.ClaimRejectionR\trejection\x12\x1d\n" +
	"\n" +
	"wire_minor\x18\a \x01(\rR\twireMinor\x12\x1b\n" +
	"\tworker_id\x18\b \x01(\tR\bworkerId\x12,\n" +
	"\x12worker_instance_id\x18\t \x01(\tR\x10workerInstanceId\x12*\n" +
	"\x11worker_release_id\x18\n" +
	" \x01(\tR\x0fworkerReleaseId\x124\n" +
	"\x16control_runtime_digest\x18\v \x01(\tR\x14controlRuntimeDigest\x12\x1d\n" +
	"\n" +
	"git_commit\x18\f \x01(\tR\tgitCommit\x12=\n" +
	"\tresources\x18\r \x01(\v2\x1f.cozy.worker.v1.WorkerResourcesR\tresourcesJ\x04\b\x04\x10\x05J\x04\b\x0e\x10\x0fR\x12wire_schema_digest\"\xe2\x03\n" +
	"\vBootFailure\x12,\n" +
	"\x12record_owner_epoch\x18\x01 \x01(\x04R\x10recordOwnerEpoch\x12:\n" +
	"\x19control_stream_generation\x18\x02 \x01(\x04R\x17controlStreamGeneration\x12$\n" +
	"\x0eworker_boot_id\x18\x03 \x01(\tR\fworkerBootId\x12\x1b\n" +
	"\tworker_id\x18\x05 \x01(\tR\bworkerId\x12,\n" +
	"\x12worker_instance_id\x18\x06 \x01(\tR\x10workerInstanceId\x129\n" +
	"\x06reason\x18\a \x01(\x0e2!.cozy.worker.v1.BootFailureReasonR\x06reason\x12\x16\n" +
	"\x06detail\x18\b \x01(\tR\x06detail\x12=\n" +
	"\tresources\x18\t \x01(\v2\x1f.cozy.worker.v1.WorkerResourcesR\tresources\x124\n" +
	"\x16control_runtime_digest\x18\n" +
	" \x01(\tR\x14controlRuntimeDigest\x12*\n" +
	"\x11worker_release_id\x18\v \x01(\tR\x0fworkerReleaseIdJ\x04\b\x04\x10\x05\"\xfe\x02\n" +
	"\x0eWorkerSnapshot\x12,\n" +
	"\x12record_owner_epoch\x18\x01 \x01(\x04R\x10recordOwnerEpoch\x12:\n" +
	"\x19control_stream_generation\x18\x02 \x01(\x04R\x17controlStreamGeneration\x12$\n" +
	"\x0eworker_boot_id\x18\x03 \x01(\tR\fworkerBootId\x12\x1f\n" +
	"\vsnapshot_id\x18\x05 \x01(\tR\n" +
	"snapshotId\x12'\n" +
	"\x0fsnapshot_digest\x18\x06 \x01(\fR\x0esnapshotDigest\x128\n" +
	"\x18snapshot_canonical_bytes\x18\a \x01(\fR\x16snapshotCanonicalBytes\x12R\n" +
	"&accepted_placement_set_canonical_bytes\x18\b \x01(\fR\"acceptedPlacementSetCanonicalBytesJ\x04\b\x04\x10\x05\"\xd1\x05\n" +
	"\x12WorkerSnapshotBody\x12E\n" +
	"\x1faccepted_desired_state_revision\x18\x01 \x01(\x04R\x1cacceptedDesiredStateRevision\x12A\n" +
	"\x1daccepted_placement_set_digest\x18\x02 \x01(\fR\x1aacceptedPlacementSetDigest\x12+\n" +
	"\x11journal_highwater\x18\x03 \x01(\x04R\x10journalHighwater\x12>\n" +
	"\fworker_phase\x18\x04 \x01(\x0e2\x1b.cozy.worker.v1.WorkerPhaseR\vworkerPhase\x12?\n" +
	"\n" +
	"placements\x18\x05 \x03(\v2\x1f.cozy.worker.v1.PlacementStatusR\n" +
	"placements\x12-\n" +
	"\x12converged_revision\x18\x06 \x01(\x04R\x11convergedRevision\x121\n" +
	"\x14admission_generation\x18\a \x01(\x04R\x13admissionGeneration\x12G\n" +
	"\x0fadmission_state\x18\b \x01(\x0e2\x1e.cozy.worker.v1.AdmissionStateR\x0eadmissionState\x126\n" +
	"\x17available_attempt_slots\x18\t \x01(\rR\x15availableAttemptSlots\x12@\n" +
	"\rheld_attempts\x18\n" +
	" \x03(\v2\x1b.cozy.worker.v1.HeldAttemptR\fheldAttempts\x12^\n" +
	"\x15artifact_transactions\x18\v \x03(\v2).cozy.worker.v1.ArtifactTransactionStatusR\x14artifactTransactions\"\xed\x01\n" +
	"\vSnapshotAck\x12,\n" +
	"\x12record_owner_epoch\x18\x01 \x01(\x04R\x10recordOwnerEpoch\x12:\n" +
	"\x19control_stream_generation\x18\x02 \x01(\x04R\x17controlStreamGeneration\x12$\n" +
	"\x0eworker_boot_id\x18\x03 \x01(\tR\fworkerBootId\x12\x1f\n" +
	"\vsnapshot_id\x18\x05 \x01(\tR\n" +
	"snapshotId\x12'\n" +
	"\x0fsnapshot_digest\x18\x06 \x01(\fR\x0esnapshotDigestJ\x04\b\x04\x10\x05\"\xa2\x04\n" +
	"\x12DesiredWorkerState\x12,\n" +
	"\x12record_owner_epoch\x18\x01 \x01(\x04R\x10recordOwnerEpoch\x12:\n" +
	"\x19control_stream_generation\x18\x02 \x01(\x04R\x17controlStreamGeneration\x12$\n" +
	"\x0eworker_boot_id\x18\x03 \x01(\tR\fworkerBootId\x12\x1a\n" +
	"\brevision\x18\x05 \x01(\x04R\brevision\x121\n" +
	"\aposture\x18\x06 \x01(\x0e2\x17.cozy.worker.v1.PostureR\aposture\x12\x1d\n" +
	"\n" +
	"wire_minor\x18\a \x01(\rR\twireMinor\x12$\n" +
	"\x0edrain_grace_ms\x18\b \x01(\x04R\fdrainGraceMs\x120\n" +
	"\x03job\x18\r \x01(\v2\x1c.cozy.worker.v1.JobDirectiveH\x00R\x03job\x12J\n" +
	"\rplacement_set\x18\x0f \x01(\v2#.cozy.worker.v1.DesiredPlacementSetH\x00R\fplacementSet\x12D\n" +
	"\vpackage_set\x18\x10 \x01(\v2!.cozy.worker.v1.DesiredPackageSetH\x00R\n" +
	"packageSetB\x06\n" +
	"\x04modeJ\x04\b\x04\x10\x05J\x04\b\t\x10\n" +
	"J\x04\b\n" +
	"\x10\vJ\x04\b\v\x10\fJ\x04\b\f\x10\r\"\x88\x01\n" +
	"\x11DesiredPackageSet\x12/\n" +
	"\x13download_delegation\x18\x01 \x01(\fR\x12downloadDelegation\x12B\n" +
	"\x1ddownload_delegation_signature\x18\x02 \x01(\fR\x1bdownloadDelegationSignature\"\x90\x01\n" +
	"\x13DesiredPlacementSet\x120\n" +
	"\x14placement_set_digest\x18\x01 \x01(\fR\x12placementSetDigest\x12A\n" +
	"\x1dplacement_set_canonical_bytes\x18\x03 \x01(\fR\x1aplacementSetCanonicalBytesJ\x04\b\x02\x10\x03\"I\n" +
	"\fPlacementSet\x129\n" +
	"\n" +
	"placements\x18\x01 \x03(\v2\x19.cozy.worker.v1.PlacementR\n" +
	"placements\"\xd9\x02\n" +
	"\x12DownloadDelegation\x12&\n" +
	"\x0fexpires_at_unix\x18\x01 \x01(\x04R\rexpiresAtUnix\x128\n" +
	"\x06models\x18\x02 \x03(\v2 .cozy.worker.v1.DownloadModelRefR\x06models\x12>\n" +
	"\bpackages\x18\x03 \x03(\v2\".cozy.worker.v1.DownloadPackageRefR\bpackages\x12\x1b\n" +
	"\trental_id\x18\x04 \x01(\tR\brentalId\x12$\n" +
	"\x0eworker_boot_id\x18\x05 \x01(\tR\fworkerBootId\x12\x1b\n" +
	"\tworker_id\x18\x06 \x01(\tR\bworkerId\x12A\n" +
	"\x1dworker_tls_certificate_digest\x18\a \x01(\fR\x1aworkerTlsCertificateDigest\"\x8c\x01\n" +
	"\x10DownloadModelRef\x12\x1a\n" +
	"\bmanifest\x18\x01 \x01(\tR\bmanifest\x12\x14\n" +
	"\x05model\x18\x02 \x01(\tR\x05model\x12\x18\n" +
	"\arelease\x18\x03 \x01(\tR\arelease\x12\x18\n" +
	"\apackage\x18\x04 \x01(\tR\apackage\x12\x12\n" +
	"\x04slot\x18\x05 \x01(\tR\x04slot\"o\n" +
	"\x12DownloadPackageRef\x12\x18\n" +
	"\apackage\x18\x01 \x01(\tR\apackage\x12\x18\n" +
	"\arelease\x18\x02 \x01(\tR\arelease\x12%\n" +
	"\x0erelease_digest\x18\x03 \x01(\tR\rreleaseDigest\"\xc3\x04\n" +
	"\tPlacement\x12!\n" +
	"\fplacement_id\x18\x01 \x01(\tR\vplacementId\x12<\n" +
	"\apackage\x18\x02 \x01(\v2 .cozy.worker.v1.PackageSelectionH\x00R\apackage\x12F\n" +
	"\vdevelopment\x18\v \x01(\v2\".cozy.worker.v1.DevelopmentPackageH\x00R\vdevelopment\x12-\n" +
	"\x12environment_digest\x18\x03 \x01(\fR\x11environmentDigest\x12B\n" +
	"\x12package_descriptor\x18\x05 \x01(\v2\x13.cozy.worker.v1.RefR\x11packageDescriptor\x12'\n" +
	"\x0fbindings_digest\x18\x06 \x01(\fR\x0ebindingsDigest\x12-\n" +
	"\x06models\x18\a \x03(\v2\x15.cozy.worker.v1.ModelR\x06models\x12<\n" +
	"\ventrypoints\x18\b \x03(\v2\x1a.cozy.worker.v1.EntrypointR\ventrypoints\x12=\n" +
	"\venvironment\x18\n" +
	" \x01(\v2\x1b.cozy.worker.v1.EnvironmentR\venvironmentB\x0e\n" +
	"\fpackage_modeJ\x04\b\x04\x10\x05J\x04\b\t\x10\n" +
	"R\x1aenvironment_receipt_digestR\rqualification\"5\n" +
	"\x03Ref\x12\x16\n" +
	"\x06digest\x18\x01 \x01(\fR\x06digest\x12\x16\n" +
	"\x06length\x18\x02 \x01(\x04R\x06length\"\xc3\x01\n" +
	"\tWheelFact\x12%\n" +
	"\x03ref\x18\x01 \x01(\v2\x13.cozy.worker.v1.RefR\x03ref\x12\"\n" +
	"\fdistribution\x18\x02 \x01(\tR\fdistribution\x12\x18\n" +
	"\aversion\x18\x03 \x01(\tR\aversion\x12\x1a\n" +
	"\bfilename\x18\x04 \x01(\tR\bfilename\x12!\n" +
	"\fimport_roots\x18\x05 \x03(\tR\vimportRoots\x12\x12\n" +
	"\x04tags\x18\x06 \x03(\tR\x04tags\"\xad\x01\n" +
	"\x10PackageSelection\x12\x18\n" +
	"\apackage\x18\x01 \x01(\tR\apackage\x12\x18\n" +
	"\arelease\x18\x02 \x01(\tR\arelease\x12%\n" +
	"\x0erelease_digest\x18\x03 \x01(\fR\rreleaseDigest\x12>\n" +
	"\rproject_wheel\x18\x04 \x01(\v2\x19.cozy.worker.v1.WheelFactR\fprojectWheel\"m\n" +
	"\x12DevelopmentPackage\x12\x18\n" +
	"\apackage\x18\x01 \x01(\tR\apackage\x12\x18\n" +
	"\arelease\x18\x02 \x01(\tR\arelease\x12#\n" +
	"\rsource_digest\x18\x03 \x01(\fR\fsourceDigest\"[\n" +
	"\vEnvironment\x121\n" +
	"\x06wheels\x18\x02 \x03(\v2\x19.cozy.worker.v1.WheelFactR\x06wheelsJ\x04\b\x01\x10\x02R\x13wheelhouse_manifest\"\x8a\x01\n" +
	"\x05Model\x12\x0e\n" +
	"\x02id\x18\x01 \x01(\tR\x02id\x12\x12\n" +
	"\x04repo\x18\x02 \x01(\tR\x04repo\x12\x18\n" +
	"\aversion\x18\x03 \x01(\tR\aversion\x12\x12\n" +
	"\x04lane\x18\x04 \x01(\tR\x04lane\x12/\n" +
	"\bmanifest\x18\x05 \x01(\v2\x13.cozy.worker.v1.RefR\bmanifest\"\x88\x01\n" +
	"\n" +
	"Entrypoint\x12\x12\n" +
	"\x04name\x18\x01 \x01(\tR\x04name\x12:\n" +
	"\x19entrypoint_binding_digest\x18\x02 \x01(\fR\x17entrypointBindingDigest\x12*\n" +
	"\x05slots\x18\x03 \x03(\v2\x14.cozy.worker.v1.SlotR\x05slots\"\xb7\x02\n" +
	"\x04Slot\x12\x12\n" +
	"\x04slot\x18\x01 \x01(\tR\x04slot\x12,\n" +
	"\x12reference_model_id\x18\x02 \x01(\tR\x10referenceModelId\x12.\n" +
	"\x06config\x18\x03 \x01(\v2\x16.cozy.worker.v1.ConfigR\x06config\x129\n" +
	"\n" +
	"components\x18\x04 \x03(\v2\x19.cozy.worker.v1.ComponentR\n" +
	"components\x12S\n" +
	"\x1bmodel_construction_contract\x18\x05 \x01(\v2\x13.cozy.worker.v1.RefR\x19modelConstructionContract\x12-\n" +
	"\x06stamps\x18\x06 \x03(\v2\x15.cozy.worker.v1.StampR\x06stamps\"h\n" +
	"\x06Config\x12/\n" +
	"\bdocument\x18\x01 \x01(\v2\x13.cozy.worker.v1.RefR\bdocument\x12-\n" +
	"\x06assets\x18\x02 \x03(\v2\x15.cozy.worker.v1.AssetR\x06assets\"]\n" +
	"\x05Asset\x12\x12\n" +
	"\x04name\x18\x01 \x01(\tR\x04name\x12\x19\n" +
	"\bmodel_id\x18\x02 \x01(\tR\amodelId\x12%\n" +
	"\x03ref\x18\x03 \x01(\v2\x13.cozy.worker.v1.RefR\x03ref\"D\n" +
	"\tComponent\x12\x1c\n" +
	"\tcomponent\x18\x01 \x01(\tR\tcomponent\x12\x19\n" +
	"\bmodel_id\x18\x02 \x01(\tR\amodelId\"O\n" +
	"\x05Stamp\x12\x1c\n" +
	"\tcomponent\x18\x01 \x01(\tR\tcomponent\x12\x10\n" +
	"\x03key\x18\x02 \x01(\tR\x03key\x12\x16\n" +
	"\x06values\x18\x03 \x03(\tR\x06values\"\xc3\x02\n" +
	"\fJobDirective\x12\x19\n" +
	"\bbuild_id\x18\x01 \x01(\tR\abuildId\x12*\n" +
	"\x11job_descriptor_id\x18\x02 \x01(\tR\x0fjobDescriptorId\x12A\n" +
	"\rresource_caps\x18\x03 \x01(\v2\x1c.cozy.worker.v1.ResourceCapsR\fresourceCaps\x12V\n" +
	"\x14publication_contract\x18\x04 \x01(\v2#.cozy.worker.v1.PublicationContractR\x13publicationContract\x12.\n" +
	"\x13reclaim_on_terminal\x18\x05 \x01(\bR\x11reclaimOnTerminal\x12!\n" +
	"\fdevice_count\x18\x06 \x01(\rR\vdeviceCount\"\xa7\b\n" +
	"\x13ObservedWorkerState\x12,\n" +
	"\x12record_owner_epoch\x18\x01 \x01(\x04R\x10recordOwnerEpoch\x12:\n" +
	"\x19control_stream_generation\x18\x02 \x01(\x04R\x17controlStreamGeneration\x12$\n" +
	"\x0eworker_boot_id\x18\x03 \x01(\tR\fworkerBootId\x12,\n" +
	"\x12applied_wire_minor\x18\b \x01(\rR\x10appliedWireMinor\x12@\n" +
	"\rheld_attempts\x18\t \x03(\v2\x1b.cozy.worker.v1.HeldAttemptR\fheldAttempts\x12-\n" +
	"\x06faults\x18\n" +
	" \x03(\v2\x15.cozy.worker.v1.FaultR\x06faults\x129\n" +
	"\bactivity\x18\v \x03(\v2\x1d.cozy.worker.v1.ActivityEventR\bactivity\x12>\n" +
	"\fjob_capacity\x18\r \x01(\v2\x1b.cozy.worker.v1.JobCapacityR\vjobCapacity\x12?\n" +
	"\n" +
	"placements\x18\x0f \x03(\v2\x1f.cozy.worker.v1.PlacementStatusR\n" +
	"placements\x121\n" +
	"\x14admission_generation\x18\x10 \x01(\x04R\x13admissionGeneration\x12G\n" +
	"\x0fadmission_state\x18\x11 \x01(\x0e2\x1e.cozy.worker.v1.AdmissionStateR\x0eadmissionState\x126\n" +
	"\x17available_attempt_slots\x18\x12 \x01(\rR\x15availableAttemptSlots\x12E\n" +
	"\x1faccepted_desired_state_revision\x18\x13 \x01(\x04R\x1cacceptedDesiredStateRevision\x12A\n" +
	"\x1daccepted_placement_set_digest\x18\x14 \x01(\fR\x1aacceptedPlacementSetDigest\x12-\n" +
	"\x12converged_revision\x18\x15 \x01(\x04R\x11convergedRevision\x12>\n" +
	"\fworker_phase\x18\x16 \x01(\x0e2\x1b.cozy.worker.v1.WorkerPhaseR\vworkerPhaseJ\x04\b\x04\x10\x05J\x04\b\x05\x10\x06J\x04\b\x06\x10\aJ\x04\b\a\x10\bJ\x04\b\f\x10\rJ\x04\b\x0e\x10\x0fJ\x04\b\x17\x10\x18J\x04\b\x18\x10\x19J\x04\b\x19\x10\x1aR\x19applied_artifact_grant_idR\x16applied_grant_revisionR\x0fartifact_intent\"\xe7\x06\n" +
	"\x0fPlacementStatus\x12!\n" +
	"\fplacement_id\x18\x01 \x01(\tR\vplacementId\x12/\n" +
	"\x13executor_generation\x18\x04 \x01(\x04R\x12executorGeneration\x12@\n" +
	"\x1cdispatchable_binding_digests\x18\x06 \x03(\fR\x1adispatchableBindingDigests\x12D\n" +
	"\x1ematerializable_binding_digests\x18\a \x03(\fR\x1cmaterializableBindingDigests\x12-\n" +
	"\x06faults\x18\b \x03(\v2\x15.cozy.worker.v1.FaultR\x06faults\x12J\n" +
	"\vaccelerator\x18\t \x01(\v2(.cozy.worker.v1.AcceleratorQualificationR\vaccelerator\x120\n" +
	"\x14placement_set_digest\x18\v \x01(\fR\x12placementSetDigest\x12N\n" +
	"\x0fmaterialization\x18\f \x01(\x0e2$.cozy.worker.v1.MaterializationStateR\x0fmaterialization\x126\n" +
	"\aserving\x18\r \x01(\x0e2\x1c.cozy.worker.v1.ServingStateR\aserving\x12R\n" +
	"&retained_fallback_placement_set_digest\x18\x0e \x01(\fR\"retainedFallbackPlacementSetDigest\x12Q\n" +
	"\vacquisition\x18\x0f \x01(\v2/.cozy.worker.v1.PlacementAcquisitionObservationR\vacquisition\x126\n" +
	"\x17package_revision_digest\x18\x10 \x01(\tR\x15packageRevisionDigest\x12-\n" +
	"\x12environment_digest\x18\x11 \x01(\tR\x11environmentDigest\x12#\n" +
	"\rconfig_digest\x18\x12 \x01(\tR\fconfigDigestJ\x04\b\x02\x10\x03J\x04\b\x03\x10\x04J\x04\b\x05\x10\x06\"\xa7\x01\n" +
	"\x1fPlacementAcquisitionObservation\x12C\n" +
	"\apackage\x18\x01 \x01(\v2).cozy.worker.v1.AcquisitionLegObservationR\apackage\x12?\n" +
	"\x05model\x18\x02 \x01(\v2).cozy.worker.v1.AcquisitionLegObservationR\x05model\"\xc9\x01\n" +
	"\x19AcquisitionLegObservation\x120\n" +
	"\x14started_monotonic_ns\x18\x01 \x01(\x04R\x12startedMonotonicNs\x12,\n" +
	"\x12ended_monotonic_ns\x18\x02 \x01(\x04R\x10endedMonotonicNs\x12)\n" +
	"\x10downloaded_bytes\x18\x03 \x01(\x04R\x0fdownloadedBytes\x12!\n" +
	"\freused_bytes\x18\x04 \x01(\x04R\vreusedBytes\"\xe1\x02\n" +
	"\x18AcceleratorQualification\x12\x1c\n" +
	"\tqualified\x18\x01 \x01(\bR\tqualified\x12\"\n" +
	"\fcapabilities\x18\x02 \x03(\tR\fcapabilities\x122\n" +
	"\x15recommended_max_bytes\x18\x03 \x01(\x04R\x13recommendedMaxBytes\x122\n" +
	"\x15pinned_capacity_bytes\x18\x04 \x01(\x04R\x13pinnedCapacityBytes\x12*\n" +
	"\x11h2d_gbps_measured\x18\x05 \x01(\rR\x0fh2dGbpsMeasured\x12*\n" +
	"\x11d2h_gbps_measured\x18\x06 \x01(\rR\x0fd2hGbpsMeasured\x12#\n" +
	"\rtorch_version\x18\a \x01(\tR\ftorchVersion\x12\x1e\n" +
	"\n" +
	"unreadable\x18\b \x03(\tR\n" +
	"unreadable\"g\n" +
	"\rActivityEvent\x12\x10\n" +
	"\x03seq\x18\x01 \x01(\x04R\x03seq\x12\x12\n" +
	"\x04kind\x18\x02 \x01(\tR\x04kind\x12\x12\n" +
	"\x04step\x18\x03 \x01(\tR\x04step\x12\x1c\n" +
	"\n" +
	"at_unix_ms\x18\x04 \x01(\x04R\batUnixMs\"\xf4\x03\n" +
	"\fAttemptOffer\x12,\n" +
	"\x12record_owner_epoch\x18\x01 \x01(\x04R\x10recordOwnerEpoch\x12:\n" +
	"\x19control_stream_generation\x18\x02 \x01(\x04R\x17controlStreamGeneration\x12$\n" +
	"\x0eworker_boot_id\x18\x03 \x01(\tR\fworkerBootId\x12\x1d\n" +
	"\n" +
	"request_id\x18\x05 \x01(\tR\trequestId\x12'\n" +
	"\x0fattempt_ordinal\x18\x06 \x01(\x04R\x0eattemptOrdinal\x124\n" +
	"\x16invocation_spec_digest\x18\a \x01(\fR\x14invocationSpecDigest\x123\n" +
	"\x05grant\x18\b \x01(\v2\x1d.cozy.worker.v1.DeliveryGrantR\x05grant\x12E\n" +
	"\x1finvocation_spec_canonical_bytes\x18\t \x01(\fR\x1cinvocationSpecCanonicalBytes\x12!\n" +
	"\fplacement_id\x18\n" +
	" \x01(\tR\vplacementId\x121\n" +
	"\x14admission_generation\x18\f \x01(\x04R\x13admissionGenerationJ\x04\b\x04\x10\x05\"\x8e\x04\n" +
	"\x0fAttemptAccepted\x12,\n" +
	"\x12record_owner_epoch\x18\x01 \x01(\x04R\x10recordOwnerEpoch\x12:\n" +
	"\x19control_stream_generation\x18\x02 \x01(\x04R\x17controlStreamGeneration\x12$\n" +
	"\x0eworker_boot_id\x18\x03 \x01(\tR\fworkerBootId\x12\x1d\n" +
	"\n" +
	"request_id\x18\x05 \x01(\tR\trequestId\x12'\n" +
	"\x0fattempt_ordinal\x18\x06 \x01(\x04R\x0eattemptOrdinal\x124\n" +
	"\x16invocation_spec_digest\x18\a \x01(\fR\x14invocationSpecDigest\x12\x1f\n" +
	"\vplan_digest\x18\b \x01(\fR\n" +
	"planDigest\x12:\n" +
	"\x19model_construction_digest\x18\t \x01(\fR\x17modelConstructionDigest\x126\n" +
	"\x04plan\x18\n" +
	" \x01(\v2\".cozy.worker.v1.AttemptPlanSummaryR\x04plan\x12!\n" +
	"\fplacement_id\x18\v \x01(\tR\vplacementId\x12/\n" +
	"\x13executor_generation\x18\f \x01(\x04R\x12executorGenerationJ\x04\b\x04\x10\x05\"\xc1\x02\n" +
	"\x12AttemptPlanSummary\x12\x1a\n" +
	"\bdelivery\x18\x01 \x01(\tR\bdelivery\x12(\n" +
	"\x0fmaterialization\x18\x02 \x01(\tR\x0fmaterialization\x12#\n" +
	"\rcompute_dtype\x18\x03 \x01(\tR\fcomputeDtype\x12\x1c\n" +
	"\tplacement\x18\x04 \x01(\tR\tplacement\x12?\n" +
	"\x1creserved_device_memory_bytes\x18\x05 \x01(\x04R\x19reservedDeviceMemoryBytes\x12.\n" +
	"\x13reserved_host_bytes\x18\x06 \x01(\x04R\x11reservedHostBytes\x121\n" +
	"\x14decision_explanation\x18\a \x01(\tR\x13decisionExplanation\"\xf4\x02\n" +
	"\rCancelAttempt\x12,\n" +
	"\x12record_owner_epoch\x18\x01 \x01(\x04R\x10recordOwnerEpoch\x12:\n" +
	"\x19control_stream_generation\x18\x02 \x01(\x04R\x17controlStreamGeneration\x12$\n" +
	"\x0eworker_boot_id\x18\x03 \x01(\tR\fworkerBootId\x12\x1d\n" +
	"\n" +
	"request_id\x18\x05 \x01(\tR\trequestId\x12'\n" +
	"\x0fattempt_ordinal\x18\x06 \x01(\x04R\x0eattemptOrdinal\x124\n" +
	"\x06reason\x18\a \x01(\x0e2\x1c.cozy.worker.v1.CancelReasonR\x06reason\x12\x19\n" +
	"\bgrace_ms\x18\b \x01(\x04R\agraceMs\x124\n" +
	"\x16invocation_spec_digest\x18\t \x01(\fR\x14invocationSpecDigestJ\x04\b\x04\x10\x05\"\xc5\x03\n" +
	"\x0eAttemptOutcome\x12,\n" +
	"\x12record_owner_epoch\x18\x01 \x01(\x04R\x10recordOwnerEpoch\x12:\n" +
	"\x19control_stream_generation\x18\x02 \x01(\x04R\x17controlStreamGeneration\x12$\n" +
	"\x0eworker_boot_id\x18\x03 \x01(\tR\fworkerBootId\x12\x1d\n" +
	"\n" +
	"request_id\x18\x05 \x01(\tR\trequestId\x12'\n" +
	"\x0fattempt_ordinal\x18\x06 \x01(\x04R\x0eattemptOrdinal\x124\n" +
	"\x16invocation_spec_digest\x18\a \x01(\fR\x14invocationSpecDigest\x12\x1d\n" +
	"\n" +
	"outcome_id\x18\b \x01(\tR\toutcomeId\x12%\n" +
	"\x0eoutcome_digest\x18\t \x01(\fR\routcomeDigest\x126\n" +
	"\x17outcome_canonical_bytes\x18\n" +
	" \x01(\fR\x15outcomeCanonicalBytes\x12!\n" +
	"\fplacement_id\x18\v \x01(\tR\vplacementIdJ\x04\b\x04\x10\x05\"\x9f\x05\n" +
	"\x12AttemptOutcomeBody\x12\x1d\n" +
	"\n" +
	"request_id\x18\x01 \x01(\tR\trequestId\x12'\n" +
	"\x0fattempt_ordinal\x18\x02 \x01(\x04R\x0eattemptOrdinal\x124\n" +
	"\x16invocation_spec_digest\x18\x03 \x01(\tR\x14invocationSpecDigest\x125\n" +
	"\x06status\x18\x04 \x01(\x0e2\x1d.cozy.worker.v1.OutcomeStatusR\x06status\x12G\n" +
	"\x0foutput_manifest\x18\x05 \x01(\v2\x1e.cozy.worker.v1.OutputManifestR\x0eoutputManifest\x128\n" +
	"\ametrics\x18\x06 \x01(\v2\x1e.cozy.worker.v1.AttemptMetricsR\ametrics\x12D\n" +
	"\rtriage_bundle\x18\a \x01(\v2\x1f.cozy.worker.v1.TriageBundleRefR\ftriageBundle\x12!\n" +
	"\fsafe_message\x18\b \x01(\tR\vsafeMessage\x122\n" +
	"\x05cause\x18\t \x01(\v2\x1c.cozy.worker.v1.OutcomeCauseR\x05cause\x126\n" +
	"\x06result\x18\n" +
	" \x01(\v2\x1e.cozy.worker.v1.ResultEnvelopeR\x06result\x12+\n" +
	"\x11execution_started\x18\v \x01(\bR\x10executionStarted\x12O\n" +
	"\x11artifact_receipts\x18\f \x03(\v2\".cozy.worker.v1.ArtifactReceiptRefR\x10artifactReceipts\"\x95\x01\n" +
	"\x12ArtifactReceiptRef\x126\n" +
	"\x17artifact_receipt_digest\x18\x01 \x01(\fR\x15artifactReceiptDigest\x12G\n" +
	" artifact_receipt_canonical_bytes\x18\x02 \x01(\fR\x1dartifactReceiptCanonicalBytes\"\xf4\x02\n" +
	"\x0fArtifactReceipt\x122\n" +
	"\x15owner_authority_scope\x18\x01 \x01(\tR\x13ownerAuthorityScope\x12\x1d\n" +
	"\n" +
	"request_id\x18\x02 \x01(\tR\trequestId\x124\n" +
	"\x16invocation_spec_digest\x18\x03 \x01(\tR\x14invocationSpecDigest\x12\x1f\n" +
	"\voutput_slot\x18\x04 \x01(\tR\n" +
	"outputSlot\x126\n" +
	"\x17artifact_transaction_id\x18\x05 \x01(\tR\x15artifactTransactionId\x126\n" +
	"\x17tensorfs_receipt_digest\x18\x06 \x01(\tR\x15tensorfsReceiptDigest\x12G\n" +
	" tensorfs_receipt_canonical_bytes\x18\a \x01(\fR\x1dtensorfsReceiptCanonicalBytes\"j\n" +
	"\x14ArtifactObjectSource\x12\x1b\n" +
	"\tobject_id\x18\x01 \x01(\tR\bobjectId\x12\x16\n" +
	"\x06length\x18\x02 \x01(\x04R\x06length\x12\x1d\n" +
	"\n" +
	"source_ref\x18\x03 \x01(\tR\tsourceRef\"\x9b\x04\n" +
	"\x13ArtifactIntentFrame\x12,\n" +
	"\x12record_owner_epoch\x18\x01 \x01(\x04R\x10recordOwnerEpoch\x12:\n" +
	"\x19control_stream_generation\x18\x02 \x01(\x04R\x17controlStreamGeneration\x12$\n" +
	"\x0eworker_boot_id\x18\x03 \x01(\tR\fworkerBootId\x12\x1d\n" +
	"\n" +
	"request_id\x18\x05 \x01(\tR\trequestId\x12'\n" +
	"\x0fattempt_ordinal\x18\x06 \x01(\x04R\x0eattemptOrdinal\x124\n" +
	"\x16invocation_spec_digest\x18\a \x01(\fR\x14invocationSpecDigest\x12\x1f\n" +
	"\voutput_slot\x18\b \x01(\tR\n" +
	"outputSlot\x12>\n" +
	"\x1btensorfs_declaration_digest\x18\t \x01(\fR\x19tensorfsDeclarationDigest\x12>\n" +
	"\x1brequested_writer_generation\x18\n" +
	" \x01(\x04R\x19requestedWriterGeneration\x12O\n" +
	"$tensorfs_declaration_canonical_bytes\x18\v \x01(\fR!tensorfsDeclarationCanonicalBytesJ\x04\b\x04\x10\x05\"\xa2\x06\n" +
	"\x0fArtifactHostAck\x12,\n" +
	"\x12record_owner_epoch\x18\x01 \x01(\x04R\x10recordOwnerEpoch\x12:\n" +
	"\x19control_stream_generation\x18\x02 \x01(\x04R\x17controlStreamGeneration\x12$\n" +
	"\x0eworker_boot_id\x18\x03 \x01(\tR\fworkerBootId\x12\x1d\n" +
	"\n" +
	"request_id\x18\x05 \x01(\tR\trequestId\x12'\n" +
	"\x0fattempt_ordinal\x18\x06 \x01(\x04R\x0eattemptOrdinal\x124\n" +
	"\x16invocation_spec_digest\x18\a \x01(\fR\x14invocationSpecDigest\x12\x1f\n" +
	"\voutput_slot\x18\b \x01(\tR\n" +
	"outputSlot\x126\n" +
	"\x17artifact_transaction_id\x18\t \x01(\tR\x15artifactTransactionId\x12+\n" +
	"\x11writer_generation\x18\n" +
	" \x01(\x04R\x10writerGeneration\x12>\n" +
	"\x1btensorfs_declaration_digest\x18\v \x01(\fR\x19tensorfsDeclarationDigest\x127\n" +
	"\x05stage\x18\f \x01(\x0e2!.cozy.worker.v1.ArtifactHostStageR\x05stage\x12=\n" +
	"\aoutcome\x18\r \x01(\x0e2#.cozy.worker.v1.ArtifactHostOutcomeR\aoutcome\x12=\n" +
	"\arefusal\x18\x0e \x01(\x0e2#.cozy.worker.v1.ArtifactHostRefusalR\arefusal\x12M\n" +
	"\x10artifact_receipt\x18\x0f \x01(\v2\".cozy.worker.v1.ArtifactReceiptRefR\x0fartifactReceipt\x12/\n" +
	"\bmanifest\x18\x10 \x01(\v2\x13.cozy.worker.v1.RefR\bmanifestJ\x04\b\x04\x10\x05\"\xb0\x05\n" +
	"\x14ArtifactReceiptFrame\x12,\n" +
	"\x12record_owner_epoch\x18\x01 \x01(\x04R\x10recordOwnerEpoch\x12:\n" +
	"\x19control_stream_generation\x18\x02 \x01(\x04R\x17controlStreamGeneration\x12$\n" +
	"\x0eworker_boot_id\x18\x03 \x01(\tR\fworkerBootId\x12\x1d\n" +
	"\n" +
	"request_id\x18\x05 \x01(\tR\trequestId\x12'\n" +
	"\x0fattempt_ordinal\x18\x06 \x01(\x04R\x0eattemptOrdinal\x124\n" +
	"\x16invocation_spec_digest\x18\a \x01(\fR\x14invocationSpecDigest\x12\x1f\n" +
	"\voutput_slot\x18\b \x01(\tR\n" +
	"outputSlot\x126\n" +
	"\x17artifact_transaction_id\x18\t \x01(\tR\x15artifactTransactionId\x12+\n" +
	"\x11writer_generation\x18\n" +
	" \x01(\x04R\x10writerGeneration\x12>\n" +
	"\x1btensorfs_declaration_digest\x18\v \x01(\fR\x19tensorfsDeclarationDigest\x12M\n" +
	"\x10artifact_receipt\x18\f \x01(\v2\".cozy.worker.v1.ArtifactReceiptRefR\x0fartifactReceipt\x12>\n" +
	"\aobjects\x18\r \x03(\v2$.cozy.worker.v1.ArtifactObjectSourceR\aobjects\x12/\n" +
	"\bmanifest\x18\x0e \x01(\v2\x13.cozy.worker.v1.RefR\bmanifestJ\x04\b\x04\x10\x05\"\xd7\x03\n" +
	"\x19ArtifactTransactionStatus\x126\n" +
	"\x17artifact_transaction_id\x18\x01 \x01(\tR\x15artifactTransactionId\x12\x1d\n" +
	"\n" +
	"request_id\x18\x02 \x01(\tR\trequestId\x12'\n" +
	"\x0fattempt_ordinal\x18\x03 \x01(\x04R\x0eattemptOrdinal\x124\n" +
	"\x16invocation_spec_digest\x18\x04 \x01(\tR\x14invocationSpecDigest\x12\x1f\n" +
	"\voutput_slot\x18\x05 \x01(\tR\n" +
	"outputSlot\x12+\n" +
	"\x11writer_generation\x18\x06 \x01(\x04R\x10writerGeneration\x12>\n" +
	"\x05state\x18\a \x01(\x0e2(.cozy.worker.v1.ArtifactTransactionStateR\x05state\x12>\n" +
	"\x1btensorfs_declaration_digest\x18\b \x01(\fR\x19tensorfsDeclarationDigest\x126\n" +
	"\x17artifact_receipt_digest\x18\t \x01(\fR\x15artifactReceiptDigest\"\x9a\x03\n" +
	"\x13ArtifactReadRequest\x12,\n" +
	"\x12record_owner_epoch\x18\x01 \x01(\x04R\x10recordOwnerEpoch\x12:\n" +
	"\x19control_stream_generation\x18\x02 \x01(\x04R\x17controlStreamGeneration\x12$\n" +
	"\x0eworker_boot_id\x18\x03 \x01(\tR\fworkerBootId\x126\n" +
	"\x17artifact_transaction_id\x18\x05 \x01(\tR\x15artifactTransactionId\x12+\n" +
	"\x11writer_generation\x18\x06 \x01(\x04R\x10writerGeneration\x12\x1b\n" +
	"\tobject_id\x18\a \x01(\tR\bobjectId\x12\x1d\n" +
	"\n" +
	"source_ref\x18\b \x01(\tR\tsourceRef\x12\x17\n" +
	"\aread_id\x18\t \x01(\tR\x06readId\x12\x16\n" +
	"\x06offset\x18\n" +
	" \x01(\x04R\x06offset\x12\x1b\n" +
	"\tmax_bytes\x18\v \x01(\rR\bmaxBytesJ\x04\b\x04\x10\x05\"\xd0\x04\n" +
	"\x12ArtifactReadResult\x12,\n" +
	"\x12record_owner_epoch\x18\x01 \x01(\x04R\x10recordOwnerEpoch\x12:\n" +
	"\x19control_stream_generation\x18\x02 \x01(\x04R\x17controlStreamGeneration\x12$\n" +
	"\x0eworker_boot_id\x18\x03 \x01(\tR\fworkerBootId\x126\n" +
	"\x17artifact_transaction_id\x18\x05 \x01(\tR\x15artifactTransactionId\x12+\n" +
	"\x11writer_generation\x18\x06 \x01(\x04R\x10writerGeneration\x12\x1b\n" +
	"\tobject_id\x18\a \x01(\tR\bobjectId\x12\x1d\n" +
	"\n" +
	"source_ref\x18\b \x01(\tR\tsourceRef\x12\x17\n" +
	"\aread_id\x18\t \x01(\tR\x06readId\x12\x16\n" +
	"\x06offset\x18\n" +
	" \x01(\x04R\x06offset\x12=\n" +
	"\aoutcome\x18\v \x01(\x0e2#.cozy.worker.v1.ArtifactReadOutcomeR\aoutcome\x12\x12\n" +
	"\x04data\x18\f \x01(\fR\x04data\x12\x1f\n" +
	"\vdata_digest\x18\r \x01(\fR\n" +
	"dataDigest\x12=\n" +
	"\arefusal\x18\x0e \x01(\x0e2#.cozy.worker.v1.ArtifactReadRefusalR\arefusal\x12\x1f\n" +
	"\vsafe_detail\x18\x0f \x01(\tR\n" +
	"safeDetailJ\x04\b\x04\x10\x05\"H\n" +
	"\x11ArtifactObjectRef\x12\x1b\n" +
	"\tobject_id\x18\x01 \x01(\tR\bobjectId\x12\x16\n" +
	"\x06length\x18\x02 \x01(\x04R\x06length\"@\n" +
	"\x14ArtifactUploadHeader\x12\x12\n" +
	"\x04name\x18\x01 \x01(\tR\x04name\x12\x14\n" +
	"\x05value\x18\x02 \x01(\tR\x05value\"\xd5\x01\n" +
	"\x13ArtifactUploadGrant\x12\x1b\n" +
	"\tobject_id\x18\x01 \x01(\tR\bobjectId\x12\x16\n" +
	"\x06length\x18\x02 \x01(\x04R\x06length\x12\x10\n" +
	"\x03url\x18\x03 \x01(\tR\x03url\x12O\n" +
	"\x10required_headers\x18\x04 \x03(\v2$.cozy.worker.v1.ArtifactUploadHeaderR\x0frequiredHeaders\x12&\n" +
	"\x0fexpires_at_unix\x18\x05 \x01(\x04R\rexpiresAtUnix\"\x97\x05\n" +
	"\x17ArtifactTransferRequest\x12,\n" +
	"\x12record_owner_epoch\x18\x01 \x01(\x04R\x10recordOwnerEpoch\x12:\n" +
	"\x19control_stream_generation\x18\x02 \x01(\x04R\x17controlStreamGeneration\x12$\n" +
	"\x0eworker_boot_id\x18\x03 \x01(\tR\fworkerBootId\x12\x1d\n" +
	"\n" +
	"request_id\x18\x05 \x01(\tR\trequestId\x12'\n" +
	"\x0fattempt_ordinal\x18\x06 \x01(\x04R\x0eattemptOrdinal\x124\n" +
	"\x16invocation_spec_digest\x18\a \x01(\fR\x14invocationSpecDigest\x12\x1f\n" +
	"\voutput_slot\x18\b \x01(\tR\n" +
	"outputSlot\x126\n" +
	"\x17artifact_transaction_id\x18\t \x01(\tR\x15artifactTransactionId\x126\n" +
	"\x17artifact_receipt_digest\x18\n" +
	" \x01(\fR\x15artifactReceiptDigest\x12!\n" +
	"\foperation_id\x18\v \x01(\tR\voperationId\x12%\n" +
	"\x0egrant_revision\x18\f \x01(\x04R\rgrantRevision\x12H\n" +
	"\fupload_grant\x18\r \x01(\v2#.cozy.worker.v1.ArtifactUploadGrantH\x00R\vuploadGrant\x127\n" +
	"\x04held\x18\x0e \x01(\v2!.cozy.worker.v1.ArtifactObjectRefH\x00R\x04heldB\n" +
	"\n" +
	"\bdecisionJ\x04\b\x04\x10\x05\"\xcf\x06\n" +
	"\x16ArtifactTransferStatus\x12,\n" +
	"\x12record_owner_epoch\x18\x01 \x01(\x04R\x10recordOwnerEpoch\x12:\n" +
	"\x19control_stream_generation\x18\x02 \x01(\x04R\x17controlStreamGeneration\x12$\n" +
	"\x0eworker_boot_id\x18\x03 \x01(\tR\fworkerBootId\x12\x1d\n" +
	"\n" +
	"request_id\x18\x05 \x01(\tR\trequestId\x12'\n" +
	"\x0fattempt_ordinal\x18\x06 \x01(\x04R\x0eattemptOrdinal\x124\n" +
	"\x16invocation_spec_digest\x18\a \x01(\fR\x14invocationSpecDigest\x12\x1f\n" +
	"\voutput_slot\x18\b \x01(\tR\n" +
	"outputSlot\x126\n" +
	"\x17artifact_transaction_id\x18\t \x01(\tR\x15artifactTransactionId\x12!\n" +
	"\foperation_id\x18\n" +
	" \x01(\tR\voperationId\x12%\n" +
	"\x0egrant_revision\x18\v \x01(\x04R\rgrantRevision\x12\x1b\n" +
	"\tobject_id\x18\f \x01(\tR\bobjectId\x12\x16\n" +
	"\x06length\x18\r \x01(\x04R\x06length\x12'\n" +
	"\x0fupdate_sequence\x18\x0e \x01(\x04R\x0eupdateSequence\x12;\n" +
	"\x05state\x18\x0f \x01(\x0e2%.cozy.worker.v1.ArtifactTransferStateR\x05state\x12+\n" +
	"\x11transferred_bytes\x18\x10 \x01(\x04R\x10transferredBytes\x12\x1f\n" +
	"\vhttp_status\x18\x11 \x01(\rR\n" +
	"httpStatus\x12\x12\n" +
	"\x04etag\x18\x12 \x01(\tR\x04etag\x12'\n" +
	"\x0fchecksum_sha256\x18\x13 \x01(\tR\x0echecksumSha256\x12\x1a\n" +
	"\battempts\x18\x14 \x01(\rR\battempts\x12\x1b\n" +
	"\tsafe_code\x18\x15 \x01(\tR\bsafeCode\x12\x1f\n" +
	"\vsafe_detail\x18\x16 \x01(\tR\n" +
	"safeDetailJ\x04\b\x04\x10\x05\"\x82\x04\n" +
	"\x16ModelSourceFileRequest\x12,\n" +
	"\x12record_owner_epoch\x18\x01 \x01(\x04R\x10recordOwnerEpoch\x12:\n" +
	"\x19control_stream_generation\x18\x02 \x01(\x04R\x17controlStreamGeneration\x12$\n" +
	"\x0eworker_boot_id\x18\x03 \x01(\tR\fworkerBootId\x12!\n" +
	"\foperation_id\x18\x05 \x01(\tR\voperationId\x126\n" +
	"\x17source_selection_digest\x18\x06 \x01(\fR\x15sourceSelectionDigest\x12\x16\n" +
	"\x06member\x18\a \x01(\tR\x06member\x12\x1b\n" +
	"\tobject_id\x18\b \x01(\tR\bobjectId\x12\x16\n" +
	"\x06length\x18\t \x01(\x04R\x06length\x12?\n" +
	"\bprovider\x18\n" +
	" \x01(\x0e2#.cozy.worker.v1.ModelSourceProviderR\bprovider\x12\x10\n" +
	"\x03url\x18\v \x01(\tR\x03url\x12&\n" +
	"\x0fexpires_at_unix\x18\f \x01(\x04R\rexpiresAtUnix\x12/\n" +
	"\x13capability_revision\x18\r \x01(\x04R\x12capabilityRevisionJ\x04\b\x04\x10\x05\"\xc9\x04\n" +
	"\x15ModelSourceFileStatus\x12,\n" +
	"\x12record_owner_epoch\x18\x01 \x01(\x04R\x10recordOwnerEpoch\x12:\n" +
	"\x19control_stream_generation\x18\x02 \x01(\x04R\x17controlStreamGeneration\x12$\n" +
	"\x0eworker_boot_id\x18\x03 \x01(\tR\fworkerBootId\x12!\n" +
	"\foperation_id\x18\x05 \x01(\tR\voperationId\x126\n" +
	"\x17source_selection_digest\x18\x06 \x01(\fR\x15sourceSelectionDigest\x12\x16\n" +
	"\x06member\x18\a \x01(\tR\x06member\x12\x1b\n" +
	"\tobject_id\x18\b \x01(\tR\bobjectId\x12\x16\n" +
	"\x06length\x18\t \x01(\x04R\x06length\x12/\n" +
	"\x13capability_revision\x18\n" +
	" \x01(\x04R\x12capabilityRevision\x12:\n" +
	"\x05state\x18\v \x01(\x0e2$.cozy.worker.v1.ModelSourceFileStateR\x05state\x12+\n" +
	"\x11transferred_bytes\x18\f \x01(\x04R\x10transferredBytes\x12\x1a\n" +
	"\battempts\x18\r \x01(\rR\battempts\x12\x1b\n" +
	"\tsafe_code\x18\x0e \x01(\tR\bsafeCode\x12\x1f\n" +
	"\vsafe_detail\x18\x0f \x01(\tR\n" +
	"safeDetailJ\x04\b\x04\x10\x05\"\x96\x03\n" +
	"\x19ModelSourcePrepareRequest\x12,\n" +
	"\x12record_owner_epoch\x18\x01 \x01(\x04R\x10recordOwnerEpoch\x12:\n" +
	"\x19control_stream_generation\x18\x02 \x01(\x04R\x17controlStreamGeneration\x12$\n" +
	"\x0eworker_boot_id\x18\x03 \x01(\tR\fworkerBootId\x12!\n" +
	"\foperation_id\x18\x05 \x01(\tR\voperationId\x126\n" +
	"\x17source_selection_digest\x18\x06 \x01(\fR\x15sourceSelectionDigest\x12>\n" +
	"\bprofiles\x18\a \x03(\v2\".cozy.worker.v1.ModelSourceProfileR\bprofiles\x12\x1d\n" +
	"\n" +
	"source_uri\x18\b \x01(\tR\tsourceUri\x12)\n" +
	"\x10declared_license\x18\t \x01(\tR\x0fdeclaredLicenseJ\x04\b\x04\x10\x05\"\xc8\x03\n" +
	"\x13ModelSourcePrepared\x12,\n" +
	"\x12record_owner_epoch\x18\x01 \x01(\x04R\x10recordOwnerEpoch\x12:\n" +
	"\x19control_stream_generation\x18\x02 \x01(\x04R\x17controlStreamGeneration\x12$\n" +
	"\x0eworker_boot_id\x18\x03 \x01(\tR\fworkerBootId\x12!\n" +
	"\foperation_id\x18\x05 \x01(\tR\voperationId\x126\n" +
	"\x17source_selection_digest\x18\x06 \x01(\fR\x15sourceSelectionDigest\x12C\n" +
	"\aoutcome\x18\a \x01(\x0e2).cozy.worker.v1.ModelSourcePrepareOutcomeR\aoutcome\x12=\n" +
	"\asources\x18\b \x03(\v2#.cozy.worker.v1.PreparedModelSourceR\asources\x12\x1b\n" +
	"\tsafe_code\x18\t \x01(\tR\bsafeCode\x12\x1f\n" +
	"\vsafe_detail\x18\n" +
	" \x01(\tR\n" +
	"safeDetailJ\x04\b\x04\x10\x05\"\x88\x04\n" +
	"\x17ArtifactFinalizeRequest\x12,\n" +
	"\x12record_owner_epoch\x18\x01 \x01(\x04R\x10recordOwnerEpoch\x12:\n" +
	"\x19control_stream_generation\x18\x02 \x01(\x04R\x17controlStreamGeneration\x12$\n" +
	"\x0eworker_boot_id\x18\x03 \x01(\tR\fworkerBootId\x12\x1d\n" +
	"\n" +
	"request_id\x18\x05 \x01(\tR\trequestId\x124\n" +
	"\x16invocation_spec_digest\x18\x06 \x01(\fR\x14invocationSpecDigest\x12\x1f\n" +
	"\voutput_slot\x18\a \x01(\tR\n" +
	"outputSlot\x12M\n" +
	"\vdisposition\x18\b \x01(\x0e2+.cozy.worker.v1.ArtifactFinalizeDispositionR\vdisposition\x126\n" +
	"\x17artifact_receipt_digest\x18\t \x01(\fR\x15artifactReceiptDigest\x12&\n" +
	"\x0fscratch_root_id\x18\n" +
	" \x01(\tR\rscratchRootId\x122\n" +
	"\x15owner_authority_scope\x18\v \x01(\tR\x13ownerAuthorityScopeJ\x04\b\x04\x10\x05\"\xea\x03\n" +
	"\x16ArtifactFinalizeResult\x12,\n" +
	"\x12record_owner_epoch\x18\x01 \x01(\x04R\x10recordOwnerEpoch\x12:\n" +
	"\x19control_stream_generation\x18\x02 \x01(\x04R\x17controlStreamGeneration\x12$\n" +
	"\x0eworker_boot_id\x18\x03 \x01(\tR\fworkerBootId\x12\x1d\n" +
	"\n" +
	"request_id\x18\x05 \x01(\tR\trequestId\x124\n" +
	"\x16invocation_spec_digest\x18\x06 \x01(\fR\x14invocationSpecDigest\x12\x1f\n" +
	"\voutput_slot\x18\a \x01(\tR\n" +
	"outputSlot\x12A\n" +
	"\aoutcome\x18\b \x01(\x0e2'.cozy.worker.v1.ArtifactFinalizeOutcomeR\aoutcome\x12M\n" +
	"\x10artifact_receipt\x18\t \x01(\v2\".cozy.worker.v1.ArtifactReceiptRefR\x0fartifactReceipt\x122\n" +
	"\x15owner_authority_scope\x18\n" +
	" \x01(\tR\x13ownerAuthorityScopeJ\x04\b\x04\x10\x05\"\x93\x02\n" +
	"\x0eResultEnvelope\x120\n" +
	"\x14result_schema_digest\x18\x01 \x01(\fR\x12resultSchemaDigest\x12#\n" +
	"\rinline_result\x18\x02 \x01(\fR\finlineResult\x12<\n" +
	"\vresult_blob\x18\x03 \x01(\v2\x1b.cozy.worker.v1.OutputEntryR\n" +
	"resultBlob\x12?\n" +
	"\vadjustments\x18\x04 \x03(\v2\x1d.cozy.worker.v1.AdjustmentRowR\vadjustments\x12%\n" +
	"\x0echeckpoint_ref\x18\x05 \x01(\tR\rcheckpointRefJ\x04\b\x06\x10\a\"u\n" +
	"\rAdjustmentRow\x12\x14\n" +
	"\x05field\x18\x01 \x01(\tR\x05field\x12\x1c\n" +
	"\trequested\x18\x02 \x01(\tR\trequested\x12\x18\n" +
	"\aapplied\x18\x03 \x01(\tR\aapplied\x12\x16\n" +
	"\x06reason\x18\x04 \x01(\tR\x06reason\"\xcb\x01\n" +
	"\fOutcomeCause\x12-\n" +
	"\x04code\x18\x01 \x01(\x0e2\x19.cozy.worker.v1.CauseCodeR\x04code\x123\n" +
	"\x06origin\x18\x02 \x01(\x0e2\x1b.cozy.worker.v1.CauseOriginR\x06origin\x12\x16\n" +
	"\x06detail\x18\x03 \x01(\tR\x06detail\x12?\n" +
	"\tshortfall\x18\x04 \x01(\v2!.cozy.worker.v1.ResourceShortfallR\tshortfall\"\xdd\x01\n" +
	"\x11ResourceShortfall\x12\x1a\n" +
	"\bresource\x18\x01 \x01(\tR\bresource\x12\x14\n" +
	"\x05scope\x18\x02 \x01(\tR\x05scope\x12!\n" +
	"\fneeded_bytes\x18\x03 \x01(\x04R\vneededBytes\x12'\n" +
	"\x0favailable_bytes\x18\x04 \x01(\x04R\x0eavailableBytes\x12#\n" +
	"\rrequest_shape\x18\x05 \x01(\tR\frequestShape\x12%\n" +
	"\x0eevidence_class\x18\x06 \x01(\tR\revidenceClass\"\xed\x02\n" +
	"\x11AttemptOutcomeAck\x12,\n" +
	"\x12record_owner_epoch\x18\x01 \x01(\x04R\x10recordOwnerEpoch\x12:\n" +
	"\x19control_stream_generation\x18\x02 \x01(\x04R\x17controlStreamGeneration\x12$\n" +
	"\x0eworker_boot_id\x18\x03 \x01(\tR\fworkerBootId\x12\x1d\n" +
	"\n" +
	"request_id\x18\x05 \x01(\tR\trequestId\x12'\n" +
	"\x0fattempt_ordinal\x18\x06 \x01(\x04R\x0eattemptOrdinal\x124\n" +
	"\x16invocation_spec_digest\x18\a \x01(\fR\x14invocationSpecDigest\x12\x1d\n" +
	"\n" +
	"outcome_id\x18\b \x01(\tR\toutcomeId\x12%\n" +
	"\x0eoutcome_digest\x18\t \x01(\fR\routcomeDigestJ\x04\b\x04\x10\x05\"\xac\x03\n" +
	"\x14JobCheckpointRequest\x12,\n" +
	"\x12record_owner_epoch\x18\x01 \x01(\x04R\x10recordOwnerEpoch\x12:\n" +
	"\x19control_stream_generation\x18\x02 \x01(\x04R\x17controlStreamGeneration\x12$\n" +
	"\x0eworker_boot_id\x18\x03 \x01(\tR\fworkerBootId\x12\x1d\n" +
	"\n" +
	"request_id\x18\x05 \x01(\tR\trequestId\x12'\n" +
	"\x0fattempt_ordinal\x18\x06 \x01(\x04R\x0eattemptOrdinal\x12#\n" +
	"\roperation_key\x18\a \x01(\tR\foperationKey\x12\x1f\n" +
	"\vlogical_key\x18\b \x01(\tR\n" +
	"logicalKey\x12%\n" +
	"\x0econtent_digest\x18\t \x01(\fR\rcontentDigest\x12\x10\n" +
	"\x03seq\x18\n" +
	" \x01(\x04R\x03seq\x127\n" +
	"\bartifact\x18\v \x01(\v2\x1b.cozy.worker.v1.OutputEntryR\bartifactJ\x04\b\x04\x10\x05\"\xf4\x03\n" +
	"\x14JobCheckpointReceipt\x12,\n" +
	"\x12record_owner_epoch\x18\x01 \x01(\x04R\x10recordOwnerEpoch\x12:\n" +
	"\x19control_stream_generation\x18\x02 \x01(\x04R\x17controlStreamGeneration\x12$\n" +
	"\x0eworker_boot_id\x18\x03 \x01(\tR\fworkerBootId\x12\x1d\n" +
	"\n" +
	"request_id\x18\x05 \x01(\tR\trequestId\x12'\n" +
	"\x0fattempt_ordinal\x18\x06 \x01(\x04R\x0eattemptOrdinal\x12#\n" +
	"\roperation_key\x18\a \x01(\tR\foperationKey\x12\x1f\n" +
	"\vlogical_key\x18\b \x01(\tR\n" +
	"logicalKey\x12%\n" +
	"\x0econtent_digest\x18\t \x01(\fR\rcontentDigest\x12\x1d\n" +
	"\n" +
	"receipt_id\x18\n" +
	" \x01(\tR\treceiptId\x12;\n" +
	"\aoutcome\x18\v \x01(\x0e2!.cozy.worker.v1.CheckpointOutcomeR\aoutcome\x125\n" +
	"\x05fault\x18\f \x01(\v2\x1f.cozy.worker.v1.CheckpointFaultR\x05faultJ\x04\b\x04\x10\x05\"\x9a\x01\n" +
	"\x0fCheckpointFault\x127\n" +
	"\x04code\x18\x01 \x01(\x0e2#.cozy.worker.v1.CheckpointFaultCodeR\x04code\x12\x16\n" +
	"\x06detail\x18\x02 \x01(\tR\x06detail\x126\n" +
	"\x17recorded_content_digest\x18\x03 \x01(\fR\x15recordedContentDigest\"\xdb\x02\n" +
	"\x10JobCheckpointAck\x12,\n" +
	"\x12record_owner_epoch\x18\x01 \x01(\x04R\x10recordOwnerEpoch\x12:\n" +
	"\x19control_stream_generation\x18\x02 \x01(\x04R\x17controlStreamGeneration\x12$\n" +
	"\x0eworker_boot_id\x18\x03 \x01(\tR\fworkerBootId\x12\x1d\n" +
	"\n" +
	"request_id\x18\x05 \x01(\tR\trequestId\x12'\n" +
	"\x0fattempt_ordinal\x18\x06 \x01(\x04R\x0eattemptOrdinal\x12#\n" +
	"\roperation_key\x18\a \x01(\tR\foperationKey\x12\x1d\n" +
	"\n" +
	"receipt_id\x18\b \x01(\tR\treceiptId\x12%\n" +
	"\x0econtent_digest\x18\t \x01(\fR\rcontentDigestJ\x04\b\x04\x10\x05\"\xc3\x01\n" +
	"\fProgressOpen\x12,\n" +
	"\x12record_owner_epoch\x18\x01 \x01(\x04R\x10recordOwnerEpoch\x12:\n" +
	"\x19control_stream_generation\x18\x02 \x01(\x04R\x17controlStreamGeneration\x12$\n" +
	"\x0eworker_boot_id\x18\x03 \x01(\tR\fworkerBootId\x12\x1d\n" +
	"\n" +
	"request_id\x18\x05 \x01(\tR\trequestIdJ\x04\b\x04\x10\x05\"\xdb\x02\n" +
	"\x0fAttemptProgress\x12,\n" +
	"\x12record_owner_epoch\x18\x01 \x01(\x04R\x10recordOwnerEpoch\x12:\n" +
	"\x19control_stream_generation\x18\x02 \x01(\x04R\x17controlStreamGeneration\x12$\n" +
	"\x0eworker_boot_id\x18\x03 \x01(\tR\fworkerBootId\x12\x1d\n" +
	"\n" +
	"request_id\x18\x05 \x01(\tR\trequestId\x12'\n" +
	"\x0fattempt_ordinal\x18\x06 \x01(\x04R\x0eattemptOrdinal\x12\x10\n" +
	"\x03seq\x18\a \x01(\x04R\x03seq\x12!\n" +
	"\fcontent_type\x18\b \x01(\tR\vcontentType\x12\x12\n" +
	"\x04data\x18\t \x01(\fR\x04data\x12!\n" +
	"\fplacement_id\x18\n" +
	" \x01(\tR\vplacementIdJ\x04\b\x04\x10\x05\"\xf2\x03\n" +
	"\x0eInvocationSpec\x126\n" +
	"\x17package_revision_digest\x18\x01 \x01(\tR\x15packageRevisionDigest\x12-\n" +
	"\x12environment_digest\x18\x02 \x01(\tR\x11environmentDigest\x12#\n" +
	"\rconfig_digest\x18\x03 \x01(\tR\fconfigDigest\x12%\n" +
	"\x0epayload_digest\x18\x04 \x01(\tR\rpayloadDigest\x124\n" +
	"\x06inputs\x18\x05 \x03(\v2\x1c.cozy.worker.v1.InputBindingR\x06inputs\x127\n" +
	"\aoutputs\x18\x06 \x03(\v2\x1d.cozy.worker.v1.OutputBindingR\aoutputs\x12(\n" +
	"\x10deadline_unix_ms\x18\a \x01(\x04R\x0edeadlineUnixMs\x12A\n" +
	"\aserving\x18\b \x01(\v2%.cozy.worker.v1.ServingInvocationSpecH\x00R\aserving\x125\n" +
	"\x03job\x18\t \x01(\v2!.cozy.worker.v1.JobInvocationSpecH\x00R\x03jobB\x06\n" +
	"\x04specR\x12package_release_id\"\x8c\x01\n" +
	"\fInputBinding\x12\x19\n" +
	"\binput_id\x18\x01 \x01(\tR\ainputId\x12\x16\n" +
	"\x06digest\x18\x02 \x01(\tR\x06digest\x12\x16\n" +
	"\x06length\x18\x03 \x01(\x04R\x06length\x12\x1b\n" +
	"\tkind_mime\x18\x04 \x01(\tR\bkindMime\x12\x14\n" +
	"\x05order\x18\x05 \x01(\rR\x05order\"f\n" +
	"\rOutputBinding\x12\x1b\n" +
	"\toutput_id\x18\x01 \x01(\tR\boutputId\x12\x1b\n" +
	"\tmime_type\x18\x02 \x01(\tR\bmimeType\x12\x1b\n" +
	"\tmax_bytes\x18\x03 \x01(\x04R\bmaxBytes\"\x81\x01\n" +
	"\x15ServingInvocationSpec\x12:\n" +
	"\x19entrypoint_binding_digest\x18\x01 \x01(\tR\x17entrypointBindingDigest\x12,\n" +
	"\x12attempt_binding_id\x18\x02 \x01(\tR\x10attemptBindingId\"\xb2\x01\n" +
	"\x11JobInvocationSpec\x12\x19\n" +
	"\bbuild_id\x18\x01 \x01(\tR\abuildId\x12*\n" +
	"\x11job_descriptor_id\x18\x02 \x01(\tR\x0fjobDescriptorId\x12V\n" +
	"\x14publication_contract\x18\x03 \x01(\v2#.cozy.worker.v1.PublicationContractR\x13publicationContract\"\xc8\x02\n" +
	"\rDeliveryGrant\x124\n" +
	"\x16invocation_spec_digest\x18\x01 \x01(\fR\x14invocationSpecDigest\x12H\n" +
	"\n" +
	"credential\x18\x02 \x01(\v2(.cozy.worker.v1.DeliveryAccessCredentialR\n" +
	"credential\x12\"\n" +
	"\rfile_base_url\x18\x03 \x01(\tR\vfileBaseUrl\x12&\n" +
	"\x0fexpires_at_unix\x18\x04 \x01(\x04R\rexpiresAtUnix\x123\n" +
	"\x06inputs\x18\x05 \x03(\v2\x1b.cozy.worker.v1.InputAccessR\x06inputs\x126\n" +
	"\aoutputs\x18\x06 \x03(\v2\x1c.cozy.worker.v1.OutputAccessR\aoutputs\":\n" +
	"\vInputAccess\x12\x19\n" +
	"\binput_id\x18\x01 \x01(\tR\ainputId\x12\x10\n" +
	"\x03url\x18\x02 \x01(\tR\x03url\"=\n" +
	"\fOutputAccess\x12\x1b\n" +
	"\toutput_id\x18\x01 \x01(\tR\boutputId\x12\x10\n" +
	"\x03url\x18\x02 \x01(\tR\x03url\"\x9d\x01\n" +
	"\x18DeliveryAccessCredential\x12\x16\n" +
	"\x06issuer\x18\x01 \x01(\tR\x06issuer\x12\x15\n" +
	"\x06key_id\x18\x02 \x01(\tR\x05keyId\x12\x14\n" +
	"\x05epoch\x18\x03 \x01(\x04R\x05epoch\x12&\n" +
	"\x0fexpires_at_unix\x18\x04 \x01(\x04R\rexpiresAtUnix\x12\x14\n" +
	"\x05token\x18\x05 \x01(\fR\x05token\"\xb8\x01\n" +
	"\fResourceCaps\x12'\n" +
	"\x0fdevice_required\x18\x01 \x01(\bR\x0edeviceRequired\x125\n" +
	"\x17max_device_memory_bytes\x18\x02 \x01(\x04R\x14maxDeviceMemoryBytes\x12\"\n" +
	"\rmax_rss_bytes\x18\x03 \x01(\x04R\vmaxRssBytes\x12$\n" +
	"\x0emax_disk_bytes\x18\x04 \x01(\x04R\fmaxDiskBytes\"i\n" +
	"\x13PublicationContract\x127\n" +
	"\aoutputs\x18\x01 \x03(\v2\x1d.cozy.worker.v1.OutputBindingR\aoutputs\x12\x19\n" +
	"\bgrant_id\x18\x02 \x01(\tR\agrantId\"\xed\x04\n" +
	"\x0fWorkerResources\x12\x1a\n" +
	"\bplatform\x18\x01 \x01(\tR\bplatform\x12\x18\n" +
	"\abackend\x18\x02 \x01(\tR\abackend\x12!\n" +
	"\fmemory_model\x18\x03 \x01(\tR\vmemoryModel\x12!\n" +
	"\fdevice_count\x18\x04 \x01(\rR\vdeviceCount\x12\x1f\n" +
	"\vdevice_name\x18\x05 \x01(\tR\n" +
	"deviceName\x129\n" +
	"\x19device_memory_total_bytes\x18\x06 \x01(\x04R\x16deviceMemoryTotalBytes\x12%\n" +
	"\x0edriver_version\x18\a \x01(\tR\rdriverVersion\x12'\n" +
	"\x0fbackend_version\x18\b \x01(\tR\x0ebackendVersion\x12/\n" +
	"\x14host_ram_total_bytes\x18\t \x01(\x04R\x11hostRamTotalBytes\x12\x1d\n" +
	"\n" +
	"vcpu_count\x18\n" +
	" \x01(\rR\tvcpuCount\x12(\n" +
	"\x10disk_total_bytes\x18\v \x01(\x04R\x0ediskTotalBytes\x12&\n" +
	"\x0fpower_cap_watts\x18\f \x01(\rR\rpowerCapWatts\x12+\n" +
	"\x11partition_profile\x18\r \x01(\tR\x10partitionProfile\x12\"\n" +
	"\finterconnect\x18\x0e \x01(\tR\finterconnect\x12\x1f\n" +
	"\vpeer_access\x18\x0f \x01(\bR\n" +
	"peerAccess\x12\x1e\n" +
	"\n" +
	"unreadable\x18\x10 \x03(\tR\n" +
	"unreadable\"\x88\x01\n" +
	"\vJobCapacity\x12$\n" +
	"\x0ejobs_in_flight\x18\x01 \x01(\rR\fjobsInFlight\x12%\n" +
	"\x0ejobs_available\x18\x02 \x01(\rR\rjobsAvailable\x12&\n" +
	"\x0ffree_disk_bytes\x18\x04 \x01(\x04R\rfreeDiskBytesJ\x04\b\x03\x10\x04\"\x8a\x03\n" +
	"\vHeldAttempt\x12\x1d\n" +
	"\n" +
	"request_id\x18\x01 \x01(\tR\trequestId\x12'\n" +
	"\x0fattempt_ordinal\x18\x02 \x01(\x04R\x0eattemptOrdinal\x12/\n" +
	"\x04kind\x18\x03 \x01(\x0e2\x1b.cozy.worker.v1.AttemptKindR\x04kind\x122\n" +
	"\x05state\x18\x04 \x01(\x0e2\x1c.cozy.worker.v1.AttemptStateR\x05state\x124\n" +
	"\x16invocation_spec_digest\x18\x05 \x01(\fR\x14invocationSpecDigest\x12!\n" +
	"\fplacement_id\x18\x06 \x01(\tR\vplacementId\x12/\n" +
	"\x13executor_generation\x18\a \x01(\x04R\x12executorGeneration\x12\x1d\n" +
	"\n" +
	"outcome_id\x18\b \x01(\tR\toutcomeId\x12%\n" +
	"\x0eoutcome_digest\x18\t \x01(\fR\routcomeDigest\"\x80\x01\n" +
	"\x05Fault\x12-\n" +
	"\x04kind\x18\x01 \x01(\x0e2\x19.cozy.worker.v1.FaultKindR\x04kind\x12\x18\n" +
	"\asubject\x18\x02 \x01(\tR\asubject\x12\x16\n" +
	"\x06reason\x18\x03 \x01(\tR\x06reason\x12\x16\n" +
	"\x06detail\x18\x04 \x01(\tR\x06detail\"\x8b\x01\n" +
	"\x0eOutputManifest\x12<\n" +
	"\x1apublication_receipt_digest\x18\x01 \x01(\tR\x18publicationReceiptDigest\x125\n" +
	"\aoutputs\x18\x03 \x03(\v2\x1b.cozy.worker.v1.OutputEntryR\aoutputsJ\x04\b\x02\x10\x03\"w\n" +
	"\vOutputEntry\x12\x1b\n" +
	"\toutput_id\x18\x01 \x01(\tR\boutputId\x12\x16\n" +
	"\x06digest\x18\x02 \x01(\fR\x06digest\x12\x16\n" +
	"\x06length\x18\x03 \x01(\x04R\x06length\x12\x1b\n" +
	"\tmime_type\x18\x04 \x01(\tR\bmimeType\"\xd7\x03\n" +
	"\x0eAttemptMetrics\x12\x1d\n" +
	"\n" +
	"runtime_ms\x18\x01 \x01(\x04R\truntimeMs\x12\x19\n" +
	"\bqueue_ms\x18\x02 \x01(\x04R\aqueueMs\x127\n" +
	"\x18peak_device_memory_bytes\x18\x03 \x01(\x04R\x15peakDeviceMemoryBytes\x12'\n" +
	"\x10rss_at_end_bytes\x18\x04 \x01(\x04R\rrssAtEndBytes\x12!\n" +
	"\foutput_count\x18\x05 \x01(\rR\voutputCount\x12!\n" +
	"\finput_tokens\x18\x06 \x01(\x04R\vinputTokens\x12#\n" +
	"\routput_tokens\x18\a \x01(\x04R\foutputTokens\x12&\n" +
	"\x0fdevice_lease_ms\x18\b \x01(\x04R\rdeviceLeaseMs\x12!\n" +
	"\fdevice_count\x18\t \x01(\rR\vdeviceCount\x12\x1d\n" +
	"\n" +
	"handler_ms\x18\n" +
	" \x01(\x04R\thandlerMs\x12'\n" +
	"\x0ffinalization_ms\x18\v \x01(\x04R\x0efinalizationMs\x12+\n" +
	"\x11unverified_fields\x18\f \x03(\tR\x10unverifiedFields\"\x80\x01\n" +
	"\x0fTriageBundleRef\x12\x1d\n" +
	"\n" +
	"subject_id\x18\x01 \x01(\tR\tsubjectId\x120\n" +
	"\x14write_receipt_digest\x18\x04 \x01(\fR\x12writeReceiptDigest\x12\x16\n" +
	"\x06length\x18\x03 \x01(\x04R\x06lengthJ\x04\b\x02\x10\x03*\xd7\x01\n" +
	"\x11LocalDownloadKind\x12#\n" +
	"\x1fLOCAL_DOWNLOAD_KIND_UNSPECIFIED\x10\x00\x12%\n" +
	"!LOCAL_DOWNLOAD_KIND_PROJECT_WHEEL\x10\x01\x12(\n" +
	"$LOCAL_DOWNLOAD_KIND_DEPENDENCY_WHEEL\x10\x02\x12&\n" +
	"\"LOCAL_DOWNLOAD_KIND_MODEL_MANIFEST\x10\x03\x12$\n" +
	" LOCAL_DOWNLOAD_KIND_MODEL_OBJECT\x10\x04*O\n" +
	"\aPosture\x12\x17\n" +
	"\x13POSTURE_UNSPECIFIED\x10\x00\x12\x15\n" +
	"\x11POSTURE_ACCEPTING\x10\x01\x12\x14\n" +
	"\x10POSTURE_DRAINING\x10\x02*\x92\x01\n" +
	"\vWorkerPhase\x12\x1c\n" +
	"\x18WORKER_PHASE_UNSPECIFIED\x10\x00\x12\x18\n" +
	"\x14WORKER_PHASE_BOOTING\x10\x01\x12\x17\n" +
	"\x13WORKER_PHASE_ONLINE\x10\x02\x12\x19\n" +
	"\x15WORKER_PHASE_DRAINING\x10\x03\x12\x17\n" +
	"\x13WORKER_PHASE_FAILED\x10\x04*\xcc\x01\n" +
	"\x14MaterializationState\x12%\n" +
	"!MATERIALIZATION_STATE_UNSPECIFIED\x10\x00\x12 \n" +
	"\x1cMATERIALIZATION_STATE_ABSENT\x10\x01\x12'\n" +
	"#MATERIALIZATION_STATE_MATERIALIZING\x10\x02\x12 \n" +
	"\x1cMATERIALIZATION_STATE_STAGED\x10\x03\x12 \n" +
	"\x1cMATERIALIZATION_STATE_FAILED\x10\x04*\xa2\x01\n" +
	"\fServingState\x12\x1d\n" +
	"\x19SERVING_STATE_UNSPECIFIED\x10\x00\x12\x19\n" +
	"\x15SERVING_STATE_OFFLINE\x10\x01\x12\x1c\n" +
	"\x18SERVING_STATE_ACTIVATING\x10\x02\x12\x1e\n" +
	"\x1aSERVING_STATE_DISPATCHABLE\x10\x03\x12\x1a\n" +
	"\x16SERVING_STATE_DRAINING\x10\x04*g\n" +
	"\x0eAdmissionState\x12\x1f\n" +
	"\x1bADMISSION_STATE_UNSPECIFIED\x10\x00\x12\x18\n" +
	"\x14ADMISSION_STATE_OPEN\x10\x01\x12\x1a\n" +
	"\x16ADMISSION_STATE_CLOSED\x10\x02*[\n" +
	"\vAttemptKind\x12\x1c\n" +
	"\x18ATTEMPT_KIND_UNSPECIFIED\x10\x00\x12\x18\n" +
	"\x14ATTEMPT_KIND_SERVING\x10\x01\x12\x14\n" +
	"\x10ATTEMPT_KIND_JOB\x10\x02*\x91\x01\n" +
	"\fAttemptState\x12\x1d\n" +
	"\x19ATTEMPT_STATE_UNSPECIFIED\x10\x00\x12\x19\n" +
	"\x15ATTEMPT_STATE_RUNNING\x10\x01\x12%\n" +
	"!ATTEMPT_STATE_OUTCOME_PENDING_ACK\x10\x02\x12 \n" +
	"\x1cATTEMPT_STATE_HELD_UNDURABLE\x10\x03*\xbf\x01\n" +
	"\rOutcomeStatus\x12\x1e\n" +
	"\x1aOUTCOME_STATUS_UNSPECIFIED\x10\x00\x12\x1c\n" +
	"\x18OUTCOME_STATUS_SUCCEEDED\x10\x01\x12\x1a\n" +
	"\x16OUTCOME_STATUS_REFUSED\x10\x02\x12\x19\n" +
	"\x15OUTCOME_STATUS_FAILED\x10\x03\x12\x1b\n" +
	"\x17OUTCOME_STATUS_CANCELED\x10\x04\x12\x1c\n" +
	"\x18OUTCOME_STATUS_ABANDONED\x10\x05*\xc4\x05\n" +
	"\tCauseCode\x12\x1a\n" +
	"\x16CAUSE_CODE_UNSPECIFIED\x10\x00\x12\x1e\n" +
	"\x1aCAUSE_CODE_INVALID_REQUEST\x10\x01\x12 \n" +
	"\x1cCAUSE_CODE_UNSUPPORTED_INPUT\x10\x02\x12\x1b\n" +
	"\x17CAUSE_CODE_LOCAL_SAFETY\x10\x03\x12\x17\n" +
	"\x13CAUSE_CODE_PROTOCOL\x10\x04\x12$\n" +
	" CAUSE_CODE_CONSTRAINT_INFEASIBLE\x10\x05\x12\x1f\n" +
	"\x1bCAUSE_CODE_AUTHOR_EXCEPTION\x10\x06\x12\x1d\n" +
	"\x19CAUSE_CODE_EXECUTOR_FAULT\x10\a\x12\x1c\n" +
	"\x18CAUSE_CODE_GRANT_EXPIRED\x10\b\x12#\n" +
	"\x1fCAUSE_CODE_ARTIFACT_UNFETCHABLE\x10\t\x12%\n" +
	"!CAUSE_CODE_CAPABILITY_UNAVAILABLE\x10\n" +
	"\x12\x1c\n" +
	"\x18CAUSE_CODE_CLIENT_CANCEL\x10\v\x12\x1f\n" +
	"\x1bCAUSE_CODE_DEADLINE_EXPIRED\x10\f\x12\x1b\n" +
	"\x17CAUSE_CODE_DRAIN_CANCEL\x10\r\x12\x1c\n" +
	"\x18CAUSE_CODE_POLICY_CANCEL\x10\x0e\x12 \n" +
	"\x1cCAUSE_CODE_SUPERSEDED_CANCEL\x10\x0f\x12#\n" +
	"\x1fCAUSE_CODE_EXECUTOR_INVALIDATED\x10\x10\x12\x1a\n" +
	"\x16CAUSE_CODE_NO_CAPACITY\x10\x11\x12)\n" +
	"%CAUSE_CODE_ADMISSION_GENERATION_STALE\x10\x12\x12 \n" +
	"\x1cCAUSE_CODE_UNKNOWN_PLACEMENT\x10\x13\x12)\n" +
	"%CAUSE_CODE_PLACEMENT_NOT_DISPATCHABLE\x10\x14*\xe2\x01\n" +
	"\vCauseOrigin\x12\x1c\n" +
	"\x18CAUSE_ORIGIN_UNSPECIFIED\x10\x00\x12\x17\n" +
	"\x13CAUSE_ORIGIN_AUTHOR\x10\x01\x12\x18\n" +
	"\x14CAUSE_ORIGIN_RUNTIME\x10\x02\x12\x19\n" +
	"\x15CAUSE_ORIGIN_EXECUTOR\x10\x03\x12\x17\n" +
	"\x13CAUSE_ORIGIN_WORKER\x10\x04\x12\x16\n" +
	"\x12CAUSE_ORIGIN_INFRA\x10\x05\x12\x17\n" +
	"\x13CAUSE_ORIGIN_CLIENT\x10\x06\x12\x1d\n" +
	"\x19CAUSE_ORIGIN_RECORD_OWNER\x10\a*\xb4\x01\n" +
	"\fCancelReason\x12\x1d\n" +
	"\x19CANCEL_REASON_UNSPECIFIED\x10\x00\x12\x18\n" +
	"\x14CANCEL_REASON_CLIENT\x10\x01\x12\x17\n" +
	"\x13CANCEL_REASON_DRAIN\x10\x02\x12\x1c\n" +
	"\x18CANCEL_REASON_SUPERSEDED\x10\x03\x12\x18\n" +
	"\x14CANCEL_REASON_POLICY\x10\x04\x12\x1a\n" +
	"\x16CANCEL_REASON_DEADLINE\x10\x05*\xeb\x02\n" +
	"\x0eClaimRejection\x12\x1f\n" +
	"\x1bCLAIM_REJECTION_UNSPECIFIED\x10\x00\x12#\n" +
	"\x1fCLAIM_REJECTION_UNAUTHENTICATED\x10\x01\x12,\n" +
	"(CLAIM_REJECTION_STALE_RECORD_OWNER_EPOCH\x10\x02\x12\x1e\n" +
	"\x1aCLAIM_REJECTION_EPOCH_HELD\x10\x03\x12&\n" +
	"\"CLAIM_REJECTION_WORKER_ID_MISMATCH\x10\x04\x12'\n" +
	"#CLAIM_REJECTION_RELEASE_ID_MISMATCH\x10\x05\x12\x1d\n" +
	"\x19CLAIM_REJECTION_UNDURABLE\x10\a\"\x04\b\x06\x10\x06\"\x04\b\b\x10\b*!CLAIM_REJECTION_UNSUPPORTED_MINOR*&CLAIM_REJECTION_SCHEMA_DIGEST_MISMATCH*\x80\x05\n" +
	"\tFaultKind\x12\x1a\n" +
	"\x16FAULT_KIND_UNSPECIFIED\x10\x00\x12\"\n" +
	"\x1eFAULT_KIND_BINDING_UNAVAILABLE\x10\x01\x12\x1f\n" +
	"\x1bFAULT_KIND_BINDING_DEGRADED\x10\x02\x12\"\n" +
	"\x1eFAULT_KIND_HARDWARE_UNSUITABLE\x10\x03\x12$\n" +
	" FAULT_KIND_ARTIFACT_FETCH_FAILED\x10\x04\x12#\n" +
	"\x1fFAULT_KIND_CREDENTIAL_UNAPPLIED\x10\x06\x12#\n" +
	"\x1fFAULT_KIND_LOCAL_SAFETY_REFUSAL\x10\a\x12 \n" +
	"\x1cFAULT_KIND_EXECUTOR_POISONED\x10\b\x12\x1d\n" +
	"\x19FAULT_KIND_CONFIG_REFUSED\x10\t\x12 \n" +
	"\x1cFAULT_KIND_UNKNOWN_PLACEMENT\x10\n" +
	"\x12(\n" +
	"$FAULT_KIND_PLACEMENT_SET_UNSUPPORTED\x10\v\x12,\n" +
	"(FAULT_KIND_PLACEMENT_SET_DIGEST_MISMATCH\x10\f\x12'\n" +
	"#FAULT_KIND_ARTIFACT_DIGEST_MISMATCH\x10\x0e\x12#\n" +
	"\x1fFAULT_KIND_FALLBACK_PIN_MISSING\x10\x0f\"\x04\b\x05\x10\x05\"\x04\b\r\x10\r\"\x04\b\x10\x10\x10*\x18FAULT_KIND_GRANT_EXPIRED*'FAULT_KIND_ENVIRONMENT_RECEIPT_MISMATCH* FAULT_KIND_AUTHORIZATION_EXPIRED*\xa1\x02\n" +
	"\x11BootFailureReason\x12#\n" +
	"\x1fBOOT_FAILURE_REASON_UNSPECIFIED\x10\x00\x12(\n" +
	"$BOOT_FAILURE_REASON_HARDWARE_VERDICT\x10\x01\x12&\n" +
	"\"BOOT_FAILURE_REASON_ARTIFACT_FAULT\x10\x02\x12$\n" +
	" BOOT_FAILURE_REASON_DRIVER_FAULT\x10\x03\x12\"\n" +
	"\x1eBOOT_FAILURE_REASON_DISK_SHAPE\x10\x04\x12&\n" +
	"\"BOOT_FAILURE_REASON_CONFIG_INVALID\x10\x05\x12#\n" +
	"\x1fBOOT_FAILURE_REASON_OTHER_FATAL\x10\x06*\xba\x01\n" +
	"\x11CheckpointOutcome\x12\"\n" +
	"\x1eCHECKPOINT_OUTCOME_UNSPECIFIED\x10\x00\x12\x1f\n" +
	"\x1bCHECKPOINT_OUTCOME_RECORDED\x10\x01\x12\x1f\n" +
	"\x1bCHECKPOINT_OUTCOME_REPLAYED\x10\x02\x12\x1f\n" +
	"\x1bCHECKPOINT_OUTCOME_CONFLICT\x10\x03\x12\x1e\n" +
	"\x1aCHECKPOINT_OUTCOME_REFUSED\x10\x04*\x90\x02\n" +
	"\x13CheckpointFaultCode\x12%\n" +
	"!CHECKPOINT_FAULT_CODE_UNSPECIFIED\x10\x00\x12+\n" +
	"'CHECKPOINT_FAULT_CODE_IDENTITY_CONFLICT\x10\x01\x12)\n" +
	"%CHECKPOINT_FAULT_CODE_UNKNOWN_ATTEMPT\x10\x02\x12&\n" +
	"\"CHECKPOINT_FAULT_CODE_NOT_JOB_MODE\x10\x03\x12(\n" +
	"$CHECKPOINT_FAULT_CODE_STALE_SEQUENCE\x10\x04\x12(\n" +
	"$CHECKPOINT_FAULT_CODE_QUOTA_EXCEEDED\x10\x05*y\n" +
	"\x11ArtifactHostStage\x12#\n" +
	"\x1fARTIFACT_HOST_STAGE_UNSPECIFIED\x10\x00\x12\x1e\n" +
	"\x1aARTIFACT_HOST_STAGE_INTENT\x10\x01\x12\x1f\n" +
	"\x1bARTIFACT_HOST_STAGE_RECEIPT\x10\x02*\xa7\x01\n" +
	"\x13ArtifactHostOutcome\x12%\n" +
	"!ARTIFACT_HOST_OUTCOME_UNSPECIFIED\x10\x00\x12\"\n" +
	"\x1eARTIFACT_HOST_OUTCOME_RECORDED\x10\x01\x12\"\n" +
	"\x1eARTIFACT_HOST_OUTCOME_REPLAYED\x10\x02\x12!\n" +
	"\x1dARTIFACT_HOST_OUTCOME_REFUSED\x10\x03*\x93\x02\n" +
	"\x13ArtifactHostRefusal\x12%\n" +
	"!ARTIFACT_HOST_REFUSAL_UNSPECIFIED\x10\x00\x12)\n" +
	"%ARTIFACT_HOST_REFUSAL_UNKNOWN_ATTEMPT\x10\x01\x12)\n" +
	"%ARTIFACT_HOST_REFUSAL_INTENT_CONFLICT\x10\x02\x12&\n" +
	"\"ARTIFACT_HOST_REFUSAL_STALE_WRITER\x10\x03\x12*\n" +
	"&ARTIFACT_HOST_REFUSAL_RECEIPT_CONFLICT\x10\x04\x12+\n" +
	"'ARTIFACT_HOST_REFUSAL_INVENTORY_INVALID\x10\x05*\x95\x01\n" +
	"\x18ArtifactTransactionState\x12*\n" +
	"&ARTIFACT_TRANSACTION_STATE_UNSPECIFIED\x10\x00\x12%\n" +
	"!ARTIFACT_TRANSACTION_STATE_INTENT\x10\x01\x12&\n" +
	"\"ARTIFACT_TRANSACTION_STATE_RECEIPT\x10\x02*\x7f\n" +
	"\x13ArtifactReadOutcome\x12%\n" +
	"!ARTIFACT_READ_OUTCOME_UNSPECIFIED\x10\x00\x12\x1e\n" +
	"\x1aARTIFACT_READ_OUTCOME_DATA\x10\x01\x12!\n" +
	"\x1dARTIFACT_READ_OUTCOME_REFUSED\x10\x02*\xc3\x02\n" +
	"\x13ArtifactReadRefusal\x12%\n" +
	"!ARTIFACT_READ_REFUSAL_UNSPECIFIED\x10\x00\x12-\n" +
	")ARTIFACT_READ_REFUSAL_UNKNOWN_TRANSACTION\x10\x01\x12&\n" +
	"\"ARTIFACT_READ_REFUSAL_STALE_WRITER\x10\x02\x12(\n" +
	"$ARTIFACT_READ_REFUSAL_UNKNOWN_OBJECT\x10\x03\x12-\n" +
	")ARTIFACT_READ_REFUSAL_SOURCE_REF_MISMATCH\x10\x04\x12'\n" +
	"#ARTIFACT_READ_REFUSAL_RANGE_INVALID\x10\x05\x12,\n" +
	"(ARTIFACT_READ_REFUSAL_SOURCE_UNAVAILABLE\x10\x06*\xa6\x02\n" +
	"\x15ArtifactTransferState\x12'\n" +
	"#ARTIFACT_TRANSFER_STATE_UNSPECIFIED\x10\x00\x12$\n" +
	" ARTIFACT_TRANSFER_STATE_ACCEPTED\x10\x01\x12%\n" +
	"!ARTIFACT_TRANSFER_STATE_UPLOADING\x10\x02\x12$\n" +
	" ARTIFACT_TRANSFER_STATE_UPLOADED\x10\x03\x12+\n" +
	"'ARTIFACT_TRANSFER_STATE_ALREADY_PRESENT\x10\x04\x12 \n" +
	"\x1cARTIFACT_TRANSFER_STATE_HELD\x10\x05\x12\"\n" +
	"\x1eARTIFACT_TRANSFER_STATE_FAILED\x10\x06*\x87\x01\n" +
	"\x13ModelSourceProvider\x12%\n" +
	"!MODEL_SOURCE_PROVIDER_UNSPECIFIED\x10\x00\x12&\n" +
	"\"MODEL_SOURCE_PROVIDER_HUGGING_FACE\x10\x01\x12!\n" +
	"\x1dMODEL_SOURCE_PROVIDER_CIVITAI\x10\x02*\xd8\x01\n" +
	"\x14ModelSourceFileState\x12'\n" +
	"#MODEL_SOURCE_FILE_STATE_UNSPECIFIED\x10\x00\x12$\n" +
	" MODEL_SOURCE_FILE_STATE_ACCEPTED\x10\x01\x12'\n" +
	"#MODEL_SOURCE_FILE_STATE_DOWNLOADING\x10\x02\x12$\n" +
	" MODEL_SOURCE_FILE_STATE_VERIFIED\x10\x03\x12\"\n" +
	"\x1eMODEL_SOURCE_FILE_STATE_FAILED\x10\x04*\xc9\x01\n" +
	"\x19ModelSourcePrepareOutcome\x12,\n" +
	"(MODEL_SOURCE_PREPARE_OUTCOME_UNSPECIFIED\x10\x00\x12)\n" +
	"%MODEL_SOURCE_PREPARE_OUTCOME_PREPARED\x10\x01\x12)\n" +
	"%MODEL_SOURCE_PREPARE_OUTCOME_REPLAYED\x10\x02\x12(\n" +
	"$MODEL_SOURCE_PREPARE_OUTCOME_REFUSED\x10\x03*\xd7\x01\n" +
	"\x1bArtifactFinalizeDisposition\x12-\n" +
	")ARTIFACT_FINALIZE_DISPOSITION_UNSPECIFIED\x10\x00\x12'\n" +
	"#ARTIFACT_FINALIZE_DISPOSITION_ADOPT\x10\x01\x12)\n" +
	"%ARTIFACT_FINALIZE_DISPOSITION_ABANDON\x10\x02\x125\n" +
	"1ARTIFACT_FINALIZE_DISPOSITION_ABANDON_UNCOMMITTED\x10\x03*\x94\x01\n" +
	"\x17ArtifactFinalizeOutcome\x12)\n" +
	"%ARTIFACT_FINALIZE_OUTCOME_UNSPECIFIED\x10\x00\x12%\n" +
	"!ARTIFACT_FINALIZE_OUTCOME_ADOPTED\x10\x01\x12'\n" +
	"#ARTIFACT_FINALIZE_OUTCOME_ABANDONED\x10\x022\xaf\x01\n" +
	"\rWorkerControl\x12L\n" +
	"\aControl\x12 .cozy.worker.v1.RecordOwnerFrame\x1a\x1b.cozy.worker.v1.WorkerFrame(\x010\x01\x12P\n" +
	"\rWatchProgress\x12\x1c.cozy.worker.v1.ProgressOpen\x1a\x1f.cozy.worker.v1.AttemptProgress0\x012\xe7\x01\n" +
	"\x12RuntimePreparation\x12f\n" +
	"\x11PreparePackageSet\x12(.cozy.worker.v1.PreparePackageSetRequest\x1a'.cozy.worker.v1.PreparePackageSetResult\x12i\n" +
	"\x12PrepareModelSource\x12).cozy.worker.v1.PrepareModelSourceRequest\x1a(.cozy.worker.v1.PrepareModelSourceResultB4\n" +
	"\x0ecozy.worker.v1P\x01Z cozy/workerprotov1;workerprotov1b\x06proto3"

var (
	file_cozy_worker_v1_worker_proto_rawDescOnce sync.Once
	file_cozy_worker_v1_worker_proto_rawDescData []byte
)

func file_cozy_worker_v1_worker_proto_rawDescGZIP() []byte {
	file_cozy_worker_v1_worker_proto_rawDescOnce.Do(func() {
		file_cozy_worker_v1_worker_proto_rawDescData = protoimpl.X.CompressGZIP(unsafe.Slice(unsafe.StringData(file_cozy_worker_v1_worker_proto_rawDesc), len(file_cozy_worker_v1_worker_proto_rawDesc)))
	})
	return file_cozy_worker_v1_worker_proto_rawDescData
}

var file_cozy_worker_v1_worker_proto_enumTypes = make([]protoimpl.EnumInfo, 29)
var file_cozy_worker_v1_worker_proto_msgTypes = make([]protoimpl.MessageInfo, 100)
var file_cozy_worker_v1_worker_proto_goTypes = []any{
	(LocalDownloadKind)(0),                  // 0: cozy.worker.v1.LocalDownloadKind
	(Posture)(0),                            // 1: cozy.worker.v1.Posture
	(WorkerPhase)(0),                        // 2: cozy.worker.v1.WorkerPhase
	(MaterializationState)(0),               // 3: cozy.worker.v1.MaterializationState
	(ServingState)(0),                       // 4: cozy.worker.v1.ServingState
	(AdmissionState)(0),                     // 5: cozy.worker.v1.AdmissionState
	(AttemptKind)(0),                        // 6: cozy.worker.v1.AttemptKind
	(AttemptState)(0),                       // 7: cozy.worker.v1.AttemptState
	(OutcomeStatus)(0),                      // 8: cozy.worker.v1.OutcomeStatus
	(CauseCode)(0),                          // 9: cozy.worker.v1.CauseCode
	(CauseOrigin)(0),                        // 10: cozy.worker.v1.CauseOrigin
	(CancelReason)(0),                       // 11: cozy.worker.v1.CancelReason
	(ClaimRejection)(0),                     // 12: cozy.worker.v1.ClaimRejection
	(FaultKind)(0),                          // 13: cozy.worker.v1.FaultKind
	(BootFailureReason)(0),                  // 14: cozy.worker.v1.BootFailureReason
	(CheckpointOutcome)(0),                  // 15: cozy.worker.v1.CheckpointOutcome
	(CheckpointFaultCode)(0),                // 16: cozy.worker.v1.CheckpointFaultCode
	(ArtifactHostStage)(0),                  // 17: cozy.worker.v1.ArtifactHostStage
	(ArtifactHostOutcome)(0),                // 18: cozy.worker.v1.ArtifactHostOutcome
	(ArtifactHostRefusal)(0),                // 19: cozy.worker.v1.ArtifactHostRefusal
	(ArtifactTransactionState)(0),           // 20: cozy.worker.v1.ArtifactTransactionState
	(ArtifactReadOutcome)(0),                // 21: cozy.worker.v1.ArtifactReadOutcome
	(ArtifactReadRefusal)(0),                // 22: cozy.worker.v1.ArtifactReadRefusal
	(ArtifactTransferState)(0),              // 23: cozy.worker.v1.ArtifactTransferState
	(ModelSourceProvider)(0),                // 24: cozy.worker.v1.ModelSourceProvider
	(ModelSourceFileState)(0),               // 25: cozy.worker.v1.ModelSourceFileState
	(ModelSourcePrepareOutcome)(0),          // 26: cozy.worker.v1.ModelSourcePrepareOutcome
	(ArtifactFinalizeDisposition)(0),        // 27: cozy.worker.v1.ArtifactFinalizeDisposition
	(ArtifactFinalizeOutcome)(0),            // 28: cozy.worker.v1.ArtifactFinalizeOutcome
	(*PreparePackageSetRequest)(nil),        // 29: cozy.worker.v1.PreparePackageSetRequest
	(*LocalDownloadFile)(nil),               // 30: cozy.worker.v1.LocalDownloadFile
	(*PreparePackageSetResult)(nil),         // 31: cozy.worker.v1.PreparePackageSetResult
	(*LocalModelSourceFile)(nil),            // 32: cozy.worker.v1.LocalModelSourceFile
	(*ModelSourceProfile)(nil),              // 33: cozy.worker.v1.ModelSourceProfile
	(*PreparedModelSource)(nil),             // 34: cozy.worker.v1.PreparedModelSource
	(*PrepareModelSourceRequest)(nil),       // 35: cozy.worker.v1.PrepareModelSourceRequest
	(*PrepareModelSourceResult)(nil),        // 36: cozy.worker.v1.PrepareModelSourceResult
	(*RecordOwnerFrame)(nil),                // 37: cozy.worker.v1.RecordOwnerFrame
	(*WorkerFrame)(nil),                     // 38: cozy.worker.v1.WorkerFrame
	(*Claim)(nil),                           // 39: cozy.worker.v1.Claim
	(*ClaimProof)(nil),                      // 40: cozy.worker.v1.ClaimProof
	(*ClaimAck)(nil),                        // 41: cozy.worker.v1.ClaimAck
	(*BootFailure)(nil),                     // 42: cozy.worker.v1.BootFailure
	(*WorkerSnapshot)(nil),                  // 43: cozy.worker.v1.WorkerSnapshot
	(*WorkerSnapshotBody)(nil),              // 44: cozy.worker.v1.WorkerSnapshotBody
	(*SnapshotAck)(nil),                     // 45: cozy.worker.v1.SnapshotAck
	(*DesiredWorkerState)(nil),              // 46: cozy.worker.v1.DesiredWorkerState
	(*DesiredPackageSet)(nil),               // 47: cozy.worker.v1.DesiredPackageSet
	(*DesiredPlacementSet)(nil),             // 48: cozy.worker.v1.DesiredPlacementSet
	(*PlacementSet)(nil),                    // 49: cozy.worker.v1.PlacementSet
	(*DownloadDelegation)(nil),              // 50: cozy.worker.v1.DownloadDelegation
	(*DownloadModelRef)(nil),                // 51: cozy.worker.v1.DownloadModelRef
	(*DownloadPackageRef)(nil),              // 52: cozy.worker.v1.DownloadPackageRef
	(*Placement)(nil),                       // 53: cozy.worker.v1.Placement
	(*Ref)(nil),                             // 54: cozy.worker.v1.Ref
	(*WheelFact)(nil),                       // 55: cozy.worker.v1.WheelFact
	(*PackageSelection)(nil),                // 56: cozy.worker.v1.PackageSelection
	(*DevelopmentPackage)(nil),              // 57: cozy.worker.v1.DevelopmentPackage
	(*Environment)(nil),                     // 58: cozy.worker.v1.Environment
	(*Model)(nil),                           // 59: cozy.worker.v1.Model
	(*Entrypoint)(nil),                      // 60: cozy.worker.v1.Entrypoint
	(*Slot)(nil),                            // 61: cozy.worker.v1.Slot
	(*Config)(nil),                          // 62: cozy.worker.v1.Config
	(*Asset)(nil),                           // 63: cozy.worker.v1.Asset
	(*Component)(nil),                       // 64: cozy.worker.v1.Component
	(*Stamp)(nil),                           // 65: cozy.worker.v1.Stamp
	(*JobDirective)(nil),                    // 66: cozy.worker.v1.JobDirective
	(*ObservedWorkerState)(nil),             // 67: cozy.worker.v1.ObservedWorkerState
	(*PlacementStatus)(nil),                 // 68: cozy.worker.v1.PlacementStatus
	(*PlacementAcquisitionObservation)(nil), // 69: cozy.worker.v1.PlacementAcquisitionObservation
	(*AcquisitionLegObservation)(nil),       // 70: cozy.worker.v1.AcquisitionLegObservation
	(*AcceleratorQualification)(nil),        // 71: cozy.worker.v1.AcceleratorQualification
	(*ActivityEvent)(nil),                   // 72: cozy.worker.v1.ActivityEvent
	(*AttemptOffer)(nil),                    // 73: cozy.worker.v1.AttemptOffer
	(*AttemptAccepted)(nil),                 // 74: cozy.worker.v1.AttemptAccepted
	(*AttemptPlanSummary)(nil),              // 75: cozy.worker.v1.AttemptPlanSummary
	(*CancelAttempt)(nil),                   // 76: cozy.worker.v1.CancelAttempt
	(*AttemptOutcome)(nil),                  // 77: cozy.worker.v1.AttemptOutcome
	(*AttemptOutcomeBody)(nil),              // 78: cozy.worker.v1.AttemptOutcomeBody
	(*ArtifactReceiptRef)(nil),              // 79: cozy.worker.v1.ArtifactReceiptRef
	(*ArtifactReceipt)(nil),                 // 80: cozy.worker.v1.ArtifactReceipt
	(*ArtifactObjectSource)(nil),            // 81: cozy.worker.v1.ArtifactObjectSource
	(*ArtifactIntentFrame)(nil),             // 82: cozy.worker.v1.ArtifactIntentFrame
	(*ArtifactHostAck)(nil),                 // 83: cozy.worker.v1.ArtifactHostAck
	(*ArtifactReceiptFrame)(nil),            // 84: cozy.worker.v1.ArtifactReceiptFrame
	(*ArtifactTransactionStatus)(nil),       // 85: cozy.worker.v1.ArtifactTransactionStatus
	(*ArtifactReadRequest)(nil),             // 86: cozy.worker.v1.ArtifactReadRequest
	(*ArtifactReadResult)(nil),              // 87: cozy.worker.v1.ArtifactReadResult
	(*ArtifactObjectRef)(nil),               // 88: cozy.worker.v1.ArtifactObjectRef
	(*ArtifactUploadHeader)(nil),            // 89: cozy.worker.v1.ArtifactUploadHeader
	(*ArtifactUploadGrant)(nil),             // 90: cozy.worker.v1.ArtifactUploadGrant
	(*ArtifactTransferRequest)(nil),         // 91: cozy.worker.v1.ArtifactTransferRequest
	(*ArtifactTransferStatus)(nil),          // 92: cozy.worker.v1.ArtifactTransferStatus
	(*ModelSourceFileRequest)(nil),          // 93: cozy.worker.v1.ModelSourceFileRequest
	(*ModelSourceFileStatus)(nil),           // 94: cozy.worker.v1.ModelSourceFileStatus
	(*ModelSourcePrepareRequest)(nil),       // 95: cozy.worker.v1.ModelSourcePrepareRequest
	(*ModelSourcePrepared)(nil),             // 96: cozy.worker.v1.ModelSourcePrepared
	(*ArtifactFinalizeRequest)(nil),         // 97: cozy.worker.v1.ArtifactFinalizeRequest
	(*ArtifactFinalizeResult)(nil),          // 98: cozy.worker.v1.ArtifactFinalizeResult
	(*ResultEnvelope)(nil),                  // 99: cozy.worker.v1.ResultEnvelope
	(*AdjustmentRow)(nil),                   // 100: cozy.worker.v1.AdjustmentRow
	(*OutcomeCause)(nil),                    // 101: cozy.worker.v1.OutcomeCause
	(*ResourceShortfall)(nil),               // 102: cozy.worker.v1.ResourceShortfall
	(*AttemptOutcomeAck)(nil),               // 103: cozy.worker.v1.AttemptOutcomeAck
	(*JobCheckpointRequest)(nil),            // 104: cozy.worker.v1.JobCheckpointRequest
	(*JobCheckpointReceipt)(nil),            // 105: cozy.worker.v1.JobCheckpointReceipt
	(*CheckpointFault)(nil),                 // 106: cozy.worker.v1.CheckpointFault
	(*JobCheckpointAck)(nil),                // 107: cozy.worker.v1.JobCheckpointAck
	(*ProgressOpen)(nil),                    // 108: cozy.worker.v1.ProgressOpen
	(*AttemptProgress)(nil),                 // 109: cozy.worker.v1.AttemptProgress
	(*InvocationSpec)(nil),                  // 110: cozy.worker.v1.InvocationSpec
	(*InputBinding)(nil),                    // 111: cozy.worker.v1.InputBinding
	(*OutputBinding)(nil),                   // 112: cozy.worker.v1.OutputBinding
	(*ServingInvocationSpec)(nil),           // 113: cozy.worker.v1.ServingInvocationSpec
	(*JobInvocationSpec)(nil),               // 114: cozy.worker.v1.JobInvocationSpec
	(*DeliveryGrant)(nil),                   // 115: cozy.worker.v1.DeliveryGrant
	(*InputAccess)(nil),                     // 116: cozy.worker.v1.InputAccess
	(*OutputAccess)(nil),                    // 117: cozy.worker.v1.OutputAccess
	(*DeliveryAccessCredential)(nil),        // 118: cozy.worker.v1.DeliveryAccessCredential
	(*ResourceCaps)(nil),                    // 119: cozy.worker.v1.ResourceCaps
	(*PublicationContract)(nil),             // 120: cozy.worker.v1.PublicationContract
	(*WorkerResources)(nil),                 // 121: cozy.worker.v1.WorkerResources
	(*JobCapacity)(nil),                     // 122: cozy.worker.v1.JobCapacity
	(*HeldAttempt)(nil),                     // 123: cozy.worker.v1.HeldAttempt
	(*Fault)(nil),                           // 124: cozy.worker.v1.Fault
	(*OutputManifest)(nil),                  // 125: cozy.worker.v1.OutputManifest
	(*OutputEntry)(nil),                     // 126: cozy.worker.v1.OutputEntry
	(*AttemptMetrics)(nil),                  // 127: cozy.worker.v1.AttemptMetrics
	(*TriageBundleRef)(nil),                 // 128: cozy.worker.v1.TriageBundleRef
}
var file_cozy_worker_v1_worker_proto_depIdxs = []int32{
	30,  // 0: cozy.worker.v1.PreparePackageSetRequest.files:type_name -> cozy.worker.v1.LocalDownloadFile
	0,   // 1: cozy.worker.v1.LocalDownloadFile.kind:type_name -> cozy.worker.v1.LocalDownloadKind
	48,  // 2: cozy.worker.v1.PreparePackageSetResult.placement_set:type_name -> cozy.worker.v1.DesiredPlacementSet
	54,  // 3: cozy.worker.v1.PreparedModelSource.manifest:type_name -> cozy.worker.v1.Ref
	33,  // 4: cozy.worker.v1.PrepareModelSourceRequest.profiles:type_name -> cozy.worker.v1.ModelSourceProfile
	32,  // 5: cozy.worker.v1.PrepareModelSourceRequest.files:type_name -> cozy.worker.v1.LocalModelSourceFile
	26,  // 6: cozy.worker.v1.PrepareModelSourceResult.outcome:type_name -> cozy.worker.v1.ModelSourcePrepareOutcome
	34,  // 7: cozy.worker.v1.PrepareModelSourceResult.sources:type_name -> cozy.worker.v1.PreparedModelSource
	39,  // 8: cozy.worker.v1.RecordOwnerFrame.claim:type_name -> cozy.worker.v1.Claim
	46,  // 9: cozy.worker.v1.RecordOwnerFrame.desired_state:type_name -> cozy.worker.v1.DesiredWorkerState
	73,  // 10: cozy.worker.v1.RecordOwnerFrame.attempt_offer:type_name -> cozy.worker.v1.AttemptOffer
	76,  // 11: cozy.worker.v1.RecordOwnerFrame.cancel_attempt:type_name -> cozy.worker.v1.CancelAttempt
	103, // 12: cozy.worker.v1.RecordOwnerFrame.outcome_ack:type_name -> cozy.worker.v1.AttemptOutcomeAck
	105, // 13: cozy.worker.v1.RecordOwnerFrame.checkpoint_receipt:type_name -> cozy.worker.v1.JobCheckpointReceipt
	45,  // 14: cozy.worker.v1.RecordOwnerFrame.snapshot_ack:type_name -> cozy.worker.v1.SnapshotAck
	97,  // 15: cozy.worker.v1.RecordOwnerFrame.artifact_finalize_request:type_name -> cozy.worker.v1.ArtifactFinalizeRequest
	83,  // 16: cozy.worker.v1.RecordOwnerFrame.artifact_host_ack:type_name -> cozy.worker.v1.ArtifactHostAck
	91,  // 17: cozy.worker.v1.RecordOwnerFrame.artifact_transfer_request:type_name -> cozy.worker.v1.ArtifactTransferRequest
	86,  // 18: cozy.worker.v1.RecordOwnerFrame.artifact_read_request:type_name -> cozy.worker.v1.ArtifactReadRequest
	93,  // 19: cozy.worker.v1.RecordOwnerFrame.model_source_file_request:type_name -> cozy.worker.v1.ModelSourceFileRequest
	95,  // 20: cozy.worker.v1.RecordOwnerFrame.model_source_prepare_request:type_name -> cozy.worker.v1.ModelSourcePrepareRequest
	41,  // 21: cozy.worker.v1.WorkerFrame.claim_ack:type_name -> cozy.worker.v1.ClaimAck
	67,  // 22: cozy.worker.v1.WorkerFrame.observed_state:type_name -> cozy.worker.v1.ObservedWorkerState
	74,  // 23: cozy.worker.v1.WorkerFrame.attempt_accepted:type_name -> cozy.worker.v1.AttemptAccepted
	77,  // 24: cozy.worker.v1.WorkerFrame.attempt_outcome:type_name -> cozy.worker.v1.AttemptOutcome
	42,  // 25: cozy.worker.v1.WorkerFrame.boot_failure:type_name -> cozy.worker.v1.BootFailure
	104, // 26: cozy.worker.v1.WorkerFrame.checkpoint_request:type_name -> cozy.worker.v1.JobCheckpointRequest
	107, // 27: cozy.worker.v1.WorkerFrame.checkpoint_ack:type_name -> cozy.worker.v1.JobCheckpointAck
	43,  // 28: cozy.worker.v1.WorkerFrame.snapshot:type_name -> cozy.worker.v1.WorkerSnapshot
	98,  // 29: cozy.worker.v1.WorkerFrame.artifact_finalize_result:type_name -> cozy.worker.v1.ArtifactFinalizeResult
	82,  // 30: cozy.worker.v1.WorkerFrame.artifact_intent:type_name -> cozy.worker.v1.ArtifactIntentFrame
	84,  // 31: cozy.worker.v1.WorkerFrame.artifact_receipt:type_name -> cozy.worker.v1.ArtifactReceiptFrame
	87,  // 32: cozy.worker.v1.WorkerFrame.artifact_read_result:type_name -> cozy.worker.v1.ArtifactReadResult
	92,  // 33: cozy.worker.v1.WorkerFrame.artifact_transfer_status:type_name -> cozy.worker.v1.ArtifactTransferStatus
	94,  // 34: cozy.worker.v1.WorkerFrame.model_source_file_status:type_name -> cozy.worker.v1.ModelSourceFileStatus
	96,  // 35: cozy.worker.v1.WorkerFrame.model_source_prepared:type_name -> cozy.worker.v1.ModelSourcePrepared
	12,  // 36: cozy.worker.v1.ClaimAck.rejection:type_name -> cozy.worker.v1.ClaimRejection
	121, // 37: cozy.worker.v1.ClaimAck.resources:type_name -> cozy.worker.v1.WorkerResources
	14,  // 38: cozy.worker.v1.BootFailure.reason:type_name -> cozy.worker.v1.BootFailureReason
	121, // 39: cozy.worker.v1.BootFailure.resources:type_name -> cozy.worker.v1.WorkerResources
	2,   // 40: cozy.worker.v1.WorkerSnapshotBody.worker_phase:type_name -> cozy.worker.v1.WorkerPhase
	68,  // 41: cozy.worker.v1.WorkerSnapshotBody.placements:type_name -> cozy.worker.v1.PlacementStatus
	5,   // 42: cozy.worker.v1.WorkerSnapshotBody.admission_state:type_name -> cozy.worker.v1.AdmissionState
	123, // 43: cozy.worker.v1.WorkerSnapshotBody.held_attempts:type_name -> cozy.worker.v1.HeldAttempt
	85,  // 44: cozy.worker.v1.WorkerSnapshotBody.artifact_transactions:type_name -> cozy.worker.v1.ArtifactTransactionStatus
	1,   // 45: cozy.worker.v1.DesiredWorkerState.posture:type_name -> cozy.worker.v1.Posture
	66,  // 46: cozy.worker.v1.DesiredWorkerState.job:type_name -> cozy.worker.v1.JobDirective
	48,  // 47: cozy.worker.v1.DesiredWorkerState.placement_set:type_name -> cozy.worker.v1.DesiredPlacementSet
	47,  // 48: cozy.worker.v1.DesiredWorkerState.package_set:type_name -> cozy.worker.v1.DesiredPackageSet
	53,  // 49: cozy.worker.v1.PlacementSet.placements:type_name -> cozy.worker.v1.Placement
	51,  // 50: cozy.worker.v1.DownloadDelegation.models:type_name -> cozy.worker.v1.DownloadModelRef
	52,  // 51: cozy.worker.v1.DownloadDelegation.packages:type_name -> cozy.worker.v1.DownloadPackageRef
	56,  // 52: cozy.worker.v1.Placement.package:type_name -> cozy.worker.v1.PackageSelection
	57,  // 53: cozy.worker.v1.Placement.development:type_name -> cozy.worker.v1.DevelopmentPackage
	54,  // 54: cozy.worker.v1.Placement.package_descriptor:type_name -> cozy.worker.v1.Ref
	59,  // 55: cozy.worker.v1.Placement.models:type_name -> cozy.worker.v1.Model
	60,  // 56: cozy.worker.v1.Placement.entrypoints:type_name -> cozy.worker.v1.Entrypoint
	58,  // 57: cozy.worker.v1.Placement.environment:type_name -> cozy.worker.v1.Environment
	54,  // 58: cozy.worker.v1.WheelFact.ref:type_name -> cozy.worker.v1.Ref
	55,  // 59: cozy.worker.v1.PackageSelection.project_wheel:type_name -> cozy.worker.v1.WheelFact
	55,  // 60: cozy.worker.v1.Environment.wheels:type_name -> cozy.worker.v1.WheelFact
	54,  // 61: cozy.worker.v1.Model.manifest:type_name -> cozy.worker.v1.Ref
	61,  // 62: cozy.worker.v1.Entrypoint.slots:type_name -> cozy.worker.v1.Slot
	62,  // 63: cozy.worker.v1.Slot.config:type_name -> cozy.worker.v1.Config
	64,  // 64: cozy.worker.v1.Slot.components:type_name -> cozy.worker.v1.Component
	54,  // 65: cozy.worker.v1.Slot.model_construction_contract:type_name -> cozy.worker.v1.Ref
	65,  // 66: cozy.worker.v1.Slot.stamps:type_name -> cozy.worker.v1.Stamp
	54,  // 67: cozy.worker.v1.Config.document:type_name -> cozy.worker.v1.Ref
	63,  // 68: cozy.worker.v1.Config.assets:type_name -> cozy.worker.v1.Asset
	54,  // 69: cozy.worker.v1.Asset.ref:type_name -> cozy.worker.v1.Ref
	119, // 70: cozy.worker.v1.JobDirective.resource_caps:type_name -> cozy.worker.v1.ResourceCaps
	120, // 71: cozy.worker.v1.JobDirective.publication_contract:type_name -> cozy.worker.v1.PublicationContract
	123, // 72: cozy.worker.v1.ObservedWorkerState.held_attempts:type_name -> cozy.worker.v1.HeldAttempt
	124, // 73: cozy.worker.v1.ObservedWorkerState.faults:type_name -> cozy.worker.v1.Fault
	72,  // 74: cozy.worker.v1.ObservedWorkerState.activity:type_name -> cozy.worker.v1.ActivityEvent
	122, // 75: cozy.worker.v1.ObservedWorkerState.job_capacity:type_name -> cozy.worker.v1.JobCapacity
	68,  // 76: cozy.worker.v1.ObservedWorkerState.placements:type_name -> cozy.worker.v1.PlacementStatus
	5,   // 77: cozy.worker.v1.ObservedWorkerState.admission_state:type_name -> cozy.worker.v1.AdmissionState
	2,   // 78: cozy.worker.v1.ObservedWorkerState.worker_phase:type_name -> cozy.worker.v1.WorkerPhase
	124, // 79: cozy.worker.v1.PlacementStatus.faults:type_name -> cozy.worker.v1.Fault
	71,  // 80: cozy.worker.v1.PlacementStatus.accelerator:type_name -> cozy.worker.v1.AcceleratorQualification
	3,   // 81: cozy.worker.v1.PlacementStatus.materialization:type_name -> cozy.worker.v1.MaterializationState
	4,   // 82: cozy.worker.v1.PlacementStatus.serving:type_name -> cozy.worker.v1.ServingState
	69,  // 83: cozy.worker.v1.PlacementStatus.acquisition:type_name -> cozy.worker.v1.PlacementAcquisitionObservation
	70,  // 84: cozy.worker.v1.PlacementAcquisitionObservation.package:type_name -> cozy.worker.v1.AcquisitionLegObservation
	70,  // 85: cozy.worker.v1.PlacementAcquisitionObservation.model:type_name -> cozy.worker.v1.AcquisitionLegObservation
	115, // 86: cozy.worker.v1.AttemptOffer.grant:type_name -> cozy.worker.v1.DeliveryGrant
	75,  // 87: cozy.worker.v1.AttemptAccepted.plan:type_name -> cozy.worker.v1.AttemptPlanSummary
	11,  // 88: cozy.worker.v1.CancelAttempt.reason:type_name -> cozy.worker.v1.CancelReason
	8,   // 89: cozy.worker.v1.AttemptOutcomeBody.status:type_name -> cozy.worker.v1.OutcomeStatus
	125, // 90: cozy.worker.v1.AttemptOutcomeBody.output_manifest:type_name -> cozy.worker.v1.OutputManifest
	127, // 91: cozy.worker.v1.AttemptOutcomeBody.metrics:type_name -> cozy.worker.v1.AttemptMetrics
	128, // 92: cozy.worker.v1.AttemptOutcomeBody.triage_bundle:type_name -> cozy.worker.v1.TriageBundleRef
	101, // 93: cozy.worker.v1.AttemptOutcomeBody.cause:type_name -> cozy.worker.v1.OutcomeCause
	99,  // 94: cozy.worker.v1.AttemptOutcomeBody.result:type_name -> cozy.worker.v1.ResultEnvelope
	79,  // 95: cozy.worker.v1.AttemptOutcomeBody.artifact_receipts:type_name -> cozy.worker.v1.ArtifactReceiptRef
	17,  // 96: cozy.worker.v1.ArtifactHostAck.stage:type_name -> cozy.worker.v1.ArtifactHostStage
	18,  // 97: cozy.worker.v1.ArtifactHostAck.outcome:type_name -> cozy.worker.v1.ArtifactHostOutcome
	19,  // 98: cozy.worker.v1.ArtifactHostAck.refusal:type_name -> cozy.worker.v1.ArtifactHostRefusal
	79,  // 99: cozy.worker.v1.ArtifactHostAck.artifact_receipt:type_name -> cozy.worker.v1.ArtifactReceiptRef
	54,  // 100: cozy.worker.v1.ArtifactHostAck.manifest:type_name -> cozy.worker.v1.Ref
	79,  // 101: cozy.worker.v1.ArtifactReceiptFrame.artifact_receipt:type_name -> cozy.worker.v1.ArtifactReceiptRef
	81,  // 102: cozy.worker.v1.ArtifactReceiptFrame.objects:type_name -> cozy.worker.v1.ArtifactObjectSource
	54,  // 103: cozy.worker.v1.ArtifactReceiptFrame.manifest:type_name -> cozy.worker.v1.Ref
	20,  // 104: cozy.worker.v1.ArtifactTransactionStatus.state:type_name -> cozy.worker.v1.ArtifactTransactionState
	21,  // 105: cozy.worker.v1.ArtifactReadResult.outcome:type_name -> cozy.worker.v1.ArtifactReadOutcome
	22,  // 106: cozy.worker.v1.ArtifactReadResult.refusal:type_name -> cozy.worker.v1.ArtifactReadRefusal
	89,  // 107: cozy.worker.v1.ArtifactUploadGrant.required_headers:type_name -> cozy.worker.v1.ArtifactUploadHeader
	90,  // 108: cozy.worker.v1.ArtifactTransferRequest.upload_grant:type_name -> cozy.worker.v1.ArtifactUploadGrant
	88,  // 109: cozy.worker.v1.ArtifactTransferRequest.held:type_name -> cozy.worker.v1.ArtifactObjectRef
	23,  // 110: cozy.worker.v1.ArtifactTransferStatus.state:type_name -> cozy.worker.v1.ArtifactTransferState
	24,  // 111: cozy.worker.v1.ModelSourceFileRequest.provider:type_name -> cozy.worker.v1.ModelSourceProvider
	25,  // 112: cozy.worker.v1.ModelSourceFileStatus.state:type_name -> cozy.worker.v1.ModelSourceFileState
	33,  // 113: cozy.worker.v1.ModelSourcePrepareRequest.profiles:type_name -> cozy.worker.v1.ModelSourceProfile
	26,  // 114: cozy.worker.v1.ModelSourcePrepared.outcome:type_name -> cozy.worker.v1.ModelSourcePrepareOutcome
	34,  // 115: cozy.worker.v1.ModelSourcePrepared.sources:type_name -> cozy.worker.v1.PreparedModelSource
	27,  // 116: cozy.worker.v1.ArtifactFinalizeRequest.disposition:type_name -> cozy.worker.v1.ArtifactFinalizeDisposition
	28,  // 117: cozy.worker.v1.ArtifactFinalizeResult.outcome:type_name -> cozy.worker.v1.ArtifactFinalizeOutcome
	79,  // 118: cozy.worker.v1.ArtifactFinalizeResult.artifact_receipt:type_name -> cozy.worker.v1.ArtifactReceiptRef
	126, // 119: cozy.worker.v1.ResultEnvelope.result_blob:type_name -> cozy.worker.v1.OutputEntry
	100, // 120: cozy.worker.v1.ResultEnvelope.adjustments:type_name -> cozy.worker.v1.AdjustmentRow
	9,   // 121: cozy.worker.v1.OutcomeCause.code:type_name -> cozy.worker.v1.CauseCode
	10,  // 122: cozy.worker.v1.OutcomeCause.origin:type_name -> cozy.worker.v1.CauseOrigin
	102, // 123: cozy.worker.v1.OutcomeCause.shortfall:type_name -> cozy.worker.v1.ResourceShortfall
	126, // 124: cozy.worker.v1.JobCheckpointRequest.artifact:type_name -> cozy.worker.v1.OutputEntry
	15,  // 125: cozy.worker.v1.JobCheckpointReceipt.outcome:type_name -> cozy.worker.v1.CheckpointOutcome
	106, // 126: cozy.worker.v1.JobCheckpointReceipt.fault:type_name -> cozy.worker.v1.CheckpointFault
	16,  // 127: cozy.worker.v1.CheckpointFault.code:type_name -> cozy.worker.v1.CheckpointFaultCode
	111, // 128: cozy.worker.v1.InvocationSpec.inputs:type_name -> cozy.worker.v1.InputBinding
	112, // 129: cozy.worker.v1.InvocationSpec.outputs:type_name -> cozy.worker.v1.OutputBinding
	113, // 130: cozy.worker.v1.InvocationSpec.serving:type_name -> cozy.worker.v1.ServingInvocationSpec
	114, // 131: cozy.worker.v1.InvocationSpec.job:type_name -> cozy.worker.v1.JobInvocationSpec
	120, // 132: cozy.worker.v1.JobInvocationSpec.publication_contract:type_name -> cozy.worker.v1.PublicationContract
	118, // 133: cozy.worker.v1.DeliveryGrant.credential:type_name -> cozy.worker.v1.DeliveryAccessCredential
	116, // 134: cozy.worker.v1.DeliveryGrant.inputs:type_name -> cozy.worker.v1.InputAccess
	117, // 135: cozy.worker.v1.DeliveryGrant.outputs:type_name -> cozy.worker.v1.OutputAccess
	112, // 136: cozy.worker.v1.PublicationContract.outputs:type_name -> cozy.worker.v1.OutputBinding
	6,   // 137: cozy.worker.v1.HeldAttempt.kind:type_name -> cozy.worker.v1.AttemptKind
	7,   // 138: cozy.worker.v1.HeldAttempt.state:type_name -> cozy.worker.v1.AttemptState
	13,  // 139: cozy.worker.v1.Fault.kind:type_name -> cozy.worker.v1.FaultKind
	126, // 140: cozy.worker.v1.OutputManifest.outputs:type_name -> cozy.worker.v1.OutputEntry
	37,  // 141: cozy.worker.v1.WorkerControl.Control:input_type -> cozy.worker.v1.RecordOwnerFrame
	108, // 142: cozy.worker.v1.WorkerControl.WatchProgress:input_type -> cozy.worker.v1.ProgressOpen
	29,  // 143: cozy.worker.v1.RuntimePreparation.PreparePackageSet:input_type -> cozy.worker.v1.PreparePackageSetRequest
	35,  // 144: cozy.worker.v1.RuntimePreparation.PrepareModelSource:input_type -> cozy.worker.v1.PrepareModelSourceRequest
	38,  // 145: cozy.worker.v1.WorkerControl.Control:output_type -> cozy.worker.v1.WorkerFrame
	109, // 146: cozy.worker.v1.WorkerControl.WatchProgress:output_type -> cozy.worker.v1.AttemptProgress
	31,  // 147: cozy.worker.v1.RuntimePreparation.PreparePackageSet:output_type -> cozy.worker.v1.PreparePackageSetResult
	36,  // 148: cozy.worker.v1.RuntimePreparation.PrepareModelSource:output_type -> cozy.worker.v1.PrepareModelSourceResult
	145, // [145:149] is the sub-list for method output_type
	141, // [141:145] is the sub-list for method input_type
	141, // [141:141] is the sub-list for extension type_name
	141, // [141:141] is the sub-list for extension extendee
	0,   // [0:141] is the sub-list for field type_name
}

func init() { file_cozy_worker_v1_worker_proto_init() }
func file_cozy_worker_v1_worker_proto_init() {
	if File_cozy_worker_v1_worker_proto != nil {
		return
	}
	file_cozy_worker_v1_worker_proto_msgTypes[8].OneofWrappers = []any{
		(*RecordOwnerFrame_Claim)(nil),
		(*RecordOwnerFrame_DesiredState)(nil),
		(*RecordOwnerFrame_AttemptOffer)(nil),
		(*RecordOwnerFrame_CancelAttempt)(nil),
		(*RecordOwnerFrame_OutcomeAck)(nil),
		(*RecordOwnerFrame_CheckpointReceipt)(nil),
		(*RecordOwnerFrame_SnapshotAck)(nil),
		(*RecordOwnerFrame_ArtifactFinalizeRequest)(nil),
		(*RecordOwnerFrame_ArtifactHostAck)(nil),
		(*RecordOwnerFrame_ArtifactTransferRequest)(nil),
		(*RecordOwnerFrame_ArtifactReadRequest)(nil),
		(*RecordOwnerFrame_ModelSourceFileRequest)(nil),
		(*RecordOwnerFrame_ModelSourcePrepareRequest)(nil),
	}
	file_cozy_worker_v1_worker_proto_msgTypes[9].OneofWrappers = []any{
		(*WorkerFrame_ClaimAck)(nil),
		(*WorkerFrame_ObservedState)(nil),
		(*WorkerFrame_AttemptAccepted)(nil),
		(*WorkerFrame_AttemptOutcome)(nil),
		(*WorkerFrame_BootFailure)(nil),
		(*WorkerFrame_CheckpointRequest)(nil),
		(*WorkerFrame_CheckpointAck)(nil),
		(*WorkerFrame_Snapshot)(nil),
		(*WorkerFrame_ArtifactFinalizeResult)(nil),
		(*WorkerFrame_ArtifactIntent)(nil),
		(*WorkerFrame_ArtifactReceipt)(nil),
		(*WorkerFrame_ArtifactReadResult)(nil),
		(*WorkerFrame_ArtifactTransferStatus)(nil),
		(*WorkerFrame_ModelSourceFileStatus)(nil),
		(*WorkerFrame_ModelSourcePrepared)(nil),
	}
	file_cozy_worker_v1_worker_proto_msgTypes[17].OneofWrappers = []any{
		(*DesiredWorkerState_Job)(nil),
		(*DesiredWorkerState_PlacementSet)(nil),
		(*DesiredWorkerState_PackageSet)(nil),
	}
	file_cozy_worker_v1_worker_proto_msgTypes[24].OneofWrappers = []any{
		(*Placement_Package)(nil),
		(*Placement_Development)(nil),
	}
	file_cozy_worker_v1_worker_proto_msgTypes[62].OneofWrappers = []any{
		(*ArtifactTransferRequest_UploadGrant)(nil),
		(*ArtifactTransferRequest_Held)(nil),
	}
	file_cozy_worker_v1_worker_proto_msgTypes[81].OneofWrappers = []any{
		(*InvocationSpec_Serving)(nil),
		(*InvocationSpec_Job)(nil),
	}
	type x struct{}
	out := protoimpl.TypeBuilder{
		File: protoimpl.DescBuilder{
			GoPackagePath: reflect.TypeOf(x{}).PkgPath(),
			RawDescriptor: unsafe.Slice(unsafe.StringData(file_cozy_worker_v1_worker_proto_rawDesc), len(file_cozy_worker_v1_worker_proto_rawDesc)),
			NumEnums:      29,
			NumMessages:   100,
			NumExtensions: 0,
			NumServices:   2,
		},
		GoTypes:           file_cozy_worker_v1_worker_proto_goTypes,
		DependencyIndexes: file_cozy_worker_v1_worker_proto_depIdxs,
		EnumInfos:         file_cozy_worker_v1_worker_proto_enumTypes,
		MessageInfos:      file_cozy_worker_v1_worker_proto_msgTypes,
	}.Build()
	File_cozy_worker_v1_worker_proto = out.File
	file_cozy_worker_v1_worker_proto_goTypes = nil
	file_cozy_worker_v1_worker_proto_depIdxs = nil
}
