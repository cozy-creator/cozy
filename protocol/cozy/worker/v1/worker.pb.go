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
// release train number. WIRE_MINIMUM and WIRE_MINOR define the compatible range. Ordinary
// additive changes preserve the floor; the explicit pre-freeze minor38 hardcut raises it.
// Clients probe ProtocolInfo before Claim/preparation; servers reject below-floor claims
// before ownership mutation. After the v1 freeze, breaking changes require a new package major.
// R7 IS AN AUTHORING RULE ONLY: `reserved` numbers and names are compiler-enforced
// tombstones against reuse; ordinary proto3 decoders do not refuse them on the wire and no
// runtime polices them. All generated bindings and fixtures move together for this hardcut.
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
// THE CONTROL STREAM CARRIES CONTROL, NOT CONTENT. No frame moves an object's bytes. Every
// content-bearing field is bounded by a constant, and every one of those constants is <=
// `MaxInlineControlBytes` (4 MiB) PER TRANSFER -- not per chunk. The distinction is the whole
// rule: a 4 MiB frame is fine, and a 4 MiB frame sent 2,048 times to move 8 GiB is the defect,
// because the total it moves is bounded by nothing. Anything larger than the ceiling travels
// as a presigned URL to the process that holds the bytes, which is what `WeightsUploadGrant`,
// `LocalPackageFileGrant` and `DeliveryGrant.outputs` already do. `MaxWeightsReadBytes` is a
// bounded RANGE of an object an owner explicitly asked for, not a quantum of an unbounded whole.
//
// The rule has NO carve-out and is mechanically enforced on the BOUND: tensorhub's
// `scripts/fence.py` convicts any content-bearing `Max*Bytes` constant above the ceiling,
// whatever the field carrying it is called. th-094 retired the last exemption, local package
// upload, which moved 1 GiB in 1 MiB frames.
//
// DOCUMENT VERSIONS. Every current pre-release document is `/1`. A digest-fenced document is NOT
// additively versioned: an unknown key REFUSES, and shape changes hardcut the `/1` definition
// across every writer, reader, and stored byte together. The canonical `format` tag is the
// message's full name plus `/1`; there are no compatibility readers or version aliases.
//
// THE ENVELOPE (every message, fields 1-3): the ownership + boot fence, checked BEFORE any body
// field is read, in this order:
//   1 `record_owner_epoch`     - durable monotonic AUTHORITY epoch, minted by whatever
//                                grants ownership. Older than the accepted claim => dropped.
//   2 `control_stream_epoch`   - worker-minted, incremented for EVERY accepted control
//                                stream; a frame from a superseded stream is dropped.
//   3 `worker_boot_id`         - minted at worker-process boot, never reused; a frame
//                                addressed to a dead boot is dropped. An id, not an epoch:
//                                it need only DIFFER, never be greater.
// FIELD 4 IS RESERVED IN EVERY MESSAGE (#446): the old global `executor_incarnation` is GONE.
// The worker BOOT id is machine truth; each placement's EXECUTOR epoch is placement truth
// (`PlacementStatus.executor_epoch`) and rides attempt-scoped facts, not the envelope. The
// attempt fence triple is (request_id, attempt_ordinal, invocation_spec_digest).
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
// ONE ADMISSION GATE (#472e/#482/#486c): per-placement `attempt_credits` are DELETED — N
// counters over ONE serialized device advertise N x the real capacity, and the defect is
// arithmetic. Execution capacity is a WORKER property (`admission_epoch` +
// `admission_state` + `available_attempt_slots`); dispatchability is a PLACEMENT property (the
// serving axis + dispatchable_binding_digests). Only `admission_epoch` FENCES; the other two are
// the GATE — "not accepting" / "full" — not a fence against a stale actor. Per-placement
// `readiness_epoch` is DELETED, not renamed: one fence, not two to keep consistent.
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
//
// THE HOST APPENDS, NEVER RE-AUTHORS (proto-025, minor 21). A pod's supervisor is the
// pod-resident half of the HOST role the Cozy daemon performs in-process locally: it downloads
// and verifies, asks the Runtime's loopback preparation for PlacementSet bytes, moves objects,
// and keeps the pod ledger. Minor 21 gives that role its own lanes so the host never rewrites
// a frame in flight: `PodHost` (owner <-> host, a second service on the pod listener),
// `RuntimeWeights` (host <-> Runtime, loopback), and a host-authored SECOND snapshot document
// beside the worker's, which passes through BYTE-IDENTICAL. The direction-annotated slots
// still on RecordOwnerFrame / WorkerFrame / DesiredWorkerState.mode are the pre-21 shape of
// the same lanes; they are RETIRING and are reserved at the first minor after every consumer
// is off them. Nothing at 21 is removed or renumbered.
//
// DEVICE LANES (proto-024, minor 22). The serialized resource inside one worker is a DEVICE
// LANE — a set of envelope-local device ordinals, one attempt seat, one ledger — not the worker
// (cr-066). The one-counter-per-serialized-resource principle behind the admission gate (#472e)
// therefore puts the seats PER LANE: `ObservedWorkerState.lanes[]`, with the worker-level
// `available_attempt_slots` provably their sum. A lane carries K >= 1 ordinals. K == 1 is every
// worker shipped so far and is byte-identical on the wire but for the new repeated field. K > 1
// is a GROUP lane: ONE executor sealed to all K devices, one seat, one attempt at a time across
// the group (cr-018's sequence-parallel door: `gpu_count` is never request concurrency). The
// RecordOwner ASSIGNS a group by pinning a placement to K ordinals (`DesiredPlacementSet.
// device_pins`, beside the digest-fenced document, never inside it: where a placement sits on
// THIS worker is not part of what it serves); the worker validates the pin against its envelope
// and the package's declared construction and refuses typed. `PlacementStatus.device_lane_id`
// reports the lane a placement landed on. Nothing at 22 is removed or renumbered.
//
// THE ATTEMPT QUEUE (proto-026, minor 23; gpu-hot.md §3-§6). The worker ORDERS its lane; the
// RecordOwner OFFERS depth. Acceptance is queue ADMISSION — journaled, hydrated, queued on a
// lane — and binds neither a plan nor an executor epoch: both bind at DEVICE ENTRY. An attempt
// on a lane is QUEUED -> RUNNING -> DEVICE_RELEASED -> OUTCOME_PENDING_ACK; the device seat is
// the RUNNING held attempt on the lane, never a count, and the post phase (encode, digest,
// write, outcome) runs beside the next occupant. Every accepted attempt appears in
// `held_attempts` from admission to ack with its lane, worker order, overtake count and budget
// (`HeldAttempt` 10-14); `DeviceLane.available_attempt_slots` becomes the offers the lane will
// still ADMIT TO ITS QUEUE now; `AttemptMetrics.post_ms` and an honest `device_lease_ms` split
// the two phases. `admission_epoch` no longer bumps on executor respawn: a queued attempt is
// bound to no executor. The four `AttemptAccepted` slots that bound plan and epoch at
// acceptance (8, 9, 10, 12) are RESERVED — a PRE-LAUNCH HARD CUT, no dual reading, no shim; an
// owner below 23 is refused typed at Claim (`attempt_queue_unsupported`). RESIDENCY
// (residency-aware-routing.md §2/§6, same minor): two observed SETS cross the wire and nothing
// else — `DeviceLane.resident_placement_ids` (executor holds device bytes on this lane now) and
// `ObservedWorkerState.held_manifests` (TensorFS manifests the verified store holds complete,
// placed or not), both also on `WorkerSnapshotBody/1`. No bytes, no headroom, no plan: the
// orchestrator routes on counts and membership; VRAM arithmetic stays where it is measured.
//
// THE PACKAGE WIRE (xs-018/xs-019/xs-017, minor 30). What a pod is told about a published
// package is the hub's own release material and nothing derived: `application`, the
// interface-declared entrypoint model-slot paths, the placed image's pinned inventory, and the
// release's hash-pinned requirements export — all hub-known before the pod exists. Runtime
// preparation receives NO files: wheels come from the package indexes under --require-hashes,
// the package interface is derived from the installed environment, and model manifests/objects
// are resolved from the local TensorFS Store the privileged host admitted them to (proto-030).
// `LocalDownloadFile` and `LocalDownloadKind` are DELETED with the per-file handoff; the
// local-package plane keeps its explicitly supplied wheels — the only files with no index
// source — and loses its `kind` rows: the project wheel is the row that MEASURES as the
// package's own distribution/release. `PackageSelection.project_wheel` and `Environment.wheels`
// are deleted: a pod that never retains wheel bytes cannot re-measure WheelFacts, so a
// published install's identity is its locked requirements export
// (`Environment.locked_requirements`) over the image inventory, not a wheel list.
//
// THE STORAGE CUT (proto-031, minor 31). DownloadDelegation no longer reports held manifests:
// TensorFS computes the missing closure locally and asks only for those bytes. Preparation names
// one immutable `install_root`, never an environment overlay. Local editable wheels stage below
// that install namespace. The proto-026 residency sets on ObservedWorkerState and
// WorkerSnapshotBody remain; the orchestrator still routes on them.

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

type ChildCallState int32

const (
	ChildCallState_CHILD_CALL_STATE_UNSPECIFIED ChildCallState = 0
	ChildCallState_CHILD_CALL_STATE_PENDING     ChildCallState = 1 // owner durably accepted; child may be queued or running
	ChildCallState_CHILD_CALL_STATE_SUCCEEDED   ChildCallState = 2
	ChildCallState_CHILD_CALL_STATE_REFUSED     ChildCallState = 3
	ChildCallState_CHILD_CALL_STATE_FAILED      ChildCallState = 4
	ChildCallState_CHILD_CALL_STATE_CANCELED    ChildCallState = 5
)

// Enum value maps for ChildCallState.
var (
	ChildCallState_name = map[int32]string{
		0: "CHILD_CALL_STATE_UNSPECIFIED",
		1: "CHILD_CALL_STATE_PENDING",
		2: "CHILD_CALL_STATE_SUCCEEDED",
		3: "CHILD_CALL_STATE_REFUSED",
		4: "CHILD_CALL_STATE_FAILED",
		5: "CHILD_CALL_STATE_CANCELED",
	}
	ChildCallState_value = map[string]int32{
		"CHILD_CALL_STATE_UNSPECIFIED": 0,
		"CHILD_CALL_STATE_PENDING":     1,
		"CHILD_CALL_STATE_SUCCEEDED":   2,
		"CHILD_CALL_STATE_REFUSED":     3,
		"CHILD_CALL_STATE_FAILED":      4,
		"CHILD_CALL_STATE_CANCELED":    5,
	}
)

func (x ChildCallState) Enum() *ChildCallState {
	p := new(ChildCallState)
	*p = x
	return p
}

func (x ChildCallState) String() string {
	return protoimpl.X.EnumStringOf(x.Descriptor(), protoreflect.EnumNumber(x))
}

func (ChildCallState) Descriptor() protoreflect.EnumDescriptor {
	return file_cozy_worker_v1_worker_proto_enumTypes[0].Descriptor()
}

func (ChildCallState) Type() protoreflect.EnumType {
	return &file_cozy_worker_v1_worker_proto_enumTypes[0]
}

func (x ChildCallState) Number() protoreflect.EnumNumber {
	return protoreflect.EnumNumber(x)
}

// Deprecated: Use ChildCallState.Descriptor instead.
func (ChildCallState) EnumDescriptor() ([]byte, []int) {
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

// The lane order (proto-026, gpu-hot.md §3): QUEUED -> RUNNING -> DEVICE_RELEASED ->
// OUTCOME_PENDING_ACK. Numbers are the normative part; they are not in phase order.
type AttemptState int32

const (
	AttemptState_ATTEMPT_STATE_UNSPECIFIED AttemptState = 0
	AttemptState_ATTEMPT_STATE_RUNNING     AttemptState = 1 // the DEVICE phase: plan priced, executor bound,
	// on the device — THE seat of its lane
	AttemptState_ATTEMPT_STATE_OUTCOME_PENDING_ACK AttemptState = 2 // was TERMINAL_PENDING_ACK
	// #507d: the attempt is HELD but its journal record is not durable. The frozen wire could
	// only say this as RUNNING + a LOCAL_SAFETY fault compound — honest, but compound.
	AttemptState_ATTEMPT_STATE_HELD_UNDURABLE AttemptState = 3
	AttemptState_ATTEMPT_STATE_QUEUED         AttemptState = 4 // accepted into the lane queue, hydrated, bound to
	// no executor (executor_epoch 0, plan_digest empty)
	AttemptState_ATTEMPT_STATE_DEVICE_RELEASED AttemptState = 5 // the device phase is over (executor replied, bytes
)

// Enum value maps for AttemptState.
var (
	AttemptState_name = map[int32]string{
		0: "ATTEMPT_STATE_UNSPECIFIED",
		1: "ATTEMPT_STATE_RUNNING",
		2: "ATTEMPT_STATE_OUTCOME_PENDING_ACK",
		3: "ATTEMPT_STATE_HELD_UNDURABLE",
		4: "ATTEMPT_STATE_QUEUED",
		5: "ATTEMPT_STATE_DEVICE_RELEASED",
	}
	AttemptState_value = map[string]int32{
		"ATTEMPT_STATE_UNSPECIFIED":         0,
		"ATTEMPT_STATE_RUNNING":             1,
		"ATTEMPT_STATE_OUTCOME_PENDING_ACK": 2,
		"ATTEMPT_STATE_HELD_UNDURABLE":      3,
		"ATTEMPT_STATE_QUEUED":              4,
		"ATTEMPT_STATE_DEVICE_RELEASED":     5,
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
	CauseCode_CAUSE_CODE_ADMISSION_EPOCH_STALE      CauseCode = 18 // the echoed epoch is not current
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
		18: "CAUSE_CODE_ADMISSION_EPOCH_STALE",
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
		"CAUSE_CODE_ADMISSION_EPOCH_STALE":      18,
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
	ClaimRejection_CLAIM_REJECTION_UNDURABLE             ClaimRejection = 7
	ClaimRejection_CLAIM_REJECTION_PROTOCOL_INCOMPATIBLE ClaimRejection = 9 // below compatibility floor; no counter changes
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
		9: "CLAIM_REJECTION_PROTOCOL_INCOMPATIBLE",
	}
	ClaimRejection_value = map[string]int32{
		"CLAIM_REJECTION_UNSPECIFIED":              0,
		"CLAIM_REJECTION_UNAUTHENTICATED":          1,
		"CLAIM_REJECTION_STALE_RECORD_OWNER_EPOCH": 2,
		"CLAIM_REJECTION_EPOCH_HELD":               3,
		"CLAIM_REJECTION_WORKER_ID_MISMATCH":       4,
		"CLAIM_REJECTION_RELEASE_ID_MISMATCH":      5,
		"CLAIM_REJECTION_UNDURABLE":                7,
		"CLAIM_REJECTION_PROTOCOL_INCOMPATIBLE":    9,
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

type PrepareStage int32

const (
	PrepareStage_PREPARE_STAGE_UNSPECIFIED PrepareStage = 0
	PrepareStage_PREPARE_STAGE_RESOLVED    PrepareStage = 1 // authority verified, plan bounded, intent journaled
	PrepareStage_PREPARE_STAGE_DOWNLOADING PrepareStage = 2
	PrepareStage_PREPARE_STAGE_PREPARING   PrepareStage = 3 // verified local files handed to Runtime preparation
	PrepareStage_PREPARE_STAGE_PREPARED    PrepareStage = 4 // terminal; placement_set present
	PrepareStage_PREPARE_STAGE_REFUSED     PrepareStage = 5 // terminal; safe_code present
)

// Enum value maps for PrepareStage.
var (
	PrepareStage_name = map[int32]string{
		0: "PREPARE_STAGE_UNSPECIFIED",
		1: "PREPARE_STAGE_RESOLVED",
		2: "PREPARE_STAGE_DOWNLOADING",
		3: "PREPARE_STAGE_PREPARING",
		4: "PREPARE_STAGE_PREPARED",
		5: "PREPARE_STAGE_REFUSED",
	}
	PrepareStage_value = map[string]int32{
		"PREPARE_STAGE_UNSPECIFIED": 0,
		"PREPARE_STAGE_RESOLVED":    1,
		"PREPARE_STAGE_DOWNLOADING": 2,
		"PREPARE_STAGE_PREPARING":   3,
		"PREPARE_STAGE_PREPARED":    4,
		"PREPARE_STAGE_REFUSED":     5,
	}
)

func (x PrepareStage) Enum() *PrepareStage {
	p := new(PrepareStage)
	*p = x
	return p
}

func (x PrepareStage) String() string {
	return protoimpl.X.EnumStringOf(x.Descriptor(), protoreflect.EnumNumber(x))
}

func (PrepareStage) Descriptor() protoreflect.EnumDescriptor {
	return file_cozy_worker_v1_worker_proto_enumTypes[17].Descriptor()
}

func (PrepareStage) Type() protoreflect.EnumType {
	return &file_cozy_worker_v1_worker_proto_enumTypes[17]
}

func (x PrepareStage) Number() protoreflect.EnumNumber {
	return protoreflect.EnumNumber(x)
}

// Deprecated: Use PrepareStage.Descriptor instead.
func (PrepareStage) EnumDescriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{17}
}

type WeightsHostStage int32

const (
	WeightsHostStage_WEIGHTS_HOST_STAGE_UNSPECIFIED WeightsHostStage = 0
	WeightsHostStage_WEIGHTS_HOST_STAGE_INTENT      WeightsHostStage = 1
	WeightsHostStage_WEIGHTS_HOST_STAGE_RECEIPT     WeightsHostStage = 2
	WeightsHostStage_WEIGHTS_HOST_STAGE_CHECKPOINT  WeightsHostStage = 3
)

// Enum value maps for WeightsHostStage.
var (
	WeightsHostStage_name = map[int32]string{
		0: "WEIGHTS_HOST_STAGE_UNSPECIFIED",
		1: "WEIGHTS_HOST_STAGE_INTENT",
		2: "WEIGHTS_HOST_STAGE_RECEIPT",
		3: "WEIGHTS_HOST_STAGE_CHECKPOINT",
	}
	WeightsHostStage_value = map[string]int32{
		"WEIGHTS_HOST_STAGE_UNSPECIFIED": 0,
		"WEIGHTS_HOST_STAGE_INTENT":      1,
		"WEIGHTS_HOST_STAGE_RECEIPT":     2,
		"WEIGHTS_HOST_STAGE_CHECKPOINT":  3,
	}
)

func (x WeightsHostStage) Enum() *WeightsHostStage {
	p := new(WeightsHostStage)
	*p = x
	return p
}

func (x WeightsHostStage) String() string {
	return protoimpl.X.EnumStringOf(x.Descriptor(), protoreflect.EnumNumber(x))
}

func (WeightsHostStage) Descriptor() protoreflect.EnumDescriptor {
	return file_cozy_worker_v1_worker_proto_enumTypes[18].Descriptor()
}

func (WeightsHostStage) Type() protoreflect.EnumType {
	return &file_cozy_worker_v1_worker_proto_enumTypes[18]
}

func (x WeightsHostStage) Number() protoreflect.EnumNumber {
	return protoreflect.EnumNumber(x)
}

// Deprecated: Use WeightsHostStage.Descriptor instead.
func (WeightsHostStage) EnumDescriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{18}
}

type WeightsHostOutcome int32

const (
	WeightsHostOutcome_WEIGHTS_HOST_OUTCOME_UNSPECIFIED WeightsHostOutcome = 0
	WeightsHostOutcome_WEIGHTS_HOST_OUTCOME_RECORDED    WeightsHostOutcome = 1
	WeightsHostOutcome_WEIGHTS_HOST_OUTCOME_REPLAYED    WeightsHostOutcome = 2
	WeightsHostOutcome_WEIGHTS_HOST_OUTCOME_REFUSED     WeightsHostOutcome = 3
)

// Enum value maps for WeightsHostOutcome.
var (
	WeightsHostOutcome_name = map[int32]string{
		0: "WEIGHTS_HOST_OUTCOME_UNSPECIFIED",
		1: "WEIGHTS_HOST_OUTCOME_RECORDED",
		2: "WEIGHTS_HOST_OUTCOME_REPLAYED",
		3: "WEIGHTS_HOST_OUTCOME_REFUSED",
	}
	WeightsHostOutcome_value = map[string]int32{
		"WEIGHTS_HOST_OUTCOME_UNSPECIFIED": 0,
		"WEIGHTS_HOST_OUTCOME_RECORDED":    1,
		"WEIGHTS_HOST_OUTCOME_REPLAYED":    2,
		"WEIGHTS_HOST_OUTCOME_REFUSED":     3,
	}
)

func (x WeightsHostOutcome) Enum() *WeightsHostOutcome {
	p := new(WeightsHostOutcome)
	*p = x
	return p
}

func (x WeightsHostOutcome) String() string {
	return protoimpl.X.EnumStringOf(x.Descriptor(), protoreflect.EnumNumber(x))
}

func (WeightsHostOutcome) Descriptor() protoreflect.EnumDescriptor {
	return file_cozy_worker_v1_worker_proto_enumTypes[19].Descriptor()
}

func (WeightsHostOutcome) Type() protoreflect.EnumType {
	return &file_cozy_worker_v1_worker_proto_enumTypes[19]
}

func (x WeightsHostOutcome) Number() protoreflect.EnumNumber {
	return protoreflect.EnumNumber(x)
}

// Deprecated: Use WeightsHostOutcome.Descriptor instead.
func (WeightsHostOutcome) EnumDescriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{19}
}

type WeightsHostRefusal int32

const (
	WeightsHostRefusal_WEIGHTS_HOST_REFUSAL_UNSPECIFIED       WeightsHostRefusal = 0
	WeightsHostRefusal_WEIGHTS_HOST_REFUSAL_UNKNOWN_ATTEMPT   WeightsHostRefusal = 1
	WeightsHostRefusal_WEIGHTS_HOST_REFUSAL_INTENT_CONFLICT   WeightsHostRefusal = 2
	WeightsHostRefusal_WEIGHTS_HOST_REFUSAL_STALE_WRITER      WeightsHostRefusal = 3
	WeightsHostRefusal_WEIGHTS_HOST_REFUSAL_RECEIPT_CONFLICT  WeightsHostRefusal = 4
	WeightsHostRefusal_WEIGHTS_HOST_REFUSAL_INVENTORY_INVALID WeightsHostRefusal = 5
)

// Enum value maps for WeightsHostRefusal.
var (
	WeightsHostRefusal_name = map[int32]string{
		0: "WEIGHTS_HOST_REFUSAL_UNSPECIFIED",
		1: "WEIGHTS_HOST_REFUSAL_UNKNOWN_ATTEMPT",
		2: "WEIGHTS_HOST_REFUSAL_INTENT_CONFLICT",
		3: "WEIGHTS_HOST_REFUSAL_STALE_WRITER",
		4: "WEIGHTS_HOST_REFUSAL_RECEIPT_CONFLICT",
		5: "WEIGHTS_HOST_REFUSAL_INVENTORY_INVALID",
	}
	WeightsHostRefusal_value = map[string]int32{
		"WEIGHTS_HOST_REFUSAL_UNSPECIFIED":       0,
		"WEIGHTS_HOST_REFUSAL_UNKNOWN_ATTEMPT":   1,
		"WEIGHTS_HOST_REFUSAL_INTENT_CONFLICT":   2,
		"WEIGHTS_HOST_REFUSAL_STALE_WRITER":      3,
		"WEIGHTS_HOST_REFUSAL_RECEIPT_CONFLICT":  4,
		"WEIGHTS_HOST_REFUSAL_INVENTORY_INVALID": 5,
	}
)

func (x WeightsHostRefusal) Enum() *WeightsHostRefusal {
	p := new(WeightsHostRefusal)
	*p = x
	return p
}

func (x WeightsHostRefusal) String() string {
	return protoimpl.X.EnumStringOf(x.Descriptor(), protoreflect.EnumNumber(x))
}

func (WeightsHostRefusal) Descriptor() protoreflect.EnumDescriptor {
	return file_cozy_worker_v1_worker_proto_enumTypes[20].Descriptor()
}

func (WeightsHostRefusal) Type() protoreflect.EnumType {
	return &file_cozy_worker_v1_worker_proto_enumTypes[20]
}

func (x WeightsHostRefusal) Number() protoreflect.EnumNumber {
	return protoreflect.EnumNumber(x)
}

// Deprecated: Use WeightsHostRefusal.Descriptor instead.
func (WeightsHostRefusal) EnumDescriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{20}
}

type WeightsTransactionState int32

const (
	WeightsTransactionState_WEIGHTS_TRANSACTION_STATE_UNSPECIFIED WeightsTransactionState = 0
	WeightsTransactionState_WEIGHTS_TRANSACTION_STATE_INTENT      WeightsTransactionState = 1
	WeightsTransactionState_WEIGHTS_TRANSACTION_STATE_RECEIPT     WeightsTransactionState = 2
)

// Enum value maps for WeightsTransactionState.
var (
	WeightsTransactionState_name = map[int32]string{
		0: "WEIGHTS_TRANSACTION_STATE_UNSPECIFIED",
		1: "WEIGHTS_TRANSACTION_STATE_INTENT",
		2: "WEIGHTS_TRANSACTION_STATE_RECEIPT",
	}
	WeightsTransactionState_value = map[string]int32{
		"WEIGHTS_TRANSACTION_STATE_UNSPECIFIED": 0,
		"WEIGHTS_TRANSACTION_STATE_INTENT":      1,
		"WEIGHTS_TRANSACTION_STATE_RECEIPT":     2,
	}
)

func (x WeightsTransactionState) Enum() *WeightsTransactionState {
	p := new(WeightsTransactionState)
	*p = x
	return p
}

func (x WeightsTransactionState) String() string {
	return protoimpl.X.EnumStringOf(x.Descriptor(), protoreflect.EnumNumber(x))
}

func (WeightsTransactionState) Descriptor() protoreflect.EnumDescriptor {
	return file_cozy_worker_v1_worker_proto_enumTypes[21].Descriptor()
}

func (WeightsTransactionState) Type() protoreflect.EnumType {
	return &file_cozy_worker_v1_worker_proto_enumTypes[21]
}

func (x WeightsTransactionState) Number() protoreflect.EnumNumber {
	return protoreflect.EnumNumber(x)
}

// Deprecated: Use WeightsTransactionState.Descriptor instead.
func (WeightsTransactionState) EnumDescriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{21}
}

type WeightsReadOutcome int32

const (
	WeightsReadOutcome_WEIGHTS_READ_OUTCOME_UNSPECIFIED WeightsReadOutcome = 0
	WeightsReadOutcome_WEIGHTS_READ_OUTCOME_DATA        WeightsReadOutcome = 1
	WeightsReadOutcome_WEIGHTS_READ_OUTCOME_REFUSED     WeightsReadOutcome = 2
)

// Enum value maps for WeightsReadOutcome.
var (
	WeightsReadOutcome_name = map[int32]string{
		0: "WEIGHTS_READ_OUTCOME_UNSPECIFIED",
		1: "WEIGHTS_READ_OUTCOME_DATA",
		2: "WEIGHTS_READ_OUTCOME_REFUSED",
	}
	WeightsReadOutcome_value = map[string]int32{
		"WEIGHTS_READ_OUTCOME_UNSPECIFIED": 0,
		"WEIGHTS_READ_OUTCOME_DATA":        1,
		"WEIGHTS_READ_OUTCOME_REFUSED":     2,
	}
)

func (x WeightsReadOutcome) Enum() *WeightsReadOutcome {
	p := new(WeightsReadOutcome)
	*p = x
	return p
}

func (x WeightsReadOutcome) String() string {
	return protoimpl.X.EnumStringOf(x.Descriptor(), protoreflect.EnumNumber(x))
}

func (WeightsReadOutcome) Descriptor() protoreflect.EnumDescriptor {
	return file_cozy_worker_v1_worker_proto_enumTypes[22].Descriptor()
}

func (WeightsReadOutcome) Type() protoreflect.EnumType {
	return &file_cozy_worker_v1_worker_proto_enumTypes[22]
}

func (x WeightsReadOutcome) Number() protoreflect.EnumNumber {
	return protoreflect.EnumNumber(x)
}

// Deprecated: Use WeightsReadOutcome.Descriptor instead.
func (WeightsReadOutcome) EnumDescriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{22}
}

type WeightsReadRefusal int32

const (
	WeightsReadRefusal_WEIGHTS_READ_REFUSAL_UNSPECIFIED         WeightsReadRefusal = 0
	WeightsReadRefusal_WEIGHTS_READ_REFUSAL_UNKNOWN_TRANSACTION WeightsReadRefusal = 1
	WeightsReadRefusal_WEIGHTS_READ_REFUSAL_STALE_WRITER        WeightsReadRefusal = 2
	WeightsReadRefusal_WEIGHTS_READ_REFUSAL_UNKNOWN_OBJECT      WeightsReadRefusal = 3
	WeightsReadRefusal_WEIGHTS_READ_REFUSAL_SOURCE_REF_MISMATCH WeightsReadRefusal = 4
	WeightsReadRefusal_WEIGHTS_READ_REFUSAL_RANGE_INVALID       WeightsReadRefusal = 5
	WeightsReadRefusal_WEIGHTS_READ_REFUSAL_SOURCE_UNAVAILABLE  WeightsReadRefusal = 6
)

// Enum value maps for WeightsReadRefusal.
var (
	WeightsReadRefusal_name = map[int32]string{
		0: "WEIGHTS_READ_REFUSAL_UNSPECIFIED",
		1: "WEIGHTS_READ_REFUSAL_UNKNOWN_TRANSACTION",
		2: "WEIGHTS_READ_REFUSAL_STALE_WRITER",
		3: "WEIGHTS_READ_REFUSAL_UNKNOWN_OBJECT",
		4: "WEIGHTS_READ_REFUSAL_SOURCE_REF_MISMATCH",
		5: "WEIGHTS_READ_REFUSAL_RANGE_INVALID",
		6: "WEIGHTS_READ_REFUSAL_SOURCE_UNAVAILABLE",
	}
	WeightsReadRefusal_value = map[string]int32{
		"WEIGHTS_READ_REFUSAL_UNSPECIFIED":         0,
		"WEIGHTS_READ_REFUSAL_UNKNOWN_TRANSACTION": 1,
		"WEIGHTS_READ_REFUSAL_STALE_WRITER":        2,
		"WEIGHTS_READ_REFUSAL_UNKNOWN_OBJECT":      3,
		"WEIGHTS_READ_REFUSAL_SOURCE_REF_MISMATCH": 4,
		"WEIGHTS_READ_REFUSAL_RANGE_INVALID":       5,
		"WEIGHTS_READ_REFUSAL_SOURCE_UNAVAILABLE":  6,
	}
)

func (x WeightsReadRefusal) Enum() *WeightsReadRefusal {
	p := new(WeightsReadRefusal)
	*p = x
	return p
}

func (x WeightsReadRefusal) String() string {
	return protoimpl.X.EnumStringOf(x.Descriptor(), protoreflect.EnumNumber(x))
}

func (WeightsReadRefusal) Descriptor() protoreflect.EnumDescriptor {
	return file_cozy_worker_v1_worker_proto_enumTypes[23].Descriptor()
}

func (WeightsReadRefusal) Type() protoreflect.EnumType {
	return &file_cozy_worker_v1_worker_proto_enumTypes[23]
}

func (x WeightsReadRefusal) Number() protoreflect.EnumNumber {
	return protoreflect.EnumNumber(x)
}

// Deprecated: Use WeightsReadRefusal.Descriptor instead.
func (WeightsReadRefusal) EnumDescriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{23}
}

type WeightsUploadOutcome int32

const (
	WeightsUploadOutcome_WEIGHTS_UPLOAD_OUTCOME_UNSPECIFIED     WeightsUploadOutcome = 0
	WeightsUploadOutcome_WEIGHTS_UPLOAD_OUTCOME_UPLOADED        WeightsUploadOutcome = 1
	WeightsUploadOutcome_WEIGHTS_UPLOAD_OUTCOME_ALREADY_PRESENT WeightsUploadOutcome = 2
	WeightsUploadOutcome_WEIGHTS_UPLOAD_OUTCOME_REFUSED         WeightsUploadOutcome = 3
)

// Enum value maps for WeightsUploadOutcome.
var (
	WeightsUploadOutcome_name = map[int32]string{
		0: "WEIGHTS_UPLOAD_OUTCOME_UNSPECIFIED",
		1: "WEIGHTS_UPLOAD_OUTCOME_UPLOADED",
		2: "WEIGHTS_UPLOAD_OUTCOME_ALREADY_PRESENT",
		3: "WEIGHTS_UPLOAD_OUTCOME_REFUSED",
	}
	WeightsUploadOutcome_value = map[string]int32{
		"WEIGHTS_UPLOAD_OUTCOME_UNSPECIFIED":     0,
		"WEIGHTS_UPLOAD_OUTCOME_UPLOADED":        1,
		"WEIGHTS_UPLOAD_OUTCOME_ALREADY_PRESENT": 2,
		"WEIGHTS_UPLOAD_OUTCOME_REFUSED":         3,
	}
)

func (x WeightsUploadOutcome) Enum() *WeightsUploadOutcome {
	p := new(WeightsUploadOutcome)
	*p = x
	return p
}

func (x WeightsUploadOutcome) String() string {
	return protoimpl.X.EnumStringOf(x.Descriptor(), protoreflect.EnumNumber(x))
}

func (WeightsUploadOutcome) Descriptor() protoreflect.EnumDescriptor {
	return file_cozy_worker_v1_worker_proto_enumTypes[24].Descriptor()
}

func (WeightsUploadOutcome) Type() protoreflect.EnumType {
	return &file_cozy_worker_v1_worker_proto_enumTypes[24]
}

func (x WeightsUploadOutcome) Number() protoreflect.EnumNumber {
	return protoreflect.EnumNumber(x)
}

// Deprecated: Use WeightsUploadOutcome.Descriptor instead.
func (WeightsUploadOutcome) EnumDescriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{24}
}

type WeightsUploadRefusal int32

const (
	WeightsUploadRefusal_WEIGHTS_UPLOAD_REFUSAL_UNSPECIFIED         WeightsUploadRefusal = 0
	WeightsUploadRefusal_WEIGHTS_UPLOAD_REFUSAL_UNKNOWN_TRANSACTION WeightsUploadRefusal = 1
	WeightsUploadRefusal_WEIGHTS_UPLOAD_REFUSAL_STALE_WRITER        WeightsUploadRefusal = 2
	WeightsUploadRefusal_WEIGHTS_UPLOAD_REFUSAL_UNKNOWN_OBJECT      WeightsUploadRefusal = 3
	WeightsUploadRefusal_WEIGHTS_UPLOAD_REFUSAL_SOURCE_REF_MISMATCH WeightsUploadRefusal = 4
	WeightsUploadRefusal_WEIGHTS_UPLOAD_REFUSAL_SOURCE_UNAVAILABLE  WeightsUploadRefusal = 5
	WeightsUploadRefusal_WEIGHTS_UPLOAD_REFUSAL_GRANT_REFUSED       WeightsUploadRefusal = 6
	WeightsUploadRefusal_WEIGHTS_UPLOAD_REFUSAL_TRANSFER_FAILED     WeightsUploadRefusal = 7
)

// Enum value maps for WeightsUploadRefusal.
var (
	WeightsUploadRefusal_name = map[int32]string{
		0: "WEIGHTS_UPLOAD_REFUSAL_UNSPECIFIED",
		1: "WEIGHTS_UPLOAD_REFUSAL_UNKNOWN_TRANSACTION",
		2: "WEIGHTS_UPLOAD_REFUSAL_STALE_WRITER",
		3: "WEIGHTS_UPLOAD_REFUSAL_UNKNOWN_OBJECT",
		4: "WEIGHTS_UPLOAD_REFUSAL_SOURCE_REF_MISMATCH",
		5: "WEIGHTS_UPLOAD_REFUSAL_SOURCE_UNAVAILABLE",
		6: "WEIGHTS_UPLOAD_REFUSAL_GRANT_REFUSED",
		7: "WEIGHTS_UPLOAD_REFUSAL_TRANSFER_FAILED",
	}
	WeightsUploadRefusal_value = map[string]int32{
		"WEIGHTS_UPLOAD_REFUSAL_UNSPECIFIED":         0,
		"WEIGHTS_UPLOAD_REFUSAL_UNKNOWN_TRANSACTION": 1,
		"WEIGHTS_UPLOAD_REFUSAL_STALE_WRITER":        2,
		"WEIGHTS_UPLOAD_REFUSAL_UNKNOWN_OBJECT":      3,
		"WEIGHTS_UPLOAD_REFUSAL_SOURCE_REF_MISMATCH": 4,
		"WEIGHTS_UPLOAD_REFUSAL_SOURCE_UNAVAILABLE":  5,
		"WEIGHTS_UPLOAD_REFUSAL_GRANT_REFUSED":       6,
		"WEIGHTS_UPLOAD_REFUSAL_TRANSFER_FAILED":     7,
	}
)

func (x WeightsUploadRefusal) Enum() *WeightsUploadRefusal {
	p := new(WeightsUploadRefusal)
	*p = x
	return p
}

func (x WeightsUploadRefusal) String() string {
	return protoimpl.X.EnumStringOf(x.Descriptor(), protoreflect.EnumNumber(x))
}

func (WeightsUploadRefusal) Descriptor() protoreflect.EnumDescriptor {
	return file_cozy_worker_v1_worker_proto_enumTypes[25].Descriptor()
}

func (WeightsUploadRefusal) Type() protoreflect.EnumType {
	return &file_cozy_worker_v1_worker_proto_enumTypes[25]
}

func (x WeightsUploadRefusal) Number() protoreflect.EnumNumber {
	return protoreflect.EnumNumber(x)
}

// Deprecated: Use WeightsUploadRefusal.Descriptor instead.
func (WeightsUploadRefusal) EnumDescriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{25}
}

type WeightsTransferState int32

const (
	WeightsTransferState_WEIGHTS_TRANSFER_STATE_UNSPECIFIED     WeightsTransferState = 0
	WeightsTransferState_WEIGHTS_TRANSFER_STATE_ACCEPTED        WeightsTransferState = 1
	WeightsTransferState_WEIGHTS_TRANSFER_STATE_UPLOADING       WeightsTransferState = 2
	WeightsTransferState_WEIGHTS_TRANSFER_STATE_UPLOADED        WeightsTransferState = 3
	WeightsTransferState_WEIGHTS_TRANSFER_STATE_ALREADY_PRESENT WeightsTransferState = 4
	WeightsTransferState_WEIGHTS_TRANSFER_STATE_HELD            WeightsTransferState = 5
	WeightsTransferState_WEIGHTS_TRANSFER_STATE_FAILED          WeightsTransferState = 6
)

// Enum value maps for WeightsTransferState.
var (
	WeightsTransferState_name = map[int32]string{
		0: "WEIGHTS_TRANSFER_STATE_UNSPECIFIED",
		1: "WEIGHTS_TRANSFER_STATE_ACCEPTED",
		2: "WEIGHTS_TRANSFER_STATE_UPLOADING",
		3: "WEIGHTS_TRANSFER_STATE_UPLOADED",
		4: "WEIGHTS_TRANSFER_STATE_ALREADY_PRESENT",
		5: "WEIGHTS_TRANSFER_STATE_HELD",
		6: "WEIGHTS_TRANSFER_STATE_FAILED",
	}
	WeightsTransferState_value = map[string]int32{
		"WEIGHTS_TRANSFER_STATE_UNSPECIFIED":     0,
		"WEIGHTS_TRANSFER_STATE_ACCEPTED":        1,
		"WEIGHTS_TRANSFER_STATE_UPLOADING":       2,
		"WEIGHTS_TRANSFER_STATE_UPLOADED":        3,
		"WEIGHTS_TRANSFER_STATE_ALREADY_PRESENT": 4,
		"WEIGHTS_TRANSFER_STATE_HELD":            5,
		"WEIGHTS_TRANSFER_STATE_FAILED":          6,
	}
)

func (x WeightsTransferState) Enum() *WeightsTransferState {
	p := new(WeightsTransferState)
	*p = x
	return p
}

func (x WeightsTransferState) String() string {
	return protoimpl.X.EnumStringOf(x.Descriptor(), protoreflect.EnumNumber(x))
}

func (WeightsTransferState) Descriptor() protoreflect.EnumDescriptor {
	return file_cozy_worker_v1_worker_proto_enumTypes[26].Descriptor()
}

func (WeightsTransferState) Type() protoreflect.EnumType {
	return &file_cozy_worker_v1_worker_proto_enumTypes[26]
}

func (x WeightsTransferState) Number() protoreflect.EnumNumber {
	return protoreflect.EnumNumber(x)
}

// Deprecated: Use WeightsTransferState.Descriptor instead.
func (WeightsTransferState) EnumDescriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{26}
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
	return file_cozy_worker_v1_worker_proto_enumTypes[27].Descriptor()
}

func (ModelSourceProvider) Type() protoreflect.EnumType {
	return &file_cozy_worker_v1_worker_proto_enumTypes[27]
}

func (x ModelSourceProvider) Number() protoreflect.EnumNumber {
	return protoreflect.EnumNumber(x)
}

// Deprecated: Use ModelSourceProvider.Descriptor instead.
func (ModelSourceProvider) EnumDescriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{27}
}

type ModelSourceFileState int32

const (
	ModelSourceFileState_MODEL_SOURCE_FILE_STATE_UNSPECIFIED ModelSourceFileState = 0
	ModelSourceFileState_MODEL_SOURCE_FILE_STATE_ACCEPTED    ModelSourceFileState = 1
	ModelSourceFileState_MODEL_SOURCE_FILE_STATE_DOWNLOADING ModelSourceFileState = 2
	ModelSourceFileState_MODEL_SOURCE_FILE_STATE_VERIFIED    ModelSourceFileState = 3
	ModelSourceFileState_MODEL_SOURCE_FILE_STATE_FAILED      ModelSourceFileState = 4
	// All conversion consumers are complete/recoverable. The physical carrier may have
	// been released, so this MUST NOT be used as LocalModelSourceFile.verified.
	ModelSourceFileState_MODEL_SOURCE_FILE_STATE_CONVERTED ModelSourceFileState = 5
)

// Enum value maps for ModelSourceFileState.
var (
	ModelSourceFileState_name = map[int32]string{
		0: "MODEL_SOURCE_FILE_STATE_UNSPECIFIED",
		1: "MODEL_SOURCE_FILE_STATE_ACCEPTED",
		2: "MODEL_SOURCE_FILE_STATE_DOWNLOADING",
		3: "MODEL_SOURCE_FILE_STATE_VERIFIED",
		4: "MODEL_SOURCE_FILE_STATE_FAILED",
		5: "MODEL_SOURCE_FILE_STATE_CONVERTED",
	}
	ModelSourceFileState_value = map[string]int32{
		"MODEL_SOURCE_FILE_STATE_UNSPECIFIED": 0,
		"MODEL_SOURCE_FILE_STATE_ACCEPTED":    1,
		"MODEL_SOURCE_FILE_STATE_DOWNLOADING": 2,
		"MODEL_SOURCE_FILE_STATE_VERIFIED":    3,
		"MODEL_SOURCE_FILE_STATE_FAILED":      4,
		"MODEL_SOURCE_FILE_STATE_CONVERTED":   5,
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
	return file_cozy_worker_v1_worker_proto_enumTypes[28].Descriptor()
}

func (ModelSourceFileState) Type() protoreflect.EnumType {
	return &file_cozy_worker_v1_worker_proto_enumTypes[28]
}

func (x ModelSourceFileState) Number() protoreflect.EnumNumber {
	return protoreflect.EnumNumber(x)
}

// Deprecated: Use ModelSourceFileState.Descriptor instead.
func (ModelSourceFileState) EnumDescriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{28}
}

type ModelSourcePrepareOutcome int32

const (
	ModelSourcePrepareOutcome_MODEL_SOURCE_PREPARE_OUTCOME_UNSPECIFIED ModelSourcePrepareOutcome = 0
	ModelSourcePrepareOutcome_MODEL_SOURCE_PREPARE_OUTCOME_PREPARED    ModelSourcePrepareOutcome = 1
	ModelSourcePrepareOutcome_MODEL_SOURCE_PREPARE_OUTCOME_REPLAYED    ModelSourcePrepareOutcome = 2
	ModelSourcePrepareOutcome_MODEL_SOURCE_PREPARE_OUTCOME_REFUSED     ModelSourcePrepareOutcome = 3
	ModelSourcePrepareOutcome_MODEL_SOURCE_PREPARE_OUTCOME_INCOMPLETE  ModelSourcePrepareOutcome = 4
)

// Enum value maps for ModelSourcePrepareOutcome.
var (
	ModelSourcePrepareOutcome_name = map[int32]string{
		0: "MODEL_SOURCE_PREPARE_OUTCOME_UNSPECIFIED",
		1: "MODEL_SOURCE_PREPARE_OUTCOME_PREPARED",
		2: "MODEL_SOURCE_PREPARE_OUTCOME_REPLAYED",
		3: "MODEL_SOURCE_PREPARE_OUTCOME_REFUSED",
		4: "MODEL_SOURCE_PREPARE_OUTCOME_INCOMPLETE",
	}
	ModelSourcePrepareOutcome_value = map[string]int32{
		"MODEL_SOURCE_PREPARE_OUTCOME_UNSPECIFIED": 0,
		"MODEL_SOURCE_PREPARE_OUTCOME_PREPARED":    1,
		"MODEL_SOURCE_PREPARE_OUTCOME_REPLAYED":    2,
		"MODEL_SOURCE_PREPARE_OUTCOME_REFUSED":     3,
		"MODEL_SOURCE_PREPARE_OUTCOME_INCOMPLETE":  4,
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
	return file_cozy_worker_v1_worker_proto_enumTypes[29].Descriptor()
}

func (ModelSourcePrepareOutcome) Type() protoreflect.EnumType {
	return &file_cozy_worker_v1_worker_proto_enumTypes[29]
}

func (x ModelSourcePrepareOutcome) Number() protoreflect.EnumNumber {
	return protoreflect.EnumNumber(x)
}

// Deprecated: Use ModelSourcePrepareOutcome.Descriptor instead.
func (ModelSourcePrepareOutcome) EnumDescriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{29}
}

type LocalPackageFileState int32

const (
	LocalPackageFileState_LOCAL_PACKAGE_FILE_STATE_UNSPECIFIED LocalPackageFileState = 0
	LocalPackageFileState_LOCAL_PACKAGE_FILE_STATE_RECEIVING   LocalPackageFileState = 1
	LocalPackageFileState_LOCAL_PACKAGE_FILE_STATE_VERIFIED    LocalPackageFileState = 2
	LocalPackageFileState_LOCAL_PACKAGE_FILE_STATE_REFUSED     LocalPackageFileState = 3
)

// Enum value maps for LocalPackageFileState.
var (
	LocalPackageFileState_name = map[int32]string{
		0: "LOCAL_PACKAGE_FILE_STATE_UNSPECIFIED",
		1: "LOCAL_PACKAGE_FILE_STATE_RECEIVING",
		2: "LOCAL_PACKAGE_FILE_STATE_VERIFIED",
		3: "LOCAL_PACKAGE_FILE_STATE_REFUSED",
	}
	LocalPackageFileState_value = map[string]int32{
		"LOCAL_PACKAGE_FILE_STATE_UNSPECIFIED": 0,
		"LOCAL_PACKAGE_FILE_STATE_RECEIVING":   1,
		"LOCAL_PACKAGE_FILE_STATE_VERIFIED":    2,
		"LOCAL_PACKAGE_FILE_STATE_REFUSED":     3,
	}
)

func (x LocalPackageFileState) Enum() *LocalPackageFileState {
	p := new(LocalPackageFileState)
	*p = x
	return p
}

func (x LocalPackageFileState) String() string {
	return protoimpl.X.EnumStringOf(x.Descriptor(), protoreflect.EnumNumber(x))
}

func (LocalPackageFileState) Descriptor() protoreflect.EnumDescriptor {
	return file_cozy_worker_v1_worker_proto_enumTypes[30].Descriptor()
}

func (LocalPackageFileState) Type() protoreflect.EnumType {
	return &file_cozy_worker_v1_worker_proto_enumTypes[30]
}

func (x LocalPackageFileState) Number() protoreflect.EnumNumber {
	return protoreflect.EnumNumber(x)
}

// Deprecated: Use LocalPackageFileState.Descriptor instead.
func (LocalPackageFileState) EnumDescriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{30}
}

type LocalPackageAbortOutcome int32

const (
	LocalPackageAbortOutcome_LOCAL_PACKAGE_ABORT_OUTCOME_UNSPECIFIED LocalPackageAbortOutcome = 0
	LocalPackageAbortOutcome_LOCAL_PACKAGE_ABORT_OUTCOME_ABANDONED   LocalPackageAbortOutcome = 1
	LocalPackageAbortOutcome_LOCAL_PACKAGE_ABORT_OUTCOME_REPLAYED    LocalPackageAbortOutcome = 2
	LocalPackageAbortOutcome_LOCAL_PACKAGE_ABORT_OUTCOME_REFUSED     LocalPackageAbortOutcome = 3
)

// Enum value maps for LocalPackageAbortOutcome.
var (
	LocalPackageAbortOutcome_name = map[int32]string{
		0: "LOCAL_PACKAGE_ABORT_OUTCOME_UNSPECIFIED",
		1: "LOCAL_PACKAGE_ABORT_OUTCOME_ABANDONED",
		2: "LOCAL_PACKAGE_ABORT_OUTCOME_REPLAYED",
		3: "LOCAL_PACKAGE_ABORT_OUTCOME_REFUSED",
	}
	LocalPackageAbortOutcome_value = map[string]int32{
		"LOCAL_PACKAGE_ABORT_OUTCOME_UNSPECIFIED": 0,
		"LOCAL_PACKAGE_ABORT_OUTCOME_ABANDONED":   1,
		"LOCAL_PACKAGE_ABORT_OUTCOME_REPLAYED":    2,
		"LOCAL_PACKAGE_ABORT_OUTCOME_REFUSED":     3,
	}
)

func (x LocalPackageAbortOutcome) Enum() *LocalPackageAbortOutcome {
	p := new(LocalPackageAbortOutcome)
	*p = x
	return p
}

func (x LocalPackageAbortOutcome) String() string {
	return protoimpl.X.EnumStringOf(x.Descriptor(), protoreflect.EnumNumber(x))
}

func (LocalPackageAbortOutcome) Descriptor() protoreflect.EnumDescriptor {
	return file_cozy_worker_v1_worker_proto_enumTypes[31].Descriptor()
}

func (LocalPackageAbortOutcome) Type() protoreflect.EnumType {
	return &file_cozy_worker_v1_worker_proto_enumTypes[31]
}

func (x LocalPackageAbortOutcome) Number() protoreflect.EnumNumber {
	return protoreflect.EnumNumber(x)
}

// Deprecated: Use LocalPackageAbortOutcome.Descriptor instead.
func (LocalPackageAbortOutcome) EnumDescriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{31}
}

type WeightsFinalizeDisposition int32

const (
	WeightsFinalizeDisposition_WEIGHTS_FINALIZE_DISPOSITION_UNSPECIFIED         WeightsFinalizeDisposition = 0
	WeightsFinalizeDisposition_WEIGHTS_FINALIZE_DISPOSITION_ADOPT               WeightsFinalizeDisposition = 1
	WeightsFinalizeDisposition_WEIGHTS_FINALIZE_DISPOSITION_ABANDON             WeightsFinalizeDisposition = 2
	WeightsFinalizeDisposition_WEIGHTS_FINALIZE_DISPOSITION_ABANDON_UNCOMMITTED WeightsFinalizeDisposition = 3
)

// Enum value maps for WeightsFinalizeDisposition.
var (
	WeightsFinalizeDisposition_name = map[int32]string{
		0: "WEIGHTS_FINALIZE_DISPOSITION_UNSPECIFIED",
		1: "WEIGHTS_FINALIZE_DISPOSITION_ADOPT",
		2: "WEIGHTS_FINALIZE_DISPOSITION_ABANDON",
		3: "WEIGHTS_FINALIZE_DISPOSITION_ABANDON_UNCOMMITTED",
	}
	WeightsFinalizeDisposition_value = map[string]int32{
		"WEIGHTS_FINALIZE_DISPOSITION_UNSPECIFIED":         0,
		"WEIGHTS_FINALIZE_DISPOSITION_ADOPT":               1,
		"WEIGHTS_FINALIZE_DISPOSITION_ABANDON":             2,
		"WEIGHTS_FINALIZE_DISPOSITION_ABANDON_UNCOMMITTED": 3,
	}
)

func (x WeightsFinalizeDisposition) Enum() *WeightsFinalizeDisposition {
	p := new(WeightsFinalizeDisposition)
	*p = x
	return p
}

func (x WeightsFinalizeDisposition) String() string {
	return protoimpl.X.EnumStringOf(x.Descriptor(), protoreflect.EnumNumber(x))
}

func (WeightsFinalizeDisposition) Descriptor() protoreflect.EnumDescriptor {
	return file_cozy_worker_v1_worker_proto_enumTypes[32].Descriptor()
}

func (WeightsFinalizeDisposition) Type() protoreflect.EnumType {
	return &file_cozy_worker_v1_worker_proto_enumTypes[32]
}

func (x WeightsFinalizeDisposition) Number() protoreflect.EnumNumber {
	return protoreflect.EnumNumber(x)
}

// Deprecated: Use WeightsFinalizeDisposition.Descriptor instead.
func (WeightsFinalizeDisposition) EnumDescriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{32}
}

type WeightsFinalizeOutcome int32

const (
	WeightsFinalizeOutcome_WEIGHTS_FINALIZE_OUTCOME_UNSPECIFIED WeightsFinalizeOutcome = 0
	WeightsFinalizeOutcome_WEIGHTS_FINALIZE_OUTCOME_ADOPTED     WeightsFinalizeOutcome = 1
	WeightsFinalizeOutcome_WEIGHTS_FINALIZE_OUTCOME_ABANDONED   WeightsFinalizeOutcome = 2
)

// Enum value maps for WeightsFinalizeOutcome.
var (
	WeightsFinalizeOutcome_name = map[int32]string{
		0: "WEIGHTS_FINALIZE_OUTCOME_UNSPECIFIED",
		1: "WEIGHTS_FINALIZE_OUTCOME_ADOPTED",
		2: "WEIGHTS_FINALIZE_OUTCOME_ABANDONED",
	}
	WeightsFinalizeOutcome_value = map[string]int32{
		"WEIGHTS_FINALIZE_OUTCOME_UNSPECIFIED": 0,
		"WEIGHTS_FINALIZE_OUTCOME_ADOPTED":     1,
		"WEIGHTS_FINALIZE_OUTCOME_ABANDONED":   2,
	}
)

func (x WeightsFinalizeOutcome) Enum() *WeightsFinalizeOutcome {
	p := new(WeightsFinalizeOutcome)
	*p = x
	return p
}

func (x WeightsFinalizeOutcome) String() string {
	return protoimpl.X.EnumStringOf(x.Descriptor(), protoreflect.EnumNumber(x))
}

func (WeightsFinalizeOutcome) Descriptor() protoreflect.EnumDescriptor {
	return file_cozy_worker_v1_worker_proto_enumTypes[33].Descriptor()
}

func (WeightsFinalizeOutcome) Type() protoreflect.EnumType {
	return &file_cozy_worker_v1_worker_proto_enumTypes[33]
}

func (x WeightsFinalizeOutcome) Number() protoreflect.EnumNumber {
	return protoreflect.EnumNumber(x)
}

// Deprecated: Use WeightsFinalizeOutcome.Descriptor instead.
func (WeightsFinalizeOutcome) EnumDescriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{33}
}

type WeightsHostEvent struct {
	state protoimpl.MessageState `protogen:"open.v1"`
	// Types that are valid to be assigned to Event:
	//
	//	*WeightsHostEvent_Intent
	//	*WeightsHostEvent_Receipt
	//	*WeightsHostEvent_Checkpoint
	Event         isWeightsHostEvent_Event `protobuf_oneof:"event"`
	unknownFields protoimpl.UnknownFields
	sizeCache     protoimpl.SizeCache
}

func (x *WeightsHostEvent) Reset() {
	*x = WeightsHostEvent{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[0]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *WeightsHostEvent) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*WeightsHostEvent) ProtoMessage() {}

func (x *WeightsHostEvent) ProtoReflect() protoreflect.Message {
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

// Deprecated: Use WeightsHostEvent.ProtoReflect.Descriptor instead.
func (*WeightsHostEvent) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{0}
}

func (x *WeightsHostEvent) GetEvent() isWeightsHostEvent_Event {
	if x != nil {
		return x.Event
	}
	return nil
}

func (x *WeightsHostEvent) GetIntent() *WeightsIntentFrame {
	if x != nil {
		if x, ok := x.Event.(*WeightsHostEvent_Intent); ok {
			return x.Intent
		}
	}
	return nil
}

func (x *WeightsHostEvent) GetReceipt() *WeightsReceiptFrame {
	if x != nil {
		if x, ok := x.Event.(*WeightsHostEvent_Receipt); ok {
			return x.Receipt
		}
	}
	return nil
}

func (x *WeightsHostEvent) GetCheckpoint() *WeightsCheckpointFrame {
	if x != nil {
		if x, ok := x.Event.(*WeightsHostEvent_Checkpoint); ok {
			return x.Checkpoint
		}
	}
	return nil
}

type isWeightsHostEvent_Event interface {
	isWeightsHostEvent_Event()
}

type WeightsHostEvent_Intent struct {
	Intent *WeightsIntentFrame `protobuf:"bytes,1,opt,name=intent,proto3,oneof"`
}

type WeightsHostEvent_Receipt struct {
	Receipt *WeightsReceiptFrame `protobuf:"bytes,2,opt,name=receipt,proto3,oneof"`
}

type WeightsHostEvent_Checkpoint struct {
	Checkpoint *WeightsCheckpointFrame `protobuf:"bytes,3,opt,name=checkpoint,proto3,oneof"`
}

func (*WeightsHostEvent_Intent) isWeightsHostEvent_Event() {}

func (*WeightsHostEvent_Receipt) isWeightsHostEvent_Event() {}

func (*WeightsHostEvent_Checkpoint) isWeightsHostEvent_Event() {}

type ProtocolInfoRequest struct {
	state         protoimpl.MessageState `protogen:"open.v1"`
	unknownFields protoimpl.UnknownFields
	sizeCache     protoimpl.SizeCache
}

func (x *ProtocolInfoRequest) Reset() {
	*x = ProtocolInfoRequest{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[1]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *ProtocolInfoRequest) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*ProtocolInfoRequest) ProtoMessage() {}

func (x *ProtocolInfoRequest) ProtoReflect() protoreflect.Message {
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

// Deprecated: Use ProtocolInfoRequest.ProtoReflect.Descriptor instead.
func (*ProtocolInfoRequest) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{1}
}

// No ownership, credential, readiness or machine state. A missing RPC or disjoint range
// refuses before any Claim/preparation side effect. Generated constants are the authority.
type ProtocolInfoResult struct {
	state            protoimpl.MessageState `protogen:"open.v1"`
	WireMinor        uint32                 `protobuf:"varint,1,opt,name=wire_minor,json=wireMinor,proto3" json:"wire_minor,omitempty"`
	MinimumWireMinor uint32                 `protobuf:"varint,2,opt,name=minimum_wire_minor,json=minimumWireMinor,proto3" json:"minimum_wire_minor,omitempty"`
	unknownFields    protoimpl.UnknownFields
	sizeCache        protoimpl.SizeCache
}

func (x *ProtocolInfoResult) Reset() {
	*x = ProtocolInfoResult{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[2]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *ProtocolInfoResult) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*ProtocolInfoResult) ProtoMessage() {}

func (x *ProtocolInfoResult) ProtoReflect() protoreflect.Message {
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

// Deprecated: Use ProtocolInfoResult.ProtoReflect.Descriptor instead.
func (*ProtocolInfoResult) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{2}
}

func (x *ProtocolInfoResult) GetWireMinor() uint32 {
	if x != nil {
		return x.WireMinor
	}
	return 0
}

func (x *ProtocolInfoResult) GetMinimumWireMinor() uint32 {
	if x != nil {
		return x.MinimumWireMinor
	}
	return 0
}

// The three preparations name the same logical sets the retiring DesiredWorkerState modes
// carried; the difference is who sends the result to the worker.
// MINOR 30 (xs-019): beside the signed set ride the facts prepare consumes, every one of them
// hub-known from the release record and the registered image before the pod exists. No package
// file, interface document, or model file travels; the pod installs from the indexes and
// resolves model manifests from its local TensorFS Store.
type PreparePackageSetCall struct {
	state          protoimpl.MessageState `protogen:"open.v1"`
	Claim          *Claim                 `protobuf:"bytes,1,opt,name=claim,proto3" json:"claim,omitempty"`
	PackageSet     *DesiredPackageSet     `protobuf:"bytes,2,opt,name=package_set,json=packageSet,proto3" json:"package_set,omitempty"`
	Application    string                 `protobuf:"bytes,3,opt,name=application,proto3" json:"application,omitempty"`                               // the release interface's application; nonempty
	ModelSlotPaths []string               `protobuf:"bytes,4,rep,name=model_slot_paths,json=modelSlotPaths,proto3" json:"model_slot_paths,omitempty"` // every interface-declared entrypoint model-slot path;
	// sorted unique; <= MaxModelSlotPaths
	ImageInventory     *ImageInventory `protobuf:"bytes,5,opt,name=image_inventory,json=imageInventory,proto3" json:"image_inventory,omitempty"`             // the placed image's exact pinned inventory
	LockedRequirements []byte          `protobuf:"bytes,6,opt,name=locked_requirements,json=lockedRequirements,proto3" json:"locked_requirements,omitempty"` // hash-pinned requirements export of the release's
	unknownFields      protoimpl.UnknownFields
	sizeCache          protoimpl.SizeCache
}

func (x *PreparePackageSetCall) Reset() {
	*x = PreparePackageSetCall{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[3]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *PreparePackageSetCall) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*PreparePackageSetCall) ProtoMessage() {}

func (x *PreparePackageSetCall) ProtoReflect() protoreflect.Message {
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

// Deprecated: Use PreparePackageSetCall.ProtoReflect.Descriptor instead.
func (*PreparePackageSetCall) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{3}
}

func (x *PreparePackageSetCall) GetClaim() *Claim {
	if x != nil {
		return x.Claim
	}
	return nil
}

func (x *PreparePackageSetCall) GetPackageSet() *DesiredPackageSet {
	if x != nil {
		return x.PackageSet
	}
	return nil
}

func (x *PreparePackageSetCall) GetApplication() string {
	if x != nil {
		return x.Application
	}
	return ""
}

func (x *PreparePackageSetCall) GetModelSlotPaths() []string {
	if x != nil {
		return x.ModelSlotPaths
	}
	return nil
}

func (x *PreparePackageSetCall) GetImageInventory() *ImageInventory {
	if x != nil {
		return x.ImageInventory
	}
	return nil
}

func (x *PreparePackageSetCall) GetLockedRequirements() []byte {
	if x != nil {
		return x.LockedRequirements
	}
	return nil
}

type PrepareLocalPackageCall struct {
	state           protoimpl.MessageState  `protogen:"open.v1"`
	Claim           *Claim                  `protobuf:"bytes,1,opt,name=claim,proto3" json:"claim,omitempty"`
	LocalPackageSet *DesiredLocalPackageSet `protobuf:"bytes,2,opt,name=local_package_set,json=localPackageSet,proto3" json:"local_package_set,omitempty"`
	unknownFields   protoimpl.UnknownFields
	sizeCache       protoimpl.SizeCache
}

func (x *PrepareLocalPackageCall) Reset() {
	*x = PrepareLocalPackageCall{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[4]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *PrepareLocalPackageCall) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*PrepareLocalPackageCall) ProtoMessage() {}

func (x *PrepareLocalPackageCall) ProtoReflect() protoreflect.Message {
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

// Deprecated: Use PrepareLocalPackageCall.ProtoReflect.Descriptor instead.
func (*PrepareLocalPackageCall) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{4}
}

func (x *PrepareLocalPackageCall) GetClaim() *Claim {
	if x != nil {
		return x.Claim
	}
	return nil
}

func (x *PrepareLocalPackageCall) GetLocalPackageSet() *DesiredLocalPackageSet {
	if x != nil {
		return x.LocalPackageSet
	}
	return nil
}

type PreparePrivatePlacementCall struct {
	state               protoimpl.MessageState      `protogen:"open.v1"`
	Claim               *Claim                      `protobuf:"bytes,1,opt,name=claim,proto3" json:"claim,omitempty"`
	PrivatePlacementSet *DesiredPrivatePlacementSet `protobuf:"bytes,2,opt,name=private_placement_set,json=privatePlacementSet,proto3" json:"private_placement_set,omitempty"`
	unknownFields       protoimpl.UnknownFields
	sizeCache           protoimpl.SizeCache
}

func (x *PreparePrivatePlacementCall) Reset() {
	*x = PreparePrivatePlacementCall{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[5]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *PreparePrivatePlacementCall) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*PreparePrivatePlacementCall) ProtoMessage() {}

func (x *PreparePrivatePlacementCall) ProtoReflect() protoreflect.Message {
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

// Deprecated: Use PreparePrivatePlacementCall.ProtoReflect.Descriptor instead.
func (*PreparePrivatePlacementCall) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{5}
}

func (x *PreparePrivatePlacementCall) GetClaim() *Claim {
	if x != nil {
		return x.Claim
	}
	return nil
}

func (x *PreparePrivatePlacementCall) GetPrivatePlacementSet() *DesiredPrivatePlacementSet {
	if x != nil {
		return x.PrivatePlacementSet
	}
	return nil
}

// One preparation's observable progress. Byte counters are monotonic within a call; the
// terminal PREPARED event carries the exact DesiredPlacementSet the owner sends next, and the
// terminal REFUSED event carries one bounded safe code and detail.
type PrepareEvent struct {
	state            protoimpl.MessageState `protogen:"open.v1"`
	Stage            PrepareStage           `protobuf:"varint,1,opt,name=stage,proto3,enum=cozy.worker.v1.PrepareStage" json:"stage,omitempty"`
	TotalBytes       uint64                 `protobuf:"varint,2,opt,name=total_bytes,json=totalBytes,proto3" json:"total_bytes,omitempty"`                   // bytes the bounded plan must land; 0 until RESOLVED
	TransferredBytes uint64                 `protobuf:"varint,3,opt,name=transferred_bytes,json=transferredBytes,proto3" json:"transferred_bytes,omitempty"` // verified so far
	PlacementSet     *DesiredPlacementSet   `protobuf:"bytes,4,opt,name=placement_set,json=placementSet,proto3" json:"placement_set,omitempty"`              // PREPARED only
	SafeCode         string                 `protobuf:"bytes,5,opt,name=safe_code,json=safeCode,proto3" json:"safe_code,omitempty"`                          // REFUSED only
	SafeDetail       string                 `protobuf:"bytes,6,opt,name=safe_detail,json=safeDetail,proto3" json:"safe_detail,omitempty"`                    // REFUSED only; bounded <= 1024 bytes, sanitized
	unknownFields    protoimpl.UnknownFields
	sizeCache        protoimpl.SizeCache
}

func (x *PrepareEvent) Reset() {
	*x = PrepareEvent{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[6]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *PrepareEvent) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*PrepareEvent) ProtoMessage() {}

func (x *PrepareEvent) ProtoReflect() protoreflect.Message {
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

// Deprecated: Use PrepareEvent.ProtoReflect.Descriptor instead.
func (*PrepareEvent) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{6}
}

func (x *PrepareEvent) GetStage() PrepareStage {
	if x != nil {
		return x.Stage
	}
	return PrepareStage_PREPARE_STAGE_UNSPECIFIED
}

func (x *PrepareEvent) GetTotalBytes() uint64 {
	if x != nil {
		return x.TotalBytes
	}
	return 0
}

func (x *PrepareEvent) GetTransferredBytes() uint64 {
	if x != nil {
		return x.TransferredBytes
	}
	return 0
}

func (x *PrepareEvent) GetPlacementSet() *DesiredPlacementSet {
	if x != nil {
		return x.PlacementSet
	}
	return nil
}

func (x *PrepareEvent) GetSafeCode() string {
	if x != nil {
		return x.SafeCode
	}
	return ""
}

func (x *PrepareEvent) GetSafeDetail() string {
	if x != nil {
		return x.SafeDetail
	}
	return ""
}

type ModelSourceFileCall struct {
	state         protoimpl.MessageState  `protogen:"open.v1"`
	Claim         *Claim                  `protobuf:"bytes,1,opt,name=claim,proto3" json:"claim,omitempty"`
	Request       *ModelSourceFileRequest `protobuf:"bytes,2,opt,name=request,proto3" json:"request,omitempty"`
	unknownFields protoimpl.UnknownFields
	sizeCache     protoimpl.SizeCache
}

func (x *ModelSourceFileCall) Reset() {
	*x = ModelSourceFileCall{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[7]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *ModelSourceFileCall) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*ModelSourceFileCall) ProtoMessage() {}

func (x *ModelSourceFileCall) ProtoReflect() protoreflect.Message {
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

// Deprecated: Use ModelSourceFileCall.ProtoReflect.Descriptor instead.
func (*ModelSourceFileCall) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{7}
}

func (x *ModelSourceFileCall) GetClaim() *Claim {
	if x != nil {
		return x.Claim
	}
	return nil
}

func (x *ModelSourceFileCall) GetRequest() *ModelSourceFileRequest {
	if x != nil {
		return x.Request
	}
	return nil
}

type ModelSourcePrepareCall struct {
	state         protoimpl.MessageState     `protogen:"open.v1"`
	Claim         *Claim                     `protobuf:"bytes,1,opt,name=claim,proto3" json:"claim,omitempty"`
	Request       *ModelSourcePrepareRequest `protobuf:"bytes,2,opt,name=request,proto3" json:"request,omitempty"`
	unknownFields protoimpl.UnknownFields
	sizeCache     protoimpl.SizeCache
}

func (x *ModelSourcePrepareCall) Reset() {
	*x = ModelSourcePrepareCall{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[8]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *ModelSourcePrepareCall) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*ModelSourcePrepareCall) ProtoMessage() {}

func (x *ModelSourcePrepareCall) ProtoReflect() protoreflect.Message {
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

// Deprecated: Use ModelSourcePrepareCall.ProtoReflect.Descriptor instead.
func (*ModelSourcePrepareCall) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{8}
}

func (x *ModelSourcePrepareCall) GetClaim() *Claim {
	if x != nil {
		return x.Claim
	}
	return nil
}

func (x *ModelSourcePrepareCall) GetRequest() *ModelSourcePrepareRequest {
	if x != nil {
		return x.Request
	}
	return nil
}

// Permanent owner disposition, never a pause. Host journals the tombstone,
// cancels outstanding fetches, drains entered conversion, then releases this
// operation's native roots. Other source operations are unaffected.
type ModelSourceReleaseCall struct {
	state                 protoimpl.MessageState `protogen:"open.v1"`
	Claim                 *Claim                 `protobuf:"bytes,1,opt,name=claim,proto3" json:"claim,omitempty"`
	OperationId           string                 `protobuf:"bytes,2,opt,name=operation_id,json=operationId,proto3" json:"operation_id,omitempty"`
	SourceSelectionDigest []byte                 `protobuf:"bytes,3,opt,name=source_selection_digest,json=sourceSelectionDigest,proto3" json:"source_selection_digest,omitempty"`
	unknownFields         protoimpl.UnknownFields
	sizeCache             protoimpl.SizeCache
}

func (x *ModelSourceReleaseCall) Reset() {
	*x = ModelSourceReleaseCall{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[9]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *ModelSourceReleaseCall) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*ModelSourceReleaseCall) ProtoMessage() {}

func (x *ModelSourceReleaseCall) ProtoReflect() protoreflect.Message {
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

// Deprecated: Use ModelSourceReleaseCall.ProtoReflect.Descriptor instead.
func (*ModelSourceReleaseCall) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{9}
}

func (x *ModelSourceReleaseCall) GetClaim() *Claim {
	if x != nil {
		return x.Claim
	}
	return nil
}

func (x *ModelSourceReleaseCall) GetOperationId() string {
	if x != nil {
		return x.OperationId
	}
	return ""
}

func (x *ModelSourceReleaseCall) GetSourceSelectionDigest() []byte {
	if x != nil {
		return x.SourceSelectionDigest
	}
	return nil
}

// Loopback-only Host -> Runtime. The Host has already checked signed ownership
// and exact source selection; Runtime serializes against native preparation.
type ReleaseModelSourceRequest struct {
	state         protoimpl.MessageState `protogen:"open.v1"`
	OperationId   string                 `protobuf:"bytes,1,opt,name=operation_id,json=operationId,proto3" json:"operation_id,omitempty"`
	unknownFields protoimpl.UnknownFields
	sizeCache     protoimpl.SizeCache
}

func (x *ReleaseModelSourceRequest) Reset() {
	*x = ReleaseModelSourceRequest{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[10]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *ReleaseModelSourceRequest) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*ReleaseModelSourceRequest) ProtoMessage() {}

func (x *ReleaseModelSourceRequest) ProtoReflect() protoreflect.Message {
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

// Deprecated: Use ReleaseModelSourceRequest.ProtoReflect.Descriptor instead.
func (*ReleaseModelSourceRequest) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{10}
}

func (x *ReleaseModelSourceRequest) GetOperationId() string {
	if x != nil {
		return x.OperationId
	}
	return ""
}

// MINOR 40. Independent result custody preserves the original native receipt and
// manifest. The RecordOwner assigns retention_id; it never enters author payloads.
// tensorfs_receipt_digest hashes the NATIVE canonical TensorFS receipt, not its
// enclosing cozy.worker.v1.WeightsReceipt document. Runtime verifies it natively.
type DerivedRetentionCall struct {
	state         protoimpl.MessageState   `protogen:"open.v1"`
	Claim         *Claim                   `protobuf:"bytes,1,opt,name=claim,proto3" json:"claim,omitempty"`
	Request       *DerivedRetentionRequest `protobuf:"bytes,2,opt,name=request,proto3" json:"request,omitempty"`
	unknownFields protoimpl.UnknownFields
	sizeCache     protoimpl.SizeCache
}

func (x *DerivedRetentionCall) Reset() {
	*x = DerivedRetentionCall{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[11]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *DerivedRetentionCall) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*DerivedRetentionCall) ProtoMessage() {}

func (x *DerivedRetentionCall) ProtoReflect() protoreflect.Message {
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

// Deprecated: Use DerivedRetentionCall.ProtoReflect.Descriptor instead.
func (*DerivedRetentionCall) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{11}
}

func (x *DerivedRetentionCall) GetClaim() *Claim {
	if x != nil {
		return x.Claim
	}
	return nil
}

func (x *DerivedRetentionCall) GetRequest() *DerivedRetentionRequest {
	if x != nil {
		return x.Request
	}
	return nil
}

type DerivedRetentionRequest struct {
	state                 protoimpl.MessageState `protogen:"open.v1"`
	WeightsTransactionId  string                 `protobuf:"bytes,1,opt,name=weights_transaction_id,json=weightsTransactionId,proto3" json:"weights_transaction_id,omitempty"`
	TensorfsReceiptDigest []byte                 `protobuf:"bytes,2,opt,name=tensorfs_receipt_digest,json=tensorfsReceiptDigest,proto3" json:"tensorfs_receipt_digest,omitempty"`
	RetentionId           string                 `protobuf:"bytes,3,opt,name=retention_id,json=retentionId,proto3" json:"retention_id,omitempty"`
	unknownFields         protoimpl.UnknownFields
	sizeCache             protoimpl.SizeCache
}

func (x *DerivedRetentionRequest) Reset() {
	*x = DerivedRetentionRequest{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[12]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *DerivedRetentionRequest) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*DerivedRetentionRequest) ProtoMessage() {}

func (x *DerivedRetentionRequest) ProtoReflect() protoreflect.Message {
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

// Deprecated: Use DerivedRetentionRequest.ProtoReflect.Descriptor instead.
func (*DerivedRetentionRequest) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{12}
}

func (x *DerivedRetentionRequest) GetWeightsTransactionId() string {
	if x != nil {
		return x.WeightsTransactionId
	}
	return ""
}

func (x *DerivedRetentionRequest) GetTensorfsReceiptDigest() []byte {
	if x != nil {
		return x.TensorfsReceiptDigest
	}
	return nil
}

func (x *DerivedRetentionRequest) GetRetentionId() string {
	if x != nil {
		return x.RetentionId
	}
	return ""
}

type DerivedRetentionResult struct {
	state                 protoimpl.MessageState `protogen:"open.v1"`
	WeightsTransactionId  string                 `protobuf:"bytes,1,opt,name=weights_transaction_id,json=weightsTransactionId,proto3" json:"weights_transaction_id,omitempty"`
	TensorfsReceiptDigest []byte                 `protobuf:"bytes,2,opt,name=tensorfs_receipt_digest,json=tensorfsReceiptDigest,proto3" json:"tensorfs_receipt_digest,omitempty"`
	RetentionId           string                 `protobuf:"bytes,3,opt,name=retention_id,json=retentionId,proto3" json:"retention_id,omitempty"`
	Manifest              *Ref                   `protobuf:"bytes,4,opt,name=manifest,proto3" json:"manifest,omitempty"` // absent only when release won before retention existed
	Released              bool                   `protobuf:"varint,5,opt,name=released,proto3" json:"released,omitempty"`
	unknownFields         protoimpl.UnknownFields
	sizeCache             protoimpl.SizeCache
}

func (x *DerivedRetentionResult) Reset() {
	*x = DerivedRetentionResult{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[13]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *DerivedRetentionResult) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*DerivedRetentionResult) ProtoMessage() {}

func (x *DerivedRetentionResult) ProtoReflect() protoreflect.Message {
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

// Deprecated: Use DerivedRetentionResult.ProtoReflect.Descriptor instead.
func (*DerivedRetentionResult) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{13}
}

func (x *DerivedRetentionResult) GetWeightsTransactionId() string {
	if x != nil {
		return x.WeightsTransactionId
	}
	return ""
}

func (x *DerivedRetentionResult) GetTensorfsReceiptDigest() []byte {
	if x != nil {
		return x.TensorfsReceiptDigest
	}
	return nil
}

func (x *DerivedRetentionResult) GetRetentionId() string {
	if x != nil {
		return x.RetentionId
	}
	return ""
}

func (x *DerivedRetentionResult) GetManifest() *Ref {
	if x != nil {
		return x.Manifest
	}
	return nil
}

func (x *DerivedRetentionResult) GetReleased() bool {
	if x != nil {
		return x.Released
	}
	return false
}

type ReleaseModelSourceResult struct {
	state         protoimpl.MessageState `protogen:"open.v1"`
	OperationId   string                 `protobuf:"bytes,1,opt,name=operation_id,json=operationId,proto3" json:"operation_id,omitempty"`
	Released      bool                   `protobuf:"varint,2,opt,name=released,proto3" json:"released,omitempty"`
	unknownFields protoimpl.UnknownFields
	sizeCache     protoimpl.SizeCache
}

func (x *ReleaseModelSourceResult) Reset() {
	*x = ReleaseModelSourceResult{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[14]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *ReleaseModelSourceResult) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*ReleaseModelSourceResult) ProtoMessage() {}

func (x *ReleaseModelSourceResult) ProtoReflect() protoreflect.Message {
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

// Deprecated: Use ReleaseModelSourceResult.ProtoReflect.Descriptor instead.
func (*ReleaseModelSourceResult) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{14}
}

func (x *ReleaseModelSourceResult) GetOperationId() string {
	if x != nil {
		return x.OperationId
	}
	return ""
}

func (x *ReleaseModelSourceResult) GetReleased() bool {
	if x != nil {
		return x.Released
	}
	return false
}

// The signed owner adopts retained source work into a different ordinary request
// on this pod. Both operations must declare the same exact source selection;
// TensorFS independently validates the old checkpoint chain and newly derived
// profile plan, creates a new operation chain, and retains its roots before the
// old owner may release. No provider reads or incomplete conversion starts here.
type ModelSourceAdoptCall struct {
	state           protoimpl.MessageState     `protogen:"open.v1"`
	Claim           *Claim                     `protobuf:"bytes,1,opt,name=claim,proto3" json:"claim,omitempty"`
	FromOperationId string                     `protobuf:"bytes,2,opt,name=from_operation_id,json=fromOperationId,proto3" json:"from_operation_id,omitempty"`
	Request         *ModelSourcePrepareRequest `protobuf:"bytes,3,opt,name=request,proto3" json:"request,omitempty"`
	unknownFields   protoimpl.UnknownFields
	sizeCache       protoimpl.SizeCache
}

func (x *ModelSourceAdoptCall) Reset() {
	*x = ModelSourceAdoptCall{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[15]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *ModelSourceAdoptCall) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*ModelSourceAdoptCall) ProtoMessage() {}

func (x *ModelSourceAdoptCall) ProtoReflect() protoreflect.Message {
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

// Deprecated: Use ModelSourceAdoptCall.ProtoReflect.Descriptor instead.
func (*ModelSourceAdoptCall) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{15}
}

func (x *ModelSourceAdoptCall) GetClaim() *Claim {
	if x != nil {
		return x.Claim
	}
	return nil
}

func (x *ModelSourceAdoptCall) GetFromOperationId() string {
	if x != nil {
		return x.FromOperationId
	}
	return ""
}

func (x *ModelSourceAdoptCall) GetRequest() *ModelSourcePrepareRequest {
	if x != nil {
		return x.Request
	}
	return nil
}

type CheckpointPageCall struct {
	state         protoimpl.MessageState `protogen:"open.v1"`
	Claim         *Claim                 `protobuf:"bytes,1,opt,name=claim,proto3" json:"claim,omitempty"`
	Request       *CheckpointPageRequest `protobuf:"bytes,2,opt,name=request,proto3" json:"request,omitempty"`
	unknownFields protoimpl.UnknownFields
	sizeCache     protoimpl.SizeCache
}

func (x *CheckpointPageCall) Reset() {
	*x = CheckpointPageCall{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[16]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *CheckpointPageCall) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*CheckpointPageCall) ProtoMessage() {}

func (x *CheckpointPageCall) ProtoReflect() protoreflect.Message {
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

// Deprecated: Use CheckpointPageCall.ProtoReflect.Descriptor instead.
func (*CheckpointPageCall) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{16}
}

func (x *CheckpointPageCall) GetClaim() *Claim {
	if x != nil {
		return x.Claim
	}
	return nil
}

func (x *CheckpointPageCall) GetRequest() *CheckpointPageRequest {
	if x != nil {
		return x.Request
	}
	return nil
}

type CheckpointTransferCall struct {
	state         protoimpl.MessageState     `protogen:"open.v1"`
	Claim         *Claim                     `protobuf:"bytes,1,opt,name=claim,proto3" json:"claim,omitempty"`
	Request       *CheckpointTransferRequest `protobuf:"bytes,2,opt,name=request,proto3" json:"request,omitempty"`
	unknownFields protoimpl.UnknownFields
	sizeCache     protoimpl.SizeCache
}

func (x *CheckpointTransferCall) Reset() {
	*x = CheckpointTransferCall{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[17]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *CheckpointTransferCall) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*CheckpointTransferCall) ProtoMessage() {}

func (x *CheckpointTransferCall) ProtoReflect() protoreflect.Message {
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

// Deprecated: Use CheckpointTransferCall.ProtoReflect.Descriptor instead.
func (*CheckpointTransferCall) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{17}
}

func (x *CheckpointTransferCall) GetClaim() *Claim {
	if x != nil {
		return x.Claim
	}
	return nil
}

func (x *CheckpointTransferCall) GetRequest() *CheckpointTransferRequest {
	if x != nil {
		return x.Request
	}
	return nil
}

type LocalPackageFetchCall struct {
	state         protoimpl.MessageState    `protogen:"open.v1"`
	Claim         *Claim                    `protobuf:"bytes,1,opt,name=claim,proto3" json:"claim,omitempty"`
	Request       *LocalPackageFetchRequest `protobuf:"bytes,2,opt,name=request,proto3" json:"request,omitempty"`
	unknownFields protoimpl.UnknownFields
	sizeCache     protoimpl.SizeCache
}

func (x *LocalPackageFetchCall) Reset() {
	*x = LocalPackageFetchCall{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[18]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *LocalPackageFetchCall) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*LocalPackageFetchCall) ProtoMessage() {}

func (x *LocalPackageFetchCall) ProtoReflect() protoreflect.Message {
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

// Deprecated: Use LocalPackageFetchCall.ProtoReflect.Descriptor instead.
func (*LocalPackageFetchCall) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{18}
}

func (x *LocalPackageFetchCall) GetClaim() *Claim {
	if x != nil {
		return x.Claim
	}
	return nil
}

func (x *LocalPackageFetchCall) GetRequest() *LocalPackageFetchRequest {
	if x != nil {
		return x.Request
	}
	return nil
}

// MINOR 39: Creator uploads private wheels directly over this dedicated PodHost
// byte stream. Code never enters WorkerControl or a Tensorhub publication/store.
// Header is first and occurs exactly once. Host reserves the exact file and
// replies with its durable received_bytes before accepting chunks. Each chunk
// starts at that exact offset, is at most 1 MiB, and receives a durable progress
// response. VERIFIED follows complete file digest verification. Disconnect keeps
// the prefix; replay the header to learn the resume offset. Existing operation
// file/count/aggregate limits and PrepareLocalPackage remain the custody contract.
type LocalPackageUploadFrame struct {
	state protoimpl.MessageState `protogen:"open.v1"`
	// Types that are valid to be assigned to Body:
	//
	//	*LocalPackageUploadFrame_Header
	//	*LocalPackageUploadFrame_Chunk
	Body          isLocalPackageUploadFrame_Body `protobuf_oneof:"body"`
	unknownFields protoimpl.UnknownFields
	sizeCache     protoimpl.SizeCache
}

func (x *LocalPackageUploadFrame) Reset() {
	*x = LocalPackageUploadFrame{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[19]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *LocalPackageUploadFrame) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*LocalPackageUploadFrame) ProtoMessage() {}

func (x *LocalPackageUploadFrame) ProtoReflect() protoreflect.Message {
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

// Deprecated: Use LocalPackageUploadFrame.ProtoReflect.Descriptor instead.
func (*LocalPackageUploadFrame) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{19}
}

func (x *LocalPackageUploadFrame) GetBody() isLocalPackageUploadFrame_Body {
	if x != nil {
		return x.Body
	}
	return nil
}

func (x *LocalPackageUploadFrame) GetHeader() *LocalPackageUploadHeader {
	if x != nil {
		if x, ok := x.Body.(*LocalPackageUploadFrame_Header); ok {
			return x.Header
		}
	}
	return nil
}

func (x *LocalPackageUploadFrame) GetChunk() *LocalPackageUploadChunk {
	if x != nil {
		if x, ok := x.Body.(*LocalPackageUploadFrame_Chunk); ok {
			return x.Chunk
		}
	}
	return nil
}

type isLocalPackageUploadFrame_Body interface {
	isLocalPackageUploadFrame_Body()
}

type LocalPackageUploadFrame_Header struct {
	Header *LocalPackageUploadHeader `protobuf:"bytes,1,opt,name=header,proto3,oneof"`
}

type LocalPackageUploadFrame_Chunk struct {
	Chunk *LocalPackageUploadChunk `protobuf:"bytes,2,opt,name=chunk,proto3,oneof"`
}

func (*LocalPackageUploadFrame_Header) isLocalPackageUploadFrame_Body() {}

func (*LocalPackageUploadFrame_Chunk) isLocalPackageUploadFrame_Body() {}

type LocalPackageUploadHeader struct {
	state         protoimpl.MessageState `protogen:"open.v1"`
	Claim         *Claim                 `protobuf:"bytes,1,opt,name=claim,proto3" json:"claim,omitempty"`
	OperationId   string                 `protobuf:"bytes,2,opt,name=operation_id,json=operationId,proto3" json:"operation_id,omitempty"`
	SourceDigest  []byte                 `protobuf:"bytes,3,opt,name=source_digest,json=sourceDigest,proto3" json:"source_digest,omitempty"`
	File          *LocalPackageFileRef   `protobuf:"bytes,4,opt,name=file,proto3" json:"file,omitempty"`
	unknownFields protoimpl.UnknownFields
	sizeCache     protoimpl.SizeCache
}

func (x *LocalPackageUploadHeader) Reset() {
	*x = LocalPackageUploadHeader{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[20]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *LocalPackageUploadHeader) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*LocalPackageUploadHeader) ProtoMessage() {}

func (x *LocalPackageUploadHeader) ProtoReflect() protoreflect.Message {
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

// Deprecated: Use LocalPackageUploadHeader.ProtoReflect.Descriptor instead.
func (*LocalPackageUploadHeader) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{20}
}

func (x *LocalPackageUploadHeader) GetClaim() *Claim {
	if x != nil {
		return x.Claim
	}
	return nil
}

func (x *LocalPackageUploadHeader) GetOperationId() string {
	if x != nil {
		return x.OperationId
	}
	return ""
}

func (x *LocalPackageUploadHeader) GetSourceDigest() []byte {
	if x != nil {
		return x.SourceDigest
	}
	return nil
}

func (x *LocalPackageUploadHeader) GetFile() *LocalPackageFileRef {
	if x != nil {
		return x.File
	}
	return nil
}

type LocalPackageUploadChunk struct {
	state         protoimpl.MessageState `protogen:"open.v1"`
	Offset        uint64                 `protobuf:"varint,1,opt,name=offset,proto3" json:"offset,omitempty"`
	Data          []byte                 `protobuf:"bytes,2,opt,name=data,proto3" json:"data,omitempty"`
	unknownFields protoimpl.UnknownFields
	sizeCache     protoimpl.SizeCache
}

func (x *LocalPackageUploadChunk) Reset() {
	*x = LocalPackageUploadChunk{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[21]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *LocalPackageUploadChunk) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*LocalPackageUploadChunk) ProtoMessage() {}

func (x *LocalPackageUploadChunk) ProtoReflect() protoreflect.Message {
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

// Deprecated: Use LocalPackageUploadChunk.ProtoReflect.Descriptor instead.
func (*LocalPackageUploadChunk) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{21}
}

func (x *LocalPackageUploadChunk) GetOffset() uint64 {
	if x != nil {
		return x.Offset
	}
	return 0
}

func (x *LocalPackageUploadChunk) GetData() []byte {
	if x != nil {
		return x.Data
	}
	return nil
}

type LocalPackageAbortCall struct {
	state         protoimpl.MessageState `protogen:"open.v1"`
	Claim         *Claim                 `protobuf:"bytes,1,opt,name=claim,proto3" json:"claim,omitempty"`
	Request       *LocalPackageAbort     `protobuf:"bytes,2,opt,name=request,proto3" json:"request,omitempty"`
	unknownFields protoimpl.UnknownFields
	sizeCache     protoimpl.SizeCache
}

func (x *LocalPackageAbortCall) Reset() {
	*x = LocalPackageAbortCall{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[22]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *LocalPackageAbortCall) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*LocalPackageAbortCall) ProtoMessage() {}

func (x *LocalPackageAbortCall) ProtoReflect() protoreflect.Message {
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

// Deprecated: Use LocalPackageAbortCall.ProtoReflect.Descriptor instead.
func (*LocalPackageAbortCall) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{22}
}

func (x *LocalPackageAbortCall) GetClaim() *Claim {
	if x != nil {
		return x.Claim
	}
	return nil
}

func (x *LocalPackageAbortCall) GetRequest() *LocalPackageAbort {
	if x != nil {
		return x.Request
	}
	return nil
}

type WeightsTransferCall struct {
	state         protoimpl.MessageState  `protogen:"open.v1"`
	Claim         *Claim                  `protobuf:"bytes,1,opt,name=claim,proto3" json:"claim,omitempty"`
	Request       *WeightsTransferRequest `protobuf:"bytes,2,opt,name=request,proto3" json:"request,omitempty"`
	unknownFields protoimpl.UnknownFields
	sizeCache     protoimpl.SizeCache
}

func (x *WeightsTransferCall) Reset() {
	*x = WeightsTransferCall{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[23]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *WeightsTransferCall) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*WeightsTransferCall) ProtoMessage() {}

func (x *WeightsTransferCall) ProtoReflect() protoreflect.Message {
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

// Deprecated: Use WeightsTransferCall.ProtoReflect.Descriptor instead.
func (*WeightsTransferCall) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{23}
}

func (x *WeightsTransferCall) GetClaim() *Claim {
	if x != nil {
		return x.Claim
	}
	return nil
}

func (x *WeightsTransferCall) GetRequest() *WeightsTransferRequest {
	if x != nil {
		return x.Request
	}
	return nil
}

// MINOR 30: the loopback form of PreparePackageSetCall. The retired per-file handoff
// (`LocalDownloadFile` with `LocalDownloadKind` PROJECT_WHEEL/DEPENDENCY_WHEEL/MODEL_MANIFEST/
// MODEL_OBJECT/PACKAGE_INTERFACE) is DELETED, not renamed; the names must not return. Runtime
// preparation no longer receives files whose final owner is TensorFS or the package
// index/install materializer.
type PreparePackageSetRequest struct {
	state              protoimpl.MessageState `protogen:"open.v1"`
	DownloadDelegation []byte                 `protobuf:"bytes,1,opt,name=download_delegation,json=downloadDelegation,proto3" json:"download_delegation,omitempty"` // exact canonical DownloadDelegation/1 bytes
	Application        string                 `protobuf:"bytes,4,opt,name=application,proto3" json:"application,omitempty"`                                         // as PreparePackageSetCall fields 3-6
	ModelSlotPaths     []string               `protobuf:"bytes,5,rep,name=model_slot_paths,json=modelSlotPaths,proto3" json:"model_slot_paths,omitempty"`
	ImageInventory     *ImageInventory        `protobuf:"bytes,6,opt,name=image_inventory,json=imageInventory,proto3" json:"image_inventory,omitempty"`
	LockedRequirements []byte                 `protobuf:"bytes,7,opt,name=locked_requirements,json=lockedRequirements,proto3" json:"locked_requirements,omitempty"`
	InstallRoot        string                 `protobuf:"bytes,8,opt,name=install_root,json=installRoot,proto3" json:"install_root,omitempty"` // absolute immutable install destination
	// MINOR 32: the delegation's other half. The same 64-byte Ed25519 signature by the rental
	// Creator key that DesiredPackageSet.download_delegation_signature carries, forwarded
	// unchanged by the pod host. Runtime's TensorFS presents `delegation <payload> <signature>`
	// as its hub credential on the tensorfs closure and presign routes; the payload alone does
	// not authenticate.
	DownloadDelegationSignature []byte `protobuf:"bytes,9,opt,name=download_delegation_signature,json=downloadDelegationSignature,proto3" json:"download_delegation_signature,omitempty"`
	unknownFields               protoimpl.UnknownFields
	sizeCache                   protoimpl.SizeCache
}

func (x *PreparePackageSetRequest) Reset() {
	*x = PreparePackageSetRequest{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[24]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *PreparePackageSetRequest) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*PreparePackageSetRequest) ProtoMessage() {}

func (x *PreparePackageSetRequest) ProtoReflect() protoreflect.Message {
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

// Deprecated: Use PreparePackageSetRequest.ProtoReflect.Descriptor instead.
func (*PreparePackageSetRequest) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{24}
}

func (x *PreparePackageSetRequest) GetDownloadDelegation() []byte {
	if x != nil {
		return x.DownloadDelegation
	}
	return nil
}

func (x *PreparePackageSetRequest) GetApplication() string {
	if x != nil {
		return x.Application
	}
	return ""
}

func (x *PreparePackageSetRequest) GetModelSlotPaths() []string {
	if x != nil {
		return x.ModelSlotPaths
	}
	return nil
}

func (x *PreparePackageSetRequest) GetImageInventory() *ImageInventory {
	if x != nil {
		return x.ImageInventory
	}
	return nil
}

func (x *PreparePackageSetRequest) GetLockedRequirements() []byte {
	if x != nil {
		return x.LockedRequirements
	}
	return nil
}

func (x *PreparePackageSetRequest) GetInstallRoot() string {
	if x != nil {
		return x.InstallRoot
	}
	return ""
}

func (x *PreparePackageSetRequest) GetDownloadDelegationSignature() []byte {
	if x != nil {
		return x.DownloadDelegationSignature
	}
	return nil
}

// The placed worker image's pinned inventory (`tensorhub.image_inventory/1` facts). The list
// the image was built from is the truth about what the image provides; the live base
// observation is its cross-check (cr-070). An inventory-owned NAME is served by the image's
// site-packages copy and never enters the venv from the lock.
type ImageInventory struct {
	state         protoimpl.MessageState `protogen:"open.v1"`
	Profile       string                 `protobuf:"bytes,1,opt,name=profile,proto3" json:"profile,omitempty"`             // the registered base image profile
	Python        string                 `protobuf:"bytes,2,opt,name=python,proto3" json:"python,omitempty"`               // exact interpreter version the image ships
	Distributions []*ImageDistribution   `protobuf:"bytes,3,rep,name=distributions,proto3" json:"distributions,omitempty"` // sorted unique by distribution;
	unknownFields protoimpl.UnknownFields
	sizeCache     protoimpl.SizeCache
}

func (x *ImageInventory) Reset() {
	*x = ImageInventory{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[25]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *ImageInventory) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*ImageInventory) ProtoMessage() {}

func (x *ImageInventory) ProtoReflect() protoreflect.Message {
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

// Deprecated: Use ImageInventory.ProtoReflect.Descriptor instead.
func (*ImageInventory) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{25}
}

func (x *ImageInventory) GetProfile() string {
	if x != nil {
		return x.Profile
	}
	return ""
}

func (x *ImageInventory) GetPython() string {
	if x != nil {
		return x.Python
	}
	return ""
}

func (x *ImageInventory) GetDistributions() []*ImageDistribution {
	if x != nil {
		return x.Distributions
	}
	return nil
}

type ImageDistribution struct {
	state         protoimpl.MessageState `protogen:"open.v1"`
	Distribution  string                 `protobuf:"bytes,1,opt,name=distribution,proto3" json:"distribution,omitempty"` // normalized lowercase distribution name
	Version       string                 `protobuf:"bytes,2,opt,name=version,proto3" json:"version,omitempty"`           // the exact == pin the image promises
	unknownFields protoimpl.UnknownFields
	sizeCache     protoimpl.SizeCache
}

func (x *ImageDistribution) Reset() {
	*x = ImageDistribution{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[26]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *ImageDistribution) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*ImageDistribution) ProtoMessage() {}

func (x *ImageDistribution) ProtoReflect() protoreflect.Message {
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

// Deprecated: Use ImageDistribution.ProtoReflect.Descriptor instead.
func (*ImageDistribution) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{26}
}

func (x *ImageDistribution) GetDistribution() string {
	if x != nil {
		return x.Distribution
	}
	return ""
}

func (x *ImageDistribution) GetVersion() string {
	if x != nil {
		return x.Version
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
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[27]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *PreparePackageSetResult) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*PreparePackageSetResult) ProtoMessage() {}

func (x *PreparePackageSetResult) ProtoReflect() protoreflect.Message {
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

// Deprecated: Use PreparePackageSetResult.ProtoReflect.Descriptor instead.
func (*PreparePackageSetResult) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{27}
}

func (x *PreparePackageSetResult) GetPlacementSet() *DesiredPlacementSet {
	if x != nil {
		return x.PlacementSet
	}
	return nil
}

// A loopback cold-download short-circuit. Empty strings mean the exact transient
// construction fits the selected CozyTensors headers — manifests and header objects the host
// already admitted to the local TensorFS Store while tensor bodies are still in flight. A
// refusal is safe to relay to the request owner and means the supervisor may cancel the
// still-running tensor-body downloads.
type CheckPackageSetCompatibilityResult struct {
	state         protoimpl.MessageState `protogen:"open.v1"`
	RefusalCode   string                 `protobuf:"bytes,1,opt,name=refusal_code,json=refusalCode,proto3" json:"refusal_code,omitempty"`
	RefusalDetail string                 `protobuf:"bytes,2,opt,name=refusal_detail,json=refusalDetail,proto3" json:"refusal_detail,omitempty"`
	unknownFields protoimpl.UnknownFields
	sizeCache     protoimpl.SizeCache
}

func (x *CheckPackageSetCompatibilityResult) Reset() {
	*x = CheckPackageSetCompatibilityResult{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[28]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *CheckPackageSetCompatibilityResult) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*CheckPackageSetCompatibilityResult) ProtoMessage() {}

func (x *CheckPackageSetCompatibilityResult) ProtoReflect() protoreflect.Message {
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

// Deprecated: Use CheckPackageSetCompatibilityResult.ProtoReflect.Descriptor instead.
func (*CheckPackageSetCompatibilityResult) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{28}
}

func (x *CheckPackageSetCompatibilityResult) GetRefusalCode() string {
	if x != nil {
		return x.RefusalCode
	}
	return ""
}

func (x *CheckPackageSetCompatibilityResult) GetRefusalDetail() string {
	if x != nil {
		return x.RefusalDetail
	}
	return ""
}

// Loopback-only local-package preparation. The supervisor substitutes its verified local paths
// for the external owner's chunks; the Runtime sees neither the control stream nor a remote URL.
// The supplied wheels are the ONLY files Runtime preparation still receives: they exist on no
// index, so their exact bytes are the install input (xs-017). Exactly one row must MEASURE as
// the project wheel — its wheel metadata names the package's own distribution and release.
type PrepareLocalPackageRequest struct {
	state         protoimpl.MessageState `protogen:"open.v1"`
	OperationId   string                 `protobuf:"bytes,1,opt,name=operation_id,json=operationId,proto3" json:"operation_id,omitempty"`
	Package       *DevelopmentPackage    `protobuf:"bytes,2,opt,name=package,proto3" json:"package,omitempty"`
	Wheels        []*LocalPackageWheel   `protobuf:"bytes,5,rep,name=wheels,proto3" json:"wheels,omitempty"`                              // sorted unique by digest; <= MaxLocalPackageFiles
	InstallRoot   string                 `protobuf:"bytes,6,opt,name=install_root,json=installRoot,proto3" json:"install_root,omitempty"` // absolute immutable install destination
	unknownFields protoimpl.UnknownFields
	sizeCache     protoimpl.SizeCache
}

func (x *PrepareLocalPackageRequest) Reset() {
	*x = PrepareLocalPackageRequest{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[29]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *PrepareLocalPackageRequest) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*PrepareLocalPackageRequest) ProtoMessage() {}

func (x *PrepareLocalPackageRequest) ProtoReflect() protoreflect.Message {
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

// Deprecated: Use PrepareLocalPackageRequest.ProtoReflect.Descriptor instead.
func (*PrepareLocalPackageRequest) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{29}
}

func (x *PrepareLocalPackageRequest) GetOperationId() string {
	if x != nil {
		return x.OperationId
	}
	return ""
}

func (x *PrepareLocalPackageRequest) GetPackage() *DevelopmentPackage {
	if x != nil {
		return x.Package
	}
	return nil
}

func (x *PrepareLocalPackageRequest) GetWheels() []*LocalPackageWheel {
	if x != nil {
		return x.Wheels
	}
	return nil
}

func (x *PrepareLocalPackageRequest) GetInstallRoot() string {
	if x != nil {
		return x.InstallRoot
	}
	return ""
}

type LocalPackageWheel struct {
	state         protoimpl.MessageState `protogen:"open.v1"`
	Digest        []byte                 `protobuf:"bytes,1,opt,name=digest,proto3" json:"digest,omitempty"`
	Filename      string                 `protobuf:"bytes,2,opt,name=filename,proto3" json:"filename,omitempty"`
	Length        uint64                 `protobuf:"varint,3,opt,name=length,proto3" json:"length,omitempty"`
	Path          string                 `protobuf:"bytes,4,opt,name=path,proto3" json:"path,omitempty"` // absolute verified supervisor-owned file
	unknownFields protoimpl.UnknownFields
	sizeCache     protoimpl.SizeCache
}

func (x *LocalPackageWheel) Reset() {
	*x = LocalPackageWheel{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[30]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *LocalPackageWheel) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*LocalPackageWheel) ProtoMessage() {}

func (x *LocalPackageWheel) ProtoReflect() protoreflect.Message {
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

// Deprecated: Use LocalPackageWheel.ProtoReflect.Descriptor instead.
func (*LocalPackageWheel) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{30}
}

func (x *LocalPackageWheel) GetDigest() []byte {
	if x != nil {
		return x.Digest
	}
	return nil
}

func (x *LocalPackageWheel) GetFilename() string {
	if x != nil {
		return x.Filename
	}
	return ""
}

func (x *LocalPackageWheel) GetLength() uint64 {
	if x != nil {
		return x.Length
	}
	return 0
}

func (x *LocalPackageWheel) GetPath() string {
	if x != nil {
		return x.Path
	}
	return ""
}

// Loopback-only model binding for one already-prepared local revision. The host has admitted
// every selected model file to the local TensorFS Store, and the Runtime has already retained
// the exact code, package interface, and Environment core named by local_revision_digest.
// Runtime resolves the delegation's manifests from that Store, joins the model/component/stamp
// facts, and alone authors the resulting PlacementSet. MINOR 30 deletes the model-file rows.
type PreparePrivatePlacementRequest struct {
	state               protoimpl.MessageState `protogen:"open.v1"`
	OperationId         string                 `protobuf:"bytes,1,opt,name=operation_id,json=operationId,proto3" json:"operation_id,omitempty"`
	LocalRevisionDigest []byte                 `protobuf:"bytes,2,opt,name=local_revision_digest,json=localRevisionDigest,proto3" json:"local_revision_digest,omitempty"`
	DownloadDelegation  []byte                 `protobuf:"bytes,3,opt,name=download_delegation,json=downloadDelegation,proto3" json:"download_delegation,omitempty"` // exact model-only canonical DownloadDelegation/1 bytes
	// MINOR 32: the same 64-byte Ed25519 signature by the rental Creator key that
	// DesiredPrivatePlacementSet.download_delegation_signature carries, forwarded unchanged by the
	// pod host so Runtime's TensorFS can present `delegation <payload> <signature>` to the hub's
	// tensorfs routes.
	DownloadDelegationSignature []byte `protobuf:"bytes,5,opt,name=download_delegation_signature,json=downloadDelegationSignature,proto3" json:"download_delegation_signature,omitempty"`
	unknownFields               protoimpl.UnknownFields
	sizeCache                   protoimpl.SizeCache
}

func (x *PreparePrivatePlacementRequest) Reset() {
	*x = PreparePrivatePlacementRequest{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[31]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *PreparePrivatePlacementRequest) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*PreparePrivatePlacementRequest) ProtoMessage() {}

func (x *PreparePrivatePlacementRequest) ProtoReflect() protoreflect.Message {
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

// Deprecated: Use PreparePrivatePlacementRequest.ProtoReflect.Descriptor instead.
func (*PreparePrivatePlacementRequest) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{31}
}

func (x *PreparePrivatePlacementRequest) GetOperationId() string {
	if x != nil {
		return x.OperationId
	}
	return ""
}

func (x *PreparePrivatePlacementRequest) GetLocalRevisionDigest() []byte {
	if x != nil {
		return x.LocalRevisionDigest
	}
	return nil
}

func (x *PreparePrivatePlacementRequest) GetDownloadDelegation() []byte {
	if x != nil {
		return x.DownloadDelegation
	}
	return nil
}

func (x *PreparePrivatePlacementRequest) GetDownloadDelegationSignature() []byte {
	if x != nil {
		return x.DownloadDelegationSignature
	}
	return nil
}

type LocalModelSourceFile struct {
	state         protoimpl.MessageState `protogen:"open.v1"`
	Member        string                 `protobuf:"bytes,1,opt,name=member,proto3" json:"member,omitempty"`                     // safe relative provider member, <= MaxModelSourceMemberBytes
	ObjectId      string                 `protobuf:"bytes,2,opt,name=object_id,json=objectId,proto3" json:"object_id,omitempty"` // lowercase sha256:<64 hex>
	Length        uint64                 `protobuf:"varint,3,opt,name=length,proto3" json:"length,omitempty"`
	Path          string                 `protobuf:"bytes,4,opt,name=path,proto3" json:"path,omitempty"`                               // absolute supervisor-owned payload destination; may be absent until verified
	HeaderPath    string                 `protobuf:"bytes,5,opt,name=header_path,json=headerPath,proto3" json:"header_path,omitempty"` // absolute supervisor-owned bounded header file; loopback only
	Verified      bool                   `protobuf:"varint,6,opt,name=verified,proto3" json:"verified,omitempty"`                      // this exact full payload has passed supervisor digest/length admission
	unknownFields protoimpl.UnknownFields
	sizeCache     protoimpl.SizeCache
}

func (x *LocalModelSourceFile) Reset() {
	*x = LocalModelSourceFile{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[32]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *LocalModelSourceFile) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*LocalModelSourceFile) ProtoMessage() {}

func (x *LocalModelSourceFile) ProtoReflect() protoreflect.Message {
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

// Deprecated: Use LocalModelSourceFile.ProtoReflect.Descriptor instead.
func (*LocalModelSourceFile) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{32}
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

func (x *LocalModelSourceFile) GetHeaderPath() string {
	if x != nil {
		return x.HeaderPath
	}
	return ""
}

func (x *LocalModelSourceFile) GetVerified() bool {
	if x != nil {
		return x.Verified
	}
	return false
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
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[33]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *ModelSourceProfile) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*ModelSourceProfile) ProtoMessage() {}

func (x *ModelSourceProfile) ProtoReflect() protoreflect.Message {
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

// Deprecated: Use ModelSourceProfile.ProtoReflect.Descriptor instead.
func (*ModelSourceProfile) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{33}
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
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[34]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *PreparedModelSource) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*PreparedModelSource) ProtoMessage() {}

func (x *PreparedModelSource) ProtoReflect() protoreflect.Message {
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

// Deprecated: Use PreparedModelSource.ProtoReflect.Descriptor instead.
func (*PreparedModelSource) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{34}
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

// A TensorFS-authored conversion checkpoint, not a new manifest or a publication receipt.
// The head binds the plan and a chain of immutable object refs. bytes is cumulative LOCAL
// checkpointed progress; it becomes remotely durable only after the record owner confirms
// custody of the head and its entire referenced prefix. A replacement pod receives only an
// acknowledged durable head and restores it through ordinary TensorFS admission.
type ModelSourceCheckpoint struct {
	state         protoimpl.MessageState `protogen:"open.v1"`
	Slot          string                 `protobuf:"bytes,1,opt,name=slot,proto3" json:"slot,omitempty"`
	Head          *Ref                   `protobuf:"bytes,2,opt,name=head,proto3" json:"head,omitempty"`
	PlanDigest    []byte                 `protobuf:"bytes,3,opt,name=plan_digest,json=planDigest,proto3" json:"plan_digest,omitempty"` // exact SHA-256, 32 bytes
	Index         uint64                 `protobuf:"varint,4,opt,name=index,proto3" json:"index,omitempty"`
	Bytes         uint64                 `protobuf:"varint,5,opt,name=bytes,proto3" json:"bytes,omitempty"`
	unknownFields protoimpl.UnknownFields
	sizeCache     protoimpl.SizeCache
}

func (x *ModelSourceCheckpoint) Reset() {
	*x = ModelSourceCheckpoint{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[35]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *ModelSourceCheckpoint) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*ModelSourceCheckpoint) ProtoMessage() {}

func (x *ModelSourceCheckpoint) ProtoReflect() protoreflect.Message {
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

// Deprecated: Use ModelSourceCheckpoint.ProtoReflect.Descriptor instead.
func (*ModelSourceCheckpoint) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{35}
}

func (x *ModelSourceCheckpoint) GetSlot() string {
	if x != nil {
		return x.Slot
	}
	return ""
}

func (x *ModelSourceCheckpoint) GetHead() *Ref {
	if x != nil {
		return x.Head
	}
	return nil
}

func (x *ModelSourceCheckpoint) GetPlanDigest() []byte {
	if x != nil {
		return x.PlanDigest
	}
	return nil
}

func (x *ModelSourceCheckpoint) GetIndex() uint64 {
	if x != nil {
		return x.Index
	}
	return 0
}

func (x *ModelSourceCheckpoint) GetBytes() uint64 {
	if x != nil {
		return x.Bytes
	}
	return 0
}

// Source and derived-output authority remain disjoint even though their immutable Link
// transport is identical. The authenticated host validates exactly one subject before
// forwarding; Runtime validates the corresponding operation/slot/plan against TensorFS.
type SourceCheckpointSubject struct {
	state                 protoimpl.MessageState `protogen:"open.v1"`
	OperationId           string                 `protobuf:"bytes,1,opt,name=operation_id,json=operationId,proto3" json:"operation_id,omitempty"`
	SourceSelectionDigest []byte                 `protobuf:"bytes,2,opt,name=source_selection_digest,json=sourceSelectionDigest,proto3" json:"source_selection_digest,omitempty"`
	Slot                  string                 `protobuf:"bytes,3,opt,name=slot,proto3" json:"slot,omitempty"`
	unknownFields         protoimpl.UnknownFields
	sizeCache             protoimpl.SizeCache
}

func (x *SourceCheckpointSubject) Reset() {
	*x = SourceCheckpointSubject{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[36]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *SourceCheckpointSubject) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*SourceCheckpointSubject) ProtoMessage() {}

func (x *SourceCheckpointSubject) ProtoReflect() protoreflect.Message {
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

// Deprecated: Use SourceCheckpointSubject.ProtoReflect.Descriptor instead.
func (*SourceCheckpointSubject) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{36}
}

func (x *SourceCheckpointSubject) GetOperationId() string {
	if x != nil {
		return x.OperationId
	}
	return ""
}

func (x *SourceCheckpointSubject) GetSourceSelectionDigest() []byte {
	if x != nil {
		return x.SourceSelectionDigest
	}
	return nil
}

func (x *SourceCheckpointSubject) GetSlot() string {
	if x != nil {
		return x.Slot
	}
	return ""
}

type WeightsCheckpointSubject struct {
	state                     protoimpl.MessageState `protogen:"open.v1"`
	RequestId                 string                 `protobuf:"bytes,1,opt,name=request_id,json=requestId,proto3" json:"request_id,omitempty"`
	InvocationSpecDigest      []byte                 `protobuf:"bytes,2,opt,name=invocation_spec_digest,json=invocationSpecDigest,proto3" json:"invocation_spec_digest,omitempty"`
	OutputSlot                string                 `protobuf:"bytes,3,opt,name=output_slot,json=outputSlot,proto3" json:"output_slot,omitempty"`
	WeightsTransactionId      string                 `protobuf:"bytes,4,opt,name=weights_transaction_id,json=weightsTransactionId,proto3" json:"weights_transaction_id,omitempty"` // supplied by Runtime, never derived by the host
	WriterEpoch               uint64                 `protobuf:"varint,5,opt,name=writer_epoch,json=writerEpoch,proto3" json:"writer_epoch,omitempty"`
	TensorfsDeclarationDigest []byte                 `protobuf:"bytes,6,opt,name=tensorfs_declaration_digest,json=tensorfsDeclarationDigest,proto3" json:"tensorfs_declaration_digest,omitempty"`
	unknownFields             protoimpl.UnknownFields
	sizeCache                 protoimpl.SizeCache
}

func (x *WeightsCheckpointSubject) Reset() {
	*x = WeightsCheckpointSubject{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[37]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *WeightsCheckpointSubject) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*WeightsCheckpointSubject) ProtoMessage() {}

func (x *WeightsCheckpointSubject) ProtoReflect() protoreflect.Message {
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

// Deprecated: Use WeightsCheckpointSubject.ProtoReflect.Descriptor instead.
func (*WeightsCheckpointSubject) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{37}
}

func (x *WeightsCheckpointSubject) GetRequestId() string {
	if x != nil {
		return x.RequestId
	}
	return ""
}

func (x *WeightsCheckpointSubject) GetInvocationSpecDigest() []byte {
	if x != nil {
		return x.InvocationSpecDigest
	}
	return nil
}

func (x *WeightsCheckpointSubject) GetOutputSlot() string {
	if x != nil {
		return x.OutputSlot
	}
	return ""
}

func (x *WeightsCheckpointSubject) GetWeightsTransactionId() string {
	if x != nil {
		return x.WeightsTransactionId
	}
	return ""
}

func (x *WeightsCheckpointSubject) GetWriterEpoch() uint64 {
	if x != nil {
		return x.WriterEpoch
	}
	return 0
}

func (x *WeightsCheckpointSubject) GetTensorfsDeclarationDigest() []byte {
	if x != nil {
		return x.TensorfsDeclarationDigest
	}
	return nil
}

type CheckpointSubject struct {
	state protoimpl.MessageState `protogen:"open.v1"`
	// Types that are valid to be assigned to Kind:
	//
	//	*CheckpointSubject_Source
	//	*CheckpointSubject_Weights
	Kind          isCheckpointSubject_Kind `protobuf_oneof:"kind"`
	unknownFields protoimpl.UnknownFields
	sizeCache     protoimpl.SizeCache
}

func (x *CheckpointSubject) Reset() {
	*x = CheckpointSubject{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[38]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *CheckpointSubject) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*CheckpointSubject) ProtoMessage() {}

func (x *CheckpointSubject) ProtoReflect() protoreflect.Message {
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

// Deprecated: Use CheckpointSubject.ProtoReflect.Descriptor instead.
func (*CheckpointSubject) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{38}
}

func (x *CheckpointSubject) GetKind() isCheckpointSubject_Kind {
	if x != nil {
		return x.Kind
	}
	return nil
}

func (x *CheckpointSubject) GetSource() *SourceCheckpointSubject {
	if x != nil {
		if x, ok := x.Kind.(*CheckpointSubject_Source); ok {
			return x.Source
		}
	}
	return nil
}

func (x *CheckpointSubject) GetWeights() *WeightsCheckpointSubject {
	if x != nil {
		if x, ok := x.Kind.(*CheckpointSubject_Weights); ok {
			return x.Weights
		}
	}
	return nil
}

type isCheckpointSubject_Kind interface {
	isCheckpointSubject_Kind()
}

type CheckpointSubject_Source struct {
	Source *SourceCheckpointSubject `protobuf:"bytes,1,opt,name=source,proto3,oneof"`
}

type CheckpointSubject_Weights struct {
	Weights *WeightsCheckpointSubject `protobuf:"bytes,2,opt,name=weights,proto3,oneof"`
}

func (*CheckpointSubject_Source) isCheckpointSubject_Kind() {}

func (*CheckpointSubject_Weights) isCheckpointSubject_Kind() {}

type CheckpointObject struct {
	state         protoimpl.MessageState `protogen:"open.v1"`
	Ref           *Ref                   `protobuf:"bytes,1,opt,name=ref,proto3" json:"ref,omitempty"`
	Manifest      bool                   `protobuf:"varint,2,opt,name=manifest,proto3" json:"manifest,omitempty"` // false is a blob; true is the manifest namespace
	unknownFields protoimpl.UnknownFields
	sizeCache     protoimpl.SizeCache
}

func (x *CheckpointObject) Reset() {
	*x = CheckpointObject{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[39]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *CheckpointObject) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*CheckpointObject) ProtoMessage() {}

func (x *CheckpointObject) ProtoReflect() protoreflect.Message {
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

// Deprecated: Use CheckpointObject.ProtoReflect.Descriptor instead.
func (*CheckpointObject) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{39}
}

func (x *CheckpointObject) GetRef() *Ref {
	if x != nil {
		return x.Ref
	}
	return nil
}

func (x *CheckpointObject) GetManifest() bool {
	if x != nil {
		return x.Manifest
	}
	return false
}

// Inspect ONE immutable checkpoint Link, in bounded pages. TensorFS verifies operation,
// slot and plan against the Link; no caller-provided object inventory becomes authority.
type CheckpointPageRequest struct {
	state              protoimpl.MessageState `protogen:"open.v1"`
	RecordOwnerEpoch   uint64                 `protobuf:"varint,1,opt,name=record_owner_epoch,json=recordOwnerEpoch,proto3" json:"record_owner_epoch,omitempty"`
	ControlStreamEpoch uint64                 `protobuf:"varint,2,opt,name=control_stream_epoch,json=controlStreamEpoch,proto3" json:"control_stream_epoch,omitempty"`
	WorkerBootId       string                 `protobuf:"bytes,3,opt,name=worker_boot_id,json=workerBootId,proto3" json:"worker_boot_id,omitempty"`
	Subject            *CheckpointSubject     `protobuf:"bytes,5,opt,name=subject,proto3" json:"subject,omitempty"`
	PlanDigest         []byte                 `protobuf:"bytes,8,opt,name=plan_digest,json=planDigest,proto3" json:"plan_digest,omitempty"`
	Head               *Ref                   `protobuf:"bytes,9,opt,name=head,proto3" json:"head,omitempty"`
	Offset             uint32                 `protobuf:"varint,10,opt,name=offset,proto3" json:"offset,omitempty"`
	Limit              uint32                 `protobuf:"varint,11,opt,name=limit,proto3" json:"limit,omitempty"` // 1..MaxCheckpointObjects
	unknownFields      protoimpl.UnknownFields
	sizeCache          protoimpl.SizeCache
}

func (x *CheckpointPageRequest) Reset() {
	*x = CheckpointPageRequest{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[40]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *CheckpointPageRequest) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*CheckpointPageRequest) ProtoMessage() {}

func (x *CheckpointPageRequest) ProtoReflect() protoreflect.Message {
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

// Deprecated: Use CheckpointPageRequest.ProtoReflect.Descriptor instead.
func (*CheckpointPageRequest) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{40}
}

func (x *CheckpointPageRequest) GetRecordOwnerEpoch() uint64 {
	if x != nil {
		return x.RecordOwnerEpoch
	}
	return 0
}

func (x *CheckpointPageRequest) GetControlStreamEpoch() uint64 {
	if x != nil {
		return x.ControlStreamEpoch
	}
	return 0
}

func (x *CheckpointPageRequest) GetWorkerBootId() string {
	if x != nil {
		return x.WorkerBootId
	}
	return ""
}

func (x *CheckpointPageRequest) GetSubject() *CheckpointSubject {
	if x != nil {
		return x.Subject
	}
	return nil
}

func (x *CheckpointPageRequest) GetPlanDigest() []byte {
	if x != nil {
		return x.PlanDigest
	}
	return nil
}

func (x *CheckpointPageRequest) GetHead() *Ref {
	if x != nil {
		return x.Head
	}
	return nil
}

func (x *CheckpointPageRequest) GetOffset() uint32 {
	if x != nil {
		return x.Offset
	}
	return 0
}

func (x *CheckpointPageRequest) GetLimit() uint32 {
	if x != nil {
		return x.Limit
	}
	return 0
}

type CheckpointPageResult struct {
	state              protoimpl.MessageState `protogen:"open.v1"`
	RecordOwnerEpoch   uint64                 `protobuf:"varint,1,opt,name=record_owner_epoch,json=recordOwnerEpoch,proto3" json:"record_owner_epoch,omitempty"`
	ControlStreamEpoch uint64                 `protobuf:"varint,2,opt,name=control_stream_epoch,json=controlStreamEpoch,proto3" json:"control_stream_epoch,omitempty"`
	WorkerBootId       string                 `protobuf:"bytes,3,opt,name=worker_boot_id,json=workerBootId,proto3" json:"worker_boot_id,omitempty"`
	Subject            *CheckpointSubject     `protobuf:"bytes,5,opt,name=subject,proto3" json:"subject,omitempty"`
	PlanDigest         []byte                 `protobuf:"bytes,8,opt,name=plan_digest,json=planDigest,proto3" json:"plan_digest,omitempty"`
	Head               *Ref                   `protobuf:"bytes,9,opt,name=head,proto3" json:"head,omitempty"`
	Index              uint64                 `protobuf:"varint,10,opt,name=index,proto3" json:"index,omitempty"`
	Bytes              uint64                 `protobuf:"varint,11,opt,name=bytes,proto3" json:"bytes,omitempty"`      // local cumulative checkpointed bytes, not remote custody
	Previous           *Ref                   `protobuf:"bytes,12,opt,name=previous,proto3" json:"previous,omitempty"` // absent at the root Link
	Progress           *Ref                   `protobuf:"bytes,13,opt,name=progress,proto3" json:"progress,omitempty"` // the immutable source journal or derived accepted-part snapshot
	Objects            []*CheckpointObject    `protobuf:"bytes,14,rep,name=objects,proto3" json:"objects,omitempty"`   // <= MaxCheckpointObjects
	NextOffset         uint32                 `protobuf:"varint,15,opt,name=next_offset,json=nextOffset,proto3" json:"next_offset,omitempty"`
	HasMore            bool                   `protobuf:"varint,16,opt,name=has_more,json=hasMore,proto3" json:"has_more,omitempty"`
	SafeCode           string                 `protobuf:"bytes,17,opt,name=safe_code,json=safeCode,proto3" json:"safe_code,omitempty"` // nonempty means refused; no partial success then
	SafeDetail         string                 `protobuf:"bytes,18,opt,name=safe_detail,json=safeDetail,proto3" json:"safe_detail,omitempty"`
	unknownFields      protoimpl.UnknownFields
	sizeCache          protoimpl.SizeCache
}

func (x *CheckpointPageResult) Reset() {
	*x = CheckpointPageResult{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[41]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *CheckpointPageResult) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*CheckpointPageResult) ProtoMessage() {}

func (x *CheckpointPageResult) ProtoReflect() protoreflect.Message {
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

// Deprecated: Use CheckpointPageResult.ProtoReflect.Descriptor instead.
func (*CheckpointPageResult) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{41}
}

func (x *CheckpointPageResult) GetRecordOwnerEpoch() uint64 {
	if x != nil {
		return x.RecordOwnerEpoch
	}
	return 0
}

func (x *CheckpointPageResult) GetControlStreamEpoch() uint64 {
	if x != nil {
		return x.ControlStreamEpoch
	}
	return 0
}

func (x *CheckpointPageResult) GetWorkerBootId() string {
	if x != nil {
		return x.WorkerBootId
	}
	return ""
}

func (x *CheckpointPageResult) GetSubject() *CheckpointSubject {
	if x != nil {
		return x.Subject
	}
	return nil
}

func (x *CheckpointPageResult) GetPlanDigest() []byte {
	if x != nil {
		return x.PlanDigest
	}
	return nil
}

func (x *CheckpointPageResult) GetHead() *Ref {
	if x != nil {
		return x.Head
	}
	return nil
}

func (x *CheckpointPageResult) GetIndex() uint64 {
	if x != nil {
		return x.Index
	}
	return 0
}

func (x *CheckpointPageResult) GetBytes() uint64 {
	if x != nil {
		return x.Bytes
	}
	return 0
}

func (x *CheckpointPageResult) GetPrevious() *Ref {
	if x != nil {
		return x.Previous
	}
	return nil
}

func (x *CheckpointPageResult) GetProgress() *Ref {
	if x != nil {
		return x.Progress
	}
	return nil
}

func (x *CheckpointPageResult) GetObjects() []*CheckpointObject {
	if x != nil {
		return x.Objects
	}
	return nil
}

func (x *CheckpointPageResult) GetNextOffset() uint32 {
	if x != nil {
		return x.NextOffset
	}
	return 0
}

func (x *CheckpointPageResult) GetHasMore() bool {
	if x != nil {
		return x.HasMore
	}
	return false
}

func (x *CheckpointPageResult) GetSafeCode() string {
	if x != nil {
		return x.SafeCode
	}
	return ""
}

func (x *CheckpointPageResult) GetSafeDetail() string {
	if x != nil {
		return x.SafeDetail
	}
	return ""
}

// One grant for one object under an exact checkpoint Link. A GET may first restore the
// named head itself, after which TensorFS validates that Link before exposing its members.
// Every other object must be a member, predecessor or progress ref of the admitted Link.
// Capabilities stay in memory; host/owner ledgers retain only identity and measured results.
type CheckpointTransferRequest struct {
	state              protoimpl.MessageState `protogen:"open.v1"`
	RecordOwnerEpoch   uint64                 `protobuf:"varint,1,opt,name=record_owner_epoch,json=recordOwnerEpoch,proto3" json:"record_owner_epoch,omitempty"`
	ControlStreamEpoch uint64                 `protobuf:"varint,2,opt,name=control_stream_epoch,json=controlStreamEpoch,proto3" json:"control_stream_epoch,omitempty"`
	WorkerBootId       string                 `protobuf:"bytes,3,opt,name=worker_boot_id,json=workerBootId,proto3" json:"worker_boot_id,omitempty"`
	Subject            *CheckpointSubject     `protobuf:"bytes,5,opt,name=subject,proto3" json:"subject,omitempty"`
	PlanDigest         []byte                 `protobuf:"bytes,8,opt,name=plan_digest,json=planDigest,proto3" json:"plan_digest,omitempty"`
	Head               *Ref                   `protobuf:"bytes,9,opt,name=head,proto3" json:"head,omitempty"`
	Object             *CheckpointObject      `protobuf:"bytes,10,opt,name=object,proto3" json:"object,omitempty"`
	TransferId         string                 `protobuf:"bytes,11,opt,name=transfer_id,json=transferId,proto3" json:"transfer_id,omitempty"`
	GrantRevision      uint64                 `protobuf:"varint,12,opt,name=grant_revision,json=grantRevision,proto3" json:"grant_revision,omitempty"`
	// Types that are valid to be assigned to Decision:
	//
	//	*CheckpointTransferRequest_UploadGrant
	//	*CheckpointTransferRequest_DownloadUrl
	Decision      isCheckpointTransferRequest_Decision `protobuf_oneof:"decision"`
	unknownFields protoimpl.UnknownFields
	sizeCache     protoimpl.SizeCache
}

func (x *CheckpointTransferRequest) Reset() {
	*x = CheckpointTransferRequest{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[42]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *CheckpointTransferRequest) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*CheckpointTransferRequest) ProtoMessage() {}

func (x *CheckpointTransferRequest) ProtoReflect() protoreflect.Message {
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

// Deprecated: Use CheckpointTransferRequest.ProtoReflect.Descriptor instead.
func (*CheckpointTransferRequest) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{42}
}

func (x *CheckpointTransferRequest) GetRecordOwnerEpoch() uint64 {
	if x != nil {
		return x.RecordOwnerEpoch
	}
	return 0
}

func (x *CheckpointTransferRequest) GetControlStreamEpoch() uint64 {
	if x != nil {
		return x.ControlStreamEpoch
	}
	return 0
}

func (x *CheckpointTransferRequest) GetWorkerBootId() string {
	if x != nil {
		return x.WorkerBootId
	}
	return ""
}

func (x *CheckpointTransferRequest) GetSubject() *CheckpointSubject {
	if x != nil {
		return x.Subject
	}
	return nil
}

func (x *CheckpointTransferRequest) GetPlanDigest() []byte {
	if x != nil {
		return x.PlanDigest
	}
	return nil
}

func (x *CheckpointTransferRequest) GetHead() *Ref {
	if x != nil {
		return x.Head
	}
	return nil
}

func (x *CheckpointTransferRequest) GetObject() *CheckpointObject {
	if x != nil {
		return x.Object
	}
	return nil
}

func (x *CheckpointTransferRequest) GetTransferId() string {
	if x != nil {
		return x.TransferId
	}
	return ""
}

func (x *CheckpointTransferRequest) GetGrantRevision() uint64 {
	if x != nil {
		return x.GrantRevision
	}
	return 0
}

func (x *CheckpointTransferRequest) GetDecision() isCheckpointTransferRequest_Decision {
	if x != nil {
		return x.Decision
	}
	return nil
}

func (x *CheckpointTransferRequest) GetUploadGrant() *WeightsUploadGrant {
	if x != nil {
		if x, ok := x.Decision.(*CheckpointTransferRequest_UploadGrant); ok {
			return x.UploadGrant
		}
	}
	return nil
}

func (x *CheckpointTransferRequest) GetDownloadUrl() string {
	if x != nil {
		if x, ok := x.Decision.(*CheckpointTransferRequest_DownloadUrl); ok {
			return x.DownloadUrl
		}
	}
	return ""
}

type isCheckpointTransferRequest_Decision interface {
	isCheckpointTransferRequest_Decision()
}

type CheckpointTransferRequest_UploadGrant struct {
	UploadGrant *WeightsUploadGrant `protobuf:"bytes,13,opt,name=upload_grant,json=uploadGrant,proto3,oneof"`
}

type CheckpointTransferRequest_DownloadUrl struct {
	DownloadUrl string `protobuf:"bytes,14,opt,name=download_url,json=downloadUrl,proto3,oneof"` // one scoped GET; digest and length come from object
}

func (*CheckpointTransferRequest_UploadGrant) isCheckpointTransferRequest_Decision() {}

func (*CheckpointTransferRequest_DownloadUrl) isCheckpointTransferRequest_Decision() {}

type CheckpointTransferStatus struct {
	state              protoimpl.MessageState `protogen:"open.v1"`
	RecordOwnerEpoch   uint64                 `protobuf:"varint,1,opt,name=record_owner_epoch,json=recordOwnerEpoch,proto3" json:"record_owner_epoch,omitempty"`
	ControlStreamEpoch uint64                 `protobuf:"varint,2,opt,name=control_stream_epoch,json=controlStreamEpoch,proto3" json:"control_stream_epoch,omitempty"`
	WorkerBootId       string                 `protobuf:"bytes,3,opt,name=worker_boot_id,json=workerBootId,proto3" json:"worker_boot_id,omitempty"`
	Subject            *CheckpointSubject     `protobuf:"bytes,5,opt,name=subject,proto3" json:"subject,omitempty"`
	PlanDigest         []byte                 `protobuf:"bytes,8,opt,name=plan_digest,json=planDigest,proto3" json:"plan_digest,omitempty"`
	Head               *Ref                   `protobuf:"bytes,9,opt,name=head,proto3" json:"head,omitempty"`
	Object             *CheckpointObject      `protobuf:"bytes,10,opt,name=object,proto3" json:"object,omitempty"`
	TransferId         string                 `protobuf:"bytes,11,opt,name=transfer_id,json=transferId,proto3" json:"transfer_id,omitempty"`
	GrantRevision      uint64                 `protobuf:"varint,12,opt,name=grant_revision,json=grantRevision,proto3" json:"grant_revision,omitempty"`
	State              WeightsTransferState   `protobuf:"varint,13,opt,name=state,proto3,enum=cozy.worker.v1.WeightsTransferState" json:"state,omitempty"` // GET success is HELD (verified local admission)
	TransferredBytes   uint64                 `protobuf:"varint,14,opt,name=transferred_bytes,json=transferredBytes,proto3" json:"transferred_bytes,omitempty"`
	HttpStatus         uint32                 `protobuf:"varint,15,opt,name=http_status,json=httpStatus,proto3" json:"http_status,omitempty"`
	ChecksumSha256     string                 `protobuf:"bytes,16,opt,name=checksum_sha256,json=checksumSha256,proto3" json:"checksum_sha256,omitempty"` // digest proven by local admission/read verification
	SafeCode           string                 `protobuf:"bytes,17,opt,name=safe_code,json=safeCode,proto3" json:"safe_code,omitempty"`
	SafeDetail         string                 `protobuf:"bytes,18,opt,name=safe_detail,json=safeDetail,proto3" json:"safe_detail,omitempty"`
	unknownFields      protoimpl.UnknownFields
	sizeCache          protoimpl.SizeCache
}

func (x *CheckpointTransferStatus) Reset() {
	*x = CheckpointTransferStatus{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[43]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *CheckpointTransferStatus) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*CheckpointTransferStatus) ProtoMessage() {}

func (x *CheckpointTransferStatus) ProtoReflect() protoreflect.Message {
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

// Deprecated: Use CheckpointTransferStatus.ProtoReflect.Descriptor instead.
func (*CheckpointTransferStatus) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{43}
}

func (x *CheckpointTransferStatus) GetRecordOwnerEpoch() uint64 {
	if x != nil {
		return x.RecordOwnerEpoch
	}
	return 0
}

func (x *CheckpointTransferStatus) GetControlStreamEpoch() uint64 {
	if x != nil {
		return x.ControlStreamEpoch
	}
	return 0
}

func (x *CheckpointTransferStatus) GetWorkerBootId() string {
	if x != nil {
		return x.WorkerBootId
	}
	return ""
}

func (x *CheckpointTransferStatus) GetSubject() *CheckpointSubject {
	if x != nil {
		return x.Subject
	}
	return nil
}

func (x *CheckpointTransferStatus) GetPlanDigest() []byte {
	if x != nil {
		return x.PlanDigest
	}
	return nil
}

func (x *CheckpointTransferStatus) GetHead() *Ref {
	if x != nil {
		return x.Head
	}
	return nil
}

func (x *CheckpointTransferStatus) GetObject() *CheckpointObject {
	if x != nil {
		return x.Object
	}
	return nil
}

func (x *CheckpointTransferStatus) GetTransferId() string {
	if x != nil {
		return x.TransferId
	}
	return ""
}

func (x *CheckpointTransferStatus) GetGrantRevision() uint64 {
	if x != nil {
		return x.GrantRevision
	}
	return 0
}

func (x *CheckpointTransferStatus) GetState() WeightsTransferState {
	if x != nil {
		return x.State
	}
	return WeightsTransferState_WEIGHTS_TRANSFER_STATE_UNSPECIFIED
}

func (x *CheckpointTransferStatus) GetTransferredBytes() uint64 {
	if x != nil {
		return x.TransferredBytes
	}
	return 0
}

func (x *CheckpointTransferStatus) GetHttpStatus() uint32 {
	if x != nil {
		return x.HttpStatus
	}
	return 0
}

func (x *CheckpointTransferStatus) GetChecksumSha256() string {
	if x != nil {
		return x.ChecksumSha256
	}
	return ""
}

func (x *CheckpointTransferStatus) GetSafeCode() string {
	if x != nil {
		return x.SafeCode
	}
	return ""
}

func (x *CheckpointTransferStatus) GetSafeDetail() string {
	if x != nil {
		return x.SafeDetail
	}
	return ""
}

type PrepareModelSourceRequest struct {
	state                 protoimpl.MessageState   `protogen:"open.v1"`
	OperationId           string                   `protobuf:"bytes,1,opt,name=operation_id,json=operationId,proto3" json:"operation_id,omitempty"`
	SourceSelectionDigest []byte                   `protobuf:"bytes,2,opt,name=source_selection_digest,json=sourceSelectionDigest,proto3" json:"source_selection_digest,omitempty"`
	Profiles              []*ModelSourceProfile    `protobuf:"bytes,3,rep,name=profiles,proto3" json:"profiles,omitempty"`                                                         // sorted unique by slot; <= MaxModelSourceProfiles
	Files                 []*LocalModelSourceFile  `protobuf:"bytes,4,rep,name=files,proto3" json:"files,omitempty"`                                                               // sorted unique by member; <= MaxModelSourceFiles
	SourceUri             string                   `protobuf:"bytes,5,opt,name=source_uri,json=sourceUri,proto3" json:"source_uri,omitempty"`                                      // credential-free pinned hf:// or civitai:// provenance
	DeclaredLicense       string                   `protobuf:"bytes,6,opt,name=declared_license,json=declaredLicense,proto3" json:"declared_license,omitempty"`                    // bounded printable provenance; empty means undeclared
	Checkpoints           []*ModelSourceCheckpoint `protobuf:"bytes,7,rep,name=checkpoints,proto3" json:"checkpoints,omitempty"`                                                   // sorted unique by slot; admitted restore heads
	AdoptFromOperationId  string                   `protobuf:"bytes,8,opt,name=adopt_from_operation_id,json=adoptFromOperationId,proto3" json:"adopt_from_operation_id,omitempty"` // signed Host adoption only; never an executor choice
	unknownFields         protoimpl.UnknownFields
	sizeCache             protoimpl.SizeCache
}

func (x *PrepareModelSourceRequest) Reset() {
	*x = PrepareModelSourceRequest{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[44]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *PrepareModelSourceRequest) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*PrepareModelSourceRequest) ProtoMessage() {}

func (x *PrepareModelSourceRequest) ProtoReflect() protoreflect.Message {
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

// Deprecated: Use PrepareModelSourceRequest.ProtoReflect.Descriptor instead.
func (*PrepareModelSourceRequest) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{44}
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

func (x *PrepareModelSourceRequest) GetCheckpoints() []*ModelSourceCheckpoint {
	if x != nil {
		return x.Checkpoints
	}
	return nil
}

func (x *PrepareModelSourceRequest) GetAdoptFromOperationId() string {
	if x != nil {
		return x.AdoptFromOperationId
	}
	return ""
}

type PrepareModelSourceResult struct {
	state         protoimpl.MessageState    `protogen:"open.v1"`
	Outcome       ModelSourcePrepareOutcome `protobuf:"varint,1,opt,name=outcome,proto3,enum=cozy.worker.v1.ModelSourcePrepareOutcome" json:"outcome,omitempty"`
	Sources       []*PreparedModelSource    `protobuf:"bytes,2,rep,name=sources,proto3" json:"sources,omitempty"` // sorted unique completed slots; exact set when complete
	SafeCode      string                    `protobuf:"bytes,3,opt,name=safe_code,json=safeCode,proto3" json:"safe_code,omitempty"`
	SafeDetail    string                    `protobuf:"bytes,4,opt,name=safe_detail,json=safeDetail,proto3" json:"safe_detail,omitempty"`
	Checkpoints   []*ModelSourceCheckpoint  `protobuf:"bytes,5,rep,name=checkpoints,proto3" json:"checkpoints,omitempty"`                       // sorted unique by slot, <= MaxModelSourceProfiles
	SpentMembers  []string                  `protobuf:"bytes,6,rep,name=spent_members,json=spentMembers,proto3" json:"spent_members,omitempty"` // sorted unique verified carriers no remaining conversion uses
	unknownFields protoimpl.UnknownFields
	sizeCache     protoimpl.SizeCache
}

func (x *PrepareModelSourceResult) Reset() {
	*x = PrepareModelSourceResult{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[45]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *PrepareModelSourceResult) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*PrepareModelSourceResult) ProtoMessage() {}

func (x *PrepareModelSourceResult) ProtoReflect() protoreflect.Message {
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

// Deprecated: Use PrepareModelSourceResult.ProtoReflect.Descriptor instead.
func (*PrepareModelSourceResult) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{45}
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

func (x *PrepareModelSourceResult) GetCheckpoints() []*ModelSourceCheckpoint {
	if x != nil {
		return x.Checkpoints
	}
	return nil
}

func (x *PrepareModelSourceResult) GetSpentMembers() []string {
	if x != nil {
		return x.SpentMembers
	}
	return nil
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
	//	*RecordOwnerFrame_WeightsFinalizeRequest
	//	*RecordOwnerFrame_WeightsHostAck
	//	*RecordOwnerFrame_WeightsTransferRequest
	//	*RecordOwnerFrame_WeightsReadRequest
	//	*RecordOwnerFrame_ModelSourceFileRequest
	//	*RecordOwnerFrame_ModelSourcePrepareRequest
	//	*RecordOwnerFrame_LocalPackageAbort
	//	*RecordOwnerFrame_WeightsUploadRequest
	//	*RecordOwnerFrame_LocalPackageFetchRequest
	//	*RecordOwnerFrame_ChildCallResult
	Msg           isRecordOwnerFrame_Msg `protobuf_oneof:"msg"`
	unknownFields protoimpl.UnknownFields
	sizeCache     protoimpl.SizeCache
}

func (x *RecordOwnerFrame) Reset() {
	*x = RecordOwnerFrame{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[46]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *RecordOwnerFrame) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*RecordOwnerFrame) ProtoMessage() {}

func (x *RecordOwnerFrame) ProtoReflect() protoreflect.Message {
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

// Deprecated: Use RecordOwnerFrame.ProtoReflect.Descriptor instead.
func (*RecordOwnerFrame) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{46}
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

func (x *RecordOwnerFrame) GetWeightsFinalizeRequest() *WeightsFinalizeRequest {
	if x != nil {
		if x, ok := x.Msg.(*RecordOwnerFrame_WeightsFinalizeRequest); ok {
			return x.WeightsFinalizeRequest
		}
	}
	return nil
}

func (x *RecordOwnerFrame) GetWeightsHostAck() *WeightsHostAck {
	if x != nil {
		if x, ok := x.Msg.(*RecordOwnerFrame_WeightsHostAck); ok {
			return x.WeightsHostAck
		}
	}
	return nil
}

func (x *RecordOwnerFrame) GetWeightsTransferRequest() *WeightsTransferRequest {
	if x != nil {
		if x, ok := x.Msg.(*RecordOwnerFrame_WeightsTransferRequest); ok {
			return x.WeightsTransferRequest
		}
	}
	return nil
}

func (x *RecordOwnerFrame) GetWeightsReadRequest() *WeightsReadRequest {
	if x != nil {
		if x, ok := x.Msg.(*RecordOwnerFrame_WeightsReadRequest); ok {
			return x.WeightsReadRequest
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

func (x *RecordOwnerFrame) GetLocalPackageAbort() *LocalPackageAbort {
	if x != nil {
		if x, ok := x.Msg.(*RecordOwnerFrame_LocalPackageAbort); ok {
			return x.LocalPackageAbort
		}
	}
	return nil
}

func (x *RecordOwnerFrame) GetWeightsUploadRequest() *WeightsUploadRequest {
	if x != nil {
		if x, ok := x.Msg.(*RecordOwnerFrame_WeightsUploadRequest); ok {
			return x.WeightsUploadRequest
		}
	}
	return nil
}

func (x *RecordOwnerFrame) GetLocalPackageFetchRequest() *LocalPackageFetchRequest {
	if x != nil {
		if x, ok := x.Msg.(*RecordOwnerFrame_LocalPackageFetchRequest); ok {
			return x.LocalPackageFetchRequest
		}
	}
	return nil
}

func (x *RecordOwnerFrame) GetChildCallResult() *ChildCallResult {
	if x != nil {
		if x, ok := x.Msg.(*RecordOwnerFrame_ChildCallResult); ok {
			return x.ChildCallResult
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

type RecordOwnerFrame_WeightsFinalizeRequest struct {
	WeightsFinalizeRequest *WeightsFinalizeRequest `protobuf:"bytes,17,opt,name=weights_finalize_request,json=weightsFinalizeRequest,proto3,oneof"`
}

type RecordOwnerFrame_WeightsHostAck struct {
	// RETIRING (proto-025): slots 19-27 below are host lanes. Their minor-21 forms are the
	// PodHost calls (owner -> host) and RuntimeWeights (host -> Runtime); they are reserved at
	// the first minor after every consumer is off them. The owner form of weights_read_request
	// (21) has no owner today and no minor-21 form; it returns additively when one needs it.
	// pod-supervisor -> Runtime injection. An external RecordOwner MUST NOT author this frame.
	WeightsHostAck *WeightsHostAck `protobuf:"bytes,19,opt,name=weights_host_ack,json=weightsHostAck,proto3,oneof"`
}

type RecordOwnerFrame_WeightsTransferRequest struct {
	// External RecordOwner -> pod-supervisor only. It MUST NOT be forwarded to Runtime.
	WeightsTransferRequest *WeightsTransferRequest `protobuf:"bytes,20,opt,name=weights_transfer_request,json=weightsTransferRequest,proto3,oneof"`
}

type RecordOwnerFrame_WeightsReadRequest struct {
	// Authenticated external RecordOwner -> pod-supervisor, or pod-supervisor -> Runtime.
	// The supervisor validates the exact durable receipt/object/range before forwarding.
	WeightsReadRequest *WeightsReadRequest `protobuf:"bytes,21,opt,name=weights_read_request,json=weightsReadRequest,proto3,oneof"`
}

type RecordOwnerFrame_ModelSourceFileRequest struct {
	ModelSourceFileRequest *ModelSourceFileRequest `protobuf:"bytes,22,opt,name=model_source_file_request,json=modelSourceFileRequest,proto3,oneof"` // external owner -> supervisor only
}

type RecordOwnerFrame_ModelSourcePrepareRequest struct {
	ModelSourcePrepareRequest *ModelSourcePrepareRequest `protobuf:"bytes,23,opt,name=model_source_prepare_request,json=modelSourcePrepareRequest,proto3,oneof"` // owner -> supervisor only
}

type RecordOwnerFrame_LocalPackageAbort struct {
	LocalPackageAbort *LocalPackageAbort `protobuf:"bytes,25,opt,name=local_package_abort,json=localPackageAbort,proto3,oneof"` // external owner -> supervisor only
}

type RecordOwnerFrame_WeightsUploadRequest struct {
	// pod-supervisor -> Runtime injection. An external RecordOwner MUST NOT author this frame;
	// the supervisor mints it only after binding the transfer to its durable receipt.
	WeightsUploadRequest *WeightsUploadRequest `protobuf:"bytes,26,opt,name=weights_upload_request,json=weightsUploadRequest,proto3,oneof"`
}

type RecordOwnerFrame_LocalPackageFetchRequest struct {
	LocalPackageFetchRequest *LocalPackageFetchRequest `protobuf:"bytes,27,opt,name=local_package_fetch_request,json=localPackageFetchRequest,proto3,oneof"` // external owner -> supervisor
}

type RecordOwnerFrame_ChildCallResult struct {
	ChildCallResult *ChildCallResult `protobuf:"bytes,28,opt,name=child_call_result,json=childCallResult,proto3,oneof"` // authenticated RecordOwner -> parent Runtime
}

func (*RecordOwnerFrame_Claim) isRecordOwnerFrame_Msg() {}

func (*RecordOwnerFrame_DesiredState) isRecordOwnerFrame_Msg() {}

func (*RecordOwnerFrame_AttemptOffer) isRecordOwnerFrame_Msg() {}

func (*RecordOwnerFrame_CancelAttempt) isRecordOwnerFrame_Msg() {}

func (*RecordOwnerFrame_OutcomeAck) isRecordOwnerFrame_Msg() {}

func (*RecordOwnerFrame_CheckpointReceipt) isRecordOwnerFrame_Msg() {}

func (*RecordOwnerFrame_SnapshotAck) isRecordOwnerFrame_Msg() {}

func (*RecordOwnerFrame_WeightsFinalizeRequest) isRecordOwnerFrame_Msg() {}

func (*RecordOwnerFrame_WeightsHostAck) isRecordOwnerFrame_Msg() {}

func (*RecordOwnerFrame_WeightsTransferRequest) isRecordOwnerFrame_Msg() {}

func (*RecordOwnerFrame_WeightsReadRequest) isRecordOwnerFrame_Msg() {}

func (*RecordOwnerFrame_ModelSourceFileRequest) isRecordOwnerFrame_Msg() {}

func (*RecordOwnerFrame_ModelSourcePrepareRequest) isRecordOwnerFrame_Msg() {}

func (*RecordOwnerFrame_LocalPackageAbort) isRecordOwnerFrame_Msg() {}

func (*RecordOwnerFrame_WeightsUploadRequest) isRecordOwnerFrame_Msg() {}

func (*RecordOwnerFrame_LocalPackageFetchRequest) isRecordOwnerFrame_Msg() {}

func (*RecordOwnerFrame_ChildCallResult) isRecordOwnerFrame_Msg() {}

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
	//	*WorkerFrame_WeightsFinalizeResult
	//	*WorkerFrame_WeightsIntent
	//	*WorkerFrame_WeightsReceipt
	//	*WorkerFrame_WeightsReadResult
	//	*WorkerFrame_WeightsTransferStatus
	//	*WorkerFrame_ModelSourceFileStatus
	//	*WorkerFrame_ModelSourcePrepared
	//	*WorkerFrame_LocalPackageFileStatus
	//	*WorkerFrame_LocalPackageAbortStatus
	//	*WorkerFrame_WeightsUploadResult
	//	*WorkerFrame_WeightsCheckpoint
	//	*WorkerFrame_WeightsTransaction
	//	*WorkerFrame_ChildCallRequest
	//	*WorkerFrame_ChildCallCancel
	Msg           isWorkerFrame_Msg `protobuf_oneof:"msg"`
	unknownFields protoimpl.UnknownFields
	sizeCache     protoimpl.SizeCache
}

func (x *WorkerFrame) Reset() {
	*x = WorkerFrame{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[47]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *WorkerFrame) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*WorkerFrame) ProtoMessage() {}

func (x *WorkerFrame) ProtoReflect() protoreflect.Message {
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

// Deprecated: Use WorkerFrame.ProtoReflect.Descriptor instead.
func (*WorkerFrame) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{47}
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

func (x *WorkerFrame) GetWeightsFinalizeResult() *WeightsFinalizeResult {
	if x != nil {
		if x, ok := x.Msg.(*WorkerFrame_WeightsFinalizeResult); ok {
			return x.WeightsFinalizeResult
		}
	}
	return nil
}

func (x *WorkerFrame) GetWeightsIntent() *WeightsIntentFrame {
	if x != nil {
		if x, ok := x.Msg.(*WorkerFrame_WeightsIntent); ok {
			return x.WeightsIntent
		}
	}
	return nil
}

func (x *WorkerFrame) GetWeightsReceipt() *WeightsReceiptFrame {
	if x != nil {
		if x, ok := x.Msg.(*WorkerFrame_WeightsReceipt); ok {
			return x.WeightsReceipt
		}
	}
	return nil
}

func (x *WorkerFrame) GetWeightsReadResult() *WeightsReadResult {
	if x != nil {
		if x, ok := x.Msg.(*WorkerFrame_WeightsReadResult); ok {
			return x.WeightsReadResult
		}
	}
	return nil
}

func (x *WorkerFrame) GetWeightsTransferStatus() *WeightsTransferStatus {
	if x != nil {
		if x, ok := x.Msg.(*WorkerFrame_WeightsTransferStatus); ok {
			return x.WeightsTransferStatus
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

func (x *WorkerFrame) GetLocalPackageFileStatus() *LocalPackageFileStatus {
	if x != nil {
		if x, ok := x.Msg.(*WorkerFrame_LocalPackageFileStatus); ok {
			return x.LocalPackageFileStatus
		}
	}
	return nil
}

func (x *WorkerFrame) GetLocalPackageAbortStatus() *LocalPackageAbortStatus {
	if x != nil {
		if x, ok := x.Msg.(*WorkerFrame_LocalPackageAbortStatus); ok {
			return x.LocalPackageAbortStatus
		}
	}
	return nil
}

func (x *WorkerFrame) GetWeightsUploadResult() *WeightsUploadResult {
	if x != nil {
		if x, ok := x.Msg.(*WorkerFrame_WeightsUploadResult); ok {
			return x.WeightsUploadResult
		}
	}
	return nil
}

func (x *WorkerFrame) GetWeightsCheckpoint() *WeightsCheckpointFrame {
	if x != nil {
		if x, ok := x.Msg.(*WorkerFrame_WeightsCheckpoint); ok {
			return x.WeightsCheckpoint
		}
	}
	return nil
}

func (x *WorkerFrame) GetWeightsTransaction() *WeightsTransactionStatus {
	if x != nil {
		if x, ok := x.Msg.(*WorkerFrame_WeightsTransaction); ok {
			return x.WeightsTransaction
		}
	}
	return nil
}

func (x *WorkerFrame) GetChildCallRequest() *ChildCallRequest {
	if x != nil {
		if x, ok := x.Msg.(*WorkerFrame_ChildCallRequest); ok {
			return x.ChildCallRequest
		}
	}
	return nil
}

func (x *WorkerFrame) GetChildCallCancel() *ChildCallCancel {
	if x != nil {
		if x, ok := x.Msg.(*WorkerFrame_ChildCallCancel); ok {
			return x.ChildCallCancel
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

type WorkerFrame_WeightsFinalizeResult struct {
	WeightsFinalizeResult *WeightsFinalizeResult `protobuf:"bytes,16,opt,name=weights_finalize_result,json=weightsFinalizeResult,proto3,oneof"`
}

type WorkerFrame_WeightsIntent struct {
	// RETIRING (proto-025): slots 17 and 19-25 below are host lanes whose minor-21 forms are
	// RuntimeWeights events and PodHost answers; they are reserved at the first minor after
	// every consumer is off them. weights_receipt (18) STAYS: the host forwards it to the owner
	// after its durable act and replays it after the snapshot barrier, exactly like an outcome.
	// Runtime -> pod-supervisor; forwarded only after the supervisor's durable host act.
	WeightsIntent *WeightsIntentFrame `protobuf:"bytes,17,opt,name=weights_intent,json=weightsIntent,proto3,oneof"`
}

type WorkerFrame_WeightsReceipt struct {
	WeightsReceipt *WeightsReceiptFrame `protobuf:"bytes,18,opt,name=weights_receipt,json=weightsReceipt,proto3,oneof"`
}

type WorkerFrame_WeightsReadResult struct {
	// Runtime -> pod-supervisor. The supervisor consumes an internal read result or forwards an
	// exact externally requested result to the authenticated RecordOwner that owns read_id.
	WeightsReadResult *WeightsReadResult `protobuf:"bytes,19,opt,name=weights_read_result,json=weightsReadResult,proto3,oneof"`
}

type WorkerFrame_WeightsTransferStatus struct {
	// pod-supervisor -> external RecordOwner only. Runtime MUST NOT author it.
	WeightsTransferStatus *WeightsTransferStatus `protobuf:"bytes,20,opt,name=weights_transfer_status,json=weightsTransferStatus,proto3,oneof"`
}

type WorkerFrame_ModelSourceFileStatus struct {
	ModelSourceFileStatus *ModelSourceFileStatus `protobuf:"bytes,21,opt,name=model_source_file_status,json=modelSourceFileStatus,proto3,oneof"` // supervisor -> external owner only
}

type WorkerFrame_ModelSourcePrepared struct {
	ModelSourcePrepared *ModelSourcePrepared `protobuf:"bytes,22,opt,name=model_source_prepared,json=modelSourcePrepared,proto3,oneof"` // supervisor -> external owner only
}

type WorkerFrame_LocalPackageFileStatus struct {
	LocalPackageFileStatus *LocalPackageFileStatus `protobuf:"bytes,23,opt,name=local_package_file_status,json=localPackageFileStatus,proto3,oneof"` // supervisor -> owner only
}

type WorkerFrame_LocalPackageAbortStatus struct {
	LocalPackageAbortStatus *LocalPackageAbortStatus `protobuf:"bytes,24,opt,name=local_package_abort_status,json=localPackageAbortStatus,proto3,oneof"` // supervisor -> owner only
}

type WorkerFrame_WeightsUploadResult struct {
	// Runtime -> pod-supervisor only. It is consumed there and never forwarded; the owner sees
	// the supervisor's durable WeightsTransferStatus instead.
	WeightsUploadResult *WeightsUploadResult `protobuf:"bytes,25,opt,name=weights_upload_result,json=weightsUploadResult,proto3,oneof"`
}

type WorkerFrame_WeightsCheckpoint struct {
	WeightsCheckpoint *WeightsCheckpointFrame `protobuf:"bytes,26,opt,name=weights_checkpoint,json=weightsCheckpoint,proto3,oneof"`
}

type WorkerFrame_WeightsTransaction struct {
	// Host -> authenticated current RecordOwner stream. The same row as its snapshot:
	// exposes the assigned writer epoch before Ready, without forwarding declaration bytes.
	// Observation only; every subsequent operation still validates the current Claim/fence.
	WeightsTransaction *WeightsTransactionStatus `protobuf:"bytes,27,opt,name=weights_transaction,json=weightsTransaction,proto3,oneof"`
}

type WorkerFrame_ChildCallRequest struct {
	ChildCallRequest *ChildCallRequest `protobuf:"bytes,28,opt,name=child_call_request,json=childCallRequest,proto3,oneof"` // Runtime -> Host -> authenticated RecordOwner
}

type WorkerFrame_ChildCallCancel struct {
	ChildCallCancel *ChildCallCancel `protobuf:"bytes,29,opt,name=child_call_cancel,json=childCallCancel,proto3,oneof"` // same parent-attempt authority
}

func (*WorkerFrame_ClaimAck) isWorkerFrame_Msg() {}

func (*WorkerFrame_ObservedState) isWorkerFrame_Msg() {}

func (*WorkerFrame_AttemptAccepted) isWorkerFrame_Msg() {}

func (*WorkerFrame_AttemptOutcome) isWorkerFrame_Msg() {}

func (*WorkerFrame_BootFailure) isWorkerFrame_Msg() {}

func (*WorkerFrame_CheckpointRequest) isWorkerFrame_Msg() {}

func (*WorkerFrame_CheckpointAck) isWorkerFrame_Msg() {}

func (*WorkerFrame_Snapshot) isWorkerFrame_Msg() {}

func (*WorkerFrame_WeightsFinalizeResult) isWorkerFrame_Msg() {}

func (*WorkerFrame_WeightsIntent) isWorkerFrame_Msg() {}

func (*WorkerFrame_WeightsReceipt) isWorkerFrame_Msg() {}

func (*WorkerFrame_WeightsReadResult) isWorkerFrame_Msg() {}

func (*WorkerFrame_WeightsTransferStatus) isWorkerFrame_Msg() {}

func (*WorkerFrame_ModelSourceFileStatus) isWorkerFrame_Msg() {}

func (*WorkerFrame_ModelSourcePrepared) isWorkerFrame_Msg() {}

func (*WorkerFrame_LocalPackageFileStatus) isWorkerFrame_Msg() {}

func (*WorkerFrame_LocalPackageAbortStatus) isWorkerFrame_Msg() {}

func (*WorkerFrame_WeightsUploadResult) isWorkerFrame_Msg() {}

func (*WorkerFrame_WeightsCheckpoint) isWorkerFrame_Msg() {}

func (*WorkerFrame_WeightsTransaction) isWorkerFrame_Msg() {}

func (*WorkerFrame_ChildCallRequest) isWorkerFrame_Msg() {}

func (*WorkerFrame_ChildCallCancel) isWorkerFrame_Msg() {}

// MINOR 40. An ordinary package call, not a graph or a second request protocol.
// The parent request owns the index; this attempt only re-establishes it. The
// owner resolves the exact target from the parent's frozen interface dependency
// join and durably accepts an ordinary child request before dispatching it.
// intent_digest = SHA256(JCS({interface_digest: "sha256:<hex>", module, export,
// request: <decoded request_canonical_bytes>})). It omits parent/attempt clocks;
// the full parent identity below independently fences transport and replay.
type ChildCallRequest struct {
	state                      protoimpl.MessageState `protogen:"open.v1"`
	RecordOwnerEpoch           uint64                 `protobuf:"varint,1,opt,name=record_owner_epoch,json=recordOwnerEpoch,proto3" json:"record_owner_epoch,omitempty"`
	ControlStreamEpoch         uint64                 `protobuf:"varint,2,opt,name=control_stream_epoch,json=controlStreamEpoch,proto3" json:"control_stream_epoch,omitempty"`
	WorkerBootId               string                 `protobuf:"bytes,3,opt,name=worker_boot_id,json=workerBootId,proto3" json:"worker_boot_id,omitempty"`
	ParentRequestId            string                 `protobuf:"bytes,4,opt,name=parent_request_id,json=parentRequestId,proto3" json:"parent_request_id,omitempty"`
	ParentAttemptOrdinal       uint64                 `protobuf:"varint,5,opt,name=parent_attempt_ordinal,json=parentAttemptOrdinal,proto3" json:"parent_attempt_ordinal,omitempty"`
	ParentInvocationSpecDigest []byte                 `protobuf:"bytes,6,opt,name=parent_invocation_spec_digest,json=parentInvocationSpecDigest,proto3" json:"parent_invocation_spec_digest,omitempty"`
	CallIndex                  uint32                 `protobuf:"varint,7,opt,name=call_index,json=callIndex,proto3" json:"call_index,omitempty"` // zero-based, less than 32
	InterfaceDigest            []byte                 `protobuf:"bytes,8,opt,name=interface_digest,json=interfaceDigest,proto3" json:"interface_digest,omitempty"`
	Module                     string                 `protobuf:"bytes,9,opt,name=module,proto3" json:"module,omitempty"`
	Export                     string                 `protobuf:"bytes,10,opt,name=export,proto3" json:"export,omitempty"`
	RequestCanonicalBytes      []byte                 `protobuf:"bytes,11,opt,name=request_canonical_bytes,json=requestCanonicalBytes,proto3" json:"request_canonical_bytes,omitempty"` // bounded 48 KiB; typed refs, never inline artifacts
	IntentDigest               []byte                 `protobuf:"bytes,12,opt,name=intent_digest,json=intentDigest,proto3" json:"intent_digest,omitempty"`
	unknownFields              protoimpl.UnknownFields
	sizeCache                  protoimpl.SizeCache
}

func (x *ChildCallRequest) Reset() {
	*x = ChildCallRequest{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[48]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *ChildCallRequest) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*ChildCallRequest) ProtoMessage() {}

func (x *ChildCallRequest) ProtoReflect() protoreflect.Message {
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

// Deprecated: Use ChildCallRequest.ProtoReflect.Descriptor instead.
func (*ChildCallRequest) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{48}
}

func (x *ChildCallRequest) GetRecordOwnerEpoch() uint64 {
	if x != nil {
		return x.RecordOwnerEpoch
	}
	return 0
}

func (x *ChildCallRequest) GetControlStreamEpoch() uint64 {
	if x != nil {
		return x.ControlStreamEpoch
	}
	return 0
}

func (x *ChildCallRequest) GetWorkerBootId() string {
	if x != nil {
		return x.WorkerBootId
	}
	return ""
}

func (x *ChildCallRequest) GetParentRequestId() string {
	if x != nil {
		return x.ParentRequestId
	}
	return ""
}

func (x *ChildCallRequest) GetParentAttemptOrdinal() uint64 {
	if x != nil {
		return x.ParentAttemptOrdinal
	}
	return 0
}

func (x *ChildCallRequest) GetParentInvocationSpecDigest() []byte {
	if x != nil {
		return x.ParentInvocationSpecDigest
	}
	return nil
}

func (x *ChildCallRequest) GetCallIndex() uint32 {
	if x != nil {
		return x.CallIndex
	}
	return 0
}

func (x *ChildCallRequest) GetInterfaceDigest() []byte {
	if x != nil {
		return x.InterfaceDigest
	}
	return nil
}

func (x *ChildCallRequest) GetModule() string {
	if x != nil {
		return x.Module
	}
	return ""
}

func (x *ChildCallRequest) GetExport() string {
	if x != nil {
		return x.Export
	}
	return ""
}

func (x *ChildCallRequest) GetRequestCanonicalBytes() []byte {
	if x != nil {
		return x.RequestCanonicalBytes
	}
	return nil
}

func (x *ChildCallRequest) GetIntentDigest() []byte {
	if x != nil {
		return x.IntentDigest
	}
	return nil
}

type ChildCallCancel struct {
	state                      protoimpl.MessageState `protogen:"open.v1"`
	RecordOwnerEpoch           uint64                 `protobuf:"varint,1,opt,name=record_owner_epoch,json=recordOwnerEpoch,proto3" json:"record_owner_epoch,omitempty"`
	ControlStreamEpoch         uint64                 `protobuf:"varint,2,opt,name=control_stream_epoch,json=controlStreamEpoch,proto3" json:"control_stream_epoch,omitempty"`
	WorkerBootId               string                 `protobuf:"bytes,3,opt,name=worker_boot_id,json=workerBootId,proto3" json:"worker_boot_id,omitempty"`
	ParentRequestId            string                 `protobuf:"bytes,4,opt,name=parent_request_id,json=parentRequestId,proto3" json:"parent_request_id,omitempty"`
	ParentAttemptOrdinal       uint64                 `protobuf:"varint,5,opt,name=parent_attempt_ordinal,json=parentAttemptOrdinal,proto3" json:"parent_attempt_ordinal,omitempty"`
	ParentInvocationSpecDigest []byte                 `protobuf:"bytes,6,opt,name=parent_invocation_spec_digest,json=parentInvocationSpecDigest,proto3" json:"parent_invocation_spec_digest,omitempty"`
	CallIndex                  uint32                 `protobuf:"varint,7,opt,name=call_index,json=callIndex,proto3" json:"call_index,omitempty"`
	IntentDigest               []byte                 `protobuf:"bytes,8,opt,name=intent_digest,json=intentDigest,proto3" json:"intent_digest,omitempty"`
	unknownFields              protoimpl.UnknownFields
	sizeCache                  protoimpl.SizeCache
}

func (x *ChildCallCancel) Reset() {
	*x = ChildCallCancel{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[49]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *ChildCallCancel) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*ChildCallCancel) ProtoMessage() {}

func (x *ChildCallCancel) ProtoReflect() protoreflect.Message {
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

// Deprecated: Use ChildCallCancel.ProtoReflect.Descriptor instead.
func (*ChildCallCancel) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{49}
}

func (x *ChildCallCancel) GetRecordOwnerEpoch() uint64 {
	if x != nil {
		return x.RecordOwnerEpoch
	}
	return 0
}

func (x *ChildCallCancel) GetControlStreamEpoch() uint64 {
	if x != nil {
		return x.ControlStreamEpoch
	}
	return 0
}

func (x *ChildCallCancel) GetWorkerBootId() string {
	if x != nil {
		return x.WorkerBootId
	}
	return ""
}

func (x *ChildCallCancel) GetParentRequestId() string {
	if x != nil {
		return x.ParentRequestId
	}
	return ""
}

func (x *ChildCallCancel) GetParentAttemptOrdinal() uint64 {
	if x != nil {
		return x.ParentAttemptOrdinal
	}
	return 0
}

func (x *ChildCallCancel) GetParentInvocationSpecDigest() []byte {
	if x != nil {
		return x.ParentInvocationSpecDigest
	}
	return nil
}

func (x *ChildCallCancel) GetCallIndex() uint32 {
	if x != nil {
		return x.CallIndex
	}
	return 0
}

func (x *ChildCallCancel) GetIntentDigest() []byte {
	if x != nil {
		return x.IntentDigest
	}
	return nil
}

type ChildCallResult struct {
	state                      protoimpl.MessageState `protogen:"open.v1"`
	RecordOwnerEpoch           uint64                 `protobuf:"varint,1,opt,name=record_owner_epoch,json=recordOwnerEpoch,proto3" json:"record_owner_epoch,omitempty"`
	ControlStreamEpoch         uint64                 `protobuf:"varint,2,opt,name=control_stream_epoch,json=controlStreamEpoch,proto3" json:"control_stream_epoch,omitempty"`
	WorkerBootId               string                 `protobuf:"bytes,3,opt,name=worker_boot_id,json=workerBootId,proto3" json:"worker_boot_id,omitempty"`
	ParentRequestId            string                 `protobuf:"bytes,4,opt,name=parent_request_id,json=parentRequestId,proto3" json:"parent_request_id,omitempty"`
	ParentAttemptOrdinal       uint64                 `protobuf:"varint,5,opt,name=parent_attempt_ordinal,json=parentAttemptOrdinal,proto3" json:"parent_attempt_ordinal,omitempty"`
	ParentInvocationSpecDigest []byte                 `protobuf:"bytes,6,opt,name=parent_invocation_spec_digest,json=parentInvocationSpecDigest,proto3" json:"parent_invocation_spec_digest,omitempty"`
	CallIndex                  uint32                 `protobuf:"varint,7,opt,name=call_index,json=callIndex,proto3" json:"call_index,omitempty"`
	IntentDigest               []byte                 `protobuf:"bytes,8,opt,name=intent_digest,json=intentDigest,proto3" json:"intent_digest,omitempty"`
	ChildRequestId             string                 `protobuf:"bytes,9,opt,name=child_request_id,json=childRequestId,proto3" json:"child_request_id,omitempty"` // absent only for refusal before durable acceptance
	State                      ChildCallState         `protobuf:"varint,10,opt,name=state,proto3,enum=cozy.worker.v1.ChildCallState" json:"state,omitempty"`
	ResultCanonicalBytes       []byte                 `protobuf:"bytes,11,opt,name=result_canonical_bytes,json=resultCanonicalBytes,proto3" json:"result_canonical_bytes,omitempty"` // SUCCEEDED only; bounded 48 KiB
	SafeCode                   string                 `protobuf:"bytes,12,opt,name=safe_code,json=safeCode,proto3" json:"safe_code,omitempty"`
	SafeDetail                 string                 `protobuf:"bytes,13,opt,name=safe_detail,json=safeDetail,proto3" json:"safe_detail,omitempty"` // bounded 1024 bytes; no remote traceback
	unknownFields              protoimpl.UnknownFields
	sizeCache                  protoimpl.SizeCache
}

func (x *ChildCallResult) Reset() {
	*x = ChildCallResult{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[50]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *ChildCallResult) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*ChildCallResult) ProtoMessage() {}

func (x *ChildCallResult) ProtoReflect() protoreflect.Message {
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

// Deprecated: Use ChildCallResult.ProtoReflect.Descriptor instead.
func (*ChildCallResult) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{50}
}

func (x *ChildCallResult) GetRecordOwnerEpoch() uint64 {
	if x != nil {
		return x.RecordOwnerEpoch
	}
	return 0
}

func (x *ChildCallResult) GetControlStreamEpoch() uint64 {
	if x != nil {
		return x.ControlStreamEpoch
	}
	return 0
}

func (x *ChildCallResult) GetWorkerBootId() string {
	if x != nil {
		return x.WorkerBootId
	}
	return ""
}

func (x *ChildCallResult) GetParentRequestId() string {
	if x != nil {
		return x.ParentRequestId
	}
	return ""
}

func (x *ChildCallResult) GetParentAttemptOrdinal() uint64 {
	if x != nil {
		return x.ParentAttemptOrdinal
	}
	return 0
}

func (x *ChildCallResult) GetParentInvocationSpecDigest() []byte {
	if x != nil {
		return x.ParentInvocationSpecDigest
	}
	return nil
}

func (x *ChildCallResult) GetCallIndex() uint32 {
	if x != nil {
		return x.CallIndex
	}
	return 0
}

func (x *ChildCallResult) GetIntentDigest() []byte {
	if x != nil {
		return x.IntentDigest
	}
	return nil
}

func (x *ChildCallResult) GetChildRequestId() string {
	if x != nil {
		return x.ChildRequestId
	}
	return ""
}

func (x *ChildCallResult) GetState() ChildCallState {
	if x != nil {
		return x.State
	}
	return ChildCallState_CHILD_CALL_STATE_UNSPECIFIED
}

func (x *ChildCallResult) GetResultCanonicalBytes() []byte {
	if x != nil {
		return x.ResultCanonicalBytes
	}
	return nil
}

func (x *ChildCallResult) GetSafeCode() string {
	if x != nil {
		return x.SafeCode
	}
	return ""
}

func (x *ChildCallResult) GetSafeDetail() string {
	if x != nil {
		return x.SafeDetail
	}
	return ""
}

type Claim struct {
	state              protoimpl.MessageState `protogen:"open.v1"`
	RecordOwnerEpoch   uint64                 `protobuf:"varint,1,opt,name=record_owner_epoch,json=recordOwnerEpoch,proto3" json:"record_owner_epoch,omitempty"`       // envelope; the claimed authority epoch
	ControlStreamEpoch uint64                 `protobuf:"varint,2,opt,name=control_stream_epoch,json=controlStreamEpoch,proto3" json:"control_stream_epoch,omitempty"` // 0 on Claim (none minted yet)
	WorkerBootId       string                 `protobuf:"bytes,3,opt,name=worker_boot_id,json=workerBootId,proto3" json:"worker_boot_id,omitempty"`                    // empty on a fresh dial; set when re-dialing a known boot
	RecordOwnerId      string                 `protobuf:"bytes,5,opt,name=record_owner_id,json=recordOwnerId,proto3" json:"record_owner_id,omitempty"`                 // stable identity of the claiming RecordOwner
	WorkerId           string                 `protobuf:"bytes,6,opt,name=worker_id,json=workerId,proto3" json:"worker_id,omitempty"`                                  // the worker identity the RecordOwner expects to be claiming
	WireMinor          uint32                 `protobuf:"varint,7,opt,name=wire_minor,json=wireMinor,proto3" json:"wire_minor,omitempty"`                              // highest minor the RECORDOWNER implements
	Proof              []byte                 `protobuf:"bytes,8,opt,name=proof,proto3" json:"proof,omitempty"`                                                        // 64-byte Ed25519 signature over canonical ClaimProof/1 on
	unknownFields      protoimpl.UnknownFields
	sizeCache          protoimpl.SizeCache
}

func (x *Claim) Reset() {
	*x = Claim{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[51]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *Claim) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*Claim) ProtoMessage() {}

func (x *Claim) ProtoReflect() protoreflect.Message {
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

// Deprecated: Use Claim.ProtoReflect.Descriptor instead.
func (*Claim) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{51}
}

func (x *Claim) GetRecordOwnerEpoch() uint64 {
	if x != nil {
		return x.RecordOwnerEpoch
	}
	return 0
}

func (x *Claim) GetControlStreamEpoch() uint64 {
	if x != nil {
		return x.ControlStreamEpoch
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
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[52]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *ClaimProof) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*ClaimProof) ProtoMessage() {}

func (x *ClaimProof) ProtoReflect() protoreflect.Message {
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

// Deprecated: Use ClaimProof.ProtoReflect.Descriptor instead.
func (*ClaimProof) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{52}
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
// is closed answers UNDURABLE without minting a control_stream_epoch. On acceptance the worker
// fences the previous stream (if any), increments control_stream_epoch, and answers with its
// identity + one WorkerSnapshot. DISPATCH STAYS CLOSED until the RecordOwner's SnapshotAck
// (02 §6) — admission_state reports CLOSED until then.
type ClaimAck struct {
	state              protoimpl.MessageState `protogen:"open.v1"`
	RecordOwnerEpoch   uint64                 `protobuf:"varint,1,opt,name=record_owner_epoch,json=recordOwnerEpoch,proto3" json:"record_owner_epoch,omitempty"`       // echo of the accepted epoch
	ControlStreamEpoch uint64                 `protobuf:"varint,2,opt,name=control_stream_epoch,json=controlStreamEpoch,proto3" json:"control_stream_epoch,omitempty"` // newly minted for this stream
	WorkerBootId       string                 `protobuf:"bytes,3,opt,name=worker_boot_id,json=workerBootId,proto3" json:"worker_boot_id,omitempty"`                    // this boot's identity
	Accepted           bool                   `protobuf:"varint,5,opt,name=accepted,proto3" json:"accepted,omitempty"`                                                 // false => `rejection` set; stream then closes
	Rejection          ClaimRejection         `protobuf:"varint,6,opt,name=rejection,proto3,enum=cozy.worker.v1.ClaimRejection" json:"rejection,omitempty"`
	WireMinor          uint32                 `protobuf:"varint,7,opt,name=wire_minor,json=wireMinor,proto3" json:"wire_minor,omitempty"` // highest minor the WORKER implements
	WorkerId           string                 `protobuf:"bytes,8,opt,name=worker_id,json=workerId,proto3" json:"worker_id,omitempty"`     // CREDENTIAL subject; cross-checked vs the worker's server
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
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[53]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *ClaimAck) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*ClaimAck) ProtoMessage() {}

func (x *ClaimAck) ProtoReflect() protoreflect.Message {
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

// Deprecated: Use ClaimAck.ProtoReflect.Descriptor instead.
func (*ClaimAck) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{53}
}

func (x *ClaimAck) GetRecordOwnerEpoch() uint64 {
	if x != nil {
		return x.RecordOwnerEpoch
	}
	return 0
}

func (x *ClaimAck) GetControlStreamEpoch() uint64 {
	if x != nil {
		return x.ControlStreamEpoch
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
	state                protoimpl.MessageState `protogen:"open.v1"`
	RecordOwnerEpoch     uint64                 `protobuf:"varint,1,opt,name=record_owner_epoch,json=recordOwnerEpoch,proto3" json:"record_owner_epoch,omitempty"`
	ControlStreamEpoch   uint64                 `protobuf:"varint,2,opt,name=control_stream_epoch,json=controlStreamEpoch,proto3" json:"control_stream_epoch,omitempty"`
	WorkerBootId         string                 `protobuf:"bytes,3,opt,name=worker_boot_id,json=workerBootId,proto3" json:"worker_boot_id,omitempty"`
	WorkerId             string                 `protobuf:"bytes,5,opt,name=worker_id,json=workerId,proto3" json:"worker_id,omitempty"`
	WorkerInstanceId     string                 `protobuf:"bytes,6,opt,name=worker_instance_id,json=workerInstanceId,proto3" json:"worker_instance_id,omitempty"`
	Reason               BootFailureReason      `protobuf:"varint,7,opt,name=reason,proto3,enum=cozy.worker.v1.BootFailureReason" json:"reason,omitempty"`
	Detail               string                 `protobuf:"bytes,8,opt,name=detail,proto3" json:"detail,omitempty"`                                                            // bounded <= 1024 bytes, sanitized
	Resources            *WorkerResources       `protobuf:"bytes,9,opt,name=resources,proto3" json:"resources,omitempty"`                                                      // whatever was measurable
	ControlRuntimeDigest string                 `protobuf:"bytes,10,opt,name=control_runtime_digest,json=controlRuntimeDigest,proto3" json:"control_runtime_digest,omitempty"` // same measured control Runtime provenance as ClaimAck 11
	WorkerReleaseId      string                 `protobuf:"bytes,11,opt,name=worker_release_id,json=workerReleaseId,proto3" json:"worker_release_id,omitempty"`
	unknownFields        protoimpl.UnknownFields
	sizeCache            protoimpl.SizeCache
}

func (x *BootFailure) Reset() {
	*x = BootFailure{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[54]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *BootFailure) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*BootFailure) ProtoMessage() {}

func (x *BootFailure) ProtoReflect() protoreflect.Message {
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

// Deprecated: Use BootFailure.ProtoReflect.Descriptor instead.
func (*BootFailure) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{54}
}

func (x *BootFailure) GetRecordOwnerEpoch() uint64 {
	if x != nil {
		return x.RecordOwnerEpoch
	}
	return 0
}

func (x *BootFailure) GetControlStreamEpoch() uint64 {
	if x != nil {
		return x.ControlStreamEpoch
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
	ControlStreamEpoch                 uint64                 `protobuf:"varint,2,opt,name=control_stream_epoch,json=controlStreamEpoch,proto3" json:"control_stream_epoch,omitempty"`
	WorkerBootId                       string                 `protobuf:"bytes,3,opt,name=worker_boot_id,json=workerBootId,proto3" json:"worker_boot_id,omitempty"`
	SnapshotId                         string                 `protobuf:"bytes,5,opt,name=snapshot_id,json=snapshotId,proto3" json:"snapshot_id,omitempty"`                                                                               // worker-minted; names this exact snapshot
	SnapshotDigest                     []byte                 `protobuf:"bytes,6,opt,name=snapshot_digest,json=snapshotDigest,proto3" json:"snapshot_digest,omitempty"`                                                                   // class (a): sha256 over EXACTLY the bytes in 7
	SnapshotCanonicalBytes             []byte                 `protobuf:"bytes,7,opt,name=snapshot_canonical_bytes,json=snapshotCanonicalBytes,proto3" json:"snapshot_canonical_bytes,omitempty"`                                         // the WorkerSnapshotBody DOCUMENT, canonical JSON bytes
	AcceptedPlacementSetCanonicalBytes []byte                 `protobuf:"bytes,8,opt,name=accepted_placement_set_canonical_bytes,json=acceptedPlacementSetCanonicalBytes,proto3" json:"accepted_placement_set_canonical_bytes,omitempty"` // the EXACT accepted set bytes as journaled
	// — never a re-serialization; its digest lives INSIDE 7
	// THE HOST'S OWN DOCUMENT (proto-025). A canonical document under a digest cannot be appended
	// to -- any addition re-authors it -- so a host standing between the RecordOwner and the
	// worker gets a second document beside the worker's, and 6/7 pass through BYTE-IDENTICAL.
	// Both empty when no host stands between them (local execution).
	HostSnapshotDigest         []byte `protobuf:"bytes,9,opt,name=host_snapshot_digest,json=hostSnapshotDigest,proto3" json:"host_snapshot_digest,omitempty"`                            // class (a): sha256 over EXACTLY the bytes in 10
	HostSnapshotCanonicalBytes []byte `protobuf:"bytes,10,opt,name=host_snapshot_canonical_bytes,json=hostSnapshotCanonicalBytes,proto3" json:"host_snapshot_canonical_bytes,omitempty"` // the HostSnapshotBody DOCUMENT, canonical JSON
	unknownFields              protoimpl.UnknownFields
	sizeCache                  protoimpl.SizeCache
}

func (x *WorkerSnapshot) Reset() {
	*x = WorkerSnapshot{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[55]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *WorkerSnapshot) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*WorkerSnapshot) ProtoMessage() {}

func (x *WorkerSnapshot) ProtoReflect() protoreflect.Message {
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

// Deprecated: Use WorkerSnapshot.ProtoReflect.Descriptor instead.
func (*WorkerSnapshot) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{55}
}

func (x *WorkerSnapshot) GetRecordOwnerEpoch() uint64 {
	if x != nil {
		return x.RecordOwnerEpoch
	}
	return 0
}

func (x *WorkerSnapshot) GetControlStreamEpoch() uint64 {
	if x != nil {
		return x.ControlStreamEpoch
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

func (x *WorkerSnapshot) GetHostSnapshotDigest() []byte {
	if x != nil {
		return x.HostSnapshotDigest
	}
	return nil
}

func (x *WorkerSnapshot) GetHostSnapshotCanonicalBytes() []byte {
	if x != nil {
		return x.HostSnapshotCanonicalBytes
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
// BOUNDEDNESS IS STRUCTURAL, NOT A PROMISE (re-based on phases at minor 23, gpu-hot.md §3):
// the post phase is bounded ONE attempt per lane, so a lane's device takes its next occupant
// only while at most one DEVICE_RELEASED attempt on it is unacked — a RecordOwner that stops
// acking idles its own device rather than growing the worker's held set. Queue room is a
// host-bytes fact (§4) and does not move with acks.
type WorkerSnapshotBody struct {
	state                        protoimpl.MessageState `protogen:"open.v1"`
	AcceptedDesiredStateRevision uint64                 `protobuf:"varint,1,opt,name=accepted_desired_state_revision,json=acceptedDesiredStateRevision,proto3" json:"accepted_desired_state_revision,omitempty"` // durably ACCEPTED intent
	AcceptedPlacementSetDigest   []byte                 `protobuf:"bytes,2,opt,name=accepted_placement_set_digest,json=acceptedPlacementSetDigest,proto3" json:"accepted_placement_set_digest,omitempty"`        // class (a); equals sha256 of the bytes the
	// enclosing WorkerSnapshot carries in field 8
	JournalHighwater      uint64             `protobuf:"varint,3,opt,name=journal_highwater,json=journalHighwater,proto3" json:"journal_highwater,omitempty"`                  // the journal position this snapshot reflects
	WorkerPhase           WorkerPhase        `protobuf:"varint,4,opt,name=worker_phase,json=workerPhase,proto3,enum=cozy.worker.v1.WorkerPhase" json:"worker_phase,omitempty"` // machine lifecycle
	Placements            []*PlacementStatus `protobuf:"bytes,5,rep,name=placements,proto3" json:"placements,omitempty"`                                                       // sorted by placement_id
	ConvergedRevision     uint64             `protobuf:"varint,6,opt,name=converged_revision,json=convergedRevision,proto3" json:"converged_revision,omitempty"`               // advances ONLY when observed satisfies accepted
	AdmissionEpoch        uint64             `protobuf:"varint,7,opt,name=admission_epoch,json=admissionEpoch,proto3" json:"admission_epoch,omitempty"`
	AdmissionState        AdmissionState     `protobuf:"varint,8,opt,name=admission_state,json=admissionState,proto3,enum=cozy.worker.v1.AdmissionState" json:"admission_state,omitempty"`
	AvailableAttemptSlots uint32             `protobuf:"varint,9,opt,name=available_attempt_slots,json=availableAttemptSlots,proto3" json:"available_attempt_slots,omitempty"` // Σ lanes[].available_attempt_slots: offers the
	// worker will still admit to its queues now
	HeldAttempts []*HeldAttempt `protobuf:"bytes,10,rep,name=held_attempts,json=heldAttempts,proto3" json:"held_attempts,omitempty"` // every accepted attempt from admission to ack
	// (QUEUED, RUNNING, DEVICE_RELEASED,
	// OUTCOME_PENDING_ACK); sorted by
	// (request_id, attempt_ordinal)
	WeightsTransactions []*WeightsTransactionStatus `protobuf:"bytes,11,rep,name=weights_transactions,json=weightsTransactions,proto3" json:"weights_transactions,omitempty"` // RETIRING (proto-025): the
	// host's rows live in HostSnapshotBody/1; no
	// worker writes this. journal_highwater (3) has
	// had no writer since the Runtime journal went.
	// Both are reserved at the retirement minor.
	Lanes []*DeviceLane `protobuf:"bytes,12,rep,name=lanes,proto3" json:"lanes,omitempty"` // proto-024: the lanes as of this snapshot,
	// sorted by lane_id — the per-lane room a
	// reconnecting RecordOwner reconciles from,
	// beside the worker-level sum in 9; each row
	// carries its resident_placement_ids (proto-026)
	HeldManifests []string `protobuf:"bytes,13,rep,name=held_manifests,json=heldManifests,proto3" json:"held_manifests,omitempty"` // proto-026 residency: the same set as
	unknownFields protoimpl.UnknownFields
	sizeCache     protoimpl.SizeCache
}

func (x *WorkerSnapshotBody) Reset() {
	*x = WorkerSnapshotBody{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[56]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *WorkerSnapshotBody) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*WorkerSnapshotBody) ProtoMessage() {}

func (x *WorkerSnapshotBody) ProtoReflect() protoreflect.Message {
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

// Deprecated: Use WorkerSnapshotBody.ProtoReflect.Descriptor instead.
func (*WorkerSnapshotBody) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{56}
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

func (x *WorkerSnapshotBody) GetAdmissionEpoch() uint64 {
	if x != nil {
		return x.AdmissionEpoch
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

func (x *WorkerSnapshotBody) GetWeightsTransactions() []*WeightsTransactionStatus {
	if x != nil {
		return x.WeightsTransactions
	}
	return nil
}

func (x *WorkerSnapshotBody) GetLanes() []*DeviceLane {
	if x != nil {
		return x.Lanes
	}
	return nil
}

func (x *WorkerSnapshotBody) GetHeldManifests() []string {
	if x != nil {
		return x.HeldManifests
	}
	return nil
}

// DOCUMENT SHAPE (not a wire message): canonical form `cozy.worker.v1.HostSnapshotBody/1`, the
// subject of WorkerSnapshot.host_snapshot_digest. Authored by the pod host from its ledger at
// every worker snapshot; never by the worker.
//
// held_outcomes are the ledger's outcomes pending ack that the worker's own held_attempts does
// NOT name -- the survivor set after a child replacement. Each replays as an AttemptOutcome
// after the barrier, exactly like the worker's own. A live child's held_attempts always covers
// the ledger's, so this list is empty until a child is replaced. weights_transactions announce
// the receipt frames the host replays after the barrier. Lanes that moved to PodHost have no
// post-barrier replay and therefore no row here: the owner asks the host.
type HostSnapshotBody struct {
	state        protoimpl.MessageState `protogen:"open.v1"`
	HeldOutcomes []*HeldAttempt         `protobuf:"bytes,1,rep,name=held_outcomes,json=heldOutcomes,proto3" json:"held_outcomes,omitempty"` // sorted by (request_id, attempt_ordinal); state is
	// always OUTCOME_PENDING_ACK
	WeightsTransactions     []*WeightsTransactionStatus `protobuf:"bytes,2,rep,name=weights_transactions,json=weightsTransactions,proto3" json:"weights_transactions,omitempty"`                // sorted by weights_transaction_id
	RetainedDesiredRevision uint64                      `protobuf:"varint,3,opt,name=retained_desired_revision,json=retainedDesiredRevision,proto3" json:"retained_desired_revision,omitempty"` // Host ledger high-water, not Runtime acceptance or convergence.
	unknownFields           protoimpl.UnknownFields
	sizeCache               protoimpl.SizeCache
}

func (x *HostSnapshotBody) Reset() {
	*x = HostSnapshotBody{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[57]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *HostSnapshotBody) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*HostSnapshotBody) ProtoMessage() {}

func (x *HostSnapshotBody) ProtoReflect() protoreflect.Message {
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

// Deprecated: Use HostSnapshotBody.ProtoReflect.Descriptor instead.
func (*HostSnapshotBody) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{57}
}

func (x *HostSnapshotBody) GetHeldOutcomes() []*HeldAttempt {
	if x != nil {
		return x.HeldOutcomes
	}
	return nil
}

func (x *HostSnapshotBody) GetWeightsTransactions() []*WeightsTransactionStatus {
	if x != nil {
		return x.WeightsTransactions
	}
	return nil
}

func (x *HostSnapshotBody) GetRetainedDesiredRevision() uint64 {
	if x != nil {
		return x.RetainedDesiredRevision
	}
	return 0
}

type SnapshotAck struct {
	state              protoimpl.MessageState `protogen:"open.v1"`
	RecordOwnerEpoch   uint64                 `protobuf:"varint,1,opt,name=record_owner_epoch,json=recordOwnerEpoch,proto3" json:"record_owner_epoch,omitempty"`
	ControlStreamEpoch uint64                 `protobuf:"varint,2,opt,name=control_stream_epoch,json=controlStreamEpoch,proto3" json:"control_stream_epoch,omitempty"`
	WorkerBootId       string                 `protobuf:"bytes,3,opt,name=worker_boot_id,json=workerBootId,proto3" json:"worker_boot_id,omitempty"`
	SnapshotId         string                 `protobuf:"bytes,5,opt,name=snapshot_id,json=snapshotId,proto3" json:"snapshot_id,omitempty"`             // echo of the exact snapshot durably reconciled
	SnapshotDigest     []byte                 `protobuf:"bytes,6,opt,name=snapshot_digest,json=snapshotDigest,proto3" json:"snapshot_digest,omitempty"` // echo; COMPARED, never recomputed by the worker. An ack
	// naming a different id/digest is NOT an ack — the barrier
	// stays closed and dispatch never opens.
	HostSnapshotDigest []byte `protobuf:"bytes,7,opt,name=host_snapshot_digest,json=hostSnapshotDigest,proto3" json:"host_snapshot_digest,omitempty"` // echo of WorkerSnapshot 9; the host compares it before
	unknownFields      protoimpl.UnknownFields
	sizeCache          protoimpl.SizeCache
}

func (x *SnapshotAck) Reset() {
	*x = SnapshotAck{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[58]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *SnapshotAck) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*SnapshotAck) ProtoMessage() {}

func (x *SnapshotAck) ProtoReflect() protoreflect.Message {
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

// Deprecated: Use SnapshotAck.ProtoReflect.Descriptor instead.
func (*SnapshotAck) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{58}
}

func (x *SnapshotAck) GetRecordOwnerEpoch() uint64 {
	if x != nil {
		return x.RecordOwnerEpoch
	}
	return 0
}

func (x *SnapshotAck) GetControlStreamEpoch() uint64 {
	if x != nil {
		return x.ControlStreamEpoch
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

func (x *SnapshotAck) GetHostSnapshotDigest() []byte {
	if x != nil {
		return x.HostSnapshotDigest
	}
	return nil
}

type DesiredWorkerState struct {
	state              protoimpl.MessageState `protogen:"open.v1"`
	RecordOwnerEpoch   uint64                 `protobuf:"varint,1,opt,name=record_owner_epoch,json=recordOwnerEpoch,proto3" json:"record_owner_epoch,omitempty"`
	ControlStreamEpoch uint64                 `protobuf:"varint,2,opt,name=control_stream_epoch,json=controlStreamEpoch,proto3" json:"control_stream_epoch,omitempty"`
	WorkerBootId       string                 `protobuf:"bytes,3,opt,name=worker_boot_id,json=workerBootId,proto3" json:"worker_boot_id,omitempty"`
	Revision           uint64                 `protobuf:"varint,5,opt,name=revision,proto3" json:"revision,omitempty"` // RecordOwner-owned, monotonic; older ignored, equal
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
	//	*DesiredWorkerState_LocalPackageSet
	//	*DesiredWorkerState_PrivatePlacementSet
	Mode          isDesiredWorkerState_Mode `protobuf_oneof:"mode"`
	unknownFields protoimpl.UnknownFields
	sizeCache     protoimpl.SizeCache
}

func (x *DesiredWorkerState) Reset() {
	*x = DesiredWorkerState{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[59]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *DesiredWorkerState) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*DesiredWorkerState) ProtoMessage() {}

func (x *DesiredWorkerState) ProtoReflect() protoreflect.Message {
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

// Deprecated: Use DesiredWorkerState.ProtoReflect.Descriptor instead.
func (*DesiredWorkerState) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{59}
}

func (x *DesiredWorkerState) GetRecordOwnerEpoch() uint64 {
	if x != nil {
		return x.RecordOwnerEpoch
	}
	return 0
}

func (x *DesiredWorkerState) GetControlStreamEpoch() uint64 {
	if x != nil {
		return x.ControlStreamEpoch
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

func (x *DesiredWorkerState) GetLocalPackageSet() *DesiredLocalPackageSet {
	if x != nil {
		if x, ok := x.Mode.(*DesiredWorkerState_LocalPackageSet); ok {
			return x.LocalPackageSet
		}
	}
	return nil
}

func (x *DesiredWorkerState) GetPrivatePlacementSet() *DesiredPrivatePlacementSet {
	if x != nil {
		if x, ok := x.Mode.(*DesiredWorkerState_PrivatePlacementSet); ok {
			return x.PrivatePlacementSet
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
	// RETIRING (proto-025): 16-18 are the pre-21 host modes. At 21 the owner prepares through
	// PodHost and sends placement_set (15) itself; these are reserved at the retirement minor.
	// Private rentals send only Creator's logical exact package/model set. The pod host
	// verifies and resolves it; Creator never reconstructs a qualified Linux/CUDA PlacementSet.
	PackageSet *DesiredPackageSet `protobuf:"bytes,16,opt,name=package_set,json=packageSet,proto3,oneof"`
}

type DesiredWorkerState_LocalPackageSet struct {
	LocalPackageSet *DesiredLocalPackageSet `protobuf:"bytes,17,opt,name=local_package_set,json=localPackageSet,proto3,oneof"`
}

type DesiredWorkerState_PrivatePlacementSet struct {
	PrivatePlacementSet *DesiredPrivatePlacementSet `protobuf:"bytes,18,opt,name=private_placement_set,json=privatePlacementSet,proto3,oneof"`
}

func (*DesiredWorkerState_Job) isDesiredWorkerState_Mode() {}

func (*DesiredWorkerState_PlacementSet) isDesiredWorkerState_Mode() {}

func (*DesiredWorkerState_PackageSet) isDesiredWorkerState_Mode() {}

func (*DesiredWorkerState_LocalPackageSet) isDesiredWorkerState_Mode() {}

func (*DesiredWorkerState_PrivatePlacementSet) isDesiredWorkerState_Mode() {}

type DesiredPackageSet struct {
	state                       protoimpl.MessageState `protogen:"open.v1"`
	DownloadDelegation          []byte                 `protobuf:"bytes,1,opt,name=download_delegation,json=downloadDelegation,proto3" json:"download_delegation,omitempty"`                              // exact canonical DownloadDelegation/1 bytes
	DownloadDelegationSignature []byte                 `protobuf:"bytes,2,opt,name=download_delegation_signature,json=downloadDelegationSignature,proto3" json:"download_delegation_signature,omitempty"` // 64-byte Ed25519 signature by the rental Creator key
	unknownFields               protoimpl.UnknownFields
	sizeCache                   protoimpl.SizeCache
}

func (x *DesiredPackageSet) Reset() {
	*x = DesiredPackageSet{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[60]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *DesiredPackageSet) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*DesiredPackageSet) ProtoMessage() {}

func (x *DesiredPackageSet) ProtoReflect() protoreflect.Message {
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

// Deprecated: Use DesiredPackageSet.ProtoReflect.Descriptor instead.
func (*DesiredPackageSet) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{60}
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

// One command-scoped local package revision. Creator streams every listed wheel first. The pod
// compares the whole inventory on replay and turns it into an ordinary PlacementSet through the
// same Runtime preparation owner used by published packages.
type DesiredLocalPackageSet struct {
	state         protoimpl.MessageState `protogen:"open.v1"`
	OperationId   string                 `protobuf:"bytes,1,opt,name=operation_id,json=operationId,proto3" json:"operation_id,omitempty"`
	Package       *DevelopmentPackage    `protobuf:"bytes,2,opt,name=package,proto3" json:"package,omitempty"`
	Files         []*LocalPackageFileRef `protobuf:"bytes,3,rep,name=files,proto3" json:"files,omitempty"` // sorted uniquely by digest
	unknownFields protoimpl.UnknownFields
	sizeCache     protoimpl.SizeCache
}

func (x *DesiredLocalPackageSet) Reset() {
	*x = DesiredLocalPackageSet{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[61]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *DesiredLocalPackageSet) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*DesiredLocalPackageSet) ProtoMessage() {}

func (x *DesiredLocalPackageSet) ProtoReflect() protoreflect.Message {
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

// Deprecated: Use DesiredLocalPackageSet.ProtoReflect.Descriptor instead.
func (*DesiredLocalPackageSet) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{61}
}

func (x *DesiredLocalPackageSet) GetOperationId() string {
	if x != nil {
		return x.OperationId
	}
	return ""
}

func (x *DesiredLocalPackageSet) GetPackage() *DevelopmentPackage {
	if x != nil {
		return x.Package
	}
	return nil
}

func (x *DesiredLocalPackageSet) GetFiles() []*LocalPackageFileRef {
	if x != nil {
		return x.Files
	}
	return nil
}

// Bind exact models to code previously admitted through DesiredLocalPackageSet. Creator signs
// a model-only DownloadDelegation (packages empty); pod-supervisor verifies and materializes it,
// then Runtime authors the joined PlacementSet over its retained local revision core.
type DesiredPrivatePlacementSet struct {
	state                       protoimpl.MessageState `protogen:"open.v1"`
	OperationId                 string                 `protobuf:"bytes,1,opt,name=operation_id,json=operationId,proto3" json:"operation_id,omitempty"`
	LocalRevisionDigest         []byte                 `protobuf:"bytes,2,opt,name=local_revision_digest,json=localRevisionDigest,proto3" json:"local_revision_digest,omitempty"`
	DownloadDelegation          []byte                 `protobuf:"bytes,3,opt,name=download_delegation,json=downloadDelegation,proto3" json:"download_delegation,omitempty"`
	DownloadDelegationSignature []byte                 `protobuf:"bytes,4,opt,name=download_delegation_signature,json=downloadDelegationSignature,proto3" json:"download_delegation_signature,omitempty"` // 64-byte Ed25519 signature by rental Creator key
	unknownFields               protoimpl.UnknownFields
	sizeCache                   protoimpl.SizeCache
}

func (x *DesiredPrivatePlacementSet) Reset() {
	*x = DesiredPrivatePlacementSet{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[62]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *DesiredPrivatePlacementSet) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*DesiredPrivatePlacementSet) ProtoMessage() {}

func (x *DesiredPrivatePlacementSet) ProtoReflect() protoreflect.Message {
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

// Deprecated: Use DesiredPrivatePlacementSet.ProtoReflect.Descriptor instead.
func (*DesiredPrivatePlacementSet) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{62}
}

func (x *DesiredPrivatePlacementSet) GetOperationId() string {
	if x != nil {
		return x.OperationId
	}
	return ""
}

func (x *DesiredPrivatePlacementSet) GetLocalRevisionDigest() []byte {
	if x != nil {
		return x.LocalRevisionDigest
	}
	return nil
}

func (x *DesiredPrivatePlacementSet) GetDownloadDelegation() []byte {
	if x != nil {
		return x.DownloadDelegation
	}
	return nil
}

func (x *DesiredPrivatePlacementSet) GetDownloadDelegationSignature() []byte {
	if x != nil {
		return x.DownloadDelegationSignature
	}
	return nil
}

type LocalPackageFileRef struct {
	state         protoimpl.MessageState `protogen:"open.v1"`
	Digest        []byte                 `protobuf:"bytes,1,opt,name=digest,proto3" json:"digest,omitempty"`
	Filename      string                 `protobuf:"bytes,2,opt,name=filename,proto3" json:"filename,omitempty"`
	Length        uint64                 `protobuf:"varint,4,opt,name=length,proto3" json:"length,omitempty"`
	unknownFields protoimpl.UnknownFields
	sizeCache     protoimpl.SizeCache
}

func (x *LocalPackageFileRef) Reset() {
	*x = LocalPackageFileRef{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[63]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *LocalPackageFileRef) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*LocalPackageFileRef) ProtoMessage() {}

func (x *LocalPackageFileRef) ProtoReflect() protoreflect.Message {
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

// Deprecated: Use LocalPackageFileRef.ProtoReflect.Descriptor instead.
func (*LocalPackageFileRef) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{63}
}

func (x *LocalPackageFileRef) GetDigest() []byte {
	if x != nil {
		return x.Digest
	}
	return nil
}

func (x *LocalPackageFileRef) GetFilename() string {
	if x != nil {
		return x.Filename
	}
	return ""
}

func (x *LocalPackageFileRef) GetLength() uint64 {
	if x != nil {
		return x.Length
	}
	return 0
}

// DOCUMENT SHAPE: canonical `cozy.worker.v1.LocalPackageRevision/1`. Creator and Runtime
// independently derive these bytes from one package interface and the exact sorted wheel inventory.
// Its digest is the local code revision that enters Placement/Invocation identity; source_digest
// remains separate provenance for the checkout that produced those bytes.
type LocalPackageRevision struct {
	state            protoimpl.MessageState `protogen:"open.v1"`
	Package          string                 `protobuf:"bytes,1,opt,name=package,proto3" json:"package,omitempty"`
	Release          string                 `protobuf:"bytes,2,opt,name=release,proto3" json:"release,omitempty"`
	SourceDigest     []byte                 `protobuf:"bytes,3,opt,name=source_digest,json=sourceDigest,proto3" json:"source_digest,omitempty"`
	PackageInterface *Ref                   `protobuf:"bytes,4,opt,name=package_interface,json=packageInterface,proto3" json:"package_interface,omitempty"`
	Files            []*LocalPackageFileRef `protobuf:"bytes,5,rep,name=files,proto3" json:"files,omitempty"` // sorted unique by digest; exactly one row measures
	unknownFields    protoimpl.UnknownFields
	sizeCache        protoimpl.SizeCache
}

func (x *LocalPackageRevision) Reset() {
	*x = LocalPackageRevision{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[64]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *LocalPackageRevision) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*LocalPackageRevision) ProtoMessage() {}

func (x *LocalPackageRevision) ProtoReflect() protoreflect.Message {
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

// Deprecated: Use LocalPackageRevision.ProtoReflect.Descriptor instead.
func (*LocalPackageRevision) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{64}
}

func (x *LocalPackageRevision) GetPackage() string {
	if x != nil {
		return x.Package
	}
	return ""
}

func (x *LocalPackageRevision) GetRelease() string {
	if x != nil {
		return x.Release
	}
	return ""
}

func (x *LocalPackageRevision) GetSourceDigest() []byte {
	if x != nil {
		return x.SourceDigest
	}
	return nil
}

func (x *LocalPackageRevision) GetPackageInterface() *Ref {
	if x != nil {
		return x.PackageInterface
	}
	return nil
}

func (x *LocalPackageRevision) GetFiles() []*LocalPackageFileRef {
	if x != nil {
		return x.Files
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
	DevicePins                 []*PlacementDevicePin  `protobuf:"bytes,4,rep,name=device_pins,json=devicePins,proto3" json:"device_pins,omitempty"`                                                     // proto-024: sorted unique by placement_id;
	unknownFields              protoimpl.UnknownFields
	sizeCache                  protoimpl.SizeCache
}

func (x *DesiredPlacementSet) Reset() {
	*x = DesiredPlacementSet{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[65]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *DesiredPlacementSet) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*DesiredPlacementSet) ProtoMessage() {}

func (x *DesiredPlacementSet) ProtoReflect() protoreflect.Message {
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

// Deprecated: Use DesiredPlacementSet.ProtoReflect.Descriptor instead.
func (*DesiredPlacementSet) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{65}
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

func (x *DesiredPlacementSet) GetDevicePins() []*PlacementDevicePin {
	if x != nil {
		return x.DevicePins
	}
	return nil
}

// Where a placement sits on THIS worker (proto-024). Ordinals are envelope-local (the worker's
// `--devices` order) and sorted unique; K > 1 pins the placement to a GROUP lane whose ONE
// executor is sealed to all K devices and whose construction must be declared for degree K.
// The pin is part of neither placement-set identity nor invocation identity; re-pinning is a
// new desired-state revision over the SAME bytes. A pin the worker cannot honour — an ordinal
// outside the envelope, a device another placement holds, a fit that fails, a package with no
// construction declared for K — is a typed placement fault (FAULT_KIND_CONFIG_REFUSED, reason
// `device_pin_infeasible` / `device_group_unsupported`); other placements are untouched.
type PlacementDevicePin struct {
	state          protoimpl.MessageState `protogen:"open.v1"`
	PlacementId    string                 `protobuf:"bytes,1,opt,name=placement_id,json=placementId,proto3" json:"placement_id,omitempty"`                  // must name one placement in placement_set_canonical_bytes
	DeviceOrdinals []uint32               `protobuf:"varint,2,rep,packed,name=device_ordinals,json=deviceOrdinals,proto3" json:"device_ordinals,omitempty"` // K >= 1, sorted unique, envelope-local
	unknownFields  protoimpl.UnknownFields
	sizeCache      protoimpl.SizeCache
}

func (x *PlacementDevicePin) Reset() {
	*x = PlacementDevicePin{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[66]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *PlacementDevicePin) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*PlacementDevicePin) ProtoMessage() {}

func (x *PlacementDevicePin) ProtoReflect() protoreflect.Message {
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

// Deprecated: Use PlacementDevicePin.ProtoReflect.Descriptor instead.
func (*PlacementDevicePin) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{66}
}

func (x *PlacementDevicePin) GetPlacementId() string {
	if x != nil {
		return x.PlacementId
	}
	return ""
}

func (x *PlacementDevicePin) GetDeviceOrdinals() []uint32 {
	if x != nil {
		return x.DeviceOrdinals
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
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[67]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *PlacementSet) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*PlacementSet) ProtoMessage() {}

func (x *PlacementSet) ProtoReflect() protoreflect.Message {
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

// Deprecated: Use PlacementSet.ProtoReflect.Descriptor instead.
func (*PlacementSet) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{67}
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
	state         protoimpl.MessageState `protogen:"open.v1"`
	ExpiresAtUnix uint64                 `protobuf:"varint,1,opt,name=expires_at_unix,json=expiresAtUnix,proto3" json:"expires_at_unix,omitempty"`
	Models        []*DownloadModelRef    `protobuf:"bytes,2,rep,name=models,proto3" json:"models,omitempty"`     // sorted unique by (package,slot,model,release,manifest)
	Packages      []*DownloadPackageRef  `protobuf:"bytes,3,rep,name=packages,proto3" json:"packages,omitempty"` // sorted unique by (package,release); empty only for
	// DesiredPrivatePlacementSet model binding
	RentalId                   string `protobuf:"bytes,4,opt,name=rental_id,json=rentalId,proto3" json:"rental_id,omitempty"`
	WorkerBootId               string `protobuf:"bytes,5,opt,name=worker_boot_id,json=workerBootId,proto3" json:"worker_boot_id,omitempty"`
	WorkerId                   string `protobuf:"bytes,6,opt,name=worker_id,json=workerId,proto3" json:"worker_id,omitempty"`
	WorkerTlsCertificateDigest []byte `protobuf:"bytes,7,opt,name=worker_tls_certificate_digest,json=workerTlsCertificateDigest,proto3" json:"worker_tls_certificate_digest,omitempty"` // sha256 of exact readiness-pinned leaf DER
	unknownFields              protoimpl.UnknownFields
	sizeCache                  protoimpl.SizeCache
}

func (x *DownloadDelegation) Reset() {
	*x = DownloadDelegation{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[68]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *DownloadDelegation) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*DownloadDelegation) ProtoMessage() {}

func (x *DownloadDelegation) ProtoReflect() protoreflect.Message {
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

// Deprecated: Use DownloadDelegation.ProtoReflect.Descriptor instead.
func (*DownloadDelegation) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{68}
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
	Slot          string                 `protobuf:"bytes,5,opt,name=slot,proto3" json:"slot,omitempty"`         // exact package-interface model-slot path; unique per package
	Lane          string                 `protobuf:"bytes,6,opt,name=lane,proto3" json:"lane,omitempty"`         // the release lane the manifest was resolved from (e.g.
	unknownFields protoimpl.UnknownFields
	sizeCache     protoimpl.SizeCache
}

func (x *DownloadModelRef) Reset() {
	*x = DownloadModelRef{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[69]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *DownloadModelRef) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*DownloadModelRef) ProtoMessage() {}

func (x *DownloadModelRef) ProtoReflect() protoreflect.Message {
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

// Deprecated: Use DownloadModelRef.ProtoReflect.Descriptor instead.
func (*DownloadModelRef) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{69}
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

func (x *DownloadModelRef) GetLane() string {
	if x != nil {
		return x.Lane
	}
	return ""
}

type DownloadPackageRef struct {
	state         protoimpl.MessageState `protogen:"open.v1"`
	Package       string                 `protobuf:"bytes,1,opt,name=package,proto3" json:"package,omitempty"` // org/name
	Release       string                 `protobuf:"bytes,2,opt,name=release,proto3" json:"release,omitempty"` // exact semver release
	unknownFields protoimpl.UnknownFields
	sizeCache     protoimpl.SizeCache
}

func (x *DownloadPackageRef) Reset() {
	*x = DownloadPackageRef{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[70]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *DownloadPackageRef) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*DownloadPackageRef) ProtoMessage() {}

func (x *DownloadPackageRef) ProtoReflect() protoreflect.Message {
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

// Deprecated: Use DownloadPackageRef.ProtoReflect.Descriptor instead.
func (*DownloadPackageRef) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{70}
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
	PackageInterface  *Ref                    `protobuf:"bytes,5,opt,name=package_interface,json=packageInterface,proto3" json:"package_interface,omitempty"`    // exact PackageInterface bytes
	BindingsDigest    []byte                  `protobuf:"bytes,6,opt,name=bindings_digest,json=bindingsDigest,proto3" json:"bindings_digest,omitempty"`          // class (a): sha256(canonical JSON of exactly
	// {entrypoints:<field 8>,models:<field 7>})
	Models        []*Model      `protobuf:"bytes,7,rep,name=models,proto3" json:"models,omitempty"`            // sorted unique by id; [] for a weightless package
	Entrypoints   []*Entrypoint `protobuf:"bytes,8,rep,name=entrypoints,proto3" json:"entrypoints,omitempty"`  // sorted unique by name
	Environment   *Environment  `protobuf:"bytes,10,opt,name=environment,proto3" json:"environment,omitempty"` // exact non-base install content (the lock, or supplied
	unknownFields protoimpl.UnknownFields
	sizeCache     protoimpl.SizeCache
}

func (x *Placement) Reset() {
	*x = Placement{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[71]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *Placement) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*Placement) ProtoMessage() {}

func (x *Placement) ProtoReflect() protoreflect.Message {
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

// Deprecated: Use Placement.ProtoReflect.Descriptor instead.
func (*Placement) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{71}
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

func (x *Placement) GetPackageInterface() *Ref {
	if x != nil {
		return x.PackageInterface
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
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[72]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *Ref) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*Ref) ProtoMessage() {}

func (x *Ref) ProtoReflect() protoreflect.Message {
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

// Deprecated: Use Ref.ProtoReflect.Descriptor instead.
func (*Ref) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{72}
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

// One exact wheel the pod actually holds bytes for: a transferred local-package project wheel
// (DevelopmentPackage) or an explicitly supplied editable wheel (Environment.local_wheels).
// Published installs carry no WheelFact — the lock's own hashes bind their exact bytes.
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
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[73]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *WheelFact) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*WheelFact) ProtoMessage() {}

func (x *WheelFact) ProtoReflect() protoreflect.Message {
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

// Deprecated: Use WheelFact.ProtoReflect.Descriptor instead.
func (*WheelFact) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{73}
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

// The directly executable published package identity. Tensorhub makes (package, release)
// immutable; the exact execution bytes are bound by the release's locked requirements export
// (Environment.locked_requirements — its hashes pin every artifact, the project wheel
// included), the derived package interface, bindings, and model refs in this Placement.
// MINOR 30 deletes the transported project wheel: the pod installs it from the org index and
// never retains bytes to re-measure.
type PackageSelection struct {
	state         protoimpl.MessageState `protogen:"open.v1"`
	Package       string                 `protobuf:"bytes,1,opt,name=package,proto3" json:"package,omitempty"` // exact org/name package identity
	Release       string                 `protobuf:"bytes,2,opt,name=release,proto3" json:"release,omitempty"` // exact semantic release
	unknownFields protoimpl.UnknownFields
	sizeCache     protoimpl.SizeCache
}

func (x *PackageSelection) Reset() {
	*x = PackageSelection{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[74]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *PackageSelection) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*PackageSelection) ProtoMessage() {}

func (x *PackageSelection) ProtoReflect() protoreflect.Message {
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

// Deprecated: Use PackageSelection.ProtoReflect.Descriptor instead.
func (*PackageSelection) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{74}
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

// Local same-user source execution is not a partially installed package. The source digest is
// the daemon-measured checkout identity; its path is a launch-local fact and never enters the
// protocol. A development Placement omits the published-only environment field.
type DevelopmentPackage struct {
	state        protoimpl.MessageState `protogen:"open.v1"`
	Package      string                 `protobuf:"bytes,1,opt,name=package,proto3" json:"package,omitempty"`                               // exact org/name package identity
	Release      string                 `protobuf:"bytes,2,opt,name=release,proto3" json:"release,omitempty"`                               // exact semantic release
	SourceDigest []byte                 `protobuf:"bytes,3,opt,name=source_digest,json=sourceDigest,proto3" json:"source_digest,omitempty"` // sha256 of the daemon's exact source snapshot
	ProjectWheel *WheelFact             `protobuf:"bytes,4,opt,name=project_wheel,json=projectWheel,proto3" json:"project_wheel,omitempty"` // present only for a remote snapshot transferred as a local
	// package; absent for a same-host live checkout
	LocalRevisionDigest []byte `protobuf:"bytes,5,opt,name=local_revision_digest,json=localRevisionDigest,proto3" json:"local_revision_digest,omitempty"` // sha256 of LocalPackageRevision/1; required with
	unknownFields       protoimpl.UnknownFields
	sizeCache           protoimpl.SizeCache
}

func (x *DevelopmentPackage) Reset() {
	*x = DevelopmentPackage{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[75]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *DevelopmentPackage) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*DevelopmentPackage) ProtoMessage() {}

func (x *DevelopmentPackage) ProtoReflect() protoreflect.Message {
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

// Deprecated: Use DevelopmentPackage.ProtoReflect.Descriptor instead.
func (*DevelopmentPackage) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{75}
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

func (x *DevelopmentPackage) GetProjectWheel() *WheelFact {
	if x != nil {
		return x.ProjectWheel
	}
	return nil
}

func (x *DevelopmentPackage) GetLocalRevisionDigest() []byte {
	if x != nil {
		return x.LocalRevisionDigest
	}
	return nil
}

// The exact non-base install content. A published install is identified by the locked
// requirements export it materialized from; a transferred editable revision by the exact
// wheels that were explicitly supplied because no index serves them.
type Environment struct {
	state              protoimpl.MessageState `protogen:"open.v1"`
	LockedRequirements *Ref                   `protobuf:"bytes,3,opt,name=locked_requirements,json=lockedRequirements,proto3" json:"locked_requirements,omitempty"` // exact export this install materialized from; the bytes
	// are an operation input (the hub keeps the lock) and are
	// not retained locally. Absent only for a transferred
	// editable revision.
	LocalWheels   []*WheelFact `protobuf:"bytes,4,rep,name=local_wheels,json=localWheels,proto3" json:"local_wheels,omitempty"` // editable only: sorted unique by normalized
	unknownFields protoimpl.UnknownFields
	sizeCache     protoimpl.SizeCache
}

func (x *Environment) Reset() {
	*x = Environment{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[76]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *Environment) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*Environment) ProtoMessage() {}

func (x *Environment) ProtoReflect() protoreflect.Message {
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

// Deprecated: Use Environment.ProtoReflect.Descriptor instead.
func (*Environment) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{76}
}

func (x *Environment) GetLockedRequirements() *Ref {
	if x != nil {
		return x.LockedRequirements
	}
	return nil
}

func (x *Environment) GetLocalWheels() []*WheelFact {
	if x != nil {
		return x.LocalWheels
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
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[77]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *Model) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*Model) ProtoMessage() {}

func (x *Model) ProtoReflect() protoreflect.Message {
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

// Deprecated: Use Model.ProtoReflect.Descriptor instead.
func (*Model) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{77}
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
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[78]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *Entrypoint) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*Entrypoint) ProtoMessage() {}

func (x *Entrypoint) ProtoReflect() protoreflect.Message {
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

// Deprecated: Use Entrypoint.ProtoReflect.Descriptor instead.
func (*Entrypoint) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{78}
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
	ReferenceModelId          string                 `protobuf:"bytes,2,opt,name=reference_model_id,json=referenceModelId,proto3" json:"reference_model_id,omitempty"`                            // must name one Placement.models id
	Components                []*Component           `protobuf:"bytes,4,rep,name=components,proto3" json:"components,omitempty"`                                                                  // ORDERED MCC destination sequence; never sorted
	ModelConstructionContract *Ref                   `protobuf:"bytes,5,opt,name=model_construction_contract,json=modelConstructionContract,proto3" json:"model_construction_contract,omitempty"` // required for qualified publication; omitted when
	// DevelopmentPackage derives it from source + local model
	Stamps        []*Stamp `protobuf:"bytes,6,rep,name=stamps,proto3" json:"stamps,omitempty"` // sorted unique by (component,key)
	unknownFields protoimpl.UnknownFields
	sizeCache     protoimpl.SizeCache
}

func (x *Slot) Reset() {
	*x = Slot{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[79]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *Slot) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*Slot) ProtoMessage() {}

func (x *Slot) ProtoReflect() protoreflect.Message {
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

// Deprecated: Use Slot.ProtoReflect.Descriptor instead.
func (*Slot) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{79}
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

type Component struct {
	state         protoimpl.MessageState `protogen:"open.v1"`
	Component     string                 `protobuf:"bytes,1,opt,name=component,proto3" json:"component,omitempty"`
	ModelId       string                 `protobuf:"bytes,2,opt,name=model_id,json=modelId,proto3" json:"model_id,omitempty"` // must name one Placement.models id
	unknownFields protoimpl.UnknownFields
	sizeCache     protoimpl.SizeCache
}

func (x *Component) Reset() {
	*x = Component{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[80]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *Component) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*Component) ProtoMessage() {}

func (x *Component) ProtoReflect() protoreflect.Message {
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

// Deprecated: Use Component.ProtoReflect.Descriptor instead.
func (*Component) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{80}
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
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[81]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *Stamp) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*Stamp) ProtoMessage() {}

func (x *Stamp) ProtoReflect() protoreflect.Message {
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

// Deprecated: Use Stamp.ProtoReflect.Descriptor instead.
func (*Stamp) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{81}
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
	// MINOR 40: exactly one CPU orchestration slot may coexist with the serial
	// ordinary job envelope. The selected parent must have frozen interface
	// dependencies, no Model/Weights capabilities, and no device grant.
	Orchestration bool `protobuf:"varint,7,opt,name=orchestration,proto3" json:"orchestration,omitempty"`
	// A complete desired state while an ordinary child runs includes its unchanged
	// CPU parent explicitly. Must have orchestration=true and no nested parent;
	// top-level orchestration=true cannot contain this field. This is an admitted
	// job set, never a program or a list of future calls.
	OrchestrationParent *JobDirective `protobuf:"bytes,8,opt,name=orchestration_parent,json=orchestrationParent,proto3" json:"orchestration_parent,omitempty"`
	unknownFields       protoimpl.UnknownFields
	sizeCache           protoimpl.SizeCache
}

func (x *JobDirective) Reset() {
	*x = JobDirective{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[82]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *JobDirective) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*JobDirective) ProtoMessage() {}

func (x *JobDirective) ProtoReflect() protoreflect.Message {
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

// Deprecated: Use JobDirective.ProtoReflect.Descriptor instead.
func (*JobDirective) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{82}
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

func (x *JobDirective) GetOrchestration() bool {
	if x != nil {
		return x.Orchestration
	}
	return false
}

func (x *JobDirective) GetOrchestrationParent() *JobDirective {
	if x != nil {
		return x.OrchestrationParent
	}
	return nil
}

// CONVERGENCE FACTS ON THE WIRE (§8; was Report). `applied_revision` is RETIRED as dishonest —
// it advanced on ACCEPTANCE, so a RecordOwner reading it learned only that its own message
// arrived. Acceptance and convergence are now two separate, checkable facts:
// converged_revision < accepted_desired_state_revision is the normal, visible state of a
// convergence in progress or a latched failure — never an error, always readable.
type ObservedWorkerState struct {
	state              protoimpl.MessageState `protogen:"open.v1"`
	RecordOwnerEpoch   uint64                 `protobuf:"varint,1,opt,name=record_owner_epoch,json=recordOwnerEpoch,proto3" json:"record_owner_epoch,omitempty"`
	ControlStreamEpoch uint64                 `protobuf:"varint,2,opt,name=control_stream_epoch,json=controlStreamEpoch,proto3" json:"control_stream_epoch,omitempty"`
	WorkerBootId       string                 `protobuf:"bytes,3,opt,name=worker_boot_id,json=workerBootId,proto3" json:"worker_boot_id,omitempty"`
	AppliedWireMinor   uint32                 `protobuf:"varint,8,opt,name=applied_wire_minor,json=appliedWireMinor,proto3" json:"applied_wire_minor,omitempty"` // stale echo is a visible skew fact, never a refusal
	HeldAttempts       []*HeldAttempt         `protobuf:"bytes,9,rep,name=held_attempts,json=heldAttempts,proto3" json:"held_attempts,omitempty"`                // every accepted attempt from admission to ack:
	// QUEUED, RUNNING, DEVICE_RELEASED, OUTCOME_PENDING_ACK
	// (proto-026: a queued attempt is no longer absent)
	Faults      []*Fault           `protobuf:"bytes,10,rep,name=faults,proto3" json:"faults,omitempty"`                              // MACHINE-scope faults; placement faults ride their status
	Activity    []*ActivityEvent   `protobuf:"bytes,11,rep,name=activity,proto3" json:"activity,omitempty"`                          // the non-attempt activity lane (durable)
	JobCapacity *JobCapacity       `protobuf:"bytes,13,opt,name=job_capacity,json=jobCapacity,proto3" json:"job_capacity,omitempty"` // job mode only
	Placements  []*PlacementStatus `protobuf:"bytes,15,rep,name=placements,proto3" json:"placements,omitempty"`                      // serving mode only; sorted by placement_id
	// The ONE worker-level admission gate (§6). The RecordOwner dispatches against the epoch it
	// last saw; the worker admits only if the echoed epoch is CURRENT, admission_state is OPEN,
	// and a slot is free. A stale epoch refuses DETERMINISTICALLY — same input, same verdict, no
	// race window. Only the epoch FENCES: state and slots say "not accepting" / "full", which is a
	// gate, not a fence against a stale actor.
	AdmissionEpoch uint64 `protobuf:"varint,16,opt,name=admission_epoch,json=admissionEpoch,proto3" json:"admission_epoch,omitempty"` // bumps whenever admission MEANING changes: window
	// resize, phase change, cutover, lane change. NOT on
	// executor respawn (proto-026, gpu-hot D10): a queued
	// attempt is bound to no executor, so a respawn
	// invalidates nothing it used to; a RUNNING attempt dies
	// with its own executor_epoch as before. record_owner_epoch,
	// control_stream_epoch and executor_epoch are unchanged.
	AdmissionState AdmissionState `protobuf:"varint,17,opt,name=admission_state,json=admissionState,proto3,enum=cozy.worker.v1.AdmissionState" json:"admission_state,omitempty"` // CLOSED is STRUCTURAL (pre-snapshot-barrier,
	// draining, mid-cutover); OPEN with zero slots is TRANSIENT
	// saturation — a RecordOwner backs off differently for each
	AvailableAttemptSlots        uint32 `protobuf:"varint,18,opt,name=available_attempt_slots,json=availableAttemptSlots,proto3" json:"available_attempt_slots,omitempty"`                        // offers this worker will still admit to its lane
	AcceptedDesiredStateRevision uint64 `protobuf:"varint,19,opt,name=accepted_desired_state_revision,json=acceptedDesiredStateRevision,proto3" json:"accepted_desired_state_revision,omitempty"` // durably ACCEPTED intent (advances on
	// acceptance — which is all it ever honestly meant)
	AcceptedPlacementSetDigest []byte `protobuf:"bytes,20,opt,name=accepted_placement_set_digest,json=acceptedPlacementSetDigest,proto3" json:"accepted_placement_set_digest,omitempty"` // class (a): the accepted set's digest
	ConvergedRevision          uint64 `protobuf:"varint,21,opt,name=converged_revision,json=convergedRevision,proto3" json:"converged_revision,omitempty"`                               // advances ONLY when observed satisfies accepted. It may
	// NEVER advance past a placement whose serving state is not
	// DISPATCHABLE for a desired spec: the field is a derived
	// fact, and a worker that advances it while observed state
	// disagrees is in breach, not merely optimistic.
	WorkerPhase WorkerPhase   `protobuf:"varint,22,opt,name=worker_phase,json=workerPhase,proto3,enum=cozy.worker.v1.WorkerPhase" json:"worker_phase,omitempty"` // machine lifecycle, out of the placement enum
	Lanes       []*DeviceLane `protobuf:"bytes,26,rep,name=lanes,proto3" json:"lanes,omitempty"`                                                                 // proto-024: every lane, sorted by lane_id; serving mode.
	// A 1-device worker reports exactly one lane. Any lane
	// change (a group forming or dissolving) bumps
	// admission_epoch; structural CLOSED stays a worker fact.
	HeldManifests []string `protobuf:"bytes,27,rep,name=held_manifests,json=heldManifests,proto3" json:"held_manifests,omitempty"` // proto-026 residency (residency-aware-routing.md §2):
	unknownFields protoimpl.UnknownFields
	sizeCache     protoimpl.SizeCache
}

func (x *ObservedWorkerState) Reset() {
	*x = ObservedWorkerState{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[83]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *ObservedWorkerState) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*ObservedWorkerState) ProtoMessage() {}

func (x *ObservedWorkerState) ProtoReflect() protoreflect.Message {
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

// Deprecated: Use ObservedWorkerState.ProtoReflect.Descriptor instead.
func (*ObservedWorkerState) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{83}
}

func (x *ObservedWorkerState) GetRecordOwnerEpoch() uint64 {
	if x != nil {
		return x.RecordOwnerEpoch
	}
	return 0
}

func (x *ObservedWorkerState) GetControlStreamEpoch() uint64 {
	if x != nil {
		return x.ControlStreamEpoch
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

func (x *ObservedWorkerState) GetAdmissionEpoch() uint64 {
	if x != nil {
		return x.AdmissionEpoch
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

func (x *ObservedWorkerState) GetLanes() []*DeviceLane {
	if x != nil {
		return x.Lanes
	}
	return nil
}

func (x *ObservedWorkerState) GetHeldManifests() []string {
	if x != nil {
		return x.HeldManifests
	}
	return nil
}

// ONE serialized resource inside the worker (proto-024/cr-066): a set of envelope-local device
// ordinals, one attempt seat, one ledger, and the placements resident on those devices. Law 8
// (one active attempt per PHYSICAL device) is a lane fact. K == 1 is a device lane; K > 1 is a
// GROUP lane — one executor sealed to all K, one seat, the ledger accounting per device, the
// whole executor torn down as a unit (executor_epoch bumps once, every device's rows clear).
// Ordinals are positions in the worker's `--devices` envelope, sorted unique, disjoint across
// lanes.
//
// The lane holds a QUEUE of accepted attempts (proto-026, gpu-hot.md §4) and orders it itself:
// resident tenant first, then co-fitting, then the rest, within an overtake budget each attempt
// carries on HeldAttempt. The device SEAT is not a count here — it is the RUNNING held attempt
// whose lane_id names this lane.
type DeviceLane struct {
	state                 protoimpl.MessageState `protogen:"open.v1"`
	LaneId                string                 `protobuf:"bytes,1,opt,name=lane_id,json=laneId,proto3" json:"lane_id,omitempty"`                                                 // worker-minted, opaque, stable while the lane exists
	DeviceOrdinals        []uint32               `protobuf:"varint,2,rep,packed,name=device_ordinals,json=deviceOrdinals,proto3" json:"device_ordinals,omitempty"`                 // K >= 1
	AvailableAttemptSlots uint32                 `protobuf:"varint,3,opt,name=available_attempt_slots,json=availableAttemptSlots,proto3" json:"available_attempt_slots,omitempty"` // offers this lane will still ADMIT TO ITS QUEUE now:
	// the depth the lane can account for in host bytes
	// (declared input + output bounds against the host
	// budget, gpu-hot.md §4) minus attempts held. Never a
	// constant; never the seat.
	PlacementIds         []string `protobuf:"bytes,4,rep,name=placement_ids,json=placementIds,proto3" json:"placement_ids,omitempty"`                           // sorted; placements assigned to this lane
	ResidentPlacementIds []string `protobuf:"bytes,5,rep,name=resident_placement_ids,json=residentPlacementIds,proto3" json:"resident_placement_ids,omitempty"` // proto-026 residency: placements whose executor
	unknownFields        protoimpl.UnknownFields
	sizeCache            protoimpl.SizeCache
}

func (x *DeviceLane) Reset() {
	*x = DeviceLane{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[84]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *DeviceLane) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*DeviceLane) ProtoMessage() {}

func (x *DeviceLane) ProtoReflect() protoreflect.Message {
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

// Deprecated: Use DeviceLane.ProtoReflect.Descriptor instead.
func (*DeviceLane) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{84}
}

func (x *DeviceLane) GetLaneId() string {
	if x != nil {
		return x.LaneId
	}
	return ""
}

func (x *DeviceLane) GetDeviceOrdinals() []uint32 {
	if x != nil {
		return x.DeviceOrdinals
	}
	return nil
}

func (x *DeviceLane) GetAvailableAttemptSlots() uint32 {
	if x != nil {
		return x.AvailableAttemptSlots
	}
	return 0
}

func (x *DeviceLane) GetPlacementIds() []string {
	if x != nil {
		return x.PlacementIds
	}
	return nil
}

func (x *DeviceLane) GetResidentPlacementIds() []string {
	if x != nil {
		return x.ResidentPlacementIds
	}
	return nil
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
	ExecutorEpoch                uint64                    `protobuf:"varint,4,opt,name=executor_epoch,json=executorEpoch,proto3" json:"executor_epoch,omitempty"` // THIS placement's executor fence; bumps on its respawn
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
	Acquisition       *PlacementAcquisitionObservation `protobuf:"bytes,15,opt,name=acquisition,proto3" json:"acquisition,omitempty"`                                      // OBSERVATION ONLY for this
	EnvironmentDigest string                           `protobuf:"bytes,17,opt,name=environment_digest,json=environmentDigest,proto3" json:"environment_digest,omitempty"` // exact InvocationSpec environment_digest
	DeviceLaneId      string                           `protobuf:"bytes,19,opt,name=device_lane_id,json=deviceLaneId,proto3" json:"device_lane_id,omitempty"`              // proto-024: the ObservedWorkerState.lanes[] entry this
	unknownFields     protoimpl.UnknownFields
	sizeCache         protoimpl.SizeCache
}

func (x *PlacementStatus) Reset() {
	*x = PlacementStatus{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[85]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *PlacementStatus) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*PlacementStatus) ProtoMessage() {}

func (x *PlacementStatus) ProtoReflect() protoreflect.Message {
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

// Deprecated: Use PlacementStatus.ProtoReflect.Descriptor instead.
func (*PlacementStatus) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{85}
}

func (x *PlacementStatus) GetPlacementId() string {
	if x != nil {
		return x.PlacementId
	}
	return ""
}

func (x *PlacementStatus) GetExecutorEpoch() uint64 {
	if x != nil {
		return x.ExecutorEpoch
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

func (x *PlacementStatus) GetEnvironmentDigest() string {
	if x != nil {
		return x.EnvironmentDigest
	}
	return ""
}

func (x *PlacementStatus) GetDeviceLaneId() string {
	if x != nil {
		return x.DeviceLaneId
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
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[86]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *PlacementAcquisitionObservation) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*PlacementAcquisitionObservation) ProtoMessage() {}

func (x *PlacementAcquisitionObservation) ProtoReflect() protoreflect.Message {
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

// Deprecated: Use PlacementAcquisitionObservation.ProtoReflect.Descriptor instead.
func (*PlacementAcquisitionObservation) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{86}
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
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[87]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *AcquisitionLegObservation) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*AcquisitionLegObservation) ProtoMessage() {}

func (x *AcquisitionLegObservation) ProtoReflect() protoreflect.Message {
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

// Deprecated: Use AcquisitionLegObservation.ProtoReflect.Descriptor instead.
func (*AcquisitionLegObservation) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{87}
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
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[88]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *AcceleratorQualification) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*AcceleratorQualification) ProtoMessage() {}

func (x *AcceleratorQualification) ProtoReflect() protoreflect.Message {
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

// Deprecated: Use AcceleratorQualification.ProtoReflect.Descriptor instead.
func (*AcceleratorQualification) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{88}
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
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[89]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *ActivityEvent) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*ActivityEvent) ProtoMessage() {}

func (x *ActivityEvent) ProtoReflect() protoreflect.Message {
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

// Deprecated: Use ActivityEvent.ProtoReflect.Descriptor instead.
func (*ActivityEvent) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{89}
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
	state              protoimpl.MessageState `protogen:"open.v1"`
	RecordOwnerEpoch   uint64                 `protobuf:"varint,1,opt,name=record_owner_epoch,json=recordOwnerEpoch,proto3" json:"record_owner_epoch,omitempty"`
	ControlStreamEpoch uint64                 `protobuf:"varint,2,opt,name=control_stream_epoch,json=controlStreamEpoch,proto3" json:"control_stream_epoch,omitempty"`
	WorkerBootId       string                 `protobuf:"bytes,3,opt,name=worker_boot_id,json=workerBootId,proto3" json:"worker_boot_id,omitempty"`
	RequestId          string                 `protobuf:"bytes,5,opt,name=request_id,json=requestId,proto3" json:"request_id,omitempty"`                 // fence (a)
	AttemptOrdinal     uint64                 `protobuf:"varint,6,opt,name=attempt_ordinal,json=attemptOrdinal,proto3" json:"attempt_ordinal,omitempty"` // fence (b); RecordOwner-bumped only. EVERY offer consumes
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
	AdmissionEpoch uint64 `protobuf:"varint,12,opt,name=admission_epoch,json=admissionEpoch,proto3" json:"admission_epoch,omitempty"` // the epoch the RecordOwner OBSERVED when it
	unknownFields  protoimpl.UnknownFields
	sizeCache      protoimpl.SizeCache
}

func (x *AttemptOffer) Reset() {
	*x = AttemptOffer{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[90]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *AttemptOffer) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*AttemptOffer) ProtoMessage() {}

func (x *AttemptOffer) ProtoReflect() protoreflect.Message {
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

// Deprecated: Use AttemptOffer.ProtoReflect.Descriptor instead.
func (*AttemptOffer) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{90}
}

func (x *AttemptOffer) GetRecordOwnerEpoch() uint64 {
	if x != nil {
		return x.RecordOwnerEpoch
	}
	return 0
}

func (x *AttemptOffer) GetControlStreamEpoch() uint64 {
	if x != nil {
		return x.ControlStreamEpoch
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

func (x *AttemptOffer) GetAdmissionEpoch() uint64 {
	if x != nil {
		return x.AdmissionEpoch
	}
	return 0
}

// ACCEPTANCE IS QUEUE ADMISSION (proto-026, gpu-hot D2): "journaled, hydrated, queued on
// lane_id". It binds neither a plan nor an executor epoch — both bind at DEVICE ENTRY and are
// OBSERVED on HeldAttempt (plan_digest from RUNNING on; executor_epoch, 0 while QUEUED). Plan
// facts ride the outcome body (AttemptMetrics).
type AttemptAccepted struct {
	state                protoimpl.MessageState `protogen:"open.v1"`
	RecordOwnerEpoch     uint64                 `protobuf:"varint,1,opt,name=record_owner_epoch,json=recordOwnerEpoch,proto3" json:"record_owner_epoch,omitempty"`
	ControlStreamEpoch   uint64                 `protobuf:"varint,2,opt,name=control_stream_epoch,json=controlStreamEpoch,proto3" json:"control_stream_epoch,omitempty"`
	WorkerBootId         string                 `protobuf:"bytes,3,opt,name=worker_boot_id,json=workerBootId,proto3" json:"worker_boot_id,omitempty"`
	RequestId            string                 `protobuf:"bytes,5,opt,name=request_id,json=requestId,proto3" json:"request_id,omitempty"`
	AttemptOrdinal       uint64                 `protobuf:"varint,6,opt,name=attempt_ordinal,json=attemptOrdinal,proto3" json:"attempt_ordinal,omitempty"`
	InvocationSpecDigest []byte                 `protobuf:"bytes,7,opt,name=invocation_spec_digest,json=invocationSpecDigest,proto3" json:"invocation_spec_digest,omitempty"` // echo
	PlacementId          string                 `protobuf:"bytes,11,opt,name=placement_id,json=placementId,proto3" json:"placement_id,omitempty"`                             // echo
	LaneId               string                 `protobuf:"bytes,13,opt,name=lane_id,json=laneId,proto3" json:"lane_id,omitempty"`                                            // proto-026: the DeviceLane the attempt was admitted to;
	unknownFields        protoimpl.UnknownFields
	sizeCache            protoimpl.SizeCache
}

func (x *AttemptAccepted) Reset() {
	*x = AttemptAccepted{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[91]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *AttemptAccepted) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*AttemptAccepted) ProtoMessage() {}

func (x *AttemptAccepted) ProtoReflect() protoreflect.Message {
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

// Deprecated: Use AttemptAccepted.ProtoReflect.Descriptor instead.
func (*AttemptAccepted) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{91}
}

func (x *AttemptAccepted) GetRecordOwnerEpoch() uint64 {
	if x != nil {
		return x.RecordOwnerEpoch
	}
	return 0
}

func (x *AttemptAccepted) GetControlStreamEpoch() uint64 {
	if x != nil {
		return x.ControlStreamEpoch
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

func (x *AttemptAccepted) GetPlacementId() string {
	if x != nil {
		return x.PlacementId
	}
	return ""
}

func (x *AttemptAccepted) GetLaneId() string {
	if x != nil {
		return x.LaneId
	}
	return ""
}

// No wire carrier at minor 23: AttemptAccepted.plan (10) is reserved and plan facts ride the
// outcome body. Kept so the rev removes no descriptor; retire it in the minor that gives plan
// facts a carrier of their own.
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
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[92]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *AttemptPlanSummary) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*AttemptPlanSummary) ProtoMessage() {}

func (x *AttemptPlanSummary) ProtoReflect() protoreflect.Message {
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

// Deprecated: Use AttemptPlanSummary.ProtoReflect.Descriptor instead.
func (*AttemptPlanSummary) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{92}
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
	state                protoimpl.MessageState `protogen:"open.v1"`
	RecordOwnerEpoch     uint64                 `protobuf:"varint,1,opt,name=record_owner_epoch,json=recordOwnerEpoch,proto3" json:"record_owner_epoch,omitempty"`
	ControlStreamEpoch   uint64                 `protobuf:"varint,2,opt,name=control_stream_epoch,json=controlStreamEpoch,proto3" json:"control_stream_epoch,omitempty"`
	WorkerBootId         string                 `protobuf:"bytes,3,opt,name=worker_boot_id,json=workerBootId,proto3" json:"worker_boot_id,omitempty"`
	RequestId            string                 `protobuf:"bytes,5,opt,name=request_id,json=requestId,proto3" json:"request_id,omitempty"`
	AttemptOrdinal       uint64                 `protobuf:"varint,6,opt,name=attempt_ordinal,json=attemptOrdinal,proto3" json:"attempt_ordinal,omitempty"`
	Reason               CancelReason           `protobuf:"varint,7,opt,name=reason,proto3,enum=cozy.worker.v1.CancelReason" json:"reason,omitempty"`
	GraceMs              uint64                 `protobuf:"varint,8,opt,name=grace_ms,json=graceMs,proto3" json:"grace_ms,omitempty"`                                         // operator override; 0 = worker default. Never author-set.
	InvocationSpecDigest []byte                 `protobuf:"bytes,9,opt,name=invocation_spec_digest,json=invocationSpecDigest,proto3" json:"invocation_spec_digest,omitempty"` // full-triple fence; stale/mismatched cancel is DROPPED
	unknownFields        protoimpl.UnknownFields
	sizeCache            protoimpl.SizeCache
}

func (x *CancelAttempt) Reset() {
	*x = CancelAttempt{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[93]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *CancelAttempt) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*CancelAttempt) ProtoMessage() {}

func (x *CancelAttempt) ProtoReflect() protoreflect.Message {
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

// Deprecated: Use CancelAttempt.ProtoReflect.Descriptor instead.
func (*CancelAttempt) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{93}
}

func (x *CancelAttempt) GetRecordOwnerEpoch() uint64 {
	if x != nil {
		return x.RecordOwnerEpoch
	}
	return 0
}

func (x *CancelAttempt) GetControlStreamEpoch() uint64 {
	if x != nil {
		return x.ControlStreamEpoch
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
	state                 protoimpl.MessageState `protogen:"open.v1"`
	RecordOwnerEpoch      uint64                 `protobuf:"varint,1,opt,name=record_owner_epoch,json=recordOwnerEpoch,proto3" json:"record_owner_epoch,omitempty"`
	ControlStreamEpoch    uint64                 `protobuf:"varint,2,opt,name=control_stream_epoch,json=controlStreamEpoch,proto3" json:"control_stream_epoch,omitempty"`
	WorkerBootId          string                 `protobuf:"bytes,3,opt,name=worker_boot_id,json=workerBootId,proto3" json:"worker_boot_id,omitempty"`
	RequestId             string                 `protobuf:"bytes,5,opt,name=request_id,json=requestId,proto3" json:"request_id,omitempty"`                                        // routing copy; must agree with the document
	AttemptOrdinal        uint64                 `protobuf:"varint,6,opt,name=attempt_ordinal,json=attemptOrdinal,proto3" json:"attempt_ordinal,omitempty"`                        // routing copy
	InvocationSpecDigest  []byte                 `protobuf:"bytes,7,opt,name=invocation_spec_digest,json=invocationSpecDigest,proto3" json:"invocation_spec_digest,omitempty"`     // routing copy
	OutcomeId             string                 `protobuf:"bytes,8,opt,name=outcome_id,json=outcomeId,proto3" json:"outcome_id,omitempty"`                                        // WORKER-minted observation id; stable across replays
	OutcomeDigest         []byte                 `protobuf:"bytes,9,opt,name=outcome_digest,json=outcomeDigest,proto3" json:"outcome_digest,omitempty"`                            // class (a): sha256 over EXACTLY the bytes in 10
	OutcomeCanonicalBytes []byte                 `protobuf:"bytes,10,opt,name=outcome_canonical_bytes,json=outcomeCanonicalBytes,proto3" json:"outcome_canonical_bytes,omitempty"` // the AttemptOutcomeBody DOCUMENT, canonical JSON bytes
	PlacementId           string                 `protobuf:"bytes,11,opt,name=placement_id,json=placementId,proto3" json:"placement_id,omitempty"`                                 // routing
	unknownFields         protoimpl.UnknownFields
	sizeCache             protoimpl.SizeCache
}

func (x *AttemptOutcome) Reset() {
	*x = AttemptOutcome{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[94]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *AttemptOutcome) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*AttemptOutcome) ProtoMessage() {}

func (x *AttemptOutcome) ProtoReflect() protoreflect.Message {
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

// Deprecated: Use AttemptOutcome.ProtoReflect.Descriptor instead.
func (*AttemptOutcome) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{94}
}

func (x *AttemptOutcome) GetRecordOwnerEpoch() uint64 {
	if x != nil {
		return x.RecordOwnerEpoch
	}
	return 0
}

func (x *AttemptOutcome) GetControlStreamEpoch() uint64 {
	if x != nil {
		return x.ControlStreamEpoch
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
	WeightsReceipts []*WeightsReceiptRef `protobuf:"bytes,12,rep,name=weights_receipts,json=weightsReceipts,proto3" json:"weights_receipts,omitempty"` // committed job outputs only; sorted by slot
	unknownFields   protoimpl.UnknownFields
	sizeCache       protoimpl.SizeCache
}

func (x *AttemptOutcomeBody) Reset() {
	*x = AttemptOutcomeBody{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[95]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *AttemptOutcomeBody) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*AttemptOutcomeBody) ProtoMessage() {}

func (x *AttemptOutcomeBody) ProtoReflect() protoreflect.Message {
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

// Deprecated: Use AttemptOutcomeBody.ProtoReflect.Descriptor instead.
func (*AttemptOutcomeBody) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{95}
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

func (x *AttemptOutcomeBody) GetWeightsReceipts() []*WeightsReceiptRef {
	if x != nil {
		return x.WeightsReceipts
	}
	return nil
}

// Carried exact bytes, not a structured duplicate. The receiver hashes 2, compares 1, then opens
// the canonical WeightsReceipt document. This reference is nested in AttemptOutcomeBody/1 and
// therefore has no format tag of its own.
type WeightsReceiptRef struct {
	state                        protoimpl.MessageState `protogen:"open.v1"`
	WeightsReceiptDigest         []byte                 `protobuf:"bytes,1,opt,name=weights_receipt_digest,json=weightsReceiptDigest,proto3" json:"weights_receipt_digest,omitempty"`
	WeightsReceiptCanonicalBytes []byte                 `protobuf:"bytes,2,opt,name=weights_receipt_canonical_bytes,json=weightsReceiptCanonicalBytes,proto3" json:"weights_receipt_canonical_bytes,omitempty"`
	unknownFields                protoimpl.UnknownFields
	sizeCache                    protoimpl.SizeCache
}

func (x *WeightsReceiptRef) Reset() {
	*x = WeightsReceiptRef{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[96]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *WeightsReceiptRef) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*WeightsReceiptRef) ProtoMessage() {}

func (x *WeightsReceiptRef) ProtoReflect() protoreflect.Message {
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

// Deprecated: Use WeightsReceiptRef.ProtoReflect.Descriptor instead.
func (*WeightsReceiptRef) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{96}
}

func (x *WeightsReceiptRef) GetWeightsReceiptDigest() []byte {
	if x != nil {
		return x.WeightsReceiptDigest
	}
	return nil
}

func (x *WeightsReceiptRef) GetWeightsReceiptCanonicalBytes() []byte {
	if x != nil {
		return x.WeightsReceiptCanonicalBytes
	}
	return nil
}

// DOCUMENT SHAPE: canonical `cozy.worker.v1.WeightsReceipt/1`. TensorFS receipt bytes remain
// opaque to this protocol package; Runtime proves their digest before authoring this document.
type WeightsReceipt struct {
	state                         protoimpl.MessageState `protogen:"open.v1"`
	OwnerAuthorityScope           string                 `protobuf:"bytes,1,opt,name=owner_authority_scope,json=ownerAuthorityScope,proto3" json:"owner_authority_scope,omitempty"`
	RequestId                     string                 `protobuf:"bytes,2,opt,name=request_id,json=requestId,proto3" json:"request_id,omitempty"`
	InvocationSpecDigest          string                 `protobuf:"bytes,3,opt,name=invocation_spec_digest,json=invocationSpecDigest,proto3" json:"invocation_spec_digest,omitempty"`
	OutputSlot                    string                 `protobuf:"bytes,4,opt,name=output_slot,json=outputSlot,proto3" json:"output_slot,omitempty"`
	WeightsTransactionId          string                 `protobuf:"bytes,5,opt,name=weights_transaction_id,json=weightsTransactionId,proto3" json:"weights_transaction_id,omitempty"`
	TensorfsReceiptDigest         string                 `protobuf:"bytes,6,opt,name=tensorfs_receipt_digest,json=tensorfsReceiptDigest,proto3" json:"tensorfs_receipt_digest,omitempty"`
	TensorfsReceiptCanonicalBytes []byte                 `protobuf:"bytes,7,opt,name=tensorfs_receipt_canonical_bytes,json=tensorfsReceiptCanonicalBytes,proto3" json:"tensorfs_receipt_canonical_bytes,omitempty"`
	unknownFields                 protoimpl.UnknownFields
	sizeCache                     protoimpl.SizeCache
}

func (x *WeightsReceipt) Reset() {
	*x = WeightsReceipt{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[97]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *WeightsReceipt) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*WeightsReceipt) ProtoMessage() {}

func (x *WeightsReceipt) ProtoReflect() protoreflect.Message {
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

// Deprecated: Use WeightsReceipt.ProtoReflect.Descriptor instead.
func (*WeightsReceipt) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{97}
}

func (x *WeightsReceipt) GetOwnerAuthorityScope() string {
	if x != nil {
		return x.OwnerAuthorityScope
	}
	return ""
}

func (x *WeightsReceipt) GetRequestId() string {
	if x != nil {
		return x.RequestId
	}
	return ""
}

func (x *WeightsReceipt) GetInvocationSpecDigest() string {
	if x != nil {
		return x.InvocationSpecDigest
	}
	return ""
}

func (x *WeightsReceipt) GetOutputSlot() string {
	if x != nil {
		return x.OutputSlot
	}
	return ""
}

func (x *WeightsReceipt) GetWeightsTransactionId() string {
	if x != nil {
		return x.WeightsTransactionId
	}
	return ""
}

func (x *WeightsReceipt) GetTensorfsReceiptDigest() string {
	if x != nil {
		return x.TensorfsReceiptDigest
	}
	return ""
}

func (x *WeightsReceipt) GetTensorfsReceiptCanonicalBytes() []byte {
	if x != nil {
		return x.TensorfsReceiptCanonicalBytes
	}
	return nil
}

// One produced TensorFS ObjectRef plus an opaque transaction-scoped reader token. source_ref uses
// `<namespace>:<token>` printable ASCII with no slash, backslash, URI, or filesystem semantics.
type WeightsObjectSource struct {
	state         protoimpl.MessageState `protogen:"open.v1"`
	ObjectId      string                 `protobuf:"bytes,1,opt,name=object_id,json=objectId,proto3" json:"object_id,omitempty"`
	Length        uint64                 `protobuf:"varint,2,opt,name=length,proto3" json:"length,omitempty"`
	SourceRef     string                 `protobuf:"bytes,3,opt,name=source_ref,json=sourceRef,proto3" json:"source_ref,omitempty"`
	unknownFields protoimpl.UnknownFields
	sizeCache     protoimpl.SizeCache
}

func (x *WeightsObjectSource) Reset() {
	*x = WeightsObjectSource{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[98]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *WeightsObjectSource) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*WeightsObjectSource) ProtoMessage() {}

func (x *WeightsObjectSource) ProtoReflect() protoreflect.Message {
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

// Deprecated: Use WeightsObjectSource.ProtoReflect.Descriptor instead.
func (*WeightsObjectSource) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{98}
}

func (x *WeightsObjectSource) GetObjectId() string {
	if x != nil {
		return x.ObjectId
	}
	return ""
}

func (x *WeightsObjectSource) GetLength() uint64 {
	if x != nil {
		return x.Length
	}
	return 0
}

func (x *WeightsObjectSource) GetSourceRef() string {
	if x != nil {
		return x.SourceRef
	}
	return ""
}

// Runtime asks its authenticated host to commit semantic intent before the first TensorFS write.
// requested_writer_epoch == 0 means first open or replacement-child rejoin. Once ACKed, an
// identical retry from the same child echoes the epoch and receives the same ACK without
// fencing itself. The attempt ordinal is routing only and does not enter transaction identity.
// The TensorFS declaration is carried directly: it is not wrapped in a second Worker Protocol
// document. Its digest and exact bytes are the durable conflict/replay fence.
type WeightsIntentFrame struct {
	state                             protoimpl.MessageState `protogen:"open.v1"`
	RecordOwnerEpoch                  uint64                 `protobuf:"varint,1,opt,name=record_owner_epoch,json=recordOwnerEpoch,proto3" json:"record_owner_epoch,omitempty"`
	ControlStreamEpoch                uint64                 `protobuf:"varint,2,opt,name=control_stream_epoch,json=controlStreamEpoch,proto3" json:"control_stream_epoch,omitempty"`
	WorkerBootId                      string                 `protobuf:"bytes,3,opt,name=worker_boot_id,json=workerBootId,proto3" json:"worker_boot_id,omitempty"`
	RequestId                         string                 `protobuf:"bytes,5,opt,name=request_id,json=requestId,proto3" json:"request_id,omitempty"`
	AttemptOrdinal                    uint64                 `protobuf:"varint,6,opt,name=attempt_ordinal,json=attemptOrdinal,proto3" json:"attempt_ordinal,omitempty"`
	InvocationSpecDigest              []byte                 `protobuf:"bytes,7,opt,name=invocation_spec_digest,json=invocationSpecDigest,proto3" json:"invocation_spec_digest,omitempty"`
	OutputSlot                        string                 `protobuf:"bytes,8,opt,name=output_slot,json=outputSlot,proto3" json:"output_slot,omitempty"`
	TensorfsDeclarationDigest         []byte                 `protobuf:"bytes,9,opt,name=tensorfs_declaration_digest,json=tensorfsDeclarationDigest,proto3" json:"tensorfs_declaration_digest,omitempty"`
	RequestedWriterEpoch              uint64                 `protobuf:"varint,10,opt,name=requested_writer_epoch,json=requestedWriterEpoch,proto3" json:"requested_writer_epoch,omitempty"`
	TensorfsDeclarationCanonicalBytes []byte                 `protobuf:"bytes,11,opt,name=tensorfs_declaration_canonical_bytes,json=tensorfsDeclarationCanonicalBytes,proto3" json:"tensorfs_declaration_canonical_bytes,omitempty"`
	// Created only by Runtime. The host validates and durably binds this supplied ID to
	// owner/request/spec/slot, refuses conflicting bindings, and echoes it; it never hashes
	// a second transaction ID. Required lowercase sha256:<64 hex>.
	WeightsTransactionId string `protobuf:"bytes,12,opt,name=weights_transaction_id,json=weightsTransactionId,proto3" json:"weights_transaction_id,omitempty"`
	unknownFields        protoimpl.UnknownFields
	sizeCache            protoimpl.SizeCache
}

func (x *WeightsIntentFrame) Reset() {
	*x = WeightsIntentFrame{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[99]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *WeightsIntentFrame) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*WeightsIntentFrame) ProtoMessage() {}

func (x *WeightsIntentFrame) ProtoReflect() protoreflect.Message {
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

// Deprecated: Use WeightsIntentFrame.ProtoReflect.Descriptor instead.
func (*WeightsIntentFrame) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{99}
}

func (x *WeightsIntentFrame) GetRecordOwnerEpoch() uint64 {
	if x != nil {
		return x.RecordOwnerEpoch
	}
	return 0
}

func (x *WeightsIntentFrame) GetControlStreamEpoch() uint64 {
	if x != nil {
		return x.ControlStreamEpoch
	}
	return 0
}

func (x *WeightsIntentFrame) GetWorkerBootId() string {
	if x != nil {
		return x.WorkerBootId
	}
	return ""
}

func (x *WeightsIntentFrame) GetRequestId() string {
	if x != nil {
		return x.RequestId
	}
	return ""
}

func (x *WeightsIntentFrame) GetAttemptOrdinal() uint64 {
	if x != nil {
		return x.AttemptOrdinal
	}
	return 0
}

func (x *WeightsIntentFrame) GetInvocationSpecDigest() []byte {
	if x != nil {
		return x.InvocationSpecDigest
	}
	return nil
}

func (x *WeightsIntentFrame) GetOutputSlot() string {
	if x != nil {
		return x.OutputSlot
	}
	return ""
}

func (x *WeightsIntentFrame) GetTensorfsDeclarationDigest() []byte {
	if x != nil {
		return x.TensorfsDeclarationDigest
	}
	return nil
}

func (x *WeightsIntentFrame) GetRequestedWriterEpoch() uint64 {
	if x != nil {
		return x.RequestedWriterEpoch
	}
	return 0
}

func (x *WeightsIntentFrame) GetTensorfsDeclarationCanonicalBytes() []byte {
	if x != nil {
		return x.TensorfsDeclarationCanonicalBytes
	}
	return nil
}

func (x *WeightsIntentFrame) GetWeightsTransactionId() string {
	if x != nil {
		return x.WeightsTransactionId
	}
	return ""
}

// One ACK type covers intent and receipt durability. The supervisor injects it into Runtime only
// after its one SQLite transaction commits. A replacement child receives the same transaction id
// and a newly fenced writer epoch; if a receipt already exists it is replayed here and no
// TensorFS write occurs. REFUSED carries no usable epoch or transaction authority.
type WeightsHostAck struct {
	state                     protoimpl.MessageState `protogen:"open.v1"`
	RecordOwnerEpoch          uint64                 `protobuf:"varint,1,opt,name=record_owner_epoch,json=recordOwnerEpoch,proto3" json:"record_owner_epoch,omitempty"`
	ControlStreamEpoch        uint64                 `protobuf:"varint,2,opt,name=control_stream_epoch,json=controlStreamEpoch,proto3" json:"control_stream_epoch,omitempty"`
	WorkerBootId              string                 `protobuf:"bytes,3,opt,name=worker_boot_id,json=workerBootId,proto3" json:"worker_boot_id,omitempty"`
	RequestId                 string                 `protobuf:"bytes,5,opt,name=request_id,json=requestId,proto3" json:"request_id,omitempty"`
	AttemptOrdinal            uint64                 `protobuf:"varint,6,opt,name=attempt_ordinal,json=attemptOrdinal,proto3" json:"attempt_ordinal,omitempty"`
	InvocationSpecDigest      []byte                 `protobuf:"bytes,7,opt,name=invocation_spec_digest,json=invocationSpecDigest,proto3" json:"invocation_spec_digest,omitempty"`
	OutputSlot                string                 `protobuf:"bytes,8,opt,name=output_slot,json=outputSlot,proto3" json:"output_slot,omitempty"`
	WeightsTransactionId      string                 `protobuf:"bytes,9,opt,name=weights_transaction_id,json=weightsTransactionId,proto3" json:"weights_transaction_id,omitempty"`
	WriterEpoch               uint64                 `protobuf:"varint,10,opt,name=writer_epoch,json=writerEpoch,proto3" json:"writer_epoch,omitempty"`
	TensorfsDeclarationDigest []byte                 `protobuf:"bytes,11,opt,name=tensorfs_declaration_digest,json=tensorfsDeclarationDigest,proto3" json:"tensorfs_declaration_digest,omitempty"`
	Stage                     WeightsHostStage       `protobuf:"varint,12,opt,name=stage,proto3,enum=cozy.worker.v1.WeightsHostStage" json:"stage,omitempty"`
	Outcome                   WeightsHostOutcome     `protobuf:"varint,13,opt,name=outcome,proto3,enum=cozy.worker.v1.WeightsHostOutcome" json:"outcome,omitempty"`
	Refusal                   WeightsHostRefusal     `protobuf:"varint,14,opt,name=refusal,proto3,enum=cozy.worker.v1.WeightsHostRefusal" json:"refusal,omitempty"`
	WeightsReceipt            *WeightsReceiptRef     `protobuf:"bytes,15,opt,name=weights_receipt,json=weightsReceipt,proto3" json:"weights_receipt,omitempty"`
	Manifest                  *Ref                   `protobuf:"bytes,16,opt,name=manifest,proto3" json:"manifest,omitempty"` // present exactly with weights_receipt; the produced Manifest identity
	// downstream nodes consume without parsing opaque TensorFS receipt bytes
	Checkpoint    *CheckpointRef `protobuf:"bytes,17,opt,name=checkpoint,proto3" json:"checkpoint,omitempty"` // INTENT: natively validated restore head; CHECKPOINT: exact
	unknownFields protoimpl.UnknownFields
	sizeCache     protoimpl.SizeCache
}

func (x *WeightsHostAck) Reset() {
	*x = WeightsHostAck{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[100]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *WeightsHostAck) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*WeightsHostAck) ProtoMessage() {}

func (x *WeightsHostAck) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[100]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use WeightsHostAck.ProtoReflect.Descriptor instead.
func (*WeightsHostAck) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{100}
}

func (x *WeightsHostAck) GetRecordOwnerEpoch() uint64 {
	if x != nil {
		return x.RecordOwnerEpoch
	}
	return 0
}

func (x *WeightsHostAck) GetControlStreamEpoch() uint64 {
	if x != nil {
		return x.ControlStreamEpoch
	}
	return 0
}

func (x *WeightsHostAck) GetWorkerBootId() string {
	if x != nil {
		return x.WorkerBootId
	}
	return ""
}

func (x *WeightsHostAck) GetRequestId() string {
	if x != nil {
		return x.RequestId
	}
	return ""
}

func (x *WeightsHostAck) GetAttemptOrdinal() uint64 {
	if x != nil {
		return x.AttemptOrdinal
	}
	return 0
}

func (x *WeightsHostAck) GetInvocationSpecDigest() []byte {
	if x != nil {
		return x.InvocationSpecDigest
	}
	return nil
}

func (x *WeightsHostAck) GetOutputSlot() string {
	if x != nil {
		return x.OutputSlot
	}
	return ""
}

func (x *WeightsHostAck) GetWeightsTransactionId() string {
	if x != nil {
		return x.WeightsTransactionId
	}
	return ""
}

func (x *WeightsHostAck) GetWriterEpoch() uint64 {
	if x != nil {
		return x.WriterEpoch
	}
	return 0
}

func (x *WeightsHostAck) GetTensorfsDeclarationDigest() []byte {
	if x != nil {
		return x.TensorfsDeclarationDigest
	}
	return nil
}

func (x *WeightsHostAck) GetStage() WeightsHostStage {
	if x != nil {
		return x.Stage
	}
	return WeightsHostStage_WEIGHTS_HOST_STAGE_UNSPECIFIED
}

func (x *WeightsHostAck) GetOutcome() WeightsHostOutcome {
	if x != nil {
		return x.Outcome
	}
	return WeightsHostOutcome_WEIGHTS_HOST_OUTCOME_UNSPECIFIED
}

func (x *WeightsHostAck) GetRefusal() WeightsHostRefusal {
	if x != nil {
		return x.Refusal
	}
	return WeightsHostRefusal_WEIGHTS_HOST_REFUSAL_UNSPECIFIED
}

func (x *WeightsHostAck) GetWeightsReceipt() *WeightsReceiptRef {
	if x != nil {
		return x.WeightsReceipt
	}
	return nil
}

func (x *WeightsHostAck) GetManifest() *Ref {
	if x != nil {
		return x.Manifest
	}
	return nil
}

func (x *WeightsHostAck) GetCheckpoint() *CheckpointRef {
	if x != nil {
		return x.Checkpoint
	}
	return nil
}

// Fixed-size TensorFS Link facts. No object inventory or model-growing role list rides here.
type CheckpointRef struct {
	state         protoimpl.MessageState `protogen:"open.v1"`
	Head          *Ref                   `protobuf:"bytes,1,opt,name=head,proto3" json:"head,omitempty"`
	PlanDigest    []byte                 `protobuf:"bytes,2,opt,name=plan_digest,json=planDigest,proto3" json:"plan_digest,omitempty"`
	Index         uint64                 `protobuf:"varint,3,opt,name=index,proto3" json:"index,omitempty"`
	Bytes         uint64                 `protobuf:"varint,4,opt,name=bytes,proto3" json:"bytes,omitempty"` // cumulative locally accepted bytes; not a remote durability receipt
	unknownFields protoimpl.UnknownFields
	sizeCache     protoimpl.SizeCache
}

func (x *CheckpointRef) Reset() {
	*x = CheckpointRef{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[101]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *CheckpointRef) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*CheckpointRef) ProtoMessage() {}

func (x *CheckpointRef) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[101]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use CheckpointRef.ProtoReflect.Descriptor instead.
func (*CheckpointRef) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{101}
}

func (x *CheckpointRef) GetHead() *Ref {
	if x != nil {
		return x.Head
	}
	return nil
}

func (x *CheckpointRef) GetPlanDigest() []byte {
	if x != nil {
		return x.PlanDigest
	}
	return nil
}

func (x *CheckpointRef) GetIndex() uint64 {
	if x != nil {
		return x.Index
	}
	return 0
}

func (x *CheckpointRef) GetBytes() uint64 {
	if x != nil {
		return x.Bytes
	}
	return 0
}

// Runtime -> host -> authenticated owner, journaled under the existing weights intent before
// ACK. The owner banks this Link closure with CheckpointPage/Transfer and retains its own
// remotely acknowledged prefix separately from this locally observed progress.
type WeightsCheckpointFrame struct {
	state                     protoimpl.MessageState `protogen:"open.v1"`
	RecordOwnerEpoch          uint64                 `protobuf:"varint,1,opt,name=record_owner_epoch,json=recordOwnerEpoch,proto3" json:"record_owner_epoch,omitempty"`
	ControlStreamEpoch        uint64                 `protobuf:"varint,2,opt,name=control_stream_epoch,json=controlStreamEpoch,proto3" json:"control_stream_epoch,omitempty"`
	WorkerBootId              string                 `protobuf:"bytes,3,opt,name=worker_boot_id,json=workerBootId,proto3" json:"worker_boot_id,omitempty"`
	RequestId                 string                 `protobuf:"bytes,5,opt,name=request_id,json=requestId,proto3" json:"request_id,omitempty"`
	AttemptOrdinal            uint64                 `protobuf:"varint,6,opt,name=attempt_ordinal,json=attemptOrdinal,proto3" json:"attempt_ordinal,omitempty"`
	InvocationSpecDigest      []byte                 `protobuf:"bytes,7,opt,name=invocation_spec_digest,json=invocationSpecDigest,proto3" json:"invocation_spec_digest,omitempty"`
	OutputSlot                string                 `protobuf:"bytes,8,opt,name=output_slot,json=outputSlot,proto3" json:"output_slot,omitempty"`
	WeightsTransactionId      string                 `protobuf:"bytes,9,opt,name=weights_transaction_id,json=weightsTransactionId,proto3" json:"weights_transaction_id,omitempty"`
	WriterEpoch               uint64                 `protobuf:"varint,10,opt,name=writer_epoch,json=writerEpoch,proto3" json:"writer_epoch,omitempty"`
	TensorfsDeclarationDigest []byte                 `protobuf:"bytes,11,opt,name=tensorfs_declaration_digest,json=tensorfsDeclarationDigest,proto3" json:"tensorfs_declaration_digest,omitempty"`
	Checkpoint                *CheckpointRef         `protobuf:"bytes,12,opt,name=checkpoint,proto3" json:"checkpoint,omitempty"`
	unknownFields             protoimpl.UnknownFields
	sizeCache                 protoimpl.SizeCache
}

func (x *WeightsCheckpointFrame) Reset() {
	*x = WeightsCheckpointFrame{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[102]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *WeightsCheckpointFrame) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*WeightsCheckpointFrame) ProtoMessage() {}

func (x *WeightsCheckpointFrame) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[102]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use WeightsCheckpointFrame.ProtoReflect.Descriptor instead.
func (*WeightsCheckpointFrame) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{102}
}

func (x *WeightsCheckpointFrame) GetRecordOwnerEpoch() uint64 {
	if x != nil {
		return x.RecordOwnerEpoch
	}
	return 0
}

func (x *WeightsCheckpointFrame) GetControlStreamEpoch() uint64 {
	if x != nil {
		return x.ControlStreamEpoch
	}
	return 0
}

func (x *WeightsCheckpointFrame) GetWorkerBootId() string {
	if x != nil {
		return x.WorkerBootId
	}
	return ""
}

func (x *WeightsCheckpointFrame) GetRequestId() string {
	if x != nil {
		return x.RequestId
	}
	return ""
}

func (x *WeightsCheckpointFrame) GetAttemptOrdinal() uint64 {
	if x != nil {
		return x.AttemptOrdinal
	}
	return 0
}

func (x *WeightsCheckpointFrame) GetInvocationSpecDigest() []byte {
	if x != nil {
		return x.InvocationSpecDigest
	}
	return nil
}

func (x *WeightsCheckpointFrame) GetOutputSlot() string {
	if x != nil {
		return x.OutputSlot
	}
	return ""
}

func (x *WeightsCheckpointFrame) GetWeightsTransactionId() string {
	if x != nil {
		return x.WeightsTransactionId
	}
	return ""
}

func (x *WeightsCheckpointFrame) GetWriterEpoch() uint64 {
	if x != nil {
		return x.WriterEpoch
	}
	return 0
}

func (x *WeightsCheckpointFrame) GetTensorfsDeclarationDigest() []byte {
	if x != nil {
		return x.TensorfsDeclarationDigest
	}
	return nil
}

func (x *WeightsCheckpointFrame) GetCheckpoint() *CheckpointRef {
	if x != nil {
		return x.Checkpoint
	}
	return nil
}

// A current intent is registered and externally observable BEFORE writer ACK. The owner
// restores its banked Link under that intent's exact authority, then calls Ready once.
// An absent checkpoint explicitly chooses no external restore; verified local progress stays.
// Same intent/head replays; changed head conflicts. Cancellation tombstones even while waiting.
type WeightsIntentReadyRequest struct {
	state              protoimpl.MessageState    `protogen:"open.v1"`
	RecordOwnerEpoch   uint64                    `protobuf:"varint,1,opt,name=record_owner_epoch,json=recordOwnerEpoch,proto3" json:"record_owner_epoch,omitempty"`
	ControlStreamEpoch uint64                    `protobuf:"varint,2,opt,name=control_stream_epoch,json=controlStreamEpoch,proto3" json:"control_stream_epoch,omitempty"`
	WorkerBootId       string                    `protobuf:"bytes,3,opt,name=worker_boot_id,json=workerBootId,proto3" json:"worker_boot_id,omitempty"`
	Weights            *WeightsCheckpointSubject `protobuf:"bytes,5,opt,name=weights,proto3" json:"weights,omitempty"`
	AttemptOrdinal     uint64                    `protobuf:"varint,6,opt,name=attempt_ordinal,json=attemptOrdinal,proto3" json:"attempt_ordinal,omitempty"`
	Checkpoint         *CheckpointRef            `protobuf:"bytes,7,opt,name=checkpoint,proto3" json:"checkpoint,omitempty"`
	unknownFields      protoimpl.UnknownFields
	sizeCache          protoimpl.SizeCache
}

func (x *WeightsIntentReadyRequest) Reset() {
	*x = WeightsIntentReadyRequest{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[103]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *WeightsIntentReadyRequest) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*WeightsIntentReadyRequest) ProtoMessage() {}

func (x *WeightsIntentReadyRequest) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[103]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use WeightsIntentReadyRequest.ProtoReflect.Descriptor instead.
func (*WeightsIntentReadyRequest) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{103}
}

func (x *WeightsIntentReadyRequest) GetRecordOwnerEpoch() uint64 {
	if x != nil {
		return x.RecordOwnerEpoch
	}
	return 0
}

func (x *WeightsIntentReadyRequest) GetControlStreamEpoch() uint64 {
	if x != nil {
		return x.ControlStreamEpoch
	}
	return 0
}

func (x *WeightsIntentReadyRequest) GetWorkerBootId() string {
	if x != nil {
		return x.WorkerBootId
	}
	return ""
}

func (x *WeightsIntentReadyRequest) GetWeights() *WeightsCheckpointSubject {
	if x != nil {
		return x.Weights
	}
	return nil
}

func (x *WeightsIntentReadyRequest) GetAttemptOrdinal() uint64 {
	if x != nil {
		return x.AttemptOrdinal
	}
	return 0
}

func (x *WeightsIntentReadyRequest) GetCheckpoint() *CheckpointRef {
	if x != nil {
		return x.Checkpoint
	}
	return nil
}

type WeightsIntentReadyCall struct {
	state         protoimpl.MessageState     `protogen:"open.v1"`
	Claim         *Claim                     `protobuf:"bytes,1,opt,name=claim,proto3" json:"claim,omitempty"`
	Request       *WeightsIntentReadyRequest `protobuf:"bytes,2,opt,name=request,proto3" json:"request,omitempty"`
	unknownFields protoimpl.UnknownFields
	sizeCache     protoimpl.SizeCache
}

func (x *WeightsIntentReadyCall) Reset() {
	*x = WeightsIntentReadyCall{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[104]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *WeightsIntentReadyCall) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*WeightsIntentReadyCall) ProtoMessage() {}

func (x *WeightsIntentReadyCall) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[104]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use WeightsIntentReadyCall.ProtoReflect.Descriptor instead.
func (*WeightsIntentReadyCall) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{104}
}

func (x *WeightsIntentReadyCall) GetClaim() *Claim {
	if x != nil {
		return x.Claim
	}
	return nil
}

func (x *WeightsIntentReadyCall) GetRequest() *WeightsIntentReadyRequest {
	if x != nil {
		return x.Request
	}
	return nil
}

// Loopback only: the host adds the exact TensorFS-authored declaration already held by its
// current intent. Runtime verifies its digest and native checkpoint closure before success.
// This never opens/fences/imports a writer. The host journals Ready and releases its existing
// INTENT ACK only after this synchronous verdict. Complete request <= MaxInlineControlBytes.
type ValidateWeightsCheckpointRequest struct {
	state                             protoimpl.MessageState     `protogen:"open.v1"`
	Intent                            *WeightsIntentReadyRequest `protobuf:"bytes,1,opt,name=intent,proto3" json:"intent,omitempty"`
	TensorfsDeclarationCanonicalBytes []byte                     `protobuf:"bytes,2,opt,name=tensorfs_declaration_canonical_bytes,json=tensorfsDeclarationCanonicalBytes,proto3" json:"tensorfs_declaration_canonical_bytes,omitempty"` // <= MaxWeightsDeclarationBytes
	unknownFields                     protoimpl.UnknownFields
	sizeCache                         protoimpl.SizeCache
}

func (x *ValidateWeightsCheckpointRequest) Reset() {
	*x = ValidateWeightsCheckpointRequest{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[105]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *ValidateWeightsCheckpointRequest) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*ValidateWeightsCheckpointRequest) ProtoMessage() {}

func (x *ValidateWeightsCheckpointRequest) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[105]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use ValidateWeightsCheckpointRequest.ProtoReflect.Descriptor instead.
func (*ValidateWeightsCheckpointRequest) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{105}
}

func (x *ValidateWeightsCheckpointRequest) GetIntent() *WeightsIntentReadyRequest {
	if x != nil {
		return x.Intent
	}
	return nil
}

func (x *ValidateWeightsCheckpointRequest) GetTensorfsDeclarationCanonicalBytes() []byte {
	if x != nil {
		return x.TensorfsDeclarationCanonicalBytes
	}
	return nil
}

type ValidateWeightsCheckpointResult struct {
	state         protoimpl.MessageState    `protogen:"open.v1"`
	Weights       *WeightsCheckpointSubject `protobuf:"bytes,1,opt,name=weights,proto3" json:"weights,omitempty"`
	Checkpoint    *CheckpointRef            `protobuf:"bytes,2,opt,name=checkpoint,proto3" json:"checkpoint,omitempty"`
	Valid         bool                      `protobuf:"varint,3,opt,name=valid,proto3" json:"valid,omitempty"`
	SafeCode      string                    `protobuf:"bytes,4,opt,name=safe_code,json=safeCode,proto3" json:"safe_code,omitempty"`
	SafeDetail    string                    `protobuf:"bytes,5,opt,name=safe_detail,json=safeDetail,proto3" json:"safe_detail,omitempty"`
	unknownFields protoimpl.UnknownFields
	sizeCache     protoimpl.SizeCache
}

func (x *ValidateWeightsCheckpointResult) Reset() {
	*x = ValidateWeightsCheckpointResult{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[106]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *ValidateWeightsCheckpointResult) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*ValidateWeightsCheckpointResult) ProtoMessage() {}

func (x *ValidateWeightsCheckpointResult) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[106]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use ValidateWeightsCheckpointResult.ProtoReflect.Descriptor instead.
func (*ValidateWeightsCheckpointResult) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{106}
}

func (x *ValidateWeightsCheckpointResult) GetWeights() *WeightsCheckpointSubject {
	if x != nil {
		return x.Weights
	}
	return nil
}

func (x *ValidateWeightsCheckpointResult) GetCheckpoint() *CheckpointRef {
	if x != nil {
		return x.Checkpoint
	}
	return nil
}

func (x *ValidateWeightsCheckpointResult) GetValid() bool {
	if x != nil {
		return x.Valid
	}
	return false
}

func (x *ValidateWeightsCheckpointResult) GetSafeCode() string {
	if x != nil {
		return x.SafeCode
	}
	return ""
}

func (x *ValidateWeightsCheckpointResult) GetSafeDetail() string {
	if x != nil {
		return x.SafeDetail
	}
	return ""
}

// Runtime sends the exact committed receipt and exact local source inventory together. The
// supervisor validates all routing copies and writer_epoch, commits both before ACK, and then
// may forward this frame to the external RecordOwner. A stale child cannot publish a late receipt.
type WeightsReceiptFrame struct {
	state                     protoimpl.MessageState `protogen:"open.v1"`
	RecordOwnerEpoch          uint64                 `protobuf:"varint,1,opt,name=record_owner_epoch,json=recordOwnerEpoch,proto3" json:"record_owner_epoch,omitempty"`
	ControlStreamEpoch        uint64                 `protobuf:"varint,2,opt,name=control_stream_epoch,json=controlStreamEpoch,proto3" json:"control_stream_epoch,omitempty"`
	WorkerBootId              string                 `protobuf:"bytes,3,opt,name=worker_boot_id,json=workerBootId,proto3" json:"worker_boot_id,omitempty"`
	RequestId                 string                 `protobuf:"bytes,5,opt,name=request_id,json=requestId,proto3" json:"request_id,omitempty"`
	AttemptOrdinal            uint64                 `protobuf:"varint,6,opt,name=attempt_ordinal,json=attemptOrdinal,proto3" json:"attempt_ordinal,omitempty"`
	InvocationSpecDigest      []byte                 `protobuf:"bytes,7,opt,name=invocation_spec_digest,json=invocationSpecDigest,proto3" json:"invocation_spec_digest,omitempty"`
	OutputSlot                string                 `protobuf:"bytes,8,opt,name=output_slot,json=outputSlot,proto3" json:"output_slot,omitempty"`
	WeightsTransactionId      string                 `protobuf:"bytes,9,opt,name=weights_transaction_id,json=weightsTransactionId,proto3" json:"weights_transaction_id,omitempty"`
	WriterEpoch               uint64                 `protobuf:"varint,10,opt,name=writer_epoch,json=writerEpoch,proto3" json:"writer_epoch,omitempty"`
	TensorfsDeclarationDigest []byte                 `protobuf:"bytes,11,opt,name=tensorfs_declaration_digest,json=tensorfsDeclarationDigest,proto3" json:"tensorfs_declaration_digest,omitempty"`
	WeightsReceipt            *WeightsReceiptRef     `protobuf:"bytes,12,opt,name=weights_receipt,json=weightsReceipt,proto3" json:"weights_receipt,omitempty"`
	Objects                   []*WeightsObjectSource `protobuf:"bytes,13,rep,name=objects,proto3" json:"objects,omitempty"` // sorted uniquely by object_id; committed in the
	// same host-ledger transaction as weights_receipt
	Manifest      *Ref `protobuf:"bytes,14,opt,name=manifest,proto3" json:"manifest,omitempty"` // exact produced TensorFS Manifest; it is also one row in objects
	unknownFields protoimpl.UnknownFields
	sizeCache     protoimpl.SizeCache
}

func (x *WeightsReceiptFrame) Reset() {
	*x = WeightsReceiptFrame{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[107]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *WeightsReceiptFrame) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*WeightsReceiptFrame) ProtoMessage() {}

func (x *WeightsReceiptFrame) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[107]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use WeightsReceiptFrame.ProtoReflect.Descriptor instead.
func (*WeightsReceiptFrame) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{107}
}

func (x *WeightsReceiptFrame) GetRecordOwnerEpoch() uint64 {
	if x != nil {
		return x.RecordOwnerEpoch
	}
	return 0
}

func (x *WeightsReceiptFrame) GetControlStreamEpoch() uint64 {
	if x != nil {
		return x.ControlStreamEpoch
	}
	return 0
}

func (x *WeightsReceiptFrame) GetWorkerBootId() string {
	if x != nil {
		return x.WorkerBootId
	}
	return ""
}

func (x *WeightsReceiptFrame) GetRequestId() string {
	if x != nil {
		return x.RequestId
	}
	return ""
}

func (x *WeightsReceiptFrame) GetAttemptOrdinal() uint64 {
	if x != nil {
		return x.AttemptOrdinal
	}
	return 0
}

func (x *WeightsReceiptFrame) GetInvocationSpecDigest() []byte {
	if x != nil {
		return x.InvocationSpecDigest
	}
	return nil
}

func (x *WeightsReceiptFrame) GetOutputSlot() string {
	if x != nil {
		return x.OutputSlot
	}
	return ""
}

func (x *WeightsReceiptFrame) GetWeightsTransactionId() string {
	if x != nil {
		return x.WeightsTransactionId
	}
	return ""
}

func (x *WeightsReceiptFrame) GetWriterEpoch() uint64 {
	if x != nil {
		return x.WriterEpoch
	}
	return 0
}

func (x *WeightsReceiptFrame) GetTensorfsDeclarationDigest() []byte {
	if x != nil {
		return x.TensorfsDeclarationDigest
	}
	return nil
}

func (x *WeightsReceiptFrame) GetWeightsReceipt() *WeightsReceiptRef {
	if x != nil {
		return x.WeightsReceipt
	}
	return nil
}

func (x *WeightsReceiptFrame) GetObjects() []*WeightsObjectSource {
	if x != nil {
		return x.Objects
	}
	return nil
}

func (x *WeightsReceiptFrame) GetManifest() *Ref {
	if x != nil {
		return x.Manifest
	}
	return nil
}

// Compact supervisor-ledger state included in WorkerSnapshotBody/1. A stateless replacement
// Runtime does not author these rows; pod-supervisor injects them before digesting the external
// snapshot. It MUST NOT put declarations, receipt bytes, or object inventory here. After the
// snapshot ACK, the supervisor replays each RECEIPT transaction's exact WeightsReceiptFrame.
type WeightsTransactionStatus struct {
	state                     protoimpl.MessageState  `protogen:"open.v1"`
	WeightsTransactionId      string                  `protobuf:"bytes,1,opt,name=weights_transaction_id,json=weightsTransactionId,proto3" json:"weights_transaction_id,omitempty"`
	RequestId                 string                  `protobuf:"bytes,2,opt,name=request_id,json=requestId,proto3" json:"request_id,omitempty"`
	AttemptOrdinal            uint64                  `protobuf:"varint,3,opt,name=attempt_ordinal,json=attemptOrdinal,proto3" json:"attempt_ordinal,omitempty"`
	InvocationSpecDigest      string                  `protobuf:"bytes,4,opt,name=invocation_spec_digest,json=invocationSpecDigest,proto3" json:"invocation_spec_digest,omitempty"`
	OutputSlot                string                  `protobuf:"bytes,5,opt,name=output_slot,json=outputSlot,proto3" json:"output_slot,omitempty"`
	WriterEpoch               uint64                  `protobuf:"varint,6,opt,name=writer_epoch,json=writerEpoch,proto3" json:"writer_epoch,omitempty"`
	State                     WeightsTransactionState `protobuf:"varint,7,opt,name=state,proto3,enum=cozy.worker.v1.WeightsTransactionState" json:"state,omitempty"`
	TensorfsDeclarationDigest []byte                  `protobuf:"bytes,8,opt,name=tensorfs_declaration_digest,json=tensorfsDeclarationDigest,proto3" json:"tensorfs_declaration_digest,omitempty"`
	WeightsReceiptDigest      []byte                  `protobuf:"bytes,9,opt,name=weights_receipt_digest,json=weightsReceiptDigest,proto3" json:"weights_receipt_digest,omitempty"` // absent in INTENT; exact replay fence in RECEIPT
	Checkpoint                *CheckpointRef          `protobuf:"bytes,10,opt,name=checkpoint,proto3" json:"checkpoint,omitempty"`                                                  // latest locally recorded progress, not remote custody
	IntentReady               bool                    `protobuf:"varint,11,opt,name=intent_ready,json=intentReady,proto3" json:"intent_ready,omitempty"`                            // the owner has finished restore (or explicitly chose no restore)
	unknownFields             protoimpl.UnknownFields
	sizeCache                 protoimpl.SizeCache
}

func (x *WeightsTransactionStatus) Reset() {
	*x = WeightsTransactionStatus{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[108]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *WeightsTransactionStatus) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*WeightsTransactionStatus) ProtoMessage() {}

func (x *WeightsTransactionStatus) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[108]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use WeightsTransactionStatus.ProtoReflect.Descriptor instead.
func (*WeightsTransactionStatus) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{108}
}

func (x *WeightsTransactionStatus) GetWeightsTransactionId() string {
	if x != nil {
		return x.WeightsTransactionId
	}
	return ""
}

func (x *WeightsTransactionStatus) GetRequestId() string {
	if x != nil {
		return x.RequestId
	}
	return ""
}

func (x *WeightsTransactionStatus) GetAttemptOrdinal() uint64 {
	if x != nil {
		return x.AttemptOrdinal
	}
	return 0
}

func (x *WeightsTransactionStatus) GetInvocationSpecDigest() string {
	if x != nil {
		return x.InvocationSpecDigest
	}
	return ""
}

func (x *WeightsTransactionStatus) GetOutputSlot() string {
	if x != nil {
		return x.OutputSlot
	}
	return ""
}

func (x *WeightsTransactionStatus) GetWriterEpoch() uint64 {
	if x != nil {
		return x.WriterEpoch
	}
	return 0
}

func (x *WeightsTransactionStatus) GetState() WeightsTransactionState {
	if x != nil {
		return x.State
	}
	return WeightsTransactionState_WEIGHTS_TRANSACTION_STATE_UNSPECIFIED
}

func (x *WeightsTransactionStatus) GetTensorfsDeclarationDigest() []byte {
	if x != nil {
		return x.TensorfsDeclarationDigest
	}
	return nil
}

func (x *WeightsTransactionStatus) GetWeightsReceiptDigest() []byte {
	if x != nil {
		return x.WeightsReceiptDigest
	}
	return nil
}

func (x *WeightsTransactionStatus) GetCheckpoint() *CheckpointRef {
	if x != nil {
		return x.Checkpoint
	}
	return nil
}

func (x *WeightsTransactionStatus) GetIntentReady() bool {
	if x != nil {
		return x.IntentReady
	}
	return false
}

// A supervisor or authenticated external RecordOwner reads a produced object without receiving a
// path or giving Runtime a network destination. The supervisor first binds one exact bounded range
// to the committed receipt inventory, then forwards this same frame to Runtime. read_id is unique
// among pending reads on one claimed stream and routes the exact result; it is not durable state.
type WeightsReadRequest struct {
	state                protoimpl.MessageState `protogen:"open.v1"`
	RecordOwnerEpoch     uint64                 `protobuf:"varint,1,opt,name=record_owner_epoch,json=recordOwnerEpoch,proto3" json:"record_owner_epoch,omitempty"`
	ControlStreamEpoch   uint64                 `protobuf:"varint,2,opt,name=control_stream_epoch,json=controlStreamEpoch,proto3" json:"control_stream_epoch,omitempty"`
	WorkerBootId         string                 `protobuf:"bytes,3,opt,name=worker_boot_id,json=workerBootId,proto3" json:"worker_boot_id,omitempty"`
	WeightsTransactionId string                 `protobuf:"bytes,5,opt,name=weights_transaction_id,json=weightsTransactionId,proto3" json:"weights_transaction_id,omitempty"`
	WriterEpoch          uint64                 `protobuf:"varint,6,opt,name=writer_epoch,json=writerEpoch,proto3" json:"writer_epoch,omitempty"`
	ObjectId             string                 `protobuf:"bytes,7,opt,name=object_id,json=objectId,proto3" json:"object_id,omitempty"`
	SourceRef            string                 `protobuf:"bytes,8,opt,name=source_ref,json=sourceRef,proto3" json:"source_ref,omitempty"`
	ReadId               string                 `protobuf:"bytes,9,opt,name=read_id,json=readId,proto3" json:"read_id,omitempty"`
	Offset               uint64                 `protobuf:"varint,10,opt,name=offset,proto3" json:"offset,omitempty"`
	MaxBytes             uint32                 `protobuf:"varint,11,opt,name=max_bytes,json=maxBytes,proto3" json:"max_bytes,omitempty"`
	unknownFields        protoimpl.UnknownFields
	sizeCache            protoimpl.SizeCache
}

func (x *WeightsReadRequest) Reset() {
	*x = WeightsReadRequest{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[109]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *WeightsReadRequest) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*WeightsReadRequest) ProtoMessage() {}

func (x *WeightsReadRequest) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[109]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use WeightsReadRequest.ProtoReflect.Descriptor instead.
func (*WeightsReadRequest) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{109}
}

func (x *WeightsReadRequest) GetRecordOwnerEpoch() uint64 {
	if x != nil {
		return x.RecordOwnerEpoch
	}
	return 0
}

func (x *WeightsReadRequest) GetControlStreamEpoch() uint64 {
	if x != nil {
		return x.ControlStreamEpoch
	}
	return 0
}

func (x *WeightsReadRequest) GetWorkerBootId() string {
	if x != nil {
		return x.WorkerBootId
	}
	return ""
}

func (x *WeightsReadRequest) GetWeightsTransactionId() string {
	if x != nil {
		return x.WeightsTransactionId
	}
	return ""
}

func (x *WeightsReadRequest) GetWriterEpoch() uint64 {
	if x != nil {
		return x.WriterEpoch
	}
	return 0
}

func (x *WeightsReadRequest) GetObjectId() string {
	if x != nil {
		return x.ObjectId
	}
	return ""
}

func (x *WeightsReadRequest) GetSourceRef() string {
	if x != nil {
		return x.SourceRef
	}
	return ""
}

func (x *WeightsReadRequest) GetReadId() string {
	if x != nil {
		return x.ReadId
	}
	return ""
}

func (x *WeightsReadRequest) GetOffset() uint64 {
	if x != nil {
		return x.Offset
	}
	return 0
}

func (x *WeightsReadRequest) GetMaxBytes() uint32 {
	if x != nil {
		return x.MaxBytes
	}
	return 0
}

type WeightsReadResult struct {
	state                protoimpl.MessageState `protogen:"open.v1"`
	RecordOwnerEpoch     uint64                 `protobuf:"varint,1,opt,name=record_owner_epoch,json=recordOwnerEpoch,proto3" json:"record_owner_epoch,omitempty"`
	ControlStreamEpoch   uint64                 `protobuf:"varint,2,opt,name=control_stream_epoch,json=controlStreamEpoch,proto3" json:"control_stream_epoch,omitempty"`
	WorkerBootId         string                 `protobuf:"bytes,3,opt,name=worker_boot_id,json=workerBootId,proto3" json:"worker_boot_id,omitempty"`
	WeightsTransactionId string                 `protobuf:"bytes,5,opt,name=weights_transaction_id,json=weightsTransactionId,proto3" json:"weights_transaction_id,omitempty"`
	WriterEpoch          uint64                 `protobuf:"varint,6,opt,name=writer_epoch,json=writerEpoch,proto3" json:"writer_epoch,omitempty"`
	ObjectId             string                 `protobuf:"bytes,7,opt,name=object_id,json=objectId,proto3" json:"object_id,omitempty"`
	SourceRef            string                 `protobuf:"bytes,8,opt,name=source_ref,json=sourceRef,proto3" json:"source_ref,omitempty"`
	ReadId               string                 `protobuf:"bytes,9,opt,name=read_id,json=readId,proto3" json:"read_id,omitempty"`
	Offset               uint64                 `protobuf:"varint,10,opt,name=offset,proto3" json:"offset,omitempty"`
	Outcome              WeightsReadOutcome     `protobuf:"varint,11,opt,name=outcome,proto3,enum=cozy.worker.v1.WeightsReadOutcome" json:"outcome,omitempty"`
	Data                 []byte                 `protobuf:"bytes,12,opt,name=data,proto3" json:"data,omitempty"`
	DataDigest           []byte                 `protobuf:"bytes,13,opt,name=data_digest,json=dataDigest,proto3" json:"data_digest,omitempty"`
	Refusal              WeightsReadRefusal     `protobuf:"varint,14,opt,name=refusal,proto3,enum=cozy.worker.v1.WeightsReadRefusal" json:"refusal,omitempty"`
	SafeDetail           string                 `protobuf:"bytes,15,opt,name=safe_detail,json=safeDetail,proto3" json:"safe_detail,omitempty"`
	unknownFields        protoimpl.UnknownFields
	sizeCache            protoimpl.SizeCache
}

func (x *WeightsReadResult) Reset() {
	*x = WeightsReadResult{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[110]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *WeightsReadResult) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*WeightsReadResult) ProtoMessage() {}

func (x *WeightsReadResult) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[110]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use WeightsReadResult.ProtoReflect.Descriptor instead.
func (*WeightsReadResult) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{110}
}

func (x *WeightsReadResult) GetRecordOwnerEpoch() uint64 {
	if x != nil {
		return x.RecordOwnerEpoch
	}
	return 0
}

func (x *WeightsReadResult) GetControlStreamEpoch() uint64 {
	if x != nil {
		return x.ControlStreamEpoch
	}
	return 0
}

func (x *WeightsReadResult) GetWorkerBootId() string {
	if x != nil {
		return x.WorkerBootId
	}
	return ""
}

func (x *WeightsReadResult) GetWeightsTransactionId() string {
	if x != nil {
		return x.WeightsTransactionId
	}
	return ""
}

func (x *WeightsReadResult) GetWriterEpoch() uint64 {
	if x != nil {
		return x.WriterEpoch
	}
	return 0
}

func (x *WeightsReadResult) GetObjectId() string {
	if x != nil {
		return x.ObjectId
	}
	return ""
}

func (x *WeightsReadResult) GetSourceRef() string {
	if x != nil {
		return x.SourceRef
	}
	return ""
}

func (x *WeightsReadResult) GetReadId() string {
	if x != nil {
		return x.ReadId
	}
	return ""
}

func (x *WeightsReadResult) GetOffset() uint64 {
	if x != nil {
		return x.Offset
	}
	return 0
}

func (x *WeightsReadResult) GetOutcome() WeightsReadOutcome {
	if x != nil {
		return x.Outcome
	}
	return WeightsReadOutcome_WEIGHTS_READ_OUTCOME_UNSPECIFIED
}

func (x *WeightsReadResult) GetData() []byte {
	if x != nil {
		return x.Data
	}
	return nil
}

func (x *WeightsReadResult) GetDataDigest() []byte {
	if x != nil {
		return x.DataDigest
	}
	return nil
}

func (x *WeightsReadResult) GetRefusal() WeightsReadRefusal {
	if x != nil {
		return x.Refusal
	}
	return WeightsReadRefusal_WEIGHTS_READ_REFUSAL_UNSPECIFIED
}

func (x *WeightsReadResult) GetSafeDetail() string {
	if x != nil {
		return x.SafeDetail
	}
	return ""
}

type WeightsObjectRef struct {
	state         protoimpl.MessageState `protogen:"open.v1"`
	ObjectId      string                 `protobuf:"bytes,1,opt,name=object_id,json=objectId,proto3" json:"object_id,omitempty"`
	Length        uint64                 `protobuf:"varint,2,opt,name=length,proto3" json:"length,omitempty"`
	unknownFields protoimpl.UnknownFields
	sizeCache     protoimpl.SizeCache
}

func (x *WeightsObjectRef) Reset() {
	*x = WeightsObjectRef{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[111]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *WeightsObjectRef) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*WeightsObjectRef) ProtoMessage() {}

func (x *WeightsObjectRef) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[111]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use WeightsObjectRef.ProtoReflect.Descriptor instead.
func (*WeightsObjectRef) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{111}
}

func (x *WeightsObjectRef) GetObjectId() string {
	if x != nil {
		return x.ObjectId
	}
	return ""
}

func (x *WeightsObjectRef) GetLength() uint64 {
	if x != nil {
		return x.Length
	}
	return 0
}

type WeightsUploadHeader struct {
	state         protoimpl.MessageState `protogen:"open.v1"`
	Name          string                 `protobuf:"bytes,1,opt,name=name,proto3" json:"name,omitempty"`
	Value         string                 `protobuf:"bytes,2,opt,name=value,proto3" json:"value,omitempty"`
	unknownFields protoimpl.UnknownFields
	sizeCache     protoimpl.SizeCache
}

func (x *WeightsUploadHeader) Reset() {
	*x = WeightsUploadHeader{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[112]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *WeightsUploadHeader) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*WeightsUploadHeader) ProtoMessage() {}

func (x *WeightsUploadHeader) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[112]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use WeightsUploadHeader.ProtoReflect.Descriptor instead.
func (*WeightsUploadHeader) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{112}
}

func (x *WeightsUploadHeader) GetName() string {
	if x != nil {
		return x.Name
	}
	return ""
}

func (x *WeightsUploadHeader) GetValue() string {
	if x != nil {
		return x.Value
	}
	return ""
}

// A th-059 upload capability: ONE object, ONE PUT, ONE short expiry, pinned to the object's own
// digest by its signed checksum header and to first-write-wins by `if-none-match: *`. url
// includes the scoped query credential, is memory-only, and MUST NOT reach a log or the host
// ledger.
//
// The supervisor validates it against the durable receipt and then hands it to the process that
// HOLDS the bytes, which is Runtime. Relaying 8 GiB back through the supervisor 4 MiB at a time
// bought nothing: the capability is already scoped to one object, Runtime cannot widen it, and
// the object store -- not the relay -- is what enforces the signed checksum. Runtime already
// spends exactly this kind of capability for job outputs (`DeliveryGrant.outputs`).
type WeightsUploadGrant struct {
	state           protoimpl.MessageState `protogen:"open.v1"`
	ObjectId        string                 `protobuf:"bytes,1,opt,name=object_id,json=objectId,proto3" json:"object_id,omitempty"`
	Length          uint64                 `protobuf:"varint,2,opt,name=length,proto3" json:"length,omitempty"`
	Url             string                 `protobuf:"bytes,3,opt,name=url,proto3" json:"url,omitempty"`
	RequiredHeaders []*WeightsUploadHeader `protobuf:"bytes,4,rep,name=required_headers,json=requiredHeaders,proto3" json:"required_headers,omitempty"`
	ExpiresAtUnix   uint64                 `protobuf:"varint,5,opt,name=expires_at_unix,json=expiresAtUnix,proto3" json:"expires_at_unix,omitempty"`
	unknownFields   protoimpl.UnknownFields
	sizeCache       protoimpl.SizeCache
}

func (x *WeightsUploadGrant) Reset() {
	*x = WeightsUploadGrant{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[113]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *WeightsUploadGrant) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*WeightsUploadGrant) ProtoMessage() {}

func (x *WeightsUploadGrant) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[113]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use WeightsUploadGrant.ProtoReflect.Descriptor instead.
func (*WeightsUploadGrant) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{113}
}

func (x *WeightsUploadGrant) GetObjectId() string {
	if x != nil {
		return x.ObjectId
	}
	return ""
}

func (x *WeightsUploadGrant) GetLength() uint64 {
	if x != nil {
		return x.Length
	}
	return 0
}

func (x *WeightsUploadGrant) GetUrl() string {
	if x != nil {
		return x.Url
	}
	return ""
}

func (x *WeightsUploadGrant) GetRequiredHeaders() []*WeightsUploadHeader {
	if x != nil {
		return x.RequiredHeaders
	}
	return nil
}

func (x *WeightsUploadGrant) GetExpiresAtUnix() uint64 {
	if x != nil {
		return x.ExpiresAtUnix
	}
	return 0
}

// Authenticated RecordOwner -> supervisor request for ONE exact object. Exact replay of one
// operation/revision compares every field; a higher grant_revision refreshes only that object's
// capability without changing weights identity. RecordOwners pipeline several requests instead
// of putting thousands of presigned URLs in one protobuf.
type WeightsTransferRequest struct {
	state                protoimpl.MessageState `protogen:"open.v1"`
	RecordOwnerEpoch     uint64                 `protobuf:"varint,1,opt,name=record_owner_epoch,json=recordOwnerEpoch,proto3" json:"record_owner_epoch,omitempty"`
	ControlStreamEpoch   uint64                 `protobuf:"varint,2,opt,name=control_stream_epoch,json=controlStreamEpoch,proto3" json:"control_stream_epoch,omitempty"`
	WorkerBootId         string                 `protobuf:"bytes,3,opt,name=worker_boot_id,json=workerBootId,proto3" json:"worker_boot_id,omitempty"`
	RequestId            string                 `protobuf:"bytes,5,opt,name=request_id,json=requestId,proto3" json:"request_id,omitempty"`
	AttemptOrdinal       uint64                 `protobuf:"varint,6,opt,name=attempt_ordinal,json=attemptOrdinal,proto3" json:"attempt_ordinal,omitempty"`
	InvocationSpecDigest []byte                 `protobuf:"bytes,7,opt,name=invocation_spec_digest,json=invocationSpecDigest,proto3" json:"invocation_spec_digest,omitempty"`
	OutputSlot           string                 `protobuf:"bytes,8,opt,name=output_slot,json=outputSlot,proto3" json:"output_slot,omitempty"`
	WeightsTransactionId string                 `protobuf:"bytes,9,opt,name=weights_transaction_id,json=weightsTransactionId,proto3" json:"weights_transaction_id,omitempty"`
	WeightsReceiptDigest []byte                 `protobuf:"bytes,10,opt,name=weights_receipt_digest,json=weightsReceiptDigest,proto3" json:"weights_receipt_digest,omitempty"`
	OperationId          string                 `protobuf:"bytes,11,opt,name=operation_id,json=operationId,proto3" json:"operation_id,omitempty"`
	GrantRevision        uint64                 `protobuf:"varint,12,opt,name=grant_revision,json=grantRevision,proto3" json:"grant_revision,omitempty"`
	// Types that are valid to be assigned to Decision:
	//
	//	*WeightsTransferRequest_UploadGrant
	//	*WeightsTransferRequest_Held
	Decision      isWeightsTransferRequest_Decision `protobuf_oneof:"decision"`
	unknownFields protoimpl.UnknownFields
	sizeCache     protoimpl.SizeCache
}

func (x *WeightsTransferRequest) Reset() {
	*x = WeightsTransferRequest{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[114]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *WeightsTransferRequest) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*WeightsTransferRequest) ProtoMessage() {}

func (x *WeightsTransferRequest) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[114]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use WeightsTransferRequest.ProtoReflect.Descriptor instead.
func (*WeightsTransferRequest) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{114}
}

func (x *WeightsTransferRequest) GetRecordOwnerEpoch() uint64 {
	if x != nil {
		return x.RecordOwnerEpoch
	}
	return 0
}

func (x *WeightsTransferRequest) GetControlStreamEpoch() uint64 {
	if x != nil {
		return x.ControlStreamEpoch
	}
	return 0
}

func (x *WeightsTransferRequest) GetWorkerBootId() string {
	if x != nil {
		return x.WorkerBootId
	}
	return ""
}

func (x *WeightsTransferRequest) GetRequestId() string {
	if x != nil {
		return x.RequestId
	}
	return ""
}

func (x *WeightsTransferRequest) GetAttemptOrdinal() uint64 {
	if x != nil {
		return x.AttemptOrdinal
	}
	return 0
}

func (x *WeightsTransferRequest) GetInvocationSpecDigest() []byte {
	if x != nil {
		return x.InvocationSpecDigest
	}
	return nil
}

func (x *WeightsTransferRequest) GetOutputSlot() string {
	if x != nil {
		return x.OutputSlot
	}
	return ""
}

func (x *WeightsTransferRequest) GetWeightsTransactionId() string {
	if x != nil {
		return x.WeightsTransactionId
	}
	return ""
}

func (x *WeightsTransferRequest) GetWeightsReceiptDigest() []byte {
	if x != nil {
		return x.WeightsReceiptDigest
	}
	return nil
}

func (x *WeightsTransferRequest) GetOperationId() string {
	if x != nil {
		return x.OperationId
	}
	return ""
}

func (x *WeightsTransferRequest) GetGrantRevision() uint64 {
	if x != nil {
		return x.GrantRevision
	}
	return 0
}

func (x *WeightsTransferRequest) GetDecision() isWeightsTransferRequest_Decision {
	if x != nil {
		return x.Decision
	}
	return nil
}

func (x *WeightsTransferRequest) GetUploadGrant() *WeightsUploadGrant {
	if x != nil {
		if x, ok := x.Decision.(*WeightsTransferRequest_UploadGrant); ok {
			return x.UploadGrant
		}
	}
	return nil
}

func (x *WeightsTransferRequest) GetHeld() *WeightsObjectRef {
	if x != nil {
		if x, ok := x.Decision.(*WeightsTransferRequest_Held); ok {
			return x.Held
		}
	}
	return nil
}

type isWeightsTransferRequest_Decision interface {
	isWeightsTransferRequest_Decision()
}

type WeightsTransferRequest_UploadGrant struct {
	UploadGrant *WeightsUploadGrant `protobuf:"bytes,13,opt,name=upload_grant,json=uploadGrant,proto3,oneof"`
}

type WeightsTransferRequest_Held struct {
	Held *WeightsObjectRef `protobuf:"bytes,14,opt,name=held,proto3,oneof"`
}

func (*WeightsTransferRequest_UploadGrant) isWeightsTransferRequest_Decision() {}

func (*WeightsTransferRequest_Held) isWeightsTransferRequest_Decision() {}

// pod-supervisor -> Runtime. The supervisor has already bound this transfer to the durable
// receipt inventory and its ledger; this frame delegates the one thing it cannot do better than
// Runtime, which is stream bytes it does not have. Runtime PUTs to the granted URL and answers
// with WeightsUploadResult. It is the supervisor's ledger, not this frame, that decides what
// was written -- custody is recorded before the writer opens and again when it closes.
type WeightsUploadRequest struct {
	state                protoimpl.MessageState `protogen:"open.v1"`
	RecordOwnerEpoch     uint64                 `protobuf:"varint,1,opt,name=record_owner_epoch,json=recordOwnerEpoch,proto3" json:"record_owner_epoch,omitempty"`
	ControlStreamEpoch   uint64                 `protobuf:"varint,2,opt,name=control_stream_epoch,json=controlStreamEpoch,proto3" json:"control_stream_epoch,omitempty"`
	WorkerBootId         string                 `protobuf:"bytes,3,opt,name=worker_boot_id,json=workerBootId,proto3" json:"worker_boot_id,omitempty"`
	WeightsTransactionId string                 `protobuf:"bytes,5,opt,name=weights_transaction_id,json=weightsTransactionId,proto3" json:"weights_transaction_id,omitempty"`
	WriterEpoch          uint64                 `protobuf:"varint,6,opt,name=writer_epoch,json=writerEpoch,proto3" json:"writer_epoch,omitempty"`
	ObjectId             string                 `protobuf:"bytes,7,opt,name=object_id,json=objectId,proto3" json:"object_id,omitempty"`
	SourceRef            string                 `protobuf:"bytes,8,opt,name=source_ref,json=sourceRef,proto3" json:"source_ref,omitempty"`
	Length               uint64                 `protobuf:"varint,9,opt,name=length,proto3" json:"length,omitempty"`
	OperationId          string                 `protobuf:"bytes,10,opt,name=operation_id,json=operationId,proto3" json:"operation_id,omitempty"`
	GrantRevision        uint64                 `protobuf:"varint,11,opt,name=grant_revision,json=grantRevision,proto3" json:"grant_revision,omitempty"`
	Grant                *WeightsUploadGrant    `protobuf:"bytes,12,opt,name=grant,proto3" json:"grant,omitempty"`
	unknownFields        protoimpl.UnknownFields
	sizeCache            protoimpl.SizeCache
}

func (x *WeightsUploadRequest) Reset() {
	*x = WeightsUploadRequest{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[115]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *WeightsUploadRequest) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*WeightsUploadRequest) ProtoMessage() {}

func (x *WeightsUploadRequest) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[115]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use WeightsUploadRequest.ProtoReflect.Descriptor instead.
func (*WeightsUploadRequest) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{115}
}

func (x *WeightsUploadRequest) GetRecordOwnerEpoch() uint64 {
	if x != nil {
		return x.RecordOwnerEpoch
	}
	return 0
}

func (x *WeightsUploadRequest) GetControlStreamEpoch() uint64 {
	if x != nil {
		return x.ControlStreamEpoch
	}
	return 0
}

func (x *WeightsUploadRequest) GetWorkerBootId() string {
	if x != nil {
		return x.WorkerBootId
	}
	return ""
}

func (x *WeightsUploadRequest) GetWeightsTransactionId() string {
	if x != nil {
		return x.WeightsTransactionId
	}
	return ""
}

func (x *WeightsUploadRequest) GetWriterEpoch() uint64 {
	if x != nil {
		return x.WriterEpoch
	}
	return 0
}

func (x *WeightsUploadRequest) GetObjectId() string {
	if x != nil {
		return x.ObjectId
	}
	return ""
}

func (x *WeightsUploadRequest) GetSourceRef() string {
	if x != nil {
		return x.SourceRef
	}
	return ""
}

func (x *WeightsUploadRequest) GetLength() uint64 {
	if x != nil {
		return x.Length
	}
	return 0
}

func (x *WeightsUploadRequest) GetOperationId() string {
	if x != nil {
		return x.OperationId
	}
	return ""
}

func (x *WeightsUploadRequest) GetGrantRevision() uint64 {
	if x != nil {
		return x.GrantRevision
	}
	return 0
}

func (x *WeightsUploadRequest) GetGrant() *WeightsUploadGrant {
	if x != nil {
		return x.Grant
	}
	return nil
}

// Runtime -> pod-supervisor. One terminal answer per WeightsUploadRequest. checksum_sha256 is
// what Runtime measured while streaming; the supervisor compares it to the durable ObjectRef,
// and the store independently enforced the same digest through the grant's signed checksum
// header, so neither party is trusted alone.
type WeightsUploadResult struct {
	state                protoimpl.MessageState `protogen:"open.v1"`
	RecordOwnerEpoch     uint64                 `protobuf:"varint,1,opt,name=record_owner_epoch,json=recordOwnerEpoch,proto3" json:"record_owner_epoch,omitempty"`
	ControlStreamEpoch   uint64                 `protobuf:"varint,2,opt,name=control_stream_epoch,json=controlStreamEpoch,proto3" json:"control_stream_epoch,omitempty"`
	WorkerBootId         string                 `protobuf:"bytes,3,opt,name=worker_boot_id,json=workerBootId,proto3" json:"worker_boot_id,omitempty"`
	WeightsTransactionId string                 `protobuf:"bytes,5,opt,name=weights_transaction_id,json=weightsTransactionId,proto3" json:"weights_transaction_id,omitempty"`
	WriterEpoch          uint64                 `protobuf:"varint,6,opt,name=writer_epoch,json=writerEpoch,proto3" json:"writer_epoch,omitempty"`
	ObjectId             string                 `protobuf:"bytes,7,opt,name=object_id,json=objectId,proto3" json:"object_id,omitempty"`
	OperationId          string                 `protobuf:"bytes,8,opt,name=operation_id,json=operationId,proto3" json:"operation_id,omitempty"`
	GrantRevision        uint64                 `protobuf:"varint,9,opt,name=grant_revision,json=grantRevision,proto3" json:"grant_revision,omitempty"`
	Outcome              WeightsUploadOutcome   `protobuf:"varint,10,opt,name=outcome,proto3,enum=cozy.worker.v1.WeightsUploadOutcome" json:"outcome,omitempty"`
	TransferredBytes     uint64                 `protobuf:"varint,11,opt,name=transferred_bytes,json=transferredBytes,proto3" json:"transferred_bytes,omitempty"`
	HttpStatus           uint32                 `protobuf:"varint,12,opt,name=http_status,json=httpStatus,proto3" json:"http_status,omitempty"`
	Etag                 string                 `protobuf:"bytes,13,opt,name=etag,proto3" json:"etag,omitempty"`
	ChecksumSha256       string                 `protobuf:"bytes,14,opt,name=checksum_sha256,json=checksumSha256,proto3" json:"checksum_sha256,omitempty"` // lowercase sha256:<64 hex> measured while streaming
	Refusal              WeightsUploadRefusal   `protobuf:"varint,15,opt,name=refusal,proto3,enum=cozy.worker.v1.WeightsUploadRefusal" json:"refusal,omitempty"`
	SafeDetail           string                 `protobuf:"bytes,16,opt,name=safe_detail,json=safeDetail,proto3" json:"safe_detail,omitempty"`
	unknownFields        protoimpl.UnknownFields
	sizeCache            protoimpl.SizeCache
}

func (x *WeightsUploadResult) Reset() {
	*x = WeightsUploadResult{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[116]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *WeightsUploadResult) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*WeightsUploadResult) ProtoMessage() {}

func (x *WeightsUploadResult) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[116]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use WeightsUploadResult.ProtoReflect.Descriptor instead.
func (*WeightsUploadResult) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{116}
}

func (x *WeightsUploadResult) GetRecordOwnerEpoch() uint64 {
	if x != nil {
		return x.RecordOwnerEpoch
	}
	return 0
}

func (x *WeightsUploadResult) GetControlStreamEpoch() uint64 {
	if x != nil {
		return x.ControlStreamEpoch
	}
	return 0
}

func (x *WeightsUploadResult) GetWorkerBootId() string {
	if x != nil {
		return x.WorkerBootId
	}
	return ""
}

func (x *WeightsUploadResult) GetWeightsTransactionId() string {
	if x != nil {
		return x.WeightsTransactionId
	}
	return ""
}

func (x *WeightsUploadResult) GetWriterEpoch() uint64 {
	if x != nil {
		return x.WriterEpoch
	}
	return 0
}

func (x *WeightsUploadResult) GetObjectId() string {
	if x != nil {
		return x.ObjectId
	}
	return ""
}

func (x *WeightsUploadResult) GetOperationId() string {
	if x != nil {
		return x.OperationId
	}
	return ""
}

func (x *WeightsUploadResult) GetGrantRevision() uint64 {
	if x != nil {
		return x.GrantRevision
	}
	return 0
}

func (x *WeightsUploadResult) GetOutcome() WeightsUploadOutcome {
	if x != nil {
		return x.Outcome
	}
	return WeightsUploadOutcome_WEIGHTS_UPLOAD_OUTCOME_UNSPECIFIED
}

func (x *WeightsUploadResult) GetTransferredBytes() uint64 {
	if x != nil {
		return x.TransferredBytes
	}
	return 0
}

func (x *WeightsUploadResult) GetHttpStatus() uint32 {
	if x != nil {
		return x.HttpStatus
	}
	return 0
}

func (x *WeightsUploadResult) GetEtag() string {
	if x != nil {
		return x.Etag
	}
	return ""
}

func (x *WeightsUploadResult) GetChecksumSha256() string {
	if x != nil {
		return x.ChecksumSha256
	}
	return ""
}

func (x *WeightsUploadResult) GetRefusal() WeightsUploadRefusal {
	if x != nil {
		return x.Refusal
	}
	return WeightsUploadRefusal_WEIGHTS_UPLOAD_REFUSAL_UNSPECIFIED
}

func (x *WeightsUploadResult) GetSafeDetail() string {
	if x != nil {
		return x.SafeDetail
	}
	return ""
}

// Durable supervisor -> RecordOwner observation. Intermediate observations are monotonic by
// update_sequence; terminal observations replay after reconnect or exact TransferRequest replay.
type WeightsTransferStatus struct {
	state                protoimpl.MessageState `protogen:"open.v1"`
	RecordOwnerEpoch     uint64                 `protobuf:"varint,1,opt,name=record_owner_epoch,json=recordOwnerEpoch,proto3" json:"record_owner_epoch,omitempty"`
	ControlStreamEpoch   uint64                 `protobuf:"varint,2,opt,name=control_stream_epoch,json=controlStreamEpoch,proto3" json:"control_stream_epoch,omitempty"`
	WorkerBootId         string                 `protobuf:"bytes,3,opt,name=worker_boot_id,json=workerBootId,proto3" json:"worker_boot_id,omitempty"`
	RequestId            string                 `protobuf:"bytes,5,opt,name=request_id,json=requestId,proto3" json:"request_id,omitempty"`
	AttemptOrdinal       uint64                 `protobuf:"varint,6,opt,name=attempt_ordinal,json=attemptOrdinal,proto3" json:"attempt_ordinal,omitempty"`
	InvocationSpecDigest []byte                 `protobuf:"bytes,7,opt,name=invocation_spec_digest,json=invocationSpecDigest,proto3" json:"invocation_spec_digest,omitempty"`
	OutputSlot           string                 `protobuf:"bytes,8,opt,name=output_slot,json=outputSlot,proto3" json:"output_slot,omitempty"`
	WeightsTransactionId string                 `protobuf:"bytes,9,opt,name=weights_transaction_id,json=weightsTransactionId,proto3" json:"weights_transaction_id,omitempty"`
	OperationId          string                 `protobuf:"bytes,10,opt,name=operation_id,json=operationId,proto3" json:"operation_id,omitempty"`
	GrantRevision        uint64                 `protobuf:"varint,11,opt,name=grant_revision,json=grantRevision,proto3" json:"grant_revision,omitempty"`
	ObjectId             string                 `protobuf:"bytes,12,opt,name=object_id,json=objectId,proto3" json:"object_id,omitempty"`
	Length               uint64                 `protobuf:"varint,13,opt,name=length,proto3" json:"length,omitempty"`
	UpdateSequence       uint64                 `protobuf:"varint,14,opt,name=update_sequence,json=updateSequence,proto3" json:"update_sequence,omitempty"`
	State                WeightsTransferState   `protobuf:"varint,15,opt,name=state,proto3,enum=cozy.worker.v1.WeightsTransferState" json:"state,omitempty"`
	TransferredBytes     uint64                 `protobuf:"varint,16,opt,name=transferred_bytes,json=transferredBytes,proto3" json:"transferred_bytes,omitempty"`
	HttpStatus           uint32                 `protobuf:"varint,17,opt,name=http_status,json=httpStatus,proto3" json:"http_status,omitempty"`
	Etag                 string                 `protobuf:"bytes,18,opt,name=etag,proto3" json:"etag,omitempty"`
	ChecksumSha256       string                 `protobuf:"bytes,19,opt,name=checksum_sha256,json=checksumSha256,proto3" json:"checksum_sha256,omitempty"`
	Attempts             uint32                 `protobuf:"varint,20,opt,name=attempts,proto3" json:"attempts,omitempty"`
	SafeCode             string                 `protobuf:"bytes,21,opt,name=safe_code,json=safeCode,proto3" json:"safe_code,omitempty"`
	SafeDetail           string                 `protobuf:"bytes,22,opt,name=safe_detail,json=safeDetail,proto3" json:"safe_detail,omitempty"`
	unknownFields        protoimpl.UnknownFields
	sizeCache            protoimpl.SizeCache
}

func (x *WeightsTransferStatus) Reset() {
	*x = WeightsTransferStatus{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[117]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *WeightsTransferStatus) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*WeightsTransferStatus) ProtoMessage() {}

func (x *WeightsTransferStatus) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[117]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use WeightsTransferStatus.ProtoReflect.Descriptor instead.
func (*WeightsTransferStatus) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{117}
}

func (x *WeightsTransferStatus) GetRecordOwnerEpoch() uint64 {
	if x != nil {
		return x.RecordOwnerEpoch
	}
	return 0
}

func (x *WeightsTransferStatus) GetControlStreamEpoch() uint64 {
	if x != nil {
		return x.ControlStreamEpoch
	}
	return 0
}

func (x *WeightsTransferStatus) GetWorkerBootId() string {
	if x != nil {
		return x.WorkerBootId
	}
	return ""
}

func (x *WeightsTransferStatus) GetRequestId() string {
	if x != nil {
		return x.RequestId
	}
	return ""
}

func (x *WeightsTransferStatus) GetAttemptOrdinal() uint64 {
	if x != nil {
		return x.AttemptOrdinal
	}
	return 0
}

func (x *WeightsTransferStatus) GetInvocationSpecDigest() []byte {
	if x != nil {
		return x.InvocationSpecDigest
	}
	return nil
}

func (x *WeightsTransferStatus) GetOutputSlot() string {
	if x != nil {
		return x.OutputSlot
	}
	return ""
}

func (x *WeightsTransferStatus) GetWeightsTransactionId() string {
	if x != nil {
		return x.WeightsTransactionId
	}
	return ""
}

func (x *WeightsTransferStatus) GetOperationId() string {
	if x != nil {
		return x.OperationId
	}
	return ""
}

func (x *WeightsTransferStatus) GetGrantRevision() uint64 {
	if x != nil {
		return x.GrantRevision
	}
	return 0
}

func (x *WeightsTransferStatus) GetObjectId() string {
	if x != nil {
		return x.ObjectId
	}
	return ""
}

func (x *WeightsTransferStatus) GetLength() uint64 {
	if x != nil {
		return x.Length
	}
	return 0
}

func (x *WeightsTransferStatus) GetUpdateSequence() uint64 {
	if x != nil {
		return x.UpdateSequence
	}
	return 0
}

func (x *WeightsTransferStatus) GetState() WeightsTransferState {
	if x != nil {
		return x.State
	}
	return WeightsTransferState_WEIGHTS_TRANSFER_STATE_UNSPECIFIED
}

func (x *WeightsTransferStatus) GetTransferredBytes() uint64 {
	if x != nil {
		return x.TransferredBytes
	}
	return 0
}

func (x *WeightsTransferStatus) GetHttpStatus() uint32 {
	if x != nil {
		return x.HttpStatus
	}
	return 0
}

func (x *WeightsTransferStatus) GetEtag() string {
	if x != nil {
		return x.Etag
	}
	return ""
}

func (x *WeightsTransferStatus) GetChecksumSha256() string {
	if x != nil {
		return x.ChecksumSha256
	}
	return ""
}

func (x *WeightsTransferStatus) GetAttempts() uint32 {
	if x != nil {
		return x.Attempts
	}
	return 0
}

func (x *WeightsTransferStatus) GetSafeCode() string {
	if x != nil {
		return x.SafeCode
	}
	return ""
}

func (x *WeightsTransferStatus) GetSafeDetail() string {
	if x != nil {
		return x.SafeDetail
	}
	return ""
}

// Creator resolves and pins provider metadata. One selected file capability at a time reaches
// pod-supervisor; Runtime performs TensorFS's bounded profile/header proof before all payloads
// arrive, then converts only operations whose physical carriers are verified. URL is memory-only; the ledger
// stores only stable identity/revision and measured progress. A higher capability_revision may
// refresh access but cannot change member/ObjectRef/source selection. Broad account tokens are not
// representable.
type ModelSourceFileRequest struct {
	state                 protoimpl.MessageState `protogen:"open.v1"`
	RecordOwnerEpoch      uint64                 `protobuf:"varint,1,opt,name=record_owner_epoch,json=recordOwnerEpoch,proto3" json:"record_owner_epoch,omitempty"`
	ControlStreamEpoch    uint64                 `protobuf:"varint,2,opt,name=control_stream_epoch,json=controlStreamEpoch,proto3" json:"control_stream_epoch,omitempty"`
	WorkerBootId          string                 `protobuf:"bytes,3,opt,name=worker_boot_id,json=workerBootId,proto3" json:"worker_boot_id,omitempty"`
	OperationId           string                 `protobuf:"bytes,5,opt,name=operation_id,json=operationId,proto3" json:"operation_id,omitempty"`
	SourceSelectionDigest []byte                 `protobuf:"bytes,6,opt,name=source_selection_digest,json=sourceSelectionDigest,proto3" json:"source_selection_digest,omitempty"`
	Member                string                 `protobuf:"bytes,7,opt,name=member,proto3" json:"member,omitempty"`
	ObjectId              string                 `protobuf:"bytes,8,opt,name=object_id,json=objectId,proto3" json:"object_id,omitempty"` // lowercase sha256:<64 hex>
	Length                uint64                 `protobuf:"varint,9,opt,name=length,proto3" json:"length,omitempty"`
	Provider              ModelSourceProvider    `protobuf:"varint,10,opt,name=provider,proto3,enum=cozy.worker.v1.ModelSourceProvider" json:"provider,omitempty"`
	// Empty declares header metadata only and returns ACCEPTED without downloading. The
	// owner declares every roster header first, restores an acknowledged checkpoint, then
	// grants nonempty URLs only for the physical carriers TensorFS still needs.
	Url                string `protobuf:"bytes,11,opt,name=url,proto3" json:"url,omitempty"`                                             // memory-only access; never echoed in status/snapshot/loopback
	ExpiresAtUnix      uint64 `protobuf:"varint,12,opt,name=expires_at_unix,json=expiresAtUnix,proto3" json:"expires_at_unix,omitempty"` // zero only for an immutable public URL
	CapabilityRevision uint64 `protobuf:"varint,13,opt,name=capability_revision,json=capabilityRevision,proto3" json:"capability_revision,omitempty"`
	// Exact safetensors 8-byte LE length + header JSON, or full JSON for an index carrier.
	// One bounded header per file frame; never the model's entire header corpus in one frame.
	// <= MaxModelSourceHeaderBytes AND the complete frame <= MaxInlineControlBytes.
	Header        []byte `protobuf:"bytes,14,opt,name=header,proto3" json:"header,omitempty"`
	unknownFields protoimpl.UnknownFields
	sizeCache     protoimpl.SizeCache
}

func (x *ModelSourceFileRequest) Reset() {
	*x = ModelSourceFileRequest{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[118]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *ModelSourceFileRequest) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*ModelSourceFileRequest) ProtoMessage() {}

func (x *ModelSourceFileRequest) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[118]
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
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{118}
}

func (x *ModelSourceFileRequest) GetRecordOwnerEpoch() uint64 {
	if x != nil {
		return x.RecordOwnerEpoch
	}
	return 0
}

func (x *ModelSourceFileRequest) GetControlStreamEpoch() uint64 {
	if x != nil {
		return x.ControlStreamEpoch
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

func (x *ModelSourceFileRequest) GetHeader() []byte {
	if x != nil {
		return x.Header
	}
	return nil
}

type ModelSourceFileStatus struct {
	state                 protoimpl.MessageState `protogen:"open.v1"`
	RecordOwnerEpoch      uint64                 `protobuf:"varint,1,opt,name=record_owner_epoch,json=recordOwnerEpoch,proto3" json:"record_owner_epoch,omitempty"`
	ControlStreamEpoch    uint64                 `protobuf:"varint,2,opt,name=control_stream_epoch,json=controlStreamEpoch,proto3" json:"control_stream_epoch,omitempty"`
	WorkerBootId          string                 `protobuf:"bytes,3,opt,name=worker_boot_id,json=workerBootId,proto3" json:"worker_boot_id,omitempty"`
	OperationId           string                 `protobuf:"bytes,5,opt,name=operation_id,json=operationId,proto3" json:"operation_id,omitempty"`
	SourceSelectionDigest []byte                 `protobuf:"bytes,6,opt,name=source_selection_digest,json=sourceSelectionDigest,proto3" json:"source_selection_digest,omitempty"`
	Member                string                 `protobuf:"bytes,7,opt,name=member,proto3" json:"member,omitempty"`
	ObjectId              string                 `protobuf:"bytes,8,opt,name=object_id,json=objectId,proto3" json:"object_id,omitempty"`
	Length                uint64                 `protobuf:"varint,9,opt,name=length,proto3" json:"length,omitempty"`
	CapabilityRevision    uint64                 `protobuf:"varint,10,opt,name=capability_revision,json=capabilityRevision,proto3" json:"capability_revision,omitempty"`
	State                 ModelSourceFileState   `protobuf:"varint,11,opt,name=state,proto3,enum=cozy.worker.v1.ModelSourceFileState" json:"state,omitempty"`
	TransferredBytes      uint64                 `protobuf:"varint,12,opt,name=transferred_bytes,json=transferredBytes,proto3" json:"transferred_bytes,omitempty"`
	Attempts              uint32                 `protobuf:"varint,13,opt,name=attempts,proto3" json:"attempts,omitempty"`
	SafeCode              string                 `protobuf:"bytes,14,opt,name=safe_code,json=safeCode,proto3" json:"safe_code,omitempty"`
	SafeDetail            string                 `protobuf:"bytes,15,opt,name=safe_detail,json=safeDetail,proto3" json:"safe_detail,omitempty"`
	unknownFields         protoimpl.UnknownFields
	sizeCache             protoimpl.SizeCache
}

func (x *ModelSourceFileStatus) Reset() {
	*x = ModelSourceFileStatus{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[119]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *ModelSourceFileStatus) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*ModelSourceFileStatus) ProtoMessage() {}

func (x *ModelSourceFileStatus) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[119]
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
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{119}
}

func (x *ModelSourceFileStatus) GetRecordOwnerEpoch() uint64 {
	if x != nil {
		return x.RecordOwnerEpoch
	}
	return 0
}

func (x *ModelSourceFileStatus) GetControlStreamEpoch() uint64 {
	if x != nil {
		return x.ControlStreamEpoch
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

// The complete selected roster and its headers are fixed before conversion. Runtime advances
// every descriptor-reviewed profile over the currently verified physical carriers and returns
// INCOMPLETE until every target is available. Pinned source URI and declared license are
// credential-free provenance, not authorization. This replaces the pre-release full-file barrier;
// deploy the generated schema and all consumers together.
type ModelSourcePrepareRequest struct {
	state                 protoimpl.MessageState   `protogen:"open.v1"`
	RecordOwnerEpoch      uint64                   `protobuf:"varint,1,opt,name=record_owner_epoch,json=recordOwnerEpoch,proto3" json:"record_owner_epoch,omitempty"`
	ControlStreamEpoch    uint64                   `protobuf:"varint,2,opt,name=control_stream_epoch,json=controlStreamEpoch,proto3" json:"control_stream_epoch,omitempty"`
	WorkerBootId          string                   `protobuf:"bytes,3,opt,name=worker_boot_id,json=workerBootId,proto3" json:"worker_boot_id,omitempty"`
	OperationId           string                   `protobuf:"bytes,5,opt,name=operation_id,json=operationId,proto3" json:"operation_id,omitempty"`
	SourceSelectionDigest []byte                   `protobuf:"bytes,6,opt,name=source_selection_digest,json=sourceSelectionDigest,proto3" json:"source_selection_digest,omitempty"`
	Profiles              []*ModelSourceProfile    `protobuf:"bytes,7,rep,name=profiles,proto3" json:"profiles,omitempty"`                                      // sorted unique by slot
	SourceUri             string                   `protobuf:"bytes,8,opt,name=source_uri,json=sourceUri,proto3" json:"source_uri,omitempty"`                   // credential-free pinned provenance
	DeclaredLicense       string                   `protobuf:"bytes,9,opt,name=declared_license,json=declaredLicense,proto3" json:"declared_license,omitempty"` // bounded printable provenance; empty means undeclared
	Checkpoints           []*ModelSourceCheckpoint `protobuf:"bytes,10,rep,name=checkpoints,proto3" json:"checkpoints,omitempty"`                               // sorted unique acknowledged durable heads
	unknownFields         protoimpl.UnknownFields
	sizeCache             protoimpl.SizeCache
}

func (x *ModelSourcePrepareRequest) Reset() {
	*x = ModelSourcePrepareRequest{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[120]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *ModelSourcePrepareRequest) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*ModelSourcePrepareRequest) ProtoMessage() {}

func (x *ModelSourcePrepareRequest) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[120]
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
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{120}
}

func (x *ModelSourcePrepareRequest) GetRecordOwnerEpoch() uint64 {
	if x != nil {
		return x.RecordOwnerEpoch
	}
	return 0
}

func (x *ModelSourcePrepareRequest) GetControlStreamEpoch() uint64 {
	if x != nil {
		return x.ControlStreamEpoch
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

func (x *ModelSourcePrepareRequest) GetCheckpoints() []*ModelSourceCheckpoint {
	if x != nil {
		return x.Checkpoints
	}
	return nil
}

type ModelSourcePrepared struct {
	state                 protoimpl.MessageState    `protogen:"open.v1"`
	RecordOwnerEpoch      uint64                    `protobuf:"varint,1,opt,name=record_owner_epoch,json=recordOwnerEpoch,proto3" json:"record_owner_epoch,omitempty"`
	ControlStreamEpoch    uint64                    `protobuf:"varint,2,opt,name=control_stream_epoch,json=controlStreamEpoch,proto3" json:"control_stream_epoch,omitempty"`
	WorkerBootId          string                    `protobuf:"bytes,3,opt,name=worker_boot_id,json=workerBootId,proto3" json:"worker_boot_id,omitempty"`
	OperationId           string                    `protobuf:"bytes,5,opt,name=operation_id,json=operationId,proto3" json:"operation_id,omitempty"`
	SourceSelectionDigest []byte                    `protobuf:"bytes,6,opt,name=source_selection_digest,json=sourceSelectionDigest,proto3" json:"source_selection_digest,omitempty"`
	Outcome               ModelSourcePrepareOutcome `protobuf:"varint,7,opt,name=outcome,proto3,enum=cozy.worker.v1.ModelSourcePrepareOutcome" json:"outcome,omitempty"`
	Sources               []*PreparedModelSource    `protobuf:"bytes,8,rep,name=sources,proto3" json:"sources,omitempty"` // sorted unique completed slots; exact set when complete
	SafeCode              string                    `protobuf:"bytes,9,opt,name=safe_code,json=safeCode,proto3" json:"safe_code,omitempty"`
	SafeDetail            string                    `protobuf:"bytes,10,opt,name=safe_detail,json=safeDetail,proto3" json:"safe_detail,omitempty"`
	Checkpoints           []*ModelSourceCheckpoint  `protobuf:"bytes,11,rep,name=checkpoints,proto3" json:"checkpoints,omitempty"` // sorted unique locally checkpointed progress
	unknownFields         protoimpl.UnknownFields
	sizeCache             protoimpl.SizeCache
}

func (x *ModelSourcePrepared) Reset() {
	*x = ModelSourcePrepared{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[121]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *ModelSourcePrepared) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*ModelSourcePrepared) ProtoMessage() {}

func (x *ModelSourcePrepared) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[121]
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
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{121}
}

func (x *ModelSourcePrepared) GetRecordOwnerEpoch() uint64 {
	if x != nil {
		return x.RecordOwnerEpoch
	}
	return 0
}

func (x *ModelSourcePrepared) GetControlStreamEpoch() uint64 {
	if x != nil {
		return x.ControlStreamEpoch
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

func (x *ModelSourcePrepared) GetCheckpoints() []*ModelSourceCheckpoint {
	if x != nil {
		return x.Checkpoints
	}
	return nil
}

// A th-094 download capability: ONE object, ONE GET, ONE short expiry. The record owner already
// PUT this wheel to the store under a checksum-pinned, first-write-wins grant of its own, so the
// object behind this URL cannot be anything but `digest`. url carries the scoped query
// credential, is memory-only, and MUST NOT reach a log, an executor, Runtime, or the host ledger.
type LocalPackageFileGrant struct {
	state         protoimpl.MessageState `protogen:"open.v1"`
	Digest        []byte                 `protobuf:"bytes,1,opt,name=digest,proto3" json:"digest,omitempty"`
	Filename      string                 `protobuf:"bytes,2,opt,name=filename,proto3" json:"filename,omitempty"`
	Length        uint64                 `protobuf:"varint,4,opt,name=length,proto3" json:"length,omitempty"`
	Url           string                 `protobuf:"bytes,5,opt,name=url,proto3" json:"url,omitempty"` // nonempty and <= MaxLocalPackageGrantURLBytes
	unknownFields protoimpl.UnknownFields
	sizeCache     protoimpl.SizeCache
}

func (x *LocalPackageFileGrant) Reset() {
	*x = LocalPackageFileGrant{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[122]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *LocalPackageFileGrant) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*LocalPackageFileGrant) ProtoMessage() {}

func (x *LocalPackageFileGrant) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[122]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use LocalPackageFileGrant.ProtoReflect.Descriptor instead.
func (*LocalPackageFileGrant) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{122}
}

func (x *LocalPackageFileGrant) GetDigest() []byte {
	if x != nil {
		return x.Digest
	}
	return nil
}

func (x *LocalPackageFileGrant) GetFilename() string {
	if x != nil {
		return x.Filename
	}
	return ""
}

func (x *LocalPackageFileGrant) GetLength() uint64 {
	if x != nil {
		return x.Length
	}
	return 0
}

func (x *LocalPackageFileGrant) GetUrl() string {
	if x != nil {
		return x.Url
	}
	return ""
}

// External RecordOwner -> pod-supervisor. This REPLACES the retired private_package_file_chunk
// relay (th-094), which moved up to 1 GiB per operation through the supervisor in 1 MiB frames.
// The record owner holds the wheels, so it uploads them to the store itself and names the
// resulting objects here: the frame carries capabilities, never content, and its whole bound is
// MaxLocalPackageFiles x MaxLocalPackageGrantURLBytes -- far under MaxInlineControlBytes.
//
// Custody ordering is unchanged. The supervisor reserves each file in its durable ledger BEFORE
// opening a reader, streams from the granted URL through its one download edge, verifies the
// SHA-256 against the reservation, advances the ledger, and answers exactly one
// LocalPackageFileStatus per file. Nothing is forwarded to Runtime.
//
// Exact replay re-answers from the ledger. Every TRANSPORT failure -- a spent capability, a
// dropped connection, a disconnected owner -- is a RESUMABLE stall rather than a refusal: the
// status stays RECEIVING with the byte count that actually landed and a safe code, and a new
// frame carrying fresh URLs continues from that prefix. Only bytes that are not the named
// object refuse, because a wrong prefix poisons every retry.
type LocalPackageFetchRequest struct {
	state              protoimpl.MessageState   `protogen:"open.v1"`
	RecordOwnerEpoch   uint64                   `protobuf:"varint,1,opt,name=record_owner_epoch,json=recordOwnerEpoch,proto3" json:"record_owner_epoch,omitempty"`
	ControlStreamEpoch uint64                   `protobuf:"varint,2,opt,name=control_stream_epoch,json=controlStreamEpoch,proto3" json:"control_stream_epoch,omitempty"`
	WorkerBootId       string                   `protobuf:"bytes,3,opt,name=worker_boot_id,json=workerBootId,proto3" json:"worker_boot_id,omitempty"`
	OperationId        string                   `protobuf:"bytes,5,opt,name=operation_id,json=operationId,proto3" json:"operation_id,omitempty"`
	SourceDigest       []byte                   `protobuf:"bytes,6,opt,name=source_digest,json=sourceDigest,proto3" json:"source_digest,omitempty"`
	Files              []*LocalPackageFileGrant `protobuf:"bytes,7,rep,name=files,proto3" json:"files,omitempty"` // sorted unique by digest; <= MaxLocalPackageFiles
	unknownFields      protoimpl.UnknownFields
	sizeCache          protoimpl.SizeCache
}

func (x *LocalPackageFetchRequest) Reset() {
	*x = LocalPackageFetchRequest{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[123]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *LocalPackageFetchRequest) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*LocalPackageFetchRequest) ProtoMessage() {}

func (x *LocalPackageFetchRequest) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[123]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use LocalPackageFetchRequest.ProtoReflect.Descriptor instead.
func (*LocalPackageFetchRequest) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{123}
}

func (x *LocalPackageFetchRequest) GetRecordOwnerEpoch() uint64 {
	if x != nil {
		return x.RecordOwnerEpoch
	}
	return 0
}

func (x *LocalPackageFetchRequest) GetControlStreamEpoch() uint64 {
	if x != nil {
		return x.ControlStreamEpoch
	}
	return 0
}

func (x *LocalPackageFetchRequest) GetWorkerBootId() string {
	if x != nil {
		return x.WorkerBootId
	}
	return ""
}

func (x *LocalPackageFetchRequest) GetOperationId() string {
	if x != nil {
		return x.OperationId
	}
	return ""
}

func (x *LocalPackageFetchRequest) GetSourceDigest() []byte {
	if x != nil {
		return x.SourceDigest
	}
	return nil
}

func (x *LocalPackageFetchRequest) GetFiles() []*LocalPackageFileGrant {
	if x != nil {
		return x.Files
	}
	return nil
}

// Durable progress/terminal observation for one exact file. Exact replay of an already verified
// prefix returns the same received byte count; changed identity under one operation refuses. A
// RECEIVING status CARRYING a safe code is a resumable stall -- the granted URL aged out, or the
// stream broke -- and the owner answers it with a fresh LocalPackageFetchRequest.
type LocalPackageFileStatus struct {
	state              protoimpl.MessageState `protogen:"open.v1"`
	RecordOwnerEpoch   uint64                 `protobuf:"varint,1,opt,name=record_owner_epoch,json=recordOwnerEpoch,proto3" json:"record_owner_epoch,omitempty"`
	ControlStreamEpoch uint64                 `protobuf:"varint,2,opt,name=control_stream_epoch,json=controlStreamEpoch,proto3" json:"control_stream_epoch,omitempty"`
	WorkerBootId       string                 `protobuf:"bytes,3,opt,name=worker_boot_id,json=workerBootId,proto3" json:"worker_boot_id,omitempty"`
	OperationId        string                 `protobuf:"bytes,5,opt,name=operation_id,json=operationId,proto3" json:"operation_id,omitempty"`
	SourceDigest       []byte                 `protobuf:"bytes,6,opt,name=source_digest,json=sourceDigest,proto3" json:"source_digest,omitempty"`
	Digest             []byte                 `protobuf:"bytes,7,opt,name=digest,proto3" json:"digest,omitempty"`
	Filename           string                 `protobuf:"bytes,8,opt,name=filename,proto3" json:"filename,omitempty"`
	Length             uint64                 `protobuf:"varint,10,opt,name=length,proto3" json:"length,omitempty"`
	State              LocalPackageFileState  `protobuf:"varint,11,opt,name=state,proto3,enum=cozy.worker.v1.LocalPackageFileState" json:"state,omitempty"`
	ReceivedBytes      uint64                 `protobuf:"varint,12,opt,name=received_bytes,json=receivedBytes,proto3" json:"received_bytes,omitempty"`
	SafeCode           string                 `protobuf:"bytes,13,opt,name=safe_code,json=safeCode,proto3" json:"safe_code,omitempty"`
	SafeDetail         string                 `protobuf:"bytes,14,opt,name=safe_detail,json=safeDetail,proto3" json:"safe_detail,omitempty"`
	unknownFields      protoimpl.UnknownFields
	sizeCache          protoimpl.SizeCache
}

func (x *LocalPackageFileStatus) Reset() {
	*x = LocalPackageFileStatus{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[124]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *LocalPackageFileStatus) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*LocalPackageFileStatus) ProtoMessage() {}

func (x *LocalPackageFileStatus) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[124]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use LocalPackageFileStatus.ProtoReflect.Descriptor instead.
func (*LocalPackageFileStatus) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{124}
}

func (x *LocalPackageFileStatus) GetRecordOwnerEpoch() uint64 {
	if x != nil {
		return x.RecordOwnerEpoch
	}
	return 0
}

func (x *LocalPackageFileStatus) GetControlStreamEpoch() uint64 {
	if x != nil {
		return x.ControlStreamEpoch
	}
	return 0
}

func (x *LocalPackageFileStatus) GetWorkerBootId() string {
	if x != nil {
		return x.WorkerBootId
	}
	return ""
}

func (x *LocalPackageFileStatus) GetOperationId() string {
	if x != nil {
		return x.OperationId
	}
	return ""
}

func (x *LocalPackageFileStatus) GetSourceDigest() []byte {
	if x != nil {
		return x.SourceDigest
	}
	return nil
}

func (x *LocalPackageFileStatus) GetDigest() []byte {
	if x != nil {
		return x.Digest
	}
	return nil
}

func (x *LocalPackageFileStatus) GetFilename() string {
	if x != nil {
		return x.Filename
	}
	return ""
}

func (x *LocalPackageFileStatus) GetLength() uint64 {
	if x != nil {
		return x.Length
	}
	return 0
}

func (x *LocalPackageFileStatus) GetState() LocalPackageFileState {
	if x != nil {
		return x.State
	}
	return LocalPackageFileState_LOCAL_PACKAGE_FILE_STATE_UNSPECIFIED
}

func (x *LocalPackageFileStatus) GetReceivedBytes() uint64 {
	if x != nil {
		return x.ReceivedBytes
	}
	return 0
}

func (x *LocalPackageFileStatus) GetSafeCode() string {
	if x != nil {
		return x.SafeCode
	}
	return ""
}

func (x *LocalPackageFileStatus) GetSafeDetail() string {
	if x != nil {
		return x.SafeDetail
	}
	return ""
}

// Durable pre-attempt transfer tombstone. Only the currently claimed RecordOwner/session may
// author it. The supervisor journals abandonment before deleting this operation's partial/final
// carriers; an already-prepared PlacementSet/environment remains intact.
type LocalPackageAbort struct {
	state               protoimpl.MessageState `protogen:"open.v1"`
	RecordOwnerEpoch    uint64                 `protobuf:"varint,1,opt,name=record_owner_epoch,json=recordOwnerEpoch,proto3" json:"record_owner_epoch,omitempty"`
	ControlStreamEpoch  uint64                 `protobuf:"varint,2,opt,name=control_stream_epoch,json=controlStreamEpoch,proto3" json:"control_stream_epoch,omitempty"`
	WorkerBootId        string                 `protobuf:"bytes,3,opt,name=worker_boot_id,json=workerBootId,proto3" json:"worker_boot_id,omitempty"`
	OperationId         string                 `protobuf:"bytes,5,opt,name=operation_id,json=operationId,proto3" json:"operation_id,omitempty"`
	SourceDigest        []byte                 `protobuf:"bytes,6,opt,name=source_digest,json=sourceDigest,proto3" json:"source_digest,omitempty"`
	LocalRevisionDigest []byte                 `protobuf:"bytes,7,opt,name=local_revision_digest,json=localRevisionDigest,proto3" json:"local_revision_digest,omitempty"`
	unknownFields       protoimpl.UnknownFields
	sizeCache           protoimpl.SizeCache
}

func (x *LocalPackageAbort) Reset() {
	*x = LocalPackageAbort{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[125]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *LocalPackageAbort) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*LocalPackageAbort) ProtoMessage() {}

func (x *LocalPackageAbort) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[125]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use LocalPackageAbort.ProtoReflect.Descriptor instead.
func (*LocalPackageAbort) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{125}
}

func (x *LocalPackageAbort) GetRecordOwnerEpoch() uint64 {
	if x != nil {
		return x.RecordOwnerEpoch
	}
	return 0
}

func (x *LocalPackageAbort) GetControlStreamEpoch() uint64 {
	if x != nil {
		return x.ControlStreamEpoch
	}
	return 0
}

func (x *LocalPackageAbort) GetWorkerBootId() string {
	if x != nil {
		return x.WorkerBootId
	}
	return ""
}

func (x *LocalPackageAbort) GetOperationId() string {
	if x != nil {
		return x.OperationId
	}
	return ""
}

func (x *LocalPackageAbort) GetSourceDigest() []byte {
	if x != nil {
		return x.SourceDigest
	}
	return nil
}

func (x *LocalPackageAbort) GetLocalRevisionDigest() []byte {
	if x != nil {
		return x.LocalRevisionDigest
	}
	return nil
}

// Durable supervisor acknowledgement. Exact replay returns REPLAYED; a changed identity or a
// later chunk/desired state under the tombstoned operation refuses with bounded safe detail.
type LocalPackageAbortStatus struct {
	state               protoimpl.MessageState   `protogen:"open.v1"`
	RecordOwnerEpoch    uint64                   `protobuf:"varint,1,opt,name=record_owner_epoch,json=recordOwnerEpoch,proto3" json:"record_owner_epoch,omitempty"`
	ControlStreamEpoch  uint64                   `protobuf:"varint,2,opt,name=control_stream_epoch,json=controlStreamEpoch,proto3" json:"control_stream_epoch,omitempty"`
	WorkerBootId        string                   `protobuf:"bytes,3,opt,name=worker_boot_id,json=workerBootId,proto3" json:"worker_boot_id,omitempty"`
	OperationId         string                   `protobuf:"bytes,5,opt,name=operation_id,json=operationId,proto3" json:"operation_id,omitempty"`
	SourceDigest        []byte                   `protobuf:"bytes,6,opt,name=source_digest,json=sourceDigest,proto3" json:"source_digest,omitempty"`
	LocalRevisionDigest []byte                   `protobuf:"bytes,7,opt,name=local_revision_digest,json=localRevisionDigest,proto3" json:"local_revision_digest,omitempty"`
	Outcome             LocalPackageAbortOutcome `protobuf:"varint,8,opt,name=outcome,proto3,enum=cozy.worker.v1.LocalPackageAbortOutcome" json:"outcome,omitempty"`
	SafeCode            string                   `protobuf:"bytes,9,opt,name=safe_code,json=safeCode,proto3" json:"safe_code,omitempty"`
	SafeDetail          string                   `protobuf:"bytes,10,opt,name=safe_detail,json=safeDetail,proto3" json:"safe_detail,omitempty"`
	unknownFields       protoimpl.UnknownFields
	sizeCache           protoimpl.SizeCache
}

func (x *LocalPackageAbortStatus) Reset() {
	*x = LocalPackageAbortStatus{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[126]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *LocalPackageAbortStatus) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*LocalPackageAbortStatus) ProtoMessage() {}

func (x *LocalPackageAbortStatus) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[126]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use LocalPackageAbortStatus.ProtoReflect.Descriptor instead.
func (*LocalPackageAbortStatus) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{126}
}

func (x *LocalPackageAbortStatus) GetRecordOwnerEpoch() uint64 {
	if x != nil {
		return x.RecordOwnerEpoch
	}
	return 0
}

func (x *LocalPackageAbortStatus) GetControlStreamEpoch() uint64 {
	if x != nil {
		return x.ControlStreamEpoch
	}
	return 0
}

func (x *LocalPackageAbortStatus) GetWorkerBootId() string {
	if x != nil {
		return x.WorkerBootId
	}
	return ""
}

func (x *LocalPackageAbortStatus) GetOperationId() string {
	if x != nil {
		return x.OperationId
	}
	return ""
}

func (x *LocalPackageAbortStatus) GetSourceDigest() []byte {
	if x != nil {
		return x.SourceDigest
	}
	return nil
}

func (x *LocalPackageAbortStatus) GetLocalRevisionDigest() []byte {
	if x != nil {
		return x.LocalRevisionDigest
	}
	return nil
}

func (x *LocalPackageAbortStatus) GetOutcome() LocalPackageAbortOutcome {
	if x != nil {
		return x.Outcome
	}
	return LocalPackageAbortOutcome_LOCAL_PACKAGE_ABORT_OUTCOME_UNSPECIFIED
}

func (x *LocalPackageAbortStatus) GetSafeCode() string {
	if x != nil {
		return x.SafeCode
	}
	return ""
}

func (x *LocalPackageAbortStatus) GetSafeDetail() string {
	if x != nil {
		return x.SafeDetail
	}
	return ""
}

// Durable owner->worker request. This is a typed protobuf frame, not a canonical document. The
// record owner journals the decision fields before send; Runtime journals the result fields
// before reply. Replay keys on the request tuple and compares the typed fields.
type WeightsFinalizeRequest struct {
	state                protoimpl.MessageState     `protogen:"open.v1"`
	RecordOwnerEpoch     uint64                     `protobuf:"varint,1,opt,name=record_owner_epoch,json=recordOwnerEpoch,proto3" json:"record_owner_epoch,omitempty"`
	ControlStreamEpoch   uint64                     `protobuf:"varint,2,opt,name=control_stream_epoch,json=controlStreamEpoch,proto3" json:"control_stream_epoch,omitempty"`
	WorkerBootId         string                     `protobuf:"bytes,3,opt,name=worker_boot_id,json=workerBootId,proto3" json:"worker_boot_id,omitempty"`
	RequestId            string                     `protobuf:"bytes,5,opt,name=request_id,json=requestId,proto3" json:"request_id,omitempty"`
	InvocationSpecDigest []byte                     `protobuf:"bytes,6,opt,name=invocation_spec_digest,json=invocationSpecDigest,proto3" json:"invocation_spec_digest,omitempty"`
	OutputSlot           string                     `protobuf:"bytes,7,opt,name=output_slot,json=outputSlot,proto3" json:"output_slot,omitempty"`
	Disposition          WeightsFinalizeDisposition `protobuf:"varint,8,opt,name=disposition,proto3,enum=cozy.worker.v1.WeightsFinalizeDisposition" json:"disposition,omitempty"`
	WeightsReceiptDigest []byte                     `protobuf:"bytes,9,opt,name=weights_receipt_digest,json=weightsReceiptDigest,proto3" json:"weights_receipt_digest,omitempty"`
	ScratchRootId        string                     `protobuf:"bytes,10,opt,name=scratch_root_id,json=scratchRootId,proto3" json:"scratch_root_id,omitempty"`
	OwnerAuthorityScope  string                     `protobuf:"bytes,11,opt,name=owner_authority_scope,json=ownerAuthorityScope,proto3" json:"owner_authority_scope,omitempty"`
	// Exact already-owned InvocationSpec/1 bytes, required and SHA-bound to field6.
	// Runtime re-validates canonical form and the declared model output even after its
	// ephemeral attempt history is gone. This is existing authority, never a new identity.
	// The complete request must remain within MaxInlineControlBytes.
	InvocationSpecCanonicalBytes []byte `protobuf:"bytes,12,opt,name=invocation_spec_canonical_bytes,json=invocationSpecCanonicalBytes,proto3" json:"invocation_spec_canonical_bytes,omitempty"`
	unknownFields                protoimpl.UnknownFields
	sizeCache                    protoimpl.SizeCache
}

func (x *WeightsFinalizeRequest) Reset() {
	*x = WeightsFinalizeRequest{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[127]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *WeightsFinalizeRequest) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*WeightsFinalizeRequest) ProtoMessage() {}

func (x *WeightsFinalizeRequest) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[127]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use WeightsFinalizeRequest.ProtoReflect.Descriptor instead.
func (*WeightsFinalizeRequest) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{127}
}

func (x *WeightsFinalizeRequest) GetRecordOwnerEpoch() uint64 {
	if x != nil {
		return x.RecordOwnerEpoch
	}
	return 0
}

func (x *WeightsFinalizeRequest) GetControlStreamEpoch() uint64 {
	if x != nil {
		return x.ControlStreamEpoch
	}
	return 0
}

func (x *WeightsFinalizeRequest) GetWorkerBootId() string {
	if x != nil {
		return x.WorkerBootId
	}
	return ""
}

func (x *WeightsFinalizeRequest) GetRequestId() string {
	if x != nil {
		return x.RequestId
	}
	return ""
}

func (x *WeightsFinalizeRequest) GetInvocationSpecDigest() []byte {
	if x != nil {
		return x.InvocationSpecDigest
	}
	return nil
}

func (x *WeightsFinalizeRequest) GetOutputSlot() string {
	if x != nil {
		return x.OutputSlot
	}
	return ""
}

func (x *WeightsFinalizeRequest) GetDisposition() WeightsFinalizeDisposition {
	if x != nil {
		return x.Disposition
	}
	return WeightsFinalizeDisposition_WEIGHTS_FINALIZE_DISPOSITION_UNSPECIFIED
}

func (x *WeightsFinalizeRequest) GetWeightsReceiptDigest() []byte {
	if x != nil {
		return x.WeightsReceiptDigest
	}
	return nil
}

func (x *WeightsFinalizeRequest) GetScratchRootId() string {
	if x != nil {
		return x.ScratchRootId
	}
	return ""
}

func (x *WeightsFinalizeRequest) GetOwnerAuthorityScope() string {
	if x != nil {
		return x.OwnerAuthorityScope
	}
	return ""
}

func (x *WeightsFinalizeRequest) GetInvocationSpecCanonicalBytes() []byte {
	if x != nil {
		return x.InvocationSpecCanonicalBytes
	}
	return nil
}

// Durable worker->owner result. A commit-first ABANDON_UNCOMMITTED race carries the released
// receipt as evidence; other outcomes may omit it. No extra acknowledgement exists.
type WeightsFinalizeResult struct {
	state                protoimpl.MessageState `protogen:"open.v1"`
	RecordOwnerEpoch     uint64                 `protobuf:"varint,1,opt,name=record_owner_epoch,json=recordOwnerEpoch,proto3" json:"record_owner_epoch,omitempty"`
	ControlStreamEpoch   uint64                 `protobuf:"varint,2,opt,name=control_stream_epoch,json=controlStreamEpoch,proto3" json:"control_stream_epoch,omitempty"`
	WorkerBootId         string                 `protobuf:"bytes,3,opt,name=worker_boot_id,json=workerBootId,proto3" json:"worker_boot_id,omitempty"`
	RequestId            string                 `protobuf:"bytes,5,opt,name=request_id,json=requestId,proto3" json:"request_id,omitempty"`
	InvocationSpecDigest []byte                 `protobuf:"bytes,6,opt,name=invocation_spec_digest,json=invocationSpecDigest,proto3" json:"invocation_spec_digest,omitempty"`
	OutputSlot           string                 `protobuf:"bytes,7,opt,name=output_slot,json=outputSlot,proto3" json:"output_slot,omitempty"`
	Outcome              WeightsFinalizeOutcome `protobuf:"varint,8,opt,name=outcome,proto3,enum=cozy.worker.v1.WeightsFinalizeOutcome" json:"outcome,omitempty"`
	WeightsReceipt       *WeightsReceiptRef     `protobuf:"bytes,9,opt,name=weights_receipt,json=weightsReceipt,proto3" json:"weights_receipt,omitempty"`
	OwnerAuthorityScope  string                 `protobuf:"bytes,10,opt,name=owner_authority_scope,json=ownerAuthorityScope,proto3" json:"owner_authority_scope,omitempty"`
	unknownFields        protoimpl.UnknownFields
	sizeCache            protoimpl.SizeCache
}

func (x *WeightsFinalizeResult) Reset() {
	*x = WeightsFinalizeResult{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[128]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *WeightsFinalizeResult) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*WeightsFinalizeResult) ProtoMessage() {}

func (x *WeightsFinalizeResult) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[128]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use WeightsFinalizeResult.ProtoReflect.Descriptor instead.
func (*WeightsFinalizeResult) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{128}
}

func (x *WeightsFinalizeResult) GetRecordOwnerEpoch() uint64 {
	if x != nil {
		return x.RecordOwnerEpoch
	}
	return 0
}

func (x *WeightsFinalizeResult) GetControlStreamEpoch() uint64 {
	if x != nil {
		return x.ControlStreamEpoch
	}
	return 0
}

func (x *WeightsFinalizeResult) GetWorkerBootId() string {
	if x != nil {
		return x.WorkerBootId
	}
	return ""
}

func (x *WeightsFinalizeResult) GetRequestId() string {
	if x != nil {
		return x.RequestId
	}
	return ""
}

func (x *WeightsFinalizeResult) GetInvocationSpecDigest() []byte {
	if x != nil {
		return x.InvocationSpecDigest
	}
	return nil
}

func (x *WeightsFinalizeResult) GetOutputSlot() string {
	if x != nil {
		return x.OutputSlot
	}
	return ""
}

func (x *WeightsFinalizeResult) GetOutcome() WeightsFinalizeOutcome {
	if x != nil {
		return x.Outcome
	}
	return WeightsFinalizeOutcome_WEIGHTS_FINALIZE_OUTCOME_UNSPECIFIED
}

func (x *WeightsFinalizeResult) GetWeightsReceipt() *WeightsReceiptRef {
	if x != nil {
		return x.WeightsReceipt
	}
	return nil
}

func (x *WeightsFinalizeResult) GetOwnerAuthorityScope() string {
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
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[129]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *ResultEnvelope) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*ResultEnvelope) ProtoMessage() {}

func (x *ResultEnvelope) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[129]
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
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{129}
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
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[130]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *AdjustmentRow) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*AdjustmentRow) ProtoMessage() {}

func (x *AdjustmentRow) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[130]
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
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{130}
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
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[131]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *OutcomeCause) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*OutcomeCause) ProtoMessage() {}

func (x *OutcomeCause) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[131]
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
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{131}
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
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[132]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *ResourceShortfall) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*ResourceShortfall) ProtoMessage() {}

func (x *ResourceShortfall) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[132]
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
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{132}
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
	state                protoimpl.MessageState `protogen:"open.v1"`
	RecordOwnerEpoch     uint64                 `protobuf:"varint,1,opt,name=record_owner_epoch,json=recordOwnerEpoch,proto3" json:"record_owner_epoch,omitempty"`
	ControlStreamEpoch   uint64                 `protobuf:"varint,2,opt,name=control_stream_epoch,json=controlStreamEpoch,proto3" json:"control_stream_epoch,omitempty"`
	WorkerBootId         string                 `protobuf:"bytes,3,opt,name=worker_boot_id,json=workerBootId,proto3" json:"worker_boot_id,omitempty"`
	RequestId            string                 `protobuf:"bytes,5,opt,name=request_id,json=requestId,proto3" json:"request_id,omitempty"`
	AttemptOrdinal       uint64                 `protobuf:"varint,6,opt,name=attempt_ordinal,json=attemptOrdinal,proto3" json:"attempt_ordinal,omitempty"`
	InvocationSpecDigest []byte                 `protobuf:"bytes,7,opt,name=invocation_spec_digest,json=invocationSpecDigest,proto3" json:"invocation_spec_digest,omitempty"`
	OutcomeId            string                 `protobuf:"bytes,8,opt,name=outcome_id,json=outcomeId,proto3" json:"outcome_id,omitempty"`             // echo; a mismatched ack is NOT an ack — replay continues
	OutcomeDigest        []byte                 `protobuf:"bytes,9,opt,name=outcome_digest,json=outcomeDigest,proto3" json:"outcome_digest,omitempty"` // echo; compared, never recomputed
	// MINOR 39: close this attempt while retaining its request-scoped writer identity,
	// epochs and artifact custody for pause/retry. Does not authorize further execution.
	// The owner sends false only after permanent abandonment or required final custody
	// and disposition have settled. Native roots are still controlled by their explicit
	// finalization/source-release operations; this flag never adopts or deletes bytes.
	RetainWork    bool `protobuf:"varint,10,opt,name=retain_work,json=retainWork,proto3" json:"retain_work,omitempty"`
	unknownFields protoimpl.UnknownFields
	sizeCache     protoimpl.SizeCache
}

func (x *AttemptOutcomeAck) Reset() {
	*x = AttemptOutcomeAck{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[133]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *AttemptOutcomeAck) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*AttemptOutcomeAck) ProtoMessage() {}

func (x *AttemptOutcomeAck) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[133]
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
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{133}
}

func (x *AttemptOutcomeAck) GetRecordOwnerEpoch() uint64 {
	if x != nil {
		return x.RecordOwnerEpoch
	}
	return 0
}

func (x *AttemptOutcomeAck) GetControlStreamEpoch() uint64 {
	if x != nil {
		return x.ControlStreamEpoch
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

func (x *AttemptOutcomeAck) GetRetainWork() bool {
	if x != nil {
		return x.RetainWork
	}
	return false
}

type JobCheckpointRequest struct {
	state              protoimpl.MessageState `protogen:"open.v1"`
	RecordOwnerEpoch   uint64                 `protobuf:"varint,1,opt,name=record_owner_epoch,json=recordOwnerEpoch,proto3" json:"record_owner_epoch,omitempty"`
	ControlStreamEpoch uint64                 `protobuf:"varint,2,opt,name=control_stream_epoch,json=controlStreamEpoch,proto3" json:"control_stream_epoch,omitempty"`
	WorkerBootId       string                 `protobuf:"bytes,3,opt,name=worker_boot_id,json=workerBootId,proto3" json:"worker_boot_id,omitempty"`
	RequestId          string                 `protobuf:"bytes,5,opt,name=request_id,json=requestId,proto3" json:"request_id,omitempty"`
	AttemptOrdinal     uint64                 `protobuf:"varint,6,opt,name=attempt_ordinal,json=attemptOrdinal,proto3" json:"attempt_ordinal,omitempty"`
	OperationKey       string                 `protobuf:"bytes,7,opt,name=operation_key,json=operationKey,proto3" json:"operation_key,omitempty"`    // worker-minted, attempt-scoped idempotency key
	LogicalKey         string                 `protobuf:"bytes,8,opt,name=logical_key,json=logicalKey,proto3" json:"logical_key,omitempty"`          // author-stable checkpoint slot
	ContentDigest      []byte                 `protobuf:"bytes,9,opt,name=content_digest,json=contentDigest,proto3" json:"content_digest,omitempty"` // class (b): the checkpoint artifact's bytes
	Seq                uint64                 `protobuf:"varint,10,opt,name=seq,proto3" json:"seq,omitempty"`                                        // ordering only, never identity
	Artifact           *OutputEntry           `protobuf:"bytes,11,opt,name=artifact,proto3" json:"artifact,omitempty"`
	unknownFields      protoimpl.UnknownFields
	sizeCache          protoimpl.SizeCache
}

func (x *JobCheckpointRequest) Reset() {
	*x = JobCheckpointRequest{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[134]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *JobCheckpointRequest) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*JobCheckpointRequest) ProtoMessage() {}

func (x *JobCheckpointRequest) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[134]
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
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{134}
}

func (x *JobCheckpointRequest) GetRecordOwnerEpoch() uint64 {
	if x != nil {
		return x.RecordOwnerEpoch
	}
	return 0
}

func (x *JobCheckpointRequest) GetControlStreamEpoch() uint64 {
	if x != nil {
		return x.ControlStreamEpoch
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
	state              protoimpl.MessageState `protogen:"open.v1"`
	RecordOwnerEpoch   uint64                 `protobuf:"varint,1,opt,name=record_owner_epoch,json=recordOwnerEpoch,proto3" json:"record_owner_epoch,omitempty"`
	ControlStreamEpoch uint64                 `protobuf:"varint,2,opt,name=control_stream_epoch,json=controlStreamEpoch,proto3" json:"control_stream_epoch,omitempty"`
	WorkerBootId       string                 `protobuf:"bytes,3,opt,name=worker_boot_id,json=workerBootId,proto3" json:"worker_boot_id,omitempty"`
	RequestId          string                 `protobuf:"bytes,5,opt,name=request_id,json=requestId,proto3" json:"request_id,omitempty"`
	AttemptOrdinal     uint64                 `protobuf:"varint,6,opt,name=attempt_ordinal,json=attemptOrdinal,proto3" json:"attempt_ordinal,omitempty"`
	OperationKey       string                 `protobuf:"bytes,7,opt,name=operation_key,json=operationKey,proto3" json:"operation_key,omitempty"`    // echo
	LogicalKey         string                 `protobuf:"bytes,8,opt,name=logical_key,json=logicalKey,proto3" json:"logical_key,omitempty"`          // echo
	ContentDigest      []byte                 `protobuf:"bytes,9,opt,name=content_digest,json=contentDigest,proto3" json:"content_digest,omitempty"` // echo; mismatch vs recorded identity is a conflict
	ReceiptId          string                 `protobuf:"bytes,10,opt,name=receipt_id,json=receiptId,proto3" json:"receipt_id,omitempty"`            // RecordOwner-minted durable record id
	Outcome            CheckpointOutcome      `protobuf:"varint,11,opt,name=outcome,proto3,enum=cozy.worker.v1.CheckpointOutcome" json:"outcome,omitempty"`
	Fault              *CheckpointFault       `protobuf:"bytes,12,opt,name=fault,proto3" json:"fault,omitempty"` // set iff CONFLICT or REFUSED
	unknownFields      protoimpl.UnknownFields
	sizeCache          protoimpl.SizeCache
}

func (x *JobCheckpointReceipt) Reset() {
	*x = JobCheckpointReceipt{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[135]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *JobCheckpointReceipt) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*JobCheckpointReceipt) ProtoMessage() {}

func (x *JobCheckpointReceipt) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[135]
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
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{135}
}

func (x *JobCheckpointReceipt) GetRecordOwnerEpoch() uint64 {
	if x != nil {
		return x.RecordOwnerEpoch
	}
	return 0
}

func (x *JobCheckpointReceipt) GetControlStreamEpoch() uint64 {
	if x != nil {
		return x.ControlStreamEpoch
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
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[136]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *CheckpointFault) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*CheckpointFault) ProtoMessage() {}

func (x *CheckpointFault) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[136]
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
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{136}
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
	state              protoimpl.MessageState `protogen:"open.v1"`
	RecordOwnerEpoch   uint64                 `protobuf:"varint,1,opt,name=record_owner_epoch,json=recordOwnerEpoch,proto3" json:"record_owner_epoch,omitempty"`
	ControlStreamEpoch uint64                 `protobuf:"varint,2,opt,name=control_stream_epoch,json=controlStreamEpoch,proto3" json:"control_stream_epoch,omitempty"`
	WorkerBootId       string                 `protobuf:"bytes,3,opt,name=worker_boot_id,json=workerBootId,proto3" json:"worker_boot_id,omitempty"`
	RequestId          string                 `protobuf:"bytes,5,opt,name=request_id,json=requestId,proto3" json:"request_id,omitempty"`
	AttemptOrdinal     uint64                 `protobuf:"varint,6,opt,name=attempt_ordinal,json=attemptOrdinal,proto3" json:"attempt_ordinal,omitempty"`
	OperationKey       string                 `protobuf:"bytes,7,opt,name=operation_key,json=operationKey,proto3" json:"operation_key,omitempty"`
	ReceiptId          string                 `protobuf:"bytes,8,opt,name=receipt_id,json=receiptId,proto3" json:"receipt_id,omitempty"` // echo; a mismatched ack is not an ack
	ContentDigest      []byte                 `protobuf:"bytes,9,opt,name=content_digest,json=contentDigest,proto3" json:"content_digest,omitempty"`
	unknownFields      protoimpl.UnknownFields
	sizeCache          protoimpl.SizeCache
}

func (x *JobCheckpointAck) Reset() {
	*x = JobCheckpointAck{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[137]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *JobCheckpointAck) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*JobCheckpointAck) ProtoMessage() {}

func (x *JobCheckpointAck) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[137]
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
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{137}
}

func (x *JobCheckpointAck) GetRecordOwnerEpoch() uint64 {
	if x != nil {
		return x.RecordOwnerEpoch
	}
	return 0
}

func (x *JobCheckpointAck) GetControlStreamEpoch() uint64 {
	if x != nil {
		return x.ControlStreamEpoch
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
	state              protoimpl.MessageState `protogen:"open.v1"`
	RecordOwnerEpoch   uint64                 `protobuf:"varint,1,opt,name=record_owner_epoch,json=recordOwnerEpoch,proto3" json:"record_owner_epoch,omitempty"`
	ControlStreamEpoch uint64                 `protobuf:"varint,2,opt,name=control_stream_epoch,json=controlStreamEpoch,proto3" json:"control_stream_epoch,omitempty"` // binds the watch to the fenced control stream
	WorkerBootId       string                 `protobuf:"bytes,3,opt,name=worker_boot_id,json=workerBootId,proto3" json:"worker_boot_id,omitempty"`
	RequestId          string                 `protobuf:"bytes,5,opt,name=request_id,json=requestId,proto3" json:"request_id,omitempty"` // empty = all attempts on this stream
	unknownFields      protoimpl.UnknownFields
	sizeCache          protoimpl.SizeCache
}

func (x *ProgressOpen) Reset() {
	*x = ProgressOpen{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[138]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *ProgressOpen) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*ProgressOpen) ProtoMessage() {}

func (x *ProgressOpen) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[138]
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
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{138}
}

func (x *ProgressOpen) GetRecordOwnerEpoch() uint64 {
	if x != nil {
		return x.RecordOwnerEpoch
	}
	return 0
}

func (x *ProgressOpen) GetControlStreamEpoch() uint64 {
	if x != nil {
		return x.ControlStreamEpoch
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
	state              protoimpl.MessageState `protogen:"open.v1"`
	RecordOwnerEpoch   uint64                 `protobuf:"varint,1,opt,name=record_owner_epoch,json=recordOwnerEpoch,proto3" json:"record_owner_epoch,omitempty"`
	ControlStreamEpoch uint64                 `protobuf:"varint,2,opt,name=control_stream_epoch,json=controlStreamEpoch,proto3" json:"control_stream_epoch,omitempty"`
	WorkerBootId       string                 `protobuf:"bytes,3,opt,name=worker_boot_id,json=workerBootId,proto3" json:"worker_boot_id,omitempty"`
	RequestId          string                 `protobuf:"bytes,5,opt,name=request_id,json=requestId,proto3" json:"request_id,omitempty"`
	AttemptOrdinal     uint64                 `protobuf:"varint,6,opt,name=attempt_ordinal,json=attemptOrdinal,proto3" json:"attempt_ordinal,omitempty"`
	Seq                uint64                 `protobuf:"varint,7,opt,name=seq,proto3" json:"seq,omitempty"` // starts at 1, strictly increasing per (request_id,
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
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[139]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *AttemptProgress) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*AttemptProgress) ProtoMessage() {}

func (x *AttemptProgress) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[139]
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
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{139}
}

func (x *AttemptProgress) GetRecordOwnerEpoch() uint64 {
	if x != nil {
		return x.RecordOwnerEpoch
	}
	return 0
}

func (x *AttemptProgress) GetControlStreamEpoch() uint64 {
	if x != nil {
		return x.ControlStreamEpoch
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
	state             protoimpl.MessageState `protogen:"open.v1"`
	EnvironmentDigest string                 `protobuf:"bytes,2,opt,name=environment_digest,json=environmentDigest,proto3" json:"environment_digest,omitempty"` // class (a), sha256:<hex>: exact selected Environment
	PayloadDigest     string                 `protobuf:"bytes,4,opt,name=payload_digest,json=payloadDigest,proto3" json:"payload_digest,omitempty"`             // class (a): the canonical typed-argument document
	Inputs            []*InputBinding        `protobuf:"bytes,5,rep,name=inputs,proto3" json:"inputs,omitempty"`                                                // ORDERED input identities — INSIDE the digest
	Outputs           []*OutputBinding       `protobuf:"bytes,6,rep,name=outputs,proto3" json:"outputs,omitempty"`                                              // output ids/kinds/limits — INSIDE the digest
	DeadlineUnixMs    uint64                 `protobuf:"varint,7,opt,name=deadline_unix_ms,json=deadlineUnixMs,proto3" json:"deadline_unix_ms,omitempty"`       // absolute attempt deadline; worker-enforced; 0 = none
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
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[140]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *InvocationSpec) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*InvocationSpec) ProtoMessage() {}

func (x *InvocationSpec) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[140]
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
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{140}
}

func (x *InvocationSpec) GetEnvironmentDigest() string {
	if x != nil {
		return x.EnvironmentDigest
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
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[141]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *InputBinding) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*InputBinding) ProtoMessage() {}

func (x *InputBinding) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[141]
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
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{141}
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
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[142]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *OutputBinding) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*OutputBinding) ProtoMessage() {}

func (x *OutputBinding) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[142]
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
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{142}
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
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[143]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *ServingInvocationSpec) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*ServingInvocationSpec) ProtoMessage() {}

func (x *ServingInvocationSpec) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[143]
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
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{143}
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
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[144]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *JobInvocationSpec) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*JobInvocationSpec) ProtoMessage() {}

func (x *JobInvocationSpec) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[144]
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
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{144}
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
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[145]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *DeliveryGrant) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*DeliveryGrant) ProtoMessage() {}

func (x *DeliveryGrant) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[145]
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
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{145}
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
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[146]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *InputAccess) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*InputAccess) ProtoMessage() {}

func (x *InputAccess) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[146]
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
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{146}
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
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[147]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *OutputAccess) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*OutputAccess) ProtoMessage() {}

func (x *OutputAccess) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[147]
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
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{147}
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
	state           protoimpl.MessageState `protogen:"open.v1"`
	Issuer          string                 `protobuf:"bytes,1,opt,name=issuer,proto3" json:"issuer,omitempty"`
	KeyId           string                 `protobuf:"bytes,2,opt,name=key_id,json=keyId,proto3" json:"key_id,omitempty"`
	CredentialEpoch uint64                 `protobuf:"varint,3,opt,name=credential_epoch,json=credentialEpoch,proto3" json:"credential_epoch,omitempty"`
	ExpiresAtUnix   uint64                 `protobuf:"varint,4,opt,name=expires_at_unix,json=expiresAtUnix,proto3" json:"expires_at_unix,omitempty"`
	Token           []byte                 `protobuf:"bytes,5,opt,name=token,proto3" json:"token,omitempty"` // STRUCTURALLY SECRET; never in a digest, a log, or an
	unknownFields   protoimpl.UnknownFields
	sizeCache       protoimpl.SizeCache
}

func (x *DeliveryAccessCredential) Reset() {
	*x = DeliveryAccessCredential{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[148]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *DeliveryAccessCredential) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*DeliveryAccessCredential) ProtoMessage() {}

func (x *DeliveryAccessCredential) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[148]
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
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{148}
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

func (x *DeliveryAccessCredential) GetCredentialEpoch() uint64 {
	if x != nil {
		return x.CredentialEpoch
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
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[149]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *ResourceCaps) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*ResourceCaps) ProtoMessage() {}

func (x *ResourceCaps) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[149]
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
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{149}
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
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[150]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *PublicationContract) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*PublicationContract) ProtoMessage() {}

func (x *PublicationContract) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[150]
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
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{150}
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
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[151]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *WorkerResources) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*WorkerResources) ProtoMessage() {}

func (x *WorkerResources) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[151]
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
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{151}
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
	JobsInFlight           uint32 `protobuf:"varint,1,opt,name=jobs_in_flight,json=jobsInFlight,proto3" json:"jobs_in_flight,omitempty"`
	JobsAvailable          uint32 `protobuf:"varint,2,opt,name=jobs_available,json=jobsAvailable,proto3" json:"jobs_available,omitempty"`
	FreeDiskBytes          uint64 `protobuf:"varint,4,opt,name=free_disk_bytes,json=freeDiskBytes,proto3" json:"free_disk_bytes,omitempty"`
	OrchestrationInFlight  uint32 `protobuf:"varint,5,opt,name=orchestration_in_flight,json=orchestrationInFlight,proto3" json:"orchestration_in_flight,omitempty"`
	OrchestrationAvailable uint32 `protobuf:"varint,6,opt,name=orchestration_available,json=orchestrationAvailable,proto3" json:"orchestration_available,omitempty"`
	unknownFields          protoimpl.UnknownFields
	sizeCache              protoimpl.SizeCache
}

func (x *JobCapacity) Reset() {
	*x = JobCapacity{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[152]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *JobCapacity) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*JobCapacity) ProtoMessage() {}

func (x *JobCapacity) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[152]
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
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{152}
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

func (x *JobCapacity) GetOrchestrationInFlight() uint32 {
	if x != nil {
		return x.OrchestrationInFlight
	}
	return 0
}

func (x *JobCapacity) GetOrchestrationAvailable() uint32 {
	if x != nil {
		return x.OrchestrationAvailable
	}
	return 0
}

// Was ActiveAttempt (#481): the set holds unacked OUTCOMES too, and those are not active. At
// minor 23 (proto-026) it holds QUEUED attempts too: every accepted attempt appears here from
// admission to ack, so the orchestrator's records are exact from the observed queue and a
// queued attempt is never absent. Lane, order and fairness are worker facts the orchestrator
// RENDERS, never sets: the worker orders its lane (gpu-hot.md §4), the orchestrator offers depth.
type HeldAttempt struct {
	state                protoimpl.MessageState `protogen:"open.v1"`
	RequestId            string                 `protobuf:"bytes,1,opt,name=request_id,json=requestId,proto3" json:"request_id,omitempty"`
	AttemptOrdinal       uint64                 `protobuf:"varint,2,opt,name=attempt_ordinal,json=attemptOrdinal,proto3" json:"attempt_ordinal,omitempty"`
	Kind                 AttemptKind            `protobuf:"varint,3,opt,name=kind,proto3,enum=cozy.worker.v1.AttemptKind" json:"kind,omitempty"`
	State                AttemptState           `protobuf:"varint,4,opt,name=state,proto3,enum=cozy.worker.v1.AttemptState" json:"state,omitempty"`
	InvocationSpecDigest []byte                 `protobuf:"bytes,5,opt,name=invocation_spec_digest,json=invocationSpecDigest,proto3" json:"invocation_spec_digest,omitempty"`
	PlacementId          string                 `protobuf:"bytes,6,opt,name=placement_id,json=placementId,proto3" json:"placement_id,omitempty"`        // the executing placement; empty in job mode
	ExecutorEpoch        uint64                 `protobuf:"varint,7,opt,name=executor_epoch,json=executorEpoch,proto3" json:"executor_epoch,omitempty"` // the epoch it runs/ran under; 0 while QUEUED — a queued
	// attempt is bound to no executor and survives a respawn
	OutcomeId     string `protobuf:"bytes,8,opt,name=outcome_id,json=outcomeId,proto3" json:"outcome_id,omitempty"`             // set iff state is OUTCOME_PENDING_ACK
	OutcomeDigest []byte `protobuf:"bytes,9,opt,name=outcome_digest,json=outcomeDigest,proto3" json:"outcome_digest,omitempty"` // class (a); set iff state is OUTCOME_PENDING_ACK
	LaneId        string `protobuf:"bytes,10,opt,name=lane_id,json=laneId,proto3" json:"lane_id,omitempty"`                     // proto-026: the DeviceLane this attempt is queued on,
	// running on, or ran on; empty in job mode
	QueuePosition uint32 `protobuf:"varint,11,opt,name=queue_position,json=queuePosition,proto3" json:"queue_position,omitempty"` // the worker's CURRENT order among this lane's QUEUED
	// attempts: 0 = next on the device. Meaningful only while
	// QUEUED; 0 otherwise. Re-derived at every device entry.
	Overtaken uint32 `protobuf:"varint,12,opt,name=overtaken,proto3" json:"overtaken,omitempty"` // times a later-admitted attempt entered the device ahead
	// of this one; monotone for the attempt's life
	OvertakeBudget uint32 `protobuf:"varint,13,opt,name=overtake_budget,json=overtakeBudget,proto3" json:"overtake_budget,omitempty"` // = the attempt's queue position at ADMISSION, the most it
	// may be overtaken; spent => picked before any class.
	// A count the queue itself yields, never a constant or env.
	PlanDigest    []byte `protobuf:"bytes,14,opt,name=plan_digest,json=planDigest,proto3" json:"plan_digest,omitempty"` // class (a): AttemptPlan (cr-007), bound at DEVICE ENTRY;
	unknownFields protoimpl.UnknownFields
	sizeCache     protoimpl.SizeCache
}

func (x *HeldAttempt) Reset() {
	*x = HeldAttempt{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[153]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *HeldAttempt) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*HeldAttempt) ProtoMessage() {}

func (x *HeldAttempt) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[153]
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
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{153}
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

func (x *HeldAttempt) GetExecutorEpoch() uint64 {
	if x != nil {
		return x.ExecutorEpoch
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

func (x *HeldAttempt) GetLaneId() string {
	if x != nil {
		return x.LaneId
	}
	return ""
}

func (x *HeldAttempt) GetQueuePosition() uint32 {
	if x != nil {
		return x.QueuePosition
	}
	return 0
}

func (x *HeldAttempt) GetOvertaken() uint32 {
	if x != nil {
		return x.Overtaken
	}
	return 0
}

func (x *HeldAttempt) GetOvertakeBudget() uint32 {
	if x != nil {
		return x.OvertakeBudget
	}
	return 0
}

func (x *HeldAttempt) GetPlanDigest() []byte {
	if x != nil {
		return x.PlanDigest
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
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[154]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *Fault) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*Fault) ProtoMessage() {}

func (x *Fault) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[154]
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
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{154}
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
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[155]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *OutputManifest) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*OutputManifest) ProtoMessage() {}

func (x *OutputManifest) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[155]
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
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{155}
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
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[156]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *OutputEntry) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*OutputEntry) ProtoMessage() {}

func (x *OutputEntry) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[156]
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
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{156}
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
	DeviceLeaseMs         uint64                 `protobuf:"varint,8,opt,name=device_lease_ms,json=deviceLeaseMs,proto3" json:"device_lease_ms,omitempty"` // worker-attested: device ENTRY -> device RELEASE (the
	// executor's invoke reply, gpu-hot.md §3). The encode is
	// EXCLUDED at minor 23 — it is post_ms — so this is the
	// device phase and nothing else.
	DeviceCount      uint32   `protobuf:"varint,9,opt,name=device_count,json=deviceCount,proto3" json:"device_count,omitempty"`
	HandlerMs        uint64   `protobuf:"varint,10,opt,name=handler_ms,json=handlerMs,proto3" json:"handler_ms,omitempty"`
	FinalizationMs   uint64   `protobuf:"varint,11,opt,name=finalization_ms,json=finalizationMs,proto3" json:"finalization_ms,omitempty"`
	UnverifiedFields []string `protobuf:"bytes,12,rep,name=unverified_fields,json=unverifiedFields,proto3" json:"unverified_fields,omitempty"`
	PostMs           uint64   `protobuf:"varint,13,opt,name=post_ms,json=postMs,proto3" json:"post_ms,omitempty"` // proto-026: device release -> outcome sent — the post
	unknownFields    protoimpl.UnknownFields
	sizeCache        protoimpl.SizeCache
}

func (x *AttemptMetrics) Reset() {
	*x = AttemptMetrics{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[157]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *AttemptMetrics) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*AttemptMetrics) ProtoMessage() {}

func (x *AttemptMetrics) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[157]
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
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{157}
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

func (x *AttemptMetrics) GetPostMs() uint64 {
	if x != nil {
		return x.PostMs
	}
	return 0
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
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[158]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *TriageBundleRef) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*TriageBundleRef) ProtoMessage() {}

func (x *TriageBundleRef) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[158]
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
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{158}
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
	"\x1bcozy/worker/v1/worker.proto\x12\x0ecozy.worker.v1\"\xe4\x01\n" +
	"\x10WeightsHostEvent\x12<\n" +
	"\x06intent\x18\x01 \x01(\v2\".cozy.worker.v1.WeightsIntentFrameH\x00R\x06intent\x12?\n" +
	"\areceipt\x18\x02 \x01(\v2#.cozy.worker.v1.WeightsReceiptFrameH\x00R\areceipt\x12H\n" +
	"\n" +
	"checkpoint\x18\x03 \x01(\v2&.cozy.worker.v1.WeightsCheckpointFrameH\x00R\n" +
	"checkpointB\a\n" +
	"\x05event\"\x15\n" +
	"\x13ProtocolInfoRequest\"a\n" +
	"\x12ProtocolInfoResult\x12\x1d\n" +
	"\n" +
	"wire_minor\x18\x01 \x01(\rR\twireMinor\x12,\n" +
	"\x12minimum_wire_minor\x18\x02 \x01(\rR\x10minimumWireMinor\"\xce\x02\n" +
	"\x15PreparePackageSetCall\x12+\n" +
	"\x05claim\x18\x01 \x01(\v2\x15.cozy.worker.v1.ClaimR\x05claim\x12B\n" +
	"\vpackage_set\x18\x02 \x01(\v2!.cozy.worker.v1.DesiredPackageSetR\n" +
	"packageSet\x12 \n" +
	"\vapplication\x18\x03 \x01(\tR\vapplication\x12(\n" +
	"\x10model_slot_paths\x18\x04 \x03(\tR\x0emodelSlotPaths\x12G\n" +
	"\x0fimage_inventory\x18\x05 \x01(\v2\x1e.cozy.worker.v1.ImageInventoryR\x0eimageInventory\x12/\n" +
	"\x13locked_requirements\x18\x06 \x01(\fR\x12lockedRequirements\"\x9a\x01\n" +
	"\x17PrepareLocalPackageCall\x12+\n" +
	"\x05claim\x18\x01 \x01(\v2\x15.cozy.worker.v1.ClaimR\x05claim\x12R\n" +
	"\x11local_package_set\x18\x02 \x01(\v2&.cozy.worker.v1.DesiredLocalPackageSetR\x0flocalPackageSet\"\xaa\x01\n" +
	"\x1bPreparePrivatePlacementCall\x12+\n" +
	"\x05claim\x18\x01 \x01(\v2\x15.cozy.worker.v1.ClaimR\x05claim\x12^\n" +
	"\x15private_placement_set\x18\x02 \x01(\v2*.cozy.worker.v1.DesiredPrivatePlacementSetR\x13privatePlacementSet\"\x98\x02\n" +
	"\fPrepareEvent\x122\n" +
	"\x05stage\x18\x01 \x01(\x0e2\x1c.cozy.worker.v1.PrepareStageR\x05stage\x12\x1f\n" +
	"\vtotal_bytes\x18\x02 \x01(\x04R\n" +
	"totalBytes\x12+\n" +
	"\x11transferred_bytes\x18\x03 \x01(\x04R\x10transferredBytes\x12H\n" +
	"\rplacement_set\x18\x04 \x01(\v2#.cozy.worker.v1.DesiredPlacementSetR\fplacementSet\x12\x1b\n" +
	"\tsafe_code\x18\x05 \x01(\tR\bsafeCode\x12\x1f\n" +
	"\vsafe_detail\x18\x06 \x01(\tR\n" +
	"safeDetail\"\x84\x01\n" +
	"\x13ModelSourceFileCall\x12+\n" +
	"\x05claim\x18\x01 \x01(\v2\x15.cozy.worker.v1.ClaimR\x05claim\x12@\n" +
	"\arequest\x18\x02 \x01(\v2&.cozy.worker.v1.ModelSourceFileRequestR\arequest\"\x8a\x01\n" +
	"\x16ModelSourcePrepareCall\x12+\n" +
	"\x05claim\x18\x01 \x01(\v2\x15.cozy.worker.v1.ClaimR\x05claim\x12C\n" +
	"\arequest\x18\x02 \x01(\v2).cozy.worker.v1.ModelSourcePrepareRequestR\arequest\"\xa0\x01\n" +
	"\x16ModelSourceReleaseCall\x12+\n" +
	"\x05claim\x18\x01 \x01(\v2\x15.cozy.worker.v1.ClaimR\x05claim\x12!\n" +
	"\foperation_id\x18\x02 \x01(\tR\voperationId\x126\n" +
	"\x17source_selection_digest\x18\x03 \x01(\fR\x15sourceSelectionDigest\">\n" +
	"\x19ReleaseModelSourceRequest\x12!\n" +
	"\foperation_id\x18\x01 \x01(\tR\voperationId\"\x86\x01\n" +
	"\x14DerivedRetentionCall\x12+\n" +
	"\x05claim\x18\x01 \x01(\v2\x15.cozy.worker.v1.ClaimR\x05claim\x12A\n" +
	"\arequest\x18\x02 \x01(\v2'.cozy.worker.v1.DerivedRetentionRequestR\arequest\"\xaa\x01\n" +
	"\x17DerivedRetentionRequest\x124\n" +
	"\x16weights_transaction_id\x18\x01 \x01(\tR\x14weightsTransactionId\x126\n" +
	"\x17tensorfs_receipt_digest\x18\x02 \x01(\fR\x15tensorfsReceiptDigest\x12!\n" +
	"\fretention_id\x18\x03 \x01(\tR\vretentionId\"\xf6\x01\n" +
	"\x16DerivedRetentionResult\x124\n" +
	"\x16weights_transaction_id\x18\x01 \x01(\tR\x14weightsTransactionId\x126\n" +
	"\x17tensorfs_receipt_digest\x18\x02 \x01(\fR\x15tensorfsReceiptDigest\x12!\n" +
	"\fretention_id\x18\x03 \x01(\tR\vretentionId\x12/\n" +
	"\bmanifest\x18\x04 \x01(\v2\x13.cozy.worker.v1.RefR\bmanifest\x12\x1a\n" +
	"\breleased\x18\x05 \x01(\bR\breleased\"Y\n" +
	"\x18ReleaseModelSourceResult\x12!\n" +
	"\foperation_id\x18\x01 \x01(\tR\voperationId\x12\x1a\n" +
	"\breleased\x18\x02 \x01(\bR\breleased\"\xb4\x01\n" +
	"\x14ModelSourceAdoptCall\x12+\n" +
	"\x05claim\x18\x01 \x01(\v2\x15.cozy.worker.v1.ClaimR\x05claim\x12*\n" +
	"\x11from_operation_id\x18\x02 \x01(\tR\x0ffromOperationId\x12C\n" +
	"\arequest\x18\x03 \x01(\v2).cozy.worker.v1.ModelSourcePrepareRequestR\arequest\"\x82\x01\n" +
	"\x12CheckpointPageCall\x12+\n" +
	"\x05claim\x18\x01 \x01(\v2\x15.cozy.worker.v1.ClaimR\x05claim\x12?\n" +
	"\arequest\x18\x02 \x01(\v2%.cozy.worker.v1.CheckpointPageRequestR\arequest\"\x8a\x01\n" +
	"\x16CheckpointTransferCall\x12+\n" +
	"\x05claim\x18\x01 \x01(\v2\x15.cozy.worker.v1.ClaimR\x05claim\x12C\n" +
	"\arequest\x18\x02 \x01(\v2).cozy.worker.v1.CheckpointTransferRequestR\arequest\"\x88\x01\n" +
	"\x15LocalPackageFetchCall\x12+\n" +
	"\x05claim\x18\x01 \x01(\v2\x15.cozy.worker.v1.ClaimR\x05claim\x12B\n" +
	"\arequest\x18\x02 \x01(\v2(.cozy.worker.v1.LocalPackageFetchRequestR\arequest\"\xa6\x01\n" +
	"\x17LocalPackageUploadFrame\x12B\n" +
	"\x06header\x18\x01 \x01(\v2(.cozy.worker.v1.LocalPackageUploadHeaderH\x00R\x06header\x12?\n" +
	"\x05chunk\x18\x02 \x01(\v2'.cozy.worker.v1.LocalPackageUploadChunkH\x00R\x05chunkB\x06\n" +
	"\x04body\"\xc8\x01\n" +
	"\x18LocalPackageUploadHeader\x12+\n" +
	"\x05claim\x18\x01 \x01(\v2\x15.cozy.worker.v1.ClaimR\x05claim\x12!\n" +
	"\foperation_id\x18\x02 \x01(\tR\voperationId\x12#\n" +
	"\rsource_digest\x18\x03 \x01(\fR\fsourceDigest\x127\n" +
	"\x04file\x18\x04 \x01(\v2#.cozy.worker.v1.LocalPackageFileRefR\x04file\"E\n" +
	"\x17LocalPackageUploadChunk\x12\x16\n" +
	"\x06offset\x18\x01 \x01(\x04R\x06offset\x12\x12\n" +
	"\x04data\x18\x02 \x01(\fR\x04data\"\x81\x01\n" +
	"\x15LocalPackageAbortCall\x12+\n" +
	"\x05claim\x18\x01 \x01(\v2\x15.cozy.worker.v1.ClaimR\x05claim\x12;\n" +
	"\arequest\x18\x02 \x01(\v2!.cozy.worker.v1.LocalPackageAbortR\arequest\"\x84\x01\n" +
	"\x13WeightsTransferCall\x12+\n" +
	"\x05claim\x18\x01 \x01(\v2\x15.cozy.worker.v1.ClaimR\x05claim\x12@\n" +
	"\arequest\x18\x02 \x01(\v2&.cozy.worker.v1.WeightsTransferRequestR\arequest\"\x9d\x03\n" +
	"\x18PreparePackageSetRequest\x12/\n" +
	"\x13download_delegation\x18\x01 \x01(\fR\x12downloadDelegation\x12 \n" +
	"\vapplication\x18\x04 \x01(\tR\vapplication\x12(\n" +
	"\x10model_slot_paths\x18\x05 \x03(\tR\x0emodelSlotPaths\x12G\n" +
	"\x0fimage_inventory\x18\x06 \x01(\v2\x1e.cozy.worker.v1.ImageInventoryR\x0eimageInventory\x12/\n" +
	"\x13locked_requirements\x18\a \x01(\fR\x12lockedRequirements\x12!\n" +
	"\finstall_root\x18\b \x01(\tR\vinstallRoot\x12B\n" +
	"\x1ddownload_delegation_signature\x18\t \x01(\fR\x1bdownloadDelegationSignatureJ\x04\b\x02\x10\x03J\x04\b\x03\x10\x04R\x05filesR\x10environment_root\"\x8b\x01\n" +
	"\x0eImageInventory\x12\x18\n" +
	"\aprofile\x18\x01 \x01(\tR\aprofile\x12\x16\n" +
	"\x06python\x18\x02 \x01(\tR\x06python\x12G\n" +
	"\rdistributions\x18\x03 \x03(\v2!.cozy.worker.v1.ImageDistributionR\rdistributions\"Q\n" +
	"\x11ImageDistribution\x12\"\n" +
	"\fdistribution\x18\x01 \x01(\tR\fdistribution\x12\x18\n" +
	"\aversion\x18\x02 \x01(\tR\aversion\"c\n" +
	"\x17PreparePackageSetResult\x12H\n" +
	"\rplacement_set\x18\x01 \x01(\v2#.cozy.worker.v1.DesiredPlacementSetR\fplacementSet\"n\n" +
	"\"CheckPackageSetCompatibilityResult\x12!\n" +
	"\frefusal_code\x18\x01 \x01(\tR\vrefusalCode\x12%\n" +
	"\x0erefusal_detail\x18\x02 \x01(\tR\rrefusalDetail\"\x80\x02\n" +
	"\x1aPrepareLocalPackageRequest\x12!\n" +
	"\foperation_id\x18\x01 \x01(\tR\voperationId\x12<\n" +
	"\apackage\x18\x02 \x01(\v2\".cozy.worker.v1.DevelopmentPackageR\apackage\x129\n" +
	"\x06wheels\x18\x05 \x03(\v2!.cozy.worker.v1.LocalPackageWheelR\x06wheels\x12!\n" +
	"\finstall_root\x18\x06 \x01(\tR\vinstallRootJ\x04\b\x03\x10\x04J\x04\b\x04\x10\x05R\x05filesR\x10environment_root\"s\n" +
	"\x11LocalPackageWheel\x12\x16\n" +
	"\x06digest\x18\x01 \x01(\fR\x06digest\x12\x1a\n" +
	"\bfilename\x18\x02 \x01(\tR\bfilename\x12\x16\n" +
	"\x06length\x18\x03 \x01(\x04R\x06length\x12\x12\n" +
	"\x04path\x18\x04 \x01(\tR\x04path\"\xf9\x01\n" +
	"\x1ePreparePrivatePlacementRequest\x12!\n" +
	"\foperation_id\x18\x01 \x01(\tR\voperationId\x122\n" +
	"\x15local_revision_digest\x18\x02 \x01(\fR\x13localRevisionDigest\x12/\n" +
	"\x13download_delegation\x18\x03 \x01(\fR\x12downloadDelegation\x12B\n" +
	"\x1ddownload_delegation_signature\x18\x05 \x01(\fR\x1bdownloadDelegationSignatureJ\x04\b\x04\x10\x05R\x05files\"\xb4\x01\n" +
	"\x14LocalModelSourceFile\x12\x16\n" +
	"\x06member\x18\x01 \x01(\tR\x06member\x12\x1b\n" +
	"\tobject_id\x18\x02 \x01(\tR\bobjectId\x12\x16\n" +
	"\x06length\x18\x03 \x01(\x04R\x06length\x12\x12\n" +
	"\x04path\x18\x04 \x01(\tR\x04path\x12\x1f\n" +
	"\vheader_path\x18\x05 \x01(\tR\n" +
	"headerPath\x12\x1a\n" +
	"\bverified\x18\x06 \x01(\bR\bverified\"B\n" +
	"\x12ModelSourceProfile\x12\x12\n" +
	"\x04slot\x18\x01 \x01(\tR\x04slot\x12\x18\n" +
	"\aprofile\x18\x02 \x01(\tR\aprofile\"\x9f\x01\n" +
	"\x13PreparedModelSource\x12\x12\n" +
	"\x04slot\x18\x01 \x01(\tR\x04slot\x12\x18\n" +
	"\aprofile\x18\x02 \x01(\tR\aprofile\x12/\n" +
	"\bmanifest\x18\x03 \x01(\v2\x13.cozy.worker.v1.RefR\bmanifestJ\x04\b\x04\x10\x05R#checkpoint_evidence_canonical_bytes\"\xa1\x01\n" +
	"\x15ModelSourceCheckpoint\x12\x12\n" +
	"\x04slot\x18\x01 \x01(\tR\x04slot\x12'\n" +
	"\x04head\x18\x02 \x01(\v2\x13.cozy.worker.v1.RefR\x04head\x12\x1f\n" +
	"\vplan_digest\x18\x03 \x01(\fR\n" +
	"planDigest\x12\x14\n" +
	"\x05index\x18\x04 \x01(\x04R\x05index\x12\x14\n" +
	"\x05bytes\x18\x05 \x01(\x04R\x05bytes\"\x88\x01\n" +
	"\x17SourceCheckpointSubject\x12!\n" +
	"\foperation_id\x18\x01 \x01(\tR\voperationId\x126\n" +
	"\x17source_selection_digest\x18\x02 \x01(\fR\x15sourceSelectionDigest\x12\x12\n" +
	"\x04slot\x18\x03 \x01(\tR\x04slot\"\xa9\x02\n" +
	"\x18WeightsCheckpointSubject\x12\x1d\n" +
	"\n" +
	"request_id\x18\x01 \x01(\tR\trequestId\x124\n" +
	"\x16invocation_spec_digest\x18\x02 \x01(\fR\x14invocationSpecDigest\x12\x1f\n" +
	"\voutput_slot\x18\x03 \x01(\tR\n" +
	"outputSlot\x124\n" +
	"\x16weights_transaction_id\x18\x04 \x01(\tR\x14weightsTransactionId\x12!\n" +
	"\fwriter_epoch\x18\x05 \x01(\x04R\vwriterEpoch\x12>\n" +
	"\x1btensorfs_declaration_digest\x18\x06 \x01(\fR\x19tensorfsDeclarationDigest\"\xa4\x01\n" +
	"\x11CheckpointSubject\x12A\n" +
	"\x06source\x18\x01 \x01(\v2'.cozy.worker.v1.SourceCheckpointSubjectH\x00R\x06source\x12D\n" +
	"\aweights\x18\x02 \x01(\v2(.cozy.worker.v1.WeightsCheckpointSubjectH\x00R\aweightsB\x06\n" +
	"\x04kind\"U\n" +
	"\x10CheckpointObject\x12%\n" +
	"\x03ref\x18\x01 \x01(\v2\x13.cozy.worker.v1.RefR\x03ref\x12\x1a\n" +
	"\bmanifest\x18\x02 \x01(\bR\bmanifest\"\xe4\x02\n" +
	"\x15CheckpointPageRequest\x12,\n" +
	"\x12record_owner_epoch\x18\x01 \x01(\x04R\x10recordOwnerEpoch\x120\n" +
	"\x14control_stream_epoch\x18\x02 \x01(\x04R\x12controlStreamEpoch\x12$\n" +
	"\x0eworker_boot_id\x18\x03 \x01(\tR\fworkerBootId\x12;\n" +
	"\asubject\x18\x05 \x01(\v2!.cozy.worker.v1.CheckpointSubjectR\asubject\x12\x1f\n" +
	"\vplan_digest\x18\b \x01(\fR\n" +
	"planDigest\x12'\n" +
	"\x04head\x18\t \x01(\v2\x13.cozy.worker.v1.RefR\x04head\x12\x16\n" +
	"\x06offset\x18\n" +
	" \x01(\rR\x06offset\x12\x14\n" +
	"\x05limit\x18\v \x01(\rR\x05limitJ\x04\b\x04\x10\x05J\x04\b\x06\x10\aJ\x04\b\a\x10\b\"\xf9\x04\n" +
	"\x14CheckpointPageResult\x12,\n" +
	"\x12record_owner_epoch\x18\x01 \x01(\x04R\x10recordOwnerEpoch\x120\n" +
	"\x14control_stream_epoch\x18\x02 \x01(\x04R\x12controlStreamEpoch\x12$\n" +
	"\x0eworker_boot_id\x18\x03 \x01(\tR\fworkerBootId\x12;\n" +
	"\asubject\x18\x05 \x01(\v2!.cozy.worker.v1.CheckpointSubjectR\asubject\x12\x1f\n" +
	"\vplan_digest\x18\b \x01(\fR\n" +
	"planDigest\x12'\n" +
	"\x04head\x18\t \x01(\v2\x13.cozy.worker.v1.RefR\x04head\x12\x14\n" +
	"\x05index\x18\n" +
	" \x01(\x04R\x05index\x12\x14\n" +
	"\x05bytes\x18\v \x01(\x04R\x05bytes\x12/\n" +
	"\bprevious\x18\f \x01(\v2\x13.cozy.worker.v1.RefR\bprevious\x12/\n" +
	"\bprogress\x18\r \x01(\v2\x13.cozy.worker.v1.RefR\bprogress\x12:\n" +
	"\aobjects\x18\x0e \x03(\v2 .cozy.worker.v1.CheckpointObjectR\aobjects\x12\x1f\n" +
	"\vnext_offset\x18\x0f \x01(\rR\n" +
	"nextOffset\x12\x19\n" +
	"\bhas_more\x18\x10 \x01(\bR\ahasMore\x12\x1b\n" +
	"\tsafe_code\x18\x11 \x01(\tR\bsafeCode\x12\x1f\n" +
	"\vsafe_detail\x18\x12 \x01(\tR\n" +
	"safeDetailJ\x04\b\x04\x10\x05J\x04\b\x06\x10\aJ\x04\b\a\x10\b\"\xb6\x04\n" +
	"\x19CheckpointTransferRequest\x12,\n" +
	"\x12record_owner_epoch\x18\x01 \x01(\x04R\x10recordOwnerEpoch\x120\n" +
	"\x14control_stream_epoch\x18\x02 \x01(\x04R\x12controlStreamEpoch\x12$\n" +
	"\x0eworker_boot_id\x18\x03 \x01(\tR\fworkerBootId\x12;\n" +
	"\asubject\x18\x05 \x01(\v2!.cozy.worker.v1.CheckpointSubjectR\asubject\x12\x1f\n" +
	"\vplan_digest\x18\b \x01(\fR\n" +
	"planDigest\x12'\n" +
	"\x04head\x18\t \x01(\v2\x13.cozy.worker.v1.RefR\x04head\x128\n" +
	"\x06object\x18\n" +
	" \x01(\v2 .cozy.worker.v1.CheckpointObjectR\x06object\x12\x1f\n" +
	"\vtransfer_id\x18\v \x01(\tR\n" +
	"transferId\x12%\n" +
	"\x0egrant_revision\x18\f \x01(\x04R\rgrantRevision\x12G\n" +
	"\fupload_grant\x18\r \x01(\v2\".cozy.worker.v1.WeightsUploadGrantH\x00R\vuploadGrant\x12#\n" +
	"\fdownload_url\x18\x0e \x01(\tH\x00R\vdownloadUrlB\n" +
	"\n" +
	"\bdecisionJ\x04\b\x04\x10\x05J\x04\b\x06\x10\aJ\x04\b\a\x10\b\"\xac\x05\n" +
	"\x18CheckpointTransferStatus\x12,\n" +
	"\x12record_owner_epoch\x18\x01 \x01(\x04R\x10recordOwnerEpoch\x120\n" +
	"\x14control_stream_epoch\x18\x02 \x01(\x04R\x12controlStreamEpoch\x12$\n" +
	"\x0eworker_boot_id\x18\x03 \x01(\tR\fworkerBootId\x12;\n" +
	"\asubject\x18\x05 \x01(\v2!.cozy.worker.v1.CheckpointSubjectR\asubject\x12\x1f\n" +
	"\vplan_digest\x18\b \x01(\fR\n" +
	"planDigest\x12'\n" +
	"\x04head\x18\t \x01(\v2\x13.cozy.worker.v1.RefR\x04head\x128\n" +
	"\x06object\x18\n" +
	" \x01(\v2 .cozy.worker.v1.CheckpointObjectR\x06object\x12\x1f\n" +
	"\vtransfer_id\x18\v \x01(\tR\n" +
	"transferId\x12%\n" +
	"\x0egrant_revision\x18\f \x01(\x04R\rgrantRevision\x12:\n" +
	"\x05state\x18\r \x01(\x0e2$.cozy.worker.v1.WeightsTransferStateR\x05state\x12+\n" +
	"\x11transferred_bytes\x18\x0e \x01(\x04R\x10transferredBytes\x12\x1f\n" +
	"\vhttp_status\x18\x0f \x01(\rR\n" +
	"httpStatus\x12'\n" +
	"\x0fchecksum_sha256\x18\x10 \x01(\tR\x0echecksumSha256\x12\x1b\n" +
	"\tsafe_code\x18\x11 \x01(\tR\bsafeCode\x12\x1f\n" +
	"\vsafe_detail\x18\x12 \x01(\tR\n" +
	"safeDetailJ\x04\b\x04\x10\x05J\x04\b\x06\x10\aJ\x04\b\a\x10\b\"\xbc\x03\n" +
	"\x19PrepareModelSourceRequest\x12!\n" +
	"\foperation_id\x18\x01 \x01(\tR\voperationId\x126\n" +
	"\x17source_selection_digest\x18\x02 \x01(\fR\x15sourceSelectionDigest\x12>\n" +
	"\bprofiles\x18\x03 \x03(\v2\".cozy.worker.v1.ModelSourceProfileR\bprofiles\x12:\n" +
	"\x05files\x18\x04 \x03(\v2$.cozy.worker.v1.LocalModelSourceFileR\x05files\x12\x1d\n" +
	"\n" +
	"source_uri\x18\x05 \x01(\tR\tsourceUri\x12)\n" +
	"\x10declared_license\x18\x06 \x01(\tR\x0fdeclaredLicense\x12G\n" +
	"\vcheckpoints\x18\a \x03(\v2%.cozy.worker.v1.ModelSourceCheckpointR\vcheckpoints\x125\n" +
	"\x17adopt_from_operation_id\x18\b \x01(\tR\x14adoptFromOperationId\"\xca\x02\n" +
	"\x18PrepareModelSourceResult\x12C\n" +
	"\aoutcome\x18\x01 \x01(\x0e2).cozy.worker.v1.ModelSourcePrepareOutcomeR\aoutcome\x12=\n" +
	"\asources\x18\x02 \x03(\v2#.cozy.worker.v1.PreparedModelSourceR\asources\x12\x1b\n" +
	"\tsafe_code\x18\x03 \x01(\tR\bsafeCode\x12\x1f\n" +
	"\vsafe_detail\x18\x04 \x01(\tR\n" +
	"safeDetail\x12G\n" +
	"\vcheckpoints\x18\x05 \x03(\v2%.cozy.worker.v1.ModelSourceCheckpointR\vcheckpoints\x12#\n" +
	"\rspent_members\x18\x06 \x03(\tR\fspentMembers\"\x88\f\n" +
	"\x10RecordOwnerFrame\x12-\n" +
	"\x05claim\x18\x05 \x01(\v2\x15.cozy.worker.v1.ClaimH\x00R\x05claim\x12I\n" +
	"\rdesired_state\x18\x06 \x01(\v2\".cozy.worker.v1.DesiredWorkerStateH\x00R\fdesiredState\x12C\n" +
	"\rattempt_offer\x18\a \x01(\v2\x1c.cozy.worker.v1.AttemptOfferH\x00R\fattemptOffer\x12F\n" +
	"\x0ecancel_attempt\x18\b \x01(\v2\x1d.cozy.worker.v1.CancelAttemptH\x00R\rcancelAttempt\x12D\n" +
	"\voutcome_ack\x18\t \x01(\v2!.cozy.worker.v1.AttemptOutcomeAckH\x00R\n" +
	"outcomeAck\x12U\n" +
	"\x12checkpoint_receipt\x18\n" +
	" \x01(\v2$.cozy.worker.v1.JobCheckpointReceiptH\x00R\x11checkpointReceipt\x12@\n" +
	"\fsnapshot_ack\x18\v \x01(\v2\x1b.cozy.worker.v1.SnapshotAckH\x00R\vsnapshotAck\x12b\n" +
	"\x18weights_finalize_request\x18\x11 \x01(\v2&.cozy.worker.v1.WeightsFinalizeRequestH\x00R\x16weightsFinalizeRequest\x12J\n" +
	"\x10weights_host_ack\x18\x13 \x01(\v2\x1e.cozy.worker.v1.WeightsHostAckH\x00R\x0eweightsHostAck\x12b\n" +
	"\x18weights_transfer_request\x18\x14 \x01(\v2&.cozy.worker.v1.WeightsTransferRequestH\x00R\x16weightsTransferRequest\x12V\n" +
	"\x14weights_read_request\x18\x15 \x01(\v2\".cozy.worker.v1.WeightsReadRequestH\x00R\x12weightsReadRequest\x12c\n" +
	"\x19model_source_file_request\x18\x16 \x01(\v2&.cozy.worker.v1.ModelSourceFileRequestH\x00R\x16modelSourceFileRequest\x12l\n" +
	"\x1cmodel_source_prepare_request\x18\x17 \x01(\v2).cozy.worker.v1.ModelSourcePrepareRequestH\x00R\x19modelSourcePrepareRequest\x12S\n" +
	"\x13local_package_abort\x18\x19 \x01(\v2!.cozy.worker.v1.LocalPackageAbortH\x00R\x11localPackageAbort\x12\\\n" +
	"\x16weights_upload_request\x18\x1a \x01(\v2$.cozy.worker.v1.WeightsUploadRequestH\x00R\x14weightsUploadRequest\x12i\n" +
	"\x1blocal_package_fetch_request\x18\x1b \x01(\v2(.cozy.worker.v1.LocalPackageFetchRequestH\x00R\x18localPackageFetchRequest\x12M\n" +
	"\x11child_call_result\x18\x1c \x01(\v2\x1f.cozy.worker.v1.ChildCallResultH\x00R\x0fchildCallResultB\x05\n" +
	"\x03msgJ\x04\b\x0e\x10\x0fJ\x04\b\x10\x10\x11J\x04\b\x12\x10\x13J\x04\b\x18\x10\x19R\x15artifact_grant_updateR\x10ensure_artifactsR\x1aprivate_package_file_chunk\"\xd8\x0e\n" +
	"\vWorkerFrame\x127\n" +
	"\tclaim_ack\x18\x05 \x01(\v2\x18.cozy.worker.v1.ClaimAckH\x00R\bclaimAck\x12L\n" +
	"\x0eobserved_state\x18\x06 \x01(\v2#.cozy.worker.v1.ObservedWorkerStateH\x00R\robservedState\x12L\n" +
	"\x10attempt_accepted\x18\a \x01(\v2\x1f.cozy.worker.v1.AttemptAcceptedH\x00R\x0fattemptAccepted\x12I\n" +
	"\x0fattempt_outcome\x18\b \x01(\v2\x1e.cozy.worker.v1.AttemptOutcomeH\x00R\x0eattemptOutcome\x12@\n" +
	"\fboot_failure\x18\t \x01(\v2\x1b.cozy.worker.v1.BootFailureH\x00R\vbootFailure\x12U\n" +
	"\x12checkpoint_request\x18\n" +
	" \x01(\v2$.cozy.worker.v1.JobCheckpointRequestH\x00R\x11checkpointRequest\x12I\n" +
	"\x0echeckpoint_ack\x18\v \x01(\v2 .cozy.worker.v1.JobCheckpointAckH\x00R\rcheckpointAck\x12<\n" +
	"\bsnapshot\x18\f \x01(\v2\x1e.cozy.worker.v1.WorkerSnapshotH\x00R\bsnapshot\x12_\n" +
	"\x17weights_finalize_result\x18\x10 \x01(\v2%.cozy.worker.v1.WeightsFinalizeResultH\x00R\x15weightsFinalizeResult\x12K\n" +
	"\x0eweights_intent\x18\x11 \x01(\v2\".cozy.worker.v1.WeightsIntentFrameH\x00R\rweightsIntent\x12N\n" +
	"\x0fweights_receipt\x18\x12 \x01(\v2#.cozy.worker.v1.WeightsReceiptFrameH\x00R\x0eweightsReceipt\x12S\n" +
	"\x13weights_read_result\x18\x13 \x01(\v2!.cozy.worker.v1.WeightsReadResultH\x00R\x11weightsReadResult\x12_\n" +
	"\x17weights_transfer_status\x18\x14 \x01(\v2%.cozy.worker.v1.WeightsTransferStatusH\x00R\x15weightsTransferStatus\x12`\n" +
	"\x18model_source_file_status\x18\x15 \x01(\v2%.cozy.worker.v1.ModelSourceFileStatusH\x00R\x15modelSourceFileStatus\x12Y\n" +
	"\x15model_source_prepared\x18\x16 \x01(\v2#.cozy.worker.v1.ModelSourcePreparedH\x00R\x13modelSourcePrepared\x12c\n" +
	"\x19local_package_file_status\x18\x17 \x01(\v2&.cozy.worker.v1.LocalPackageFileStatusH\x00R\x16localPackageFileStatus\x12f\n" +
	"\x1alocal_package_abort_status\x18\x18 \x01(\v2'.cozy.worker.v1.LocalPackageAbortStatusH\x00R\x17localPackageAbortStatus\x12Y\n" +
	"\x15weights_upload_result\x18\x19 \x01(\v2#.cozy.worker.v1.WeightsUploadResultH\x00R\x13weightsUploadResult\x12W\n" +
	"\x12weights_checkpoint\x18\x1a \x01(\v2&.cozy.worker.v1.WeightsCheckpointFrameH\x00R\x11weightsCheckpoint\x12[\n" +
	"\x13weights_transaction\x18\x1b \x01(\v2(.cozy.worker.v1.WeightsTransactionStatusH\x00R\x12weightsTransaction\x12P\n" +
	"\x12child_call_request\x18\x1c \x01(\v2 .cozy.worker.v1.ChildCallRequestH\x00R\x10childCallRequest\x12M\n" +
	"\x11child_call_cancel\x18\x1d \x01(\v2\x1f.cozy.worker.v1.ChildCallCancelH\x00R\x0fchildCallCancelB\x05\n" +
	"\x03msgJ\x04\b\r\x10\x0eJ\x04\b\x0e\x10\x0fJ\x04\b\x0f\x10\x10\"\x94\x04\n" +
	"\x10ChildCallRequest\x12,\n" +
	"\x12record_owner_epoch\x18\x01 \x01(\x04R\x10recordOwnerEpoch\x120\n" +
	"\x14control_stream_epoch\x18\x02 \x01(\x04R\x12controlStreamEpoch\x12$\n" +
	"\x0eworker_boot_id\x18\x03 \x01(\tR\fworkerBootId\x12*\n" +
	"\x11parent_request_id\x18\x04 \x01(\tR\x0fparentRequestId\x124\n" +
	"\x16parent_attempt_ordinal\x18\x05 \x01(\x04R\x14parentAttemptOrdinal\x12A\n" +
	"\x1dparent_invocation_spec_digest\x18\x06 \x01(\fR\x1aparentInvocationSpecDigest\x12\x1d\n" +
	"\n" +
	"call_index\x18\a \x01(\rR\tcallIndex\x12)\n" +
	"\x10interface_digest\x18\b \x01(\fR\x0finterfaceDigest\x12\x16\n" +
	"\x06module\x18\t \x01(\tR\x06module\x12\x16\n" +
	"\x06export\x18\n" +
	" \x01(\tR\x06export\x126\n" +
	"\x17request_canonical_bytes\x18\v \x01(\fR\x15requestCanonicalBytes\x12#\n" +
	"\rintent_digest\x18\f \x01(\fR\fintentDigest\"\x80\x03\n" +
	"\x0fChildCallCancel\x12,\n" +
	"\x12record_owner_epoch\x18\x01 \x01(\x04R\x10recordOwnerEpoch\x120\n" +
	"\x14control_stream_epoch\x18\x02 \x01(\x04R\x12controlStreamEpoch\x12$\n" +
	"\x0eworker_boot_id\x18\x03 \x01(\tR\fworkerBootId\x12*\n" +
	"\x11parent_request_id\x18\x04 \x01(\tR\x0fparentRequestId\x124\n" +
	"\x16parent_attempt_ordinal\x18\x05 \x01(\x04R\x14parentAttemptOrdinal\x12A\n" +
	"\x1dparent_invocation_spec_digest\x18\x06 \x01(\fR\x1aparentInvocationSpecDigest\x12\x1d\n" +
	"\n" +
	"call_index\x18\a \x01(\rR\tcallIndex\x12#\n" +
	"\rintent_digest\x18\b \x01(\fR\fintentDigest\"\xd4\x04\n" +
	"\x0fChildCallResult\x12,\n" +
	"\x12record_owner_epoch\x18\x01 \x01(\x04R\x10recordOwnerEpoch\x120\n" +
	"\x14control_stream_epoch\x18\x02 \x01(\x04R\x12controlStreamEpoch\x12$\n" +
	"\x0eworker_boot_id\x18\x03 \x01(\tR\fworkerBootId\x12*\n" +
	"\x11parent_request_id\x18\x04 \x01(\tR\x0fparentRequestId\x124\n" +
	"\x16parent_attempt_ordinal\x18\x05 \x01(\x04R\x14parentAttemptOrdinal\x12A\n" +
	"\x1dparent_invocation_spec_digest\x18\x06 \x01(\fR\x1aparentInvocationSpecDigest\x12\x1d\n" +
	"\n" +
	"call_index\x18\a \x01(\rR\tcallIndex\x12#\n" +
	"\rintent_digest\x18\b \x01(\fR\fintentDigest\x12(\n" +
	"\x10child_request_id\x18\t \x01(\tR\x0echildRequestId\x124\n" +
	"\x05state\x18\n" +
	" \x01(\x0e2\x1e.cozy.worker.v1.ChildCallStateR\x05state\x124\n" +
	"\x16result_canonical_bytes\x18\v \x01(\fR\x14resultCanonicalBytes\x12\x1b\n" +
	"\tsafe_code\x18\f \x01(\tR\bsafeCode\x12\x1f\n" +
	"\vsafe_detail\x18\r \x01(\tR\n" +
	"safeDetail\"\xa7\x02\n" +
	"\x05Claim\x12,\n" +
	"\x12record_owner_epoch\x18\x01 \x01(\x04R\x10recordOwnerEpoch\x120\n" +
	"\x14control_stream_epoch\x18\x02 \x01(\x04R\x12controlStreamEpoch\x12$\n" +
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
	"\x1dworker_tls_certificate_digest\x18\x04 \x01(\fR\x1aworkerTlsCertificateDigest\"\xb4\x04\n" +
	"\bClaimAck\x12,\n" +
	"\x12record_owner_epoch\x18\x01 \x01(\x04R\x10recordOwnerEpoch\x120\n" +
	"\x14control_stream_epoch\x18\x02 \x01(\x04R\x12controlStreamEpoch\x12$\n" +
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
	"\tresources\x18\r \x01(\v2\x1f.cozy.worker.v1.WorkerResourcesR\tresourcesJ\x04\b\x04\x10\x05J\x04\b\x0e\x10\x0fR\x12wire_schema_digest\"\xd8\x03\n" +
	"\vBootFailure\x12,\n" +
	"\x12record_owner_epoch\x18\x01 \x01(\x04R\x10recordOwnerEpoch\x120\n" +
	"\x14control_stream_epoch\x18\x02 \x01(\x04R\x12controlStreamEpoch\x12$\n" +
	"\x0eworker_boot_id\x18\x03 \x01(\tR\fworkerBootId\x12\x1b\n" +
	"\tworker_id\x18\x05 \x01(\tR\bworkerId\x12,\n" +
	"\x12worker_instance_id\x18\x06 \x01(\tR\x10workerInstanceId\x129\n" +
	"\x06reason\x18\a \x01(\x0e2!.cozy.worker.v1.BootFailureReasonR\x06reason\x12\x16\n" +
	"\x06detail\x18\b \x01(\tR\x06detail\x12=\n" +
	"\tresources\x18\t \x01(\v2\x1f.cozy.worker.v1.WorkerResourcesR\tresources\x124\n" +
	"\x16control_runtime_digest\x18\n" +
	" \x01(\tR\x14controlRuntimeDigest\x12*\n" +
	"\x11worker_release_id\x18\v \x01(\tR\x0fworkerReleaseIdJ\x04\b\x04\x10\x05\"\xe9\x03\n" +
	"\x0eWorkerSnapshot\x12,\n" +
	"\x12record_owner_epoch\x18\x01 \x01(\x04R\x10recordOwnerEpoch\x120\n" +
	"\x14control_stream_epoch\x18\x02 \x01(\x04R\x12controlStreamEpoch\x12$\n" +
	"\x0eworker_boot_id\x18\x03 \x01(\tR\fworkerBootId\x12\x1f\n" +
	"\vsnapshot_id\x18\x05 \x01(\tR\n" +
	"snapshotId\x12'\n" +
	"\x0fsnapshot_digest\x18\x06 \x01(\fR\x0esnapshotDigest\x128\n" +
	"\x18snapshot_canonical_bytes\x18\a \x01(\fR\x16snapshotCanonicalBytes\x12R\n" +
	"&accepted_placement_set_canonical_bytes\x18\b \x01(\fR\"acceptedPlacementSetCanonicalBytes\x120\n" +
	"\x14host_snapshot_digest\x18\t \x01(\fR\x12hostSnapshotDigest\x12A\n" +
	"\x1dhost_snapshot_canonical_bytes\x18\n" +
	" \x01(\fR\x1ahostSnapshotCanonicalBytesJ\x04\b\x04\x10\x05\"\x9d\x06\n" +
	"\x12WorkerSnapshotBody\x12E\n" +
	"\x1faccepted_desired_state_revision\x18\x01 \x01(\x04R\x1cacceptedDesiredStateRevision\x12A\n" +
	"\x1daccepted_placement_set_digest\x18\x02 \x01(\fR\x1aacceptedPlacementSetDigest\x12+\n" +
	"\x11journal_highwater\x18\x03 \x01(\x04R\x10journalHighwater\x12>\n" +
	"\fworker_phase\x18\x04 \x01(\x0e2\x1b.cozy.worker.v1.WorkerPhaseR\vworkerPhase\x12?\n" +
	"\n" +
	"placements\x18\x05 \x03(\v2\x1f.cozy.worker.v1.PlacementStatusR\n" +
	"placements\x12-\n" +
	"\x12converged_revision\x18\x06 \x01(\x04R\x11convergedRevision\x12'\n" +
	"\x0fadmission_epoch\x18\a \x01(\x04R\x0eadmissionEpoch\x12G\n" +
	"\x0fadmission_state\x18\b \x01(\x0e2\x1e.cozy.worker.v1.AdmissionStateR\x0eadmissionState\x126\n" +
	"\x17available_attempt_slots\x18\t \x01(\rR\x15availableAttemptSlots\x12@\n" +
	"\rheld_attempts\x18\n" +
	" \x03(\v2\x1b.cozy.worker.v1.HeldAttemptR\fheldAttempts\x12[\n" +
	"\x14weights_transactions\x18\v \x03(\v2(.cozy.worker.v1.WeightsTransactionStatusR\x13weightsTransactions\x120\n" +
	"\x05lanes\x18\f \x03(\v2\x1a.cozy.worker.v1.DeviceLaneR\x05lanes\x12%\n" +
	"\x0eheld_manifests\x18\r \x03(\tR\rheldManifests\"\xed\x01\n" +
	"\x10HostSnapshotBody\x12@\n" +
	"\rheld_outcomes\x18\x01 \x03(\v2\x1b.cozy.worker.v1.HeldAttemptR\fheldOutcomes\x12[\n" +
	"\x14weights_transactions\x18\x02 \x03(\v2(.cozy.worker.v1.WeightsTransactionStatusR\x13weightsTransactions\x12:\n" +
	"\x19retained_desired_revision\x18\x03 \x01(\x04R\x17retainedDesiredRevision\"\x95\x02\n" +
	"\vSnapshotAck\x12,\n" +
	"\x12record_owner_epoch\x18\x01 \x01(\x04R\x10recordOwnerEpoch\x120\n" +
	"\x14control_stream_epoch\x18\x02 \x01(\x04R\x12controlStreamEpoch\x12$\n" +
	"\x0eworker_boot_id\x18\x03 \x01(\tR\fworkerBootId\x12\x1f\n" +
	"\vsnapshot_id\x18\x05 \x01(\tR\n" +
	"snapshotId\x12'\n" +
	"\x0fsnapshot_digest\x18\x06 \x01(\fR\x0esnapshotDigest\x120\n" +
	"\x14host_snapshot_digest\x18\a \x01(\fR\x12hostSnapshotDigestJ\x04\b\x04\x10\x05\"\xd0\x05\n" +
	"\x12DesiredWorkerState\x12,\n" +
	"\x12record_owner_epoch\x18\x01 \x01(\x04R\x10recordOwnerEpoch\x120\n" +
	"\x14control_stream_epoch\x18\x02 \x01(\x04R\x12controlStreamEpoch\x12$\n" +
	"\x0eworker_boot_id\x18\x03 \x01(\tR\fworkerBootId\x12\x1a\n" +
	"\brevision\x18\x05 \x01(\x04R\brevision\x121\n" +
	"\aposture\x18\x06 \x01(\x0e2\x17.cozy.worker.v1.PostureR\aposture\x12\x1d\n" +
	"\n" +
	"wire_minor\x18\a \x01(\rR\twireMinor\x12$\n" +
	"\x0edrain_grace_ms\x18\b \x01(\x04R\fdrainGraceMs\x120\n" +
	"\x03job\x18\r \x01(\v2\x1c.cozy.worker.v1.JobDirectiveH\x00R\x03job\x12J\n" +
	"\rplacement_set\x18\x0f \x01(\v2#.cozy.worker.v1.DesiredPlacementSetH\x00R\fplacementSet\x12D\n" +
	"\vpackage_set\x18\x10 \x01(\v2!.cozy.worker.v1.DesiredPackageSetH\x00R\n" +
	"packageSet\x12T\n" +
	"\x11local_package_set\x18\x11 \x01(\v2&.cozy.worker.v1.DesiredLocalPackageSetH\x00R\x0flocalPackageSet\x12`\n" +
	"\x15private_placement_set\x18\x12 \x01(\v2*.cozy.worker.v1.DesiredPrivatePlacementSetH\x00R\x13privatePlacementSetB\x06\n" +
	"\x04modeJ\x04\b\x04\x10\x05J\x04\b\t\x10\n" +
	"J\x04\b\n" +
	"\x10\vJ\x04\b\v\x10\fJ\x04\b\f\x10\r\"\x88\x01\n" +
	"\x11DesiredPackageSet\x12/\n" +
	"\x13download_delegation\x18\x01 \x01(\fR\x12downloadDelegation\x12B\n" +
	"\x1ddownload_delegation_signature\x18\x02 \x01(\fR\x1bdownloadDelegationSignature\"\xb4\x01\n" +
	"\x16DesiredLocalPackageSet\x12!\n" +
	"\foperation_id\x18\x01 \x01(\tR\voperationId\x12<\n" +
	"\apackage\x18\x02 \x01(\v2\".cozy.worker.v1.DevelopmentPackageR\apackage\x129\n" +
	"\x05files\x18\x03 \x03(\v2#.cozy.worker.v1.LocalPackageFileRefR\x05files\"\xe8\x01\n" +
	"\x1aDesiredPrivatePlacementSet\x12!\n" +
	"\foperation_id\x18\x01 \x01(\tR\voperationId\x122\n" +
	"\x15local_revision_digest\x18\x02 \x01(\fR\x13localRevisionDigest\x12/\n" +
	"\x13download_delegation\x18\x03 \x01(\fR\x12downloadDelegation\x12B\n" +
	"\x1ddownload_delegation_signature\x18\x04 \x01(\fR\x1bdownloadDelegationSignature\"m\n" +
	"\x13LocalPackageFileRef\x12\x16\n" +
	"\x06digest\x18\x01 \x01(\fR\x06digest\x12\x1a\n" +
	"\bfilename\x18\x02 \x01(\tR\bfilename\x12\x16\n" +
	"\x06length\x18\x04 \x01(\x04R\x06lengthJ\x04\b\x03\x10\x04R\x04kind\"\xec\x01\n" +
	"\x14LocalPackageRevision\x12\x18\n" +
	"\apackage\x18\x01 \x01(\tR\apackage\x12\x18\n" +
	"\arelease\x18\x02 \x01(\tR\arelease\x12#\n" +
	"\rsource_digest\x18\x03 \x01(\fR\fsourceDigest\x12@\n" +
	"\x11package_interface\x18\x04 \x01(\v2\x13.cozy.worker.v1.RefR\x10packageInterface\x129\n" +
	"\x05files\x18\x05 \x03(\v2#.cozy.worker.v1.LocalPackageFileRefR\x05files\"\xd5\x01\n" +
	"\x13DesiredPlacementSet\x120\n" +
	"\x14placement_set_digest\x18\x01 \x01(\fR\x12placementSetDigest\x12A\n" +
	"\x1dplacement_set_canonical_bytes\x18\x03 \x01(\fR\x1aplacementSetCanonicalBytes\x12C\n" +
	"\vdevice_pins\x18\x04 \x03(\v2\".cozy.worker.v1.PlacementDevicePinR\n" +
	"devicePinsJ\x04\b\x02\x10\x03\"`\n" +
	"\x12PlacementDevicePin\x12!\n" +
	"\fplacement_id\x18\x01 \x01(\tR\vplacementId\x12'\n" +
	"\x0fdevice_ordinals\x18\x02 \x03(\rR\x0edeviceOrdinals\"I\n" +
	"\fPlacementSet\x129\n" +
	"\n" +
	"placements\x18\x01 \x03(\v2\x19.cozy.worker.v1.PlacementR\n" +
	"placements\"\xef\x02\n" +
	"\x12DownloadDelegation\x12&\n" +
	"\x0fexpires_at_unix\x18\x01 \x01(\x04R\rexpiresAtUnix\x128\n" +
	"\x06models\x18\x02 \x03(\v2 .cozy.worker.v1.DownloadModelRefR\x06models\x12>\n" +
	"\bpackages\x18\x03 \x03(\v2\".cozy.worker.v1.DownloadPackageRefR\bpackages\x12\x1b\n" +
	"\trental_id\x18\x04 \x01(\tR\brentalId\x12$\n" +
	"\x0eworker_boot_id\x18\x05 \x01(\tR\fworkerBootId\x12\x1b\n" +
	"\tworker_id\x18\x06 \x01(\tR\bworkerId\x12A\n" +
	"\x1dworker_tls_certificate_digest\x18\a \x01(\fR\x1aworkerTlsCertificateDigestJ\x04\b\b\x10\tR\x0eheld_manifests\"\xa0\x01\n" +
	"\x10DownloadModelRef\x12\x1a\n" +
	"\bmanifest\x18\x01 \x01(\tR\bmanifest\x12\x14\n" +
	"\x05model\x18\x02 \x01(\tR\x05model\x12\x18\n" +
	"\arelease\x18\x03 \x01(\tR\arelease\x12\x18\n" +
	"\apackage\x18\x04 \x01(\tR\apackage\x12\x12\n" +
	"\x04slot\x18\x05 \x01(\tR\x04slot\x12\x12\n" +
	"\x04lane\x18\x06 \x01(\tR\x04lane\"^\n" +
	"\x12DownloadPackageRef\x12\x18\n" +
	"\apackage\x18\x01 \x01(\tR\apackage\x12\x18\n" +
	"\arelease\x18\x02 \x01(\tR\areleaseJ\x04\b\x03\x10\x04R\x0erelease_digest\"\xc1\x04\n" +
	"\tPlacement\x12!\n" +
	"\fplacement_id\x18\x01 \x01(\tR\vplacementId\x12<\n" +
	"\apackage\x18\x02 \x01(\v2 .cozy.worker.v1.PackageSelectionH\x00R\apackage\x12F\n" +
	"\vdevelopment\x18\v \x01(\v2\".cozy.worker.v1.DevelopmentPackageH\x00R\vdevelopment\x12-\n" +
	"\x12environment_digest\x18\x03 \x01(\fR\x11environmentDigest\x12@\n" +
	"\x11package_interface\x18\x05 \x01(\v2\x13.cozy.worker.v1.RefR\x10packageInterface\x12'\n" +
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
	"\x04tags\x18\x06 \x03(\tR\x04tags\"q\n" +
	"\x10PackageSelection\x12\x18\n" +
	"\apackage\x18\x01 \x01(\tR\apackage\x12\x18\n" +
	"\arelease\x18\x02 \x01(\tR\areleaseJ\x04\b\x03\x10\x04J\x04\b\x04\x10\x05R\x0erelease_digestR\rproject_wheel\"\xe1\x01\n" +
	"\x12DevelopmentPackage\x12\x18\n" +
	"\apackage\x18\x01 \x01(\tR\apackage\x12\x18\n" +
	"\arelease\x18\x02 \x01(\tR\arelease\x12#\n" +
	"\rsource_digest\x18\x03 \x01(\fR\fsourceDigest\x12>\n" +
	"\rproject_wheel\x18\x04 \x01(\v2\x19.cozy.worker.v1.WheelFactR\fprojectWheel\x122\n" +
	"\x15local_revision_digest\x18\x05 \x01(\fR\x13localRevisionDigest\"\xba\x01\n" +
	"\vEnvironment\x12D\n" +
	"\x13locked_requirements\x18\x03 \x01(\v2\x13.cozy.worker.v1.RefR\x12lockedRequirements\x12<\n" +
	"\flocal_wheels\x18\x04 \x03(\v2\x19.cozy.worker.v1.WheelFactR\vlocalWheelsJ\x04\b\x01\x10\x02J\x04\b\x02\x10\x03R\x13wheelhouse_manifestR\x06wheels\"\x8a\x01\n" +
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
	"\x05slots\x18\x03 \x03(\v2\x14.cozy.worker.v1.SlotR\x05slots\"\x95\x02\n" +
	"\x04Slot\x12\x12\n" +
	"\x04slot\x18\x01 \x01(\tR\x04slot\x12,\n" +
	"\x12reference_model_id\x18\x02 \x01(\tR\x10referenceModelId\x129\n" +
	"\n" +
	"components\x18\x04 \x03(\v2\x19.cozy.worker.v1.ComponentR\n" +
	"components\x12S\n" +
	"\x1bmodel_construction_contract\x18\x05 \x01(\v2\x13.cozy.worker.v1.RefR\x19modelConstructionContract\x12-\n" +
	"\x06stamps\x18\x06 \x03(\v2\x15.cozy.worker.v1.StampR\x06stampsJ\x04\b\x03\x10\x04R\x06config\"D\n" +
	"\tComponent\x12\x1c\n" +
	"\tcomponent\x18\x01 \x01(\tR\tcomponent\x12\x19\n" +
	"\bmodel_id\x18\x02 \x01(\tR\amodelId\"O\n" +
	"\x05Stamp\x12\x1c\n" +
	"\tcomponent\x18\x01 \x01(\tR\tcomponent\x12\x10\n" +
	"\x03key\x18\x02 \x01(\tR\x03key\x12\x16\n" +
	"\x06values\x18\x03 \x03(\tR\x06values\"\xba\x03\n" +
	"\fJobDirective\x12\x19\n" +
	"\bbuild_id\x18\x01 \x01(\tR\abuildId\x12*\n" +
	"\x11job_descriptor_id\x18\x02 \x01(\tR\x0fjobDescriptorId\x12A\n" +
	"\rresource_caps\x18\x03 \x01(\v2\x1c.cozy.worker.v1.ResourceCapsR\fresourceCaps\x12V\n" +
	"\x14publication_contract\x18\x04 \x01(\v2#.cozy.worker.v1.PublicationContractR\x13publicationContract\x12.\n" +
	"\x13reclaim_on_terminal\x18\x05 \x01(\bR\x11reclaimOnTerminal\x12!\n" +
	"\fdevice_count\x18\x06 \x01(\rR\vdeviceCount\x12$\n" +
	"\rorchestration\x18\a \x01(\bR\rorchestration\x12O\n" +
	"\x14orchestration_parent\x18\b \x01(\v2\x1c.cozy.worker.v1.JobDirectiveR\x13orchestrationParent\"\xec\b\n" +
	"\x13ObservedWorkerState\x12,\n" +
	"\x12record_owner_epoch\x18\x01 \x01(\x04R\x10recordOwnerEpoch\x120\n" +
	"\x14control_stream_epoch\x18\x02 \x01(\x04R\x12controlStreamEpoch\x12$\n" +
	"\x0eworker_boot_id\x18\x03 \x01(\tR\fworkerBootId\x12,\n" +
	"\x12applied_wire_minor\x18\b \x01(\rR\x10appliedWireMinor\x12@\n" +
	"\rheld_attempts\x18\t \x03(\v2\x1b.cozy.worker.v1.HeldAttemptR\fheldAttempts\x12-\n" +
	"\x06faults\x18\n" +
	" \x03(\v2\x15.cozy.worker.v1.FaultR\x06faults\x129\n" +
	"\bactivity\x18\v \x03(\v2\x1d.cozy.worker.v1.ActivityEventR\bactivity\x12>\n" +
	"\fjob_capacity\x18\r \x01(\v2\x1b.cozy.worker.v1.JobCapacityR\vjobCapacity\x12?\n" +
	"\n" +
	"placements\x18\x0f \x03(\v2\x1f.cozy.worker.v1.PlacementStatusR\n" +
	"placements\x12'\n" +
	"\x0fadmission_epoch\x18\x10 \x01(\x04R\x0eadmissionEpoch\x12G\n" +
	"\x0fadmission_state\x18\x11 \x01(\x0e2\x1e.cozy.worker.v1.AdmissionStateR\x0eadmissionState\x126\n" +
	"\x17available_attempt_slots\x18\x12 \x01(\rR\x15availableAttemptSlots\x12E\n" +
	"\x1faccepted_desired_state_revision\x18\x13 \x01(\x04R\x1cacceptedDesiredStateRevision\x12A\n" +
	"\x1daccepted_placement_set_digest\x18\x14 \x01(\fR\x1aacceptedPlacementSetDigest\x12-\n" +
	"\x12converged_revision\x18\x15 \x01(\x04R\x11convergedRevision\x12>\n" +
	"\fworker_phase\x18\x16 \x01(\x0e2\x1b.cozy.worker.v1.WorkerPhaseR\vworkerPhase\x120\n" +
	"\x05lanes\x18\x1a \x03(\v2\x1a.cozy.worker.v1.DeviceLaneR\x05lanes\x12%\n" +
	"\x0eheld_manifests\x18\x1b \x03(\tR\rheldManifestsJ\x04\b\x04\x10\x05J\x04\b\x05\x10\x06J\x04\b\x06\x10\aJ\x04\b\a\x10\bJ\x04\b\f\x10\rJ\x04\b\x0e\x10\x0fJ\x04\b\x17\x10\x18J\x04\b\x18\x10\x19J\x04\b\x19\x10\x1aR\x19applied_artifact_grant_idR\x16applied_grant_revisionR\x0fartifact_intent\"\xe1\x01\n" +
	"\n" +
	"DeviceLane\x12\x17\n" +
	"\alane_id\x18\x01 \x01(\tR\x06laneId\x12'\n" +
	"\x0fdevice_ordinals\x18\x02 \x03(\rR\x0edeviceOrdinals\x126\n" +
	"\x17available_attempt_slots\x18\x03 \x01(\rR\x15availableAttemptSlots\x12#\n" +
	"\rplacement_ids\x18\x04 \x03(\tR\fplacementIds\x124\n" +
	"\x16resident_placement_ids\x18\x05 \x03(\tR\x14residentPlacementIds\"\xda\x06\n" +
	"\x0fPlacementStatus\x12!\n" +
	"\fplacement_id\x18\x01 \x01(\tR\vplacementId\x12%\n" +
	"\x0eexecutor_epoch\x18\x04 \x01(\x04R\rexecutorEpoch\x12@\n" +
	"\x1cdispatchable_binding_digests\x18\x06 \x03(\fR\x1adispatchableBindingDigests\x12D\n" +
	"\x1ematerializable_binding_digests\x18\a \x03(\fR\x1cmaterializableBindingDigests\x12-\n" +
	"\x06faults\x18\b \x03(\v2\x15.cozy.worker.v1.FaultR\x06faults\x12J\n" +
	"\vaccelerator\x18\t \x01(\v2(.cozy.worker.v1.AcceleratorQualificationR\vaccelerator\x120\n" +
	"\x14placement_set_digest\x18\v \x01(\fR\x12placementSetDigest\x12N\n" +
	"\x0fmaterialization\x18\f \x01(\x0e2$.cozy.worker.v1.MaterializationStateR\x0fmaterialization\x126\n" +
	"\aserving\x18\r \x01(\x0e2\x1c.cozy.worker.v1.ServingStateR\aserving\x12R\n" +
	"&retained_fallback_placement_set_digest\x18\x0e \x01(\fR\"retainedFallbackPlacementSetDigest\x12Q\n" +
	"\vacquisition\x18\x0f \x01(\v2/.cozy.worker.v1.PlacementAcquisitionObservationR\vacquisition\x12-\n" +
	"\x12environment_digest\x18\x11 \x01(\tR\x11environmentDigest\x12$\n" +
	"\x0edevice_lane_id\x18\x13 \x01(\tR\fdeviceLaneIdJ\x04\b\x02\x10\x03J\x04\b\x03\x10\x04J\x04\b\x05\x10\x06J\x04\b\x10\x10\x11J\x04\b\x12\x10\x13R\x17package_revision_digestR\rconfig_digest\"\xa7\x01\n" +
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
	"at_unix_ms\x18\x04 \x01(\x04R\batUnixMs\"\xe0\x03\n" +
	"\fAttemptOffer\x12,\n" +
	"\x12record_owner_epoch\x18\x01 \x01(\x04R\x10recordOwnerEpoch\x120\n" +
	"\x14control_stream_epoch\x18\x02 \x01(\x04R\x12controlStreamEpoch\x12$\n" +
	"\x0eworker_boot_id\x18\x03 \x01(\tR\fworkerBootId\x12\x1d\n" +
	"\n" +
	"request_id\x18\x05 \x01(\tR\trequestId\x12'\n" +
	"\x0fattempt_ordinal\x18\x06 \x01(\x04R\x0eattemptOrdinal\x124\n" +
	"\x16invocation_spec_digest\x18\a \x01(\fR\x14invocationSpecDigest\x123\n" +
	"\x05grant\x18\b \x01(\v2\x1d.cozy.worker.v1.DeliveryGrantR\x05grant\x12E\n" +
	"\x1finvocation_spec_canonical_bytes\x18\t \x01(\fR\x1cinvocationSpecCanonicalBytes\x12!\n" +
	"\fplacement_id\x18\n" +
	" \x01(\tR\vplacementId\x12'\n" +
	"\x0fadmission_epoch\x18\f \x01(\x04R\x0eadmissionEpochJ\x04\b\x04\x10\x05\"\xad\x03\n" +
	"\x0fAttemptAccepted\x12,\n" +
	"\x12record_owner_epoch\x18\x01 \x01(\x04R\x10recordOwnerEpoch\x120\n" +
	"\x14control_stream_epoch\x18\x02 \x01(\x04R\x12controlStreamEpoch\x12$\n" +
	"\x0eworker_boot_id\x18\x03 \x01(\tR\fworkerBootId\x12\x1d\n" +
	"\n" +
	"request_id\x18\x05 \x01(\tR\trequestId\x12'\n" +
	"\x0fattempt_ordinal\x18\x06 \x01(\x04R\x0eattemptOrdinal\x124\n" +
	"\x16invocation_spec_digest\x18\a \x01(\fR\x14invocationSpecDigest\x12!\n" +
	"\fplacement_id\x18\v \x01(\tR\vplacementId\x12\x17\n" +
	"\alane_id\x18\r \x01(\tR\x06laneIdJ\x04\b\x04\x10\x05J\x04\b\b\x10\tJ\x04\b\t\x10\n" +
	"J\x04\b\n" +
	"\x10\vJ\x04\b\f\x10\rR\vplan_digestR\x19model_construction_digestR\x04planR\x0eexecutor_epoch\"\xc1\x02\n" +
	"\x12AttemptPlanSummary\x12\x1a\n" +
	"\bdelivery\x18\x01 \x01(\tR\bdelivery\x12(\n" +
	"\x0fmaterialization\x18\x02 \x01(\tR\x0fmaterialization\x12#\n" +
	"\rcompute_dtype\x18\x03 \x01(\tR\fcomputeDtype\x12\x1c\n" +
	"\tplacement\x18\x04 \x01(\tR\tplacement\x12?\n" +
	"\x1creserved_device_memory_bytes\x18\x05 \x01(\x04R\x19reservedDeviceMemoryBytes\x12.\n" +
	"\x13reserved_host_bytes\x18\x06 \x01(\x04R\x11reservedHostBytes\x121\n" +
	"\x14decision_explanation\x18\a \x01(\tR\x13decisionExplanation\"\xea\x02\n" +
	"\rCancelAttempt\x12,\n" +
	"\x12record_owner_epoch\x18\x01 \x01(\x04R\x10recordOwnerEpoch\x120\n" +
	"\x14control_stream_epoch\x18\x02 \x01(\x04R\x12controlStreamEpoch\x12$\n" +
	"\x0eworker_boot_id\x18\x03 \x01(\tR\fworkerBootId\x12\x1d\n" +
	"\n" +
	"request_id\x18\x05 \x01(\tR\trequestId\x12'\n" +
	"\x0fattempt_ordinal\x18\x06 \x01(\x04R\x0eattemptOrdinal\x124\n" +
	"\x06reason\x18\a \x01(\x0e2\x1c.cozy.worker.v1.CancelReasonR\x06reason\x12\x19\n" +
	"\bgrace_ms\x18\b \x01(\x04R\agraceMs\x124\n" +
	"\x16invocation_spec_digest\x18\t \x01(\fR\x14invocationSpecDigestJ\x04\b\x04\x10\x05\"\xbb\x03\n" +
	"\x0eAttemptOutcome\x12,\n" +
	"\x12record_owner_epoch\x18\x01 \x01(\x04R\x10recordOwnerEpoch\x120\n" +
	"\x14control_stream_epoch\x18\x02 \x01(\x04R\x12controlStreamEpoch\x12$\n" +
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
	"\fplacement_id\x18\v \x01(\tR\vplacementIdJ\x04\b\x04\x10\x05\"\x9c\x05\n" +
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
	"\x11execution_started\x18\v \x01(\bR\x10executionStarted\x12L\n" +
	"\x10weights_receipts\x18\f \x03(\v2!.cozy.worker.v1.WeightsReceiptRefR\x0fweightsReceipts\"\x90\x01\n" +
	"\x11WeightsReceiptRef\x124\n" +
	"\x16weights_receipt_digest\x18\x01 \x01(\fR\x14weightsReceiptDigest\x12E\n" +
	"\x1fweights_receipt_canonical_bytes\x18\x02 \x01(\fR\x1cweightsReceiptCanonicalBytes\"\x9c\x03\n" +
	"\x0eWeightsReceipt\x122\n" +
	"\x15owner_authority_scope\x18\x01 \x01(\tR\x13ownerAuthorityScope\x12\x1d\n" +
	"\n" +
	"request_id\x18\x02 \x01(\tR\trequestId\x124\n" +
	"\x16invocation_spec_digest\x18\x03 \x01(\tR\x14invocationSpecDigest\x12\x1f\n" +
	"\voutput_slot\x18\x04 \x01(\tR\n" +
	"outputSlot\x124\n" +
	"\x16weights_transaction_id\x18\x05 \x01(\tR\x14weightsTransactionId\x126\n" +
	"\x17tensorfs_receipt_digest\x18\x06 \x01(\tR\x15tensorfsReceiptDigest\x12G\n" +
	" tensorfs_receipt_canonical_bytes\x18\a \x01(\fR\x1dtensorfsReceiptCanonicalBytesJ\x04\b\b\x10\tR#checkpoint_evidence_canonical_bytes\"i\n" +
	"\x13WeightsObjectSource\x12\x1b\n" +
	"\tobject_id\x18\x01 \x01(\tR\bobjectId\x12\x16\n" +
	"\x06length\x18\x02 \x01(\x04R\x06length\x12\x1d\n" +
	"\n" +
	"source_ref\x18\x03 \x01(\tR\tsourceRef\"\xbc\x04\n" +
	"\x12WeightsIntentFrame\x12,\n" +
	"\x12record_owner_epoch\x18\x01 \x01(\x04R\x10recordOwnerEpoch\x120\n" +
	"\x14control_stream_epoch\x18\x02 \x01(\x04R\x12controlStreamEpoch\x12$\n" +
	"\x0eworker_boot_id\x18\x03 \x01(\tR\fworkerBootId\x12\x1d\n" +
	"\n" +
	"request_id\x18\x05 \x01(\tR\trequestId\x12'\n" +
	"\x0fattempt_ordinal\x18\x06 \x01(\x04R\x0eattemptOrdinal\x124\n" +
	"\x16invocation_spec_digest\x18\a \x01(\fR\x14invocationSpecDigest\x12\x1f\n" +
	"\voutput_slot\x18\b \x01(\tR\n" +
	"outputSlot\x12>\n" +
	"\x1btensorfs_declaration_digest\x18\t \x01(\fR\x19tensorfsDeclarationDigest\x124\n" +
	"\x16requested_writer_epoch\x18\n" +
	" \x01(\x04R\x14requestedWriterEpoch\x12O\n" +
	"$tensorfs_declaration_canonical_bytes\x18\v \x01(\fR!tensorfsDeclarationCanonicalBytes\x124\n" +
	"\x16weights_transaction_id\x18\f \x01(\tR\x14weightsTransactionIdJ\x04\b\x04\x10\x05\"\xc4\x06\n" +
	"\x0eWeightsHostAck\x12,\n" +
	"\x12record_owner_epoch\x18\x01 \x01(\x04R\x10recordOwnerEpoch\x120\n" +
	"\x14control_stream_epoch\x18\x02 \x01(\x04R\x12controlStreamEpoch\x12$\n" +
	"\x0eworker_boot_id\x18\x03 \x01(\tR\fworkerBootId\x12\x1d\n" +
	"\n" +
	"request_id\x18\x05 \x01(\tR\trequestId\x12'\n" +
	"\x0fattempt_ordinal\x18\x06 \x01(\x04R\x0eattemptOrdinal\x124\n" +
	"\x16invocation_spec_digest\x18\a \x01(\fR\x14invocationSpecDigest\x12\x1f\n" +
	"\voutput_slot\x18\b \x01(\tR\n" +
	"outputSlot\x124\n" +
	"\x16weights_transaction_id\x18\t \x01(\tR\x14weightsTransactionId\x12!\n" +
	"\fwriter_epoch\x18\n" +
	" \x01(\x04R\vwriterEpoch\x12>\n" +
	"\x1btensorfs_declaration_digest\x18\v \x01(\fR\x19tensorfsDeclarationDigest\x126\n" +
	"\x05stage\x18\f \x01(\x0e2 .cozy.worker.v1.WeightsHostStageR\x05stage\x12<\n" +
	"\aoutcome\x18\r \x01(\x0e2\".cozy.worker.v1.WeightsHostOutcomeR\aoutcome\x12<\n" +
	"\arefusal\x18\x0e \x01(\x0e2\".cozy.worker.v1.WeightsHostRefusalR\arefusal\x12J\n" +
	"\x0fweights_receipt\x18\x0f \x01(\v2!.cozy.worker.v1.WeightsReceiptRefR\x0eweightsReceipt\x12/\n" +
	"\bmanifest\x18\x10 \x01(\v2\x13.cozy.worker.v1.RefR\bmanifest\x12=\n" +
	"\n" +
	"checkpoint\x18\x11 \x01(\v2\x1d.cozy.worker.v1.CheckpointRefR\n" +
	"checkpointJ\x04\b\x04\x10\x05\"\x85\x01\n" +
	"\rCheckpointRef\x12'\n" +
	"\x04head\x18\x01 \x01(\v2\x13.cozy.worker.v1.RefR\x04head\x12\x1f\n" +
	"\vplan_digest\x18\x02 \x01(\fR\n" +
	"planDigest\x12\x14\n" +
	"\x05index\x18\x03 \x01(\x04R\x05index\x12\x14\n" +
	"\x05bytes\x18\x04 \x01(\x04R\x05bytes\"\x9b\x04\n" +
	"\x16WeightsCheckpointFrame\x12,\n" +
	"\x12record_owner_epoch\x18\x01 \x01(\x04R\x10recordOwnerEpoch\x120\n" +
	"\x14control_stream_epoch\x18\x02 \x01(\x04R\x12controlStreamEpoch\x12$\n" +
	"\x0eworker_boot_id\x18\x03 \x01(\tR\fworkerBootId\x12\x1d\n" +
	"\n" +
	"request_id\x18\x05 \x01(\tR\trequestId\x12'\n" +
	"\x0fattempt_ordinal\x18\x06 \x01(\x04R\x0eattemptOrdinal\x124\n" +
	"\x16invocation_spec_digest\x18\a \x01(\fR\x14invocationSpecDigest\x12\x1f\n" +
	"\voutput_slot\x18\b \x01(\tR\n" +
	"outputSlot\x124\n" +
	"\x16weights_transaction_id\x18\t \x01(\tR\x14weightsTransactionId\x12!\n" +
	"\fwriter_epoch\x18\n" +
	" \x01(\x04R\vwriterEpoch\x12>\n" +
	"\x1btensorfs_declaration_digest\x18\v \x01(\fR\x19tensorfsDeclarationDigest\x12=\n" +
	"\n" +
	"checkpoint\x18\f \x01(\v2\x1d.cozy.worker.v1.CheckpointRefR\n" +
	"checkpointJ\x04\b\x04\x10\x05\"\xd3\x02\n" +
	"\x19WeightsIntentReadyRequest\x12,\n" +
	"\x12record_owner_epoch\x18\x01 \x01(\x04R\x10recordOwnerEpoch\x120\n" +
	"\x14control_stream_epoch\x18\x02 \x01(\x04R\x12controlStreamEpoch\x12$\n" +
	"\x0eworker_boot_id\x18\x03 \x01(\tR\fworkerBootId\x12B\n" +
	"\aweights\x18\x05 \x01(\v2(.cozy.worker.v1.WeightsCheckpointSubjectR\aweights\x12'\n" +
	"\x0fattempt_ordinal\x18\x06 \x01(\x04R\x0eattemptOrdinal\x12=\n" +
	"\n" +
	"checkpoint\x18\a \x01(\v2\x1d.cozy.worker.v1.CheckpointRefR\n" +
	"checkpointJ\x04\b\x04\x10\x05\"\x8a\x01\n" +
	"\x16WeightsIntentReadyCall\x12+\n" +
	"\x05claim\x18\x01 \x01(\v2\x15.cozy.worker.v1.ClaimR\x05claim\x12C\n" +
	"\arequest\x18\x02 \x01(\v2).cozy.worker.v1.WeightsIntentReadyRequestR\arequest\"\xb6\x01\n" +
	" ValidateWeightsCheckpointRequest\x12A\n" +
	"\x06intent\x18\x01 \x01(\v2).cozy.worker.v1.WeightsIntentReadyRequestR\x06intent\x12O\n" +
	"$tensorfs_declaration_canonical_bytes\x18\x02 \x01(\fR!tensorfsDeclarationCanonicalBytes\"\xf8\x01\n" +
	"\x1fValidateWeightsCheckpointResult\x12B\n" +
	"\aweights\x18\x01 \x01(\v2(.cozy.worker.v1.WeightsCheckpointSubjectR\aweights\x12=\n" +
	"\n" +
	"checkpoint\x18\x02 \x01(\v2\x1d.cozy.worker.v1.CheckpointRefR\n" +
	"checkpoint\x12\x14\n" +
	"\x05valid\x18\x03 \x01(\bR\x05valid\x12\x1b\n" +
	"\tsafe_code\x18\x04 \x01(\tR\bsafeCode\x12\x1f\n" +
	"\vsafe_detail\x18\x05 \x01(\tR\n" +
	"safeDetail\"\x95\x05\n" +
	"\x13WeightsReceiptFrame\x12,\n" +
	"\x12record_owner_epoch\x18\x01 \x01(\x04R\x10recordOwnerEpoch\x120\n" +
	"\x14control_stream_epoch\x18\x02 \x01(\x04R\x12controlStreamEpoch\x12$\n" +
	"\x0eworker_boot_id\x18\x03 \x01(\tR\fworkerBootId\x12\x1d\n" +
	"\n" +
	"request_id\x18\x05 \x01(\tR\trequestId\x12'\n" +
	"\x0fattempt_ordinal\x18\x06 \x01(\x04R\x0eattemptOrdinal\x124\n" +
	"\x16invocation_spec_digest\x18\a \x01(\fR\x14invocationSpecDigest\x12\x1f\n" +
	"\voutput_slot\x18\b \x01(\tR\n" +
	"outputSlot\x124\n" +
	"\x16weights_transaction_id\x18\t \x01(\tR\x14weightsTransactionId\x12!\n" +
	"\fwriter_epoch\x18\n" +
	" \x01(\x04R\vwriterEpoch\x12>\n" +
	"\x1btensorfs_declaration_digest\x18\v \x01(\fR\x19tensorfsDeclarationDigest\x12J\n" +
	"\x0fweights_receipt\x18\f \x01(\v2!.cozy.worker.v1.WeightsReceiptRefR\x0eweightsReceipt\x12=\n" +
	"\aobjects\x18\r \x03(\v2#.cozy.worker.v1.WeightsObjectSourceR\aobjects\x12/\n" +
	"\bmanifest\x18\x0e \x01(\v2\x13.cozy.worker.v1.RefR\bmanifestJ\x04\b\x04\x10\x05\"\xa9\x04\n" +
	"\x18WeightsTransactionStatus\x124\n" +
	"\x16weights_transaction_id\x18\x01 \x01(\tR\x14weightsTransactionId\x12\x1d\n" +
	"\n" +
	"request_id\x18\x02 \x01(\tR\trequestId\x12'\n" +
	"\x0fattempt_ordinal\x18\x03 \x01(\x04R\x0eattemptOrdinal\x124\n" +
	"\x16invocation_spec_digest\x18\x04 \x01(\tR\x14invocationSpecDigest\x12\x1f\n" +
	"\voutput_slot\x18\x05 \x01(\tR\n" +
	"outputSlot\x12!\n" +
	"\fwriter_epoch\x18\x06 \x01(\x04R\vwriterEpoch\x12=\n" +
	"\x05state\x18\a \x01(\x0e2'.cozy.worker.v1.WeightsTransactionStateR\x05state\x12>\n" +
	"\x1btensorfs_declaration_digest\x18\b \x01(\fR\x19tensorfsDeclarationDigest\x124\n" +
	"\x16weights_receipt_digest\x18\t \x01(\fR\x14weightsReceiptDigest\x12=\n" +
	"\n" +
	"checkpoint\x18\n" +
	" \x01(\v2\x1d.cozy.worker.v1.CheckpointRefR\n" +
	"checkpoint\x12!\n" +
	"\fintent_ready\x18\v \x01(\bR\vintentReady\"\x83\x03\n" +
	"\x12WeightsReadRequest\x12,\n" +
	"\x12record_owner_epoch\x18\x01 \x01(\x04R\x10recordOwnerEpoch\x120\n" +
	"\x14control_stream_epoch\x18\x02 \x01(\x04R\x12controlStreamEpoch\x12$\n" +
	"\x0eworker_boot_id\x18\x03 \x01(\tR\fworkerBootId\x124\n" +
	"\x16weights_transaction_id\x18\x05 \x01(\tR\x14weightsTransactionId\x12!\n" +
	"\fwriter_epoch\x18\x06 \x01(\x04R\vwriterEpoch\x12\x1b\n" +
	"\tobject_id\x18\a \x01(\tR\bobjectId\x12\x1d\n" +
	"\n" +
	"source_ref\x18\b \x01(\tR\tsourceRef\x12\x17\n" +
	"\aread_id\x18\t \x01(\tR\x06readId\x12\x16\n" +
	"\x06offset\x18\n" +
	" \x01(\x04R\x06offset\x12\x1b\n" +
	"\tmax_bytes\x18\v \x01(\rR\bmaxBytesJ\x04\b\x04\x10\x05\"\xb7\x04\n" +
	"\x11WeightsReadResult\x12,\n" +
	"\x12record_owner_epoch\x18\x01 \x01(\x04R\x10recordOwnerEpoch\x120\n" +
	"\x14control_stream_epoch\x18\x02 \x01(\x04R\x12controlStreamEpoch\x12$\n" +
	"\x0eworker_boot_id\x18\x03 \x01(\tR\fworkerBootId\x124\n" +
	"\x16weights_transaction_id\x18\x05 \x01(\tR\x14weightsTransactionId\x12!\n" +
	"\fwriter_epoch\x18\x06 \x01(\x04R\vwriterEpoch\x12\x1b\n" +
	"\tobject_id\x18\a \x01(\tR\bobjectId\x12\x1d\n" +
	"\n" +
	"source_ref\x18\b \x01(\tR\tsourceRef\x12\x17\n" +
	"\aread_id\x18\t \x01(\tR\x06readId\x12\x16\n" +
	"\x06offset\x18\n" +
	" \x01(\x04R\x06offset\x12<\n" +
	"\aoutcome\x18\v \x01(\x0e2\".cozy.worker.v1.WeightsReadOutcomeR\aoutcome\x12\x12\n" +
	"\x04data\x18\f \x01(\fR\x04data\x12\x1f\n" +
	"\vdata_digest\x18\r \x01(\fR\n" +
	"dataDigest\x12<\n" +
	"\arefusal\x18\x0e \x01(\x0e2\".cozy.worker.v1.WeightsReadRefusalR\arefusal\x12\x1f\n" +
	"\vsafe_detail\x18\x0f \x01(\tR\n" +
	"safeDetailJ\x04\b\x04\x10\x05\"G\n" +
	"\x10WeightsObjectRef\x12\x1b\n" +
	"\tobject_id\x18\x01 \x01(\tR\bobjectId\x12\x16\n" +
	"\x06length\x18\x02 \x01(\x04R\x06length\"?\n" +
	"\x13WeightsUploadHeader\x12\x12\n" +
	"\x04name\x18\x01 \x01(\tR\x04name\x12\x14\n" +
	"\x05value\x18\x02 \x01(\tR\x05value\"\xd3\x01\n" +
	"\x12WeightsUploadGrant\x12\x1b\n" +
	"\tobject_id\x18\x01 \x01(\tR\bobjectId\x12\x16\n" +
	"\x06length\x18\x02 \x01(\x04R\x06length\x12\x10\n" +
	"\x03url\x18\x03 \x01(\tR\x03url\x12N\n" +
	"\x10required_headers\x18\x04 \x03(\v2#.cozy.worker.v1.WeightsUploadHeaderR\x0frequiredHeaders\x12&\n" +
	"\x0fexpires_at_unix\x18\x05 \x01(\x04R\rexpiresAtUnix\"\x86\x05\n" +
	"\x16WeightsTransferRequest\x12,\n" +
	"\x12record_owner_epoch\x18\x01 \x01(\x04R\x10recordOwnerEpoch\x120\n" +
	"\x14control_stream_epoch\x18\x02 \x01(\x04R\x12controlStreamEpoch\x12$\n" +
	"\x0eworker_boot_id\x18\x03 \x01(\tR\fworkerBootId\x12\x1d\n" +
	"\n" +
	"request_id\x18\x05 \x01(\tR\trequestId\x12'\n" +
	"\x0fattempt_ordinal\x18\x06 \x01(\x04R\x0eattemptOrdinal\x124\n" +
	"\x16invocation_spec_digest\x18\a \x01(\fR\x14invocationSpecDigest\x12\x1f\n" +
	"\voutput_slot\x18\b \x01(\tR\n" +
	"outputSlot\x124\n" +
	"\x16weights_transaction_id\x18\t \x01(\tR\x14weightsTransactionId\x124\n" +
	"\x16weights_receipt_digest\x18\n" +
	" \x01(\fR\x14weightsReceiptDigest\x12!\n" +
	"\foperation_id\x18\v \x01(\tR\voperationId\x12%\n" +
	"\x0egrant_revision\x18\f \x01(\x04R\rgrantRevision\x12G\n" +
	"\fupload_grant\x18\r \x01(\v2\".cozy.worker.v1.WeightsUploadGrantH\x00R\vuploadGrant\x126\n" +
	"\x04held\x18\x0e \x01(\v2 .cozy.worker.v1.WeightsObjectRefH\x00R\x04heldB\n" +
	"\n" +
	"\bdecisionJ\x04\b\x04\x10\x05\"\xd3\x03\n" +
	"\x14WeightsUploadRequest\x12,\n" +
	"\x12record_owner_epoch\x18\x01 \x01(\x04R\x10recordOwnerEpoch\x120\n" +
	"\x14control_stream_epoch\x18\x02 \x01(\x04R\x12controlStreamEpoch\x12$\n" +
	"\x0eworker_boot_id\x18\x03 \x01(\tR\fworkerBootId\x124\n" +
	"\x16weights_transaction_id\x18\x05 \x01(\tR\x14weightsTransactionId\x12!\n" +
	"\fwriter_epoch\x18\x06 \x01(\x04R\vwriterEpoch\x12\x1b\n" +
	"\tobject_id\x18\a \x01(\tR\bobjectId\x12\x1d\n" +
	"\n" +
	"source_ref\x18\b \x01(\tR\tsourceRef\x12\x16\n" +
	"\x06length\x18\t \x01(\x04R\x06length\x12!\n" +
	"\foperation_id\x18\n" +
	" \x01(\tR\voperationId\x12%\n" +
	"\x0egrant_revision\x18\v \x01(\x04R\rgrantRevision\x128\n" +
	"\x05grant\x18\f \x01(\v2\".cozy.worker.v1.WeightsUploadGrantR\x05grantJ\x04\b\x04\x10\x05\"\x8d\x05\n" +
	"\x13WeightsUploadResult\x12,\n" +
	"\x12record_owner_epoch\x18\x01 \x01(\x04R\x10recordOwnerEpoch\x120\n" +
	"\x14control_stream_epoch\x18\x02 \x01(\x04R\x12controlStreamEpoch\x12$\n" +
	"\x0eworker_boot_id\x18\x03 \x01(\tR\fworkerBootId\x124\n" +
	"\x16weights_transaction_id\x18\x05 \x01(\tR\x14weightsTransactionId\x12!\n" +
	"\fwriter_epoch\x18\x06 \x01(\x04R\vwriterEpoch\x12\x1b\n" +
	"\tobject_id\x18\a \x01(\tR\bobjectId\x12!\n" +
	"\foperation_id\x18\b \x01(\tR\voperationId\x12%\n" +
	"\x0egrant_revision\x18\t \x01(\x04R\rgrantRevision\x12>\n" +
	"\aoutcome\x18\n" +
	" \x01(\x0e2$.cozy.worker.v1.WeightsUploadOutcomeR\aoutcome\x12+\n" +
	"\x11transferred_bytes\x18\v \x01(\x04R\x10transferredBytes\x12\x1f\n" +
	"\vhttp_status\x18\f \x01(\rR\n" +
	"httpStatus\x12\x12\n" +
	"\x04etag\x18\r \x01(\tR\x04etag\x12'\n" +
	"\x0fchecksum_sha256\x18\x0e \x01(\tR\x0echecksumSha256\x12>\n" +
	"\arefusal\x18\x0f \x01(\x0e2$.cozy.worker.v1.WeightsUploadRefusalR\arefusal\x12\x1f\n" +
	"\vsafe_detail\x18\x10 \x01(\tR\n" +
	"safeDetailJ\x04\b\x04\x10\x05\"\xc1\x06\n" +
	"\x15WeightsTransferStatus\x12,\n" +
	"\x12record_owner_epoch\x18\x01 \x01(\x04R\x10recordOwnerEpoch\x120\n" +
	"\x14control_stream_epoch\x18\x02 \x01(\x04R\x12controlStreamEpoch\x12$\n" +
	"\x0eworker_boot_id\x18\x03 \x01(\tR\fworkerBootId\x12\x1d\n" +
	"\n" +
	"request_id\x18\x05 \x01(\tR\trequestId\x12'\n" +
	"\x0fattempt_ordinal\x18\x06 \x01(\x04R\x0eattemptOrdinal\x124\n" +
	"\x16invocation_spec_digest\x18\a \x01(\fR\x14invocationSpecDigest\x12\x1f\n" +
	"\voutput_slot\x18\b \x01(\tR\n" +
	"outputSlot\x124\n" +
	"\x16weights_transaction_id\x18\t \x01(\tR\x14weightsTransactionId\x12!\n" +
	"\foperation_id\x18\n" +
	" \x01(\tR\voperationId\x12%\n" +
	"\x0egrant_revision\x18\v \x01(\x04R\rgrantRevision\x12\x1b\n" +
	"\tobject_id\x18\f \x01(\tR\bobjectId\x12\x16\n" +
	"\x06length\x18\r \x01(\x04R\x06length\x12'\n" +
	"\x0fupdate_sequence\x18\x0e \x01(\x04R\x0eupdateSequence\x12:\n" +
	"\x05state\x18\x0f \x01(\x0e2$.cozy.worker.v1.WeightsTransferStateR\x05state\x12+\n" +
	"\x11transferred_bytes\x18\x10 \x01(\x04R\x10transferredBytes\x12\x1f\n" +
	"\vhttp_status\x18\x11 \x01(\rR\n" +
	"httpStatus\x12\x12\n" +
	"\x04etag\x18\x12 \x01(\tR\x04etag\x12'\n" +
	"\x0fchecksum_sha256\x18\x13 \x01(\tR\x0echecksumSha256\x12\x1a\n" +
	"\battempts\x18\x14 \x01(\rR\battempts\x12\x1b\n" +
	"\tsafe_code\x18\x15 \x01(\tR\bsafeCode\x12\x1f\n" +
	"\vsafe_detail\x18\x16 \x01(\tR\n" +
	"safeDetailJ\x04\b\x04\x10\x05\"\x90\x04\n" +
	"\x16ModelSourceFileRequest\x12,\n" +
	"\x12record_owner_epoch\x18\x01 \x01(\x04R\x10recordOwnerEpoch\x120\n" +
	"\x14control_stream_epoch\x18\x02 \x01(\x04R\x12controlStreamEpoch\x12$\n" +
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
	"\x13capability_revision\x18\r \x01(\x04R\x12capabilityRevision\x12\x16\n" +
	"\x06header\x18\x0e \x01(\fR\x06headerJ\x04\b\x04\x10\x05\"\xbf\x04\n" +
	"\x15ModelSourceFileStatus\x12,\n" +
	"\x12record_owner_epoch\x18\x01 \x01(\x04R\x10recordOwnerEpoch\x120\n" +
	"\x14control_stream_epoch\x18\x02 \x01(\x04R\x12controlStreamEpoch\x12$\n" +
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
	"safeDetailJ\x04\b\x04\x10\x05\"\xd5\x03\n" +
	"\x19ModelSourcePrepareRequest\x12,\n" +
	"\x12record_owner_epoch\x18\x01 \x01(\x04R\x10recordOwnerEpoch\x120\n" +
	"\x14control_stream_epoch\x18\x02 \x01(\x04R\x12controlStreamEpoch\x12$\n" +
	"\x0eworker_boot_id\x18\x03 \x01(\tR\fworkerBootId\x12!\n" +
	"\foperation_id\x18\x05 \x01(\tR\voperationId\x126\n" +
	"\x17source_selection_digest\x18\x06 \x01(\fR\x15sourceSelectionDigest\x12>\n" +
	"\bprofiles\x18\a \x03(\v2\".cozy.worker.v1.ModelSourceProfileR\bprofiles\x12\x1d\n" +
	"\n" +
	"source_uri\x18\b \x01(\tR\tsourceUri\x12)\n" +
	"\x10declared_license\x18\t \x01(\tR\x0fdeclaredLicense\x12G\n" +
	"\vcheckpoints\x18\n" +
	" \x03(\v2%.cozy.worker.v1.ModelSourceCheckpointR\vcheckpointsJ\x04\b\x04\x10\x05\"\x87\x04\n" +
	"\x13ModelSourcePrepared\x12,\n" +
	"\x12record_owner_epoch\x18\x01 \x01(\x04R\x10recordOwnerEpoch\x120\n" +
	"\x14control_stream_epoch\x18\x02 \x01(\x04R\x12controlStreamEpoch\x12$\n" +
	"\x0eworker_boot_id\x18\x03 \x01(\tR\fworkerBootId\x12!\n" +
	"\foperation_id\x18\x05 \x01(\tR\voperationId\x126\n" +
	"\x17source_selection_digest\x18\x06 \x01(\fR\x15sourceSelectionDigest\x12C\n" +
	"\aoutcome\x18\a \x01(\x0e2).cozy.worker.v1.ModelSourcePrepareOutcomeR\aoutcome\x12=\n" +
	"\asources\x18\b \x03(\v2#.cozy.worker.v1.PreparedModelSourceR\asources\x12\x1b\n" +
	"\tsafe_code\x18\t \x01(\tR\bsafeCode\x12\x1f\n" +
	"\vsafe_detail\x18\n" +
	" \x01(\tR\n" +
	"safeDetail\x12G\n" +
	"\vcheckpoints\x18\v \x03(\v2%.cozy.worker.v1.ModelSourceCheckpointR\vcheckpointsJ\x04\b\x04\x10\x05\"\x81\x01\n" +
	"\x15LocalPackageFileGrant\x12\x16\n" +
	"\x06digest\x18\x01 \x01(\fR\x06digest\x12\x1a\n" +
	"\bfilename\x18\x02 \x01(\tR\bfilename\x12\x16\n" +
	"\x06length\x18\x04 \x01(\x04R\x06length\x12\x10\n" +
	"\x03url\x18\x05 \x01(\tR\x03urlJ\x04\b\x03\x10\x04R\x04kind\"\xab\x02\n" +
	"\x18LocalPackageFetchRequest\x12,\n" +
	"\x12record_owner_epoch\x18\x01 \x01(\x04R\x10recordOwnerEpoch\x120\n" +
	"\x14control_stream_epoch\x18\x02 \x01(\x04R\x12controlStreamEpoch\x12$\n" +
	"\x0eworker_boot_id\x18\x03 \x01(\tR\fworkerBootId\x12!\n" +
	"\foperation_id\x18\x05 \x01(\tR\voperationId\x12#\n" +
	"\rsource_digest\x18\x06 \x01(\fR\fsourceDigest\x12;\n" +
	"\x05files\x18\a \x03(\v2%.cozy.worker.v1.LocalPackageFileGrantR\x05filesJ\x04\b\x04\x10\x05\"\xe6\x03\n" +
	"\x16LocalPackageFileStatus\x12,\n" +
	"\x12record_owner_epoch\x18\x01 \x01(\x04R\x10recordOwnerEpoch\x120\n" +
	"\x14control_stream_epoch\x18\x02 \x01(\x04R\x12controlStreamEpoch\x12$\n" +
	"\x0eworker_boot_id\x18\x03 \x01(\tR\fworkerBootId\x12!\n" +
	"\foperation_id\x18\x05 \x01(\tR\voperationId\x12#\n" +
	"\rsource_digest\x18\x06 \x01(\fR\fsourceDigest\x12\x16\n" +
	"\x06digest\x18\a \x01(\fR\x06digest\x12\x1a\n" +
	"\bfilename\x18\b \x01(\tR\bfilename\x12\x16\n" +
	"\x06length\x18\n" +
	" \x01(\x04R\x06length\x12;\n" +
	"\x05state\x18\v \x01(\x0e2%.cozy.worker.v1.LocalPackageFileStateR\x05state\x12%\n" +
	"\x0ereceived_bytes\x18\f \x01(\x04R\rreceivedBytes\x12\x1b\n" +
	"\tsafe_code\x18\r \x01(\tR\bsafeCode\x12\x1f\n" +
	"\vsafe_detail\x18\x0e \x01(\tR\n" +
	"safeDetailJ\x04\b\x04\x10\x05J\x04\b\t\x10\n" +
	"R\x04kind\"\x9b\x02\n" +
	"\x11LocalPackageAbort\x12,\n" +
	"\x12record_owner_epoch\x18\x01 \x01(\x04R\x10recordOwnerEpoch\x120\n" +
	"\x14control_stream_epoch\x18\x02 \x01(\x04R\x12controlStreamEpoch\x12$\n" +
	"\x0eworker_boot_id\x18\x03 \x01(\tR\fworkerBootId\x12!\n" +
	"\foperation_id\x18\x05 \x01(\tR\voperationId\x12#\n" +
	"\rsource_digest\x18\x06 \x01(\fR\fsourceDigest\x122\n" +
	"\x15local_revision_digest\x18\a \x01(\fR\x13localRevisionDigestJ\x04\b\x04\x10\x05\"\xa3\x03\n" +
	"\x17LocalPackageAbortStatus\x12,\n" +
	"\x12record_owner_epoch\x18\x01 \x01(\x04R\x10recordOwnerEpoch\x120\n" +
	"\x14control_stream_epoch\x18\x02 \x01(\x04R\x12controlStreamEpoch\x12$\n" +
	"\x0eworker_boot_id\x18\x03 \x01(\tR\fworkerBootId\x12!\n" +
	"\foperation_id\x18\x05 \x01(\tR\voperationId\x12#\n" +
	"\rsource_digest\x18\x06 \x01(\fR\fsourceDigest\x122\n" +
	"\x15local_revision_digest\x18\a \x01(\fR\x13localRevisionDigest\x12B\n" +
	"\aoutcome\x18\b \x01(\x0e2(.cozy.worker.v1.LocalPackageAbortOutcomeR\aoutcome\x12\x1b\n" +
	"\tsafe_code\x18\t \x01(\tR\bsafeCode\x12\x1f\n" +
	"\vsafe_detail\x18\n" +
	" \x01(\tR\n" +
	"safeDetailJ\x04\b\x04\x10\x05\"\xc1\x04\n" +
	"\x16WeightsFinalizeRequest\x12,\n" +
	"\x12record_owner_epoch\x18\x01 \x01(\x04R\x10recordOwnerEpoch\x120\n" +
	"\x14control_stream_epoch\x18\x02 \x01(\x04R\x12controlStreamEpoch\x12$\n" +
	"\x0eworker_boot_id\x18\x03 \x01(\tR\fworkerBootId\x12\x1d\n" +
	"\n" +
	"request_id\x18\x05 \x01(\tR\trequestId\x124\n" +
	"\x16invocation_spec_digest\x18\x06 \x01(\fR\x14invocationSpecDigest\x12\x1f\n" +
	"\voutput_slot\x18\a \x01(\tR\n" +
	"outputSlot\x12L\n" +
	"\vdisposition\x18\b \x01(\x0e2*.cozy.worker.v1.WeightsFinalizeDispositionR\vdisposition\x124\n" +
	"\x16weights_receipt_digest\x18\t \x01(\fR\x14weightsReceiptDigest\x12&\n" +
	"\x0fscratch_root_id\x18\n" +
	" \x01(\tR\rscratchRootId\x122\n" +
	"\x15owner_authority_scope\x18\v \x01(\tR\x13ownerAuthorityScope\x12E\n" +
	"\x1finvocation_spec_canonical_bytes\x18\f \x01(\fR\x1cinvocationSpecCanonicalBytesJ\x04\b\x04\x10\x05\"\xdb\x03\n" +
	"\x15WeightsFinalizeResult\x12,\n" +
	"\x12record_owner_epoch\x18\x01 \x01(\x04R\x10recordOwnerEpoch\x120\n" +
	"\x14control_stream_epoch\x18\x02 \x01(\x04R\x12controlStreamEpoch\x12$\n" +
	"\x0eworker_boot_id\x18\x03 \x01(\tR\fworkerBootId\x12\x1d\n" +
	"\n" +
	"request_id\x18\x05 \x01(\tR\trequestId\x124\n" +
	"\x16invocation_spec_digest\x18\x06 \x01(\fR\x14invocationSpecDigest\x12\x1f\n" +
	"\voutput_slot\x18\a \x01(\tR\n" +
	"outputSlot\x12@\n" +
	"\aoutcome\x18\b \x01(\x0e2&.cozy.worker.v1.WeightsFinalizeOutcomeR\aoutcome\x12J\n" +
	"\x0fweights_receipt\x18\t \x01(\v2!.cozy.worker.v1.WeightsReceiptRefR\x0eweightsReceipt\x122\n" +
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
	"\x0eevidence_class\x18\x06 \x01(\tR\revidenceClass\"\x84\x03\n" +
	"\x11AttemptOutcomeAck\x12,\n" +
	"\x12record_owner_epoch\x18\x01 \x01(\x04R\x10recordOwnerEpoch\x120\n" +
	"\x14control_stream_epoch\x18\x02 \x01(\x04R\x12controlStreamEpoch\x12$\n" +
	"\x0eworker_boot_id\x18\x03 \x01(\tR\fworkerBootId\x12\x1d\n" +
	"\n" +
	"request_id\x18\x05 \x01(\tR\trequestId\x12'\n" +
	"\x0fattempt_ordinal\x18\x06 \x01(\x04R\x0eattemptOrdinal\x124\n" +
	"\x16invocation_spec_digest\x18\a \x01(\fR\x14invocationSpecDigest\x12\x1d\n" +
	"\n" +
	"outcome_id\x18\b \x01(\tR\toutcomeId\x12%\n" +
	"\x0eoutcome_digest\x18\t \x01(\fR\routcomeDigest\x12\x1f\n" +
	"\vretain_work\x18\n" +
	" \x01(\bR\n" +
	"retainWorkJ\x04\b\x04\x10\x05\"\xa2\x03\n" +
	"\x14JobCheckpointRequest\x12,\n" +
	"\x12record_owner_epoch\x18\x01 \x01(\x04R\x10recordOwnerEpoch\x120\n" +
	"\x14control_stream_epoch\x18\x02 \x01(\x04R\x12controlStreamEpoch\x12$\n" +
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
	"\bartifact\x18\v \x01(\v2\x1b.cozy.worker.v1.OutputEntryR\bartifactJ\x04\b\x04\x10\x05\"\xea\x03\n" +
	"\x14JobCheckpointReceipt\x12,\n" +
	"\x12record_owner_epoch\x18\x01 \x01(\x04R\x10recordOwnerEpoch\x120\n" +
	"\x14control_stream_epoch\x18\x02 \x01(\x04R\x12controlStreamEpoch\x12$\n" +
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
	"\x17recorded_content_digest\x18\x03 \x01(\fR\x15recordedContentDigest\"\xd1\x02\n" +
	"\x10JobCheckpointAck\x12,\n" +
	"\x12record_owner_epoch\x18\x01 \x01(\x04R\x10recordOwnerEpoch\x120\n" +
	"\x14control_stream_epoch\x18\x02 \x01(\x04R\x12controlStreamEpoch\x12$\n" +
	"\x0eworker_boot_id\x18\x03 \x01(\tR\fworkerBootId\x12\x1d\n" +
	"\n" +
	"request_id\x18\x05 \x01(\tR\trequestId\x12'\n" +
	"\x0fattempt_ordinal\x18\x06 \x01(\x04R\x0eattemptOrdinal\x12#\n" +
	"\roperation_key\x18\a \x01(\tR\foperationKey\x12\x1d\n" +
	"\n" +
	"receipt_id\x18\b \x01(\tR\treceiptId\x12%\n" +
	"\x0econtent_digest\x18\t \x01(\fR\rcontentDigestJ\x04\b\x04\x10\x05\"\xb9\x01\n" +
	"\fProgressOpen\x12,\n" +
	"\x12record_owner_epoch\x18\x01 \x01(\x04R\x10recordOwnerEpoch\x120\n" +
	"\x14control_stream_epoch\x18\x02 \x01(\x04R\x12controlStreamEpoch\x12$\n" +
	"\x0eworker_boot_id\x18\x03 \x01(\tR\fworkerBootId\x12\x1d\n" +
	"\n" +
	"request_id\x18\x05 \x01(\tR\trequestIdJ\x04\b\x04\x10\x05\"\xd1\x02\n" +
	"\x0fAttemptProgress\x12,\n" +
	"\x12record_owner_epoch\x18\x01 \x01(\x04R\x10recordOwnerEpoch\x120\n" +
	"\x14control_stream_epoch\x18\x02 \x01(\x04R\x12controlStreamEpoch\x12$\n" +
	"\x0eworker_boot_id\x18\x03 \x01(\tR\fworkerBootId\x12\x1d\n" +
	"\n" +
	"request_id\x18\x05 \x01(\tR\trequestId\x12'\n" +
	"\x0fattempt_ordinal\x18\x06 \x01(\x04R\x0eattemptOrdinal\x12\x10\n" +
	"\x03seq\x18\a \x01(\x04R\x03seq\x12!\n" +
	"\fcontent_type\x18\b \x01(\tR\vcontentType\x12\x12\n" +
	"\x04data\x18\t \x01(\fR\x04data\x12!\n" +
	"\fplacement_id\x18\n" +
	" \x01(\tR\vplacementIdJ\x04\b\x04\x10\x05\"\xc9\x03\n" +
	"\x0eInvocationSpec\x12-\n" +
	"\x12environment_digest\x18\x02 \x01(\tR\x11environmentDigest\x12%\n" +
	"\x0epayload_digest\x18\x04 \x01(\tR\rpayloadDigest\x124\n" +
	"\x06inputs\x18\x05 \x03(\v2\x1c.cozy.worker.v1.InputBindingR\x06inputs\x127\n" +
	"\aoutputs\x18\x06 \x03(\v2\x1d.cozy.worker.v1.OutputBindingR\aoutputs\x12(\n" +
	"\x10deadline_unix_ms\x18\a \x01(\x04R\x0edeadlineUnixMs\x12A\n" +
	"\aserving\x18\b \x01(\v2%.cozy.worker.v1.ServingInvocationSpecH\x00R\aserving\x125\n" +
	"\x03job\x18\t \x01(\v2!.cozy.worker.v1.JobInvocationSpecH\x00R\x03jobB\x06\n" +
	"\x04specJ\x04\b\x01\x10\x02J\x04\b\x03\x10\x04R\x17package_revision_digestR\x12package_release_idR\rconfig_digest\"\x8c\x01\n" +
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
	"\x03url\x18\x02 \x01(\tR\x03url\"\xb2\x01\n" +
	"\x18DeliveryAccessCredential\x12\x16\n" +
	"\x06issuer\x18\x01 \x01(\tR\x06issuer\x12\x15\n" +
	"\x06key_id\x18\x02 \x01(\tR\x05keyId\x12)\n" +
	"\x10credential_epoch\x18\x03 \x01(\x04R\x0fcredentialEpoch\x12&\n" +
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
	"unreadable\"\xf9\x01\n" +
	"\vJobCapacity\x12$\n" +
	"\x0ejobs_in_flight\x18\x01 \x01(\rR\fjobsInFlight\x12%\n" +
	"\x0ejobs_available\x18\x02 \x01(\rR\rjobsAvailable\x12&\n" +
	"\x0ffree_disk_bytes\x18\x04 \x01(\x04R\rfreeDiskBytes\x126\n" +
	"\x17orchestration_in_flight\x18\x05 \x01(\rR\x15orchestrationInFlight\x127\n" +
	"\x17orchestration_available\x18\x06 \x01(\rR\x16orchestrationAvailableJ\x04\b\x03\x10\x04\"\xa8\x04\n" +
	"\vHeldAttempt\x12\x1d\n" +
	"\n" +
	"request_id\x18\x01 \x01(\tR\trequestId\x12'\n" +
	"\x0fattempt_ordinal\x18\x02 \x01(\x04R\x0eattemptOrdinal\x12/\n" +
	"\x04kind\x18\x03 \x01(\x0e2\x1b.cozy.worker.v1.AttemptKindR\x04kind\x122\n" +
	"\x05state\x18\x04 \x01(\x0e2\x1c.cozy.worker.v1.AttemptStateR\x05state\x124\n" +
	"\x16invocation_spec_digest\x18\x05 \x01(\fR\x14invocationSpecDigest\x12!\n" +
	"\fplacement_id\x18\x06 \x01(\tR\vplacementId\x12%\n" +
	"\x0eexecutor_epoch\x18\a \x01(\x04R\rexecutorEpoch\x12\x1d\n" +
	"\n" +
	"outcome_id\x18\b \x01(\tR\toutcomeId\x12%\n" +
	"\x0eoutcome_digest\x18\t \x01(\fR\routcomeDigest\x12\x17\n" +
	"\alane_id\x18\n" +
	" \x01(\tR\x06laneId\x12%\n" +
	"\x0equeue_position\x18\v \x01(\rR\rqueuePosition\x12\x1c\n" +
	"\tovertaken\x18\f \x01(\rR\tovertaken\x12'\n" +
	"\x0fovertake_budget\x18\r \x01(\rR\x0eovertakeBudget\x12\x1f\n" +
	"\vplan_digest\x18\x0e \x01(\fR\n" +
	"planDigest\"\x80\x01\n" +
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
	"\tmime_type\x18\x04 \x01(\tR\bmimeType\"\xf0\x03\n" +
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
	"\x11unverified_fields\x18\f \x03(\tR\x10unverifiedFields\x12\x17\n" +
	"\apost_ms\x18\r \x01(\x04R\x06postMs\"\x80\x01\n" +
	"\x0fTriageBundleRef\x12\x1d\n" +
	"\n" +
	"subject_id\x18\x01 \x01(\tR\tsubjectId\x120\n" +
	"\x14write_receipt_digest\x18\x04 \x01(\fR\x12writeReceiptDigest\x12\x16\n" +
	"\x06length\x18\x03 \x01(\x04R\x06lengthJ\x04\b\x02\x10\x03*\xca\x01\n" +
	"\x0eChildCallState\x12 \n" +
	"\x1cCHILD_CALL_STATE_UNSPECIFIED\x10\x00\x12\x1c\n" +
	"\x18CHILD_CALL_STATE_PENDING\x10\x01\x12\x1e\n" +
	"\x1aCHILD_CALL_STATE_SUCCEEDED\x10\x02\x12\x1c\n" +
	"\x18CHILD_CALL_STATE_REFUSED\x10\x03\x12\x1b\n" +
	"\x17CHILD_CALL_STATE_FAILED\x10\x04\x12\x1d\n" +
	"\x19CHILD_CALL_STATE_CANCELED\x10\x05*O\n" +
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
	"\x10ATTEMPT_KIND_JOB\x10\x02*\xce\x01\n" +
	"\fAttemptState\x12\x1d\n" +
	"\x19ATTEMPT_STATE_UNSPECIFIED\x10\x00\x12\x19\n" +
	"\x15ATTEMPT_STATE_RUNNING\x10\x01\x12%\n" +
	"!ATTEMPT_STATE_OUTCOME_PENDING_ACK\x10\x02\x12 \n" +
	"\x1cATTEMPT_STATE_HELD_UNDURABLE\x10\x03\x12\x18\n" +
	"\x14ATTEMPT_STATE_QUEUED\x10\x04\x12!\n" +
	"\x1dATTEMPT_STATE_DEVICE_RELEASED\x10\x05*\xbf\x01\n" +
	"\rOutcomeStatus\x12\x1e\n" +
	"\x1aOUTCOME_STATUS_UNSPECIFIED\x10\x00\x12\x1c\n" +
	"\x18OUTCOME_STATUS_SUCCEEDED\x10\x01\x12\x1a\n" +
	"\x16OUTCOME_STATUS_REFUSED\x10\x02\x12\x19\n" +
	"\x15OUTCOME_STATUS_FAILED\x10\x03\x12\x1b\n" +
	"\x17OUTCOME_STATUS_CANCELED\x10\x04\x12\x1c\n" +
	"\x18OUTCOME_STATUS_ABANDONED\x10\x05*\xbf\x05\n" +
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
	"\x16CAUSE_CODE_NO_CAPACITY\x10\x11\x12$\n" +
	" CAUSE_CODE_ADMISSION_EPOCH_STALE\x10\x12\x12 \n" +
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
	"\x16CANCEL_REASON_DEADLINE\x10\x05*\x96\x03\n" +
	"\x0eClaimRejection\x12\x1f\n" +
	"\x1bCLAIM_REJECTION_UNSPECIFIED\x10\x00\x12#\n" +
	"\x1fCLAIM_REJECTION_UNAUTHENTICATED\x10\x01\x12,\n" +
	"(CLAIM_REJECTION_STALE_RECORD_OWNER_EPOCH\x10\x02\x12\x1e\n" +
	"\x1aCLAIM_REJECTION_EPOCH_HELD\x10\x03\x12&\n" +
	"\"CLAIM_REJECTION_WORKER_ID_MISMATCH\x10\x04\x12'\n" +
	"#CLAIM_REJECTION_RELEASE_ID_MISMATCH\x10\x05\x12\x1d\n" +
	"\x19CLAIM_REJECTION_UNDURABLE\x10\a\x12)\n" +
	"%CLAIM_REJECTION_PROTOCOL_INCOMPATIBLE\x10\t\"\x04\b\x06\x10\x06\"\x04\b\b\x10\b*!CLAIM_REJECTION_UNSUPPORTED_MINOR*&CLAIM_REJECTION_SCHEMA_DIGEST_MISMATCH*\x80\x05\n" +
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
	"$CHECKPOINT_FAULT_CODE_QUOTA_EXCEEDED\x10\x05*\xbc\x01\n" +
	"\fPrepareStage\x12\x1d\n" +
	"\x19PREPARE_STAGE_UNSPECIFIED\x10\x00\x12\x1a\n" +
	"\x16PREPARE_STAGE_RESOLVED\x10\x01\x12\x1d\n" +
	"\x19PREPARE_STAGE_DOWNLOADING\x10\x02\x12\x1b\n" +
	"\x17PREPARE_STAGE_PREPARING\x10\x03\x12\x1a\n" +
	"\x16PREPARE_STAGE_PREPARED\x10\x04\x12\x19\n" +
	"\x15PREPARE_STAGE_REFUSED\x10\x05*\x98\x01\n" +
	"\x10WeightsHostStage\x12\"\n" +
	"\x1eWEIGHTS_HOST_STAGE_UNSPECIFIED\x10\x00\x12\x1d\n" +
	"\x19WEIGHTS_HOST_STAGE_INTENT\x10\x01\x12\x1e\n" +
	"\x1aWEIGHTS_HOST_STAGE_RECEIPT\x10\x02\x12!\n" +
	"\x1dWEIGHTS_HOST_STAGE_CHECKPOINT\x10\x03*\xa2\x01\n" +
	"\x12WeightsHostOutcome\x12$\n" +
	" WEIGHTS_HOST_OUTCOME_UNSPECIFIED\x10\x00\x12!\n" +
	"\x1dWEIGHTS_HOST_OUTCOME_RECORDED\x10\x01\x12!\n" +
	"\x1dWEIGHTS_HOST_OUTCOME_REPLAYED\x10\x02\x12 \n" +
	"\x1cWEIGHTS_HOST_OUTCOME_REFUSED\x10\x03*\x8c\x02\n" +
	"\x12WeightsHostRefusal\x12$\n" +
	" WEIGHTS_HOST_REFUSAL_UNSPECIFIED\x10\x00\x12(\n" +
	"$WEIGHTS_HOST_REFUSAL_UNKNOWN_ATTEMPT\x10\x01\x12(\n" +
	"$WEIGHTS_HOST_REFUSAL_INTENT_CONFLICT\x10\x02\x12%\n" +
	"!WEIGHTS_HOST_REFUSAL_STALE_WRITER\x10\x03\x12)\n" +
	"%WEIGHTS_HOST_REFUSAL_RECEIPT_CONFLICT\x10\x04\x12*\n" +
	"&WEIGHTS_HOST_REFUSAL_INVENTORY_INVALID\x10\x05*\x91\x01\n" +
	"\x17WeightsTransactionState\x12)\n" +
	"%WEIGHTS_TRANSACTION_STATE_UNSPECIFIED\x10\x00\x12$\n" +
	" WEIGHTS_TRANSACTION_STATE_INTENT\x10\x01\x12%\n" +
	"!WEIGHTS_TRANSACTION_STATE_RECEIPT\x10\x02*{\n" +
	"\x12WeightsReadOutcome\x12$\n" +
	" WEIGHTS_READ_OUTCOME_UNSPECIFIED\x10\x00\x12\x1d\n" +
	"\x19WEIGHTS_READ_OUTCOME_DATA\x10\x01\x12 \n" +
	"\x1cWEIGHTS_READ_OUTCOME_REFUSED\x10\x02*\xbb\x02\n" +
	"\x12WeightsReadRefusal\x12$\n" +
	" WEIGHTS_READ_REFUSAL_UNSPECIFIED\x10\x00\x12,\n" +
	"(WEIGHTS_READ_REFUSAL_UNKNOWN_TRANSACTION\x10\x01\x12%\n" +
	"!WEIGHTS_READ_REFUSAL_STALE_WRITER\x10\x02\x12'\n" +
	"#WEIGHTS_READ_REFUSAL_UNKNOWN_OBJECT\x10\x03\x12,\n" +
	"(WEIGHTS_READ_REFUSAL_SOURCE_REF_MISMATCH\x10\x04\x12&\n" +
	"\"WEIGHTS_READ_REFUSAL_RANGE_INVALID\x10\x05\x12+\n" +
	"'WEIGHTS_READ_REFUSAL_SOURCE_UNAVAILABLE\x10\x06*\xb3\x01\n" +
	"\x14WeightsUploadOutcome\x12&\n" +
	"\"WEIGHTS_UPLOAD_OUTCOME_UNSPECIFIED\x10\x00\x12#\n" +
	"\x1fWEIGHTS_UPLOAD_OUTCOME_UPLOADED\x10\x01\x12*\n" +
	"&WEIGHTS_UPLOAD_OUTCOME_ALREADY_PRESENT\x10\x02\x12\"\n" +
	"\x1eWEIGHTS_UPLOAD_OUTCOME_REFUSED\x10\x03*\xf7\x02\n" +
	"\x14WeightsUploadRefusal\x12&\n" +
	"\"WEIGHTS_UPLOAD_REFUSAL_UNSPECIFIED\x10\x00\x12.\n" +
	"*WEIGHTS_UPLOAD_REFUSAL_UNKNOWN_TRANSACTION\x10\x01\x12'\n" +
	"#WEIGHTS_UPLOAD_REFUSAL_STALE_WRITER\x10\x02\x12)\n" +
	"%WEIGHTS_UPLOAD_REFUSAL_UNKNOWN_OBJECT\x10\x03\x12.\n" +
	"*WEIGHTS_UPLOAD_REFUSAL_SOURCE_REF_MISMATCH\x10\x04\x12-\n" +
	")WEIGHTS_UPLOAD_REFUSAL_SOURCE_UNAVAILABLE\x10\x05\x12(\n" +
	"$WEIGHTS_UPLOAD_REFUSAL_GRANT_REFUSED\x10\x06\x12*\n" +
	"&WEIGHTS_UPLOAD_REFUSAL_TRANSFER_FAILED\x10\a*\x9e\x02\n" +
	"\x14WeightsTransferState\x12&\n" +
	"\"WEIGHTS_TRANSFER_STATE_UNSPECIFIED\x10\x00\x12#\n" +
	"\x1fWEIGHTS_TRANSFER_STATE_ACCEPTED\x10\x01\x12$\n" +
	" WEIGHTS_TRANSFER_STATE_UPLOADING\x10\x02\x12#\n" +
	"\x1fWEIGHTS_TRANSFER_STATE_UPLOADED\x10\x03\x12*\n" +
	"&WEIGHTS_TRANSFER_STATE_ALREADY_PRESENT\x10\x04\x12\x1f\n" +
	"\x1bWEIGHTS_TRANSFER_STATE_HELD\x10\x05\x12!\n" +
	"\x1dWEIGHTS_TRANSFER_STATE_FAILED\x10\x06*\x87\x01\n" +
	"\x13ModelSourceProvider\x12%\n" +
	"!MODEL_SOURCE_PROVIDER_UNSPECIFIED\x10\x00\x12&\n" +
	"\"MODEL_SOURCE_PROVIDER_HUGGING_FACE\x10\x01\x12!\n" +
	"\x1dMODEL_SOURCE_PROVIDER_CIVITAI\x10\x02*\xff\x01\n" +
	"\x14ModelSourceFileState\x12'\n" +
	"#MODEL_SOURCE_FILE_STATE_UNSPECIFIED\x10\x00\x12$\n" +
	" MODEL_SOURCE_FILE_STATE_ACCEPTED\x10\x01\x12'\n" +
	"#MODEL_SOURCE_FILE_STATE_DOWNLOADING\x10\x02\x12$\n" +
	" MODEL_SOURCE_FILE_STATE_VERIFIED\x10\x03\x12\"\n" +
	"\x1eMODEL_SOURCE_FILE_STATE_FAILED\x10\x04\x12%\n" +
	"!MODEL_SOURCE_FILE_STATE_CONVERTED\x10\x05*\xf6\x01\n" +
	"\x19ModelSourcePrepareOutcome\x12,\n" +
	"(MODEL_SOURCE_PREPARE_OUTCOME_UNSPECIFIED\x10\x00\x12)\n" +
	"%MODEL_SOURCE_PREPARE_OUTCOME_PREPARED\x10\x01\x12)\n" +
	"%MODEL_SOURCE_PREPARE_OUTCOME_REPLAYED\x10\x02\x12(\n" +
	"$MODEL_SOURCE_PREPARE_OUTCOME_REFUSED\x10\x03\x12+\n" +
	"'MODEL_SOURCE_PREPARE_OUTCOME_INCOMPLETE\x10\x04*\xb6\x01\n" +
	"\x15LocalPackageFileState\x12(\n" +
	"$LOCAL_PACKAGE_FILE_STATE_UNSPECIFIED\x10\x00\x12&\n" +
	"\"LOCAL_PACKAGE_FILE_STATE_RECEIVING\x10\x01\x12%\n" +
	"!LOCAL_PACKAGE_FILE_STATE_VERIFIED\x10\x02\x12$\n" +
	" LOCAL_PACKAGE_FILE_STATE_REFUSED\x10\x03*\xc5\x01\n" +
	"\x18LocalPackageAbortOutcome\x12+\n" +
	"'LOCAL_PACKAGE_ABORT_OUTCOME_UNSPECIFIED\x10\x00\x12)\n" +
	"%LOCAL_PACKAGE_ABORT_OUTCOME_ABANDONED\x10\x01\x12(\n" +
	"$LOCAL_PACKAGE_ABORT_OUTCOME_REPLAYED\x10\x02\x12'\n" +
	"#LOCAL_PACKAGE_ABORT_OUTCOME_REFUSED\x10\x03*\xd2\x01\n" +
	"\x1aWeightsFinalizeDisposition\x12,\n" +
	"(WEIGHTS_FINALIZE_DISPOSITION_UNSPECIFIED\x10\x00\x12&\n" +
	"\"WEIGHTS_FINALIZE_DISPOSITION_ADOPT\x10\x01\x12(\n" +
	"$WEIGHTS_FINALIZE_DISPOSITION_ABANDON\x10\x02\x124\n" +
	"0WEIGHTS_FINALIZE_DISPOSITION_ABANDON_UNCOMMITTED\x10\x03*\x90\x01\n" +
	"\x16WeightsFinalizeOutcome\x12(\n" +
	"$WEIGHTS_FINALIZE_OUTCOME_UNSPECIFIED\x10\x00\x12$\n" +
	" WEIGHTS_FINALIZE_OUTCOME_ADOPTED\x10\x01\x12&\n" +
	"\"WEIGHTS_FINALIZE_OUTCOME_ABANDONED\x10\x022\xaf\x01\n" +
	"\rWorkerControl\x12L\n" +
	"\aControl\x12 .cozy.worker.v1.RecordOwnerFrame\x1a\x1b.cozy.worker.v1.WorkerFrame(\x010\x01\x12P\n" +
	"\rWatchProgress\x12\x1c.cozy.worker.v1.ProgressOpen\x1a\x1f.cozy.worker.v1.AttemptProgress0\x012\xa7\n" +
	"\n" +
	"\x12RuntimePreparation\x12W\n" +
	"\fProtocolInfo\x12#.cozy.worker.v1.ProtocolInfoRequest\x1a\".cozy.worker.v1.ProtocolInfoResult\x12|\n" +
	"\x1cCheckPackageSetCompatibility\x12(.cozy.worker.v1.PreparePackageSetRequest\x1a2.cozy.worker.v1.CheckPackageSetCompatibilityResult\x12f\n" +
	"\x11PreparePackageSet\x12(.cozy.worker.v1.PreparePackageSetRequest\x1a'.cozy.worker.v1.PreparePackageSetResult\x12i\n" +
	"\x12PrepareModelSource\x12).cozy.worker.v1.PrepareModelSourceRequest\x1a(.cozy.worker.v1.PrepareModelSourceResult\x12i\n" +
	"\x12ReleaseModelSource\x12).cozy.worker.v1.ReleaseModelSourceRequest\x1a(.cozy.worker.v1.ReleaseModelSourceResult\x12f\n" +
	"\x13RetainDerivedResult\x12'.cozy.worker.v1.DerivedRetentionRequest\x1a&.cozy.worker.v1.DerivedRetentionResult\x12j\n" +
	"\x17ReleaseDerivedRetention\x12'.cozy.worker.v1.DerivedRetentionRequest\x1a&.cozy.worker.v1.DerivedRetentionResult\x12~\n" +
	"\x19ValidateWeightsCheckpoint\x120.cozy.worker.v1.ValidateWeightsCheckpointRequest\x1a/.cozy.worker.v1.ValidateWeightsCheckpointResult\x12]\n" +
	"\x0eCheckpointPage\x12%.cozy.worker.v1.CheckpointPageRequest\x1a$.cozy.worker.v1.CheckpointPageResult\x12i\n" +
	"\x12CheckpointTransfer\x12).cozy.worker.v1.CheckpointTransferRequest\x1a(.cozy.worker.v1.CheckpointTransferStatus\x12j\n" +
	"\x13PrepareLocalPackage\x12*.cozy.worker.v1.PrepareLocalPackageRequest\x1a'.cozy.worker.v1.PreparePackageSetResult\x12r\n" +
	"\x17PreparePrivatePlacement\x12..cozy.worker.v1.PreparePrivatePlacementRequest\x1a'.cozy.worker.v1.PreparePackageSetResult2\xb7\x01\n" +
	"\x0eRuntimeWeights\x12P\n" +
	"\bExchange\x12\x1e.cozy.worker.v1.WeightsHostAck\x1a .cozy.worker.v1.WeightsHostEvent(\x010\x01\x12S\n" +
	"\x06Upload\x12$.cozy.worker.v1.WeightsUploadRequest\x1a#.cozy.worker.v1.WeightsUploadResult2\x98\r\n" +
	"\aPodHost\x12W\n" +
	"\fProtocolInfo\x12#.cozy.worker.v1.ProtocolInfoRequest\x1a\".cozy.worker.v1.ProtocolInfoResult\x12Z\n" +
	"\x11PreparePackageSet\x12%.cozy.worker.v1.PreparePackageSetCall\x1a\x1c.cozy.worker.v1.PrepareEvent0\x01\x12^\n" +
	"\x13PrepareLocalPackage\x12'.cozy.worker.v1.PrepareLocalPackageCall\x1a\x1c.cozy.worker.v1.PrepareEvent0\x01\x12f\n" +
	"\x17PreparePrivatePlacement\x12+.cozy.worker.v1.PreparePrivatePlacementCall\x1a\x1c.cozy.worker.v1.PrepareEvent0\x01\x12_\n" +
	"\x0fModelSourceFile\x12#.cozy.worker.v1.ModelSourceFileCall\x1a%.cozy.worker.v1.ModelSourceFileStatus0\x01\x12a\n" +
	"\x12ModelSourcePrepare\x12&.cozy.worker.v1.ModelSourcePrepareCall\x1a#.cozy.worker.v1.ModelSourcePrepared\x12f\n" +
	"\x12ModelSourceRelease\x12&.cozy.worker.v1.ModelSourceReleaseCall\x1a(.cozy.worker.v1.ReleaseModelSourceResult\x12c\n" +
	"\x13RetainDerivedResult\x12$.cozy.worker.v1.DerivedRetentionCall\x1a&.cozy.worker.v1.DerivedRetentionResult\x12g\n" +
	"\x17ReleaseDerivedRetention\x12$.cozy.worker.v1.DerivedRetentionCall\x1a&.cozy.worker.v1.DerivedRetentionResult\x12]\n" +
	"\x10ModelSourceAdopt\x12$.cozy.worker.v1.ModelSourceAdoptCall\x1a#.cozy.worker.v1.ModelSourcePrepared\x12Z\n" +
	"\x0eCheckpointPage\x12\".cozy.worker.v1.CheckpointPageCall\x1a$.cozy.worker.v1.CheckpointPageResult\x12f\n" +
	"\x12CheckpointTransfer\x12&.cozy.worker.v1.CheckpointTransferCall\x1a(.cozy.worker.v1.CheckpointTransferStatus\x12d\n" +
	"\x11LocalPackageFetch\x12%.cozy.worker.v1.LocalPackageFetchCall\x1a&.cozy.worker.v1.LocalPackageFileStatus0\x01\x12i\n" +
	"\x12LocalPackageUpload\x12'.cozy.worker.v1.LocalPackageUploadFrame\x1a&.cozy.worker.v1.LocalPackageFileStatus(\x010\x01\x12c\n" +
	"\x11LocalPackageAbort\x12%.cozy.worker.v1.LocalPackageAbortCall\x1a'.cozy.worker.v1.LocalPackageAbortStatus\x12_\n" +
	"\x0fWeightsTransfer\x12#.cozy.worker.v1.WeightsTransferCall\x1a%.cozy.worker.v1.WeightsTransferStatus0\x01\x12\\\n" +
	"\x12WeightsIntentReady\x12&.cozy.worker.v1.WeightsIntentReadyCall\x1a\x1e.cozy.worker.v1.WeightsHostAckB4\n" +
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

var file_cozy_worker_v1_worker_proto_enumTypes = make([]protoimpl.EnumInfo, 34)
var file_cozy_worker_v1_worker_proto_msgTypes = make([]protoimpl.MessageInfo, 159)
var file_cozy_worker_v1_worker_proto_goTypes = []any{
	(ChildCallState)(0),                        // 0: cozy.worker.v1.ChildCallState
	(Posture)(0),                               // 1: cozy.worker.v1.Posture
	(WorkerPhase)(0),                           // 2: cozy.worker.v1.WorkerPhase
	(MaterializationState)(0),                  // 3: cozy.worker.v1.MaterializationState
	(ServingState)(0),                          // 4: cozy.worker.v1.ServingState
	(AdmissionState)(0),                        // 5: cozy.worker.v1.AdmissionState
	(AttemptKind)(0),                           // 6: cozy.worker.v1.AttemptKind
	(AttemptState)(0),                          // 7: cozy.worker.v1.AttemptState
	(OutcomeStatus)(0),                         // 8: cozy.worker.v1.OutcomeStatus
	(CauseCode)(0),                             // 9: cozy.worker.v1.CauseCode
	(CauseOrigin)(0),                           // 10: cozy.worker.v1.CauseOrigin
	(CancelReason)(0),                          // 11: cozy.worker.v1.CancelReason
	(ClaimRejection)(0),                        // 12: cozy.worker.v1.ClaimRejection
	(FaultKind)(0),                             // 13: cozy.worker.v1.FaultKind
	(BootFailureReason)(0),                     // 14: cozy.worker.v1.BootFailureReason
	(CheckpointOutcome)(0),                     // 15: cozy.worker.v1.CheckpointOutcome
	(CheckpointFaultCode)(0),                   // 16: cozy.worker.v1.CheckpointFaultCode
	(PrepareStage)(0),                          // 17: cozy.worker.v1.PrepareStage
	(WeightsHostStage)(0),                      // 18: cozy.worker.v1.WeightsHostStage
	(WeightsHostOutcome)(0),                    // 19: cozy.worker.v1.WeightsHostOutcome
	(WeightsHostRefusal)(0),                    // 20: cozy.worker.v1.WeightsHostRefusal
	(WeightsTransactionState)(0),               // 21: cozy.worker.v1.WeightsTransactionState
	(WeightsReadOutcome)(0),                    // 22: cozy.worker.v1.WeightsReadOutcome
	(WeightsReadRefusal)(0),                    // 23: cozy.worker.v1.WeightsReadRefusal
	(WeightsUploadOutcome)(0),                  // 24: cozy.worker.v1.WeightsUploadOutcome
	(WeightsUploadRefusal)(0),                  // 25: cozy.worker.v1.WeightsUploadRefusal
	(WeightsTransferState)(0),                  // 26: cozy.worker.v1.WeightsTransferState
	(ModelSourceProvider)(0),                   // 27: cozy.worker.v1.ModelSourceProvider
	(ModelSourceFileState)(0),                  // 28: cozy.worker.v1.ModelSourceFileState
	(ModelSourcePrepareOutcome)(0),             // 29: cozy.worker.v1.ModelSourcePrepareOutcome
	(LocalPackageFileState)(0),                 // 30: cozy.worker.v1.LocalPackageFileState
	(LocalPackageAbortOutcome)(0),              // 31: cozy.worker.v1.LocalPackageAbortOutcome
	(WeightsFinalizeDisposition)(0),            // 32: cozy.worker.v1.WeightsFinalizeDisposition
	(WeightsFinalizeOutcome)(0),                // 33: cozy.worker.v1.WeightsFinalizeOutcome
	(*WeightsHostEvent)(nil),                   // 34: cozy.worker.v1.WeightsHostEvent
	(*ProtocolInfoRequest)(nil),                // 35: cozy.worker.v1.ProtocolInfoRequest
	(*ProtocolInfoResult)(nil),                 // 36: cozy.worker.v1.ProtocolInfoResult
	(*PreparePackageSetCall)(nil),              // 37: cozy.worker.v1.PreparePackageSetCall
	(*PrepareLocalPackageCall)(nil),            // 38: cozy.worker.v1.PrepareLocalPackageCall
	(*PreparePrivatePlacementCall)(nil),        // 39: cozy.worker.v1.PreparePrivatePlacementCall
	(*PrepareEvent)(nil),                       // 40: cozy.worker.v1.PrepareEvent
	(*ModelSourceFileCall)(nil),                // 41: cozy.worker.v1.ModelSourceFileCall
	(*ModelSourcePrepareCall)(nil),             // 42: cozy.worker.v1.ModelSourcePrepareCall
	(*ModelSourceReleaseCall)(nil),             // 43: cozy.worker.v1.ModelSourceReleaseCall
	(*ReleaseModelSourceRequest)(nil),          // 44: cozy.worker.v1.ReleaseModelSourceRequest
	(*DerivedRetentionCall)(nil),               // 45: cozy.worker.v1.DerivedRetentionCall
	(*DerivedRetentionRequest)(nil),            // 46: cozy.worker.v1.DerivedRetentionRequest
	(*DerivedRetentionResult)(nil),             // 47: cozy.worker.v1.DerivedRetentionResult
	(*ReleaseModelSourceResult)(nil),           // 48: cozy.worker.v1.ReleaseModelSourceResult
	(*ModelSourceAdoptCall)(nil),               // 49: cozy.worker.v1.ModelSourceAdoptCall
	(*CheckpointPageCall)(nil),                 // 50: cozy.worker.v1.CheckpointPageCall
	(*CheckpointTransferCall)(nil),             // 51: cozy.worker.v1.CheckpointTransferCall
	(*LocalPackageFetchCall)(nil),              // 52: cozy.worker.v1.LocalPackageFetchCall
	(*LocalPackageUploadFrame)(nil),            // 53: cozy.worker.v1.LocalPackageUploadFrame
	(*LocalPackageUploadHeader)(nil),           // 54: cozy.worker.v1.LocalPackageUploadHeader
	(*LocalPackageUploadChunk)(nil),            // 55: cozy.worker.v1.LocalPackageUploadChunk
	(*LocalPackageAbortCall)(nil),              // 56: cozy.worker.v1.LocalPackageAbortCall
	(*WeightsTransferCall)(nil),                // 57: cozy.worker.v1.WeightsTransferCall
	(*PreparePackageSetRequest)(nil),           // 58: cozy.worker.v1.PreparePackageSetRequest
	(*ImageInventory)(nil),                     // 59: cozy.worker.v1.ImageInventory
	(*ImageDistribution)(nil),                  // 60: cozy.worker.v1.ImageDistribution
	(*PreparePackageSetResult)(nil),            // 61: cozy.worker.v1.PreparePackageSetResult
	(*CheckPackageSetCompatibilityResult)(nil), // 62: cozy.worker.v1.CheckPackageSetCompatibilityResult
	(*PrepareLocalPackageRequest)(nil),         // 63: cozy.worker.v1.PrepareLocalPackageRequest
	(*LocalPackageWheel)(nil),                  // 64: cozy.worker.v1.LocalPackageWheel
	(*PreparePrivatePlacementRequest)(nil),     // 65: cozy.worker.v1.PreparePrivatePlacementRequest
	(*LocalModelSourceFile)(nil),               // 66: cozy.worker.v1.LocalModelSourceFile
	(*ModelSourceProfile)(nil),                 // 67: cozy.worker.v1.ModelSourceProfile
	(*PreparedModelSource)(nil),                // 68: cozy.worker.v1.PreparedModelSource
	(*ModelSourceCheckpoint)(nil),              // 69: cozy.worker.v1.ModelSourceCheckpoint
	(*SourceCheckpointSubject)(nil),            // 70: cozy.worker.v1.SourceCheckpointSubject
	(*WeightsCheckpointSubject)(nil),           // 71: cozy.worker.v1.WeightsCheckpointSubject
	(*CheckpointSubject)(nil),                  // 72: cozy.worker.v1.CheckpointSubject
	(*CheckpointObject)(nil),                   // 73: cozy.worker.v1.CheckpointObject
	(*CheckpointPageRequest)(nil),              // 74: cozy.worker.v1.CheckpointPageRequest
	(*CheckpointPageResult)(nil),               // 75: cozy.worker.v1.CheckpointPageResult
	(*CheckpointTransferRequest)(nil),          // 76: cozy.worker.v1.CheckpointTransferRequest
	(*CheckpointTransferStatus)(nil),           // 77: cozy.worker.v1.CheckpointTransferStatus
	(*PrepareModelSourceRequest)(nil),          // 78: cozy.worker.v1.PrepareModelSourceRequest
	(*PrepareModelSourceResult)(nil),           // 79: cozy.worker.v1.PrepareModelSourceResult
	(*RecordOwnerFrame)(nil),                   // 80: cozy.worker.v1.RecordOwnerFrame
	(*WorkerFrame)(nil),                        // 81: cozy.worker.v1.WorkerFrame
	(*ChildCallRequest)(nil),                   // 82: cozy.worker.v1.ChildCallRequest
	(*ChildCallCancel)(nil),                    // 83: cozy.worker.v1.ChildCallCancel
	(*ChildCallResult)(nil),                    // 84: cozy.worker.v1.ChildCallResult
	(*Claim)(nil),                              // 85: cozy.worker.v1.Claim
	(*ClaimProof)(nil),                         // 86: cozy.worker.v1.ClaimProof
	(*ClaimAck)(nil),                           // 87: cozy.worker.v1.ClaimAck
	(*BootFailure)(nil),                        // 88: cozy.worker.v1.BootFailure
	(*WorkerSnapshot)(nil),                     // 89: cozy.worker.v1.WorkerSnapshot
	(*WorkerSnapshotBody)(nil),                 // 90: cozy.worker.v1.WorkerSnapshotBody
	(*HostSnapshotBody)(nil),                   // 91: cozy.worker.v1.HostSnapshotBody
	(*SnapshotAck)(nil),                        // 92: cozy.worker.v1.SnapshotAck
	(*DesiredWorkerState)(nil),                 // 93: cozy.worker.v1.DesiredWorkerState
	(*DesiredPackageSet)(nil),                  // 94: cozy.worker.v1.DesiredPackageSet
	(*DesiredLocalPackageSet)(nil),             // 95: cozy.worker.v1.DesiredLocalPackageSet
	(*DesiredPrivatePlacementSet)(nil),         // 96: cozy.worker.v1.DesiredPrivatePlacementSet
	(*LocalPackageFileRef)(nil),                // 97: cozy.worker.v1.LocalPackageFileRef
	(*LocalPackageRevision)(nil),               // 98: cozy.worker.v1.LocalPackageRevision
	(*DesiredPlacementSet)(nil),                // 99: cozy.worker.v1.DesiredPlacementSet
	(*PlacementDevicePin)(nil),                 // 100: cozy.worker.v1.PlacementDevicePin
	(*PlacementSet)(nil),                       // 101: cozy.worker.v1.PlacementSet
	(*DownloadDelegation)(nil),                 // 102: cozy.worker.v1.DownloadDelegation
	(*DownloadModelRef)(nil),                   // 103: cozy.worker.v1.DownloadModelRef
	(*DownloadPackageRef)(nil),                 // 104: cozy.worker.v1.DownloadPackageRef
	(*Placement)(nil),                          // 105: cozy.worker.v1.Placement
	(*Ref)(nil),                                // 106: cozy.worker.v1.Ref
	(*WheelFact)(nil),                          // 107: cozy.worker.v1.WheelFact
	(*PackageSelection)(nil),                   // 108: cozy.worker.v1.PackageSelection
	(*DevelopmentPackage)(nil),                 // 109: cozy.worker.v1.DevelopmentPackage
	(*Environment)(nil),                        // 110: cozy.worker.v1.Environment
	(*Model)(nil),                              // 111: cozy.worker.v1.Model
	(*Entrypoint)(nil),                         // 112: cozy.worker.v1.Entrypoint
	(*Slot)(nil),                               // 113: cozy.worker.v1.Slot
	(*Component)(nil),                          // 114: cozy.worker.v1.Component
	(*Stamp)(nil),                              // 115: cozy.worker.v1.Stamp
	(*JobDirective)(nil),                       // 116: cozy.worker.v1.JobDirective
	(*ObservedWorkerState)(nil),                // 117: cozy.worker.v1.ObservedWorkerState
	(*DeviceLane)(nil),                         // 118: cozy.worker.v1.DeviceLane
	(*PlacementStatus)(nil),                    // 119: cozy.worker.v1.PlacementStatus
	(*PlacementAcquisitionObservation)(nil),    // 120: cozy.worker.v1.PlacementAcquisitionObservation
	(*AcquisitionLegObservation)(nil),          // 121: cozy.worker.v1.AcquisitionLegObservation
	(*AcceleratorQualification)(nil),           // 122: cozy.worker.v1.AcceleratorQualification
	(*ActivityEvent)(nil),                      // 123: cozy.worker.v1.ActivityEvent
	(*AttemptOffer)(nil),                       // 124: cozy.worker.v1.AttemptOffer
	(*AttemptAccepted)(nil),                    // 125: cozy.worker.v1.AttemptAccepted
	(*AttemptPlanSummary)(nil),                 // 126: cozy.worker.v1.AttemptPlanSummary
	(*CancelAttempt)(nil),                      // 127: cozy.worker.v1.CancelAttempt
	(*AttemptOutcome)(nil),                     // 128: cozy.worker.v1.AttemptOutcome
	(*AttemptOutcomeBody)(nil),                 // 129: cozy.worker.v1.AttemptOutcomeBody
	(*WeightsReceiptRef)(nil),                  // 130: cozy.worker.v1.WeightsReceiptRef
	(*WeightsReceipt)(nil),                     // 131: cozy.worker.v1.WeightsReceipt
	(*WeightsObjectSource)(nil),                // 132: cozy.worker.v1.WeightsObjectSource
	(*WeightsIntentFrame)(nil),                 // 133: cozy.worker.v1.WeightsIntentFrame
	(*WeightsHostAck)(nil),                     // 134: cozy.worker.v1.WeightsHostAck
	(*CheckpointRef)(nil),                      // 135: cozy.worker.v1.CheckpointRef
	(*WeightsCheckpointFrame)(nil),             // 136: cozy.worker.v1.WeightsCheckpointFrame
	(*WeightsIntentReadyRequest)(nil),          // 137: cozy.worker.v1.WeightsIntentReadyRequest
	(*WeightsIntentReadyCall)(nil),             // 138: cozy.worker.v1.WeightsIntentReadyCall
	(*ValidateWeightsCheckpointRequest)(nil),   // 139: cozy.worker.v1.ValidateWeightsCheckpointRequest
	(*ValidateWeightsCheckpointResult)(nil),    // 140: cozy.worker.v1.ValidateWeightsCheckpointResult
	(*WeightsReceiptFrame)(nil),                // 141: cozy.worker.v1.WeightsReceiptFrame
	(*WeightsTransactionStatus)(nil),           // 142: cozy.worker.v1.WeightsTransactionStatus
	(*WeightsReadRequest)(nil),                 // 143: cozy.worker.v1.WeightsReadRequest
	(*WeightsReadResult)(nil),                  // 144: cozy.worker.v1.WeightsReadResult
	(*WeightsObjectRef)(nil),                   // 145: cozy.worker.v1.WeightsObjectRef
	(*WeightsUploadHeader)(nil),                // 146: cozy.worker.v1.WeightsUploadHeader
	(*WeightsUploadGrant)(nil),                 // 147: cozy.worker.v1.WeightsUploadGrant
	(*WeightsTransferRequest)(nil),             // 148: cozy.worker.v1.WeightsTransferRequest
	(*WeightsUploadRequest)(nil),               // 149: cozy.worker.v1.WeightsUploadRequest
	(*WeightsUploadResult)(nil),                // 150: cozy.worker.v1.WeightsUploadResult
	(*WeightsTransferStatus)(nil),              // 151: cozy.worker.v1.WeightsTransferStatus
	(*ModelSourceFileRequest)(nil),             // 152: cozy.worker.v1.ModelSourceFileRequest
	(*ModelSourceFileStatus)(nil),              // 153: cozy.worker.v1.ModelSourceFileStatus
	(*ModelSourcePrepareRequest)(nil),          // 154: cozy.worker.v1.ModelSourcePrepareRequest
	(*ModelSourcePrepared)(nil),                // 155: cozy.worker.v1.ModelSourcePrepared
	(*LocalPackageFileGrant)(nil),              // 156: cozy.worker.v1.LocalPackageFileGrant
	(*LocalPackageFetchRequest)(nil),           // 157: cozy.worker.v1.LocalPackageFetchRequest
	(*LocalPackageFileStatus)(nil),             // 158: cozy.worker.v1.LocalPackageFileStatus
	(*LocalPackageAbort)(nil),                  // 159: cozy.worker.v1.LocalPackageAbort
	(*LocalPackageAbortStatus)(nil),            // 160: cozy.worker.v1.LocalPackageAbortStatus
	(*WeightsFinalizeRequest)(nil),             // 161: cozy.worker.v1.WeightsFinalizeRequest
	(*WeightsFinalizeResult)(nil),              // 162: cozy.worker.v1.WeightsFinalizeResult
	(*ResultEnvelope)(nil),                     // 163: cozy.worker.v1.ResultEnvelope
	(*AdjustmentRow)(nil),                      // 164: cozy.worker.v1.AdjustmentRow
	(*OutcomeCause)(nil),                       // 165: cozy.worker.v1.OutcomeCause
	(*ResourceShortfall)(nil),                  // 166: cozy.worker.v1.ResourceShortfall
	(*AttemptOutcomeAck)(nil),                  // 167: cozy.worker.v1.AttemptOutcomeAck
	(*JobCheckpointRequest)(nil),               // 168: cozy.worker.v1.JobCheckpointRequest
	(*JobCheckpointReceipt)(nil),               // 169: cozy.worker.v1.JobCheckpointReceipt
	(*CheckpointFault)(nil),                    // 170: cozy.worker.v1.CheckpointFault
	(*JobCheckpointAck)(nil),                   // 171: cozy.worker.v1.JobCheckpointAck
	(*ProgressOpen)(nil),                       // 172: cozy.worker.v1.ProgressOpen
	(*AttemptProgress)(nil),                    // 173: cozy.worker.v1.AttemptProgress
	(*InvocationSpec)(nil),                     // 174: cozy.worker.v1.InvocationSpec
	(*InputBinding)(nil),                       // 175: cozy.worker.v1.InputBinding
	(*OutputBinding)(nil),                      // 176: cozy.worker.v1.OutputBinding
	(*ServingInvocationSpec)(nil),              // 177: cozy.worker.v1.ServingInvocationSpec
	(*JobInvocationSpec)(nil),                  // 178: cozy.worker.v1.JobInvocationSpec
	(*DeliveryGrant)(nil),                      // 179: cozy.worker.v1.DeliveryGrant
	(*InputAccess)(nil),                        // 180: cozy.worker.v1.InputAccess
	(*OutputAccess)(nil),                       // 181: cozy.worker.v1.OutputAccess
	(*DeliveryAccessCredential)(nil),           // 182: cozy.worker.v1.DeliveryAccessCredential
	(*ResourceCaps)(nil),                       // 183: cozy.worker.v1.ResourceCaps
	(*PublicationContract)(nil),                // 184: cozy.worker.v1.PublicationContract
	(*WorkerResources)(nil),                    // 185: cozy.worker.v1.WorkerResources
	(*JobCapacity)(nil),                        // 186: cozy.worker.v1.JobCapacity
	(*HeldAttempt)(nil),                        // 187: cozy.worker.v1.HeldAttempt
	(*Fault)(nil),                              // 188: cozy.worker.v1.Fault
	(*OutputManifest)(nil),                     // 189: cozy.worker.v1.OutputManifest
	(*OutputEntry)(nil),                        // 190: cozy.worker.v1.OutputEntry
	(*AttemptMetrics)(nil),                     // 191: cozy.worker.v1.AttemptMetrics
	(*TriageBundleRef)(nil),                    // 192: cozy.worker.v1.TriageBundleRef
}
var file_cozy_worker_v1_worker_proto_depIdxs = []int32{
	133, // 0: cozy.worker.v1.WeightsHostEvent.intent:type_name -> cozy.worker.v1.WeightsIntentFrame
	141, // 1: cozy.worker.v1.WeightsHostEvent.receipt:type_name -> cozy.worker.v1.WeightsReceiptFrame
	136, // 2: cozy.worker.v1.WeightsHostEvent.checkpoint:type_name -> cozy.worker.v1.WeightsCheckpointFrame
	85,  // 3: cozy.worker.v1.PreparePackageSetCall.claim:type_name -> cozy.worker.v1.Claim
	94,  // 4: cozy.worker.v1.PreparePackageSetCall.package_set:type_name -> cozy.worker.v1.DesiredPackageSet
	59,  // 5: cozy.worker.v1.PreparePackageSetCall.image_inventory:type_name -> cozy.worker.v1.ImageInventory
	85,  // 6: cozy.worker.v1.PrepareLocalPackageCall.claim:type_name -> cozy.worker.v1.Claim
	95,  // 7: cozy.worker.v1.PrepareLocalPackageCall.local_package_set:type_name -> cozy.worker.v1.DesiredLocalPackageSet
	85,  // 8: cozy.worker.v1.PreparePrivatePlacementCall.claim:type_name -> cozy.worker.v1.Claim
	96,  // 9: cozy.worker.v1.PreparePrivatePlacementCall.private_placement_set:type_name -> cozy.worker.v1.DesiredPrivatePlacementSet
	17,  // 10: cozy.worker.v1.PrepareEvent.stage:type_name -> cozy.worker.v1.PrepareStage
	99,  // 11: cozy.worker.v1.PrepareEvent.placement_set:type_name -> cozy.worker.v1.DesiredPlacementSet
	85,  // 12: cozy.worker.v1.ModelSourceFileCall.claim:type_name -> cozy.worker.v1.Claim
	152, // 13: cozy.worker.v1.ModelSourceFileCall.request:type_name -> cozy.worker.v1.ModelSourceFileRequest
	85,  // 14: cozy.worker.v1.ModelSourcePrepareCall.claim:type_name -> cozy.worker.v1.Claim
	154, // 15: cozy.worker.v1.ModelSourcePrepareCall.request:type_name -> cozy.worker.v1.ModelSourcePrepareRequest
	85,  // 16: cozy.worker.v1.ModelSourceReleaseCall.claim:type_name -> cozy.worker.v1.Claim
	85,  // 17: cozy.worker.v1.DerivedRetentionCall.claim:type_name -> cozy.worker.v1.Claim
	46,  // 18: cozy.worker.v1.DerivedRetentionCall.request:type_name -> cozy.worker.v1.DerivedRetentionRequest
	106, // 19: cozy.worker.v1.DerivedRetentionResult.manifest:type_name -> cozy.worker.v1.Ref
	85,  // 20: cozy.worker.v1.ModelSourceAdoptCall.claim:type_name -> cozy.worker.v1.Claim
	154, // 21: cozy.worker.v1.ModelSourceAdoptCall.request:type_name -> cozy.worker.v1.ModelSourcePrepareRequest
	85,  // 22: cozy.worker.v1.CheckpointPageCall.claim:type_name -> cozy.worker.v1.Claim
	74,  // 23: cozy.worker.v1.CheckpointPageCall.request:type_name -> cozy.worker.v1.CheckpointPageRequest
	85,  // 24: cozy.worker.v1.CheckpointTransferCall.claim:type_name -> cozy.worker.v1.Claim
	76,  // 25: cozy.worker.v1.CheckpointTransferCall.request:type_name -> cozy.worker.v1.CheckpointTransferRequest
	85,  // 26: cozy.worker.v1.LocalPackageFetchCall.claim:type_name -> cozy.worker.v1.Claim
	157, // 27: cozy.worker.v1.LocalPackageFetchCall.request:type_name -> cozy.worker.v1.LocalPackageFetchRequest
	54,  // 28: cozy.worker.v1.LocalPackageUploadFrame.header:type_name -> cozy.worker.v1.LocalPackageUploadHeader
	55,  // 29: cozy.worker.v1.LocalPackageUploadFrame.chunk:type_name -> cozy.worker.v1.LocalPackageUploadChunk
	85,  // 30: cozy.worker.v1.LocalPackageUploadHeader.claim:type_name -> cozy.worker.v1.Claim
	97,  // 31: cozy.worker.v1.LocalPackageUploadHeader.file:type_name -> cozy.worker.v1.LocalPackageFileRef
	85,  // 32: cozy.worker.v1.LocalPackageAbortCall.claim:type_name -> cozy.worker.v1.Claim
	159, // 33: cozy.worker.v1.LocalPackageAbortCall.request:type_name -> cozy.worker.v1.LocalPackageAbort
	85,  // 34: cozy.worker.v1.WeightsTransferCall.claim:type_name -> cozy.worker.v1.Claim
	148, // 35: cozy.worker.v1.WeightsTransferCall.request:type_name -> cozy.worker.v1.WeightsTransferRequest
	59,  // 36: cozy.worker.v1.PreparePackageSetRequest.image_inventory:type_name -> cozy.worker.v1.ImageInventory
	60,  // 37: cozy.worker.v1.ImageInventory.distributions:type_name -> cozy.worker.v1.ImageDistribution
	99,  // 38: cozy.worker.v1.PreparePackageSetResult.placement_set:type_name -> cozy.worker.v1.DesiredPlacementSet
	109, // 39: cozy.worker.v1.PrepareLocalPackageRequest.package:type_name -> cozy.worker.v1.DevelopmentPackage
	64,  // 40: cozy.worker.v1.PrepareLocalPackageRequest.wheels:type_name -> cozy.worker.v1.LocalPackageWheel
	106, // 41: cozy.worker.v1.PreparedModelSource.manifest:type_name -> cozy.worker.v1.Ref
	106, // 42: cozy.worker.v1.ModelSourceCheckpoint.head:type_name -> cozy.worker.v1.Ref
	70,  // 43: cozy.worker.v1.CheckpointSubject.source:type_name -> cozy.worker.v1.SourceCheckpointSubject
	71,  // 44: cozy.worker.v1.CheckpointSubject.weights:type_name -> cozy.worker.v1.WeightsCheckpointSubject
	106, // 45: cozy.worker.v1.CheckpointObject.ref:type_name -> cozy.worker.v1.Ref
	72,  // 46: cozy.worker.v1.CheckpointPageRequest.subject:type_name -> cozy.worker.v1.CheckpointSubject
	106, // 47: cozy.worker.v1.CheckpointPageRequest.head:type_name -> cozy.worker.v1.Ref
	72,  // 48: cozy.worker.v1.CheckpointPageResult.subject:type_name -> cozy.worker.v1.CheckpointSubject
	106, // 49: cozy.worker.v1.CheckpointPageResult.head:type_name -> cozy.worker.v1.Ref
	106, // 50: cozy.worker.v1.CheckpointPageResult.previous:type_name -> cozy.worker.v1.Ref
	106, // 51: cozy.worker.v1.CheckpointPageResult.progress:type_name -> cozy.worker.v1.Ref
	73,  // 52: cozy.worker.v1.CheckpointPageResult.objects:type_name -> cozy.worker.v1.CheckpointObject
	72,  // 53: cozy.worker.v1.CheckpointTransferRequest.subject:type_name -> cozy.worker.v1.CheckpointSubject
	106, // 54: cozy.worker.v1.CheckpointTransferRequest.head:type_name -> cozy.worker.v1.Ref
	73,  // 55: cozy.worker.v1.CheckpointTransferRequest.object:type_name -> cozy.worker.v1.CheckpointObject
	147, // 56: cozy.worker.v1.CheckpointTransferRequest.upload_grant:type_name -> cozy.worker.v1.WeightsUploadGrant
	72,  // 57: cozy.worker.v1.CheckpointTransferStatus.subject:type_name -> cozy.worker.v1.CheckpointSubject
	106, // 58: cozy.worker.v1.CheckpointTransferStatus.head:type_name -> cozy.worker.v1.Ref
	73,  // 59: cozy.worker.v1.CheckpointTransferStatus.object:type_name -> cozy.worker.v1.CheckpointObject
	26,  // 60: cozy.worker.v1.CheckpointTransferStatus.state:type_name -> cozy.worker.v1.WeightsTransferState
	67,  // 61: cozy.worker.v1.PrepareModelSourceRequest.profiles:type_name -> cozy.worker.v1.ModelSourceProfile
	66,  // 62: cozy.worker.v1.PrepareModelSourceRequest.files:type_name -> cozy.worker.v1.LocalModelSourceFile
	69,  // 63: cozy.worker.v1.PrepareModelSourceRequest.checkpoints:type_name -> cozy.worker.v1.ModelSourceCheckpoint
	29,  // 64: cozy.worker.v1.PrepareModelSourceResult.outcome:type_name -> cozy.worker.v1.ModelSourcePrepareOutcome
	68,  // 65: cozy.worker.v1.PrepareModelSourceResult.sources:type_name -> cozy.worker.v1.PreparedModelSource
	69,  // 66: cozy.worker.v1.PrepareModelSourceResult.checkpoints:type_name -> cozy.worker.v1.ModelSourceCheckpoint
	85,  // 67: cozy.worker.v1.RecordOwnerFrame.claim:type_name -> cozy.worker.v1.Claim
	93,  // 68: cozy.worker.v1.RecordOwnerFrame.desired_state:type_name -> cozy.worker.v1.DesiredWorkerState
	124, // 69: cozy.worker.v1.RecordOwnerFrame.attempt_offer:type_name -> cozy.worker.v1.AttemptOffer
	127, // 70: cozy.worker.v1.RecordOwnerFrame.cancel_attempt:type_name -> cozy.worker.v1.CancelAttempt
	167, // 71: cozy.worker.v1.RecordOwnerFrame.outcome_ack:type_name -> cozy.worker.v1.AttemptOutcomeAck
	169, // 72: cozy.worker.v1.RecordOwnerFrame.checkpoint_receipt:type_name -> cozy.worker.v1.JobCheckpointReceipt
	92,  // 73: cozy.worker.v1.RecordOwnerFrame.snapshot_ack:type_name -> cozy.worker.v1.SnapshotAck
	161, // 74: cozy.worker.v1.RecordOwnerFrame.weights_finalize_request:type_name -> cozy.worker.v1.WeightsFinalizeRequest
	134, // 75: cozy.worker.v1.RecordOwnerFrame.weights_host_ack:type_name -> cozy.worker.v1.WeightsHostAck
	148, // 76: cozy.worker.v1.RecordOwnerFrame.weights_transfer_request:type_name -> cozy.worker.v1.WeightsTransferRequest
	143, // 77: cozy.worker.v1.RecordOwnerFrame.weights_read_request:type_name -> cozy.worker.v1.WeightsReadRequest
	152, // 78: cozy.worker.v1.RecordOwnerFrame.model_source_file_request:type_name -> cozy.worker.v1.ModelSourceFileRequest
	154, // 79: cozy.worker.v1.RecordOwnerFrame.model_source_prepare_request:type_name -> cozy.worker.v1.ModelSourcePrepareRequest
	159, // 80: cozy.worker.v1.RecordOwnerFrame.local_package_abort:type_name -> cozy.worker.v1.LocalPackageAbort
	149, // 81: cozy.worker.v1.RecordOwnerFrame.weights_upload_request:type_name -> cozy.worker.v1.WeightsUploadRequest
	157, // 82: cozy.worker.v1.RecordOwnerFrame.local_package_fetch_request:type_name -> cozy.worker.v1.LocalPackageFetchRequest
	84,  // 83: cozy.worker.v1.RecordOwnerFrame.child_call_result:type_name -> cozy.worker.v1.ChildCallResult
	87,  // 84: cozy.worker.v1.WorkerFrame.claim_ack:type_name -> cozy.worker.v1.ClaimAck
	117, // 85: cozy.worker.v1.WorkerFrame.observed_state:type_name -> cozy.worker.v1.ObservedWorkerState
	125, // 86: cozy.worker.v1.WorkerFrame.attempt_accepted:type_name -> cozy.worker.v1.AttemptAccepted
	128, // 87: cozy.worker.v1.WorkerFrame.attempt_outcome:type_name -> cozy.worker.v1.AttemptOutcome
	88,  // 88: cozy.worker.v1.WorkerFrame.boot_failure:type_name -> cozy.worker.v1.BootFailure
	168, // 89: cozy.worker.v1.WorkerFrame.checkpoint_request:type_name -> cozy.worker.v1.JobCheckpointRequest
	171, // 90: cozy.worker.v1.WorkerFrame.checkpoint_ack:type_name -> cozy.worker.v1.JobCheckpointAck
	89,  // 91: cozy.worker.v1.WorkerFrame.snapshot:type_name -> cozy.worker.v1.WorkerSnapshot
	162, // 92: cozy.worker.v1.WorkerFrame.weights_finalize_result:type_name -> cozy.worker.v1.WeightsFinalizeResult
	133, // 93: cozy.worker.v1.WorkerFrame.weights_intent:type_name -> cozy.worker.v1.WeightsIntentFrame
	141, // 94: cozy.worker.v1.WorkerFrame.weights_receipt:type_name -> cozy.worker.v1.WeightsReceiptFrame
	144, // 95: cozy.worker.v1.WorkerFrame.weights_read_result:type_name -> cozy.worker.v1.WeightsReadResult
	151, // 96: cozy.worker.v1.WorkerFrame.weights_transfer_status:type_name -> cozy.worker.v1.WeightsTransferStatus
	153, // 97: cozy.worker.v1.WorkerFrame.model_source_file_status:type_name -> cozy.worker.v1.ModelSourceFileStatus
	155, // 98: cozy.worker.v1.WorkerFrame.model_source_prepared:type_name -> cozy.worker.v1.ModelSourcePrepared
	158, // 99: cozy.worker.v1.WorkerFrame.local_package_file_status:type_name -> cozy.worker.v1.LocalPackageFileStatus
	160, // 100: cozy.worker.v1.WorkerFrame.local_package_abort_status:type_name -> cozy.worker.v1.LocalPackageAbortStatus
	150, // 101: cozy.worker.v1.WorkerFrame.weights_upload_result:type_name -> cozy.worker.v1.WeightsUploadResult
	136, // 102: cozy.worker.v1.WorkerFrame.weights_checkpoint:type_name -> cozy.worker.v1.WeightsCheckpointFrame
	142, // 103: cozy.worker.v1.WorkerFrame.weights_transaction:type_name -> cozy.worker.v1.WeightsTransactionStatus
	82,  // 104: cozy.worker.v1.WorkerFrame.child_call_request:type_name -> cozy.worker.v1.ChildCallRequest
	83,  // 105: cozy.worker.v1.WorkerFrame.child_call_cancel:type_name -> cozy.worker.v1.ChildCallCancel
	0,   // 106: cozy.worker.v1.ChildCallResult.state:type_name -> cozy.worker.v1.ChildCallState
	12,  // 107: cozy.worker.v1.ClaimAck.rejection:type_name -> cozy.worker.v1.ClaimRejection
	185, // 108: cozy.worker.v1.ClaimAck.resources:type_name -> cozy.worker.v1.WorkerResources
	14,  // 109: cozy.worker.v1.BootFailure.reason:type_name -> cozy.worker.v1.BootFailureReason
	185, // 110: cozy.worker.v1.BootFailure.resources:type_name -> cozy.worker.v1.WorkerResources
	2,   // 111: cozy.worker.v1.WorkerSnapshotBody.worker_phase:type_name -> cozy.worker.v1.WorkerPhase
	119, // 112: cozy.worker.v1.WorkerSnapshotBody.placements:type_name -> cozy.worker.v1.PlacementStatus
	5,   // 113: cozy.worker.v1.WorkerSnapshotBody.admission_state:type_name -> cozy.worker.v1.AdmissionState
	187, // 114: cozy.worker.v1.WorkerSnapshotBody.held_attempts:type_name -> cozy.worker.v1.HeldAttempt
	142, // 115: cozy.worker.v1.WorkerSnapshotBody.weights_transactions:type_name -> cozy.worker.v1.WeightsTransactionStatus
	118, // 116: cozy.worker.v1.WorkerSnapshotBody.lanes:type_name -> cozy.worker.v1.DeviceLane
	187, // 117: cozy.worker.v1.HostSnapshotBody.held_outcomes:type_name -> cozy.worker.v1.HeldAttempt
	142, // 118: cozy.worker.v1.HostSnapshotBody.weights_transactions:type_name -> cozy.worker.v1.WeightsTransactionStatus
	1,   // 119: cozy.worker.v1.DesiredWorkerState.posture:type_name -> cozy.worker.v1.Posture
	116, // 120: cozy.worker.v1.DesiredWorkerState.job:type_name -> cozy.worker.v1.JobDirective
	99,  // 121: cozy.worker.v1.DesiredWorkerState.placement_set:type_name -> cozy.worker.v1.DesiredPlacementSet
	94,  // 122: cozy.worker.v1.DesiredWorkerState.package_set:type_name -> cozy.worker.v1.DesiredPackageSet
	95,  // 123: cozy.worker.v1.DesiredWorkerState.local_package_set:type_name -> cozy.worker.v1.DesiredLocalPackageSet
	96,  // 124: cozy.worker.v1.DesiredWorkerState.private_placement_set:type_name -> cozy.worker.v1.DesiredPrivatePlacementSet
	109, // 125: cozy.worker.v1.DesiredLocalPackageSet.package:type_name -> cozy.worker.v1.DevelopmentPackage
	97,  // 126: cozy.worker.v1.DesiredLocalPackageSet.files:type_name -> cozy.worker.v1.LocalPackageFileRef
	106, // 127: cozy.worker.v1.LocalPackageRevision.package_interface:type_name -> cozy.worker.v1.Ref
	97,  // 128: cozy.worker.v1.LocalPackageRevision.files:type_name -> cozy.worker.v1.LocalPackageFileRef
	100, // 129: cozy.worker.v1.DesiredPlacementSet.device_pins:type_name -> cozy.worker.v1.PlacementDevicePin
	105, // 130: cozy.worker.v1.PlacementSet.placements:type_name -> cozy.worker.v1.Placement
	103, // 131: cozy.worker.v1.DownloadDelegation.models:type_name -> cozy.worker.v1.DownloadModelRef
	104, // 132: cozy.worker.v1.DownloadDelegation.packages:type_name -> cozy.worker.v1.DownloadPackageRef
	108, // 133: cozy.worker.v1.Placement.package:type_name -> cozy.worker.v1.PackageSelection
	109, // 134: cozy.worker.v1.Placement.development:type_name -> cozy.worker.v1.DevelopmentPackage
	106, // 135: cozy.worker.v1.Placement.package_interface:type_name -> cozy.worker.v1.Ref
	111, // 136: cozy.worker.v1.Placement.models:type_name -> cozy.worker.v1.Model
	112, // 137: cozy.worker.v1.Placement.entrypoints:type_name -> cozy.worker.v1.Entrypoint
	110, // 138: cozy.worker.v1.Placement.environment:type_name -> cozy.worker.v1.Environment
	106, // 139: cozy.worker.v1.WheelFact.ref:type_name -> cozy.worker.v1.Ref
	107, // 140: cozy.worker.v1.DevelopmentPackage.project_wheel:type_name -> cozy.worker.v1.WheelFact
	106, // 141: cozy.worker.v1.Environment.locked_requirements:type_name -> cozy.worker.v1.Ref
	107, // 142: cozy.worker.v1.Environment.local_wheels:type_name -> cozy.worker.v1.WheelFact
	106, // 143: cozy.worker.v1.Model.manifest:type_name -> cozy.worker.v1.Ref
	113, // 144: cozy.worker.v1.Entrypoint.slots:type_name -> cozy.worker.v1.Slot
	114, // 145: cozy.worker.v1.Slot.components:type_name -> cozy.worker.v1.Component
	106, // 146: cozy.worker.v1.Slot.model_construction_contract:type_name -> cozy.worker.v1.Ref
	115, // 147: cozy.worker.v1.Slot.stamps:type_name -> cozy.worker.v1.Stamp
	183, // 148: cozy.worker.v1.JobDirective.resource_caps:type_name -> cozy.worker.v1.ResourceCaps
	184, // 149: cozy.worker.v1.JobDirective.publication_contract:type_name -> cozy.worker.v1.PublicationContract
	116, // 150: cozy.worker.v1.JobDirective.orchestration_parent:type_name -> cozy.worker.v1.JobDirective
	187, // 151: cozy.worker.v1.ObservedWorkerState.held_attempts:type_name -> cozy.worker.v1.HeldAttempt
	188, // 152: cozy.worker.v1.ObservedWorkerState.faults:type_name -> cozy.worker.v1.Fault
	123, // 153: cozy.worker.v1.ObservedWorkerState.activity:type_name -> cozy.worker.v1.ActivityEvent
	186, // 154: cozy.worker.v1.ObservedWorkerState.job_capacity:type_name -> cozy.worker.v1.JobCapacity
	119, // 155: cozy.worker.v1.ObservedWorkerState.placements:type_name -> cozy.worker.v1.PlacementStatus
	5,   // 156: cozy.worker.v1.ObservedWorkerState.admission_state:type_name -> cozy.worker.v1.AdmissionState
	2,   // 157: cozy.worker.v1.ObservedWorkerState.worker_phase:type_name -> cozy.worker.v1.WorkerPhase
	118, // 158: cozy.worker.v1.ObservedWorkerState.lanes:type_name -> cozy.worker.v1.DeviceLane
	188, // 159: cozy.worker.v1.PlacementStatus.faults:type_name -> cozy.worker.v1.Fault
	122, // 160: cozy.worker.v1.PlacementStatus.accelerator:type_name -> cozy.worker.v1.AcceleratorQualification
	3,   // 161: cozy.worker.v1.PlacementStatus.materialization:type_name -> cozy.worker.v1.MaterializationState
	4,   // 162: cozy.worker.v1.PlacementStatus.serving:type_name -> cozy.worker.v1.ServingState
	120, // 163: cozy.worker.v1.PlacementStatus.acquisition:type_name -> cozy.worker.v1.PlacementAcquisitionObservation
	121, // 164: cozy.worker.v1.PlacementAcquisitionObservation.package:type_name -> cozy.worker.v1.AcquisitionLegObservation
	121, // 165: cozy.worker.v1.PlacementAcquisitionObservation.model:type_name -> cozy.worker.v1.AcquisitionLegObservation
	179, // 166: cozy.worker.v1.AttemptOffer.grant:type_name -> cozy.worker.v1.DeliveryGrant
	11,  // 167: cozy.worker.v1.CancelAttempt.reason:type_name -> cozy.worker.v1.CancelReason
	8,   // 168: cozy.worker.v1.AttemptOutcomeBody.status:type_name -> cozy.worker.v1.OutcomeStatus
	189, // 169: cozy.worker.v1.AttemptOutcomeBody.output_manifest:type_name -> cozy.worker.v1.OutputManifest
	191, // 170: cozy.worker.v1.AttemptOutcomeBody.metrics:type_name -> cozy.worker.v1.AttemptMetrics
	192, // 171: cozy.worker.v1.AttemptOutcomeBody.triage_bundle:type_name -> cozy.worker.v1.TriageBundleRef
	165, // 172: cozy.worker.v1.AttemptOutcomeBody.cause:type_name -> cozy.worker.v1.OutcomeCause
	163, // 173: cozy.worker.v1.AttemptOutcomeBody.result:type_name -> cozy.worker.v1.ResultEnvelope
	130, // 174: cozy.worker.v1.AttemptOutcomeBody.weights_receipts:type_name -> cozy.worker.v1.WeightsReceiptRef
	18,  // 175: cozy.worker.v1.WeightsHostAck.stage:type_name -> cozy.worker.v1.WeightsHostStage
	19,  // 176: cozy.worker.v1.WeightsHostAck.outcome:type_name -> cozy.worker.v1.WeightsHostOutcome
	20,  // 177: cozy.worker.v1.WeightsHostAck.refusal:type_name -> cozy.worker.v1.WeightsHostRefusal
	130, // 178: cozy.worker.v1.WeightsHostAck.weights_receipt:type_name -> cozy.worker.v1.WeightsReceiptRef
	106, // 179: cozy.worker.v1.WeightsHostAck.manifest:type_name -> cozy.worker.v1.Ref
	135, // 180: cozy.worker.v1.WeightsHostAck.checkpoint:type_name -> cozy.worker.v1.CheckpointRef
	106, // 181: cozy.worker.v1.CheckpointRef.head:type_name -> cozy.worker.v1.Ref
	135, // 182: cozy.worker.v1.WeightsCheckpointFrame.checkpoint:type_name -> cozy.worker.v1.CheckpointRef
	71,  // 183: cozy.worker.v1.WeightsIntentReadyRequest.weights:type_name -> cozy.worker.v1.WeightsCheckpointSubject
	135, // 184: cozy.worker.v1.WeightsIntentReadyRequest.checkpoint:type_name -> cozy.worker.v1.CheckpointRef
	85,  // 185: cozy.worker.v1.WeightsIntentReadyCall.claim:type_name -> cozy.worker.v1.Claim
	137, // 186: cozy.worker.v1.WeightsIntentReadyCall.request:type_name -> cozy.worker.v1.WeightsIntentReadyRequest
	137, // 187: cozy.worker.v1.ValidateWeightsCheckpointRequest.intent:type_name -> cozy.worker.v1.WeightsIntentReadyRequest
	71,  // 188: cozy.worker.v1.ValidateWeightsCheckpointResult.weights:type_name -> cozy.worker.v1.WeightsCheckpointSubject
	135, // 189: cozy.worker.v1.ValidateWeightsCheckpointResult.checkpoint:type_name -> cozy.worker.v1.CheckpointRef
	130, // 190: cozy.worker.v1.WeightsReceiptFrame.weights_receipt:type_name -> cozy.worker.v1.WeightsReceiptRef
	132, // 191: cozy.worker.v1.WeightsReceiptFrame.objects:type_name -> cozy.worker.v1.WeightsObjectSource
	106, // 192: cozy.worker.v1.WeightsReceiptFrame.manifest:type_name -> cozy.worker.v1.Ref
	21,  // 193: cozy.worker.v1.WeightsTransactionStatus.state:type_name -> cozy.worker.v1.WeightsTransactionState
	135, // 194: cozy.worker.v1.WeightsTransactionStatus.checkpoint:type_name -> cozy.worker.v1.CheckpointRef
	22,  // 195: cozy.worker.v1.WeightsReadResult.outcome:type_name -> cozy.worker.v1.WeightsReadOutcome
	23,  // 196: cozy.worker.v1.WeightsReadResult.refusal:type_name -> cozy.worker.v1.WeightsReadRefusal
	146, // 197: cozy.worker.v1.WeightsUploadGrant.required_headers:type_name -> cozy.worker.v1.WeightsUploadHeader
	147, // 198: cozy.worker.v1.WeightsTransferRequest.upload_grant:type_name -> cozy.worker.v1.WeightsUploadGrant
	145, // 199: cozy.worker.v1.WeightsTransferRequest.held:type_name -> cozy.worker.v1.WeightsObjectRef
	147, // 200: cozy.worker.v1.WeightsUploadRequest.grant:type_name -> cozy.worker.v1.WeightsUploadGrant
	24,  // 201: cozy.worker.v1.WeightsUploadResult.outcome:type_name -> cozy.worker.v1.WeightsUploadOutcome
	25,  // 202: cozy.worker.v1.WeightsUploadResult.refusal:type_name -> cozy.worker.v1.WeightsUploadRefusal
	26,  // 203: cozy.worker.v1.WeightsTransferStatus.state:type_name -> cozy.worker.v1.WeightsTransferState
	27,  // 204: cozy.worker.v1.ModelSourceFileRequest.provider:type_name -> cozy.worker.v1.ModelSourceProvider
	28,  // 205: cozy.worker.v1.ModelSourceFileStatus.state:type_name -> cozy.worker.v1.ModelSourceFileState
	67,  // 206: cozy.worker.v1.ModelSourcePrepareRequest.profiles:type_name -> cozy.worker.v1.ModelSourceProfile
	69,  // 207: cozy.worker.v1.ModelSourcePrepareRequest.checkpoints:type_name -> cozy.worker.v1.ModelSourceCheckpoint
	29,  // 208: cozy.worker.v1.ModelSourcePrepared.outcome:type_name -> cozy.worker.v1.ModelSourcePrepareOutcome
	68,  // 209: cozy.worker.v1.ModelSourcePrepared.sources:type_name -> cozy.worker.v1.PreparedModelSource
	69,  // 210: cozy.worker.v1.ModelSourcePrepared.checkpoints:type_name -> cozy.worker.v1.ModelSourceCheckpoint
	156, // 211: cozy.worker.v1.LocalPackageFetchRequest.files:type_name -> cozy.worker.v1.LocalPackageFileGrant
	30,  // 212: cozy.worker.v1.LocalPackageFileStatus.state:type_name -> cozy.worker.v1.LocalPackageFileState
	31,  // 213: cozy.worker.v1.LocalPackageAbortStatus.outcome:type_name -> cozy.worker.v1.LocalPackageAbortOutcome
	32,  // 214: cozy.worker.v1.WeightsFinalizeRequest.disposition:type_name -> cozy.worker.v1.WeightsFinalizeDisposition
	33,  // 215: cozy.worker.v1.WeightsFinalizeResult.outcome:type_name -> cozy.worker.v1.WeightsFinalizeOutcome
	130, // 216: cozy.worker.v1.WeightsFinalizeResult.weights_receipt:type_name -> cozy.worker.v1.WeightsReceiptRef
	190, // 217: cozy.worker.v1.ResultEnvelope.result_blob:type_name -> cozy.worker.v1.OutputEntry
	164, // 218: cozy.worker.v1.ResultEnvelope.adjustments:type_name -> cozy.worker.v1.AdjustmentRow
	9,   // 219: cozy.worker.v1.OutcomeCause.code:type_name -> cozy.worker.v1.CauseCode
	10,  // 220: cozy.worker.v1.OutcomeCause.origin:type_name -> cozy.worker.v1.CauseOrigin
	166, // 221: cozy.worker.v1.OutcomeCause.shortfall:type_name -> cozy.worker.v1.ResourceShortfall
	190, // 222: cozy.worker.v1.JobCheckpointRequest.artifact:type_name -> cozy.worker.v1.OutputEntry
	15,  // 223: cozy.worker.v1.JobCheckpointReceipt.outcome:type_name -> cozy.worker.v1.CheckpointOutcome
	170, // 224: cozy.worker.v1.JobCheckpointReceipt.fault:type_name -> cozy.worker.v1.CheckpointFault
	16,  // 225: cozy.worker.v1.CheckpointFault.code:type_name -> cozy.worker.v1.CheckpointFaultCode
	175, // 226: cozy.worker.v1.InvocationSpec.inputs:type_name -> cozy.worker.v1.InputBinding
	176, // 227: cozy.worker.v1.InvocationSpec.outputs:type_name -> cozy.worker.v1.OutputBinding
	177, // 228: cozy.worker.v1.InvocationSpec.serving:type_name -> cozy.worker.v1.ServingInvocationSpec
	178, // 229: cozy.worker.v1.InvocationSpec.job:type_name -> cozy.worker.v1.JobInvocationSpec
	184, // 230: cozy.worker.v1.JobInvocationSpec.publication_contract:type_name -> cozy.worker.v1.PublicationContract
	182, // 231: cozy.worker.v1.DeliveryGrant.credential:type_name -> cozy.worker.v1.DeliveryAccessCredential
	180, // 232: cozy.worker.v1.DeliveryGrant.inputs:type_name -> cozy.worker.v1.InputAccess
	181, // 233: cozy.worker.v1.DeliveryGrant.outputs:type_name -> cozy.worker.v1.OutputAccess
	176, // 234: cozy.worker.v1.PublicationContract.outputs:type_name -> cozy.worker.v1.OutputBinding
	6,   // 235: cozy.worker.v1.HeldAttempt.kind:type_name -> cozy.worker.v1.AttemptKind
	7,   // 236: cozy.worker.v1.HeldAttempt.state:type_name -> cozy.worker.v1.AttemptState
	13,  // 237: cozy.worker.v1.Fault.kind:type_name -> cozy.worker.v1.FaultKind
	190, // 238: cozy.worker.v1.OutputManifest.outputs:type_name -> cozy.worker.v1.OutputEntry
	80,  // 239: cozy.worker.v1.WorkerControl.Control:input_type -> cozy.worker.v1.RecordOwnerFrame
	172, // 240: cozy.worker.v1.WorkerControl.WatchProgress:input_type -> cozy.worker.v1.ProgressOpen
	35,  // 241: cozy.worker.v1.RuntimePreparation.ProtocolInfo:input_type -> cozy.worker.v1.ProtocolInfoRequest
	58,  // 242: cozy.worker.v1.RuntimePreparation.CheckPackageSetCompatibility:input_type -> cozy.worker.v1.PreparePackageSetRequest
	58,  // 243: cozy.worker.v1.RuntimePreparation.PreparePackageSet:input_type -> cozy.worker.v1.PreparePackageSetRequest
	78,  // 244: cozy.worker.v1.RuntimePreparation.PrepareModelSource:input_type -> cozy.worker.v1.PrepareModelSourceRequest
	44,  // 245: cozy.worker.v1.RuntimePreparation.ReleaseModelSource:input_type -> cozy.worker.v1.ReleaseModelSourceRequest
	46,  // 246: cozy.worker.v1.RuntimePreparation.RetainDerivedResult:input_type -> cozy.worker.v1.DerivedRetentionRequest
	46,  // 247: cozy.worker.v1.RuntimePreparation.ReleaseDerivedRetention:input_type -> cozy.worker.v1.DerivedRetentionRequest
	139, // 248: cozy.worker.v1.RuntimePreparation.ValidateWeightsCheckpoint:input_type -> cozy.worker.v1.ValidateWeightsCheckpointRequest
	74,  // 249: cozy.worker.v1.RuntimePreparation.CheckpointPage:input_type -> cozy.worker.v1.CheckpointPageRequest
	76,  // 250: cozy.worker.v1.RuntimePreparation.CheckpointTransfer:input_type -> cozy.worker.v1.CheckpointTransferRequest
	63,  // 251: cozy.worker.v1.RuntimePreparation.PrepareLocalPackage:input_type -> cozy.worker.v1.PrepareLocalPackageRequest
	65,  // 252: cozy.worker.v1.RuntimePreparation.PreparePrivatePlacement:input_type -> cozy.worker.v1.PreparePrivatePlacementRequest
	134, // 253: cozy.worker.v1.RuntimeWeights.Exchange:input_type -> cozy.worker.v1.WeightsHostAck
	149, // 254: cozy.worker.v1.RuntimeWeights.Upload:input_type -> cozy.worker.v1.WeightsUploadRequest
	35,  // 255: cozy.worker.v1.PodHost.ProtocolInfo:input_type -> cozy.worker.v1.ProtocolInfoRequest
	37,  // 256: cozy.worker.v1.PodHost.PreparePackageSet:input_type -> cozy.worker.v1.PreparePackageSetCall
	38,  // 257: cozy.worker.v1.PodHost.PrepareLocalPackage:input_type -> cozy.worker.v1.PrepareLocalPackageCall
	39,  // 258: cozy.worker.v1.PodHost.PreparePrivatePlacement:input_type -> cozy.worker.v1.PreparePrivatePlacementCall
	41,  // 259: cozy.worker.v1.PodHost.ModelSourceFile:input_type -> cozy.worker.v1.ModelSourceFileCall
	42,  // 260: cozy.worker.v1.PodHost.ModelSourcePrepare:input_type -> cozy.worker.v1.ModelSourcePrepareCall
	43,  // 261: cozy.worker.v1.PodHost.ModelSourceRelease:input_type -> cozy.worker.v1.ModelSourceReleaseCall
	45,  // 262: cozy.worker.v1.PodHost.RetainDerivedResult:input_type -> cozy.worker.v1.DerivedRetentionCall
	45,  // 263: cozy.worker.v1.PodHost.ReleaseDerivedRetention:input_type -> cozy.worker.v1.DerivedRetentionCall
	49,  // 264: cozy.worker.v1.PodHost.ModelSourceAdopt:input_type -> cozy.worker.v1.ModelSourceAdoptCall
	50,  // 265: cozy.worker.v1.PodHost.CheckpointPage:input_type -> cozy.worker.v1.CheckpointPageCall
	51,  // 266: cozy.worker.v1.PodHost.CheckpointTransfer:input_type -> cozy.worker.v1.CheckpointTransferCall
	52,  // 267: cozy.worker.v1.PodHost.LocalPackageFetch:input_type -> cozy.worker.v1.LocalPackageFetchCall
	53,  // 268: cozy.worker.v1.PodHost.LocalPackageUpload:input_type -> cozy.worker.v1.LocalPackageUploadFrame
	56,  // 269: cozy.worker.v1.PodHost.LocalPackageAbort:input_type -> cozy.worker.v1.LocalPackageAbortCall
	57,  // 270: cozy.worker.v1.PodHost.WeightsTransfer:input_type -> cozy.worker.v1.WeightsTransferCall
	138, // 271: cozy.worker.v1.PodHost.WeightsIntentReady:input_type -> cozy.worker.v1.WeightsIntentReadyCall
	81,  // 272: cozy.worker.v1.WorkerControl.Control:output_type -> cozy.worker.v1.WorkerFrame
	173, // 273: cozy.worker.v1.WorkerControl.WatchProgress:output_type -> cozy.worker.v1.AttemptProgress
	36,  // 274: cozy.worker.v1.RuntimePreparation.ProtocolInfo:output_type -> cozy.worker.v1.ProtocolInfoResult
	62,  // 275: cozy.worker.v1.RuntimePreparation.CheckPackageSetCompatibility:output_type -> cozy.worker.v1.CheckPackageSetCompatibilityResult
	61,  // 276: cozy.worker.v1.RuntimePreparation.PreparePackageSet:output_type -> cozy.worker.v1.PreparePackageSetResult
	79,  // 277: cozy.worker.v1.RuntimePreparation.PrepareModelSource:output_type -> cozy.worker.v1.PrepareModelSourceResult
	48,  // 278: cozy.worker.v1.RuntimePreparation.ReleaseModelSource:output_type -> cozy.worker.v1.ReleaseModelSourceResult
	47,  // 279: cozy.worker.v1.RuntimePreparation.RetainDerivedResult:output_type -> cozy.worker.v1.DerivedRetentionResult
	47,  // 280: cozy.worker.v1.RuntimePreparation.ReleaseDerivedRetention:output_type -> cozy.worker.v1.DerivedRetentionResult
	140, // 281: cozy.worker.v1.RuntimePreparation.ValidateWeightsCheckpoint:output_type -> cozy.worker.v1.ValidateWeightsCheckpointResult
	75,  // 282: cozy.worker.v1.RuntimePreparation.CheckpointPage:output_type -> cozy.worker.v1.CheckpointPageResult
	77,  // 283: cozy.worker.v1.RuntimePreparation.CheckpointTransfer:output_type -> cozy.worker.v1.CheckpointTransferStatus
	61,  // 284: cozy.worker.v1.RuntimePreparation.PrepareLocalPackage:output_type -> cozy.worker.v1.PreparePackageSetResult
	61,  // 285: cozy.worker.v1.RuntimePreparation.PreparePrivatePlacement:output_type -> cozy.worker.v1.PreparePackageSetResult
	34,  // 286: cozy.worker.v1.RuntimeWeights.Exchange:output_type -> cozy.worker.v1.WeightsHostEvent
	150, // 287: cozy.worker.v1.RuntimeWeights.Upload:output_type -> cozy.worker.v1.WeightsUploadResult
	36,  // 288: cozy.worker.v1.PodHost.ProtocolInfo:output_type -> cozy.worker.v1.ProtocolInfoResult
	40,  // 289: cozy.worker.v1.PodHost.PreparePackageSet:output_type -> cozy.worker.v1.PrepareEvent
	40,  // 290: cozy.worker.v1.PodHost.PrepareLocalPackage:output_type -> cozy.worker.v1.PrepareEvent
	40,  // 291: cozy.worker.v1.PodHost.PreparePrivatePlacement:output_type -> cozy.worker.v1.PrepareEvent
	153, // 292: cozy.worker.v1.PodHost.ModelSourceFile:output_type -> cozy.worker.v1.ModelSourceFileStatus
	155, // 293: cozy.worker.v1.PodHost.ModelSourcePrepare:output_type -> cozy.worker.v1.ModelSourcePrepared
	48,  // 294: cozy.worker.v1.PodHost.ModelSourceRelease:output_type -> cozy.worker.v1.ReleaseModelSourceResult
	47,  // 295: cozy.worker.v1.PodHost.RetainDerivedResult:output_type -> cozy.worker.v1.DerivedRetentionResult
	47,  // 296: cozy.worker.v1.PodHost.ReleaseDerivedRetention:output_type -> cozy.worker.v1.DerivedRetentionResult
	155, // 297: cozy.worker.v1.PodHost.ModelSourceAdopt:output_type -> cozy.worker.v1.ModelSourcePrepared
	75,  // 298: cozy.worker.v1.PodHost.CheckpointPage:output_type -> cozy.worker.v1.CheckpointPageResult
	77,  // 299: cozy.worker.v1.PodHost.CheckpointTransfer:output_type -> cozy.worker.v1.CheckpointTransferStatus
	158, // 300: cozy.worker.v1.PodHost.LocalPackageFetch:output_type -> cozy.worker.v1.LocalPackageFileStatus
	158, // 301: cozy.worker.v1.PodHost.LocalPackageUpload:output_type -> cozy.worker.v1.LocalPackageFileStatus
	160, // 302: cozy.worker.v1.PodHost.LocalPackageAbort:output_type -> cozy.worker.v1.LocalPackageAbortStatus
	151, // 303: cozy.worker.v1.PodHost.WeightsTransfer:output_type -> cozy.worker.v1.WeightsTransferStatus
	134, // 304: cozy.worker.v1.PodHost.WeightsIntentReady:output_type -> cozy.worker.v1.WeightsHostAck
	272, // [272:305] is the sub-list for method output_type
	239, // [239:272] is the sub-list for method input_type
	239, // [239:239] is the sub-list for extension type_name
	239, // [239:239] is the sub-list for extension extendee
	0,   // [0:239] is the sub-list for field type_name
}

func init() { file_cozy_worker_v1_worker_proto_init() }
func file_cozy_worker_v1_worker_proto_init() {
	if File_cozy_worker_v1_worker_proto != nil {
		return
	}
	file_cozy_worker_v1_worker_proto_msgTypes[0].OneofWrappers = []any{
		(*WeightsHostEvent_Intent)(nil),
		(*WeightsHostEvent_Receipt)(nil),
		(*WeightsHostEvent_Checkpoint)(nil),
	}
	file_cozy_worker_v1_worker_proto_msgTypes[19].OneofWrappers = []any{
		(*LocalPackageUploadFrame_Header)(nil),
		(*LocalPackageUploadFrame_Chunk)(nil),
	}
	file_cozy_worker_v1_worker_proto_msgTypes[38].OneofWrappers = []any{
		(*CheckpointSubject_Source)(nil),
		(*CheckpointSubject_Weights)(nil),
	}
	file_cozy_worker_v1_worker_proto_msgTypes[42].OneofWrappers = []any{
		(*CheckpointTransferRequest_UploadGrant)(nil),
		(*CheckpointTransferRequest_DownloadUrl)(nil),
	}
	file_cozy_worker_v1_worker_proto_msgTypes[46].OneofWrappers = []any{
		(*RecordOwnerFrame_Claim)(nil),
		(*RecordOwnerFrame_DesiredState)(nil),
		(*RecordOwnerFrame_AttemptOffer)(nil),
		(*RecordOwnerFrame_CancelAttempt)(nil),
		(*RecordOwnerFrame_OutcomeAck)(nil),
		(*RecordOwnerFrame_CheckpointReceipt)(nil),
		(*RecordOwnerFrame_SnapshotAck)(nil),
		(*RecordOwnerFrame_WeightsFinalizeRequest)(nil),
		(*RecordOwnerFrame_WeightsHostAck)(nil),
		(*RecordOwnerFrame_WeightsTransferRequest)(nil),
		(*RecordOwnerFrame_WeightsReadRequest)(nil),
		(*RecordOwnerFrame_ModelSourceFileRequest)(nil),
		(*RecordOwnerFrame_ModelSourcePrepareRequest)(nil),
		(*RecordOwnerFrame_LocalPackageAbort)(nil),
		(*RecordOwnerFrame_WeightsUploadRequest)(nil),
		(*RecordOwnerFrame_LocalPackageFetchRequest)(nil),
		(*RecordOwnerFrame_ChildCallResult)(nil),
	}
	file_cozy_worker_v1_worker_proto_msgTypes[47].OneofWrappers = []any{
		(*WorkerFrame_ClaimAck)(nil),
		(*WorkerFrame_ObservedState)(nil),
		(*WorkerFrame_AttemptAccepted)(nil),
		(*WorkerFrame_AttemptOutcome)(nil),
		(*WorkerFrame_BootFailure)(nil),
		(*WorkerFrame_CheckpointRequest)(nil),
		(*WorkerFrame_CheckpointAck)(nil),
		(*WorkerFrame_Snapshot)(nil),
		(*WorkerFrame_WeightsFinalizeResult)(nil),
		(*WorkerFrame_WeightsIntent)(nil),
		(*WorkerFrame_WeightsReceipt)(nil),
		(*WorkerFrame_WeightsReadResult)(nil),
		(*WorkerFrame_WeightsTransferStatus)(nil),
		(*WorkerFrame_ModelSourceFileStatus)(nil),
		(*WorkerFrame_ModelSourcePrepared)(nil),
		(*WorkerFrame_LocalPackageFileStatus)(nil),
		(*WorkerFrame_LocalPackageAbortStatus)(nil),
		(*WorkerFrame_WeightsUploadResult)(nil),
		(*WorkerFrame_WeightsCheckpoint)(nil),
		(*WorkerFrame_WeightsTransaction)(nil),
		(*WorkerFrame_ChildCallRequest)(nil),
		(*WorkerFrame_ChildCallCancel)(nil),
	}
	file_cozy_worker_v1_worker_proto_msgTypes[59].OneofWrappers = []any{
		(*DesiredWorkerState_Job)(nil),
		(*DesiredWorkerState_PlacementSet)(nil),
		(*DesiredWorkerState_PackageSet)(nil),
		(*DesiredWorkerState_LocalPackageSet)(nil),
		(*DesiredWorkerState_PrivatePlacementSet)(nil),
	}
	file_cozy_worker_v1_worker_proto_msgTypes[71].OneofWrappers = []any{
		(*Placement_Package)(nil),
		(*Placement_Development)(nil),
	}
	file_cozy_worker_v1_worker_proto_msgTypes[114].OneofWrappers = []any{
		(*WeightsTransferRequest_UploadGrant)(nil),
		(*WeightsTransferRequest_Held)(nil),
	}
	file_cozy_worker_v1_worker_proto_msgTypes[140].OneofWrappers = []any{
		(*InvocationSpec_Serving)(nil),
		(*InvocationSpec_Job)(nil),
	}
	type x struct{}
	out := protoimpl.TypeBuilder{
		File: protoimpl.DescBuilder{
			GoPackagePath: reflect.TypeOf(x{}).PkgPath(),
			RawDescriptor: unsafe.Slice(unsafe.StringData(file_cozy_worker_v1_worker_proto_rawDesc), len(file_cozy_worker_v1_worker_proto_rawDesc)),
			NumEnums:      34,
			NumMessages:   159,
			NumExtensions: 0,
			NumServices:   4,
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
