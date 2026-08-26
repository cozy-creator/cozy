// cozy.worker.v1 — the record-owner<->worker wire contract. Schema only; no behavior.
//
// Authority: tracker-v2/worker-protocol/01-wire-schema.md (schema laws), 02-lifecycle.md
// (behavior), 03-versioning-conformance.md (versioning). Issue: tracker/tensorhub/th-024.
//
// ORIENTATION (2026-08-25 re-landing; supersedes the worker-dials arrangement): the WORKER
// (the cozy-runtime supervisor, "WorkerSupervisor") HOSTS this service; the RECORD-PLANE OWNER
// ("RecordOwner" — cozy-creator's coordinator, a web-mode pod's co-resident owner module, or
// tensorhub's orchestrator via th-007) DIALS it. The orientation is IDENTICAL local and remote;
// only channel establishment differs: a Unix socket (Windows: loopback) co-resident, TLS over a
// private network or authenticated overlay remotely. A worker control port is NEVER required to
// be publicly exposed for topology symmetry.
//
// VERSIONING (03 §1, §2 R1-R8): the MAJOR is the package path (`cozy.worker.v1`); the MINOR is
// `wire_minor`, a linear-train number declared at Claim/ClaimAck — never a negotiation. R1-R8
// as recorded in 03. R7 IS AN AUTHORING RULE ONLY (2026-08-25 correction): `reserved` numbers
// and names are compiler-enforced tombstones against REUSE; ordinary proto3 decoders do not
// refuse them on the wire (field names do not exist in binary protobuf at all), and no runtime
// polices them. This contract is UNRELEASED; this file is an in-place revision (the fourth:
// 5b07b79 -> 732763b -> 8d90461 -> this, decision #446) and renumbers freely, per the
// 2026-08-24 precedent.
//
// IDENTITY IS CANONICAL BYTES, PROTOBUF IS TRANSPORT (unchanged from 732763b): no digest is
// ever computed over protobuf-marshaled bytes. A meaning-fencing digest is the SHA-256 of a
// DOCUMENT's exact canonical bytes (RFC 8785-shaped canonical JSON under tfs-013's writer).
// Where the fence requires byte agreement the canonical bytes travel IN the message as a
// `bytes` field instead of a structured body; the receiver recomputes sha256 over the resident
// bytes. DIGEST CLASSES: (a) document identity over canonical JSON bytes; (b) content digest
// over opaque artifact bytes. On the wire a class-(a)/(b) digest is raw 32-byte `bytes`; inside
// a canonical document the same digest is spelled `"sha256:<64 lowercase hex>"`.
// BYTES NAMING LAW: a `bytes` field named `digest`/`*_digest` IS a 32-byte SHA-256; every other
// `bytes` field is an opaque payload (base64 in documents).
//
// THE ENVELOPE (every message, fields 1-3): the ownership + boot fence, checked BEFORE any body
// field is read, in this order:
//   1 `owner_epoch`         - durable monotonic AUTHORITY generation, minted by whatever grants
//                             ownership (the hub's pod lease/CAS at Wave 2; the private owner's
//                             exclusive record-plane lease). A frame with an older epoch than
//                             the worker's accepted claim is dropped.
//   2 `control_generation`  - worker-minted, incremented for EVERY accepted control stream; a
//                             frame from a superseded stream is dropped.
//   3 `worker_boot_id`      - minted at supervisor boot, never reused (the old `session_id`);
//                             a frame addressed to a dead boot is dropped.
// FIELD 4 IS RESERVED IN EVERY MESSAGE (#446): the old global `executor_incarnation` is GONE.
// The worker BOOT generation (`worker_boot_id`) is machine truth; each deployment's EXECUTOR
// generation is deployment truth (`DeploymentStatus.executor_generation`, bumping on that
// executor's respawn) and rides attempt-scoped facts (`AttemptAccepted`, `ActiveAttempt`), not
// the envelope. An attempt never survives its executor generation, so no owner->worker frame
// needs to name one: the (request_id, attempt, invocation_digest) triple resolves against the
// journal.
// LAUNCH TIER (single-owner-first): private pods ship single-owner — a Claim while a live
// fenced stream exists REFUSES unless its owner_epoch is strictly higher; the machinery that
// MINTS competing epochs (hub lease/CAS) is Wave-2 and arms with the fleet.
//
// DEPLOYMENTS (#446, from #425/#444): th-024 addresses the MACHINE; `deployment_id` is the
// owner-minted routing + journal key for one hosted endpoint deployment. The desired state is
// a digest-addressed DeploymentSet; status, readiness, capacity credits, fault, and executor
// generation are all PER-DEPLOYMENT. LAUNCH ENFORCES len(deployments) <= 1: a longer set is a
// typed refusal (FAULT_KIND_DEPLOYMENT_SET_UNSUPPORTED) with the directive unapplied — the
// wire shape is multi-deployment so stacking lands without another hardcut.
//
// ONEOF-LAST RULE (kept from 732763b, transport determinism only, never identity): every oneof
// carries the highest field numbers in its message.
//
// ENUM VALUE NAMING (kept): every value is prefixed with its enum's SCREAMING_SNAKE type name.
//
// THINNING (2026-08-25, "no writer, no field"): the purge lane, AppliedAdapter, ArtifactGrant/
// ArtifactSubject and DrainPolicy had no writer in any shipped consumer and are REDUCED TO
// RESERVATIONS at their sites. PURGE RESERVATION: private-artifact purge (cr-013/tfs-014, post-
// M9) rides Control as a durable command/acknowledgement pair when it lands — reserved oneof
// slots 14 (OwnerFrame.purge_command) and 14 (WorkerFrame.purge_report) hold its place; no
// separate RPC returns.

// Code generated by protoc-gen-go. DO NOT EDIT.
// versions:
// 	protoc-gen-go v1.36.11
// 	protoc        v6.31.1
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
	return file_cozy_worker_v1_worker_proto_enumTypes[0].Descriptor()
}

func (Posture) Type() protoreflect.EnumType {
	return &file_cozy_worker_v1_worker_proto_enumTypes[0]
}

func (x Posture) Number() protoreflect.EnumNumber {
	return protoreflect.EnumNumber(x)
}

// Deprecated: Use Posture.Descriptor instead.
func (Posture) EnumDescriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{0}
}

type IntakeState int32

const (
	IntakeState_INTAKE_STATE_UNSPECIFIED IntakeState = 0
	IntakeState_INTAKE_STATE_BOOTING     IntakeState = 1
	IntakeState_INTAKE_STATE_DOWNLOADING IntakeState = 2
	IntakeState_INTAKE_STATE_LOADING     IntakeState = 3
	IntakeState_INTAKE_STATE_READY       IntakeState = 4
	IntakeState_INTAKE_STATE_DRAINING    IntakeState = 5
	IntakeState_INTAKE_STATE_ERROR       IntakeState = 6
)

// Enum value maps for IntakeState.
var (
	IntakeState_name = map[int32]string{
		0: "INTAKE_STATE_UNSPECIFIED",
		1: "INTAKE_STATE_BOOTING",
		2: "INTAKE_STATE_DOWNLOADING",
		3: "INTAKE_STATE_LOADING",
		4: "INTAKE_STATE_READY",
		5: "INTAKE_STATE_DRAINING",
		6: "INTAKE_STATE_ERROR",
	}
	IntakeState_value = map[string]int32{
		"INTAKE_STATE_UNSPECIFIED": 0,
		"INTAKE_STATE_BOOTING":     1,
		"INTAKE_STATE_DOWNLOADING": 2,
		"INTAKE_STATE_LOADING":     3,
		"INTAKE_STATE_READY":       4,
		"INTAKE_STATE_DRAINING":    5,
		"INTAKE_STATE_ERROR":       6,
	}
)

func (x IntakeState) Enum() *IntakeState {
	p := new(IntakeState)
	*p = x
	return p
}

func (x IntakeState) String() string {
	return protoimpl.X.EnumStringOf(x.Descriptor(), protoreflect.EnumNumber(x))
}

func (IntakeState) Descriptor() protoreflect.EnumDescriptor {
	return file_cozy_worker_v1_worker_proto_enumTypes[1].Descriptor()
}

func (IntakeState) Type() protoreflect.EnumType {
	return &file_cozy_worker_v1_worker_proto_enumTypes[1]
}

func (x IntakeState) Number() protoreflect.EnumNumber {
	return protoreflect.EnumNumber(x)
}

// Deprecated: Use IntakeState.Descriptor instead.
func (IntakeState) EnumDescriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{1}
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
	return file_cozy_worker_v1_worker_proto_enumTypes[2].Descriptor()
}

func (AttemptKind) Type() protoreflect.EnumType {
	return &file_cozy_worker_v1_worker_proto_enumTypes[2]
}

func (x AttemptKind) Number() protoreflect.EnumNumber {
	return protoreflect.EnumNumber(x)
}

// Deprecated: Use AttemptKind.Descriptor instead.
func (AttemptKind) EnumDescriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{2}
}

type AttemptState int32

const (
	AttemptState_ATTEMPT_STATE_UNSPECIFIED          AttemptState = 0
	AttemptState_ATTEMPT_STATE_RUNNING              AttemptState = 1
	AttemptState_ATTEMPT_STATE_TERMINAL_PENDING_ACK AttemptState = 2
)

// Enum value maps for AttemptState.
var (
	AttemptState_name = map[int32]string{
		0: "ATTEMPT_STATE_UNSPECIFIED",
		1: "ATTEMPT_STATE_RUNNING",
		2: "ATTEMPT_STATE_TERMINAL_PENDING_ACK",
	}
	AttemptState_value = map[string]int32{
		"ATTEMPT_STATE_UNSPECIFIED":          0,
		"ATTEMPT_STATE_RUNNING":              1,
		"ATTEMPT_STATE_TERMINAL_PENDING_ACK": 2,
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
	return file_cozy_worker_v1_worker_proto_enumTypes[3].Descriptor()
}

func (AttemptState) Type() protoreflect.EnumType {
	return &file_cozy_worker_v1_worker_proto_enumTypes[3]
}

func (x AttemptState) Number() protoreflect.EnumNumber {
	return protoreflect.EnumNumber(x)
}

// Deprecated: Use AttemptState.Descriptor instead.
func (AttemptState) EnumDescriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{3}
}

type TerminalStatus int32

const (
	TerminalStatus_TERMINAL_STATUS_UNSPECIFIED TerminalStatus = 0
	TerminalStatus_TERMINAL_STATUS_SUCCEEDED   TerminalStatus = 1
	TerminalStatus_TERMINAL_STATUS_REFUSED     TerminalStatus = 2
	TerminalStatus_TERMINAL_STATUS_FAILED      TerminalStatus = 3
	TerminalStatus_TERMINAL_STATUS_CANCELED    TerminalStatus = 4
	TerminalStatus_TERMINAL_STATUS_ABANDONED   TerminalStatus = 5
)

// Enum value maps for TerminalStatus.
var (
	TerminalStatus_name = map[int32]string{
		0: "TERMINAL_STATUS_UNSPECIFIED",
		1: "TERMINAL_STATUS_SUCCEEDED",
		2: "TERMINAL_STATUS_REFUSED",
		3: "TERMINAL_STATUS_FAILED",
		4: "TERMINAL_STATUS_CANCELED",
		5: "TERMINAL_STATUS_ABANDONED",
	}
	TerminalStatus_value = map[string]int32{
		"TERMINAL_STATUS_UNSPECIFIED": 0,
		"TERMINAL_STATUS_SUCCEEDED":   1,
		"TERMINAL_STATUS_REFUSED":     2,
		"TERMINAL_STATUS_FAILED":      3,
		"TERMINAL_STATUS_CANCELED":    4,
		"TERMINAL_STATUS_ABANDONED":   5,
	}
)

func (x TerminalStatus) Enum() *TerminalStatus {
	p := new(TerminalStatus)
	*p = x
	return p
}

func (x TerminalStatus) String() string {
	return protoimpl.X.EnumStringOf(x.Descriptor(), protoreflect.EnumNumber(x))
}

func (TerminalStatus) Descriptor() protoreflect.EnumDescriptor {
	return file_cozy_worker_v1_worker_proto_enumTypes[4].Descriptor()
}

func (TerminalStatus) Type() protoreflect.EnumType {
	return &file_cozy_worker_v1_worker_proto_enumTypes[4]
}

func (x TerminalStatus) Number() protoreflect.EnumNumber {
	return protoreflect.EnumNumber(x)
}

// Deprecated: Use TerminalStatus.Descriptor instead.
func (TerminalStatus) EnumDescriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{4}
}

type CauseCode int32

const (
	CauseCode_CAUSE_CODE_UNSPECIFIED CauseCode = 0
	// REFUSED
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
	}
	CauseCode_value = map[string]int32{
		"CAUSE_CODE_UNSPECIFIED":            0,
		"CAUSE_CODE_INVALID_REQUEST":        1,
		"CAUSE_CODE_UNSUPPORTED_INPUT":      2,
		"CAUSE_CODE_LOCAL_SAFETY":           3,
		"CAUSE_CODE_PROTOCOL":               4,
		"CAUSE_CODE_CONSTRAINT_INFEASIBLE":  5,
		"CAUSE_CODE_AUTHOR_EXCEPTION":       6,
		"CAUSE_CODE_EXECUTOR_FAULT":         7,
		"CAUSE_CODE_GRANT_EXPIRED":          8,
		"CAUSE_CODE_ARTIFACT_UNFETCHABLE":   9,
		"CAUSE_CODE_CAPABILITY_UNAVAILABLE": 10,
		"CAUSE_CODE_CLIENT_CANCEL":          11,
		"CAUSE_CODE_DEADLINE_EXPIRED":       12,
		"CAUSE_CODE_DRAIN_CANCEL":           13,
		"CAUSE_CODE_POLICY_CANCEL":          14,
		"CAUSE_CODE_SUPERSEDED_CANCEL":      15,
		"CAUSE_CODE_EXECUTOR_INVALIDATED":   16,
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
	return file_cozy_worker_v1_worker_proto_enumTypes[5].Descriptor()
}

func (CauseCode) Type() protoreflect.EnumType {
	return &file_cozy_worker_v1_worker_proto_enumTypes[5]
}

func (x CauseCode) Number() protoreflect.EnumNumber {
	return protoreflect.EnumNumber(x)
}

// Deprecated: Use CauseCode.Descriptor instead.
func (CauseCode) EnumDescriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{5}
}

type CauseOrigin int32

const (
	CauseOrigin_CAUSE_ORIGIN_UNSPECIFIED CauseOrigin = 0
	CauseOrigin_CAUSE_ORIGIN_AUTHOR      CauseOrigin = 1
	CauseOrigin_CAUSE_ORIGIN_RUNTIME     CauseOrigin = 2
	CauseOrigin_CAUSE_ORIGIN_EXECUTOR    CauseOrigin = 3
	CauseOrigin_CAUSE_ORIGIN_SUPERVISOR  CauseOrigin = 4
	CauseOrigin_CAUSE_ORIGIN_INFRA       CauseOrigin = 5
	CauseOrigin_CAUSE_ORIGIN_CLIENT      CauseOrigin = 6
	CauseOrigin_CAUSE_ORIGIN_COORDINATOR CauseOrigin = 7
)

// Enum value maps for CauseOrigin.
var (
	CauseOrigin_name = map[int32]string{
		0: "CAUSE_ORIGIN_UNSPECIFIED",
		1: "CAUSE_ORIGIN_AUTHOR",
		2: "CAUSE_ORIGIN_RUNTIME",
		3: "CAUSE_ORIGIN_EXECUTOR",
		4: "CAUSE_ORIGIN_SUPERVISOR",
		5: "CAUSE_ORIGIN_INFRA",
		6: "CAUSE_ORIGIN_CLIENT",
		7: "CAUSE_ORIGIN_COORDINATOR",
	}
	CauseOrigin_value = map[string]int32{
		"CAUSE_ORIGIN_UNSPECIFIED": 0,
		"CAUSE_ORIGIN_AUTHOR":      1,
		"CAUSE_ORIGIN_RUNTIME":     2,
		"CAUSE_ORIGIN_EXECUTOR":    3,
		"CAUSE_ORIGIN_SUPERVISOR":  4,
		"CAUSE_ORIGIN_INFRA":       5,
		"CAUSE_ORIGIN_CLIENT":      6,
		"CAUSE_ORIGIN_COORDINATOR": 7,
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
	return file_cozy_worker_v1_worker_proto_enumTypes[6].Descriptor()
}

func (CauseOrigin) Type() protoreflect.EnumType {
	return &file_cozy_worker_v1_worker_proto_enumTypes[6]
}

func (x CauseOrigin) Number() protoreflect.EnumNumber {
	return protoreflect.EnumNumber(x)
}

// Deprecated: Use CauseOrigin.Descriptor instead.
func (CauseOrigin) EnumDescriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{6}
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
	return file_cozy_worker_v1_worker_proto_enumTypes[7].Descriptor()
}

func (CancelReason) Type() protoreflect.EnumType {
	return &file_cozy_worker_v1_worker_proto_enumTypes[7]
}

func (x CancelReason) Number() protoreflect.EnumNumber {
	return protoreflect.EnumNumber(x)
}

// Deprecated: Use CancelReason.Descriptor instead.
func (CancelReason) EnumDescriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{7}
}

type ClaimRejection int32

const (
	ClaimRejection_CLAIM_REJECTION_UNSPECIFIED         ClaimRejection = 0
	ClaimRejection_CLAIM_REJECTION_UNAUTHENTICATED     ClaimRejection = 1
	ClaimRejection_CLAIM_REJECTION_STALE_OWNER_EPOCH   ClaimRejection = 2 // epoch older than the accepted claim
	ClaimRejection_CLAIM_REJECTION_EPOCH_HELD          ClaimRejection = 3 // equal epoch, different controller_id
	ClaimRejection_CLAIM_REJECTION_WORKER_ID_MISMATCH  ClaimRejection = 4
	ClaimRejection_CLAIM_REJECTION_RELEASE_ID_MISMATCH ClaimRejection = 5
	ClaimRejection_CLAIM_REJECTION_UNSUPPORTED_MINOR   ClaimRejection = 6
)

// Enum value maps for ClaimRejection.
var (
	ClaimRejection_name = map[int32]string{
		0: "CLAIM_REJECTION_UNSPECIFIED",
		1: "CLAIM_REJECTION_UNAUTHENTICATED",
		2: "CLAIM_REJECTION_STALE_OWNER_EPOCH",
		3: "CLAIM_REJECTION_EPOCH_HELD",
		4: "CLAIM_REJECTION_WORKER_ID_MISMATCH",
		5: "CLAIM_REJECTION_RELEASE_ID_MISMATCH",
		6: "CLAIM_REJECTION_UNSUPPORTED_MINOR",
	}
	ClaimRejection_value = map[string]int32{
		"CLAIM_REJECTION_UNSPECIFIED":         0,
		"CLAIM_REJECTION_UNAUTHENTICATED":     1,
		"CLAIM_REJECTION_STALE_OWNER_EPOCH":   2,
		"CLAIM_REJECTION_EPOCH_HELD":          3,
		"CLAIM_REJECTION_WORKER_ID_MISMATCH":  4,
		"CLAIM_REJECTION_RELEASE_ID_MISMATCH": 5,
		"CLAIM_REJECTION_UNSUPPORTED_MINOR":   6,
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
	return file_cozy_worker_v1_worker_proto_enumTypes[8].Descriptor()
}

func (ClaimRejection) Type() protoreflect.EnumType {
	return &file_cozy_worker_v1_worker_proto_enumTypes[8]
}

func (x ClaimRejection) Number() protoreflect.EnumNumber {
	return protoreflect.EnumNumber(x)
}

// Deprecated: Use ClaimRejection.Descriptor instead.
func (ClaimRejection) EnumDescriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{8}
}

type FaultKind int32

const (
	FaultKind_FAULT_KIND_UNSPECIFIED                FaultKind = 0
	FaultKind_FAULT_KIND_BINDING_UNAVAILABLE        FaultKind = 1
	FaultKind_FAULT_KIND_BINDING_DEGRADED           FaultKind = 2
	FaultKind_FAULT_KIND_HARDWARE_UNSUITABLE        FaultKind = 3
	FaultKind_FAULT_KIND_ARTIFACT_FETCH_FAILED      FaultKind = 4
	FaultKind_FAULT_KIND_GRANT_EXPIRED              FaultKind = 5
	FaultKind_FAULT_KIND_CREDENTIAL_UNAPPLIED       FaultKind = 6
	FaultKind_FAULT_KIND_LOCAL_SAFETY_REFUSAL       FaultKind = 7
	FaultKind_FAULT_KIND_EXECUTOR_POISONED          FaultKind = 8
	FaultKind_FAULT_KIND_CONFIG_REFUSED             FaultKind = 9
	FaultKind_FAULT_KIND_UNKNOWN_DEPLOYMENT         FaultKind = 10
	FaultKind_FAULT_KIND_DEPLOYMENT_SET_UNSUPPORTED FaultKind = 11 // launch len<=1 breach or an unsupportable set;
)

// Enum value maps for FaultKind.
var (
	FaultKind_name = map[int32]string{
		0:  "FAULT_KIND_UNSPECIFIED",
		1:  "FAULT_KIND_BINDING_UNAVAILABLE",
		2:  "FAULT_KIND_BINDING_DEGRADED",
		3:  "FAULT_KIND_HARDWARE_UNSUITABLE",
		4:  "FAULT_KIND_ARTIFACT_FETCH_FAILED",
		5:  "FAULT_KIND_GRANT_EXPIRED",
		6:  "FAULT_KIND_CREDENTIAL_UNAPPLIED",
		7:  "FAULT_KIND_LOCAL_SAFETY_REFUSAL",
		8:  "FAULT_KIND_EXECUTOR_POISONED",
		9:  "FAULT_KIND_CONFIG_REFUSED",
		10: "FAULT_KIND_UNKNOWN_DEPLOYMENT",
		11: "FAULT_KIND_DEPLOYMENT_SET_UNSUPPORTED",
	}
	FaultKind_value = map[string]int32{
		"FAULT_KIND_UNSPECIFIED":                0,
		"FAULT_KIND_BINDING_UNAVAILABLE":        1,
		"FAULT_KIND_BINDING_DEGRADED":           2,
		"FAULT_KIND_HARDWARE_UNSUITABLE":        3,
		"FAULT_KIND_ARTIFACT_FETCH_FAILED":      4,
		"FAULT_KIND_GRANT_EXPIRED":              5,
		"FAULT_KIND_CREDENTIAL_UNAPPLIED":       6,
		"FAULT_KIND_LOCAL_SAFETY_REFUSAL":       7,
		"FAULT_KIND_EXECUTOR_POISONED":          8,
		"FAULT_KIND_CONFIG_REFUSED":             9,
		"FAULT_KIND_UNKNOWN_DEPLOYMENT":         10,
		"FAULT_KIND_DEPLOYMENT_SET_UNSUPPORTED": 11,
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
	return file_cozy_worker_v1_worker_proto_enumTypes[9].Descriptor()
}

func (FaultKind) Type() protoreflect.EnumType {
	return &file_cozy_worker_v1_worker_proto_enumTypes[9]
}

func (x FaultKind) Number() protoreflect.EnumNumber {
	return protoreflect.EnumNumber(x)
}

// Deprecated: Use FaultKind.Descriptor instead.
func (FaultKind) EnumDescriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{9}
}

type BootFailureReason int32

const (
	BootFailureReason_BOOT_FAILURE_REASON_UNSPECIFIED      BootFailureReason = 0
	BootFailureReason_BOOT_FAILURE_REASON_HARDWARE_VERDICT BootFailureReason = 1
	BootFailureReason_BOOT_FAILURE_REASON_IMAGE_FAULT      BootFailureReason = 2
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
		2: "BOOT_FAILURE_REASON_IMAGE_FAULT",
		3: "BOOT_FAILURE_REASON_DRIVER_FAULT",
		4: "BOOT_FAILURE_REASON_DISK_SHAPE",
		5: "BOOT_FAILURE_REASON_CONFIG_INVALID",
		6: "BOOT_FAILURE_REASON_OTHER_FATAL",
	}
	BootFailureReason_value = map[string]int32{
		"BOOT_FAILURE_REASON_UNSPECIFIED":      0,
		"BOOT_FAILURE_REASON_HARDWARE_VERDICT": 1,
		"BOOT_FAILURE_REASON_IMAGE_FAULT":      2,
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
	return file_cozy_worker_v1_worker_proto_enumTypes[10].Descriptor()
}

func (BootFailureReason) Type() protoreflect.EnumType {
	return &file_cozy_worker_v1_worker_proto_enumTypes[10]
}

func (x BootFailureReason) Number() protoreflect.EnumNumber {
	return protoreflect.EnumNumber(x)
}

// Deprecated: Use BootFailureReason.Descriptor instead.
func (BootFailureReason) EnumDescriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{10}
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
	return file_cozy_worker_v1_worker_proto_enumTypes[11].Descriptor()
}

func (CheckpointOutcome) Type() protoreflect.EnumType {
	return &file_cozy_worker_v1_worker_proto_enumTypes[11]
}

func (x CheckpointOutcome) Number() protoreflect.EnumNumber {
	return protoreflect.EnumNumber(x)
}

// Deprecated: Use CheckpointOutcome.Descriptor instead.
func (CheckpointOutcome) EnumDescriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{11}
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
	return file_cozy_worker_v1_worker_proto_enumTypes[12].Descriptor()
}

func (CheckpointFaultCode) Type() protoreflect.EnumType {
	return &file_cozy_worker_v1_worker_proto_enumTypes[12]
}

func (x CheckpointFaultCode) Number() protoreflect.EnumNumber {
	return protoreflect.EnumNumber(x)
}

// Deprecated: Use CheckpointFaultCode.Descriptor instead.
func (CheckpointFaultCode) EnumDescriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{12}
}

type OwnerFrame struct {
	state protoimpl.MessageState `protogen:"open.v1"`
	// Types that are valid to be assigned to Msg:
	//
	//	*OwnerFrame_Claim
	//	*OwnerFrame_Directive
	//	*OwnerFrame_StartAttempt
	//	*OwnerFrame_CancelAttempt
	//	*OwnerFrame_TerminalAck
	//	*OwnerFrame_CheckpointReceipt
	//	*OwnerFrame_SnapshotAck
	Msg           isOwnerFrame_Msg `protobuf_oneof:"msg"`
	unknownFields protoimpl.UnknownFields
	sizeCache     protoimpl.SizeCache
}

func (x *OwnerFrame) Reset() {
	*x = OwnerFrame{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[0]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *OwnerFrame) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*OwnerFrame) ProtoMessage() {}

func (x *OwnerFrame) ProtoReflect() protoreflect.Message {
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

// Deprecated: Use OwnerFrame.ProtoReflect.Descriptor instead.
func (*OwnerFrame) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{0}
}

func (x *OwnerFrame) GetMsg() isOwnerFrame_Msg {
	if x != nil {
		return x.Msg
	}
	return nil
}

func (x *OwnerFrame) GetClaim() *Claim {
	if x != nil {
		if x, ok := x.Msg.(*OwnerFrame_Claim); ok {
			return x.Claim
		}
	}
	return nil
}

func (x *OwnerFrame) GetDirective() *Directive {
	if x != nil {
		if x, ok := x.Msg.(*OwnerFrame_Directive); ok {
			return x.Directive
		}
	}
	return nil
}

func (x *OwnerFrame) GetStartAttempt() *StartAttempt {
	if x != nil {
		if x, ok := x.Msg.(*OwnerFrame_StartAttempt); ok {
			return x.StartAttempt
		}
	}
	return nil
}

func (x *OwnerFrame) GetCancelAttempt() *CancelAttempt {
	if x != nil {
		if x, ok := x.Msg.(*OwnerFrame_CancelAttempt); ok {
			return x.CancelAttempt
		}
	}
	return nil
}

func (x *OwnerFrame) GetTerminalAck() *TerminalAck {
	if x != nil {
		if x, ok := x.Msg.(*OwnerFrame_TerminalAck); ok {
			return x.TerminalAck
		}
	}
	return nil
}

func (x *OwnerFrame) GetCheckpointReceipt() *JobCheckpointReceipt {
	if x != nil {
		if x, ok := x.Msg.(*OwnerFrame_CheckpointReceipt); ok {
			return x.CheckpointReceipt
		}
	}
	return nil
}

func (x *OwnerFrame) GetSnapshotAck() *SnapshotAck {
	if x != nil {
		if x, ok := x.Msg.(*OwnerFrame_SnapshotAck); ok {
			return x.SnapshotAck
		}
	}
	return nil
}

type isOwnerFrame_Msg interface {
	isOwnerFrame_Msg()
}

type OwnerFrame_Claim struct {
	Claim *Claim `protobuf:"bytes,5,opt,name=claim,proto3,oneof"`
}

type OwnerFrame_Directive struct {
	Directive *Directive `protobuf:"bytes,6,opt,name=directive,proto3,oneof"`
}

type OwnerFrame_StartAttempt struct {
	StartAttempt *StartAttempt `protobuf:"bytes,7,opt,name=start_attempt,json=startAttempt,proto3,oneof"`
}

type OwnerFrame_CancelAttempt struct {
	CancelAttempt *CancelAttempt `protobuf:"bytes,8,opt,name=cancel_attempt,json=cancelAttempt,proto3,oneof"`
}

type OwnerFrame_TerminalAck struct {
	TerminalAck *TerminalAck `protobuf:"bytes,9,opt,name=terminal_ack,json=terminalAck,proto3,oneof"`
}

type OwnerFrame_CheckpointReceipt struct {
	CheckpointReceipt *JobCheckpointReceipt `protobuf:"bytes,10,opt,name=checkpoint_receipt,json=checkpointReceipt,proto3,oneof"`
}

type OwnerFrame_SnapshotAck struct {
	SnapshotAck *SnapshotAck `protobuf:"bytes,11,opt,name=snapshot_ack,json=snapshotAck,proto3,oneof"`
}

func (*OwnerFrame_Claim) isOwnerFrame_Msg() {}

func (*OwnerFrame_Directive) isOwnerFrame_Msg() {}

func (*OwnerFrame_StartAttempt) isOwnerFrame_Msg() {}

func (*OwnerFrame_CancelAttempt) isOwnerFrame_Msg() {}

func (*OwnerFrame_TerminalAck) isOwnerFrame_Msg() {}

func (*OwnerFrame_CheckpointReceipt) isOwnerFrame_Msg() {}

func (*OwnerFrame_SnapshotAck) isOwnerFrame_Msg() {}

type WorkerFrame struct {
	state protoimpl.MessageState `protogen:"open.v1"`
	// Types that are valid to be assigned to Msg:
	//
	//	*WorkerFrame_ClaimAck
	//	*WorkerFrame_Report
	//	*WorkerFrame_AttemptAccepted
	//	*WorkerFrame_AttemptTerminal
	//	*WorkerFrame_BootFailure
	//	*WorkerFrame_CheckpointRequest
	//	*WorkerFrame_CheckpointAck
	//	*WorkerFrame_SnapshotBegin
	//	*WorkerFrame_SnapshotEnd
	//	*WorkerFrame_SnapshotEntry
	Msg           isWorkerFrame_Msg `protobuf_oneof:"msg"`
	unknownFields protoimpl.UnknownFields
	sizeCache     protoimpl.SizeCache
}

func (x *WorkerFrame) Reset() {
	*x = WorkerFrame{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[1]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *WorkerFrame) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*WorkerFrame) ProtoMessage() {}

func (x *WorkerFrame) ProtoReflect() protoreflect.Message {
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

// Deprecated: Use WorkerFrame.ProtoReflect.Descriptor instead.
func (*WorkerFrame) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{1}
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

func (x *WorkerFrame) GetReport() *Report {
	if x != nil {
		if x, ok := x.Msg.(*WorkerFrame_Report); ok {
			return x.Report
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

func (x *WorkerFrame) GetAttemptTerminal() *AttemptTerminal {
	if x != nil {
		if x, ok := x.Msg.(*WorkerFrame_AttemptTerminal); ok {
			return x.AttemptTerminal
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

func (x *WorkerFrame) GetSnapshotBegin() *SnapshotBegin {
	if x != nil {
		if x, ok := x.Msg.(*WorkerFrame_SnapshotBegin); ok {
			return x.SnapshotBegin
		}
	}
	return nil
}

func (x *WorkerFrame) GetSnapshotEnd() *SnapshotEnd {
	if x != nil {
		if x, ok := x.Msg.(*WorkerFrame_SnapshotEnd); ok {
			return x.SnapshotEnd
		}
	}
	return nil
}

func (x *WorkerFrame) GetSnapshotEntry() *SnapshotEntry {
	if x != nil {
		if x, ok := x.Msg.(*WorkerFrame_SnapshotEntry); ok {
			return x.SnapshotEntry
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

type WorkerFrame_Report struct {
	Report *Report `protobuf:"bytes,6,opt,name=report,proto3,oneof"`
}

type WorkerFrame_AttemptAccepted struct {
	AttemptAccepted *AttemptAccepted `protobuf:"bytes,7,opt,name=attempt_accepted,json=attemptAccepted,proto3,oneof"`
}

type WorkerFrame_AttemptTerminal struct {
	AttemptTerminal *AttemptTerminal `protobuf:"bytes,8,opt,name=attempt_terminal,json=attemptTerminal,proto3,oneof"`
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

type WorkerFrame_SnapshotBegin struct {
	SnapshotBegin *SnapshotBegin `protobuf:"bytes,12,opt,name=snapshot_begin,json=snapshotBegin,proto3,oneof"`
}

type WorkerFrame_SnapshotEnd struct {
	SnapshotEnd *SnapshotEnd `protobuf:"bytes,13,opt,name=snapshot_end,json=snapshotEnd,proto3,oneof"`
}

type WorkerFrame_SnapshotEntry struct {
	SnapshotEntry *SnapshotEntry `protobuf:"bytes,15,opt,name=snapshot_entry,json=snapshotEntry,proto3,oneof"`
}

func (*WorkerFrame_ClaimAck) isWorkerFrame_Msg() {}

func (*WorkerFrame_Report) isWorkerFrame_Msg() {}

func (*WorkerFrame_AttemptAccepted) isWorkerFrame_Msg() {}

func (*WorkerFrame_AttemptTerminal) isWorkerFrame_Msg() {}

func (*WorkerFrame_BootFailure) isWorkerFrame_Msg() {}

func (*WorkerFrame_CheckpointRequest) isWorkerFrame_Msg() {}

func (*WorkerFrame_CheckpointAck) isWorkerFrame_Msg() {}

func (*WorkerFrame_SnapshotBegin) isWorkerFrame_Msg() {}

func (*WorkerFrame_SnapshotEnd) isWorkerFrame_Msg() {}

func (*WorkerFrame_SnapshotEntry) isWorkerFrame_Msg() {}

type Claim struct {
	state             protoimpl.MessageState `protogen:"open.v1"`
	OwnerEpoch        uint64                 `protobuf:"varint,1,opt,name=owner_epoch,json=ownerEpoch,proto3" json:"owner_epoch,omitempty"`                      // envelope; the claimed authority generation
	ControlGeneration uint64                 `protobuf:"varint,2,opt,name=control_generation,json=controlGeneration,proto3" json:"control_generation,omitempty"` // 0 on Claim (none minted yet)
	WorkerBootId      string                 `protobuf:"bytes,3,opt,name=worker_boot_id,json=workerBootId,proto3" json:"worker_boot_id,omitempty"`               // empty on a fresh dial; set when re-dialing a known boot
	ControllerId      string                 `protobuf:"bytes,5,opt,name=controller_id,json=controllerId,proto3" json:"controller_id,omitempty"`                 // stable identity of the claiming owner
	WorkerId          string                 `protobuf:"bytes,6,opt,name=worker_id,json=workerId,proto3" json:"worker_id,omitempty"`                             // the worker identity the owner expects to be claiming
	WireMinor         uint32                 `protobuf:"varint,7,opt,name=wire_minor,json=wireMinor,proto3" json:"wire_minor,omitempty"`                         // highest minor the OWNER implements
	Proof             []byte                 `protobuf:"bytes,8,opt,name=proof,proto3" json:"proof,omitempty"`                                                   // controller-authority proof where the transport does not
	unknownFields     protoimpl.UnknownFields
	sizeCache         protoimpl.SizeCache
}

func (x *Claim) Reset() {
	*x = Claim{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[2]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *Claim) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*Claim) ProtoMessage() {}

func (x *Claim) ProtoReflect() protoreflect.Message {
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

// Deprecated: Use Claim.ProtoReflect.Descriptor instead.
func (*Claim) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{2}
}

func (x *Claim) GetOwnerEpoch() uint64 {
	if x != nil {
		return x.OwnerEpoch
	}
	return 0
}

func (x *Claim) GetControlGeneration() uint64 {
	if x != nil {
		return x.ControlGeneration
	}
	return 0
}

func (x *Claim) GetWorkerBootId() string {
	if x != nil {
		return x.WorkerBootId
	}
	return ""
}

func (x *Claim) GetControllerId() string {
	if x != nil {
		return x.ControllerId
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

// Claim acceptance: the worker fences the previous stream (if any), increments
// control_generation, and answers with its identity + a recovery snapshot (SnapshotBegin..End
// follow immediately). DISPATCH STAYS CLOSED until the owner's SnapshotAck (02 §6).
type ClaimAck struct {
	state             protoimpl.MessageState `protogen:"open.v1"`
	OwnerEpoch        uint64                 `protobuf:"varint,1,opt,name=owner_epoch,json=ownerEpoch,proto3" json:"owner_epoch,omitempty"`                      // echo of the accepted epoch
	ControlGeneration uint64                 `protobuf:"varint,2,opt,name=control_generation,json=controlGeneration,proto3" json:"control_generation,omitempty"` // newly minted for this stream
	WorkerBootId      string                 `protobuf:"bytes,3,opt,name=worker_boot_id,json=workerBootId,proto3" json:"worker_boot_id,omitempty"`               // this boot's identity
	Accepted          bool                   `protobuf:"varint,5,opt,name=accepted,proto3" json:"accepted,omitempty"`                                            // false => `rejection` set; stream then closes
	Rejection         ClaimRejection         `protobuf:"varint,6,opt,name=rejection,proto3,enum=cozy.worker.v1.ClaimRejection" json:"rejection,omitempty"`
	WireMinor         uint32                 `protobuf:"varint,7,opt,name=wire_minor,json=wireMinor,proto3" json:"wire_minor,omitempty"` // highest minor the WORKER implements
	WorkerId          string                 `protobuf:"bytes,8,opt,name=worker_id,json=workerId,proto3" json:"worker_id,omitempty"`     // CREDENTIAL subject; cross-checked vs the worker's server
	// identity (02 §4)
	InstanceId string `protobuf:"bytes,9,opt,name=instance_id,json=instanceId,proto3" json:"instance_id,omitempty"` // ONE provisioned instance lifetime; terminal-replay
	// authorization keys on this, never worker_id
	ReleaseId     string           `protobuf:"bytes,10,opt,name=release_id,json=releaseId,proto3" json:"release_id,omitempty"`
	ImageDigest   string           `protobuf:"bytes,11,opt,name=image_digest,json=imageDigest,proto3" json:"image_digest,omitempty"` // class (b), `sha256:<hex>` (provenance)
	GitCommit     string           `protobuf:"bytes,12,opt,name=git_commit,json=gitCommit,proto3" json:"git_commit,omitempty"`
	Resources     *WorkerResources `protobuf:"bytes,13,opt,name=resources,proto3" json:"resources,omitempty"` // torch-free supervisor statics; never re-sent mid-stream.
	unknownFields protoimpl.UnknownFields
	sizeCache     protoimpl.SizeCache
}

func (x *ClaimAck) Reset() {
	*x = ClaimAck{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[3]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *ClaimAck) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*ClaimAck) ProtoMessage() {}

func (x *ClaimAck) ProtoReflect() protoreflect.Message {
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

// Deprecated: Use ClaimAck.ProtoReflect.Descriptor instead.
func (*ClaimAck) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{3}
}

func (x *ClaimAck) GetOwnerEpoch() uint64 {
	if x != nil {
		return x.OwnerEpoch
	}
	return 0
}

func (x *ClaimAck) GetControlGeneration() uint64 {
	if x != nil {
		return x.ControlGeneration
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

func (x *ClaimAck) GetInstanceId() string {
	if x != nil {
		return x.InstanceId
	}
	return ""
}

func (x *ClaimAck) GetReleaseId() string {
	if x != nil {
		return x.ReleaseId
	}
	return ""
}

func (x *ClaimAck) GetImageDigest() string {
	if x != nil {
		return x.ImageDigest
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
	state             protoimpl.MessageState `protogen:"open.v1"`
	OwnerEpoch        uint64                 `protobuf:"varint,1,opt,name=owner_epoch,json=ownerEpoch,proto3" json:"owner_epoch,omitempty"`
	ControlGeneration uint64                 `protobuf:"varint,2,opt,name=control_generation,json=controlGeneration,proto3" json:"control_generation,omitempty"`
	WorkerBootId      string                 `protobuf:"bytes,3,opt,name=worker_boot_id,json=workerBootId,proto3" json:"worker_boot_id,omitempty"`
	WorkerId          string                 `protobuf:"bytes,5,opt,name=worker_id,json=workerId,proto3" json:"worker_id,omitempty"`
	InstanceId        string                 `protobuf:"bytes,6,opt,name=instance_id,json=instanceId,proto3" json:"instance_id,omitempty"`
	Reason            BootFailureReason      `protobuf:"varint,7,opt,name=reason,proto3,enum=cozy.worker.v1.BootFailureReason" json:"reason,omitempty"`
	Detail            string                 `protobuf:"bytes,8,opt,name=detail,proto3" json:"detail,omitempty"`       // bounded <= 1024 bytes, sanitized
	Resources         *WorkerResources       `protobuf:"bytes,9,opt,name=resources,proto3" json:"resources,omitempty"` // whatever was measurable
	ImageDigest       string                 `protobuf:"bytes,10,opt,name=image_digest,json=imageDigest,proto3" json:"image_digest,omitempty"`
	ReleaseId         string                 `protobuf:"bytes,11,opt,name=release_id,json=releaseId,proto3" json:"release_id,omitempty"`
	unknownFields     protoimpl.UnknownFields
	sizeCache         protoimpl.SizeCache
}

func (x *BootFailure) Reset() {
	*x = BootFailure{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[4]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *BootFailure) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*BootFailure) ProtoMessage() {}

func (x *BootFailure) ProtoReflect() protoreflect.Message {
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

// Deprecated: Use BootFailure.ProtoReflect.Descriptor instead.
func (*BootFailure) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{4}
}

func (x *BootFailure) GetOwnerEpoch() uint64 {
	if x != nil {
		return x.OwnerEpoch
	}
	return 0
}

func (x *BootFailure) GetControlGeneration() uint64 {
	if x != nil {
		return x.ControlGeneration
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

func (x *BootFailure) GetInstanceId() string {
	if x != nil {
		return x.InstanceId
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

func (x *BootFailure) GetImageDigest() string {
	if x != nil {
		return x.ImageDigest
	}
	return ""
}

func (x *BootFailure) GetReleaseId() string {
	if x != nil {
		return x.ReleaseId
	}
	return ""
}

type SnapshotBegin struct {
	state             protoimpl.MessageState `protogen:"open.v1"`
	OwnerEpoch        uint64                 `protobuf:"varint,1,opt,name=owner_epoch,json=ownerEpoch,proto3" json:"owner_epoch,omitempty"`
	ControlGeneration uint64                 `protobuf:"varint,2,opt,name=control_generation,json=controlGeneration,proto3" json:"control_generation,omitempty"`
	WorkerBootId      string                 `protobuf:"bytes,3,opt,name=worker_boot_id,json=workerBootId,proto3" json:"worker_boot_id,omitempty"`
	SnapshotId        string                 `protobuf:"bytes,5,opt,name=snapshot_id,json=snapshotId,proto3" json:"snapshot_id,omitempty"`                    // worker-minted; names this exact snapshot
	JournalHighwater  uint64                 `protobuf:"varint,6,opt,name=journal_highwater,json=journalHighwater,proto3" json:"journal_highwater,omitempty"` // the journal position this snapshot reflects
	unknownFields     protoimpl.UnknownFields
	sizeCache         protoimpl.SizeCache
}

func (x *SnapshotBegin) Reset() {
	*x = SnapshotBegin{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[5]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *SnapshotBegin) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*SnapshotBegin) ProtoMessage() {}

func (x *SnapshotBegin) ProtoReflect() protoreflect.Message {
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

// Deprecated: Use SnapshotBegin.ProtoReflect.Descriptor instead.
func (*SnapshotBegin) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{5}
}

func (x *SnapshotBegin) GetOwnerEpoch() uint64 {
	if x != nil {
		return x.OwnerEpoch
	}
	return 0
}

func (x *SnapshotBegin) GetControlGeneration() uint64 {
	if x != nil {
		return x.ControlGeneration
	}
	return 0
}

func (x *SnapshotBegin) GetWorkerBootId() string {
	if x != nil {
		return x.WorkerBootId
	}
	return ""
}

func (x *SnapshotBegin) GetSnapshotId() string {
	if x != nil {
		return x.SnapshotId
	}
	return ""
}

func (x *SnapshotBegin) GetJournalHighwater() uint64 {
	if x != nil {
		return x.JournalHighwater
	}
	return 0
}

type SnapshotEntry struct {
	state             protoimpl.MessageState `protogen:"open.v1"`
	OwnerEpoch        uint64                 `protobuf:"varint,1,opt,name=owner_epoch,json=ownerEpoch,proto3" json:"owner_epoch,omitempty"`
	ControlGeneration uint64                 `protobuf:"varint,2,opt,name=control_generation,json=controlGeneration,proto3" json:"control_generation,omitempty"`
	WorkerBootId      string                 `protobuf:"bytes,3,opt,name=worker_boot_id,json=workerBootId,proto3" json:"worker_boot_id,omitempty"`
	SnapshotId        string                 `protobuf:"bytes,5,opt,name=snapshot_id,json=snapshotId,proto3" json:"snapshot_id,omitempty"`
	Attempt           *ActiveAttempt         `protobuf:"bytes,6,opt,name=attempt,proto3" json:"attempt,omitempty"` // one recovered/held attempt (accepted, or terminal
	unknownFields     protoimpl.UnknownFields
	sizeCache         protoimpl.SizeCache
}

func (x *SnapshotEntry) Reset() {
	*x = SnapshotEntry{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[6]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *SnapshotEntry) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*SnapshotEntry) ProtoMessage() {}

func (x *SnapshotEntry) ProtoReflect() protoreflect.Message {
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

// Deprecated: Use SnapshotEntry.ProtoReflect.Descriptor instead.
func (*SnapshotEntry) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{6}
}

func (x *SnapshotEntry) GetOwnerEpoch() uint64 {
	if x != nil {
		return x.OwnerEpoch
	}
	return 0
}

func (x *SnapshotEntry) GetControlGeneration() uint64 {
	if x != nil {
		return x.ControlGeneration
	}
	return 0
}

func (x *SnapshotEntry) GetWorkerBootId() string {
	if x != nil {
		return x.WorkerBootId
	}
	return ""
}

func (x *SnapshotEntry) GetSnapshotId() string {
	if x != nil {
		return x.SnapshotId
	}
	return ""
}

func (x *SnapshotEntry) GetAttempt() *ActiveAttempt {
	if x != nil {
		return x.Attempt
	}
	return nil
}

type SnapshotEnd struct {
	state             protoimpl.MessageState `protogen:"open.v1"`
	OwnerEpoch        uint64                 `protobuf:"varint,1,opt,name=owner_epoch,json=ownerEpoch,proto3" json:"owner_epoch,omitempty"`
	ControlGeneration uint64                 `protobuf:"varint,2,opt,name=control_generation,json=controlGeneration,proto3" json:"control_generation,omitempty"`
	WorkerBootId      string                 `protobuf:"bytes,3,opt,name=worker_boot_id,json=workerBootId,proto3" json:"worker_boot_id,omitempty"`
	SnapshotId        string                 `protobuf:"bytes,5,opt,name=snapshot_id,json=snapshotId,proto3" json:"snapshot_id,omitempty"`
	EntryCount        uint32                 `protobuf:"varint,6,opt,name=entry_count,json=entryCount,proto3" json:"entry_count,omitempty"` // must equal the entries sent; a gap is a refusal
	unknownFields     protoimpl.UnknownFields
	sizeCache         protoimpl.SizeCache
}

func (x *SnapshotEnd) Reset() {
	*x = SnapshotEnd{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[7]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *SnapshotEnd) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*SnapshotEnd) ProtoMessage() {}

func (x *SnapshotEnd) ProtoReflect() protoreflect.Message {
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

// Deprecated: Use SnapshotEnd.ProtoReflect.Descriptor instead.
func (*SnapshotEnd) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{7}
}

func (x *SnapshotEnd) GetOwnerEpoch() uint64 {
	if x != nil {
		return x.OwnerEpoch
	}
	return 0
}

func (x *SnapshotEnd) GetControlGeneration() uint64 {
	if x != nil {
		return x.ControlGeneration
	}
	return 0
}

func (x *SnapshotEnd) GetWorkerBootId() string {
	if x != nil {
		return x.WorkerBootId
	}
	return ""
}

func (x *SnapshotEnd) GetSnapshotId() string {
	if x != nil {
		return x.SnapshotId
	}
	return ""
}

func (x *SnapshotEnd) GetEntryCount() uint32 {
	if x != nil {
		return x.EntryCount
	}
	return 0
}

type SnapshotAck struct {
	state             protoimpl.MessageState `protogen:"open.v1"`
	OwnerEpoch        uint64                 `protobuf:"varint,1,opt,name=owner_epoch,json=ownerEpoch,proto3" json:"owner_epoch,omitempty"`
	ControlGeneration uint64                 `protobuf:"varint,2,opt,name=control_generation,json=controlGeneration,proto3" json:"control_generation,omitempty"`
	WorkerBootId      string                 `protobuf:"bytes,3,opt,name=worker_boot_id,json=workerBootId,proto3" json:"worker_boot_id,omitempty"`
	SnapshotId        string                 `protobuf:"bytes,5,opt,name=snapshot_id,json=snapshotId,proto3" json:"snapshot_id,omitempty"` // the exact snapshot durably reconciled; dispatch opens
	unknownFields     protoimpl.UnknownFields
	sizeCache         protoimpl.SizeCache
}

func (x *SnapshotAck) Reset() {
	*x = SnapshotAck{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[8]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *SnapshotAck) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*SnapshotAck) ProtoMessage() {}

func (x *SnapshotAck) ProtoReflect() protoreflect.Message {
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

// Deprecated: Use SnapshotAck.ProtoReflect.Descriptor instead.
func (*SnapshotAck) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{8}
}

func (x *SnapshotAck) GetOwnerEpoch() uint64 {
	if x != nil {
		return x.OwnerEpoch
	}
	return 0
}

func (x *SnapshotAck) GetControlGeneration() uint64 {
	if x != nil {
		return x.ControlGeneration
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

type Directive struct {
	state             protoimpl.MessageState `protogen:"open.v1"`
	OwnerEpoch        uint64                 `protobuf:"varint,1,opt,name=owner_epoch,json=ownerEpoch,proto3" json:"owner_epoch,omitempty"`
	ControlGeneration uint64                 `protobuf:"varint,2,opt,name=control_generation,json=controlGeneration,proto3" json:"control_generation,omitempty"`
	WorkerBootId      string                 `protobuf:"bytes,3,opt,name=worker_boot_id,json=workerBootId,proto3" json:"worker_boot_id,omitempty"`
	Revision          uint64                 `protobuf:"varint,5,opt,name=revision,proto3" json:"revision,omitempty"` // owner-owned, monotonic; older ignored, equal idempotent,
	// newer applies; equal revision + different body is a
	// protocol error (02 §3)
	Posture      Posture `protobuf:"varint,6,opt,name=posture,proto3,enum=cozy.worker.v1.Posture" json:"posture,omitempty"`
	WireMinor    uint32  `protobuf:"varint,7,opt,name=wire_minor,json=wireMinor,proto3" json:"wire_minor,omitempty"`            // the minor the owner speaks on this stream; echoed
	DrainGraceMs uint64  `protobuf:"varint,8,opt,name=drain_grace_ms,json=drainGraceMs,proto3" json:"drain_grace_ms,omitempty"` // draining posture only: grace before forceful; 0 = the
	// worker's runtime-constant default
	//
	// Types that are valid to be assigned to Mode:
	//
	//	*Directive_Job
	//	*Directive_DeploymentSet
	Mode          isDirective_Mode `protobuf_oneof:"mode"`
	unknownFields protoimpl.UnknownFields
	sizeCache     protoimpl.SizeCache
}

func (x *Directive) Reset() {
	*x = Directive{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[9]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *Directive) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*Directive) ProtoMessage() {}

func (x *Directive) ProtoReflect() protoreflect.Message {
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

// Deprecated: Use Directive.ProtoReflect.Descriptor instead.
func (*Directive) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{9}
}

func (x *Directive) GetOwnerEpoch() uint64 {
	if x != nil {
		return x.OwnerEpoch
	}
	return 0
}

func (x *Directive) GetControlGeneration() uint64 {
	if x != nil {
		return x.ControlGeneration
	}
	return 0
}

func (x *Directive) GetWorkerBootId() string {
	if x != nil {
		return x.WorkerBootId
	}
	return ""
}

func (x *Directive) GetRevision() uint64 {
	if x != nil {
		return x.Revision
	}
	return 0
}

func (x *Directive) GetPosture() Posture {
	if x != nil {
		return x.Posture
	}
	return Posture_POSTURE_UNSPECIFIED
}

func (x *Directive) GetWireMinor() uint32 {
	if x != nil {
		return x.WireMinor
	}
	return 0
}

func (x *Directive) GetDrainGraceMs() uint64 {
	if x != nil {
		return x.DrainGraceMs
	}
	return 0
}

func (x *Directive) GetMode() isDirective_Mode {
	if x != nil {
		return x.Mode
	}
	return nil
}

func (x *Directive) GetJob() *JobDirective {
	if x != nil {
		if x, ok := x.Mode.(*Directive_Job); ok {
			return x.Job
		}
	}
	return nil
}

func (x *Directive) GetDeploymentSet() *DeploymentSetDirective {
	if x != nil {
		if x, ok := x.Mode.(*Directive_DeploymentSet); ok {
			return x.DeploymentSet
		}
	}
	return nil
}

type isDirective_Mode interface {
	isDirective_Mode()
}

type Directive_Job struct {
	Job *JobDirective `protobuf:"bytes,13,opt,name=job,proto3,oneof"`
}

type Directive_DeploymentSet struct {
	DeploymentSet *DeploymentSetDirective `protobuf:"bytes,15,opt,name=deployment_set,json=deploymentSet,proto3,oneof"`
}

func (*Directive_Job) isDirective_Mode() {}

func (*Directive_DeploymentSet) isDirective_Mode() {}

// The serving-mode desired state: a digest-addressed DeploymentSet. The owner authors BOTH the
// structured set and its document digest (ONE-DIGESTER RULE: the worker ECHOES the digest in
// Report.applied_set_digest and never recomputes it — the digest is the set's NAME for status
// attribution, not a byte fence).
type DeploymentSetDirective struct {
	state     protoimpl.MessageState `protogen:"open.v1"`
	SetDigest []byte                 `protobuf:"bytes,1,opt,name=set_digest,json=setDigest,proto3" json:"set_digest,omitempty"` // class (a): sha256 of the canonical DeploymentSet/1
	// document below
	Set           *DeploymentSet `protobuf:"bytes,2,opt,name=set,proto3" json:"set,omitempty"`
	unknownFields protoimpl.UnknownFields
	sizeCache     protoimpl.SizeCache
}

func (x *DeploymentSetDirective) Reset() {
	*x = DeploymentSetDirective{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[10]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *DeploymentSetDirective) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*DeploymentSetDirective) ProtoMessage() {}

func (x *DeploymentSetDirective) ProtoReflect() protoreflect.Message {
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

// Deprecated: Use DeploymentSetDirective.ProtoReflect.Descriptor instead.
func (*DeploymentSetDirective) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{10}
}

func (x *DeploymentSetDirective) GetSetDigest() []byte {
	if x != nil {
		return x.SetDigest
	}
	return nil
}

func (x *DeploymentSetDirective) GetSet() *DeploymentSet {
	if x != nil {
		return x.Set
	}
	return nil
}

// DOCUMENT SHAPE (also carried structured on the wire): canonical form
// `cozy.worker.v1.DeploymentSet/1`. Deployments sorted by deployment_id.
// LAUNCH ENFORCES len(deployments) <= 1 (header note): a longer set refuses typed
// (FAULT_KIND_DEPLOYMENT_SET_UNSUPPORTED) and the directive is UNAPPLIED — applied_revision
// does not advance.
type DeploymentSet struct {
	state         protoimpl.MessageState `protogen:"open.v1"`
	Deployments   []*Deployment          `protobuf:"bytes,1,rep,name=deployments,proto3" json:"deployments,omitempty"`
	unknownFields protoimpl.UnknownFields
	sizeCache     protoimpl.SizeCache
}

func (x *DeploymentSet) Reset() {
	*x = DeploymentSet{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[11]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *DeploymentSet) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*DeploymentSet) ProtoMessage() {}

func (x *DeploymentSet) ProtoReflect() protoreflect.Message {
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

// Deprecated: Use DeploymentSet.ProtoReflect.Descriptor instead.
func (*DeploymentSet) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{11}
}

func (x *DeploymentSet) GetDeployments() []*Deployment {
	if x != nil {
		return x.Deployments
	}
	return nil
}

type Deployment struct {
	state protoimpl.MessageState `protogen:"open.v1"`
	// until the build unit lands; future owner
	// tensorhub-build.md / th-038 / cl-017
	DeploymentId string `protobuf:"bytes,1,opt,name=deployment_id,json=deploymentId,proto3" json:"deployment_id,omitempty"` // owner-minted routing + journal key (#444); NEVER part of
	// invocation identity (InvocationSpec digests exclude it)
	EndpointReleaseId        string   `protobuf:"bytes,2,opt,name=endpoint_release_id,json=endpointReleaseId,proto3" json:"endpoint_release_id,omitempty"`
	EntrypointBindingPlanIds []string `protobuf:"bytes,3,rep,name=entrypoint_binding_plan_ids,json=entrypointBindingPlanIds,proto3" json:"entrypoint_binding_plan_ids,omitempty"` // expected set; sorted lexicographic
	unknownFields            protoimpl.UnknownFields
	sizeCache                protoimpl.SizeCache
}

func (x *Deployment) Reset() {
	*x = Deployment{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[12]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *Deployment) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*Deployment) ProtoMessage() {}

func (x *Deployment) ProtoReflect() protoreflect.Message {
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

// Deprecated: Use Deployment.ProtoReflect.Descriptor instead.
func (*Deployment) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{12}
}

func (x *Deployment) GetDeploymentId() string {
	if x != nil {
		return x.DeploymentId
	}
	return ""
}

func (x *Deployment) GetEndpointReleaseId() string {
	if x != nil {
		return x.EndpointReleaseId
	}
	return ""
}

func (x *Deployment) GetEntrypointBindingPlanIds() []string {
	if x != nil {
		return x.EntrypointBindingPlanIds
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
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[13]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *JobDirective) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*JobDirective) ProtoMessage() {}

func (x *JobDirective) ProtoReflect() protoreflect.Message {
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

// Deprecated: Use JobDirective.ProtoReflect.Descriptor instead.
func (*JobDirective) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{13}
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

// ONE-DIGESTER RULE unchanged: the owner canonicalizes; the worker echoes plain fields.
// Serving truth is PER-DEPLOYMENT (#446): `deployments` carries one DeploymentStatus per
// hosted deployment; `intake_state` here is MACHINE scope (the supervisor's own lifecycle).
// Modes exclusive: serving mode populates deployments; job mode populates job_capacity.
type Report struct {
	state             protoimpl.MessageState `protogen:"open.v1"`
	OwnerEpoch        uint64                 `protobuf:"varint,1,opt,name=owner_epoch,json=ownerEpoch,proto3" json:"owner_epoch,omitempty"`
	ControlGeneration uint64                 `protobuf:"varint,2,opt,name=control_generation,json=controlGeneration,proto3" json:"control_generation,omitempty"`
	WorkerBootId      string                 `protobuf:"bytes,3,opt,name=worker_boot_id,json=workerBootId,proto3" json:"worker_boot_id,omitempty"`
	AppliedRevision   uint64                 `protobuf:"varint,5,opt,name=applied_revision,json=appliedRevision,proto3" json:"applied_revision,omitempty"`
	IntakeState       IntakeState            `protobuf:"varint,6,opt,name=intake_state,json=intakeState,proto3,enum=cozy.worker.v1.IntakeState" json:"intake_state,omitempty"` // MACHINE scope: the supervisor's lifecycle. Per-deployment
	// READY (first_request_servable) lives in DeploymentStatus.
	AppliedWireMinor uint32           `protobuf:"varint,8,opt,name=applied_wire_minor,json=appliedWireMinor,proto3" json:"applied_wire_minor,omitempty"` // stale echo is a visible skew fact, never a refusal
	ActiveAttempts   []*ActiveAttempt `protobuf:"bytes,9,rep,name=active_attempts,json=activeAttempts,proto3" json:"active_attempts,omitempty"`          // running + completed-with-unshipped-result
	Faults           []*Fault         `protobuf:"bytes,10,rep,name=faults,proto3" json:"faults,omitempty"`                                               // MACHINE-scope faults; deployment faults ride their status
	Activity         []*ActivityEvent `protobuf:"bytes,11,rep,name=activity,proto3" json:"activity,omitempty"`                                           // the non-attempt activity lane (durable)
	JobCapacity      *JobCapacity     `protobuf:"bytes,13,opt,name=job_capacity,json=jobCapacity,proto3" json:"job_capacity,omitempty"`                  // job mode only
	AppliedSetDigest []byte           `protobuf:"bytes,14,opt,name=applied_set_digest,json=appliedSetDigest,proto3" json:"applied_set_digest,omitempty"` // echo of the applied DeploymentSetDirective.set_digest;
	// empty until a set applies (serving mode only)
	Deployments   []*DeploymentStatus `protobuf:"bytes,15,rep,name=deployments,proto3" json:"deployments,omitempty"` // serving mode only
	unknownFields protoimpl.UnknownFields
	sizeCache     protoimpl.SizeCache
}

func (x *Report) Reset() {
	*x = Report{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[14]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *Report) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*Report) ProtoMessage() {}

func (x *Report) ProtoReflect() protoreflect.Message {
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

// Deprecated: Use Report.ProtoReflect.Descriptor instead.
func (*Report) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{14}
}

func (x *Report) GetOwnerEpoch() uint64 {
	if x != nil {
		return x.OwnerEpoch
	}
	return 0
}

func (x *Report) GetControlGeneration() uint64 {
	if x != nil {
		return x.ControlGeneration
	}
	return 0
}

func (x *Report) GetWorkerBootId() string {
	if x != nil {
		return x.WorkerBootId
	}
	return ""
}

func (x *Report) GetAppliedRevision() uint64 {
	if x != nil {
		return x.AppliedRevision
	}
	return 0
}

func (x *Report) GetIntakeState() IntakeState {
	if x != nil {
		return x.IntakeState
	}
	return IntakeState_INTAKE_STATE_UNSPECIFIED
}

func (x *Report) GetAppliedWireMinor() uint32 {
	if x != nil {
		return x.AppliedWireMinor
	}
	return 0
}

func (x *Report) GetActiveAttempts() []*ActiveAttempt {
	if x != nil {
		return x.ActiveAttempts
	}
	return nil
}

func (x *Report) GetFaults() []*Fault {
	if x != nil {
		return x.Faults
	}
	return nil
}

func (x *Report) GetActivity() []*ActivityEvent {
	if x != nil {
		return x.Activity
	}
	return nil
}

func (x *Report) GetJobCapacity() *JobCapacity {
	if x != nil {
		return x.JobCapacity
	}
	return nil
}

func (x *Report) GetAppliedSetDigest() []byte {
	if x != nil {
		return x.AppliedSetDigest
	}
	return nil
}

func (x *Report) GetDeployments() []*DeploymentStatus {
	if x != nil {
		return x.Deployments
	}
	return nil
}

// Per-deployment truth (#446): status, readiness, capacity credits, faults, executor
// generation, and — once this deployment's executor has probed the device — the qualified
// accelerator facts (#430: measured at boot, never inferred).
type DeploymentStatus struct {
	state          protoimpl.MessageState `protogen:"open.v1"`
	DeploymentId   string                 `protobuf:"bytes,1,opt,name=deployment_id,json=deploymentId,proto3" json:"deployment_id,omitempty"`
	IntakeState    IntakeState            `protobuf:"varint,2,opt,name=intake_state,json=intakeState,proto3,enum=cozy.worker.v1.IntakeState" json:"intake_state,omitempty"` // READY means first_request_servable for THIS deployment
	ReadinessEpoch uint64                 `protobuf:"varint,3,opt,name=readiness_epoch,json=readinessEpoch,proto3" json:"readiness_epoch,omitempty"`                        // bumps whenever readiness/capacity meaning changes;
	// credits are bound to it (#441)
	ExecutorGeneration uint64 `protobuf:"varint,4,opt,name=executor_generation,json=executorGeneration,proto3" json:"executor_generation,omitempty"` // THIS deployment's executor fence; bumps on its respawn
	AttemptCredits     uint32 `protobuf:"varint,5,opt,name=attempt_credits,json=attemptCredits,proto3" json:"attempt_credits,omitempty"`             // bounded admission credits at this readiness_epoch (#441);
	// replaces headroom byte counts
	ReadyEntrypointBindingPlanIds    []string                  `protobuf:"bytes,6,rep,name=ready_entrypoint_binding_plan_ids,json=readyEntrypointBindingPlanIds,proto3" json:"ready_entrypoint_binding_plan_ids,omitempty"`
	LoadableEntrypointBindingPlanIds []string                  `protobuf:"bytes,7,rep,name=loadable_entrypoint_binding_plan_ids,json=loadableEntrypointBindingPlanIds,proto3" json:"loadable_entrypoint_binding_plan_ids,omitempty"` // DISJOINT from ready
	Faults                           []*Fault                  `protobuf:"bytes,8,rep,name=faults,proto3" json:"faults,omitempty"`                                                                                                   // deployment-scoped
	Accelerator                      *AcceleratorQualification `protobuf:"bytes,9,opt,name=accelerator,proto3" json:"accelerator,omitempty"`                                                                                         // absent = present-but-unqualified (#446)
	unknownFields                    protoimpl.UnknownFields
	sizeCache                        protoimpl.SizeCache
}

func (x *DeploymentStatus) Reset() {
	*x = DeploymentStatus{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[15]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *DeploymentStatus) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*DeploymentStatus) ProtoMessage() {}

func (x *DeploymentStatus) ProtoReflect() protoreflect.Message {
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

// Deprecated: Use DeploymentStatus.ProtoReflect.Descriptor instead.
func (*DeploymentStatus) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{15}
}

func (x *DeploymentStatus) GetDeploymentId() string {
	if x != nil {
		return x.DeploymentId
	}
	return ""
}

func (x *DeploymentStatus) GetIntakeState() IntakeState {
	if x != nil {
		return x.IntakeState
	}
	return IntakeState_INTAKE_STATE_UNSPECIFIED
}

func (x *DeploymentStatus) GetReadinessEpoch() uint64 {
	if x != nil {
		return x.ReadinessEpoch
	}
	return 0
}

func (x *DeploymentStatus) GetExecutorGeneration() uint64 {
	if x != nil {
		return x.ExecutorGeneration
	}
	return 0
}

func (x *DeploymentStatus) GetAttemptCredits() uint32 {
	if x != nil {
		return x.AttemptCredits
	}
	return 0
}

func (x *DeploymentStatus) GetReadyEntrypointBindingPlanIds() []string {
	if x != nil {
		return x.ReadyEntrypointBindingPlanIds
	}
	return nil
}

func (x *DeploymentStatus) GetLoadableEntrypointBindingPlanIds() []string {
	if x != nil {
		return x.LoadableEntrypointBindingPlanIds
	}
	return nil
}

func (x *DeploymentStatus) GetFaults() []*Fault {
	if x != nil {
		return x.Faults
	}
	return nil
}

func (x *DeploymentStatus) GetAccelerator() *AcceleratorQualification {
	if x != nil {
		return x.Accelerator
	}
	return nil
}

// Executor-probed capability facts (#430: MEASURED, never inferred from chip generation or OS
// version). Vocabulary IS the AcceleratorFacts record's (#422) — one facts schema, no second
// wire vocabulary. The torch-free supervisor NEVER fabricates these (it cannot measure them);
// until an executor qualifies the device, the accelerator is PRESENT-BUT-UNQUALIFIED and
// this message is absent from DeploymentStatus.
type AcceleratorQualification struct {
	state        protoimpl.MessageState `protogen:"open.v1"`
	Qualified    bool                   `protobuf:"varint,1,opt,name=qualified,proto3" json:"qualified,omitempty"`
	Capabilities []string               `protobuf:"bytes,2,rep,name=capabilities,proto3" json:"capabilities,omitempty"` // sorted facts vocabulary (bf16, fp8_compute,
	// pinned_h2d_streams, non_blocking_safe,
	// attention_upcast_required); presence = measured true
	RecommendedMaxBytes uint64 `protobuf:"varint,3,opt,name=recommended_max_bytes,json=recommendedMaxBytes,proto3" json:"recommended_max_bytes,omitempty"` // the working-set envelope the ledger prices against
	// (unified: torch.mps recommended_max; discrete: device
	// total)
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
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[16]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *AcceleratorQualification) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*AcceleratorQualification) ProtoMessage() {}

func (x *AcceleratorQualification) ProtoReflect() protoreflect.Message {
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

// Deprecated: Use AcceleratorQualification.ProtoReflect.Descriptor instead.
func (*AcceleratorQualification) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{16}
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
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[17]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *ActivityEvent) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*ActivityEvent) ProtoMessage() {}

func (x *ActivityEvent) ProtoReflect() protoreflect.Message {
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

// Deprecated: Use ActivityEvent.ProtoReflect.Descriptor instead.
func (*ActivityEvent) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{17}
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

type StartAttempt struct {
	state             protoimpl.MessageState `protogen:"open.v1"`
	OwnerEpoch        uint64                 `protobuf:"varint,1,opt,name=owner_epoch,json=ownerEpoch,proto3" json:"owner_epoch,omitempty"`
	ControlGeneration uint64                 `protobuf:"varint,2,opt,name=control_generation,json=controlGeneration,proto3" json:"control_generation,omitempty"`
	WorkerBootId      string                 `protobuf:"bytes,3,opt,name=worker_boot_id,json=workerBootId,proto3" json:"worker_boot_id,omitempty"`
	RequestId         string                 `protobuf:"bytes,5,opt,name=request_id,json=requestId,proto3" json:"request_id,omitempty"` // fence (a)
	Attempt           uint64                 `protobuf:"varint,6,opt,name=attempt,proto3" json:"attempt,omitempty"`                     // fence (b); ordinal, owner-bumped only, after the prior
	// attempt's journaled terminal
	InvocationDigest []byte `protobuf:"bytes,7,opt,name=invocation_digest,json=invocationDigest,proto3" json:"invocation_digest,omitempty"` // fence (c), class (a): sha256 over EXACTLY the bytes in
	// 9; the receiver RECOMPUTES and refuses on mismatch
	Grant               *DeliveryGrant `protobuf:"bytes,8,opt,name=grant,proto3" json:"grant,omitempty"`                                                        // expiring, refreshable; OUTSIDE the digest
	InvocationCanonical []byte         `protobuf:"bytes,9,opt,name=invocation_canonical,json=invocationCanonical,proto3" json:"invocation_canonical,omitempty"` // the InvocationSpec DOCUMENT, canonical JSON bytes;
	// immutable for the attempt; parsed under unknown-field
	// REFUSAL
	DeploymentId  string `protobuf:"bytes,10,opt,name=deployment_id,json=deploymentId,proto3" json:"deployment_id,omitempty"` // the target deployment (serving mode; REQUIRED there,
	unknownFields protoimpl.UnknownFields
	sizeCache     protoimpl.SizeCache
}

func (x *StartAttempt) Reset() {
	*x = StartAttempt{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[18]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *StartAttempt) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*StartAttempt) ProtoMessage() {}

func (x *StartAttempt) ProtoReflect() protoreflect.Message {
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

// Deprecated: Use StartAttempt.ProtoReflect.Descriptor instead.
func (*StartAttempt) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{18}
}

func (x *StartAttempt) GetOwnerEpoch() uint64 {
	if x != nil {
		return x.OwnerEpoch
	}
	return 0
}

func (x *StartAttempt) GetControlGeneration() uint64 {
	if x != nil {
		return x.ControlGeneration
	}
	return 0
}

func (x *StartAttempt) GetWorkerBootId() string {
	if x != nil {
		return x.WorkerBootId
	}
	return ""
}

func (x *StartAttempt) GetRequestId() string {
	if x != nil {
		return x.RequestId
	}
	return ""
}

func (x *StartAttempt) GetAttempt() uint64 {
	if x != nil {
		return x.Attempt
	}
	return 0
}

func (x *StartAttempt) GetInvocationDigest() []byte {
	if x != nil {
		return x.InvocationDigest
	}
	return nil
}

func (x *StartAttempt) GetGrant() *DeliveryGrant {
	if x != nil {
		return x.Grant
	}
	return nil
}

func (x *StartAttempt) GetInvocationCanonical() []byte {
	if x != nil {
		return x.InvocationCanonical
	}
	return nil
}

func (x *StartAttempt) GetDeploymentId() string {
	if x != nil {
		return x.DeploymentId
	}
	return ""
}

type AttemptAccepted struct {
	state                   protoimpl.MessageState `protogen:"open.v1"`
	OwnerEpoch              uint64                 `protobuf:"varint,1,opt,name=owner_epoch,json=ownerEpoch,proto3" json:"owner_epoch,omitempty"`
	ControlGeneration       uint64                 `protobuf:"varint,2,opt,name=control_generation,json=controlGeneration,proto3" json:"control_generation,omitempty"`
	WorkerBootId            string                 `protobuf:"bytes,3,opt,name=worker_boot_id,json=workerBootId,proto3" json:"worker_boot_id,omitempty"`
	RequestId               string                 `protobuf:"bytes,5,opt,name=request_id,json=requestId,proto3" json:"request_id,omitempty"`
	Attempt                 uint64                 `protobuf:"varint,6,opt,name=attempt,proto3" json:"attempt,omitempty"`
	InvocationDigest        []byte                 `protobuf:"bytes,7,opt,name=invocation_digest,json=invocationDigest,proto3" json:"invocation_digest,omitempty"`                        // echo
	PlanDigest              []byte                 `protobuf:"bytes,8,opt,name=plan_digest,json=planDigest,proto3" json:"plan_digest,omitempty"`                                          // class (a): AttemptPlan (cr-007), fixed at acceptance
	ModelConstructionDigest []byte                 `protobuf:"bytes,9,opt,name=model_construction_digest,json=modelConstructionDigest,proto3" json:"model_construction_digest,omitempty"` // class (a): ModelConstructionContract (cr-004)
	Plan                    *AttemptPlanSummary    `protobuf:"bytes,10,opt,name=plan,proto3" json:"plan,omitempty"`                                                                       // the CLOSED observable projection of the chosen plan
	DeploymentId            string                 `protobuf:"bytes,11,opt,name=deployment_id,json=deploymentId,proto3" json:"deployment_id,omitempty"`                                   // echo (#446)
	ExecutorGeneration      uint64                 `protobuf:"varint,12,opt,name=executor_generation,json=executorGeneration,proto3" json:"executor_generation,omitempty"`                // the deployment executor generation the attempt runs
	unknownFields           protoimpl.UnknownFields
	sizeCache               protoimpl.SizeCache
}

func (x *AttemptAccepted) Reset() {
	*x = AttemptAccepted{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[19]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *AttemptAccepted) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*AttemptAccepted) ProtoMessage() {}

func (x *AttemptAccepted) ProtoReflect() protoreflect.Message {
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

// Deprecated: Use AttemptAccepted.ProtoReflect.Descriptor instead.
func (*AttemptAccepted) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{19}
}

func (x *AttemptAccepted) GetOwnerEpoch() uint64 {
	if x != nil {
		return x.OwnerEpoch
	}
	return 0
}

func (x *AttemptAccepted) GetControlGeneration() uint64 {
	if x != nil {
		return x.ControlGeneration
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

func (x *AttemptAccepted) GetAttempt() uint64 {
	if x != nil {
		return x.Attempt
	}
	return 0
}

func (x *AttemptAccepted) GetInvocationDigest() []byte {
	if x != nil {
		return x.InvocationDigest
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

func (x *AttemptAccepted) GetDeploymentId() string {
	if x != nil {
		return x.DeploymentId
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
	state                     protoimpl.MessageState `protogen:"open.v1"`
	Delivery                  string                 `protobuf:"bytes,1,opt,name=delivery,proto3" json:"delivery,omitempty"`               // native | float
	Materialization           string                 `protobuf:"bytes,2,opt,name=materialization,proto3" json:"materialization,omitempty"` // aot_decode | staged_decode | jit_decode
	ComputeDtype              string                 `protobuf:"bytes,3,opt,name=compute_dtype,json=computeDtype,proto3" json:"compute_dtype,omitempty"`
	Placement                 string                 `protobuf:"bytes,4,opt,name=placement,proto3" json:"placement,omitempty"` // all_resident | component_staged | host_offload | streamed
	ReservedDeviceMemoryBytes uint64                 `protobuf:"varint,5,opt,name=reserved_device_memory_bytes,json=reservedDeviceMemoryBytes,proto3" json:"reserved_device_memory_bytes,omitempty"`
	ReservedHostBytes         uint64                 `protobuf:"varint,6,opt,name=reserved_host_bytes,json=reservedHostBytes,proto3" json:"reserved_host_bytes,omitempty"`
	DecisionExplanation       string                 `protobuf:"bytes,7,opt,name=decision_explanation,json=decisionExplanation,proto3" json:"decision_explanation,omitempty"` // bounded <= 512 bytes diagnostic prose; not parseable
	unknownFields             protoimpl.UnknownFields
	sizeCache                 protoimpl.SizeCache
}

func (x *AttemptPlanSummary) Reset() {
	*x = AttemptPlanSummary{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[20]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *AttemptPlanSummary) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*AttemptPlanSummary) ProtoMessage() {}

func (x *AttemptPlanSummary) ProtoReflect() protoreflect.Message {
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

// Deprecated: Use AttemptPlanSummary.ProtoReflect.Descriptor instead.
func (*AttemptPlanSummary) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{20}
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

// No deployment_id here (#446): the fence triple resolves against the journaled acceptance,
// which records the deployment.
type CancelAttempt struct {
	state             protoimpl.MessageState `protogen:"open.v1"`
	OwnerEpoch        uint64                 `protobuf:"varint,1,opt,name=owner_epoch,json=ownerEpoch,proto3" json:"owner_epoch,omitempty"`
	ControlGeneration uint64                 `protobuf:"varint,2,opt,name=control_generation,json=controlGeneration,proto3" json:"control_generation,omitempty"`
	WorkerBootId      string                 `protobuf:"bytes,3,opt,name=worker_boot_id,json=workerBootId,proto3" json:"worker_boot_id,omitempty"`
	RequestId         string                 `protobuf:"bytes,5,opt,name=request_id,json=requestId,proto3" json:"request_id,omitempty"`
	Attempt           uint64                 `protobuf:"varint,6,opt,name=attempt,proto3" json:"attempt,omitempty"`
	Reason            CancelReason           `protobuf:"varint,7,opt,name=reason,proto3,enum=cozy.worker.v1.CancelReason" json:"reason,omitempty"`
	GraceMs           uint64                 `protobuf:"varint,8,opt,name=grace_ms,json=graceMs,proto3" json:"grace_ms,omitempty"`                           // operator override; 0 = worker default. Never author-set.
	InvocationDigest  []byte                 `protobuf:"bytes,9,opt,name=invocation_digest,json=invocationDigest,proto3" json:"invocation_digest,omitempty"` // full-triple fence; stale/mismatched cancel is DROPPED
	unknownFields     protoimpl.UnknownFields
	sizeCache         protoimpl.SizeCache
}

func (x *CancelAttempt) Reset() {
	*x = CancelAttempt{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[21]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *CancelAttempt) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*CancelAttempt) ProtoMessage() {}

func (x *CancelAttempt) ProtoReflect() protoreflect.Message {
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

// Deprecated: Use CancelAttempt.ProtoReflect.Descriptor instead.
func (*CancelAttempt) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{21}
}

func (x *CancelAttempt) GetOwnerEpoch() uint64 {
	if x != nil {
		return x.OwnerEpoch
	}
	return 0
}

func (x *CancelAttempt) GetControlGeneration() uint64 {
	if x != nil {
		return x.ControlGeneration
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

func (x *CancelAttempt) GetAttempt() uint64 {
	if x != nil {
		return x.Attempt
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

func (x *CancelAttempt) GetInvocationDigest() []byte {
	if x != nil {
		return x.InvocationDigest
	}
	return nil
}

// The envelope around a journaled TerminalBody document (the ATTEMPT OUTCOME — one immutable
// terminal per attempt; the customer-visible settlement is the owner's RequestClosure and never
// crosses this wire). Body travels as canonical bytes; routing copies must agree with it.
// deployment_id is a routing fact from the journaled acceptance (no document counterpart).
type AttemptTerminal struct {
	state             protoimpl.MessageState `protogen:"open.v1"`
	OwnerEpoch        uint64                 `protobuf:"varint,1,opt,name=owner_epoch,json=ownerEpoch,proto3" json:"owner_epoch,omitempty"`
	ControlGeneration uint64                 `protobuf:"varint,2,opt,name=control_generation,json=controlGeneration,proto3" json:"control_generation,omitempty"`
	WorkerBootId      string                 `protobuf:"bytes,3,opt,name=worker_boot_id,json=workerBootId,proto3" json:"worker_boot_id,omitempty"`
	RequestId         string                 `protobuf:"bytes,5,opt,name=request_id,json=requestId,proto3" json:"request_id,omitempty"`                          // routing copy; must agree with the document
	Attempt           uint64                 `protobuf:"varint,6,opt,name=attempt,proto3" json:"attempt,omitempty"`                                              // routing copy
	InvocationDigest  []byte                 `protobuf:"bytes,7,opt,name=invocation_digest,json=invocationDigest,proto3" json:"invocation_digest,omitempty"`     // routing copy
	TerminalId        string                 `protobuf:"bytes,8,opt,name=terminal_id,json=terminalId,proto3" json:"terminal_id,omitempty"`                       // WORKER-minted observation id; stable across replays
	TerminalDigest    []byte                 `protobuf:"bytes,9,opt,name=terminal_digest,json=terminalDigest,proto3" json:"terminal_digest,omitempty"`           // class (a): sha256 over EXACTLY the bytes in 10
	TerminalCanonical []byte                 `protobuf:"bytes,10,opt,name=terminal_canonical,json=terminalCanonical,proto3" json:"terminal_canonical,omitempty"` // the TerminalBody DOCUMENT, canonical JSON bytes
	DeploymentId      string                 `protobuf:"bytes,11,opt,name=deployment_id,json=deploymentId,proto3" json:"deployment_id,omitempty"`                // routing (#446)
	unknownFields     protoimpl.UnknownFields
	sizeCache         protoimpl.SizeCache
}

func (x *AttemptTerminal) Reset() {
	*x = AttemptTerminal{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[22]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *AttemptTerminal) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*AttemptTerminal) ProtoMessage() {}

func (x *AttemptTerminal) ProtoReflect() protoreflect.Message {
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

// Deprecated: Use AttemptTerminal.ProtoReflect.Descriptor instead.
func (*AttemptTerminal) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{22}
}

func (x *AttemptTerminal) GetOwnerEpoch() uint64 {
	if x != nil {
		return x.OwnerEpoch
	}
	return 0
}

func (x *AttemptTerminal) GetControlGeneration() uint64 {
	if x != nil {
		return x.ControlGeneration
	}
	return 0
}

func (x *AttemptTerminal) GetWorkerBootId() string {
	if x != nil {
		return x.WorkerBootId
	}
	return ""
}

func (x *AttemptTerminal) GetRequestId() string {
	if x != nil {
		return x.RequestId
	}
	return ""
}

func (x *AttemptTerminal) GetAttempt() uint64 {
	if x != nil {
		return x.Attempt
	}
	return 0
}

func (x *AttemptTerminal) GetInvocationDigest() []byte {
	if x != nil {
		return x.InvocationDigest
	}
	return nil
}

func (x *AttemptTerminal) GetTerminalId() string {
	if x != nil {
		return x.TerminalId
	}
	return ""
}

func (x *AttemptTerminal) GetTerminalDigest() []byte {
	if x != nil {
		return x.TerminalDigest
	}
	return nil
}

func (x *AttemptTerminal) GetTerminalCanonical() []byte {
	if x != nil {
		return x.TerminalCanonical
	}
	return nil
}

func (x *AttemptTerminal) GetDeploymentId() string {
	if x != nil {
		return x.DeploymentId
	}
	return ""
}

// DOCUMENT SHAPE (not a wire message): canonical form `cozy.worker.v1.TerminalBody/1`.
type TerminalBody struct {
	state            protoimpl.MessageState `protogen:"open.v1"`
	RequestId        string                 `protobuf:"bytes,1,opt,name=request_id,json=requestId,proto3" json:"request_id,omitempty"`
	Attempt          uint64                 `protobuf:"varint,2,opt,name=attempt,proto3" json:"attempt,omitempty"`
	InvocationDigest string                 `protobuf:"bytes,3,opt,name=invocation_digest,json=invocationDigest,proto3" json:"invocation_digest,omitempty"` // sha256:<hex> spelling inside the document
	Status           TerminalStatus         `protobuf:"varint,4,opt,name=status,proto3,enum=cozy.worker.v1.TerminalStatus" json:"status,omitempty"`         // closed NEUTRAL vocabulary; retryability is an owner
	// PROJECTION, unrepresentable as a worker wire fact
	OutputManifest *OutputManifest  `protobuf:"bytes,5,opt,name=output_manifest,json=outputManifest,proto3" json:"output_manifest,omitempty"`
	Metrics        *AttemptMetrics  `protobuf:"bytes,6,opt,name=metrics,proto3" json:"metrics,omitempty"`
	TriageBundle   *TriageBundleRef `protobuf:"bytes,7,opt,name=triage_bundle,json=triageBundle,proto3" json:"triage_bundle,omitempty"` // observation only
	SafeMessage    string           `protobuf:"bytes,8,opt,name=safe_message,json=safeMessage,proto3" json:"safe_message,omitempty"`    // bounded <= 4096 bytes, sanitized
	Cause          *TerminalCause   `protobuf:"bytes,9,opt,name=cause,proto3" json:"cause,omitempty"`                                   // REQUIRED
	Result         *ResultEnvelope  `protobuf:"bytes,10,opt,name=result,proto3" json:"result,omitempty"`                                // the typed function result (success terminals)
	unknownFields  protoimpl.UnknownFields
	sizeCache      protoimpl.SizeCache
}

func (x *TerminalBody) Reset() {
	*x = TerminalBody{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[23]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *TerminalBody) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*TerminalBody) ProtoMessage() {}

func (x *TerminalBody) ProtoReflect() protoreflect.Message {
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

// Deprecated: Use TerminalBody.ProtoReflect.Descriptor instead.
func (*TerminalBody) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{23}
}

func (x *TerminalBody) GetRequestId() string {
	if x != nil {
		return x.RequestId
	}
	return ""
}

func (x *TerminalBody) GetAttempt() uint64 {
	if x != nil {
		return x.Attempt
	}
	return 0
}

func (x *TerminalBody) GetInvocationDigest() string {
	if x != nil {
		return x.InvocationDigest
	}
	return ""
}

func (x *TerminalBody) GetStatus() TerminalStatus {
	if x != nil {
		return x.Status
	}
	return TerminalStatus_TERMINAL_STATUS_UNSPECIFIED
}

func (x *TerminalBody) GetOutputManifest() *OutputManifest {
	if x != nil {
		return x.OutputManifest
	}
	return nil
}

func (x *TerminalBody) GetMetrics() *AttemptMetrics {
	if x != nil {
		return x.Metrics
	}
	return nil
}

func (x *TerminalBody) GetTriageBundle() *TriageBundleRef {
	if x != nil {
		return x.TriageBundle
	}
	return nil
}

func (x *TerminalBody) GetSafeMessage() string {
	if x != nil {
		return x.SafeMessage
	}
	return ""
}

func (x *TerminalBody) GetCause() *TerminalCause {
	if x != nil {
		return x.Cause
	}
	return nil
}

func (x *TerminalBody) GetResult() *ResultEnvelope {
	if x != nil {
		return x.Result
	}
	return nil
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
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[24]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *ResultEnvelope) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*ResultEnvelope) ProtoMessage() {}

func (x *ResultEnvelope) ProtoReflect() protoreflect.Message {
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

// Deprecated: Use ResultEnvelope.ProtoReflect.Descriptor instead.
func (*ResultEnvelope) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{24}
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
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[25]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *AdjustmentRow) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*AdjustmentRow) ProtoMessage() {}

func (x *AdjustmentRow) ProtoReflect() protoreflect.Message {
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

// Deprecated: Use AdjustmentRow.ProtoReflect.Descriptor instead.
func (*AdjustmentRow) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{25}
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

type TerminalCause struct {
	state         protoimpl.MessageState `protogen:"open.v1"`
	Code          CauseCode              `protobuf:"varint,1,opt,name=code,proto3,enum=cozy.worker.v1.CauseCode" json:"code,omitempty"`
	Origin        CauseOrigin            `protobuf:"varint,2,opt,name=origin,proto3,enum=cozy.worker.v1.CauseOrigin" json:"origin,omitempty"`
	Detail        string                 `protobuf:"bytes,3,opt,name=detail,proto3" json:"detail,omitempty"` // bounded <= 1024 bytes, sanitized
	Shortfall     *ResourceShortfall     `protobuf:"bytes,4,opt,name=shortfall,proto3" json:"shortfall,omitempty"`
	unknownFields protoimpl.UnknownFields
	sizeCache     protoimpl.SizeCache
}

func (x *TerminalCause) Reset() {
	*x = TerminalCause{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[26]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *TerminalCause) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*TerminalCause) ProtoMessage() {}

func (x *TerminalCause) ProtoReflect() protoreflect.Message {
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

// Deprecated: Use TerminalCause.ProtoReflect.Descriptor instead.
func (*TerminalCause) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{26}
}

func (x *TerminalCause) GetCode() CauseCode {
	if x != nil {
		return x.Code
	}
	return CauseCode_CAUSE_CODE_UNSPECIFIED
}

func (x *TerminalCause) GetOrigin() CauseOrigin {
	if x != nil {
		return x.Origin
	}
	return CauseOrigin_CAUSE_ORIGIN_UNSPECIFIED
}

func (x *TerminalCause) GetDetail() string {
	if x != nil {
		return x.Detail
	}
	return ""
}

func (x *TerminalCause) GetShortfall() *ResourceShortfall {
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
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[27]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *ResourceShortfall) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*ResourceShortfall) ProtoMessage() {}

func (x *ResourceShortfall) ProtoReflect() protoreflect.Message {
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

// Deprecated: Use ResourceShortfall.ProtoReflect.Descriptor instead.
func (*ResourceShortfall) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{27}
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

type TerminalAck struct {
	state             protoimpl.MessageState `protogen:"open.v1"`
	OwnerEpoch        uint64                 `protobuf:"varint,1,opt,name=owner_epoch,json=ownerEpoch,proto3" json:"owner_epoch,omitempty"`
	ControlGeneration uint64                 `protobuf:"varint,2,opt,name=control_generation,json=controlGeneration,proto3" json:"control_generation,omitempty"`
	WorkerBootId      string                 `protobuf:"bytes,3,opt,name=worker_boot_id,json=workerBootId,proto3" json:"worker_boot_id,omitempty"`
	RequestId         string                 `protobuf:"bytes,5,opt,name=request_id,json=requestId,proto3" json:"request_id,omitempty"`
	Attempt           uint64                 `protobuf:"varint,6,opt,name=attempt,proto3" json:"attempt,omitempty"`
	InvocationDigest  []byte                 `protobuf:"bytes,7,opt,name=invocation_digest,json=invocationDigest,proto3" json:"invocation_digest,omitempty"`
	TerminalId        string                 `protobuf:"bytes,8,opt,name=terminal_id,json=terminalId,proto3" json:"terminal_id,omitempty"`             // echo; a mismatched ack is NOT an ack — replay continues
	TerminalDigest    []byte                 `protobuf:"bytes,9,opt,name=terminal_digest,json=terminalDigest,proto3" json:"terminal_digest,omitempty"` // echo; compared, never recomputed
	unknownFields     protoimpl.UnknownFields
	sizeCache         protoimpl.SizeCache
}

func (x *TerminalAck) Reset() {
	*x = TerminalAck{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[28]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *TerminalAck) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*TerminalAck) ProtoMessage() {}

func (x *TerminalAck) ProtoReflect() protoreflect.Message {
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

// Deprecated: Use TerminalAck.ProtoReflect.Descriptor instead.
func (*TerminalAck) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{28}
}

func (x *TerminalAck) GetOwnerEpoch() uint64 {
	if x != nil {
		return x.OwnerEpoch
	}
	return 0
}

func (x *TerminalAck) GetControlGeneration() uint64 {
	if x != nil {
		return x.ControlGeneration
	}
	return 0
}

func (x *TerminalAck) GetWorkerBootId() string {
	if x != nil {
		return x.WorkerBootId
	}
	return ""
}

func (x *TerminalAck) GetRequestId() string {
	if x != nil {
		return x.RequestId
	}
	return ""
}

func (x *TerminalAck) GetAttempt() uint64 {
	if x != nil {
		return x.Attempt
	}
	return 0
}

func (x *TerminalAck) GetInvocationDigest() []byte {
	if x != nil {
		return x.InvocationDigest
	}
	return nil
}

func (x *TerminalAck) GetTerminalId() string {
	if x != nil {
		return x.TerminalId
	}
	return ""
}

func (x *TerminalAck) GetTerminalDigest() []byte {
	if x != nil {
		return x.TerminalDigest
	}
	return nil
}

type JobCheckpointRequest struct {
	state             protoimpl.MessageState `protogen:"open.v1"`
	OwnerEpoch        uint64                 `protobuf:"varint,1,opt,name=owner_epoch,json=ownerEpoch,proto3" json:"owner_epoch,omitempty"`
	ControlGeneration uint64                 `protobuf:"varint,2,opt,name=control_generation,json=controlGeneration,proto3" json:"control_generation,omitempty"`
	WorkerBootId      string                 `protobuf:"bytes,3,opt,name=worker_boot_id,json=workerBootId,proto3" json:"worker_boot_id,omitempty"`
	RequestId         string                 `protobuf:"bytes,5,opt,name=request_id,json=requestId,proto3" json:"request_id,omitempty"`
	Attempt           uint64                 `protobuf:"varint,6,opt,name=attempt,proto3" json:"attempt,omitempty"`
	OperationKey      string                 `protobuf:"bytes,7,opt,name=operation_key,json=operationKey,proto3" json:"operation_key,omitempty"`    // worker-minted, attempt-scoped idempotency key
	LogicalKey        string                 `protobuf:"bytes,8,opt,name=logical_key,json=logicalKey,proto3" json:"logical_key,omitempty"`          // author-stable checkpoint slot
	ContentDigest     []byte                 `protobuf:"bytes,9,opt,name=content_digest,json=contentDigest,proto3" json:"content_digest,omitempty"` // class (b): the checkpoint artifact's bytes
	Seq               uint64                 `protobuf:"varint,10,opt,name=seq,proto3" json:"seq,omitempty"`                                        // ordering only, never identity
	Artifact          *OutputEntry           `protobuf:"bytes,11,opt,name=artifact,proto3" json:"artifact,omitempty"`
	unknownFields     protoimpl.UnknownFields
	sizeCache         protoimpl.SizeCache
}

func (x *JobCheckpointRequest) Reset() {
	*x = JobCheckpointRequest{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[29]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *JobCheckpointRequest) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*JobCheckpointRequest) ProtoMessage() {}

func (x *JobCheckpointRequest) ProtoReflect() protoreflect.Message {
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

// Deprecated: Use JobCheckpointRequest.ProtoReflect.Descriptor instead.
func (*JobCheckpointRequest) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{29}
}

func (x *JobCheckpointRequest) GetOwnerEpoch() uint64 {
	if x != nil {
		return x.OwnerEpoch
	}
	return 0
}

func (x *JobCheckpointRequest) GetControlGeneration() uint64 {
	if x != nil {
		return x.ControlGeneration
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

func (x *JobCheckpointRequest) GetAttempt() uint64 {
	if x != nil {
		return x.Attempt
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
	state             protoimpl.MessageState `protogen:"open.v1"`
	OwnerEpoch        uint64                 `protobuf:"varint,1,opt,name=owner_epoch,json=ownerEpoch,proto3" json:"owner_epoch,omitempty"`
	ControlGeneration uint64                 `protobuf:"varint,2,opt,name=control_generation,json=controlGeneration,proto3" json:"control_generation,omitempty"`
	WorkerBootId      string                 `protobuf:"bytes,3,opt,name=worker_boot_id,json=workerBootId,proto3" json:"worker_boot_id,omitempty"`
	RequestId         string                 `protobuf:"bytes,5,opt,name=request_id,json=requestId,proto3" json:"request_id,omitempty"`
	Attempt           uint64                 `protobuf:"varint,6,opt,name=attempt,proto3" json:"attempt,omitempty"`
	OperationKey      string                 `protobuf:"bytes,7,opt,name=operation_key,json=operationKey,proto3" json:"operation_key,omitempty"`    // echo
	LogicalKey        string                 `protobuf:"bytes,8,opt,name=logical_key,json=logicalKey,proto3" json:"logical_key,omitempty"`          // echo
	ContentDigest     []byte                 `protobuf:"bytes,9,opt,name=content_digest,json=contentDigest,proto3" json:"content_digest,omitempty"` // echo; mismatch vs recorded identity is a conflict
	ReceiptId         string                 `protobuf:"bytes,10,opt,name=receipt_id,json=receiptId,proto3" json:"receipt_id,omitempty"`            // owner-minted durable record id
	Outcome           CheckpointOutcome      `protobuf:"varint,11,opt,name=outcome,proto3,enum=cozy.worker.v1.CheckpointOutcome" json:"outcome,omitempty"`
	Fault             *CheckpointFault       `protobuf:"bytes,12,opt,name=fault,proto3" json:"fault,omitempty"` // set iff CONFLICT or REFUSED
	unknownFields     protoimpl.UnknownFields
	sizeCache         protoimpl.SizeCache
}

func (x *JobCheckpointReceipt) Reset() {
	*x = JobCheckpointReceipt{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[30]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *JobCheckpointReceipt) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*JobCheckpointReceipt) ProtoMessage() {}

func (x *JobCheckpointReceipt) ProtoReflect() protoreflect.Message {
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

// Deprecated: Use JobCheckpointReceipt.ProtoReflect.Descriptor instead.
func (*JobCheckpointReceipt) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{30}
}

func (x *JobCheckpointReceipt) GetOwnerEpoch() uint64 {
	if x != nil {
		return x.OwnerEpoch
	}
	return 0
}

func (x *JobCheckpointReceipt) GetControlGeneration() uint64 {
	if x != nil {
		return x.ControlGeneration
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

func (x *JobCheckpointReceipt) GetAttempt() uint64 {
	if x != nil {
		return x.Attempt
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
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[31]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *CheckpointFault) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*CheckpointFault) ProtoMessage() {}

func (x *CheckpointFault) ProtoReflect() protoreflect.Message {
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

// Deprecated: Use CheckpointFault.ProtoReflect.Descriptor instead.
func (*CheckpointFault) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{31}
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
	state             protoimpl.MessageState `protogen:"open.v1"`
	OwnerEpoch        uint64                 `protobuf:"varint,1,opt,name=owner_epoch,json=ownerEpoch,proto3" json:"owner_epoch,omitempty"`
	ControlGeneration uint64                 `protobuf:"varint,2,opt,name=control_generation,json=controlGeneration,proto3" json:"control_generation,omitempty"`
	WorkerBootId      string                 `protobuf:"bytes,3,opt,name=worker_boot_id,json=workerBootId,proto3" json:"worker_boot_id,omitempty"`
	RequestId         string                 `protobuf:"bytes,5,opt,name=request_id,json=requestId,proto3" json:"request_id,omitempty"`
	Attempt           uint64                 `protobuf:"varint,6,opt,name=attempt,proto3" json:"attempt,omitempty"`
	OperationKey      string                 `protobuf:"bytes,7,opt,name=operation_key,json=operationKey,proto3" json:"operation_key,omitempty"`
	ReceiptId         string                 `protobuf:"bytes,8,opt,name=receipt_id,json=receiptId,proto3" json:"receipt_id,omitempty"` // echo; a mismatched ack is not an ack
	ContentDigest     []byte                 `protobuf:"bytes,9,opt,name=content_digest,json=contentDigest,proto3" json:"content_digest,omitempty"`
	unknownFields     protoimpl.UnknownFields
	sizeCache         protoimpl.SizeCache
}

func (x *JobCheckpointAck) Reset() {
	*x = JobCheckpointAck{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[32]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *JobCheckpointAck) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*JobCheckpointAck) ProtoMessage() {}

func (x *JobCheckpointAck) ProtoReflect() protoreflect.Message {
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

// Deprecated: Use JobCheckpointAck.ProtoReflect.Descriptor instead.
func (*JobCheckpointAck) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{32}
}

func (x *JobCheckpointAck) GetOwnerEpoch() uint64 {
	if x != nil {
		return x.OwnerEpoch
	}
	return 0
}

func (x *JobCheckpointAck) GetControlGeneration() uint64 {
	if x != nil {
		return x.ControlGeneration
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

func (x *JobCheckpointAck) GetAttempt() uint64 {
	if x != nil {
		return x.Attempt
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
	state             protoimpl.MessageState `protogen:"open.v1"`
	OwnerEpoch        uint64                 `protobuf:"varint,1,opt,name=owner_epoch,json=ownerEpoch,proto3" json:"owner_epoch,omitempty"`
	ControlGeneration uint64                 `protobuf:"varint,2,opt,name=control_generation,json=controlGeneration,proto3" json:"control_generation,omitempty"` // binds the watch to the fenced control stream
	WorkerBootId      string                 `protobuf:"bytes,3,opt,name=worker_boot_id,json=workerBootId,proto3" json:"worker_boot_id,omitempty"`
	RequestId         string                 `protobuf:"bytes,5,opt,name=request_id,json=requestId,proto3" json:"request_id,omitempty"` // empty = all attempts on this stream
	unknownFields     protoimpl.UnknownFields
	sizeCache         protoimpl.SizeCache
}

func (x *ProgressOpen) Reset() {
	*x = ProgressOpen{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[33]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *ProgressOpen) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*ProgressOpen) ProtoMessage() {}

func (x *ProgressOpen) ProtoReflect() protoreflect.Message {
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

// Deprecated: Use ProgressOpen.ProtoReflect.Descriptor instead.
func (*ProgressOpen) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{33}
}

func (x *ProgressOpen) GetOwnerEpoch() uint64 {
	if x != nil {
		return x.OwnerEpoch
	}
	return 0
}

func (x *ProgressOpen) GetControlGeneration() uint64 {
	if x != nil {
		return x.ControlGeneration
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
	state             protoimpl.MessageState `protogen:"open.v1"`
	OwnerEpoch        uint64                 `protobuf:"varint,1,opt,name=owner_epoch,json=ownerEpoch,proto3" json:"owner_epoch,omitempty"`
	ControlGeneration uint64                 `protobuf:"varint,2,opt,name=control_generation,json=controlGeneration,proto3" json:"control_generation,omitempty"`
	WorkerBootId      string                 `protobuf:"bytes,3,opt,name=worker_boot_id,json=workerBootId,proto3" json:"worker_boot_id,omitempty"`
	RequestId         string                 `protobuf:"bytes,5,opt,name=request_id,json=requestId,proto3" json:"request_id,omitempty"`
	Attempt           uint64                 `protobuf:"varint,6,opt,name=attempt,proto3" json:"attempt,omitempty"`
	Seq               uint64                 `protobuf:"varint,7,opt,name=seq,proto3" json:"seq,omitempty"` // starts at 1, strictly increasing per (request_id,
	// attempt); gaps are VISIBLE and the only loss allowed
	ContentType   string `protobuf:"bytes,8,opt,name=content_type,json=contentType,proto3" json:"content_type,omitempty"`
	Data          []byte `protobuf:"bytes,9,opt,name=data,proto3" json:"data,omitempty"`                                      // bounded <= 65536 bytes; NEVER terminal
	DeploymentId  string `protobuf:"bytes,10,opt,name=deployment_id,json=deploymentId,proto3" json:"deployment_id,omitempty"` // routing (#446)
	unknownFields protoimpl.UnknownFields
	sizeCache     protoimpl.SizeCache
}

func (x *AttemptProgress) Reset() {
	*x = AttemptProgress{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[34]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *AttemptProgress) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*AttemptProgress) ProtoMessage() {}

func (x *AttemptProgress) ProtoReflect() protoreflect.Message {
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

// Deprecated: Use AttemptProgress.ProtoReflect.Descriptor instead.
func (*AttemptProgress) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{34}
}

func (x *AttemptProgress) GetOwnerEpoch() uint64 {
	if x != nil {
		return x.OwnerEpoch
	}
	return 0
}

func (x *AttemptProgress) GetControlGeneration() uint64 {
	if x != nil {
		return x.ControlGeneration
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

func (x *AttemptProgress) GetAttempt() uint64 {
	if x != nil {
		return x.Attempt
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

func (x *AttemptProgress) GetDeploymentId() string {
	if x != nil {
		return x.DeploymentId
	}
	return ""
}

// DOCUMENT SHAPE (not a wire message): the subject of `invocation_digest`. Canonical form
// `cozy.worker.v1.InvocationSpec/1`. The key set is CLOSED (law 4): no human model/adapter
// ref is spellable; unknown keys refuse on read and are unwritable on author.
// deployment_id is DELIBERATELY absent (#446): the same invocation is the same work wherever
// it routes; the deployment is wire routing, never meaning.
type InvocationSpec struct {
	state             protoimpl.MessageState `protogen:"open.v1"`
	EndpointReleaseId string                 `protobuf:"bytes,1,opt,name=endpoint_release_id,json=endpointReleaseId,proto3" json:"endpoint_release_id,omitempty"`
	ImageDigest       string                 `protobuf:"bytes,2,opt,name=image_digest,json=imageDigest,proto3" json:"image_digest,omitempty"`             // class (b), sha256:<hex>: exact execution environment
	ConfigDigest      string                 `protobuf:"bytes,3,opt,name=config_digest,json=configDigest,proto3" json:"config_digest,omitempty"`          // class (a): the evaluated-config document (cr-003)
	PayloadDigest     string                 `protobuf:"bytes,4,opt,name=payload_digest,json=payloadDigest,proto3" json:"payload_digest,omitempty"`       // class (a): the canonical typed-argument document
	Inputs            []*InputBinding        `protobuf:"bytes,5,rep,name=inputs,proto3" json:"inputs,omitempty"`                                          // ORDERED input identities — INSIDE the digest
	Outputs           []*OutputBinding       `protobuf:"bytes,6,rep,name=outputs,proto3" json:"outputs,omitempty"`                                        // output ids/kinds/limits — INSIDE the digest
	DeadlineUnixMs    uint64                 `protobuf:"varint,7,opt,name=deadline_unix_ms,json=deadlineUnixMs,proto3" json:"deadline_unix_ms,omitempty"` // absolute attempt deadline; supervisor-enforced; 0 = none
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
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[35]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *InvocationSpec) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*InvocationSpec) ProtoMessage() {}

func (x *InvocationSpec) ProtoReflect() protoreflect.Message {
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

// Deprecated: Use InvocationSpec.ProtoReflect.Descriptor instead.
func (*InvocationSpec) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{35}
}

func (x *InvocationSpec) GetEndpointReleaseId() string {
	if x != nil {
		return x.EndpointReleaseId
	}
	return ""
}

func (x *InvocationSpec) GetImageDigest() string {
	if x != nil {
		return x.ImageDigest
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
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[36]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *InputBinding) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*InputBinding) ProtoMessage() {}

func (x *InputBinding) ProtoReflect() protoreflect.Message {
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

// Deprecated: Use InputBinding.ProtoReflect.Descriptor instead.
func (*InputBinding) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{36}
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
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[37]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *OutputBinding) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*OutputBinding) ProtoMessage() {}

func (x *OutputBinding) ProtoReflect() protoreflect.Message {
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

// Deprecated: Use OutputBinding.ProtoReflect.Descriptor instead.
func (*OutputBinding) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{37}
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
	EntrypointBindingPlanId string                 `protobuf:"bytes,1,opt,name=entrypoint_binding_plan_id,json=entrypointBindingPlanId,proto3" json:"entrypoint_binding_plan_id,omitempty"` // class (a) id, th-004's document
	AttemptBindingId        string                 `protobuf:"bytes,2,opt,name=attempt_binding_id,json=attemptBindingId,proto3" json:"attempt_binding_id,omitempty"`                        // equal to field 1 when no adapters
	unknownFields           protoimpl.UnknownFields
	sizeCache               protoimpl.SizeCache
}

func (x *ServingInvocationSpec) Reset() {
	*x = ServingInvocationSpec{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[38]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *ServingInvocationSpec) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*ServingInvocationSpec) ProtoMessage() {}

func (x *ServingInvocationSpec) ProtoReflect() protoreflect.Message {
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

// Deprecated: Use ServingInvocationSpec.ProtoReflect.Descriptor instead.
func (*ServingInvocationSpec) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{38}
}

func (x *ServingInvocationSpec) GetEntrypointBindingPlanId() string {
	if x != nil {
		return x.EntrypointBindingPlanId
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
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[39]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *JobInvocationSpec) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*JobInvocationSpec) ProtoMessage() {}

func (x *JobInvocationSpec) ProtoReflect() protoreflect.Message {
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

// Deprecated: Use JobInvocationSpec.ProtoReflect.Descriptor instead.
func (*JobInvocationSpec) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{39}
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

// Refreshable ACCESS: urls, token, expiry — and the invocation it serves. Nothing else.
type DeliveryGrant struct {
	state            protoimpl.MessageState `protogen:"open.v1"`
	InvocationDigest []byte                 `protobuf:"bytes,1,opt,name=invocation_digest,json=invocationDigest,proto3" json:"invocation_digest,omitempty"` // the subject this grant serves; revalidated on refresh
	Credential       *Credential            `protobuf:"bytes,2,opt,name=credential,proto3" json:"credential,omitempty"`                                     // per-attempt scoped capability token
	FileBaseUrl      string                 `protobuf:"bytes,3,opt,name=file_base_url,json=fileBaseUrl,proto3" json:"file_base_url,omitempty"`
	ExpiresAtUnix    uint64                 `protobuf:"varint,4,opt,name=expires_at_unix,json=expiresAtUnix,proto3" json:"expires_at_unix,omitempty"`
	Inputs           []*InputAccess         `protobuf:"bytes,5,rep,name=inputs,proto3" json:"inputs,omitempty"`   // sorted by input_id; ids must match the spec's set
	Outputs          []*OutputAccess        `protobuf:"bytes,6,rep,name=outputs,proto3" json:"outputs,omitempty"` // sorted by output_id
	unknownFields    protoimpl.UnknownFields
	sizeCache        protoimpl.SizeCache
}

func (x *DeliveryGrant) Reset() {
	*x = DeliveryGrant{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[40]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *DeliveryGrant) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*DeliveryGrant) ProtoMessage() {}

func (x *DeliveryGrant) ProtoReflect() protoreflect.Message {
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

// Deprecated: Use DeliveryGrant.ProtoReflect.Descriptor instead.
func (*DeliveryGrant) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{40}
}

func (x *DeliveryGrant) GetInvocationDigest() []byte {
	if x != nil {
		return x.InvocationDigest
	}
	return nil
}

func (x *DeliveryGrant) GetCredential() *Credential {
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
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[41]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *InputAccess) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*InputAccess) ProtoMessage() {}

func (x *InputAccess) ProtoReflect() protoreflect.Message {
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

// Deprecated: Use InputAccess.ProtoReflect.Descriptor instead.
func (*InputAccess) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{41}
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
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[42]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *OutputAccess) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*OutputAccess) ProtoMessage() {}

func (x *OutputAccess) ProtoReflect() protoreflect.Message {
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

// Deprecated: Use OutputAccess.ProtoReflect.Descriptor instead.
func (*OutputAccess) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{42}
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

type Credential struct {
	state         protoimpl.MessageState `protogen:"open.v1"`
	Issuer        string                 `protobuf:"bytes,1,opt,name=issuer,proto3" json:"issuer,omitempty"`
	KeyId         string                 `protobuf:"bytes,2,opt,name=key_id,json=keyId,proto3" json:"key_id,omitempty"`
	Epoch         uint64                 `protobuf:"varint,3,opt,name=epoch,proto3" json:"epoch,omitempty"`
	ExpiresAtUnix uint64                 `protobuf:"varint,4,opt,name=expires_at_unix,json=expiresAtUnix,proto3" json:"expires_at_unix,omitempty"`
	Token         []byte                 `protobuf:"bytes,5,opt,name=token,proto3" json:"token,omitempty"` // STRUCTURALLY SECRET; never in a digest/log/Report
	unknownFields protoimpl.UnknownFields
	sizeCache     protoimpl.SizeCache
}

func (x *Credential) Reset() {
	*x = Credential{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[43]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *Credential) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*Credential) ProtoMessage() {}

func (x *Credential) ProtoReflect() protoreflect.Message {
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

// Deprecated: Use Credential.ProtoReflect.Descriptor instead.
func (*Credential) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{43}
}

func (x *Credential) GetIssuer() string {
	if x != nil {
		return x.Issuer
	}
	return ""
}

func (x *Credential) GetKeyId() string {
	if x != nil {
		return x.KeyId
	}
	return ""
}

func (x *Credential) GetEpoch() uint64 {
	if x != nil {
		return x.Epoch
	}
	return 0
}

func (x *Credential) GetExpiresAtUnix() uint64 {
	if x != nil {
		return x.ExpiresAtUnix
	}
	return 0
}

func (x *Credential) GetToken() []byte {
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
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[44]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *ResourceCaps) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*ResourceCaps) ProtoMessage() {}

func (x *ResourceCaps) ProtoReflect() protoreflect.Message {
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

// Deprecated: Use ResourceCaps.ProtoReflect.Descriptor instead.
func (*ResourceCaps) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{44}
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
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[45]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *PublicationContract) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*PublicationContract) ProtoMessage() {}

func (x *PublicationContract) ProtoReflect() protoreflect.Message {
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

// Deprecated: Use PublicationContract.ProtoReflect.Descriptor instead.
func (*PublicationContract) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{45}
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

// Torch-free supervisor-measurable STATICS (NVML / sysctl / OS facts), sent once at ClaimAck
// and never re-sent. Accelerator-neutral (#446): backend + memory_model use the
// AcceleratorFacts vocabulary (#422/#423). The supervisor never fabricates capability facts —
// a discovered accelerator is PRESENT-BUT-UNQUALIFIED until a deployment executor qualifies
// it (DeploymentStatus.accelerator). torch_version is gone from here: a torch-free process
// cannot honestly report one.
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
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[46]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *WorkerResources) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*WorkerResources) ProtoMessage() {}

func (x *WorkerResources) ProtoReflect() protoreflect.Message {
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

// Deprecated: Use WorkerResources.ProtoReflect.Descriptor instead.
func (*WorkerResources) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{46}
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
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[47]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *JobCapacity) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*JobCapacity) ProtoMessage() {}

func (x *JobCapacity) ProtoReflect() protoreflect.Message {
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

// Deprecated: Use JobCapacity.ProtoReflect.Descriptor instead.
func (*JobCapacity) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{47}
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

type ActiveAttempt struct {
	state              protoimpl.MessageState `protogen:"open.v1"`
	RequestId          string                 `protobuf:"bytes,1,opt,name=request_id,json=requestId,proto3" json:"request_id,omitempty"`
	Attempt            uint64                 `protobuf:"varint,2,opt,name=attempt,proto3" json:"attempt,omitempty"`
	Kind               AttemptKind            `protobuf:"varint,3,opt,name=kind,proto3,enum=cozy.worker.v1.AttemptKind" json:"kind,omitempty"`
	State              AttemptState           `protobuf:"varint,4,opt,name=state,proto3,enum=cozy.worker.v1.AttemptState" json:"state,omitempty"`
	InvocationDigest   []byte                 `protobuf:"bytes,5,opt,name=invocation_digest,json=invocationDigest,proto3" json:"invocation_digest,omitempty"`
	DeploymentId       string                 `protobuf:"bytes,6,opt,name=deployment_id,json=deploymentId,proto3" json:"deployment_id,omitempty"`                    // the executing deployment (#446); empty in job mode
	ExecutorGeneration uint64                 `protobuf:"varint,7,opt,name=executor_generation,json=executorGeneration,proto3" json:"executor_generation,omitempty"` // the generation it runs/ran under (#446)
	unknownFields      protoimpl.UnknownFields
	sizeCache          protoimpl.SizeCache
}

func (x *ActiveAttempt) Reset() {
	*x = ActiveAttempt{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[48]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *ActiveAttempt) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*ActiveAttempt) ProtoMessage() {}

func (x *ActiveAttempt) ProtoReflect() protoreflect.Message {
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

// Deprecated: Use ActiveAttempt.ProtoReflect.Descriptor instead.
func (*ActiveAttempt) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{48}
}

func (x *ActiveAttempt) GetRequestId() string {
	if x != nil {
		return x.RequestId
	}
	return ""
}

func (x *ActiveAttempt) GetAttempt() uint64 {
	if x != nil {
		return x.Attempt
	}
	return 0
}

func (x *ActiveAttempt) GetKind() AttemptKind {
	if x != nil {
		return x.Kind
	}
	return AttemptKind_ATTEMPT_KIND_UNSPECIFIED
}

func (x *ActiveAttempt) GetState() AttemptState {
	if x != nil {
		return x.State
	}
	return AttemptState_ATTEMPT_STATE_UNSPECIFIED
}

func (x *ActiveAttempt) GetInvocationDigest() []byte {
	if x != nil {
		return x.InvocationDigest
	}
	return nil
}

func (x *ActiveAttempt) GetDeploymentId() string {
	if x != nil {
		return x.DeploymentId
	}
	return ""
}

func (x *ActiveAttempt) GetExecutorGeneration() uint64 {
	if x != nil {
		return x.ExecutorGeneration
	}
	return 0
}

type Fault struct {
	state         protoimpl.MessageState `protogen:"open.v1"`
	Kind          FaultKind              `protobuf:"varint,1,opt,name=kind,proto3,enum=cozy.worker.v1.FaultKind" json:"kind,omitempty"`
	Subject       string                 `protobuf:"bytes,2,opt,name=subject,proto3" json:"subject,omitempty"`
	Reason        string                 `protobuf:"bytes,3,opt,name=reason,proto3" json:"reason,omitempty"` // bounded <= 1024 bytes
	Detail        string                 `protobuf:"bytes,4,opt,name=detail,proto3" json:"detail,omitempty"` // bounded <= 1024 bytes
	unknownFields protoimpl.UnknownFields
	sizeCache     protoimpl.SizeCache
}

func (x *Fault) Reset() {
	*x = Fault{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[49]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *Fault) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*Fault) ProtoMessage() {}

func (x *Fault) ProtoReflect() protoreflect.Message {
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

// Deprecated: Use Fault.ProtoReflect.Descriptor instead.
func (*Fault) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{49}
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
	state         protoimpl.MessageState `protogen:"open.v1"`
	ManifestId    string                 `protobuf:"bytes,1,opt,name=manifest_id,json=manifestId,proto3" json:"manifest_id,omitempty"` // the PublicationReceipt's canonical digest (sha256:<hex>)
	Outputs       []*OutputEntry         `protobuf:"bytes,3,rep,name=outputs,proto3" json:"outputs,omitempty"`                         // sorted by output_id
	unknownFields protoimpl.UnknownFields
	sizeCache     protoimpl.SizeCache
}

func (x *OutputManifest) Reset() {
	*x = OutputManifest{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[50]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *OutputManifest) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*OutputManifest) ProtoMessage() {}

func (x *OutputManifest) ProtoReflect() protoreflect.Message {
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

// Deprecated: Use OutputManifest.ProtoReflect.Descriptor instead.
func (*OutputManifest) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{50}
}

func (x *OutputManifest) GetManifestId() string {
	if x != nil {
		return x.ManifestId
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
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[51]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *OutputEntry) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*OutputEntry) ProtoMessage() {}

func (x *OutputEntry) ProtoReflect() protoreflect.Message {
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

// Deprecated: Use OutputEntry.ProtoReflect.Descriptor instead.
func (*OutputEntry) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{51}
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
	DeviceLeaseMs         uint64                 `protobuf:"varint,8,opt,name=device_lease_ms,json=deviceLeaseMs,proto3" json:"device_lease_ms,omitempty"` // supervisor-attested
	DeviceCount           uint32                 `protobuf:"varint,9,opt,name=device_count,json=deviceCount,proto3" json:"device_count,omitempty"`
	HandlerMs             uint64                 `protobuf:"varint,10,opt,name=handler_ms,json=handlerMs,proto3" json:"handler_ms,omitempty"`
	FinalizationMs        uint64                 `protobuf:"varint,11,opt,name=finalization_ms,json=finalizationMs,proto3" json:"finalization_ms,omitempty"`
	UnverifiedFields      []string               `protobuf:"bytes,12,rep,name=unverified_fields,json=unverifiedFields,proto3" json:"unverified_fields,omitempty"`
	unknownFields         protoimpl.UnknownFields
	sizeCache             protoimpl.SizeCache
}

func (x *AttemptMetrics) Reset() {
	*x = AttemptMetrics{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[52]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *AttemptMetrics) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*AttemptMetrics) ProtoMessage() {}

func (x *AttemptMetrics) ProtoReflect() protoreflect.Message {
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

// Deprecated: Use AttemptMetrics.ProtoReflect.Descriptor instead.
func (*AttemptMetrics) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{52}
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
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[53]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *TriageBundleRef) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*TriageBundleRef) ProtoMessage() {}

func (x *TriageBundleRef) ProtoReflect() protoreflect.Message {
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

// Deprecated: Use TriageBundleRef.ProtoReflect.Descriptor instead.
func (*TriageBundleRef) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{53}
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
	"\x1bcozy/worker/v1/worker.proto\x12\x0ecozy.worker.v1\"\xeb\x03\n" +
	"\n" +
	"OwnerFrame\x12-\n" +
	"\x05claim\x18\x05 \x01(\v2\x15.cozy.worker.v1.ClaimH\x00R\x05claim\x129\n" +
	"\tdirective\x18\x06 \x01(\v2\x19.cozy.worker.v1.DirectiveH\x00R\tdirective\x12C\n" +
	"\rstart_attempt\x18\a \x01(\v2\x1c.cozy.worker.v1.StartAttemptH\x00R\fstartAttempt\x12F\n" +
	"\x0ecancel_attempt\x18\b \x01(\v2\x1d.cozy.worker.v1.CancelAttemptH\x00R\rcancelAttempt\x12@\n" +
	"\fterminal_ack\x18\t \x01(\v2\x1b.cozy.worker.v1.TerminalAckH\x00R\vterminalAck\x12U\n" +
	"\x12checkpoint_receipt\x18\n" +
	" \x01(\v2$.cozy.worker.v1.JobCheckpointReceiptH\x00R\x11checkpointReceipt\x12@\n" +
	"\fsnapshot_ack\x18\v \x01(\v2\x1b.cozy.worker.v1.SnapshotAckH\x00R\vsnapshotAckB\x05\n" +
	"\x03msgJ\x04\b\x0e\x10\x0f\"\xd7\x05\n" +
	"\vWorkerFrame\x127\n" +
	"\tclaim_ack\x18\x05 \x01(\v2\x18.cozy.worker.v1.ClaimAckH\x00R\bclaimAck\x120\n" +
	"\x06report\x18\x06 \x01(\v2\x16.cozy.worker.v1.ReportH\x00R\x06report\x12L\n" +
	"\x10attempt_accepted\x18\a \x01(\v2\x1f.cozy.worker.v1.AttemptAcceptedH\x00R\x0fattemptAccepted\x12L\n" +
	"\x10attempt_terminal\x18\b \x01(\v2\x1f.cozy.worker.v1.AttemptTerminalH\x00R\x0fattemptTerminal\x12@\n" +
	"\fboot_failure\x18\t \x01(\v2\x1b.cozy.worker.v1.BootFailureH\x00R\vbootFailure\x12U\n" +
	"\x12checkpoint_request\x18\n" +
	" \x01(\v2$.cozy.worker.v1.JobCheckpointRequestH\x00R\x11checkpointRequest\x12I\n" +
	"\x0echeckpoint_ack\x18\v \x01(\v2 .cozy.worker.v1.JobCheckpointAckH\x00R\rcheckpointAck\x12F\n" +
	"\x0esnapshot_begin\x18\f \x01(\v2\x1d.cozy.worker.v1.SnapshotBeginH\x00R\rsnapshotBegin\x12@\n" +
	"\fsnapshot_end\x18\r \x01(\v2\x1b.cozy.worker.v1.SnapshotEndH\x00R\vsnapshotEnd\x12F\n" +
	"\x0esnapshot_entry\x18\x0f \x01(\v2\x1d.cozy.worker.v1.SnapshotEntryH\x00R\rsnapshotEntryB\x05\n" +
	"\x03msgJ\x04\b\x0e\x10\x0f\"\xfa\x01\n" +
	"\x05Claim\x12\x1f\n" +
	"\vowner_epoch\x18\x01 \x01(\x04R\n" +
	"ownerEpoch\x12-\n" +
	"\x12control_generation\x18\x02 \x01(\x04R\x11controlGeneration\x12$\n" +
	"\x0eworker_boot_id\x18\x03 \x01(\tR\fworkerBootId\x12#\n" +
	"\rcontroller_id\x18\x05 \x01(\tR\fcontrollerId\x12\x1b\n" +
	"\tworker_id\x18\x06 \x01(\tR\bworkerId\x12\x1d\n" +
	"\n" +
	"wire_minor\x18\a \x01(\rR\twireMinor\x12\x14\n" +
	"\x05proof\x18\b \x01(\fR\x05proofJ\x04\b\x04\x10\x05\"\xdd\x03\n" +
	"\bClaimAck\x12\x1f\n" +
	"\vowner_epoch\x18\x01 \x01(\x04R\n" +
	"ownerEpoch\x12-\n" +
	"\x12control_generation\x18\x02 \x01(\x04R\x11controlGeneration\x12$\n" +
	"\x0eworker_boot_id\x18\x03 \x01(\tR\fworkerBootId\x12\x1a\n" +
	"\baccepted\x18\x05 \x01(\bR\baccepted\x12<\n" +
	"\trejection\x18\x06 \x01(\x0e2\x1e.cozy.worker.v1.ClaimRejectionR\trejection\x12\x1d\n" +
	"\n" +
	"wire_minor\x18\a \x01(\rR\twireMinor\x12\x1b\n" +
	"\tworker_id\x18\b \x01(\tR\bworkerId\x12\x1f\n" +
	"\vinstance_id\x18\t \x01(\tR\n" +
	"instanceId\x12\x1d\n" +
	"\n" +
	"release_id\x18\n" +
	" \x01(\tR\treleaseId\x12!\n" +
	"\fimage_digest\x18\v \x01(\tR\vimageDigest\x12\x1d\n" +
	"\n" +
	"git_commit\x18\f \x01(\tR\tgitCommit\x12=\n" +
	"\tresources\x18\r \x01(\v2\x1f.cozy.worker.v1.WorkerResourcesR\tresourcesJ\x04\b\x04\x10\x05\"\x9b\x03\n" +
	"\vBootFailure\x12\x1f\n" +
	"\vowner_epoch\x18\x01 \x01(\x04R\n" +
	"ownerEpoch\x12-\n" +
	"\x12control_generation\x18\x02 \x01(\x04R\x11controlGeneration\x12$\n" +
	"\x0eworker_boot_id\x18\x03 \x01(\tR\fworkerBootId\x12\x1b\n" +
	"\tworker_id\x18\x05 \x01(\tR\bworkerId\x12\x1f\n" +
	"\vinstance_id\x18\x06 \x01(\tR\n" +
	"instanceId\x129\n" +
	"\x06reason\x18\a \x01(\x0e2!.cozy.worker.v1.BootFailureReasonR\x06reason\x12\x16\n" +
	"\x06detail\x18\b \x01(\tR\x06detail\x12=\n" +
	"\tresources\x18\t \x01(\v2\x1f.cozy.worker.v1.WorkerResourcesR\tresources\x12!\n" +
	"\fimage_digest\x18\n" +
	" \x01(\tR\vimageDigest\x12\x1d\n" +
	"\n" +
	"release_id\x18\v \x01(\tR\treleaseIdJ\x04\b\x04\x10\x05\"\xd9\x01\n" +
	"\rSnapshotBegin\x12\x1f\n" +
	"\vowner_epoch\x18\x01 \x01(\x04R\n" +
	"ownerEpoch\x12-\n" +
	"\x12control_generation\x18\x02 \x01(\x04R\x11controlGeneration\x12$\n" +
	"\x0eworker_boot_id\x18\x03 \x01(\tR\fworkerBootId\x12\x1f\n" +
	"\vsnapshot_id\x18\x05 \x01(\tR\n" +
	"snapshotId\x12+\n" +
	"\x11journal_highwater\x18\x06 \x01(\x04R\x10journalHighwaterJ\x04\b\x04\x10\x05\"\xe5\x01\n" +
	"\rSnapshotEntry\x12\x1f\n" +
	"\vowner_epoch\x18\x01 \x01(\x04R\n" +
	"ownerEpoch\x12-\n" +
	"\x12control_generation\x18\x02 \x01(\x04R\x11controlGeneration\x12$\n" +
	"\x0eworker_boot_id\x18\x03 \x01(\tR\fworkerBootId\x12\x1f\n" +
	"\vsnapshot_id\x18\x05 \x01(\tR\n" +
	"snapshotId\x127\n" +
	"\aattempt\x18\x06 \x01(\v2\x1d.cozy.worker.v1.ActiveAttemptR\aattemptJ\x04\b\x04\x10\x05\"\xcb\x01\n" +
	"\vSnapshotEnd\x12\x1f\n" +
	"\vowner_epoch\x18\x01 \x01(\x04R\n" +
	"ownerEpoch\x12-\n" +
	"\x12control_generation\x18\x02 \x01(\x04R\x11controlGeneration\x12$\n" +
	"\x0eworker_boot_id\x18\x03 \x01(\tR\fworkerBootId\x12\x1f\n" +
	"\vsnapshot_id\x18\x05 \x01(\tR\n" +
	"snapshotId\x12\x1f\n" +
	"\ventry_count\x18\x06 \x01(\rR\n" +
	"entryCountJ\x04\b\x04\x10\x05\"\xaa\x01\n" +
	"\vSnapshotAck\x12\x1f\n" +
	"\vowner_epoch\x18\x01 \x01(\x04R\n" +
	"ownerEpoch\x12-\n" +
	"\x12control_generation\x18\x02 \x01(\x04R\x11controlGeneration\x12$\n" +
	"\x0eworker_boot_id\x18\x03 \x01(\tR\fworkerBootId\x12\x1f\n" +
	"\vsnapshot_id\x18\x05 \x01(\tR\n" +
	"snapshotIdJ\x04\b\x04\x10\x05\"\xbe\x03\n" +
	"\tDirective\x12\x1f\n" +
	"\vowner_epoch\x18\x01 \x01(\x04R\n" +
	"ownerEpoch\x12-\n" +
	"\x12control_generation\x18\x02 \x01(\x04R\x11controlGeneration\x12$\n" +
	"\x0eworker_boot_id\x18\x03 \x01(\tR\fworkerBootId\x12\x1a\n" +
	"\brevision\x18\x05 \x01(\x04R\brevision\x121\n" +
	"\aposture\x18\x06 \x01(\x0e2\x17.cozy.worker.v1.PostureR\aposture\x12\x1d\n" +
	"\n" +
	"wire_minor\x18\a \x01(\rR\twireMinor\x12$\n" +
	"\x0edrain_grace_ms\x18\b \x01(\x04R\fdrainGraceMs\x120\n" +
	"\x03job\x18\r \x01(\v2\x1c.cozy.worker.v1.JobDirectiveH\x00R\x03job\x12O\n" +
	"\x0edeployment_set\x18\x0f \x01(\v2&.cozy.worker.v1.DeploymentSetDirectiveH\x00R\rdeploymentSetB\x06\n" +
	"\x04modeJ\x04\b\t\x10\n" +
	"J\x04\b\n" +
	"\x10\vJ\x04\b\v\x10\fJ\x04\b\f\x10\rJ\x04\b\x04\x10\x05\"h\n" +
	"\x16DeploymentSetDirective\x12\x1d\n" +
	"\n" +
	"set_digest\x18\x01 \x01(\fR\tsetDigest\x12/\n" +
	"\x03set\x18\x02 \x01(\v2\x1d.cozy.worker.v1.DeploymentSetR\x03set\"M\n" +
	"\rDeploymentSet\x12<\n" +
	"\vdeployments\x18\x01 \x03(\v2\x1a.cozy.worker.v1.DeploymentR\vdeployments\"\xa6\x01\n" +
	"\n" +
	"Deployment\x12#\n" +
	"\rdeployment_id\x18\x01 \x01(\tR\fdeploymentId\x12.\n" +
	"\x13endpoint_release_id\x18\x02 \x01(\tR\x11endpointReleaseId\x12=\n" +
	"\x1bentrypoint_binding_plan_ids\x18\x03 \x03(\tR\x18entrypointBindingPlanIdsJ\x04\b\x04\x10\x05\"\xc3\x02\n" +
	"\fJobDirective\x12\x19\n" +
	"\bbuild_id\x18\x01 \x01(\tR\abuildId\x12*\n" +
	"\x11job_descriptor_id\x18\x02 \x01(\tR\x0fjobDescriptorId\x12A\n" +
	"\rresource_caps\x18\x03 \x01(\v2\x1c.cozy.worker.v1.ResourceCapsR\fresourceCaps\x12V\n" +
	"\x14publication_contract\x18\x04 \x01(\v2#.cozy.worker.v1.PublicationContractR\x13publicationContract\x12.\n" +
	"\x13reclaim_on_terminal\x18\x05 \x01(\bR\x11reclaimOnTerminal\x12!\n" +
	"\fdevice_count\x18\x06 \x01(\rR\vdeviceCount\"\x8d\x05\n" +
	"\x06Report\x12\x1f\n" +
	"\vowner_epoch\x18\x01 \x01(\x04R\n" +
	"ownerEpoch\x12-\n" +
	"\x12control_generation\x18\x02 \x01(\x04R\x11controlGeneration\x12$\n" +
	"\x0eworker_boot_id\x18\x03 \x01(\tR\fworkerBootId\x12)\n" +
	"\x10applied_revision\x18\x05 \x01(\x04R\x0fappliedRevision\x12>\n" +
	"\fintake_state\x18\x06 \x01(\x0e2\x1b.cozy.worker.v1.IntakeStateR\vintakeState\x12,\n" +
	"\x12applied_wire_minor\x18\b \x01(\rR\x10appliedWireMinor\x12F\n" +
	"\x0factive_attempts\x18\t \x03(\v2\x1d.cozy.worker.v1.ActiveAttemptR\x0eactiveAttempts\x12-\n" +
	"\x06faults\x18\n" +
	" \x03(\v2\x15.cozy.worker.v1.FaultR\x06faults\x129\n" +
	"\bactivity\x18\v \x03(\v2\x1d.cozy.worker.v1.ActivityEventR\bactivity\x12>\n" +
	"\fjob_capacity\x18\r \x01(\v2\x1b.cozy.worker.v1.JobCapacityR\vjobCapacity\x12,\n" +
	"\x12applied_set_digest\x18\x0e \x01(\fR\x10appliedSetDigest\x12B\n" +
	"\vdeployments\x18\x0f \x03(\v2 .cozy.worker.v1.DeploymentStatusR\vdeploymentsJ\x04\b\x04\x10\x05J\x04\b\a\x10\bJ\x04\b\f\x10\r\"\x8f\x04\n" +
	"\x10DeploymentStatus\x12#\n" +
	"\rdeployment_id\x18\x01 \x01(\tR\fdeploymentId\x12>\n" +
	"\fintake_state\x18\x02 \x01(\x0e2\x1b.cozy.worker.v1.IntakeStateR\vintakeState\x12'\n" +
	"\x0freadiness_epoch\x18\x03 \x01(\x04R\x0ereadinessEpoch\x12/\n" +
	"\x13executor_generation\x18\x04 \x01(\x04R\x12executorGeneration\x12'\n" +
	"\x0fattempt_credits\x18\x05 \x01(\rR\x0eattemptCredits\x12H\n" +
	"!ready_entrypoint_binding_plan_ids\x18\x06 \x03(\tR\x1dreadyEntrypointBindingPlanIds\x12N\n" +
	"$loadable_entrypoint_binding_plan_ids\x18\a \x03(\tR loadableEntrypointBindingPlanIds\x12-\n" +
	"\x06faults\x18\b \x03(\v2\x15.cozy.worker.v1.FaultR\x06faults\x12J\n" +
	"\vaccelerator\x18\t \x01(\v2(.cozy.worker.v1.AcceleratorQualificationR\vaccelerator\"\xe1\x02\n" +
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
	"at_unix_ms\x18\x04 \x01(\x04R\batUnixMs\"\xfd\x02\n" +
	"\fStartAttempt\x12\x1f\n" +
	"\vowner_epoch\x18\x01 \x01(\x04R\n" +
	"ownerEpoch\x12-\n" +
	"\x12control_generation\x18\x02 \x01(\x04R\x11controlGeneration\x12$\n" +
	"\x0eworker_boot_id\x18\x03 \x01(\tR\fworkerBootId\x12\x1d\n" +
	"\n" +
	"request_id\x18\x05 \x01(\tR\trequestId\x12\x18\n" +
	"\aattempt\x18\x06 \x01(\x04R\aattempt\x12+\n" +
	"\x11invocation_digest\x18\a \x01(\fR\x10invocationDigest\x123\n" +
	"\x05grant\x18\b \x01(\v2\x1d.cozy.worker.v1.DeliveryGrantR\x05grant\x121\n" +
	"\x14invocation_canonical\x18\t \x01(\fR\x13invocationCanonical\x12#\n" +
	"\rdeployment_id\x18\n" +
	" \x01(\tR\fdeploymentIdJ\x04\b\x04\x10\x05\"\xde\x03\n" +
	"\x0fAttemptAccepted\x12\x1f\n" +
	"\vowner_epoch\x18\x01 \x01(\x04R\n" +
	"ownerEpoch\x12-\n" +
	"\x12control_generation\x18\x02 \x01(\x04R\x11controlGeneration\x12$\n" +
	"\x0eworker_boot_id\x18\x03 \x01(\tR\fworkerBootId\x12\x1d\n" +
	"\n" +
	"request_id\x18\x05 \x01(\tR\trequestId\x12\x18\n" +
	"\aattempt\x18\x06 \x01(\x04R\aattempt\x12+\n" +
	"\x11invocation_digest\x18\a \x01(\fR\x10invocationDigest\x12\x1f\n" +
	"\vplan_digest\x18\b \x01(\fR\n" +
	"planDigest\x12:\n" +
	"\x19model_construction_digest\x18\t \x01(\fR\x17modelConstructionDigest\x126\n" +
	"\x04plan\x18\n" +
	" \x01(\v2\".cozy.worker.v1.AttemptPlanSummaryR\x04plan\x12#\n" +
	"\rdeployment_id\x18\v \x01(\tR\fdeploymentId\x12/\n" +
	"\x13executor_generation\x18\f \x01(\x04R\x12executorGenerationJ\x04\b\x04\x10\x05\"\xc1\x02\n" +
	"\x12AttemptPlanSummary\x12\x1a\n" +
	"\bdelivery\x18\x01 \x01(\tR\bdelivery\x12(\n" +
	"\x0fmaterialization\x18\x02 \x01(\tR\x0fmaterialization\x12#\n" +
	"\rcompute_dtype\x18\x03 \x01(\tR\fcomputeDtype\x12\x1c\n" +
	"\tplacement\x18\x04 \x01(\tR\tplacement\x12?\n" +
	"\x1creserved_device_memory_bytes\x18\x05 \x01(\x04R\x19reservedDeviceMemoryBytes\x12.\n" +
	"\x13reserved_host_bytes\x18\x06 \x01(\x04R\x11reservedHostBytes\x121\n" +
	"\x14decision_explanation\x18\a \x01(\tR\x13decisionExplanation\"\xc2\x02\n" +
	"\rCancelAttempt\x12\x1f\n" +
	"\vowner_epoch\x18\x01 \x01(\x04R\n" +
	"ownerEpoch\x12-\n" +
	"\x12control_generation\x18\x02 \x01(\x04R\x11controlGeneration\x12$\n" +
	"\x0eworker_boot_id\x18\x03 \x01(\tR\fworkerBootId\x12\x1d\n" +
	"\n" +
	"request_id\x18\x05 \x01(\tR\trequestId\x12\x18\n" +
	"\aattempt\x18\x06 \x01(\x04R\aattempt\x124\n" +
	"\x06reason\x18\a \x01(\x0e2\x1c.cozy.worker.v1.CancelReasonR\x06reason\x12\x19\n" +
	"\bgrace_ms\x18\b \x01(\x04R\agraceMs\x12+\n" +
	"\x11invocation_digest\x18\t \x01(\fR\x10invocationDigestJ\x04\b\x04\x10\x05\"\x91\x03\n" +
	"\x0fAttemptTerminal\x12\x1f\n" +
	"\vowner_epoch\x18\x01 \x01(\x04R\n" +
	"ownerEpoch\x12-\n" +
	"\x12control_generation\x18\x02 \x01(\x04R\x11controlGeneration\x12$\n" +
	"\x0eworker_boot_id\x18\x03 \x01(\tR\fworkerBootId\x12\x1d\n" +
	"\n" +
	"request_id\x18\x05 \x01(\tR\trequestId\x12\x18\n" +
	"\aattempt\x18\x06 \x01(\x04R\aattempt\x12+\n" +
	"\x11invocation_digest\x18\a \x01(\fR\x10invocationDigest\x12\x1f\n" +
	"\vterminal_id\x18\b \x01(\tR\n" +
	"terminalId\x12'\n" +
	"\x0fterminal_digest\x18\t \x01(\fR\x0eterminalDigest\x12-\n" +
	"\x12terminal_canonical\x18\n" +
	" \x01(\fR\x11terminalCanonical\x12#\n" +
	"\rdeployment_id\x18\v \x01(\tR\fdeploymentIdJ\x04\b\x04\x10\x05\"\x85\x04\n" +
	"\fTerminalBody\x12\x1d\n" +
	"\n" +
	"request_id\x18\x01 \x01(\tR\trequestId\x12\x18\n" +
	"\aattempt\x18\x02 \x01(\x04R\aattempt\x12+\n" +
	"\x11invocation_digest\x18\x03 \x01(\tR\x10invocationDigest\x126\n" +
	"\x06status\x18\x04 \x01(\x0e2\x1e.cozy.worker.v1.TerminalStatusR\x06status\x12G\n" +
	"\x0foutput_manifest\x18\x05 \x01(\v2\x1e.cozy.worker.v1.OutputManifestR\x0eoutputManifest\x128\n" +
	"\ametrics\x18\x06 \x01(\v2\x1e.cozy.worker.v1.AttemptMetricsR\ametrics\x12D\n" +
	"\rtriage_bundle\x18\a \x01(\v2\x1f.cozy.worker.v1.TriageBundleRefR\ftriageBundle\x12!\n" +
	"\fsafe_message\x18\b \x01(\tR\vsafeMessage\x123\n" +
	"\x05cause\x18\t \x01(\v2\x1d.cozy.worker.v1.TerminalCauseR\x05cause\x126\n" +
	"\x06result\x18\n" +
	" \x01(\v2\x1e.cozy.worker.v1.ResultEnvelopeR\x06result\"\x93\x02\n" +
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
	"\x06reason\x18\x04 \x01(\tR\x06reason\"\xcc\x01\n" +
	"\rTerminalCause\x12-\n" +
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
	"\x0eevidence_class\x18\x06 \x01(\tR\revidenceClass\"\xb9\x02\n" +
	"\vTerminalAck\x12\x1f\n" +
	"\vowner_epoch\x18\x01 \x01(\x04R\n" +
	"ownerEpoch\x12-\n" +
	"\x12control_generation\x18\x02 \x01(\x04R\x11controlGeneration\x12$\n" +
	"\x0eworker_boot_id\x18\x03 \x01(\tR\fworkerBootId\x12\x1d\n" +
	"\n" +
	"request_id\x18\x05 \x01(\tR\trequestId\x12\x18\n" +
	"\aattempt\x18\x06 \x01(\x04R\aattempt\x12+\n" +
	"\x11invocation_digest\x18\a \x01(\fR\x10invocationDigest\x12\x1f\n" +
	"\vterminal_id\x18\b \x01(\tR\n" +
	"terminalId\x12'\n" +
	"\x0fterminal_digest\x18\t \x01(\fR\x0eterminalDigestJ\x04\b\x04\x10\x05\"\x83\x03\n" +
	"\x14JobCheckpointRequest\x12\x1f\n" +
	"\vowner_epoch\x18\x01 \x01(\x04R\n" +
	"ownerEpoch\x12-\n" +
	"\x12control_generation\x18\x02 \x01(\x04R\x11controlGeneration\x12$\n" +
	"\x0eworker_boot_id\x18\x03 \x01(\tR\fworkerBootId\x12\x1d\n" +
	"\n" +
	"request_id\x18\x05 \x01(\tR\trequestId\x12\x18\n" +
	"\aattempt\x18\x06 \x01(\x04R\aattempt\x12#\n" +
	"\roperation_key\x18\a \x01(\tR\foperationKey\x12\x1f\n" +
	"\vlogical_key\x18\b \x01(\tR\n" +
	"logicalKey\x12%\n" +
	"\x0econtent_digest\x18\t \x01(\fR\rcontentDigest\x12\x10\n" +
	"\x03seq\x18\n" +
	" \x01(\x04R\x03seq\x127\n" +
	"\bartifact\x18\v \x01(\v2\x1b.cozy.worker.v1.OutputEntryR\bartifactJ\x04\b\x04\x10\x05\"\xcb\x03\n" +
	"\x14JobCheckpointReceipt\x12\x1f\n" +
	"\vowner_epoch\x18\x01 \x01(\x04R\n" +
	"ownerEpoch\x12-\n" +
	"\x12control_generation\x18\x02 \x01(\x04R\x11controlGeneration\x12$\n" +
	"\x0eworker_boot_id\x18\x03 \x01(\tR\fworkerBootId\x12\x1d\n" +
	"\n" +
	"request_id\x18\x05 \x01(\tR\trequestId\x12\x18\n" +
	"\aattempt\x18\x06 \x01(\x04R\aattempt\x12#\n" +
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
	"\x17recorded_content_digest\x18\x03 \x01(\fR\x15recordedContentDigest\"\xb2\x02\n" +
	"\x10JobCheckpointAck\x12\x1f\n" +
	"\vowner_epoch\x18\x01 \x01(\x04R\n" +
	"ownerEpoch\x12-\n" +
	"\x12control_generation\x18\x02 \x01(\x04R\x11controlGeneration\x12$\n" +
	"\x0eworker_boot_id\x18\x03 \x01(\tR\fworkerBootId\x12\x1d\n" +
	"\n" +
	"request_id\x18\x05 \x01(\tR\trequestId\x12\x18\n" +
	"\aattempt\x18\x06 \x01(\x04R\aattempt\x12#\n" +
	"\roperation_key\x18\a \x01(\tR\foperationKey\x12\x1d\n" +
	"\n" +
	"receipt_id\x18\b \x01(\tR\treceiptId\x12%\n" +
	"\x0econtent_digest\x18\t \x01(\fR\rcontentDigestJ\x04\b\x04\x10\x05\"\xa9\x01\n" +
	"\fProgressOpen\x12\x1f\n" +
	"\vowner_epoch\x18\x01 \x01(\x04R\n" +
	"ownerEpoch\x12-\n" +
	"\x12control_generation\x18\x02 \x01(\x04R\x11controlGeneration\x12$\n" +
	"\x0eworker_boot_id\x18\x03 \x01(\tR\fworkerBootId\x12\x1d\n" +
	"\n" +
	"request_id\x18\x05 \x01(\tR\trequestIdJ\x04\b\x04\x10\x05\"\xb4\x02\n" +
	"\x0fAttemptProgress\x12\x1f\n" +
	"\vowner_epoch\x18\x01 \x01(\x04R\n" +
	"ownerEpoch\x12-\n" +
	"\x12control_generation\x18\x02 \x01(\x04R\x11controlGeneration\x12$\n" +
	"\x0eworker_boot_id\x18\x03 \x01(\tR\fworkerBootId\x12\x1d\n" +
	"\n" +
	"request_id\x18\x05 \x01(\tR\trequestId\x12\x18\n" +
	"\aattempt\x18\x06 \x01(\x04R\aattempt\x12\x10\n" +
	"\x03seq\x18\a \x01(\x04R\x03seq\x12!\n" +
	"\fcontent_type\x18\b \x01(\tR\vcontentType\x12\x12\n" +
	"\x04data\x18\t \x01(\fR\x04data\x12#\n" +
	"\rdeployment_id\x18\n" +
	" \x01(\tR\fdeploymentIdJ\x04\b\x04\x10\x05\"\xca\x03\n" +
	"\x0eInvocationSpec\x12.\n" +
	"\x13endpoint_release_id\x18\x01 \x01(\tR\x11endpointReleaseId\x12!\n" +
	"\fimage_digest\x18\x02 \x01(\tR\vimageDigest\x12#\n" +
	"\rconfig_digest\x18\x03 \x01(\tR\fconfigDigest\x12%\n" +
	"\x0epayload_digest\x18\x04 \x01(\tR\rpayloadDigest\x124\n" +
	"\x06inputs\x18\x05 \x03(\v2\x1c.cozy.worker.v1.InputBindingR\x06inputs\x127\n" +
	"\aoutputs\x18\x06 \x03(\v2\x1d.cozy.worker.v1.OutputBindingR\aoutputs\x12(\n" +
	"\x10deadline_unix_ms\x18\a \x01(\x04R\x0edeadlineUnixMs\x12A\n" +
	"\aserving\x18\b \x01(\v2%.cozy.worker.v1.ServingInvocationSpecH\x00R\aserving\x125\n" +
	"\x03job\x18\t \x01(\v2!.cozy.worker.v1.JobInvocationSpecH\x00R\x03jobB\x06\n" +
	"\x04spec\"\x8c\x01\n" +
	"\fInputBinding\x12\x19\n" +
	"\binput_id\x18\x01 \x01(\tR\ainputId\x12\x16\n" +
	"\x06digest\x18\x02 \x01(\tR\x06digest\x12\x16\n" +
	"\x06length\x18\x03 \x01(\x04R\x06length\x12\x1b\n" +
	"\tkind_mime\x18\x04 \x01(\tR\bkindMime\x12\x14\n" +
	"\x05order\x18\x05 \x01(\rR\x05order\"f\n" +
	"\rOutputBinding\x12\x1b\n" +
	"\toutput_id\x18\x01 \x01(\tR\boutputId\x12\x1b\n" +
	"\tmime_type\x18\x02 \x01(\tR\bmimeType\x12\x1b\n" +
	"\tmax_bytes\x18\x03 \x01(\x04R\bmaxBytes\"\x82\x01\n" +
	"\x15ServingInvocationSpec\x12;\n" +
	"\x1aentrypoint_binding_plan_id\x18\x01 \x01(\tR\x17entrypointBindingPlanId\x12,\n" +
	"\x12attempt_binding_id\x18\x02 \x01(\tR\x10attemptBindingId\"\xb2\x01\n" +
	"\x11JobInvocationSpec\x12\x19\n" +
	"\bbuild_id\x18\x01 \x01(\tR\abuildId\x12*\n" +
	"\x11job_descriptor_id\x18\x02 \x01(\tR\x0fjobDescriptorId\x12V\n" +
	"\x14publication_contract\x18\x03 \x01(\v2#.cozy.worker.v1.PublicationContractR\x13publicationContract\"\xb1\x02\n" +
	"\rDeliveryGrant\x12+\n" +
	"\x11invocation_digest\x18\x01 \x01(\fR\x10invocationDigest\x12:\n" +
	"\n" +
	"credential\x18\x02 \x01(\v2\x1a.cozy.worker.v1.CredentialR\n" +
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
	"\x03url\x18\x02 \x01(\tR\x03url\"\x8f\x01\n" +
	"\n" +
	"Credential\x12\x16\n" +
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
	"\x0ffree_disk_bytes\x18\x04 \x01(\x04R\rfreeDiskBytesJ\x04\b\x03\x10\x04\"\xb0\x02\n" +
	"\rActiveAttempt\x12\x1d\n" +
	"\n" +
	"request_id\x18\x01 \x01(\tR\trequestId\x12\x18\n" +
	"\aattempt\x18\x02 \x01(\x04R\aattempt\x12/\n" +
	"\x04kind\x18\x03 \x01(\x0e2\x1b.cozy.worker.v1.AttemptKindR\x04kind\x122\n" +
	"\x05state\x18\x04 \x01(\x0e2\x1c.cozy.worker.v1.AttemptStateR\x05state\x12+\n" +
	"\x11invocation_digest\x18\x05 \x01(\fR\x10invocationDigest\x12#\n" +
	"\rdeployment_id\x18\x06 \x01(\tR\fdeploymentId\x12/\n" +
	"\x13executor_generation\x18\a \x01(\x04R\x12executorGeneration\"\x80\x01\n" +
	"\x05Fault\x12-\n" +
	"\x04kind\x18\x01 \x01(\x0e2\x19.cozy.worker.v1.FaultKindR\x04kind\x12\x18\n" +
	"\asubject\x18\x02 \x01(\tR\asubject\x12\x16\n" +
	"\x06reason\x18\x03 \x01(\tR\x06reason\x12\x16\n" +
	"\x06detail\x18\x04 \x01(\tR\x06detail\"h\n" +
	"\x0eOutputManifest\x12\x1f\n" +
	"\vmanifest_id\x18\x01 \x01(\tR\n" +
	"manifestId\x125\n" +
	"\aoutputs\x18\x03 \x03(\v2\x1b.cozy.worker.v1.OutputEntryR\aoutputs\"w\n" +
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
	"\x11unverified_fields\x18\f \x03(\tR\x10unverifiedFields\"z\n" +
	"\x0fTriageBundleRef\x12\x1d\n" +
	"\n" +
	"subject_id\x18\x01 \x01(\tR\tsubjectId\x120\n" +
	"\x14write_receipt_digest\x18\x04 \x01(\fR\x12writeReceiptDigest\x12\x16\n" +
	"\x06length\x18\x03 \x01(\x04R\x06length*O\n" +
	"\aPosture\x12\x17\n" +
	"\x13POSTURE_UNSPECIFIED\x10\x00\x12\x15\n" +
	"\x11POSTURE_ACCEPTING\x10\x01\x12\x14\n" +
	"\x10POSTURE_DRAINING\x10\x02*\xc8\x01\n" +
	"\vIntakeState\x12\x1c\n" +
	"\x18INTAKE_STATE_UNSPECIFIED\x10\x00\x12\x18\n" +
	"\x14INTAKE_STATE_BOOTING\x10\x01\x12\x1c\n" +
	"\x18INTAKE_STATE_DOWNLOADING\x10\x02\x12\x18\n" +
	"\x14INTAKE_STATE_LOADING\x10\x03\x12\x16\n" +
	"\x12INTAKE_STATE_READY\x10\x04\x12\x19\n" +
	"\x15INTAKE_STATE_DRAINING\x10\x05\x12\x16\n" +
	"\x12INTAKE_STATE_ERROR\x10\x06*[\n" +
	"\vAttemptKind\x12\x1c\n" +
	"\x18ATTEMPT_KIND_UNSPECIFIED\x10\x00\x12\x18\n" +
	"\x14ATTEMPT_KIND_SERVING\x10\x01\x12\x14\n" +
	"\x10ATTEMPT_KIND_JOB\x10\x02*p\n" +
	"\fAttemptState\x12\x1d\n" +
	"\x19ATTEMPT_STATE_UNSPECIFIED\x10\x00\x12\x19\n" +
	"\x15ATTEMPT_STATE_RUNNING\x10\x01\x12&\n" +
	"\"ATTEMPT_STATE_TERMINAL_PENDING_ACK\x10\x02*\xc6\x01\n" +
	"\x0eTerminalStatus\x12\x1f\n" +
	"\x1bTERMINAL_STATUS_UNSPECIFIED\x10\x00\x12\x1d\n" +
	"\x19TERMINAL_STATUS_SUCCEEDED\x10\x01\x12\x1b\n" +
	"\x17TERMINAL_STATUS_REFUSED\x10\x02\x12\x1a\n" +
	"\x16TERMINAL_STATUS_FAILED\x10\x03\x12\x1c\n" +
	"\x18TERMINAL_STATUS_CANCELED\x10\x04\x12\x1d\n" +
	"\x19TERMINAL_STATUS_ABANDONED\x10\x05*\xb0\x04\n" +
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
	"\x1fCAUSE_CODE_EXECUTOR_INVALIDATED\x10\x10*\xe5\x01\n" +
	"\vCauseOrigin\x12\x1c\n" +
	"\x18CAUSE_ORIGIN_UNSPECIFIED\x10\x00\x12\x17\n" +
	"\x13CAUSE_ORIGIN_AUTHOR\x10\x01\x12\x18\n" +
	"\x14CAUSE_ORIGIN_RUNTIME\x10\x02\x12\x19\n" +
	"\x15CAUSE_ORIGIN_EXECUTOR\x10\x03\x12\x1b\n" +
	"\x17CAUSE_ORIGIN_SUPERVISOR\x10\x04\x12\x16\n" +
	"\x12CAUSE_ORIGIN_INFRA\x10\x05\x12\x17\n" +
	"\x13CAUSE_ORIGIN_CLIENT\x10\x06\x12\x1c\n" +
	"\x18CAUSE_ORIGIN_COORDINATOR\x10\a*\xb4\x01\n" +
	"\fCancelReason\x12\x1d\n" +
	"\x19CANCEL_REASON_UNSPECIFIED\x10\x00\x12\x18\n" +
	"\x14CANCEL_REASON_CLIENT\x10\x01\x12\x17\n" +
	"\x13CANCEL_REASON_DRAIN\x10\x02\x12\x1c\n" +
	"\x18CANCEL_REASON_SUPERSEDED\x10\x03\x12\x18\n" +
	"\x14CANCEL_REASON_POLICY\x10\x04\x12\x1a\n" +
	"\x16CANCEL_REASON_DEADLINE\x10\x05*\x95\x02\n" +
	"\x0eClaimRejection\x12\x1f\n" +
	"\x1bCLAIM_REJECTION_UNSPECIFIED\x10\x00\x12#\n" +
	"\x1fCLAIM_REJECTION_UNAUTHENTICATED\x10\x01\x12%\n" +
	"!CLAIM_REJECTION_STALE_OWNER_EPOCH\x10\x02\x12\x1e\n" +
	"\x1aCLAIM_REJECTION_EPOCH_HELD\x10\x03\x12&\n" +
	"\"CLAIM_REJECTION_WORKER_ID_MISMATCH\x10\x04\x12'\n" +
	"#CLAIM_REJECTION_RELEASE_ID_MISMATCH\x10\x05\x12%\n" +
	"!CLAIM_REJECTION_UNSUPPORTED_MINOR\x10\x06*\xad\x03\n" +
	"\tFaultKind\x12\x1a\n" +
	"\x16FAULT_KIND_UNSPECIFIED\x10\x00\x12\"\n" +
	"\x1eFAULT_KIND_BINDING_UNAVAILABLE\x10\x01\x12\x1f\n" +
	"\x1bFAULT_KIND_BINDING_DEGRADED\x10\x02\x12\"\n" +
	"\x1eFAULT_KIND_HARDWARE_UNSUITABLE\x10\x03\x12$\n" +
	" FAULT_KIND_ARTIFACT_FETCH_FAILED\x10\x04\x12\x1c\n" +
	"\x18FAULT_KIND_GRANT_EXPIRED\x10\x05\x12#\n" +
	"\x1fFAULT_KIND_CREDENTIAL_UNAPPLIED\x10\x06\x12#\n" +
	"\x1fFAULT_KIND_LOCAL_SAFETY_REFUSAL\x10\a\x12 \n" +
	"\x1cFAULT_KIND_EXECUTOR_POISONED\x10\b\x12\x1d\n" +
	"\x19FAULT_KIND_CONFIG_REFUSED\x10\t\x12!\n" +
	"\x1dFAULT_KIND_UNKNOWN_DEPLOYMENT\x10\n" +
	"\x12)\n" +
	"%FAULT_KIND_DEPLOYMENT_SET_UNSUPPORTED\x10\v*\x9e\x02\n" +
	"\x11BootFailureReason\x12#\n" +
	"\x1fBOOT_FAILURE_REASON_UNSPECIFIED\x10\x00\x12(\n" +
	"$BOOT_FAILURE_REASON_HARDWARE_VERDICT\x10\x01\x12#\n" +
	"\x1fBOOT_FAILURE_REASON_IMAGE_FAULT\x10\x02\x12$\n" +
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
	"$CHECKPOINT_FAULT_CODE_QUOTA_EXCEEDED\x10\x052\xa9\x01\n" +
	"\rWorkerControl\x12F\n" +
	"\aControl\x12\x1a.cozy.worker.v1.OwnerFrame\x1a\x1b.cozy.worker.v1.WorkerFrame(\x010\x01\x12P\n" +
	"\rWatchProgress\x12\x1c.cozy.worker.v1.ProgressOpen\x1a\x1f.cozy.worker.v1.AttemptProgress0\x01B4\n" +
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

var file_cozy_worker_v1_worker_proto_enumTypes = make([]protoimpl.EnumInfo, 13)
var file_cozy_worker_v1_worker_proto_msgTypes = make([]protoimpl.MessageInfo, 54)
var file_cozy_worker_v1_worker_proto_goTypes = []any{
	(Posture)(0),                     // 0: cozy.worker.v1.Posture
	(IntakeState)(0),                 // 1: cozy.worker.v1.IntakeState
	(AttemptKind)(0),                 // 2: cozy.worker.v1.AttemptKind
	(AttemptState)(0),                // 3: cozy.worker.v1.AttemptState
	(TerminalStatus)(0),              // 4: cozy.worker.v1.TerminalStatus
	(CauseCode)(0),                   // 5: cozy.worker.v1.CauseCode
	(CauseOrigin)(0),                 // 6: cozy.worker.v1.CauseOrigin
	(CancelReason)(0),                // 7: cozy.worker.v1.CancelReason
	(ClaimRejection)(0),              // 8: cozy.worker.v1.ClaimRejection
	(FaultKind)(0),                   // 9: cozy.worker.v1.FaultKind
	(BootFailureReason)(0),           // 10: cozy.worker.v1.BootFailureReason
	(CheckpointOutcome)(0),           // 11: cozy.worker.v1.CheckpointOutcome
	(CheckpointFaultCode)(0),         // 12: cozy.worker.v1.CheckpointFaultCode
	(*OwnerFrame)(nil),               // 13: cozy.worker.v1.OwnerFrame
	(*WorkerFrame)(nil),              // 14: cozy.worker.v1.WorkerFrame
	(*Claim)(nil),                    // 15: cozy.worker.v1.Claim
	(*ClaimAck)(nil),                 // 16: cozy.worker.v1.ClaimAck
	(*BootFailure)(nil),              // 17: cozy.worker.v1.BootFailure
	(*SnapshotBegin)(nil),            // 18: cozy.worker.v1.SnapshotBegin
	(*SnapshotEntry)(nil),            // 19: cozy.worker.v1.SnapshotEntry
	(*SnapshotEnd)(nil),              // 20: cozy.worker.v1.SnapshotEnd
	(*SnapshotAck)(nil),              // 21: cozy.worker.v1.SnapshotAck
	(*Directive)(nil),                // 22: cozy.worker.v1.Directive
	(*DeploymentSetDirective)(nil),   // 23: cozy.worker.v1.DeploymentSetDirective
	(*DeploymentSet)(nil),            // 24: cozy.worker.v1.DeploymentSet
	(*Deployment)(nil),               // 25: cozy.worker.v1.Deployment
	(*JobDirective)(nil),             // 26: cozy.worker.v1.JobDirective
	(*Report)(nil),                   // 27: cozy.worker.v1.Report
	(*DeploymentStatus)(nil),         // 28: cozy.worker.v1.DeploymentStatus
	(*AcceleratorQualification)(nil), // 29: cozy.worker.v1.AcceleratorQualification
	(*ActivityEvent)(nil),            // 30: cozy.worker.v1.ActivityEvent
	(*StartAttempt)(nil),             // 31: cozy.worker.v1.StartAttempt
	(*AttemptAccepted)(nil),          // 32: cozy.worker.v1.AttemptAccepted
	(*AttemptPlanSummary)(nil),       // 33: cozy.worker.v1.AttemptPlanSummary
	(*CancelAttempt)(nil),            // 34: cozy.worker.v1.CancelAttempt
	(*AttemptTerminal)(nil),          // 35: cozy.worker.v1.AttemptTerminal
	(*TerminalBody)(nil),             // 36: cozy.worker.v1.TerminalBody
	(*ResultEnvelope)(nil),           // 37: cozy.worker.v1.ResultEnvelope
	(*AdjustmentRow)(nil),            // 38: cozy.worker.v1.AdjustmentRow
	(*TerminalCause)(nil),            // 39: cozy.worker.v1.TerminalCause
	(*ResourceShortfall)(nil),        // 40: cozy.worker.v1.ResourceShortfall
	(*TerminalAck)(nil),              // 41: cozy.worker.v1.TerminalAck
	(*JobCheckpointRequest)(nil),     // 42: cozy.worker.v1.JobCheckpointRequest
	(*JobCheckpointReceipt)(nil),     // 43: cozy.worker.v1.JobCheckpointReceipt
	(*CheckpointFault)(nil),          // 44: cozy.worker.v1.CheckpointFault
	(*JobCheckpointAck)(nil),         // 45: cozy.worker.v1.JobCheckpointAck
	(*ProgressOpen)(nil),             // 46: cozy.worker.v1.ProgressOpen
	(*AttemptProgress)(nil),          // 47: cozy.worker.v1.AttemptProgress
	(*InvocationSpec)(nil),           // 48: cozy.worker.v1.InvocationSpec
	(*InputBinding)(nil),             // 49: cozy.worker.v1.InputBinding
	(*OutputBinding)(nil),            // 50: cozy.worker.v1.OutputBinding
	(*ServingInvocationSpec)(nil),    // 51: cozy.worker.v1.ServingInvocationSpec
	(*JobInvocationSpec)(nil),        // 52: cozy.worker.v1.JobInvocationSpec
	(*DeliveryGrant)(nil),            // 53: cozy.worker.v1.DeliveryGrant
	(*InputAccess)(nil),              // 54: cozy.worker.v1.InputAccess
	(*OutputAccess)(nil),             // 55: cozy.worker.v1.OutputAccess
	(*Credential)(nil),               // 56: cozy.worker.v1.Credential
	(*ResourceCaps)(nil),             // 57: cozy.worker.v1.ResourceCaps
	(*PublicationContract)(nil),      // 58: cozy.worker.v1.PublicationContract
	(*WorkerResources)(nil),          // 59: cozy.worker.v1.WorkerResources
	(*JobCapacity)(nil),              // 60: cozy.worker.v1.JobCapacity
	(*ActiveAttempt)(nil),            // 61: cozy.worker.v1.ActiveAttempt
	(*Fault)(nil),                    // 62: cozy.worker.v1.Fault
	(*OutputManifest)(nil),           // 63: cozy.worker.v1.OutputManifest
	(*OutputEntry)(nil),              // 64: cozy.worker.v1.OutputEntry
	(*AttemptMetrics)(nil),           // 65: cozy.worker.v1.AttemptMetrics
	(*TriageBundleRef)(nil),          // 66: cozy.worker.v1.TriageBundleRef
}
var file_cozy_worker_v1_worker_proto_depIdxs = []int32{
	15, // 0: cozy.worker.v1.OwnerFrame.claim:type_name -> cozy.worker.v1.Claim
	22, // 1: cozy.worker.v1.OwnerFrame.directive:type_name -> cozy.worker.v1.Directive
	31, // 2: cozy.worker.v1.OwnerFrame.start_attempt:type_name -> cozy.worker.v1.StartAttempt
	34, // 3: cozy.worker.v1.OwnerFrame.cancel_attempt:type_name -> cozy.worker.v1.CancelAttempt
	41, // 4: cozy.worker.v1.OwnerFrame.terminal_ack:type_name -> cozy.worker.v1.TerminalAck
	43, // 5: cozy.worker.v1.OwnerFrame.checkpoint_receipt:type_name -> cozy.worker.v1.JobCheckpointReceipt
	21, // 6: cozy.worker.v1.OwnerFrame.snapshot_ack:type_name -> cozy.worker.v1.SnapshotAck
	16, // 7: cozy.worker.v1.WorkerFrame.claim_ack:type_name -> cozy.worker.v1.ClaimAck
	27, // 8: cozy.worker.v1.WorkerFrame.report:type_name -> cozy.worker.v1.Report
	32, // 9: cozy.worker.v1.WorkerFrame.attempt_accepted:type_name -> cozy.worker.v1.AttemptAccepted
	35, // 10: cozy.worker.v1.WorkerFrame.attempt_terminal:type_name -> cozy.worker.v1.AttemptTerminal
	17, // 11: cozy.worker.v1.WorkerFrame.boot_failure:type_name -> cozy.worker.v1.BootFailure
	42, // 12: cozy.worker.v1.WorkerFrame.checkpoint_request:type_name -> cozy.worker.v1.JobCheckpointRequest
	45, // 13: cozy.worker.v1.WorkerFrame.checkpoint_ack:type_name -> cozy.worker.v1.JobCheckpointAck
	18, // 14: cozy.worker.v1.WorkerFrame.snapshot_begin:type_name -> cozy.worker.v1.SnapshotBegin
	20, // 15: cozy.worker.v1.WorkerFrame.snapshot_end:type_name -> cozy.worker.v1.SnapshotEnd
	19, // 16: cozy.worker.v1.WorkerFrame.snapshot_entry:type_name -> cozy.worker.v1.SnapshotEntry
	8,  // 17: cozy.worker.v1.ClaimAck.rejection:type_name -> cozy.worker.v1.ClaimRejection
	59, // 18: cozy.worker.v1.ClaimAck.resources:type_name -> cozy.worker.v1.WorkerResources
	10, // 19: cozy.worker.v1.BootFailure.reason:type_name -> cozy.worker.v1.BootFailureReason
	59, // 20: cozy.worker.v1.BootFailure.resources:type_name -> cozy.worker.v1.WorkerResources
	61, // 21: cozy.worker.v1.SnapshotEntry.attempt:type_name -> cozy.worker.v1.ActiveAttempt
	0,  // 22: cozy.worker.v1.Directive.posture:type_name -> cozy.worker.v1.Posture
	26, // 23: cozy.worker.v1.Directive.job:type_name -> cozy.worker.v1.JobDirective
	23, // 24: cozy.worker.v1.Directive.deployment_set:type_name -> cozy.worker.v1.DeploymentSetDirective
	24, // 25: cozy.worker.v1.DeploymentSetDirective.set:type_name -> cozy.worker.v1.DeploymentSet
	25, // 26: cozy.worker.v1.DeploymentSet.deployments:type_name -> cozy.worker.v1.Deployment
	57, // 27: cozy.worker.v1.JobDirective.resource_caps:type_name -> cozy.worker.v1.ResourceCaps
	58, // 28: cozy.worker.v1.JobDirective.publication_contract:type_name -> cozy.worker.v1.PublicationContract
	1,  // 29: cozy.worker.v1.Report.intake_state:type_name -> cozy.worker.v1.IntakeState
	61, // 30: cozy.worker.v1.Report.active_attempts:type_name -> cozy.worker.v1.ActiveAttempt
	62, // 31: cozy.worker.v1.Report.faults:type_name -> cozy.worker.v1.Fault
	30, // 32: cozy.worker.v1.Report.activity:type_name -> cozy.worker.v1.ActivityEvent
	60, // 33: cozy.worker.v1.Report.job_capacity:type_name -> cozy.worker.v1.JobCapacity
	28, // 34: cozy.worker.v1.Report.deployments:type_name -> cozy.worker.v1.DeploymentStatus
	1,  // 35: cozy.worker.v1.DeploymentStatus.intake_state:type_name -> cozy.worker.v1.IntakeState
	62, // 36: cozy.worker.v1.DeploymentStatus.faults:type_name -> cozy.worker.v1.Fault
	29, // 37: cozy.worker.v1.DeploymentStatus.accelerator:type_name -> cozy.worker.v1.AcceleratorQualification
	53, // 38: cozy.worker.v1.StartAttempt.grant:type_name -> cozy.worker.v1.DeliveryGrant
	33, // 39: cozy.worker.v1.AttemptAccepted.plan:type_name -> cozy.worker.v1.AttemptPlanSummary
	7,  // 40: cozy.worker.v1.CancelAttempt.reason:type_name -> cozy.worker.v1.CancelReason
	4,  // 41: cozy.worker.v1.TerminalBody.status:type_name -> cozy.worker.v1.TerminalStatus
	63, // 42: cozy.worker.v1.TerminalBody.output_manifest:type_name -> cozy.worker.v1.OutputManifest
	65, // 43: cozy.worker.v1.TerminalBody.metrics:type_name -> cozy.worker.v1.AttemptMetrics
	66, // 44: cozy.worker.v1.TerminalBody.triage_bundle:type_name -> cozy.worker.v1.TriageBundleRef
	39, // 45: cozy.worker.v1.TerminalBody.cause:type_name -> cozy.worker.v1.TerminalCause
	37, // 46: cozy.worker.v1.TerminalBody.result:type_name -> cozy.worker.v1.ResultEnvelope
	64, // 47: cozy.worker.v1.ResultEnvelope.result_blob:type_name -> cozy.worker.v1.OutputEntry
	38, // 48: cozy.worker.v1.ResultEnvelope.adjustments:type_name -> cozy.worker.v1.AdjustmentRow
	5,  // 49: cozy.worker.v1.TerminalCause.code:type_name -> cozy.worker.v1.CauseCode
	6,  // 50: cozy.worker.v1.TerminalCause.origin:type_name -> cozy.worker.v1.CauseOrigin
	40, // 51: cozy.worker.v1.TerminalCause.shortfall:type_name -> cozy.worker.v1.ResourceShortfall
	64, // 52: cozy.worker.v1.JobCheckpointRequest.artifact:type_name -> cozy.worker.v1.OutputEntry
	11, // 53: cozy.worker.v1.JobCheckpointReceipt.outcome:type_name -> cozy.worker.v1.CheckpointOutcome
	44, // 54: cozy.worker.v1.JobCheckpointReceipt.fault:type_name -> cozy.worker.v1.CheckpointFault
	12, // 55: cozy.worker.v1.CheckpointFault.code:type_name -> cozy.worker.v1.CheckpointFaultCode
	49, // 56: cozy.worker.v1.InvocationSpec.inputs:type_name -> cozy.worker.v1.InputBinding
	50, // 57: cozy.worker.v1.InvocationSpec.outputs:type_name -> cozy.worker.v1.OutputBinding
	51, // 58: cozy.worker.v1.InvocationSpec.serving:type_name -> cozy.worker.v1.ServingInvocationSpec
	52, // 59: cozy.worker.v1.InvocationSpec.job:type_name -> cozy.worker.v1.JobInvocationSpec
	58, // 60: cozy.worker.v1.JobInvocationSpec.publication_contract:type_name -> cozy.worker.v1.PublicationContract
	56, // 61: cozy.worker.v1.DeliveryGrant.credential:type_name -> cozy.worker.v1.Credential
	54, // 62: cozy.worker.v1.DeliveryGrant.inputs:type_name -> cozy.worker.v1.InputAccess
	55, // 63: cozy.worker.v1.DeliveryGrant.outputs:type_name -> cozy.worker.v1.OutputAccess
	50, // 64: cozy.worker.v1.PublicationContract.outputs:type_name -> cozy.worker.v1.OutputBinding
	2,  // 65: cozy.worker.v1.ActiveAttempt.kind:type_name -> cozy.worker.v1.AttemptKind
	3,  // 66: cozy.worker.v1.ActiveAttempt.state:type_name -> cozy.worker.v1.AttemptState
	9,  // 67: cozy.worker.v1.Fault.kind:type_name -> cozy.worker.v1.FaultKind
	64, // 68: cozy.worker.v1.OutputManifest.outputs:type_name -> cozy.worker.v1.OutputEntry
	13, // 69: cozy.worker.v1.WorkerControl.Control:input_type -> cozy.worker.v1.OwnerFrame
	46, // 70: cozy.worker.v1.WorkerControl.WatchProgress:input_type -> cozy.worker.v1.ProgressOpen
	14, // 71: cozy.worker.v1.WorkerControl.Control:output_type -> cozy.worker.v1.WorkerFrame
	47, // 72: cozy.worker.v1.WorkerControl.WatchProgress:output_type -> cozy.worker.v1.AttemptProgress
	71, // [71:73] is the sub-list for method output_type
	69, // [69:71] is the sub-list for method input_type
	69, // [69:69] is the sub-list for extension type_name
	69, // [69:69] is the sub-list for extension extendee
	0,  // [0:69] is the sub-list for field type_name
}

func init() { file_cozy_worker_v1_worker_proto_init() }
func file_cozy_worker_v1_worker_proto_init() {
	if File_cozy_worker_v1_worker_proto != nil {
		return
	}
	file_cozy_worker_v1_worker_proto_msgTypes[0].OneofWrappers = []any{
		(*OwnerFrame_Claim)(nil),
		(*OwnerFrame_Directive)(nil),
		(*OwnerFrame_StartAttempt)(nil),
		(*OwnerFrame_CancelAttempt)(nil),
		(*OwnerFrame_TerminalAck)(nil),
		(*OwnerFrame_CheckpointReceipt)(nil),
		(*OwnerFrame_SnapshotAck)(nil),
	}
	file_cozy_worker_v1_worker_proto_msgTypes[1].OneofWrappers = []any{
		(*WorkerFrame_ClaimAck)(nil),
		(*WorkerFrame_Report)(nil),
		(*WorkerFrame_AttemptAccepted)(nil),
		(*WorkerFrame_AttemptTerminal)(nil),
		(*WorkerFrame_BootFailure)(nil),
		(*WorkerFrame_CheckpointRequest)(nil),
		(*WorkerFrame_CheckpointAck)(nil),
		(*WorkerFrame_SnapshotBegin)(nil),
		(*WorkerFrame_SnapshotEnd)(nil),
		(*WorkerFrame_SnapshotEntry)(nil),
	}
	file_cozy_worker_v1_worker_proto_msgTypes[9].OneofWrappers = []any{
		(*Directive_Job)(nil),
		(*Directive_DeploymentSet)(nil),
	}
	file_cozy_worker_v1_worker_proto_msgTypes[35].OneofWrappers = []any{
		(*InvocationSpec_Serving)(nil),
		(*InvocationSpec_Job)(nil),
	}
	type x struct{}
	out := protoimpl.TypeBuilder{
		File: protoimpl.DescBuilder{
			GoPackagePath: reflect.TypeOf(x{}).PkgPath(),
			RawDescriptor: unsafe.Slice(unsafe.StringData(file_cozy_worker_v1_worker_proto_rawDesc), len(file_cozy_worker_v1_worker_proto_rawDesc)),
			NumEnums:      13,
			NumMessages:   54,
			NumExtensions: 0,
			NumServices:   1,
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
