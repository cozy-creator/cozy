// cozy.worker.v1 — the worker<->hub wire contract. Schema only; this package has no behavior.
//
// Authority: tracker-v2/worker-protocol/01-wire-schema.md (schema), 02-lifecycle.md (behavior),
// 03-versioning-conformance.md (versioning). Issue: tracker/tensorhub/th-024.
//
// VERSIONING (03 §1, §2 R1-R8):
//   - The MAJOR is in the package path. `cozy.worker.v1` IS major 1; the gRPC method paths
//     (/cozy.worker.v1.Worker/Control, .../StreamProgress, .../Purge) route before a body byte
//     is parsed. An unsupported major surfaces as gRPC UNIMPLEMENTED and is fatal-no-retry.
//   - The MINOR is `wire_minor` (uint32): a linear-train number, declared once on
//     Register/RegisterAck, carried on Directive, echoed on Report. effective = min(W, H).
//     Never a negotiation, never a capability list, never a wire-admission gate.
//     The current minor of THIS file is the WIRE_MINOR constant at the repo root.
//   - R1/R2: field numbers are assigned monotonically and never reused; the (number, name)
//     pair is frozen at introduction. Meaning is keyed to the NUMBER.
//   - R3: a meaning change (type, cardinality, unit, semantics) is a NEW number + NEW name;
//     the old number AND old name move to `reserved`.
//   - R5: within a major, deprecate (stop producing) and reserve (prevent reuse). Structural
//     removal is a major.
//   - R6: `reserved` is scoped to one major and never inherited. 01 §7.2 predicted this major
//     would carry NO `reserved` entries; ten messages carry them, for two reasons, neither a
//     placeholder. (i) The ONEOF-LAST RULE vacated numbers that 01 §3 assigns to live meanings.
//     (ii) The canonical-bytes redesign moved whole document BODIES out of their wire messages
//     (StartAttempt, AttemptTerminal, PurgeArtifact) and deleted three fields outright
//     (OutputManifest.digest, AppliedAdapter.scale, TriageBundleRef.write_receipt). Either way
//     the tombstone turns a doc-faithful misreading into a compile error rather than a silent
//     wrong-slot write. Where a meaning only MOVED, the name moves with it and only the NUMBER
//     is reserved; where a field was deleted or renamed under R3, that is recorded at the
//     message.
//   - R7: a reserved number or name on the wire is MALFORMED (refused), not an ignorable
//     unknown field. Never-assigned numbers are skipped per normal proto3 semantics.
//   - R8: no `map` anywhere; no `required`; every field's absence-default equals its
//     pre-introduction behavior, which is what makes additive minors degrade both ways.
//
// IDENTITY IS CANONICAL BYTES, PROTOBUF IS TRANSPORT (th-024 redesign, 2026-08-24; supersedes
// the earlier "digest over marshaled protobuf" design — see the decisions row):
//
//   No digest in this protocol is ever computed over protobuf-marshaled bytes. Protobuf
//   specifies NO field emission order, so marshaled bytes are not a canonical format: two
//   encoders agreeing is an implementation property that guarantees nothing for a third
//   implementation or a future library version.
//
//   Wherever a digest fences MEANING, it is the SHA-256 of a DOCUMENT's exact canonical
//   bytes — RFC 8785-shaped canonical JSON under tfs-013's writer rules (tensorfs-v2:
//   sorted keys, no whitespace, integer-only numerics in +/-(2^53-1), printable-ASCII fields,
//   only \" and \\ escapes, lowercase `sha256:` digests, duplicate-key and unknown-field
//   refusal, and the RE-EMIT LAW: stored bytes are the canonical bytes of their own content).
//   Where the fence requires BYTE AGREEMENT — i.e. the receiver must recompute the digest
//   rather than take the sender's word — the canonical bytes travel IN the message, as a
//   `bytes` field, INSTEAD OF the structured body. Verification is then sha256 over bytes
//   already present: a third implementation needs no canonicalizer to VERIFY, only to AUTHOR.
//   Two representations of one document never travel together, because two representations
//   that must agree are a second canonicalization free to disagree.
//
// DIGEST CLASSES. Every digest-bearing field in this file is exactly one of:
//   (a) DOCUMENT IDENTITY over canonical JSON bytes. The document is named at the field.
//       Protocol-owned documents are the `*Body`/ExecutionSpec DOCUMENT SHAPES below; foreign
//       documents (AttemptPlan, ModelConstructionContract, EntrypointBindingPlan,
//       AttemptBinding, WorkerTriageBundle, OutputManifest, LocalArtifactAbsenceReceipt,
//       evaluated config, result surface) freeze in their OWNING issues under the same writer
//       and are transported here as identity only — cited, never redefined.
//   (b) CONTENT digest over opaque artifact bytes (input assets, outputs, checkpoint
//       artifacts, OCI images). Not a JSON document; no canonicalization is involved.
// On the wire a digest is raw 32-byte SHA-256 `bytes`; inside a canonical document the SAME
// digest is spelled `"sha256:<64 lowercase hex>"`. One identity, two spellings.
//
// BYTES NAMING LAW (mechanical, so both spellings are derivable from the descriptor alone):
// a `bytes` field named `digest` or `*_digest` IS a 32-byte SHA-256 and canonicalizes as
// `sha256:<hex>`; every other `bytes` field is an opaque payload and canonicalizes as base64.
//
// WIRE DETERMINISM (belt and braces, NOT identity): no `map` anywhere; every set is a
// `repeated` with a fixed sort order stated at the field; times are uint64 unix
// seconds/millis (never Timestamp); human identity strings are `string` in canonical form;
// secrets are `bytes`.
//
// ONEOF-LAST RULE (th-024, kept from a MEASURED divergence — see the decisions row):
// protobuf does not specify field emission order, and the two shipped bindings disagree.
// The Python/upb encoder emits in ascending field-number order; the Go encoder emits `oneof`
// members AFTER all regular fields, so a mid-message oneof makes the two produce DIFFERENT
// bytes for the same logical message. In every message the `oneof` members therefore carry
// the HIGHEST field numbers, so the two orders coincide. This is a DETERMINISM convenience
// for diffing and caching the transport encoding; it is NOT an identity guarantee and no
// fence in this protocol rests on it. Numbers vacated by the rule are `reserved` at their
// message so a reader implementing the field tables of 01 §3 verbatim gets a COMPILE ERROR
// rather than a silent wrong-slot write.
//
// ENUM VALUE NAMING (th-024 implementation rule, diverges from the value names printed in
// 01 §2/§4): proto3 scopes enum VALUES in the enclosing package, not the enum, so the spec's
// unprefixed names collide (DRAINING in Posture and IntakeState, GRANT_EXPIRED in CauseCode
// and FaultKind, CLIENT in CauseOrigin and CancelReason) and do not compile. Every value is
// therefore prefixed with its enum's SCREAMING_SNAKE type name. This is the only scheme that
// stays stable under R2: with it, a future enum can never force the rename of an existing
// value. Numbers are unchanged from the spec and are the normative part.

// Code generated by protoc-gen-go. DO NOT EDIT.
// versions:
// 	protoc-gen-go v1.36.11
// 	protoc        v6.31.1
// source: cozy/worker/v1/worker.proto

package workerprotov1

import (
	protoreflect "google.golang.org/protobuf/reflect/protoreflect"
	protoimpl "google.golang.org/protobuf/runtime/protoimpl"
	emptypb "google.golang.org/protobuf/types/known/emptypb"
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
	IntakeState_INTAKE_STATE_READY       IntakeState = 4 // first_request_servable, never merely "connected"
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

// The ENTIRE terminal vocabulary, fenced like a frozen token list: a new meaning is a new
// MAJOR, never a new value. RETRYABLE and FATAL are gone from the wire by design.
// TERMINAL_STATUS_UNSPECIFIED is wire-invalid on a sent terminal; it exists only as the
// proto3 zero value.
type TerminalStatus int32

const (
	TerminalStatus_TERMINAL_STATUS_UNSPECIFIED TerminalStatus = 0
	TerminalStatus_TERMINAL_STATUS_SUCCEEDED   TerminalStatus = 1 // success; output manifest attached
	TerminalStatus_TERMINAL_STATUS_REFUSED     TerminalStatus = 2 // typed refusal; cause names which kind
	TerminalStatus_TERMINAL_STATUS_FAILED      TerminalStatus = 3 // execution failure; cause separates deterministic-body
	// failure from infra fault
	TerminalStatus_TERMINAL_STATUS_CANCELED TerminalStatus = 4 // cancel committed; cause separates client / deadline /
	// drain / policy / supersession
	TerminalStatus_TERMINAL_STATUS_ABANDONED TerminalStatus = 5 // accepted-but-incomplete, lost to executor invalidation
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
	CauseCode_CAUSE_CODE_INVALID_REQUEST       CauseCode = 1 // author-raised typed 4xx-class fact
	CauseCode_CAUSE_CODE_UNSUPPORTED_INPUT     CauseCode = 2 // valid but unsupported semantic combination
	CauseCode_CAUSE_CODE_LOCAL_SAFETY          CauseCode = 3 // byte/contract/fit/mapping validation refused
	CauseCode_CAUSE_CODE_PROTOCOL              CauseCode = 4 // fence/idempotency refusal surfaced as a terminal
	CauseCode_CAUSE_CODE_CONSTRAINT_INFEASIBLE CauseCode = 5 // explicit request constraint no plan satisfies
	// FAILED
	CauseCode_CAUSE_CODE_AUTHOR_EXCEPTION       CauseCode = 6  // deterministic body failure - terminal, never requeued
	CauseCode_CAUSE_CODE_EXECUTOR_FAULT         CauseCode = 7  // OOM / kernel fault / crash - infra class
	CauseCode_CAUSE_CODE_GRANT_EXPIRED          CauseCode = 8  // resolver fallback - infra class
	CauseCode_CAUSE_CODE_ARTIFACT_UNFETCHABLE   CauseCode = 9  // resolver fallback - infra class
	CauseCode_CAUSE_CODE_CAPABILITY_UNAVAILABLE CauseCode = 10 // capability mint/spool failure - infra class
	// CANCELED
	CauseCode_CAUSE_CODE_CLIENT_CANCEL     CauseCode = 11
	CauseCode_CAUSE_CODE_DEADLINE_EXPIRED  CauseCode = 12 // supervisor-enforced; distinguishable from client
	CauseCode_CAUSE_CODE_DRAIN_CANCEL      CauseCode = 13
	CauseCode_CAUSE_CODE_POLICY_CANCEL     CauseCode = 14
	CauseCode_CAUSE_CODE_SUPERSEDED_CANCEL CauseCode = 15 // explicit journaled supersession
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
	CauseOrigin_CAUSE_ORIGIN_AUTHOR      CauseOrigin = 1 // endpoint/job body
	CauseOrigin_CAUSE_ORIGIN_RUNTIME     CauseOrigin = 2 // cozy-runtime in-process machinery
	CauseOrigin_CAUSE_ORIGIN_EXECUTOR    CauseOrigin = 3 // device executor process / CUDA
	CauseOrigin_CAUSE_ORIGIN_SUPERVISOR  CauseOrigin = 4 // the worker-local supervisor (control parent) process
	CauseOrigin_CAUSE_ORIGIN_INFRA       CauseOrigin = 5 // transport/storage/grant plumbing
	CauseOrigin_CAUSE_ORIGIN_CLIENT      CauseOrigin = 6 // caller action
	CauseOrigin_CAUSE_ORIGIN_COORDINATOR CauseOrigin = 7 // hub decision (drain, policy, supersession)
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
	CancelReason_CANCEL_REASON_DEADLINE    CancelReason = 5 // a hub-side deadline decision; the supervisor also
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

type RegisterRejection int32

const (
	RegisterRejection_REGISTER_REJECTION_UNSPECIFIED         RegisterRejection = 0
	RegisterRejection_REGISTER_REJECTION_UNAUTHENTICATED     RegisterRejection = 1
	RegisterRejection_REGISTER_REJECTION_SESSION_COLLISION   RegisterRejection = 2 // session_id already bound to another live worker
	RegisterRejection_REGISTER_REJECTION_STALE_SESSION       RegisterRejection = 3
	RegisterRejection_REGISTER_REJECTION_WORKER_ID_MISMATCH  RegisterRejection = 4
	RegisterRejection_REGISTER_REJECTION_RELEASE_ID_MISMATCH RegisterRejection = 5
	RegisterRejection_REGISTER_REJECTION_UNSUPPORTED_MINOR   RegisterRejection = 6
)

// Enum value maps for RegisterRejection.
var (
	RegisterRejection_name = map[int32]string{
		0: "REGISTER_REJECTION_UNSPECIFIED",
		1: "REGISTER_REJECTION_UNAUTHENTICATED",
		2: "REGISTER_REJECTION_SESSION_COLLISION",
		3: "REGISTER_REJECTION_STALE_SESSION",
		4: "REGISTER_REJECTION_WORKER_ID_MISMATCH",
		5: "REGISTER_REJECTION_RELEASE_ID_MISMATCH",
		6: "REGISTER_REJECTION_UNSUPPORTED_MINOR",
	}
	RegisterRejection_value = map[string]int32{
		"REGISTER_REJECTION_UNSPECIFIED":         0,
		"REGISTER_REJECTION_UNAUTHENTICATED":     1,
		"REGISTER_REJECTION_SESSION_COLLISION":   2,
		"REGISTER_REJECTION_STALE_SESSION":       3,
		"REGISTER_REJECTION_WORKER_ID_MISMATCH":  4,
		"REGISTER_REJECTION_RELEASE_ID_MISMATCH": 5,
		"REGISTER_REJECTION_UNSUPPORTED_MINOR":   6,
	}
)

func (x RegisterRejection) Enum() *RegisterRejection {
	p := new(RegisterRejection)
	*p = x
	return p
}

func (x RegisterRejection) String() string {
	return protoimpl.X.EnumStringOf(x.Descriptor(), protoreflect.EnumNumber(x))
}

func (RegisterRejection) Descriptor() protoreflect.EnumDescriptor {
	return file_cozy_worker_v1_worker_proto_enumTypes[8].Descriptor()
}

func (RegisterRejection) Type() protoreflect.EnumType {
	return &file_cozy_worker_v1_worker_proto_enumTypes[8]
}

func (x RegisterRejection) Number() protoreflect.EnumNumber {
	return protoreflect.EnumNumber(x)
}

// Deprecated: Use RegisterRejection.Descriptor instead.
func (RegisterRejection) EnumDescriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{8}
}

type FaultKind int32

const (
	FaultKind_FAULT_KIND_UNSPECIFIED         FaultKind = 0
	FaultKind_FAULT_KIND_BINDING_UNAVAILABLE FaultKind = 1
	FaultKind_FAULT_KIND_BINDING_DEGRADED    FaultKind = 2
	FaultKind_FAULT_KIND_HARDWARE_UNSUITABLE FaultKind = 3 // STRUCTURAL impossibility only (arch/encoding);
	// never fit/VRAM - fit DEGRADES, it does not fault
	FaultKind_FAULT_KIND_ARTIFACT_FETCH_FAILED FaultKind = 4
	FaultKind_FAULT_KIND_GRANT_EXPIRED         FaultKind = 5
	FaultKind_FAULT_KIND_CREDENTIAL_UNAPPLIED  FaultKind = 6
	FaultKind_FAULT_KIND_LOCAL_SAFETY_REFUSAL  FaultKind = 7
	FaultKind_FAULT_KIND_EXECUTOR_POISONED     FaultKind = 8
	FaultKind_FAULT_KIND_CONFIG_REFUSED        FaultKind = 9
	FaultKind_FAULT_KIND_UNKNOWN_DEPLOYMENT    FaultKind = 10
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
	}
	FaultKind_value = map[string]int32{
		"FAULT_KIND_UNSPECIFIED":           0,
		"FAULT_KIND_BINDING_UNAVAILABLE":   1,
		"FAULT_KIND_BINDING_DEGRADED":      2,
		"FAULT_KIND_HARDWARE_UNSUITABLE":   3,
		"FAULT_KIND_ARTIFACT_FETCH_FAILED": 4,
		"FAULT_KIND_GRANT_EXPIRED":         5,
		"FAULT_KIND_CREDENTIAL_UNAPPLIED":  6,
		"FAULT_KIND_LOCAL_SAFETY_REFUSAL":  7,
		"FAULT_KIND_EXECUTOR_POISONED":     8,
		"FAULT_KIND_CONFIG_REFUSED":        9,
		"FAULT_KIND_UNKNOWN_DEPLOYMENT":    10,
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

type SubjectKind int32

const (
	SubjectKind_SUBJECT_KIND_UNSPECIFIED      SubjectKind = 0
	SubjectKind_SUBJECT_KIND_OBJECT           SubjectKind = 1
	SubjectKind_SUBJECT_KIND_SNAPSHOT         SubjectKind = 2
	SubjectKind_SUBJECT_KIND_TENSOR_HEADER    SubjectKind = 3
	SubjectKind_SUBJECT_KIND_CHECKPOINT       SubjectKind = 4
	SubjectKind_SUBJECT_KIND_ADAPTER_ARTIFACT SubjectKind = 5
	SubjectKind_SUBJECT_KIND_LAYOUT           SubjectKind = 6
	SubjectKind_SUBJECT_KIND_RELEASE          SubjectKind = 7
)

// Enum value maps for SubjectKind.
var (
	SubjectKind_name = map[int32]string{
		0: "SUBJECT_KIND_UNSPECIFIED",
		1: "SUBJECT_KIND_OBJECT",
		2: "SUBJECT_KIND_SNAPSHOT",
		3: "SUBJECT_KIND_TENSOR_HEADER",
		4: "SUBJECT_KIND_CHECKPOINT",
		5: "SUBJECT_KIND_ADAPTER_ARTIFACT",
		6: "SUBJECT_KIND_LAYOUT",
		7: "SUBJECT_KIND_RELEASE",
	}
	SubjectKind_value = map[string]int32{
		"SUBJECT_KIND_UNSPECIFIED":      0,
		"SUBJECT_KIND_OBJECT":           1,
		"SUBJECT_KIND_SNAPSHOT":         2,
		"SUBJECT_KIND_TENSOR_HEADER":    3,
		"SUBJECT_KIND_CHECKPOINT":       4,
		"SUBJECT_KIND_ADAPTER_ARTIFACT": 5,
		"SUBJECT_KIND_LAYOUT":           6,
		"SUBJECT_KIND_RELEASE":          7,
	}
)

func (x SubjectKind) Enum() *SubjectKind {
	p := new(SubjectKind)
	*p = x
	return p
}

func (x SubjectKind) String() string {
	return protoimpl.X.EnumStringOf(x.Descriptor(), protoreflect.EnumNumber(x))
}

func (SubjectKind) Descriptor() protoreflect.EnumDescriptor {
	return file_cozy_worker_v1_worker_proto_enumTypes[10].Descriptor()
}

func (SubjectKind) Type() protoreflect.EnumType {
	return &file_cozy_worker_v1_worker_proto_enumTypes[10]
}

func (x SubjectKind) Number() protoreflect.EnumNumber {
	return protoreflect.EnumNumber(x)
}

// Deprecated: Use SubjectKind.Descriptor instead.
func (SubjectKind) EnumDescriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{10}
}

type PurgeFaultCode int32

const (
	PurgeFaultCode_PURGE_FAULT_CODE_UNSPECIFIED                PurgeFaultCode = 0
	PurgeFaultCode_PURGE_FAULT_CODE_STALE_TOMBSTONE_GENERATION PurgeFaultCode = 1
	PurgeFaultCode_PURGE_FAULT_CODE_STALE_WORKER_GENERATION    PurgeFaultCode = 2
	PurgeFaultCode_PURGE_FAULT_CODE_CHANGED_BODY_REPLAY        PurgeFaultCode = 3
	PurgeFaultCode_PURGE_FAULT_CODE_CROSS_TENANT_SUBJECT       PurgeFaultCode = 4
	PurgeFaultCode_PURGE_FAULT_CODE_LIVE_UNRELATED_BINDING     PurgeFaultCode = 5
	PurgeFaultCode_PURGE_FAULT_CODE_ERASURE_FAILED             PurgeFaultCode = 6
	PurgeFaultCode_PURGE_FAULT_CODE_UNKNOWN_SUBJECT            PurgeFaultCode = 7
	PurgeFaultCode_PURGE_FAULT_CODE_DEADLINE_EXCEEDED          PurgeFaultCode = 8
)

// Enum value maps for PurgeFaultCode.
var (
	PurgeFaultCode_name = map[int32]string{
		0: "PURGE_FAULT_CODE_UNSPECIFIED",
		1: "PURGE_FAULT_CODE_STALE_TOMBSTONE_GENERATION",
		2: "PURGE_FAULT_CODE_STALE_WORKER_GENERATION",
		3: "PURGE_FAULT_CODE_CHANGED_BODY_REPLAY",
		4: "PURGE_FAULT_CODE_CROSS_TENANT_SUBJECT",
		5: "PURGE_FAULT_CODE_LIVE_UNRELATED_BINDING",
		6: "PURGE_FAULT_CODE_ERASURE_FAILED",
		7: "PURGE_FAULT_CODE_UNKNOWN_SUBJECT",
		8: "PURGE_FAULT_CODE_DEADLINE_EXCEEDED",
	}
	PurgeFaultCode_value = map[string]int32{
		"PURGE_FAULT_CODE_UNSPECIFIED":                0,
		"PURGE_FAULT_CODE_STALE_TOMBSTONE_GENERATION": 1,
		"PURGE_FAULT_CODE_STALE_WORKER_GENERATION":    2,
		"PURGE_FAULT_CODE_CHANGED_BODY_REPLAY":        3,
		"PURGE_FAULT_CODE_CROSS_TENANT_SUBJECT":       4,
		"PURGE_FAULT_CODE_LIVE_UNRELATED_BINDING":     5,
		"PURGE_FAULT_CODE_ERASURE_FAILED":             6,
		"PURGE_FAULT_CODE_UNKNOWN_SUBJECT":            7,
		"PURGE_FAULT_CODE_DEADLINE_EXCEEDED":          8,
	}
)

func (x PurgeFaultCode) Enum() *PurgeFaultCode {
	p := new(PurgeFaultCode)
	*p = x
	return p
}

func (x PurgeFaultCode) String() string {
	return protoimpl.X.EnumStringOf(x.Descriptor(), protoreflect.EnumNumber(x))
}

func (PurgeFaultCode) Descriptor() protoreflect.EnumDescriptor {
	return file_cozy_worker_v1_worker_proto_enumTypes[11].Descriptor()
}

func (PurgeFaultCode) Type() protoreflect.EnumType {
	return &file_cozy_worker_v1_worker_proto_enumTypes[11]
}

func (x PurgeFaultCode) Number() protoreflect.EnumNumber {
	return protoreflect.EnumNumber(x)
}

// Deprecated: Use PurgeFaultCode.Descriptor instead.
func (PurgeFaultCode) EnumDescriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{11}
}

type BootFailureReason int32

const (
	BootFailureReason_BOOT_FAILURE_REASON_UNSPECIFIED      BootFailureReason = 0
	BootFailureReason_BOOT_FAILURE_REASON_HARDWARE_VERDICT BootFailureReason = 1 // device absent/wedged/wrong shape
	BootFailureReason_BOOT_FAILURE_REASON_IMAGE_FAULT      BootFailureReason = 2
	BootFailureReason_BOOT_FAILURE_REASON_DRIVER_FAULT     BootFailureReason = 3
	BootFailureReason_BOOT_FAILURE_REASON_DISK_SHAPE       BootFailureReason = 4 // ENOSPC-class: a claim about the pod shape
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
	return file_cozy_worker_v1_worker_proto_enumTypes[12].Descriptor()
}

func (BootFailureReason) Type() protoreflect.EnumType {
	return &file_cozy_worker_v1_worker_proto_enumTypes[12]
}

func (x BootFailureReason) Number() protoreflect.EnumNumber {
	return protoreflect.EnumNumber(x)
}

// Deprecated: Use BootFailureReason.Descriptor instead.
func (BootFailureReason) EnumDescriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{12}
}

type CheckpointOutcome int32

const (
	CheckpointOutcome_CHECKPOINT_OUTCOME_UNSPECIFIED CheckpointOutcome = 0
	CheckpointOutcome_CHECKPOINT_OUTCOME_RECORDED    CheckpointOutcome = 1 // first durable record of this identity
	CheckpointOutcome_CHECKPOINT_OUTCOME_REPLAYED    CheckpointOutcome = 2 // exact identity already recorded; no duplicate effect
	CheckpointOutcome_CHECKPOINT_OUTCOME_CONFLICT    CheckpointOutcome = 3 // same (operation_key, logical_key), different digest
	CheckpointOutcome_CHECKPOINT_OUTCOME_REFUSED     CheckpointOutcome = 4 // typed refusal (see CheckpointFault)
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
	return file_cozy_worker_v1_worker_proto_enumTypes[13].Descriptor()
}

func (CheckpointOutcome) Type() protoreflect.EnumType {
	return &file_cozy_worker_v1_worker_proto_enumTypes[13]
}

func (x CheckpointOutcome) Number() protoreflect.EnumNumber {
	return protoreflect.EnumNumber(x)
}

// Deprecated: Use CheckpointOutcome.Descriptor instead.
func (CheckpointOutcome) EnumDescriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{13}
}

type CheckpointFaultCode int32

const (
	CheckpointFaultCode_CHECKPOINT_FAULT_CODE_UNSPECIFIED       CheckpointFaultCode = 0
	CheckpointFaultCode_CHECKPOINT_FAULT_CODE_IDENTITY_CONFLICT CheckpointFaultCode = 1 // digest differs from the recorded one
	CheckpointFaultCode_CHECKPOINT_FAULT_CODE_UNKNOWN_ATTEMPT   CheckpointFaultCode = 2 // no such held (request_id, attempt)
	CheckpointFaultCode_CHECKPOINT_FAULT_CODE_NOT_JOB_MODE      CheckpointFaultCode = 3 // a serving attempt has no checkpoint lane
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
	return file_cozy_worker_v1_worker_proto_enumTypes[14].Descriptor()
}

func (CheckpointFaultCode) Type() protoreflect.EnumType {
	return &file_cozy_worker_v1_worker_proto_enumTypes[14]
}

func (x CheckpointFaultCode) Number() protoreflect.EnumNumber {
	return protoreflect.EnumNumber(x)
}

// Deprecated: Use CheckpointFaultCode.Descriptor instead.
func (CheckpointFaultCode) EnumDescriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{14}
}

type WorkerMessage struct {
	state protoimpl.MessageState `protogen:"open.v1"`
	// Types that are valid to be assigned to Msg:
	//
	//	*WorkerMessage_Register
	//	*WorkerMessage_Report
	//	*WorkerMessage_AttemptAccepted
	//	*WorkerMessage_AttemptTerminal
	//	*WorkerMessage_BootFailure
	//	*WorkerMessage_CheckpointRequest
	//	*WorkerMessage_CheckpointAck
	Msg           isWorkerMessage_Msg `protobuf_oneof:"msg"`
	unknownFields protoimpl.UnknownFields
	sizeCache     protoimpl.SizeCache
}

func (x *WorkerMessage) Reset() {
	*x = WorkerMessage{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[0]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *WorkerMessage) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*WorkerMessage) ProtoMessage() {}

func (x *WorkerMessage) ProtoReflect() protoreflect.Message {
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

// Deprecated: Use WorkerMessage.ProtoReflect.Descriptor instead.
func (*WorkerMessage) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{0}
}

func (x *WorkerMessage) GetMsg() isWorkerMessage_Msg {
	if x != nil {
		return x.Msg
	}
	return nil
}

func (x *WorkerMessage) GetRegister() *Register {
	if x != nil {
		if x, ok := x.Msg.(*WorkerMessage_Register); ok {
			return x.Register
		}
	}
	return nil
}

func (x *WorkerMessage) GetReport() *Report {
	if x != nil {
		if x, ok := x.Msg.(*WorkerMessage_Report); ok {
			return x.Report
		}
	}
	return nil
}

func (x *WorkerMessage) GetAttemptAccepted() *AttemptAccepted {
	if x != nil {
		if x, ok := x.Msg.(*WorkerMessage_AttemptAccepted); ok {
			return x.AttemptAccepted
		}
	}
	return nil
}

func (x *WorkerMessage) GetAttemptTerminal() *AttemptTerminal {
	if x != nil {
		if x, ok := x.Msg.(*WorkerMessage_AttemptTerminal); ok {
			return x.AttemptTerminal
		}
	}
	return nil
}

func (x *WorkerMessage) GetBootFailure() *BootFailure {
	if x != nil {
		if x, ok := x.Msg.(*WorkerMessage_BootFailure); ok {
			return x.BootFailure
		}
	}
	return nil
}

func (x *WorkerMessage) GetCheckpointRequest() *JobCheckpointRequest {
	if x != nil {
		if x, ok := x.Msg.(*WorkerMessage_CheckpointRequest); ok {
			return x.CheckpointRequest
		}
	}
	return nil
}

func (x *WorkerMessage) GetCheckpointAck() *JobCheckpointAck {
	if x != nil {
		if x, ok := x.Msg.(*WorkerMessage_CheckpointAck); ok {
			return x.CheckpointAck
		}
	}
	return nil
}

type isWorkerMessage_Msg interface {
	isWorkerMessage_Msg()
}

type WorkerMessage_Register struct {
	Register *Register `protobuf:"bytes,1,opt,name=register,proto3,oneof"`
}

type WorkerMessage_Report struct {
	Report *Report `protobuf:"bytes,2,opt,name=report,proto3,oneof"`
}

type WorkerMessage_AttemptAccepted struct {
	AttemptAccepted *AttemptAccepted `protobuf:"bytes,3,opt,name=attempt_accepted,json=attemptAccepted,proto3,oneof"`
}

type WorkerMessage_AttemptTerminal struct {
	AttemptTerminal *AttemptTerminal `protobuf:"bytes,4,opt,name=attempt_terminal,json=attemptTerminal,proto3,oneof"`
}

type WorkerMessage_BootFailure struct {
	// Valid as the stream's ONLY message: the boot-fatal verdict sent INSTEAD OF Register.
	BootFailure *BootFailure `protobuf:"bytes,5,opt,name=boot_failure,json=bootFailure,proto3,oneof"`
}

type WorkerMessage_CheckpointRequest struct {
	CheckpointRequest *JobCheckpointRequest `protobuf:"bytes,6,opt,name=checkpoint_request,json=checkpointRequest,proto3,oneof"`
}

type WorkerMessage_CheckpointAck struct {
	CheckpointAck *JobCheckpointAck `protobuf:"bytes,7,opt,name=checkpoint_ack,json=checkpointAck,proto3,oneof"`
}

func (*WorkerMessage_Register) isWorkerMessage_Msg() {}

func (*WorkerMessage_Report) isWorkerMessage_Msg() {}

func (*WorkerMessage_AttemptAccepted) isWorkerMessage_Msg() {}

func (*WorkerMessage_AttemptTerminal) isWorkerMessage_Msg() {}

func (*WorkerMessage_BootFailure) isWorkerMessage_Msg() {}

func (*WorkerMessage_CheckpointRequest) isWorkerMessage_Msg() {}

func (*WorkerMessage_CheckpointAck) isWorkerMessage_Msg() {}

type CoordinatorMessage struct {
	state protoimpl.MessageState `protogen:"open.v1"`
	// Types that are valid to be assigned to Msg:
	//
	//	*CoordinatorMessage_RegisterAck
	//	*CoordinatorMessage_Directive
	//	*CoordinatorMessage_StartAttempt
	//	*CoordinatorMessage_CancelAttempt
	//	*CoordinatorMessage_TerminalAck
	//	*CoordinatorMessage_CheckpointReceipt
	Msg           isCoordinatorMessage_Msg `protobuf_oneof:"msg"`
	unknownFields protoimpl.UnknownFields
	sizeCache     protoimpl.SizeCache
}

func (x *CoordinatorMessage) Reset() {
	*x = CoordinatorMessage{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[1]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *CoordinatorMessage) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*CoordinatorMessage) ProtoMessage() {}

func (x *CoordinatorMessage) ProtoReflect() protoreflect.Message {
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

// Deprecated: Use CoordinatorMessage.ProtoReflect.Descriptor instead.
func (*CoordinatorMessage) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{1}
}

func (x *CoordinatorMessage) GetMsg() isCoordinatorMessage_Msg {
	if x != nil {
		return x.Msg
	}
	return nil
}

func (x *CoordinatorMessage) GetRegisterAck() *RegisterAck {
	if x != nil {
		if x, ok := x.Msg.(*CoordinatorMessage_RegisterAck); ok {
			return x.RegisterAck
		}
	}
	return nil
}

func (x *CoordinatorMessage) GetDirective() *Directive {
	if x != nil {
		if x, ok := x.Msg.(*CoordinatorMessage_Directive); ok {
			return x.Directive
		}
	}
	return nil
}

func (x *CoordinatorMessage) GetStartAttempt() *StartAttempt {
	if x != nil {
		if x, ok := x.Msg.(*CoordinatorMessage_StartAttempt); ok {
			return x.StartAttempt
		}
	}
	return nil
}

func (x *CoordinatorMessage) GetCancelAttempt() *CancelAttempt {
	if x != nil {
		if x, ok := x.Msg.(*CoordinatorMessage_CancelAttempt); ok {
			return x.CancelAttempt
		}
	}
	return nil
}

func (x *CoordinatorMessage) GetTerminalAck() *TerminalAck {
	if x != nil {
		if x, ok := x.Msg.(*CoordinatorMessage_TerminalAck); ok {
			return x.TerminalAck
		}
	}
	return nil
}

func (x *CoordinatorMessage) GetCheckpointReceipt() *JobCheckpointReceipt {
	if x != nil {
		if x, ok := x.Msg.(*CoordinatorMessage_CheckpointReceipt); ok {
			return x.CheckpointReceipt
		}
	}
	return nil
}

type isCoordinatorMessage_Msg interface {
	isCoordinatorMessage_Msg()
}

type CoordinatorMessage_RegisterAck struct {
	RegisterAck *RegisterAck `protobuf:"bytes,1,opt,name=register_ack,json=registerAck,proto3,oneof"`
}

type CoordinatorMessage_Directive struct {
	Directive *Directive `protobuf:"bytes,2,opt,name=directive,proto3,oneof"`
}

type CoordinatorMessage_StartAttempt struct {
	StartAttempt *StartAttempt `protobuf:"bytes,3,opt,name=start_attempt,json=startAttempt,proto3,oneof"`
}

type CoordinatorMessage_CancelAttempt struct {
	CancelAttempt *CancelAttempt `protobuf:"bytes,4,opt,name=cancel_attempt,json=cancelAttempt,proto3,oneof"`
}

type CoordinatorMessage_TerminalAck struct {
	TerminalAck *TerminalAck `protobuf:"bytes,5,opt,name=terminal_ack,json=terminalAck,proto3,oneof"`
}

type CoordinatorMessage_CheckpointReceipt struct {
	CheckpointReceipt *JobCheckpointReceipt `protobuf:"bytes,6,opt,name=checkpoint_receipt,json=checkpointReceipt,proto3,oneof"`
}

func (*CoordinatorMessage_RegisterAck) isCoordinatorMessage_Msg() {}

func (*CoordinatorMessage_Directive) isCoordinatorMessage_Msg() {}

func (*CoordinatorMessage_StartAttempt) isCoordinatorMessage_Msg() {}

func (*CoordinatorMessage_CancelAttempt) isCoordinatorMessage_Msg() {}

func (*CoordinatorMessage_TerminalAck) isCoordinatorMessage_Msg() {}

func (*CoordinatorMessage_CheckpointReceipt) isCoordinatorMessage_Msg() {}

type PurgeMessage struct {
	state protoimpl.MessageState `protogen:"open.v1"`
	// Types that are valid to be assigned to Msg:
	//
	//	*PurgeMessage_PurgeArtifact
	//	*PurgeMessage_PurgeArtifactReport
	Msg           isPurgeMessage_Msg `protobuf_oneof:"msg"`
	unknownFields protoimpl.UnknownFields
	sizeCache     protoimpl.SizeCache
}

func (x *PurgeMessage) Reset() {
	*x = PurgeMessage{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[2]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *PurgeMessage) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*PurgeMessage) ProtoMessage() {}

func (x *PurgeMessage) ProtoReflect() protoreflect.Message {
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

// Deprecated: Use PurgeMessage.ProtoReflect.Descriptor instead.
func (*PurgeMessage) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{2}
}

func (x *PurgeMessage) GetMsg() isPurgeMessage_Msg {
	if x != nil {
		return x.Msg
	}
	return nil
}

func (x *PurgeMessage) GetPurgeArtifact() *PurgeArtifact {
	if x != nil {
		if x, ok := x.Msg.(*PurgeMessage_PurgeArtifact); ok {
			return x.PurgeArtifact
		}
	}
	return nil
}

func (x *PurgeMessage) GetPurgeArtifactReport() *PurgeArtifactReport {
	if x != nil {
		if x, ok := x.Msg.(*PurgeMessage_PurgeArtifactReport); ok {
			return x.PurgeArtifactReport
		}
	}
	return nil
}

type isPurgeMessage_Msg interface {
	isPurgeMessage_Msg()
}

type PurgeMessage_PurgeArtifact struct {
	PurgeArtifact *PurgeArtifact `protobuf:"bytes,1,opt,name=purge_artifact,json=purgeArtifact,proto3,oneof"` // hub -> worker
}

type PurgeMessage_PurgeArtifactReport struct {
	PurgeArtifactReport *PurgeArtifactReport `protobuf:"bytes,2,opt,name=purge_artifact_report,json=purgeArtifactReport,proto3,oneof"` // worker -> hub
}

func (*PurgeMessage_PurgeArtifact) isPurgeMessage_Msg() {}

func (*PurgeMessage_PurgeArtifactReport) isPurgeMessage_Msg() {}

type Register struct {
	state               protoimpl.MessageState `protogen:"open.v1"`
	SessionId           string                 `protobuf:"bytes,1,opt,name=session_id,json=sessionId,proto3" json:"session_id,omitempty"`                                // §6 fence; control-parent-minted, never reused
	ExecutorIncarnation uint64                 `protobuf:"varint,2,opt,name=executor_incarnation,json=executorIncarnation,proto3" json:"executor_incarnation,omitempty"` // §6 fence
	WireMinor           uint32                 `protobuf:"varint,3,opt,name=wire_minor,json=wireMinor,proto3" json:"wire_minor,omitempty"`                               // highest minor THIS worker implements; absent = major.0
	WorkerId            string                 `protobuf:"bytes,4,opt,name=worker_id,json=workerId,proto3" json:"worker_id,omitempty"`                                   // CREDENTIAL subject; cross-checked vs auth JWT `sub`
	InstanceId          string                 `protobuf:"bytes,5,opt,name=instance_id,json=instanceId,proto3" json:"instance_id,omitempty"`                             // ONE provisioned instance lifetime; terminal-replay
	// authorization keys on this, never on worker_id
	ReleaseId   string `protobuf:"bytes,6,opt,name=release_id,json=releaseId,proto3" json:"release_id,omitempty"`       // cozy-runtime release identity
	ImageDigest string `protobuf:"bytes,7,opt,name=image_digest,json=imageDigest,proto3" json:"image_digest,omitempty"` // class (b): OCI image content digest, `sha256:<hex>`
	// (provenance, not identity)
	GitCommit string           `protobuf:"bytes,8,opt,name=git_commit,json=gitCommit,proto3" json:"git_commit,omitempty"` // provenance
	Resources *WorkerResources `protobuf:"bytes,9,opt,name=resources,proto3" json:"resources,omitempty"`                  // static per-boot facts; never re-sent mid-stream
	// The RECOVERED JOURNAL, reported first: accepted attempts and terminals pending ack that
	// survived restart. The hub closes/acknowledges these BEFORE minting any next ordinal -
	// a new session_id never manufactures absence (02 §6.2).
	RecoveredAttempts []*ActiveAttempt `protobuf:"bytes,10,rep,name=recovered_attempts,json=recoveredAttempts,proto3" json:"recovered_attempts,omitempty"`
	unknownFields     protoimpl.UnknownFields
	sizeCache         protoimpl.SizeCache
}

func (x *Register) Reset() {
	*x = Register{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[3]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *Register) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*Register) ProtoMessage() {}

func (x *Register) ProtoReflect() protoreflect.Message {
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

// Deprecated: Use Register.ProtoReflect.Descriptor instead.
func (*Register) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{3}
}

func (x *Register) GetSessionId() string {
	if x != nil {
		return x.SessionId
	}
	return ""
}

func (x *Register) GetExecutorIncarnation() uint64 {
	if x != nil {
		return x.ExecutorIncarnation
	}
	return 0
}

func (x *Register) GetWireMinor() uint32 {
	if x != nil {
		return x.WireMinor
	}
	return 0
}

func (x *Register) GetWorkerId() string {
	if x != nil {
		return x.WorkerId
	}
	return ""
}

func (x *Register) GetInstanceId() string {
	if x != nil {
		return x.InstanceId
	}
	return ""
}

func (x *Register) GetReleaseId() string {
	if x != nil {
		return x.ReleaseId
	}
	return ""
}

func (x *Register) GetImageDigest() string {
	if x != nil {
		return x.ImageDigest
	}
	return ""
}

func (x *Register) GetGitCommit() string {
	if x != nil {
		return x.GitCommit
	}
	return ""
}

func (x *Register) GetResources() *WorkerResources {
	if x != nil {
		return x.Resources
	}
	return nil
}

func (x *Register) GetRecoveredAttempts() []*ActiveAttempt {
	if x != nil {
		return x.RecoveredAttempts
	}
	return nil
}

// Authenticated boot-fatal delivery: the worker could open a stream but cannot become a
// registered worker. Closed vocabulary; hardware/image/provider facts attached so the
// coordinator can classify the pod SHAPE without logs.
type BootFailure struct {
	state         protoimpl.MessageState `protogen:"open.v1"`
	WorkerId      string                 `protobuf:"bytes,1,opt,name=worker_id,json=workerId,proto3" json:"worker_id,omitempty"` // cross-checked vs auth subject, same as Register
	InstanceId    string                 `protobuf:"bytes,2,opt,name=instance_id,json=instanceId,proto3" json:"instance_id,omitempty"`
	Reason        BootFailureReason      `protobuf:"varint,3,opt,name=reason,proto3,enum=cozy.worker.v1.BootFailureReason" json:"reason,omitempty"`
	Detail        string                 `protobuf:"bytes,4,opt,name=detail,proto3" json:"detail,omitempty"`       // bounded <= 1024 bytes, sanitized
	Resources     *WorkerResources       `protobuf:"bytes,5,opt,name=resources,proto3" json:"resources,omitempty"` // whatever was measurable; unreadable fields named there
	ImageDigest   string                 `protobuf:"bytes,6,opt,name=image_digest,json=imageDigest,proto3" json:"image_digest,omitempty"`
	ReleaseId     string                 `protobuf:"bytes,7,opt,name=release_id,json=releaseId,proto3" json:"release_id,omitempty"`
	unknownFields protoimpl.UnknownFields
	sizeCache     protoimpl.SizeCache
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

type RegisterAck struct {
	state               protoimpl.MessageState `protogen:"open.v1"`
	SessionId           string                 `protobuf:"bytes,1,opt,name=session_id,json=sessionId,proto3" json:"session_id,omitempty"`
	ExecutorIncarnation uint64                 `protobuf:"varint,2,opt,name=executor_incarnation,json=executorIncarnation,proto3" json:"executor_incarnation,omitempty"`
	WireMinor           uint32                 `protobuf:"varint,3,opt,name=wire_minor,json=wireMinor,proto3" json:"wire_minor,omitempty"`                      // highest minor the HUB implements for this major
	Accepted            bool                   `protobuf:"varint,4,opt,name=accepted,proto3" json:"accepted,omitempty"`                                         // false => `rejection` is set
	Rejection           RegisterRejection      `protobuf:"varint,5,opt,name=rejection,proto3,enum=cozy.worker.v1.RegisterRejection" json:"rejection,omitempty"` // set iff !accepted
	unknownFields       protoimpl.UnknownFields
	sizeCache           protoimpl.SizeCache
}

func (x *RegisterAck) Reset() {
	*x = RegisterAck{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[5]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *RegisterAck) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*RegisterAck) ProtoMessage() {}

func (x *RegisterAck) ProtoReflect() protoreflect.Message {
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

// Deprecated: Use RegisterAck.ProtoReflect.Descriptor instead.
func (*RegisterAck) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{5}
}

func (x *RegisterAck) GetSessionId() string {
	if x != nil {
		return x.SessionId
	}
	return ""
}

func (x *RegisterAck) GetExecutorIncarnation() uint64 {
	if x != nil {
		return x.ExecutorIncarnation
	}
	return 0
}

func (x *RegisterAck) GetWireMinor() uint32 {
	if x != nil {
		return x.WireMinor
	}
	return 0
}

func (x *RegisterAck) GetAccepted() bool {
	if x != nil {
		return x.Accepted
	}
	return false
}

func (x *RegisterAck) GetRejection() RegisterRejection {
	if x != nil {
		return x.Rejection
	}
	return RegisterRejection_REGISTER_REJECTION_UNSPECIFIED
}

// A Directive is a discriminated COMPLETE replacement. There is no merge and no delta: a
// field absent in a new Directive means "absent", never "unchanged". It is preposition for
// future dispatch only and never mutates an already-accepted ExecutionSpec (02 §3).
type Directive struct {
	state protoimpl.MessageState `protogen:"open.v1"`
	// by the ONEOF-LAST RULE. Same meaning, new numbers.
	SessionId           string `protobuf:"bytes,1,opt,name=session_id,json=sessionId,proto3" json:"session_id,omitempty"`
	ExecutorIncarnation uint64 `protobuf:"varint,2,opt,name=executor_incarnation,json=executorIncarnation,proto3" json:"executor_incarnation,omitempty"`
	Revision            uint64 `protobuf:"varint,3,opt,name=revision,proto3" json:"revision,omitempty"` // hub-owned, monotonic; older ignored, equal idempotent,
	// newer applies. Equal revision + different body is a
	// protocol error (02 §3) - "different body" is decided by
	// canonicalizing this message under the same writer and
	// comparing bytes. No digest field exists because the
	// comparison is one peer's, against what it applied; the
	// obligation is th-007's and cr-007's, not the schema's.
	Posture       Posture        `protobuf:"varint,4,opt,name=posture,proto3,enum=cozy.worker.v1.Posture" json:"posture,omitempty"`
	DrainPolicy   *DrainPolicy   `protobuf:"bytes,5,opt,name=drain_policy,json=drainPolicy,proto3" json:"drain_policy,omitempty"`
	ArtifactGrant *ArtifactGrant `protobuf:"bytes,6,opt,name=artifact_grant,json=artifactGrant,proto3" json:"artifact_grant,omitempty"` // standing fetch authority for this directive's artifacts
	Credential    *Credential    `protobuf:"bytes,7,opt,name=credential,proto3" json:"credential,omitempty"`                            // worker credential: issuer/key/epoch/expiry + secret token
	WireMinor     uint32         `protobuf:"varint,10,opt,name=wire_minor,json=wireMinor,proto3" json:"wire_minor,omitempty"`           // the minor the hub is speaking on this session; rides
	// the revision envelope, echoed by Report (03 §1.3)
	//
	// Types that are valid to be assigned to Mode:
	//
	//	*Directive_Serving
	//	*Directive_Job
	Mode          isDirective_Mode `protobuf_oneof:"mode"`
	unknownFields protoimpl.UnknownFields
	sizeCache     protoimpl.SizeCache
}

func (x *Directive) Reset() {
	*x = Directive{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[6]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *Directive) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*Directive) ProtoMessage() {}

func (x *Directive) ProtoReflect() protoreflect.Message {
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

// Deprecated: Use Directive.ProtoReflect.Descriptor instead.
func (*Directive) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{6}
}

func (x *Directive) GetSessionId() string {
	if x != nil {
		return x.SessionId
	}
	return ""
}

func (x *Directive) GetExecutorIncarnation() uint64 {
	if x != nil {
		return x.ExecutorIncarnation
	}
	return 0
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

func (x *Directive) GetDrainPolicy() *DrainPolicy {
	if x != nil {
		return x.DrainPolicy
	}
	return nil
}

func (x *Directive) GetArtifactGrant() *ArtifactGrant {
	if x != nil {
		return x.ArtifactGrant
	}
	return nil
}

func (x *Directive) GetCredential() *Credential {
	if x != nil {
		return x.Credential
	}
	return nil
}

func (x *Directive) GetWireMinor() uint32 {
	if x != nil {
		return x.WireMinor
	}
	return 0
}

func (x *Directive) GetMode() isDirective_Mode {
	if x != nil {
		return x.Mode
	}
	return nil
}

func (x *Directive) GetServing() *ServingDirective {
	if x != nil {
		if x, ok := x.Mode.(*Directive_Serving); ok {
			return x.Serving
		}
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

type isDirective_Mode interface {
	isDirective_Mode()
}

type Directive_Serving struct {
	Serving *ServingDirective `protobuf:"bytes,11,opt,name=serving,proto3,oneof"` // pinned binding constituents + expected plan set
}

type Directive_Job struct {
	Job *JobDirective `protobuf:"bytes,12,opt,name=job,proto3,oneof"` // build/JobDescriptor + caps + reclaim; NO serving binding
}

func (*Directive_Serving) isDirective_Mode() {}

func (*Directive_Job) isDirective_Mode() {}

type ServingDirective struct {
	state             protoimpl.MessageState `protogen:"open.v1"`
	EndpointReleaseId string                 `protobuf:"bytes,1,opt,name=endpoint_release_id,json=endpointReleaseId,proto3" json:"endpoint_release_id,omitempty"` // endpoint code identity. A pinned binding
	// has NO id of its own; its constituents
	// name it.
	EntrypointBindingPlanIds []string `protobuf:"bytes,2,rep,name=entrypoint_binding_plan_ids,json=entrypointBindingPlanIds,proto3" json:"entrypoint_binding_plan_ids,omitempty"` // expected set; sorted lexicographic ascending
	unknownFields            protoimpl.UnknownFields
	sizeCache                protoimpl.SizeCache
}

func (x *ServingDirective) Reset() {
	*x = ServingDirective{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[7]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *ServingDirective) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*ServingDirective) ProtoMessage() {}

func (x *ServingDirective) ProtoReflect() protoreflect.Message {
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

// Deprecated: Use ServingDirective.ProtoReflect.Descriptor instead.
func (*ServingDirective) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{7}
}

func (x *ServingDirective) GetEndpointReleaseId() string {
	if x != nil {
		return x.EndpointReleaseId
	}
	return ""
}

func (x *ServingDirective) GetEntrypointBindingPlanIds() []string {
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
	ReclaimOnTerminal   bool                   `protobuf:"varint,5,opt,name=reclaim_on_terminal,json=reclaimOnTerminal,proto3" json:"reclaim_on_terminal,omitempty"` // mandatory reclaim after terminal (bounded job)
	GpuCount            uint32                 `protobuf:"varint,6,opt,name=gpu_count,json=gpuCount,proto3" json:"gpu_count,omitempty"`                              // devices this ONE attempt owns. Multi-GPU admission
	unknownFields       protoimpl.UnknownFields
	sizeCache           protoimpl.SizeCache
}

func (x *JobDirective) Reset() {
	*x = JobDirective{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[8]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *JobDirective) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*JobDirective) ProtoMessage() {}

func (x *JobDirective) ProtoReflect() protoreflect.Message {
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

// Deprecated: Use JobDirective.ProtoReflect.Descriptor instead.
func (*JobDirective) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{8}
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

func (x *JobDirective) GetGpuCount() uint32 {
	if x != nil {
		return x.GpuCount
	}
	return 0
}

// ONE-DIGESTER RULE: every two-sided canonical document has exactly one canonicalizer -
// the hub. The worker echoes applied state as plain fields and NEVER digests posture/config
// itself; a worker-side digest would be a second canonicalization free to disagree.
type Report struct {
	state protoimpl.MessageState `protogen:"open.v1"`
	// 13/14 by the ONEOF-LAST RULE. Same meaning.
	SessionId           string      `protobuf:"bytes,1,opt,name=session_id,json=sessionId,proto3" json:"session_id,omitempty"`
	ExecutorIncarnation uint64      `protobuf:"varint,2,opt,name=executor_incarnation,json=executorIncarnation,proto3" json:"executor_incarnation,omitempty"`
	AppliedRevision     uint64      `protobuf:"varint,3,opt,name=applied_revision,json=appliedRevision,proto3" json:"applied_revision,omitempty"`                     // the Directive revision the worker has applied
	IntakeState         IntakeState `protobuf:"varint,4,opt,name=intake_state,json=intakeState,proto3,enum=cozy.worker.v1.IntakeState" json:"intake_state,omitempty"` // READY means first_request_servable, never "connected"
	CredentialEpoch     uint64      `protobuf:"varint,5,opt,name=credential_epoch,json=credentialEpoch,proto3" json:"credential_epoch,omitempty"`                     // the credential epoch the worker has APPLIED (rotation
	// echo). A divergence from the hub's minted epoch is
	// always observable; no rotation can be silently dropped.
	ReadinessEpoch uint64           `protobuf:"varint,6,opt,name=readiness_epoch,json=readinessEpoch,proto3" json:"readiness_epoch,omitempty"` // executor readiness epoch; >= executor_incarnation (§6.3)
	ActiveAttempts []*ActiveAttempt `protobuf:"bytes,9,rep,name=active_attempts,json=activeAttempts,proto3" json:"active_attempts,omitempty"`  // running + completed-with-unshipped-result
	Faults         []*Fault         `protobuf:"bytes,10,rep,name=faults,proto3" json:"faults,omitempty"`                                       // typed refusal/degrade facts
	// The NON-ATTEMPT activity lane: typed, sequence-bearing boot/worker steps riding the
	// DURABLE Report projection - never overloaded onto lossy AttemptProgress.
	Activity         []*ActivityEvent `protobuf:"bytes,11,rep,name=activity,proto3" json:"activity,omitempty"`
	AppliedWireMinor uint32           `protobuf:"varint,12,opt,name=applied_wire_minor,json=appliedWireMinor,proto3" json:"applied_wire_minor,omitempty"` // the minor the worker applied; a stale echo is a
	// VISIBLE skew fact, never a refusal (03 §4.3)
	//
	// Types that are valid to be assigned to Capacity:
	//
	//	*Report_ServingCapacity
	//	*Report_JobCapacity
	Capacity      isReport_Capacity `protobuf_oneof:"capacity"`
	unknownFields protoimpl.UnknownFields
	sizeCache     protoimpl.SizeCache
}

func (x *Report) Reset() {
	*x = Report{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[9]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *Report) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*Report) ProtoMessage() {}

func (x *Report) ProtoReflect() protoreflect.Message {
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

// Deprecated: Use Report.ProtoReflect.Descriptor instead.
func (*Report) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{9}
}

func (x *Report) GetSessionId() string {
	if x != nil {
		return x.SessionId
	}
	return ""
}

func (x *Report) GetExecutorIncarnation() uint64 {
	if x != nil {
		return x.ExecutorIncarnation
	}
	return 0
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

func (x *Report) GetCredentialEpoch() uint64 {
	if x != nil {
		return x.CredentialEpoch
	}
	return 0
}

func (x *Report) GetReadinessEpoch() uint64 {
	if x != nil {
		return x.ReadinessEpoch
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

func (x *Report) GetAppliedWireMinor() uint32 {
	if x != nil {
		return x.AppliedWireMinor
	}
	return 0
}

func (x *Report) GetCapacity() isReport_Capacity {
	if x != nil {
		return x.Capacity
	}
	return nil
}

func (x *Report) GetServingCapacity() *ServingCapacity {
	if x != nil {
		if x, ok := x.Capacity.(*Report_ServingCapacity); ok {
			return x.ServingCapacity
		}
	}
	return nil
}

func (x *Report) GetJobCapacity() *JobCapacity {
	if x != nil {
		if x, ok := x.Capacity.(*Report_JobCapacity); ok {
			return x.JobCapacity
		}
	}
	return nil
}

type isReport_Capacity interface {
	isReport_Capacity()
}

type Report_ServingCapacity struct {
	ServingCapacity *ServingCapacity `protobuf:"bytes,13,opt,name=serving_capacity,json=servingCapacity,proto3,oneof"`
}

type Report_JobCapacity struct {
	JobCapacity *JobCapacity `protobuf:"bytes,14,opt,name=job_capacity,json=jobCapacity,proto3,oneof"`
}

func (*Report_ServingCapacity) isReport_Capacity() {}

func (*Report_JobCapacity) isReport_Capacity() {}

type ActivityEvent struct {
	state         protoimpl.MessageState `protogen:"open.v1"`
	Seq           uint64                 `protobuf:"varint,1,opt,name=seq,proto3" json:"seq,omitempty"`  // monotone per session; supervisor-owned
	Kind          string                 `protobuf:"bytes,2,opt,name=kind,proto3" json:"kind,omitempty"` // closed step vocabulary (boot stages, engine_boot, recovery)
	Step          string                 `protobuf:"bytes,3,opt,name=step,proto3" json:"step,omitempty"` // bounded <= 256 bytes
	AtUnixMs      uint64                 `protobuf:"varint,4,opt,name=at_unix_ms,json=atUnixMs,proto3" json:"at_unix_ms,omitempty"`
	unknownFields protoimpl.UnknownFields
	sizeCache     protoimpl.SizeCache
}

func (x *ActivityEvent) Reset() {
	*x = ActivityEvent{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[10]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *ActivityEvent) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*ActivityEvent) ProtoMessage() {}

func (x *ActivityEvent) ProtoReflect() protoreflect.Message {
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

// Deprecated: Use ActivityEvent.ProtoReflect.Descriptor instead.
func (*ActivityEvent) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{10}
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
	state protoimpl.MessageState `protogen:"open.v1"`
	// travels as its CANONICAL BYTES at 8: the digest fence
	// requires byte agreement, and a structured twin beside
	// the bytes would be a second canonicalization.
	SessionId           string `protobuf:"bytes,1,opt,name=session_id,json=sessionId,proto3" json:"session_id,omitempty"`
	ExecutorIncarnation uint64 `protobuf:"varint,2,opt,name=executor_incarnation,json=executorIncarnation,proto3" json:"executor_incarnation,omitempty"`
	RequestId           string `protobuf:"bytes,3,opt,name=request_id,json=requestId,proto3" json:"request_id,omitempty"` // attempt fence (a)
	Attempt             uint64 `protobuf:"varint,4,opt,name=attempt,proto3" json:"attempt,omitempty"`                     // attempt fence (b); ordinal, starts at 1, bumped ONLY
	// by the hub, and only after the prior attempt's
	// journaled terminal. The worker echoes, never invents.
	ExecSpecDigest []byte `protobuf:"bytes,6,opt,name=exec_spec_digest,json=execSpecDigest,proto3" json:"exec_spec_digest,omitempty"` // attempt fence (c), digest class (a): sha256 over
	// EXACTLY the bytes in field 8. The receiver RECOMPUTES
	// over those resident bytes and refuses on mismatch - a
	// digest never bypasses the lower check.
	Grant             *DeliveryGrant `protobuf:"bytes,7,opt,name=grant,proto3" json:"grant,omitempty"`                                                    // expiring, refreshable; OUTSIDE the digest
	ExecSpecCanonical []byte         `protobuf:"bytes,8,opt,name=exec_spec_canonical,json=execSpecCanonical,proto3" json:"exec_spec_canonical,omitempty"` // the ExecutionSpec DOCUMENT (§5), canonical JSON bytes.
	unknownFields     protoimpl.UnknownFields
	sizeCache         protoimpl.SizeCache
}

func (x *StartAttempt) Reset() {
	*x = StartAttempt{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[11]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *StartAttempt) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*StartAttempt) ProtoMessage() {}

func (x *StartAttempt) ProtoReflect() protoreflect.Message {
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

// Deprecated: Use StartAttempt.ProtoReflect.Descriptor instead.
func (*StartAttempt) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{11}
}

func (x *StartAttempt) GetSessionId() string {
	if x != nil {
		return x.SessionId
	}
	return ""
}

func (x *StartAttempt) GetExecutorIncarnation() uint64 {
	if x != nil {
		return x.ExecutorIncarnation
	}
	return 0
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

func (x *StartAttempt) GetExecSpecDigest() []byte {
	if x != nil {
		return x.ExecSpecDigest
	}
	return nil
}

func (x *StartAttempt) GetGrant() *DeliveryGrant {
	if x != nil {
		return x.Grant
	}
	return nil
}

func (x *StartAttempt) GetExecSpecCanonical() []byte {
	if x != nil {
		return x.ExecSpecCanonical
	}
	return nil
}

type AttemptAccepted struct {
	state               protoimpl.MessageState `protogen:"open.v1"`
	SessionId           string                 `protobuf:"bytes,1,opt,name=session_id,json=sessionId,proto3" json:"session_id,omitempty"`
	ExecutorIncarnation uint64                 `protobuf:"varint,2,opt,name=executor_incarnation,json=executorIncarnation,proto3" json:"executor_incarnation,omitempty"`
	RequestId           string                 `protobuf:"bytes,3,opt,name=request_id,json=requestId,proto3" json:"request_id,omitempty"`
	Attempt             uint64                 `protobuf:"varint,4,opt,name=attempt,proto3" json:"attempt,omitempty"`
	ExecSpecDigest      []byte                 `protobuf:"bytes,5,opt,name=exec_spec_digest,json=execSpecDigest,proto3" json:"exec_spec_digest,omitempty"` // the digest the worker accepted (class (a), echo)
	PlanDigest          []byte                 `protobuf:"bytes,6,opt,name=plan_digest,json=planDigest,proto3" json:"plan_digest,omitempty"`               // class (a): the AttemptPlan DOCUMENT's identity, fixed
	// at acceptance. That document is cozy-runtime's (cr-007)
	// under tfs-013's writer; identity only here.
	ModelConstructionDigest []byte `protobuf:"bytes,7,opt,name=model_construction_digest,json=modelConstructionDigest,proto3" json:"model_construction_digest,omitempty"` // class (a): the ModelConstructionContract DOCUMENT's
	// identity (cr-004). Identity only here.
	// The CLOSED observable projection of the chosen plan. A hash alone is not observable
	// evidence: the coordinator sees WHAT was chosen, not only that something was.
	Plan          *AttemptPlanSummary `protobuf:"bytes,8,opt,name=plan,proto3" json:"plan,omitempty"`
	unknownFields protoimpl.UnknownFields
	sizeCache     protoimpl.SizeCache
}

func (x *AttemptAccepted) Reset() {
	*x = AttemptAccepted{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[12]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *AttemptAccepted) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*AttemptAccepted) ProtoMessage() {}

func (x *AttemptAccepted) ProtoReflect() protoreflect.Message {
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

// Deprecated: Use AttemptAccepted.ProtoReflect.Descriptor instead.
func (*AttemptAccepted) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{12}
}

func (x *AttemptAccepted) GetSessionId() string {
	if x != nil {
		return x.SessionId
	}
	return ""
}

func (x *AttemptAccepted) GetExecutorIncarnation() uint64 {
	if x != nil {
		return x.ExecutorIncarnation
	}
	return 0
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

func (x *AttemptAccepted) GetExecSpecDigest() []byte {
	if x != nil {
		return x.ExecSpecDigest
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

type AttemptPlanSummary struct {
	state             protoimpl.MessageState `protogen:"open.v1"`
	Delivery          string                 `protobuf:"bytes,1,opt,name=delivery,proto3" json:"delivery,omitempty"`               // native | float
	Materialization   string                 `protobuf:"bytes,2,opt,name=materialization,proto3" json:"materialization,omitempty"` // aot_decode | staged_decode | jit_decode (float only)
	ComputeDtype      string                 `protobuf:"bytes,3,opt,name=compute_dtype,json=computeDtype,proto3" json:"compute_dtype,omitempty"`
	Placement         string                 `protobuf:"bytes,4,opt,name=placement,proto3" json:"placement,omitempty"` // all_resident | component_staged | host_offload | streamed
	ReservedVramBytes uint64                 `protobuf:"varint,5,opt,name=reserved_vram_bytes,json=reservedVramBytes,proto3" json:"reserved_vram_bytes,omitempty"`
	ReservedHostBytes uint64                 `protobuf:"varint,6,opt,name=reserved_host_bytes,json=reservedHostBytes,proto3" json:"reserved_host_bytes,omitempty"`
	QuantifiedChoice  string                 `protobuf:"bytes,7,opt,name=quantified_choice,json=quantifiedChoice,proto3" json:"quantified_choice,omitempty"` // bounded <= 512 bytes human-diagnostic prose; NOT
	unknownFields     protoimpl.UnknownFields
	sizeCache         protoimpl.SizeCache
}

func (x *AttemptPlanSummary) Reset() {
	*x = AttemptPlanSummary{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[13]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *AttemptPlanSummary) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*AttemptPlanSummary) ProtoMessage() {}

func (x *AttemptPlanSummary) ProtoReflect() protoreflect.Message {
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

// Deprecated: Use AttemptPlanSummary.ProtoReflect.Descriptor instead.
func (*AttemptPlanSummary) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{13}
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

func (x *AttemptPlanSummary) GetReservedVramBytes() uint64 {
	if x != nil {
		return x.ReservedVramBytes
	}
	return 0
}

func (x *AttemptPlanSummary) GetReservedHostBytes() uint64 {
	if x != nil {
		return x.ReservedHostBytes
	}
	return 0
}

func (x *AttemptPlanSummary) GetQuantifiedChoice() string {
	if x != nil {
		return x.QuantifiedChoice
	}
	return ""
}

type CancelAttempt struct {
	state               protoimpl.MessageState `protogen:"open.v1"`
	SessionId           string                 `protobuf:"bytes,1,opt,name=session_id,json=sessionId,proto3" json:"session_id,omitempty"`
	ExecutorIncarnation uint64                 `protobuf:"varint,2,opt,name=executor_incarnation,json=executorIncarnation,proto3" json:"executor_incarnation,omitempty"`
	RequestId           string                 `protobuf:"bytes,3,opt,name=request_id,json=requestId,proto3" json:"request_id,omitempty"`
	Attempt             uint64                 `protobuf:"varint,4,opt,name=attempt,proto3" json:"attempt,omitempty"` // the attempt the hub believes current
	Reason              CancelReason           `protobuf:"varint,5,opt,name=reason,proto3,enum=cozy.worker.v1.CancelReason" json:"reason,omitempty"`
	GraceMs             uint64                 `protobuf:"varint,6,opt,name=grace_ms,json=graceMs,proto3" json:"grace_ms,omitempty"` // operator override door; 0 = the worker's
	// runtime-constant default grace. Never an author knob.
	ExecSpecDigest []byte `protobuf:"bytes,7,opt,name=exec_spec_digest,json=execSpecDigest,proto3" json:"exec_spec_digest,omitempty"` // full-triple fence, class (a): compared against the
	unknownFields  protoimpl.UnknownFields
	sizeCache      protoimpl.SizeCache
}

func (x *CancelAttempt) Reset() {
	*x = CancelAttempt{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[14]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *CancelAttempt) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*CancelAttempt) ProtoMessage() {}

func (x *CancelAttempt) ProtoReflect() protoreflect.Message {
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

// Deprecated: Use CancelAttempt.ProtoReflect.Descriptor instead.
func (*CancelAttempt) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{14}
}

func (x *CancelAttempt) GetSessionId() string {
	if x != nil {
		return x.SessionId
	}
	return ""
}

func (x *CancelAttempt) GetExecutorIncarnation() uint64 {
	if x != nil {
		return x.ExecutorIncarnation
	}
	return 0
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

func (x *CancelAttempt) GetExecSpecDigest() []byte {
	if x != nil {
		return x.ExecSpecDigest
	}
	return nil
}

// The envelope around a journaled TerminalBody document. The body travels as canonical bytes
// because its fence needs byte agreement in two places at once: the hub must recompute the
// digest to know the id binds this content, and a RECONNECT re-wraps the SAME body in a fresh
// session envelope - which is only meaningful if "the same body" is a byte fact. Fields 1/2
// are outside the document for exactly that reason; 3/4/5 are routing copies the receiver
// MUST check against the document and refuse on divergence.
type AttemptTerminal struct {
	state protoimpl.MessageState `protogen:"open.v1"`
	// safe_message, cause, result: they ARE the terminal
	// body and now live in the TerminalBody document at 15.
	SessionId           string `protobuf:"bytes,1,opt,name=session_id,json=sessionId,proto3" json:"session_id,omitempty"`
	ExecutorIncarnation uint64 `protobuf:"varint,2,opt,name=executor_incarnation,json=executorIncarnation,proto3" json:"executor_incarnation,omitempty"`
	RequestId           string `protobuf:"bytes,3,opt,name=request_id,json=requestId,proto3" json:"request_id,omitempty"`                  // routing copy of the document's field; must agree
	Attempt             uint64 `protobuf:"varint,4,opt,name=attempt,proto3" json:"attempt,omitempty"`                                      // routing copy; must agree
	ExecSpecDigest      []byte `protobuf:"bytes,5,opt,name=exec_spec_digest,json=execSpecDigest,proto3" json:"exec_spec_digest,omitempty"` // routing copy; must agree
	TerminalId          string `protobuf:"bytes,12,opt,name=terminal_id,json=terminalId,proto3" json:"terminal_id,omitempty"`              // WORKER-minted observation id; fixed at journal time,
	// stable across every replay. OUTSIDE the body, so the
	// digest fences content and the id names the observation.
	TerminalDigest []byte `protobuf:"bytes,13,opt,name=terminal_digest,json=terminalDigest,proto3" json:"terminal_digest,omitempty"` // digest class (a): sha256 over EXACTLY the bytes in 15,
	// fixed at journal time and byte-stable across replays.
	TerminalCanonical []byte `protobuf:"bytes,15,opt,name=terminal_canonical,json=terminalCanonical,proto3" json:"terminal_canonical,omitempty"` // the TerminalBody DOCUMENT, canonical JSON bytes.
	unknownFields     protoimpl.UnknownFields
	sizeCache         protoimpl.SizeCache
}

func (x *AttemptTerminal) Reset() {
	*x = AttemptTerminal{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[15]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *AttemptTerminal) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*AttemptTerminal) ProtoMessage() {}

func (x *AttemptTerminal) ProtoReflect() protoreflect.Message {
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

// Deprecated: Use AttemptTerminal.ProtoReflect.Descriptor instead.
func (*AttemptTerminal) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{15}
}

func (x *AttemptTerminal) GetSessionId() string {
	if x != nil {
		return x.SessionId
	}
	return ""
}

func (x *AttemptTerminal) GetExecutorIncarnation() uint64 {
	if x != nil {
		return x.ExecutorIncarnation
	}
	return 0
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

func (x *AttemptTerminal) GetExecSpecDigest() []byte {
	if x != nil {
		return x.ExecSpecDigest
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

// DOCUMENT SHAPE (not a wire message): the journaled terminal body, the subject of
// `terminal_digest`. Canonical form `cozy.worker.v1.TerminalBody/1`.
type TerminalBody struct {
	state          protoimpl.MessageState `protogen:"open.v1"`
	RequestId      string                 `protobuf:"bytes,1,opt,name=request_id,json=requestId,proto3" json:"request_id,omitempty"`
	Attempt        uint64                 `protobuf:"varint,2,opt,name=attempt,proto3" json:"attempt,omitempty"`
	ExecSpecDigest []byte                 `protobuf:"bytes,3,opt,name=exec_spec_digest,json=execSpecDigest,proto3" json:"exec_spec_digest,omitempty"` // which execution this terminal closes
	Status         TerminalStatus         `protobuf:"varint,4,opt,name=status,proto3,enum=cozy.worker.v1.TerminalStatus" json:"status,omitempty"`     // closed NEUTRAL vocabulary. Retryability is a
	// coordinator PROJECTION over (status, cause, origin)
	// and is unrepresentable as a worker-authored wire fact.
	OutputManifest *OutputManifest  `protobuf:"bytes,5,opt,name=output_manifest,json=outputManifest,proto3" json:"output_manifest,omitempty"` // journaled-before-send; replayed until TerminalAck
	Metrics        *AttemptMetrics  `protobuf:"bytes,6,opt,name=metrics,proto3" json:"metrics,omitempty"`                                     // bounded worker-observed measurements
	TriageBundle   *TriageBundleRef `protobuf:"bytes,7,opt,name=triage_bundle,json=triageBundle,proto3" json:"triage_bundle,omitempty"`       // optional; OBSERVATION ONLY - it can never replace or
	// override any terminal field
	SafeMessage string `protobuf:"bytes,8,opt,name=safe_message,json=safeMessage,proto3" json:"safe_message,omitempty"` // bounded <= 4096 bytes, sanitized: no secrets, paths,
	// or signed URLs
	Cause         *TerminalCause  `protobuf:"bytes,9,opt,name=cause,proto3" json:"cause,omitempty"`    // REQUIRED on every sent terminal
	Result        *ResultEnvelope `protobuf:"bytes,10,opt,name=result,proto3" json:"result,omitempty"` // the TYPED FUNCTION RESULT on success terminals. An
	unknownFields protoimpl.UnknownFields
	sizeCache     protoimpl.SizeCache
}

func (x *TerminalBody) Reset() {
	*x = TerminalBody{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[16]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *TerminalBody) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*TerminalBody) ProtoMessage() {}

func (x *TerminalBody) ProtoReflect() protoreflect.Message {
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

// Deprecated: Use TerminalBody.ProtoReflect.Descriptor instead.
func (*TerminalBody) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{16}
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

func (x *TerminalBody) GetExecSpecDigest() []byte {
	if x != nil {
		return x.ExecSpecDigest
	}
	return nil
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
	state              protoimpl.MessageState `protogen:"open.v1"`
	ResultSchemaDigest []byte                 `protobuf:"bytes,1,opt,name=result_schema_digest,json=resultSchemaDigest,proto3" json:"result_schema_digest,omitempty"` // the exact annotated result-surface digest
	InlineResult       []byte                 `protobuf:"bytes,2,opt,name=inline_result,json=inlineResult,proto3" json:"inline_result,omitempty"`                     // canonical typed result bytes, <= 4 MiB; the inline
	// door. XOR with field 3.
	ResultBlob    *OutputEntry      `protobuf:"bytes,3,opt,name=result_blob,json=resultBlob,proto3" json:"result_blob,omitempty"`          // by immutable blob receipt when large. XOR with field 2.
	Adjustments   []*AdjustmentRow  `protobuf:"bytes,4,rep,name=adjustments,proto3" json:"adjustments,omitempty"`                          // caller-visible adjusted/clamp/warn rows
	CheckpointRef string            `protobuf:"bytes,5,opt,name=checkpoint_ref,json=checkpointRef,proto3" json:"checkpoint_ref,omitempty"` // reproduction provenance
	Adapters      []*AppliedAdapter `protobuf:"bytes,6,rep,name=adapters,proto3" json:"adapters,omitempty"`
	unknownFields protoimpl.UnknownFields
	sizeCache     protoimpl.SizeCache
}

func (x *ResultEnvelope) Reset() {
	*x = ResultEnvelope{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[17]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *ResultEnvelope) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*ResultEnvelope) ProtoMessage() {}

func (x *ResultEnvelope) ProtoReflect() protoreflect.Message {
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

// Deprecated: Use ResultEnvelope.ProtoReflect.Descriptor instead.
func (*ResultEnvelope) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{17}
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

func (x *ResultEnvelope) GetAdapters() []*AppliedAdapter {
	if x != nil {
		return x.Adapters
	}
	return nil
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
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[18]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *AdjustmentRow) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*AdjustmentRow) ProtoMessage() {}

func (x *AdjustmentRow) ProtoReflect() protoreflect.Message {
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

// Deprecated: Use AdjustmentRow.ProtoReflect.Descriptor instead.
func (*AdjustmentRow) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{18}
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

type AppliedAdapter struct {
	state protoimpl.MessageState `protogen:"open.v1"`
	// document, and tfs-013's writer is INTEGER-ONLY, so a
	// float has no canonical spelling in a document at all.
	// It is also redundant: `resolved_adapter_plan_id` already
	// binds the exact canonical scale. The fixed-point
	// spelling itself freezes at tfs-012, not here.
	Ref string `protobuf:"bytes,1,opt,name=ref,proto3" json:"ref,omitempty"` // the `resolved_adapter_plan_id` that was applied -
	// adapter slot + contract + exact canonical scale +
	// composition contract, digest class (a), owned upstream
	Kind          string `protobuf:"bytes,3,opt,name=kind,proto3" json:"kind,omitempty"` // typed stamp fact (turbo, style, ...)
	unknownFields protoimpl.UnknownFields
	sizeCache     protoimpl.SizeCache
}

func (x *AppliedAdapter) Reset() {
	*x = AppliedAdapter{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[19]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *AppliedAdapter) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*AppliedAdapter) ProtoMessage() {}

func (x *AppliedAdapter) ProtoReflect() protoreflect.Message {
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

// Deprecated: Use AppliedAdapter.ProtoReflect.Descriptor instead.
func (*AppliedAdapter) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{19}
}

func (x *AppliedAdapter) GetRef() string {
	if x != nil {
		return x.Ref
	}
	return ""
}

func (x *AppliedAdapter) GetKind() string {
	if x != nil {
		return x.Kind
	}
	return ""
}

type TerminalCause struct {
	state         protoimpl.MessageState `protogen:"open.v1"`
	Code          CauseCode              `protobuf:"varint,1,opt,name=code,proto3,enum=cozy.worker.v1.CauseCode" json:"code,omitempty"`       // closed per-status cause vocabulary
	Origin        CauseOrigin            `protobuf:"varint,2,opt,name=origin,proto3,enum=cozy.worker.v1.CauseOrigin" json:"origin,omitempty"` // which layer produced the fact
	Detail        string                 `protobuf:"bytes,3,opt,name=detail,proto3" json:"detail,omitempty"`                                  // bounded <= 1024 bytes, sanitized
	Shortfall     *ResourceShortfall     `protobuf:"bytes,4,opt,name=shortfall,proto3" json:"shortfall,omitempty"`                            // set on capacity-class causes: the STRUCTURED
	unknownFields protoimpl.UnknownFields
	sizeCache     protoimpl.SizeCache
}

func (x *TerminalCause) Reset() {
	*x = TerminalCause{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[20]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *TerminalCause) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*TerminalCause) ProtoMessage() {}

func (x *TerminalCause) ProtoReflect() protoreflect.Message {
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

// Deprecated: Use TerminalCause.ProtoReflect.Descriptor instead.
func (*TerminalCause) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{20}
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
	Resource       string                 `protobuf:"bytes,1,opt,name=resource,proto3" json:"resource,omitempty"` // vram | host_ram | pinned | disk | ...
	Scope          string                 `protobuf:"bytes,2,opt,name=scope,proto3" json:"scope,omitempty"`       // step/component the shortfall bit (fill, denoise, encode)
	NeededBytes    uint64                 `protobuf:"varint,3,opt,name=needed_bytes,json=neededBytes,proto3" json:"needed_bytes,omitempty"`
	AvailableBytes uint64                 `protobuf:"varint,4,opt,name=available_bytes,json=availableBytes,proto3" json:"available_bytes,omitempty"`
	RequestShape   string                 `protobuf:"bytes,5,opt,name=request_shape,json=requestShape,proto3" json:"request_shape,omitempty"`    // bounded normalized-feature rendering
	EvidenceClass  string                 `protobuf:"bytes,6,opt,name=evidence_class,json=evidenceClass,proto3" json:"evidence_class,omitempty"` // measured | fitted | prior
	unknownFields  protoimpl.UnknownFields
	sizeCache      protoimpl.SizeCache
}

func (x *ResourceShortfall) Reset() {
	*x = ResourceShortfall{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[21]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *ResourceShortfall) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*ResourceShortfall) ProtoMessage() {}

func (x *ResourceShortfall) ProtoReflect() protoreflect.Message {
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

// Deprecated: Use ResourceShortfall.ProtoReflect.Descriptor instead.
func (*ResourceShortfall) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{21}
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
	state               protoimpl.MessageState `protogen:"open.v1"`
	SessionId           string                 `protobuf:"bytes,1,opt,name=session_id,json=sessionId,proto3" json:"session_id,omitempty"`
	ExecutorIncarnation uint64                 `protobuf:"varint,2,opt,name=executor_incarnation,json=executorIncarnation,proto3" json:"executor_incarnation,omitempty"`
	RequestId           string                 `protobuf:"bytes,3,opt,name=request_id,json=requestId,proto3" json:"request_id,omitempty"`
	Attempt             uint64                 `protobuf:"varint,4,opt,name=attempt,proto3" json:"attempt,omitempty"`
	ExecSpecDigest      []byte                 `protobuf:"bytes,5,opt,name=exec_spec_digest,json=execSpecDigest,proto3" json:"exec_spec_digest,omitempty"` // which execution's terminal is acknowledged (echo)
	TerminalId          string                 `protobuf:"bytes,6,opt,name=terminal_id,json=terminalId,proto3" json:"terminal_id,omitempty"`               // echo: the exact journaled observation acknowledged
	TerminalDigest      []byte                 `protobuf:"bytes,7,opt,name=terminal_digest,json=terminalDigest,proto3" json:"terminal_digest,omitempty"`   // echo of the TerminalBody identity: an ack whose digest
	unknownFields       protoimpl.UnknownFields
	sizeCache           protoimpl.SizeCache
}

func (x *TerminalAck) Reset() {
	*x = TerminalAck{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[22]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *TerminalAck) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*TerminalAck) ProtoMessage() {}

func (x *TerminalAck) ProtoReflect() protoreflect.Message {
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

// Deprecated: Use TerminalAck.ProtoReflect.Descriptor instead.
func (*TerminalAck) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{22}
}

func (x *TerminalAck) GetSessionId() string {
	if x != nil {
		return x.SessionId
	}
	return ""
}

func (x *TerminalAck) GetExecutorIncarnation() uint64 {
	if x != nil {
		return x.ExecutorIncarnation
	}
	return 0
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

func (x *TerminalAck) GetExecSpecDigest() []byte {
	if x != nil {
		return x.ExecSpecDigest
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
	state               protoimpl.MessageState `protogen:"open.v1"`
	SessionId           string                 `protobuf:"bytes,1,opt,name=session_id,json=sessionId,proto3" json:"session_id,omitempty"`
	ExecutorIncarnation uint64                 `protobuf:"varint,2,opt,name=executor_incarnation,json=executorIncarnation,proto3" json:"executor_incarnation,omitempty"`
	RequestId           string                 `protobuf:"bytes,3,opt,name=request_id,json=requestId,proto3" json:"request_id,omitempty"`
	Attempt             uint64                 `protobuf:"varint,4,opt,name=attempt,proto3" json:"attempt,omitempty"`
	OperationKey        string                 `protobuf:"bytes,5,opt,name=operation_key,json=operationKey,proto3" json:"operation_key,omitempty"` // worker-minted, attempt-scoped idempotency key for THIS
	// write operation; stable across replays
	LogicalKey    string `protobuf:"bytes,6,opt,name=logical_key,json=logicalKey,proto3" json:"logical_key,omitempty"`          // the author-stable checkpoint slot (e.g. "step-4000")
	ContentDigest []byte `protobuf:"bytes,7,opt,name=content_digest,json=contentDigest,proto3" json:"content_digest,omitempty"` // digest class (b): the CONTENT digest of the checkpoint
	// artifact's bytes in the store (a tensorfs snapshot,
	// tfs-013: checkpoint_id = snapshot_id). Not a JSON
	// document; nothing here canonicalizes it.
	Seq uint64 `protobuf:"varint,8,opt,name=seq,proto3" json:"seq,omitempty"` // monotone per (request_id, attempt); ORDERING only,
	// never identity
	Artifact      *OutputEntry `protobuf:"bytes,9,opt,name=artifact,proto3" json:"artifact,omitempty"` // the immutable blob receipt of what was written
	unknownFields protoimpl.UnknownFields
	sizeCache     protoimpl.SizeCache
}

func (x *JobCheckpointRequest) Reset() {
	*x = JobCheckpointRequest{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[23]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *JobCheckpointRequest) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*JobCheckpointRequest) ProtoMessage() {}

func (x *JobCheckpointRequest) ProtoReflect() protoreflect.Message {
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

// Deprecated: Use JobCheckpointRequest.ProtoReflect.Descriptor instead.
func (*JobCheckpointRequest) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{23}
}

func (x *JobCheckpointRequest) GetSessionId() string {
	if x != nil {
		return x.SessionId
	}
	return ""
}

func (x *JobCheckpointRequest) GetExecutorIncarnation() uint64 {
	if x != nil {
		return x.ExecutorIncarnation
	}
	return 0
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
	state               protoimpl.MessageState `protogen:"open.v1"`
	SessionId           string                 `protobuf:"bytes,1,opt,name=session_id,json=sessionId,proto3" json:"session_id,omitempty"`
	ExecutorIncarnation uint64                 `protobuf:"varint,2,opt,name=executor_incarnation,json=executorIncarnation,proto3" json:"executor_incarnation,omitempty"`
	RequestId           string                 `protobuf:"bytes,3,opt,name=request_id,json=requestId,proto3" json:"request_id,omitempty"`
	Attempt             uint64                 `protobuf:"varint,4,opt,name=attempt,proto3" json:"attempt,omitempty"`
	OperationKey        string                 `protobuf:"bytes,5,opt,name=operation_key,json=operationKey,proto3" json:"operation_key,omitempty"`    // echo
	LogicalKey          string                 `protobuf:"bytes,6,opt,name=logical_key,json=logicalKey,proto3" json:"logical_key,omitempty"`          // echo
	ContentDigest       []byte                 `protobuf:"bytes,7,opt,name=content_digest,json=contentDigest,proto3" json:"content_digest,omitempty"` // echo of the class (b) content digest; a mismatch
	// against the recorded identity is a conflict, never a
	// silent overwrite
	ReceiptId     string            `protobuf:"bytes,8,opt,name=receipt_id,json=receiptId,proto3" json:"receipt_id,omitempty"` // hub-minted durable record id; stable across replays
	Outcome       CheckpointOutcome `protobuf:"varint,9,opt,name=outcome,proto3,enum=cozy.worker.v1.CheckpointOutcome" json:"outcome,omitempty"`
	Fault         *CheckpointFault  `protobuf:"bytes,10,opt,name=fault,proto3" json:"fault,omitempty"` // set iff outcome is CONFLICT or REFUSED
	unknownFields protoimpl.UnknownFields
	sizeCache     protoimpl.SizeCache
}

func (x *JobCheckpointReceipt) Reset() {
	*x = JobCheckpointReceipt{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[24]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *JobCheckpointReceipt) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*JobCheckpointReceipt) ProtoMessage() {}

func (x *JobCheckpointReceipt) ProtoReflect() protoreflect.Message {
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

// Deprecated: Use JobCheckpointReceipt.ProtoReflect.Descriptor instead.
func (*JobCheckpointReceipt) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{24}
}

func (x *JobCheckpointReceipt) GetSessionId() string {
	if x != nil {
		return x.SessionId
	}
	return ""
}

func (x *JobCheckpointReceipt) GetExecutorIncarnation() uint64 {
	if x != nil {
		return x.ExecutorIncarnation
	}
	return 0
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
	Detail                string                 `protobuf:"bytes,2,opt,name=detail,proto3" json:"detail,omitempty"`                                                              // bounded <= 1024 bytes
	RecordedContentDigest []byte                 `protobuf:"bytes,3,opt,name=recorded_content_digest,json=recordedContentDigest,proto3" json:"recorded_content_digest,omitempty"` // on CONFLICT: the digest already recorded for this
	unknownFields         protoimpl.UnknownFields
	sizeCache             protoimpl.SizeCache
}

func (x *CheckpointFault) Reset() {
	*x = CheckpointFault{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[25]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *CheckpointFault) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*CheckpointFault) ProtoMessage() {}

func (x *CheckpointFault) ProtoReflect() protoreflect.Message {
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

// Deprecated: Use CheckpointFault.ProtoReflect.Descriptor instead.
func (*CheckpointFault) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{25}
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
	state               protoimpl.MessageState `protogen:"open.v1"`
	SessionId           string                 `protobuf:"bytes,1,opt,name=session_id,json=sessionId,proto3" json:"session_id,omitempty"`
	ExecutorIncarnation uint64                 `protobuf:"varint,2,opt,name=executor_incarnation,json=executorIncarnation,proto3" json:"executor_incarnation,omitempty"`
	RequestId           string                 `protobuf:"bytes,3,opt,name=request_id,json=requestId,proto3" json:"request_id,omitempty"`
	Attempt             uint64                 `protobuf:"varint,4,opt,name=attempt,proto3" json:"attempt,omitempty"`
	OperationKey        string                 `protobuf:"bytes,5,opt,name=operation_key,json=operationKey,proto3" json:"operation_key,omitempty"`
	ReceiptId           string                 `protobuf:"bytes,6,opt,name=receipt_id,json=receiptId,proto3" json:"receipt_id,omitempty"`             // echo: an ack whose receipt_id or content_digest
	ContentDigest       []byte                 `protobuf:"bytes,7,opt,name=content_digest,json=contentDigest,proto3" json:"content_digest,omitempty"` // mismatches the journaled receipt is NOT an ack
	unknownFields       protoimpl.UnknownFields
	sizeCache           protoimpl.SizeCache
}

func (x *JobCheckpointAck) Reset() {
	*x = JobCheckpointAck{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[26]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *JobCheckpointAck) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*JobCheckpointAck) ProtoMessage() {}

func (x *JobCheckpointAck) ProtoReflect() protoreflect.Message {
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

// Deprecated: Use JobCheckpointAck.ProtoReflect.Descriptor instead.
func (*JobCheckpointAck) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{26}
}

func (x *JobCheckpointAck) GetSessionId() string {
	if x != nil {
		return x.SessionId
	}
	return ""
}

func (x *JobCheckpointAck) GetExecutorIncarnation() uint64 {
	if x != nil {
		return x.ExecutorIncarnation
	}
	return 0
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

type AttemptProgress struct {
	state               protoimpl.MessageState `protogen:"open.v1"`
	SessionId           string                 `protobuf:"bytes,1,opt,name=session_id,json=sessionId,proto3" json:"session_id,omitempty"`
	ExecutorIncarnation uint64                 `protobuf:"varint,2,opt,name=executor_incarnation,json=executorIncarnation,proto3" json:"executor_incarnation,omitempty"`
	RequestId           string                 `protobuf:"bytes,3,opt,name=request_id,json=requestId,proto3" json:"request_id,omitempty"`
	Attempt             uint64                 `protobuf:"varint,4,opt,name=attempt,proto3" json:"attempt,omitempty"`
	Seq                 uint64                 `protobuf:"varint,5,opt,name=seq,proto3" json:"seq,omitempty"` // starts at 1, strictly increasing per (request_id,
	// attempt). Gaps are VISIBLE and are the only loss this
	// lane may incur.
	ContentType string `protobuf:"bytes,6,opt,name=content_type,json=contentType,proto3" json:"content_type,omitempty"` // "text/plain" | "application/json" | "audio/*" | ... ;
	// "application/x-cozy-event+json" carries the ctx-event
	// envelope {type in progress|log|warning|checkpoint|
	// metric, payload}
	Data          []byte `protobuf:"bytes,7,opt,name=data,proto3" json:"data,omitempty"` // bounded <= 65536 bytes per message; NEVER terminal.
	unknownFields protoimpl.UnknownFields
	sizeCache     protoimpl.SizeCache
}

func (x *AttemptProgress) Reset() {
	*x = AttemptProgress{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[27]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *AttemptProgress) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*AttemptProgress) ProtoMessage() {}

func (x *AttemptProgress) ProtoReflect() protoreflect.Message {
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

// Deprecated: Use AttemptProgress.ProtoReflect.Descriptor instead.
func (*AttemptProgress) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{27}
}

func (x *AttemptProgress) GetSessionId() string {
	if x != nil {
		return x.SessionId
	}
	return ""
}

func (x *AttemptProgress) GetExecutorIncarnation() uint64 {
	if x != nil {
		return x.ExecutorIncarnation
	}
	return 0
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

type PurgeArtifact struct {
	state protoimpl.MessageState `protogen:"open.v1"`
	// target_instance_id, target_generation, deadline_unix:
	// they ARE the purge body and now live in the PurgeBody
	// document at 11. "Same id + changed bytes refuses" is a
	// byte fence, so the worker must recompute it.
	SessionId           string `protobuf:"bytes,1,opt,name=session_id,json=sessionId,proto3" json:"session_id,omitempty"`
	ExecutorIncarnation uint64 `protobuf:"varint,2,opt,name=executor_incarnation,json=executorIncarnation,proto3" json:"executor_incarnation,omitempty"`
	PurgeId             string `protobuf:"bytes,3,opt,name=purge_id,json=purgeId,proto3" json:"purge_id,omitempty"`                                     // hub-minted command id (replay/dedup key)
	PurgeBodyDigest     []byte `protobuf:"bytes,4,opt,name=purge_body_digest,json=purgeBodyDigest,proto3" json:"purge_body_digest,omitempty"`           // digest class (a): sha256 over EXACTLY the bytes in 11
	PurgeBodyCanonical  []byte `protobuf:"bytes,11,opt,name=purge_body_canonical,json=purgeBodyCanonical,proto3" json:"purge_body_canonical,omitempty"` // the PurgeBody DOCUMENT, canonical JSON bytes
	unknownFields       protoimpl.UnknownFields
	sizeCache           protoimpl.SizeCache
}

func (x *PurgeArtifact) Reset() {
	*x = PurgeArtifact{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[28]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *PurgeArtifact) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*PurgeArtifact) ProtoMessage() {}

func (x *PurgeArtifact) ProtoReflect() protoreflect.Message {
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

// Deprecated: Use PurgeArtifact.ProtoReflect.Descriptor instead.
func (*PurgeArtifact) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{28}
}

func (x *PurgeArtifact) GetSessionId() string {
	if x != nil {
		return x.SessionId
	}
	return ""
}

func (x *PurgeArtifact) GetExecutorIncarnation() uint64 {
	if x != nil {
		return x.ExecutorIncarnation
	}
	return 0
}

func (x *PurgeArtifact) GetPurgeId() string {
	if x != nil {
		return x.PurgeId
	}
	return ""
}

func (x *PurgeArtifact) GetPurgeBodyDigest() []byte {
	if x != nil {
		return x.PurgeBodyDigest
	}
	return nil
}

func (x *PurgeArtifact) GetPurgeBodyCanonical() []byte {
	if x != nil {
		return x.PurgeBodyCanonical
	}
	return nil
}

// DOCUMENT SHAPE (not a wire message): the subject of `purge_body_digest`. Canonical form
// `cozy.worker.v1.PurgeBody/1`.
type PurgeBody struct {
	state               protoimpl.MessageState `protogen:"open.v1"`
	TenantId            string                 `protobuf:"bytes,1,opt,name=tenant_id,json=tenantId,proto3" json:"tenant_id,omitempty"`                                   // the exact tenant the subjects belong to
	Subjects            []*ArtifactSubject     `protobuf:"bytes,2,rep,name=subjects,proto3" json:"subjects,omitempty"`                                                   // sorted by (kind, digest)
	TombstoneGeneration uint64                 `protobuf:"varint,3,opt,name=tombstone_generation,json=tombstoneGeneration,proto3" json:"tombstone_generation,omitempty"` // TensorFS generation fence; a stale tombstone refuses
	TargetInstanceId    string                 `protobuf:"bytes,4,opt,name=target_instance_id,json=targetInstanceId,proto3" json:"target_instance_id,omitempty"`
	TargetGeneration    uint64                 `protobuf:"varint,5,opt,name=target_generation,json=targetGeneration,proto3" json:"target_generation,omitempty"` // the worker executor_incarnation targeted; stale refuses
	DeadlineUnix        uint64                 `protobuf:"varint,6,opt,name=deadline_unix,json=deadlineUnix,proto3" json:"deadline_unix,omitempty"`
	unknownFields       protoimpl.UnknownFields
	sizeCache           protoimpl.SizeCache
}

func (x *PurgeBody) Reset() {
	*x = PurgeBody{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[29]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *PurgeBody) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*PurgeBody) ProtoMessage() {}

func (x *PurgeBody) ProtoReflect() protoreflect.Message {
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

// Deprecated: Use PurgeBody.ProtoReflect.Descriptor instead.
func (*PurgeBody) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{29}
}

func (x *PurgeBody) GetTenantId() string {
	if x != nil {
		return x.TenantId
	}
	return ""
}

func (x *PurgeBody) GetSubjects() []*ArtifactSubject {
	if x != nil {
		return x.Subjects
	}
	return nil
}

func (x *PurgeBody) GetTombstoneGeneration() uint64 {
	if x != nil {
		return x.TombstoneGeneration
	}
	return 0
}

func (x *PurgeBody) GetTargetInstanceId() string {
	if x != nil {
		return x.TargetInstanceId
	}
	return ""
}

func (x *PurgeBody) GetTargetGeneration() uint64 {
	if x != nil {
		return x.TargetGeneration
	}
	return 0
}

func (x *PurgeBody) GetDeadlineUnix() uint64 {
	if x != nil {
		return x.DeadlineUnix
	}
	return 0
}

type PurgeArtifactReport struct {
	state protoimpl.MessageState `protogen:"open.v1"`
	// 9/10/11 by the ONEOF-LAST RULE. Same meaning.
	SessionId                  string `protobuf:"bytes,1,opt,name=session_id,json=sessionId,proto3" json:"session_id,omitempty"`
	ExecutorIncarnation        uint64 `protobuf:"varint,2,opt,name=executor_incarnation,json=executorIncarnation,proto3" json:"executor_incarnation,omitempty"`
	PurgeId                    string `protobuf:"bytes,3,opt,name=purge_id,json=purgeId,proto3" json:"purge_id,omitempty"`
	PurgeBodyDigest            []byte `protobuf:"bytes,4,opt,name=purge_body_digest,json=purgeBodyDigest,proto3" json:"purge_body_digest,omitempty"`
	AppliedTombstoneGeneration uint64 `protobuf:"varint,8,opt,name=applied_tombstone_generation,json=appliedTombstoneGeneration,proto3" json:"applied_tombstone_generation,omitempty"` // echo of what the worker applied
	// Types that are valid to be assigned to Outcome:
	//
	//	*PurgeArtifactReport_Pending
	//	*PurgeArtifactReport_Absent
	//	*PurgeArtifactReport_Fault
	Outcome       isPurgeArtifactReport_Outcome `protobuf_oneof:"outcome"`
	unknownFields protoimpl.UnknownFields
	sizeCache     protoimpl.SizeCache
}

func (x *PurgeArtifactReport) Reset() {
	*x = PurgeArtifactReport{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[30]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *PurgeArtifactReport) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*PurgeArtifactReport) ProtoMessage() {}

func (x *PurgeArtifactReport) ProtoReflect() protoreflect.Message {
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

// Deprecated: Use PurgeArtifactReport.ProtoReflect.Descriptor instead.
func (*PurgeArtifactReport) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{30}
}

func (x *PurgeArtifactReport) GetSessionId() string {
	if x != nil {
		return x.SessionId
	}
	return ""
}

func (x *PurgeArtifactReport) GetExecutorIncarnation() uint64 {
	if x != nil {
		return x.ExecutorIncarnation
	}
	return 0
}

func (x *PurgeArtifactReport) GetPurgeId() string {
	if x != nil {
		return x.PurgeId
	}
	return ""
}

func (x *PurgeArtifactReport) GetPurgeBodyDigest() []byte {
	if x != nil {
		return x.PurgeBodyDigest
	}
	return nil
}

func (x *PurgeArtifactReport) GetAppliedTombstoneGeneration() uint64 {
	if x != nil {
		return x.AppliedTombstoneGeneration
	}
	return 0
}

func (x *PurgeArtifactReport) GetOutcome() isPurgeArtifactReport_Outcome {
	if x != nil {
		return x.Outcome
	}
	return nil
}

func (x *PurgeArtifactReport) GetPending() *PendingLeases {
	if x != nil {
		if x, ok := x.Outcome.(*PurgeArtifactReport_Pending); ok {
			return x.Pending
		}
	}
	return nil
}

func (x *PurgeArtifactReport) GetAbsent() *AbsenceEvidence {
	if x != nil {
		if x, ok := x.Outcome.(*PurgeArtifactReport_Absent); ok {
			return x.Absent
		}
	}
	return nil
}

func (x *PurgeArtifactReport) GetFault() *PurgeFault {
	if x != nil {
		if x, ok := x.Outcome.(*PurgeArtifactReport_Fault); ok {
			return x.Fault
		}
	}
	return nil
}

type isPurgeArtifactReport_Outcome interface {
	isPurgeArtifactReport_Outcome()
}

type PurgeArtifactReport_Pending struct {
	Pending *PendingLeases `protobuf:"bytes,9,opt,name=pending,proto3,oneof"` // live leases still hold the subject
}

type PurgeArtifactReport_Absent struct {
	Absent *AbsenceEvidence `protobuf:"bytes,10,opt,name=absent,proto3,oneof"` // canonical local-absence evidence (purge complete)
}

type PurgeArtifactReport_Fault struct {
	Fault *PurgeFault `protobuf:"bytes,11,opt,name=fault,proto3,oneof"` // typed refusal
}

func (*PurgeArtifactReport_Pending) isPurgeArtifactReport_Outcome() {}

func (*PurgeArtifactReport_Absent) isPurgeArtifactReport_Outcome() {}

func (*PurgeArtifactReport_Fault) isPurgeArtifactReport_Outcome() {}

// DOCUMENT SHAPE (not a wire message): the subject of `exec_spec_digest`. Canonical form
// `cozy.worker.v1.ExecutionSpec/1`.
type ExecutionSpec struct {
	state protoimpl.MessageState `protogen:"open.v1"`
	// the ONEOF-LAST RULE. This message is the DIGEST
	// SUBJECT, so the rule is load-bearing here above all.
	EndpointReleaseId string `protobuf:"bytes,1,opt,name=endpoint_release_id,json=endpointReleaseId,proto3" json:"endpoint_release_id,omitempty"` // the EndpointRelease's identity; nothing else here is
	// "the release"
	ImageDigest string `protobuf:"bytes,2,opt,name=image_digest,json=imageDigest,proto3" json:"image_digest,omitempty"` // class (b): OCI image content digest, spelled
	// `sha256:<hex>`, of the exact execution environment
	ConfigDigest string `protobuf:"bytes,3,opt,name=config_digest,json=configDigest,proto3" json:"config_digest,omitempty"` // class (a): the evaluated-config DOCUMENT's identity
	// (the admission fact), spelled `sha256:<hex>`. Owned by
	// the config surface (cr-003); identity only here.
	DeadlineUnixMs uint64 `protobuf:"varint,6,opt,name=deadline_unix_ms,json=deadlineUnixMs,proto3" json:"deadline_unix_ms,omitempty"` // absolute attempt deadline; INSIDE the digest - an
	// admission fact the SUPERVISOR enforces locally
	// (ctx.deadline's source). 0 = none.
	//
	// Types that are valid to be assigned to Spec:
	//
	//	*ExecutionSpec_Serving
	//	*ExecutionSpec_Job
	Spec          isExecutionSpec_Spec `protobuf_oneof:"spec"`
	unknownFields protoimpl.UnknownFields
	sizeCache     protoimpl.SizeCache
}

func (x *ExecutionSpec) Reset() {
	*x = ExecutionSpec{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[31]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *ExecutionSpec) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*ExecutionSpec) ProtoMessage() {}

func (x *ExecutionSpec) ProtoReflect() protoreflect.Message {
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

// Deprecated: Use ExecutionSpec.ProtoReflect.Descriptor instead.
func (*ExecutionSpec) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{31}
}

func (x *ExecutionSpec) GetEndpointReleaseId() string {
	if x != nil {
		return x.EndpointReleaseId
	}
	return ""
}

func (x *ExecutionSpec) GetImageDigest() string {
	if x != nil {
		return x.ImageDigest
	}
	return ""
}

func (x *ExecutionSpec) GetConfigDigest() string {
	if x != nil {
		return x.ConfigDigest
	}
	return ""
}

func (x *ExecutionSpec) GetDeadlineUnixMs() uint64 {
	if x != nil {
		return x.DeadlineUnixMs
	}
	return 0
}

func (x *ExecutionSpec) GetSpec() isExecutionSpec_Spec {
	if x != nil {
		return x.Spec
	}
	return nil
}

func (x *ExecutionSpec) GetServing() *ServingExecutionSpec {
	if x != nil {
		if x, ok := x.Spec.(*ExecutionSpec_Serving); ok {
			return x.Serving
		}
	}
	return nil
}

func (x *ExecutionSpec) GetJob() *JobExecutionSpec {
	if x != nil {
		if x, ok := x.Spec.(*ExecutionSpec_Job); ok {
			return x.Job
		}
	}
	return nil
}

type isExecutionSpec_Spec interface {
	isExecutionSpec_Spec()
}

type ExecutionSpec_Serving struct {
	Serving *ServingExecutionSpec `protobuf:"bytes,7,opt,name=serving,proto3,oneof"`
}

type ExecutionSpec_Job struct {
	Job *JobExecutionSpec `protobuf:"bytes,8,opt,name=job,proto3,oneof"`
}

func (*ExecutionSpec_Serving) isExecutionSpec_Spec() {}

func (*ExecutionSpec_Job) isExecutionSpec_Spec() {}

type ServingExecutionSpec struct {
	state protoimpl.MessageState `protogen:"open.v1"`
	// Both ids are class (a) DOCUMENT identities spelled `sha256:<hex>`. Both documents are
	// TENSORHUB's (th-004), frozen there under tfs-013's writer - this protocol transports the
	// identities and never canonicalizes either document.
	EntrypointBindingPlanId string `protobuf:"bytes,1,opt,name=entrypoint_binding_plan_id,json=entrypointBindingPlanId,proto3" json:"entrypoint_binding_plan_id,omitempty"` // sha256 of the EntrypointBindingPlan document
	AttemptBindingId        string `protobuf:"bytes,2,opt,name=attempt_binding_id,json=attemptBindingId,proto3" json:"attempt_binding_id,omitempty"`                        // sha256 of the AttemptBinding document (that plan
	unknownFields           protoimpl.UnknownFields
	sizeCache               protoimpl.SizeCache
}

func (x *ServingExecutionSpec) Reset() {
	*x = ServingExecutionSpec{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[32]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *ServingExecutionSpec) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*ServingExecutionSpec) ProtoMessage() {}

func (x *ServingExecutionSpec) ProtoReflect() protoreflect.Message {
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

// Deprecated: Use ServingExecutionSpec.ProtoReflect.Descriptor instead.
func (*ServingExecutionSpec) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{32}
}

func (x *ServingExecutionSpec) GetEntrypointBindingPlanId() string {
	if x != nil {
		return x.EntrypointBindingPlanId
	}
	return ""
}

func (x *ServingExecutionSpec) GetAttemptBindingId() string {
	if x != nil {
		return x.AttemptBindingId
	}
	return ""
}

type JobExecutionSpec struct {
	state               protoimpl.MessageState `protogen:"open.v1"`
	BuildId             string                 `protobuf:"bytes,1,opt,name=build_id,json=buildId,proto3" json:"build_id,omitempty"`
	JobDescriptorId     string                 `protobuf:"bytes,2,opt,name=job_descriptor_id,json=jobDescriptorId,proto3" json:"job_descriptor_id,omitempty"`
	PublicationContract *PublicationContract   `protobuf:"bytes,3,opt,name=publication_contract,json=publicationContract,proto3" json:"publication_contract,omitempty"` // bounded publication contract
	unknownFields       protoimpl.UnknownFields
	sizeCache           protoimpl.SizeCache
}

func (x *JobExecutionSpec) Reset() {
	*x = JobExecutionSpec{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[33]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *JobExecutionSpec) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*JobExecutionSpec) ProtoMessage() {}

func (x *JobExecutionSpec) ProtoReflect() protoreflect.Message {
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

// Deprecated: Use JobExecutionSpec.ProtoReflect.Descriptor instead.
func (*JobExecutionSpec) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{33}
}

func (x *JobExecutionSpec) GetBuildId() string {
	if x != nil {
		return x.BuildId
	}
	return ""
}

func (x *JobExecutionSpec) GetJobDescriptorId() string {
	if x != nil {
		return x.JobDescriptorId
	}
	return ""
}

func (x *JobExecutionSpec) GetPublicationContract() *PublicationContract {
	if x != nil {
		return x.PublicationContract
	}
	return nil
}

type DeliveryGrant struct {
	state         protoimpl.MessageState `protogen:"open.v1"`
	Credential    *Credential            `protobuf:"bytes,1,opt,name=credential,proto3" json:"credential,omitempty"`                        // per-attempt scoped capability token (epoch inside)
	FileBaseUrl   string                 `protobuf:"bytes,2,opt,name=file_base_url,json=fileBaseUrl,proto3" json:"file_base_url,omitempty"` // the base the token authenticates to (its other half)
	ExpiresAtUnix uint64                 `protobuf:"varint,3,opt,name=expires_at_unix,json=expiresAtUnix,proto3" json:"expires_at_unix,omitempty"`
	Inputs        []*InputLocation       `protobuf:"bytes,4,rep,name=inputs,proto3" json:"inputs,omitempty"`   // sorted by input_id
	Outputs       []*OutputDestination   `protobuf:"bytes,5,rep,name=outputs,proto3" json:"outputs,omitempty"` // sorted by output_id
	unknownFields protoimpl.UnknownFields
	sizeCache     protoimpl.SizeCache
}

func (x *DeliveryGrant) Reset() {
	*x = DeliveryGrant{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[34]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *DeliveryGrant) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*DeliveryGrant) ProtoMessage() {}

func (x *DeliveryGrant) ProtoReflect() protoreflect.Message {
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

// Deprecated: Use DeliveryGrant.ProtoReflect.Descriptor instead.
func (*DeliveryGrant) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{34}
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

func (x *DeliveryGrant) GetInputs() []*InputLocation {
	if x != nil {
		return x.Inputs
	}
	return nil
}

func (x *DeliveryGrant) GetOutputs() []*OutputDestination {
	if x != nil {
		return x.Outputs
	}
	return nil
}

type InputLocation struct {
	state  protoimpl.MessageState `protogen:"open.v1"`
	Digest []byte                 `protobuf:"bytes,1,opt,name=digest,proto3" json:"digest,omitempty"` // class (b): CONTENT digest of the input asset
	Url    string                 `protobuf:"bytes,2,opt,name=url,proto3" json:"url,omitempty"`       // presigned download URL; relative to file_base_url if
	// not absolute
	Length  uint64 `protobuf:"varint,3,opt,name=length,proto3" json:"length,omitempty"`                 // expected length (validates; never folded into digest)
	InputId string `protobuf:"bytes,4,opt,name=input_id,json=inputId,proto3" json:"input_id,omitempty"` // STABLE field/input identity: the payload field path,
	// list index included. Inputs are IDENTIFIED, ORDERED
	// subjects, never a digest-sorted anonymous list.
	Order         uint32 `protobuf:"varint,5,opt,name=order,proto3" json:"order,omitempty"`                      // declared order within the containing list; 0 for scalars
	KindMime      string `protobuf:"bytes,6,opt,name=kind_mime,json=kindMime,proto3" json:"kind_mime,omitempty"` // declared asset kind/MIME from the typed field
	unknownFields protoimpl.UnknownFields
	sizeCache     protoimpl.SizeCache
}

func (x *InputLocation) Reset() {
	*x = InputLocation{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[35]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *InputLocation) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*InputLocation) ProtoMessage() {}

func (x *InputLocation) ProtoReflect() protoreflect.Message {
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

// Deprecated: Use InputLocation.ProtoReflect.Descriptor instead.
func (*InputLocation) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{35}
}

func (x *InputLocation) GetDigest() []byte {
	if x != nil {
		return x.Digest
	}
	return nil
}

func (x *InputLocation) GetUrl() string {
	if x != nil {
		return x.Url
	}
	return ""
}

func (x *InputLocation) GetLength() uint64 {
	if x != nil {
		return x.Length
	}
	return 0
}

func (x *InputLocation) GetInputId() string {
	if x != nil {
		return x.InputId
	}
	return ""
}

func (x *InputLocation) GetOrder() uint32 {
	if x != nil {
		return x.Order
	}
	return 0
}

func (x *InputLocation) GetKindMime() string {
	if x != nil {
		return x.KindMime
	}
	return ""
}

type OutputDestination struct {
	state         protoimpl.MessageState `protogen:"open.v1"`
	OutputId      string                 `protobuf:"bytes,1,opt,name=output_id,json=outputId,proto3" json:"output_id,omitempty"`
	Url           string                 `protobuf:"bytes,2,opt,name=url,proto3" json:"url,omitempty"` // presigned upload URL
	MaxBytes      uint64                 `protobuf:"varint,3,opt,name=max_bytes,json=maxBytes,proto3" json:"max_bytes,omitempty"`
	unknownFields protoimpl.UnknownFields
	sizeCache     protoimpl.SizeCache
}

func (x *OutputDestination) Reset() {
	*x = OutputDestination{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[36]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *OutputDestination) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*OutputDestination) ProtoMessage() {}

func (x *OutputDestination) ProtoReflect() protoreflect.Message {
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

// Deprecated: Use OutputDestination.ProtoReflect.Descriptor instead.
func (*OutputDestination) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{36}
}

func (x *OutputDestination) GetOutputId() string {
	if x != nil {
		return x.OutputId
	}
	return ""
}

func (x *OutputDestination) GetUrl() string {
	if x != nil {
		return x.Url
	}
	return ""
}

func (x *OutputDestination) GetMaxBytes() uint64 {
	if x != nil {
		return x.MaxBytes
	}
	return 0
}

type Credential struct {
	state  protoimpl.MessageState `protogen:"open.v1"`
	Issuer string                 `protobuf:"bytes,1,opt,name=issuer,proto3" json:"issuer,omitempty"`
	KeyId  string                 `protobuf:"bytes,2,opt,name=key_id,json=keyId,proto3" json:"key_id,omitempty"`
	Epoch  uint64                 `protobuf:"varint,3,opt,name=epoch,proto3" json:"epoch,omitempty"` // rotation label; the ONLY non-secret credential
	// identifier
	ExpiresAtUnix uint64 `protobuf:"varint,4,opt,name=expires_at_unix,json=expiresAtUnix,proto3" json:"expires_at_unix,omitempty"`
	Token         []byte `protobuf:"bytes,5,opt,name=token,proto3" json:"token,omitempty"` // STRUCTURALLY SECRET: `bytes` so no logging formatter
	unknownFields protoimpl.UnknownFields
	sizeCache     protoimpl.SizeCache
}

func (x *Credential) Reset() {
	*x = Credential{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[37]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *Credential) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*Credential) ProtoMessage() {}

func (x *Credential) ProtoReflect() protoreflect.Message {
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

// Deprecated: Use Credential.ProtoReflect.Descriptor instead.
func (*Credential) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{37}
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

type ArtifactGrant struct {
	state         protoimpl.MessageState `protogen:"open.v1"`
	FileBaseUrl   string                 `protobuf:"bytes,1,opt,name=file_base_url,json=fileBaseUrl,proto3" json:"file_base_url,omitempty"`
	ExpiresAtUnix uint64                 `protobuf:"varint,2,opt,name=expires_at_unix,json=expiresAtUnix,proto3" json:"expires_at_unix,omitempty"`
	Subjects      []*ArtifactSubject     `protobuf:"bytes,3,rep,name=subjects,proto3" json:"subjects,omitempty"` // sorted by (kind, digest)
	unknownFields protoimpl.UnknownFields
	sizeCache     protoimpl.SizeCache
}

func (x *ArtifactGrant) Reset() {
	*x = ArtifactGrant{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[38]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *ArtifactGrant) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*ArtifactGrant) ProtoMessage() {}

func (x *ArtifactGrant) ProtoReflect() protoreflect.Message {
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

// Deprecated: Use ArtifactGrant.ProtoReflect.Descriptor instead.
func (*ArtifactGrant) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{38}
}

func (x *ArtifactGrant) GetFileBaseUrl() string {
	if x != nil {
		return x.FileBaseUrl
	}
	return ""
}

func (x *ArtifactGrant) GetExpiresAtUnix() uint64 {
	if x != nil {
		return x.ExpiresAtUnix
	}
	return 0
}

func (x *ArtifactGrant) GetSubjects() []*ArtifactSubject {
	if x != nil {
		return x.Subjects
	}
	return nil
}

type ArtifactSubject struct {
	state         protoimpl.MessageState `protogen:"open.v1"`
	Kind          SubjectKind            `protobuf:"varint,1,opt,name=kind,proto3,enum=cozy.worker.v1.SubjectKind" json:"kind,omitempty"`
	Digest        []byte                 `protobuf:"bytes,2,opt,name=digest,proto3" json:"digest,omitempty"` // class (b): the artifact's CONTENT digest
	unknownFields protoimpl.UnknownFields
	sizeCache     protoimpl.SizeCache
}

func (x *ArtifactSubject) Reset() {
	*x = ArtifactSubject{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[39]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *ArtifactSubject) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*ArtifactSubject) ProtoMessage() {}

func (x *ArtifactSubject) ProtoReflect() protoreflect.Message {
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

// Deprecated: Use ArtifactSubject.ProtoReflect.Descriptor instead.
func (*ArtifactSubject) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{39}
}

func (x *ArtifactSubject) GetKind() SubjectKind {
	if x != nil {
		return x.Kind
	}
	return SubjectKind_SUBJECT_KIND_UNSPECIFIED
}

func (x *ArtifactSubject) GetDigest() []byte {
	if x != nil {
		return x.Digest
	}
	return nil
}

type DrainPolicy struct {
	state                 protoimpl.MessageState `protogen:"open.v1"`
	DeadlineMs            uint64                 `protobuf:"varint,1,opt,name=deadline_ms,json=deadlineMs,proto3" json:"deadline_ms,omitempty"` // grace budget; 0 = wait without a lifecycle deadline
	ForceCancelAtDeadline bool                   `protobuf:"varint,2,opt,name=force_cancel_at_deadline,json=forceCancelAtDeadline,proto3" json:"force_cancel_at_deadline,omitempty"`
	unknownFields         protoimpl.UnknownFields
	sizeCache             protoimpl.SizeCache
}

func (x *DrainPolicy) Reset() {
	*x = DrainPolicy{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[40]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *DrainPolicy) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*DrainPolicy) ProtoMessage() {}

func (x *DrainPolicy) ProtoReflect() protoreflect.Message {
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

// Deprecated: Use DrainPolicy.ProtoReflect.Descriptor instead.
func (*DrainPolicy) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{40}
}

func (x *DrainPolicy) GetDeadlineMs() uint64 {
	if x != nil {
		return x.DeadlineMs
	}
	return 0
}

func (x *DrainPolicy) GetForceCancelAtDeadline() bool {
	if x != nil {
		return x.ForceCancelAtDeadline
	}
	return false
}

type ResourceCaps struct {
	state         protoimpl.MessageState `protogen:"open.v1"`
	GpuRequired   bool                   `protobuf:"varint,1,opt,name=gpu_required,json=gpuRequired,proto3" json:"gpu_required,omitempty"`
	MaxVramBytes  uint64                 `protobuf:"varint,2,opt,name=max_vram_bytes,json=maxVramBytes,proto3" json:"max_vram_bytes,omitempty"`
	MaxRssBytes   uint64                 `protobuf:"varint,3,opt,name=max_rss_bytes,json=maxRssBytes,proto3" json:"max_rss_bytes,omitempty"`
	MaxDiskBytes  uint64                 `protobuf:"varint,4,opt,name=max_disk_bytes,json=maxDiskBytes,proto3" json:"max_disk_bytes,omitempty"`
	unknownFields protoimpl.UnknownFields
	sizeCache     protoimpl.SizeCache
}

func (x *ResourceCaps) Reset() {
	*x = ResourceCaps{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[41]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *ResourceCaps) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*ResourceCaps) ProtoMessage() {}

func (x *ResourceCaps) ProtoReflect() protoreflect.Message {
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

// Deprecated: Use ResourceCaps.ProtoReflect.Descriptor instead.
func (*ResourceCaps) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{41}
}

func (x *ResourceCaps) GetGpuRequired() bool {
	if x != nil {
		return x.GpuRequired
	}
	return false
}

func (x *ResourceCaps) GetMaxVramBytes() uint64 {
	if x != nil {
		return x.MaxVramBytes
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
	Outputs       []*OutputSpec          `protobuf:"bytes,1,rep,name=outputs,proto3" json:"outputs,omitempty"`                // sorted by output_id
	GrantId       string                 `protobuf:"bytes,2,opt,name=grant_id,json=grantId,proto3" json:"grant_id,omitempty"` // the publication grant authorizing writes
	unknownFields protoimpl.UnknownFields
	sizeCache     protoimpl.SizeCache
}

func (x *PublicationContract) Reset() {
	*x = PublicationContract{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[42]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *PublicationContract) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*PublicationContract) ProtoMessage() {}

func (x *PublicationContract) ProtoReflect() protoreflect.Message {
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

// Deprecated: Use PublicationContract.ProtoReflect.Descriptor instead.
func (*PublicationContract) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{42}
}

func (x *PublicationContract) GetOutputs() []*OutputSpec {
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

type OutputSpec struct {
	state         protoimpl.MessageState `protogen:"open.v1"`
	OutputId      string                 `protobuf:"bytes,1,opt,name=output_id,json=outputId,proto3" json:"output_id,omitempty"`
	MimeType      string                 `protobuf:"bytes,2,opt,name=mime_type,json=mimeType,proto3" json:"mime_type,omitempty"`
	MaxBytes      uint64                 `protobuf:"varint,3,opt,name=max_bytes,json=maxBytes,proto3" json:"max_bytes,omitempty"`
	unknownFields protoimpl.UnknownFields
	sizeCache     protoimpl.SizeCache
}

func (x *OutputSpec) Reset() {
	*x = OutputSpec{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[43]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *OutputSpec) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*OutputSpec) ProtoMessage() {}

func (x *OutputSpec) ProtoReflect() protoreflect.Message {
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

// Deprecated: Use OutputSpec.ProtoReflect.Descriptor instead.
func (*OutputSpec) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{43}
}

func (x *OutputSpec) GetOutputId() string {
	if x != nil {
		return x.OutputId
	}
	return ""
}

func (x *OutputSpec) GetMimeType() string {
	if x != nil {
		return x.MimeType
	}
	return ""
}

func (x *OutputSpec) GetMaxBytes() uint64 {
	if x != nil {
		return x.MaxBytes
	}
	return 0
}

type WorkerResources struct {
	state               protoimpl.MessageState `protogen:"open.v1"`
	GpuCount            uint32                 `protobuf:"varint,1,opt,name=gpu_count,json=gpuCount,proto3" json:"gpu_count,omitempty"`
	GpuName             string                 `protobuf:"bytes,2,opt,name=gpu_name,json=gpuName,proto3" json:"gpu_name,omitempty"`
	GpuSm               uint32                 `protobuf:"varint,3,opt,name=gpu_sm,json=gpuSm,proto3" json:"gpu_sm,omitempty"`
	VramTotalBytes      uint64                 `protobuf:"varint,4,opt,name=vram_total_bytes,json=vramTotalBytes,proto3" json:"vram_total_bytes,omitempty"`
	DriverVersion       string                 `protobuf:"bytes,5,opt,name=driver_version,json=driverVersion,proto3" json:"driver_version,omitempty"`
	CudaVersion         string                 `protobuf:"bytes,6,opt,name=cuda_version,json=cudaVersion,proto3" json:"cuda_version,omitempty"`
	TorchVersion        string                 `protobuf:"bytes,7,opt,name=torch_version,json=torchVersion,proto3" json:"torch_version,omitempty"` // TELEMETRY ONLY; never a capability gate
	Platform            string                 `protobuf:"bytes,8,opt,name=platform,proto3" json:"platform,omitempty"`
	HostRamTotalBytes   uint64                 `protobuf:"varint,9,opt,name=host_ram_total_bytes,json=hostRamTotalBytes,proto3" json:"host_ram_total_bytes,omitempty"`
	VcpuCount           uint32                 `protobuf:"varint,10,opt,name=vcpu_count,json=vcpuCount,proto3" json:"vcpu_count,omitempty"`
	DiskTotalBytes      uint64                 `protobuf:"varint,11,opt,name=disk_total_bytes,json=diskTotalBytes,proto3" json:"disk_total_bytes,omitempty"`
	PinnedCapacityBytes uint64                 `protobuf:"varint,12,opt,name=pinned_capacity_bytes,json=pinnedCapacityBytes,proto3" json:"pinned_capacity_bytes,omitempty"`
	PowerCapWatts       uint32                 `protobuf:"varint,13,opt,name=power_cap_watts,json=powerCapWatts,proto3" json:"power_cap_watts,omitempty"` // NVML-read device configuration (probe-key term)
	MigProfile          string                 `protobuf:"bytes,14,opt,name=mig_profile,json=migProfile,proto3" json:"mig_profile,omitempty"`             // empty = whole device
	Interconnect        string                 `protobuf:"bytes,15,opt,name=interconnect,proto3" json:"interconnect,omitempty"`                           // pcie3|pcie4|pcie5|nvlink... (transfer-coefficient key)
	PeerAccess          bool                   `protobuf:"varint,16,opt,name=peer_access,json=peerAccess,proto3" json:"peer_access,omitempty"`
	H2DGbpsMeasured     uint32                 `protobuf:"varint,17,opt,name=h2d_gbps_measured,json=h2dGbpsMeasured,proto3" json:"h2d_gbps_measured,omitempty"`
	D2HGbpsMeasured     uint32                 `protobuf:"varint,18,opt,name=d2h_gbps_measured,json=d2hGbpsMeasured,proto3" json:"d2h_gbps_measured,omitempty"`
	Unreadable          []string               `protobuf:"bytes,19,rep,name=unreadable,proto3" json:"unreadable,omitempty"` // field NAMES that were unreadable/unmeasured.
	unknownFields       protoimpl.UnknownFields
	sizeCache           protoimpl.SizeCache
}

func (x *WorkerResources) Reset() {
	*x = WorkerResources{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[44]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *WorkerResources) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*WorkerResources) ProtoMessage() {}

func (x *WorkerResources) ProtoReflect() protoreflect.Message {
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

// Deprecated: Use WorkerResources.ProtoReflect.Descriptor instead.
func (*WorkerResources) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{44}
}

func (x *WorkerResources) GetGpuCount() uint32 {
	if x != nil {
		return x.GpuCount
	}
	return 0
}

func (x *WorkerResources) GetGpuName() string {
	if x != nil {
		return x.GpuName
	}
	return ""
}

func (x *WorkerResources) GetGpuSm() uint32 {
	if x != nil {
		return x.GpuSm
	}
	return 0
}

func (x *WorkerResources) GetVramTotalBytes() uint64 {
	if x != nil {
		return x.VramTotalBytes
	}
	return 0
}

func (x *WorkerResources) GetDriverVersion() string {
	if x != nil {
		return x.DriverVersion
	}
	return ""
}

func (x *WorkerResources) GetCudaVersion() string {
	if x != nil {
		return x.CudaVersion
	}
	return ""
}

func (x *WorkerResources) GetTorchVersion() string {
	if x != nil {
		return x.TorchVersion
	}
	return ""
}

func (x *WorkerResources) GetPlatform() string {
	if x != nil {
		return x.Platform
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

func (x *WorkerResources) GetPinnedCapacityBytes() uint64 {
	if x != nil {
		return x.PinnedCapacityBytes
	}
	return 0
}

func (x *WorkerResources) GetPowerCapWatts() uint32 {
	if x != nil {
		return x.PowerCapWatts
	}
	return 0
}

func (x *WorkerResources) GetMigProfile() string {
	if x != nil {
		return x.MigProfile
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

func (x *WorkerResources) GetH2DGbpsMeasured() uint32 {
	if x != nil {
		return x.H2DGbpsMeasured
	}
	return 0
}

func (x *WorkerResources) GetD2HGbpsMeasured() uint32 {
	if x != nil {
		return x.D2HGbpsMeasured
	}
	return 0
}

func (x *WorkerResources) GetUnreadable() []string {
	if x != nil {
		return x.Unreadable
	}
	return nil
}

type ServingCapacity struct {
	state                         protoimpl.MessageState `protogen:"open.v1"`
	ReadyEntrypointBindingPlanIds []string               `protobuf:"bytes,1,rep,name=ready_entrypoint_binding_plan_ids,json=readyEntrypointBindingPlanIds,proto3" json:"ready_entrypoint_binding_plan_ids,omitempty"` // dispatchable now: a real
	// forward completed. Sorted.
	LoadableEntrypointBindingPlanIds []string `protobuf:"bytes,2,rep,name=loadable_entrypoint_binding_plan_ids,json=loadableEntrypointBindingPlanIds,proto3" json:"loadable_entrypoint_binding_plan_ids,omitempty"` // materializing; DISJOINT from
	// ready. Sorted.
	FreeVramBytes uint64 `protobuf:"varint,3,opt,name=free_vram_bytes,json=freeVramBytes,proto3" json:"free_vram_bytes,omitempty"` // measured, quantized
	unknownFields protoimpl.UnknownFields
	sizeCache     protoimpl.SizeCache
}

func (x *ServingCapacity) Reset() {
	*x = ServingCapacity{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[45]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *ServingCapacity) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*ServingCapacity) ProtoMessage() {}

func (x *ServingCapacity) ProtoReflect() protoreflect.Message {
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

// Deprecated: Use ServingCapacity.ProtoReflect.Descriptor instead.
func (*ServingCapacity) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{45}
}

func (x *ServingCapacity) GetReadyEntrypointBindingPlanIds() []string {
	if x != nil {
		return x.ReadyEntrypointBindingPlanIds
	}
	return nil
}

func (x *ServingCapacity) GetLoadableEntrypointBindingPlanIds() []string {
	if x != nil {
		return x.LoadableEntrypointBindingPlanIds
	}
	return nil
}

func (x *ServingCapacity) GetFreeVramBytes() uint64 {
	if x != nil {
		return x.FreeVramBytes
	}
	return 0
}

type JobCapacity struct {
	state         protoimpl.MessageState `protogen:"open.v1"`
	JobsInFlight  uint32                 `protobuf:"varint,1,opt,name=jobs_in_flight,json=jobsInFlight,proto3" json:"jobs_in_flight,omitempty"`
	JobsAvailable uint32                 `protobuf:"varint,2,opt,name=jobs_available,json=jobsAvailable,proto3" json:"jobs_available,omitempty"`
	FreeVramBytes uint64                 `protobuf:"varint,3,opt,name=free_vram_bytes,json=freeVramBytes,proto3" json:"free_vram_bytes,omitempty"`
	FreeDiskBytes uint64                 `protobuf:"varint,4,opt,name=free_disk_bytes,json=freeDiskBytes,proto3" json:"free_disk_bytes,omitempty"`
	unknownFields protoimpl.UnknownFields
	sizeCache     protoimpl.SizeCache
}

func (x *JobCapacity) Reset() {
	*x = JobCapacity{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[46]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *JobCapacity) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*JobCapacity) ProtoMessage() {}

func (x *JobCapacity) ProtoReflect() protoreflect.Message {
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

// Deprecated: Use JobCapacity.ProtoReflect.Descriptor instead.
func (*JobCapacity) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{46}
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

func (x *JobCapacity) GetFreeVramBytes() uint64 {
	if x != nil {
		return x.FreeVramBytes
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
	state          protoimpl.MessageState `protogen:"open.v1"`
	RequestId      string                 `protobuf:"bytes,1,opt,name=request_id,json=requestId,proto3" json:"request_id,omitempty"`
	Attempt        uint64                 `protobuf:"varint,2,opt,name=attempt,proto3" json:"attempt,omitempty"`
	Kind           AttemptKind            `protobuf:"varint,3,opt,name=kind,proto3,enum=cozy.worker.v1.AttemptKind" json:"kind,omitempty"`
	State          AttemptState           `protobuf:"varint,4,opt,name=state,proto3,enum=cozy.worker.v1.AttemptState" json:"state,omitempty"`
	ExecSpecDigest []byte                 `protobuf:"bytes,5,opt,name=exec_spec_digest,json=execSpecDigest,proto3" json:"exec_spec_digest,omitempty"`
	unknownFields  protoimpl.UnknownFields
	sizeCache      protoimpl.SizeCache
}

func (x *ActiveAttempt) Reset() {
	*x = ActiveAttempt{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[47]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *ActiveAttempt) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*ActiveAttempt) ProtoMessage() {}

func (x *ActiveAttempt) ProtoReflect() protoreflect.Message {
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

// Deprecated: Use ActiveAttempt.ProtoReflect.Descriptor instead.
func (*ActiveAttempt) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{47}
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

func (x *ActiveAttempt) GetExecSpecDigest() []byte {
	if x != nil {
		return x.ExecSpecDigest
	}
	return nil
}

type Fault struct {
	state         protoimpl.MessageState `protogen:"open.v1"`
	Kind          FaultKind              `protobuf:"varint,1,opt,name=kind,proto3,enum=cozy.worker.v1.FaultKind" json:"kind,omitempty"`
	Subject       string                 `protobuf:"bytes,2,opt,name=subject,proto3" json:"subject,omitempty"` // the binding/deployment/attempt this faults
	Reason        string                 `protobuf:"bytes,3,opt,name=reason,proto3" json:"reason,omitempty"`   // bounded <= 1024 bytes
	Detail        string                 `protobuf:"bytes,4,opt,name=detail,proto3" json:"detail,omitempty"`   // bounded <= 1024 bytes
	unknownFields protoimpl.UnknownFields
	sizeCache     protoimpl.SizeCache
}

func (x *Fault) Reset() {
	*x = Fault{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[48]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *Fault) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*Fault) ProtoMessage() {}

func (x *Fault) ProtoReflect() protoreflect.Message {
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

// Deprecated: Use Fault.ProtoReflect.Descriptor instead.
func (*Fault) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{48}
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

// The OutputManifest DOCUMENT is cr-017's (tfs-013's foreign-document list); this protocol
// carries the shape it transports inside TerminalBody and never re-digests it.
type OutputManifest struct {
	state protoimpl.MessageState `protogen:"open.v1"`
	// INSIDE the TerminalBody document is already fenced by
	// `terminal_digest`; a second digest over a sub-object
	// would be an independent canonicalization of a nested
	// document, i.e. the defect this redesign removed.
	ManifestId    string         `protobuf:"bytes,1,opt,name=manifest_id,json=manifestId,proto3" json:"manifest_id,omitempty"`
	Outputs       []*OutputEntry `protobuf:"bytes,3,rep,name=outputs,proto3" json:"outputs,omitempty"` // sorted by output_id
	unknownFields protoimpl.UnknownFields
	sizeCache     protoimpl.SizeCache
}

func (x *OutputManifest) Reset() {
	*x = OutputManifest{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[49]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *OutputManifest) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*OutputManifest) ProtoMessage() {}

func (x *OutputManifest) ProtoReflect() protoreflect.Message {
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

// Deprecated: Use OutputManifest.ProtoReflect.Descriptor instead.
func (*OutputManifest) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{49}
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
	Digest        []byte                 `protobuf:"bytes,2,opt,name=digest,proto3" json:"digest,omitempty"` // digest class (b): CONTENT digest of the written object
	Length        uint64                 `protobuf:"varint,3,opt,name=length,proto3" json:"length,omitempty"`
	MimeType      string                 `protobuf:"bytes,4,opt,name=mime_type,json=mimeType,proto3" json:"mime_type,omitempty"`
	unknownFields protoimpl.UnknownFields
	sizeCache     protoimpl.SizeCache
}

func (x *OutputEntry) Reset() {
	*x = OutputEntry{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[50]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *OutputEntry) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*OutputEntry) ProtoMessage() {}

func (x *OutputEntry) ProtoReflect() protoreflect.Message {
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

// Deprecated: Use OutputEntry.ProtoReflect.Descriptor instead.
func (*OutputEntry) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{50}
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
	state            protoimpl.MessageState `protogen:"open.v1"`
	RuntimeMs        uint64                 `protobuf:"varint,1,opt,name=runtime_ms,json=runtimeMs,proto3" json:"runtime_ms,omitempty"`
	QueueMs          uint64                 `protobuf:"varint,2,opt,name=queue_ms,json=queueMs,proto3" json:"queue_ms,omitempty"`
	PeakVramBytes    uint64                 `protobuf:"varint,3,opt,name=peak_vram_bytes,json=peakVramBytes,proto3" json:"peak_vram_bytes,omitempty"`
	RssAtEndBytes    uint64                 `protobuf:"varint,4,opt,name=rss_at_end_bytes,json=rssAtEndBytes,proto3" json:"rss_at_end_bytes,omitempty"`
	OutputCount      uint32                 `protobuf:"varint,5,opt,name=output_count,json=outputCount,proto3" json:"output_count,omitempty"`
	InputTokens      uint64                 `protobuf:"varint,6,opt,name=input_tokens,json=inputTokens,proto3" json:"input_tokens,omitempty"`                // ANALYTICS at launch, never settlement input:
	OutputTokens     uint64                 `protobuf:"varint,7,opt,name=output_tokens,json=outputTokens,proto3" json:"output_tokens,omitempty"`             // settlement reads coordinator-owned frozen inputs
	DeviceLeaseMs    uint64                 `protobuf:"varint,8,opt,name=device_lease_ms,json=deviceLeaseMs,proto3" json:"device_lease_ms,omitempty"`        // supervisor-ATTESTED: device lease acquire -> release
	GpuCount         uint32                 `protobuf:"varint,9,opt,name=gpu_count,json=gpuCount,proto3" json:"gpu_count,omitempty"`                         // devices the attempt actually leased
	HandlerMs        uint64                 `protobuf:"varint,10,opt,name=handler_ms,json=handlerMs,proto3" json:"handler_ms,omitempty"`                     // author-code window (supervisor-clamped)
	FinalizationMs   uint64                 `protobuf:"varint,11,opt,name=finalization_ms,json=finalizationMs,proto3" json:"finalization_ms,omitempty"`      // output tail (encode/mux/upload), post lease release
	UnverifiedFields []string               `protobuf:"bytes,12,rep,name=unverified_fields,json=unverifiedFields,proto3" json:"unverified_fields,omitempty"` // PER-FIELD TRUST: names the fields the supervisor
	unknownFields    protoimpl.UnknownFields
	sizeCache        protoimpl.SizeCache
}

func (x *AttemptMetrics) Reset() {
	*x = AttemptMetrics{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[51]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *AttemptMetrics) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*AttemptMetrics) ProtoMessage() {}

func (x *AttemptMetrics) ProtoReflect() protoreflect.Message {
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

// Deprecated: Use AttemptMetrics.ProtoReflect.Descriptor instead.
func (*AttemptMetrics) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{51}
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

func (x *AttemptMetrics) GetPeakVramBytes() uint64 {
	if x != nil {
		return x.PeakVramBytes
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

func (x *AttemptMetrics) GetGpuCount() uint32 {
	if x != nil {
		return x.GpuCount
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
	state protoimpl.MessageState `protogen:"open.v1"`
	// NAMING LAW covers it (a digest field is named *_digest,
	// which is what makes its `sha256:` spelling derivable
	// from the descriptor instead of from a hand-kept table).
	SubjectId string `protobuf:"bytes,1,opt,name=subject_id,json=subjectId,proto3" json:"subject_id,omitempty"` // the bounded WorkerTriageBundle subject. The document is
	// cr-011's under tfs-013's writer; identity only here.
	WriteReceiptDigest []byte `protobuf:"bytes,4,opt,name=write_receipt_digest,json=writeReceiptDigest,proto3" json:"write_receipt_digest,omitempty"` // digest class (a): its write receipt's identity
	Length             uint64 `protobuf:"varint,3,opt,name=length,proto3" json:"length,omitempty"`                                                    // bounded <= 1048576 bytes. Raw unbounded logs and
	unknownFields      protoimpl.UnknownFields
	sizeCache          protoimpl.SizeCache
}

func (x *TriageBundleRef) Reset() {
	*x = TriageBundleRef{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[52]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *TriageBundleRef) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*TriageBundleRef) ProtoMessage() {}

func (x *TriageBundleRef) ProtoReflect() protoreflect.Message {
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

// Deprecated: Use TriageBundleRef.ProtoReflect.Descriptor instead.
func (*TriageBundleRef) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{52}
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

type PendingLeases struct {
	state         protoimpl.MessageState `protogen:"open.v1"`
	LeaseIds      []string               `protobuf:"bytes,1,rep,name=lease_ids,json=leaseIds,proto3" json:"lease_ids,omitempty"` // sorted lexicographic ascending
	unknownFields protoimpl.UnknownFields
	sizeCache     protoimpl.SizeCache
}

func (x *PendingLeases) Reset() {
	*x = PendingLeases{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[53]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *PendingLeases) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*PendingLeases) ProtoMessage() {}

func (x *PendingLeases) ProtoReflect() protoreflect.Message {
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

// Deprecated: Use PendingLeases.ProtoReflect.Descriptor instead.
func (*PendingLeases) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{53}
}

func (x *PendingLeases) GetLeaseIds() []string {
	if x != nil {
		return x.LeaseIds
	}
	return nil
}

type AbsenceEvidence struct {
	state          protoimpl.MessageState `protogen:"open.v1"`
	EvidenceId     string                 `protobuf:"bytes,1,opt,name=evidence_id,json=evidenceId,proto3" json:"evidence_id,omitempty"`             // LocalArtifactAbsenceReceipt id
	EvidenceDigest []byte                 `protobuf:"bytes,2,opt,name=evidence_digest,json=evidenceDigest,proto3" json:"evidence_digest,omitempty"` // class (a): that receipt DOCUMENT's identity. The
	unknownFields  protoimpl.UnknownFields
	sizeCache      protoimpl.SizeCache
}

func (x *AbsenceEvidence) Reset() {
	*x = AbsenceEvidence{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[54]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *AbsenceEvidence) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*AbsenceEvidence) ProtoMessage() {}

func (x *AbsenceEvidence) ProtoReflect() protoreflect.Message {
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

// Deprecated: Use AbsenceEvidence.ProtoReflect.Descriptor instead.
func (*AbsenceEvidence) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{54}
}

func (x *AbsenceEvidence) GetEvidenceId() string {
	if x != nil {
		return x.EvidenceId
	}
	return ""
}

func (x *AbsenceEvidence) GetEvidenceDigest() []byte {
	if x != nil {
		return x.EvidenceDigest
	}
	return nil
}

type PurgeFault struct {
	state         protoimpl.MessageState `protogen:"open.v1"`
	Code          PurgeFaultCode         `protobuf:"varint,1,opt,name=code,proto3,enum=cozy.worker.v1.PurgeFaultCode" json:"code,omitempty"`
	Detail        string                 `protobuf:"bytes,2,opt,name=detail,proto3" json:"detail,omitempty"` // bounded <= 1024 bytes
	unknownFields protoimpl.UnknownFields
	sizeCache     protoimpl.SizeCache
}

func (x *PurgeFault) Reset() {
	*x = PurgeFault{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[55]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *PurgeFault) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*PurgeFault) ProtoMessage() {}

func (x *PurgeFault) ProtoReflect() protoreflect.Message {
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

// Deprecated: Use PurgeFault.ProtoReflect.Descriptor instead.
func (*PurgeFault) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{55}
}

func (x *PurgeFault) GetCode() PurgeFaultCode {
	if x != nil {
		return x.Code
	}
	return PurgeFaultCode_PURGE_FAULT_CODE_UNSPECIFIED
}

func (x *PurgeFault) GetDetail() string {
	if x != nil {
		return x.Detail
	}
	return ""
}

var File_cozy_worker_v1_worker_proto protoreflect.FileDescriptor

const file_cozy_worker_v1_worker_proto_rawDesc = "" +
	"\n" +
	"\x1bcozy/worker/v1/worker.proto\x12\x0ecozy.worker.v1\x1a\x1bgoogle/protobuf/empty.proto\"\x80\x04\n" +
	"\rWorkerMessage\x126\n" +
	"\bregister\x18\x01 \x01(\v2\x18.cozy.worker.v1.RegisterH\x00R\bregister\x120\n" +
	"\x06report\x18\x02 \x01(\v2\x16.cozy.worker.v1.ReportH\x00R\x06report\x12L\n" +
	"\x10attempt_accepted\x18\x03 \x01(\v2\x1f.cozy.worker.v1.AttemptAcceptedH\x00R\x0fattemptAccepted\x12L\n" +
	"\x10attempt_terminal\x18\x04 \x01(\v2\x1f.cozy.worker.v1.AttemptTerminalH\x00R\x0fattemptTerminal\x12@\n" +
	"\fboot_failure\x18\x05 \x01(\v2\x1b.cozy.worker.v1.BootFailureH\x00R\vbootFailure\x12U\n" +
	"\x12checkpoint_request\x18\x06 \x01(\v2$.cozy.worker.v1.JobCheckpointRequestH\x00R\x11checkpointRequest\x12I\n" +
	"\x0echeckpoint_ack\x18\a \x01(\v2 .cozy.worker.v1.JobCheckpointAckH\x00R\rcheckpointAckB\x05\n" +
	"\x03msg\"\xbe\x03\n" +
	"\x12CoordinatorMessage\x12@\n" +
	"\fregister_ack\x18\x01 \x01(\v2\x1b.cozy.worker.v1.RegisterAckH\x00R\vregisterAck\x129\n" +
	"\tdirective\x18\x02 \x01(\v2\x19.cozy.worker.v1.DirectiveH\x00R\tdirective\x12C\n" +
	"\rstart_attempt\x18\x03 \x01(\v2\x1c.cozy.worker.v1.StartAttemptH\x00R\fstartAttempt\x12F\n" +
	"\x0ecancel_attempt\x18\x04 \x01(\v2\x1d.cozy.worker.v1.CancelAttemptH\x00R\rcancelAttempt\x12@\n" +
	"\fterminal_ack\x18\x05 \x01(\v2\x1b.cozy.worker.v1.TerminalAckH\x00R\vterminalAck\x12U\n" +
	"\x12checkpoint_receipt\x18\x06 \x01(\v2$.cozy.worker.v1.JobCheckpointReceiptH\x00R\x11checkpointReceiptB\x05\n" +
	"\x03msg\"\xb8\x01\n" +
	"\fPurgeMessage\x12F\n" +
	"\x0epurge_artifact\x18\x01 \x01(\v2\x1d.cozy.worker.v1.PurgeArtifactH\x00R\rpurgeArtifact\x12Y\n" +
	"\x15purge_artifact_report\x18\x02 \x01(\v2#.cozy.worker.v1.PurgeArtifactReportH\x00R\x13purgeArtifactReportB\x05\n" +
	"\x03msg\"\xa7\x03\n" +
	"\bRegister\x12\x1d\n" +
	"\n" +
	"session_id\x18\x01 \x01(\tR\tsessionId\x121\n" +
	"\x14executor_incarnation\x18\x02 \x01(\x04R\x13executorIncarnation\x12\x1d\n" +
	"\n" +
	"wire_minor\x18\x03 \x01(\rR\twireMinor\x12\x1b\n" +
	"\tworker_id\x18\x04 \x01(\tR\bworkerId\x12\x1f\n" +
	"\vinstance_id\x18\x05 \x01(\tR\n" +
	"instanceId\x12\x1d\n" +
	"\n" +
	"release_id\x18\x06 \x01(\tR\treleaseId\x12!\n" +
	"\fimage_digest\x18\a \x01(\tR\vimageDigest\x12\x1d\n" +
	"\n" +
	"git_commit\x18\b \x01(\tR\tgitCommit\x12=\n" +
	"\tresources\x18\t \x01(\v2\x1f.cozy.worker.v1.WorkerResourcesR\tresources\x12L\n" +
	"\x12recovered_attempts\x18\n" +
	" \x03(\v2\x1d.cozy.worker.v1.ActiveAttemptR\x11recoveredAttempts\"\x9f\x02\n" +
	"\vBootFailure\x12\x1b\n" +
	"\tworker_id\x18\x01 \x01(\tR\bworkerId\x12\x1f\n" +
	"\vinstance_id\x18\x02 \x01(\tR\n" +
	"instanceId\x129\n" +
	"\x06reason\x18\x03 \x01(\x0e2!.cozy.worker.v1.BootFailureReasonR\x06reason\x12\x16\n" +
	"\x06detail\x18\x04 \x01(\tR\x06detail\x12=\n" +
	"\tresources\x18\x05 \x01(\v2\x1f.cozy.worker.v1.WorkerResourcesR\tresources\x12!\n" +
	"\fimage_digest\x18\x06 \x01(\tR\vimageDigest\x12\x1d\n" +
	"\n" +
	"release_id\x18\a \x01(\tR\treleaseId\"\xdb\x01\n" +
	"\vRegisterAck\x12\x1d\n" +
	"\n" +
	"session_id\x18\x01 \x01(\tR\tsessionId\x121\n" +
	"\x14executor_incarnation\x18\x02 \x01(\x04R\x13executorIncarnation\x12\x1d\n" +
	"\n" +
	"wire_minor\x18\x03 \x01(\rR\twireMinor\x12\x1a\n" +
	"\baccepted\x18\x04 \x01(\bR\baccepted\x12?\n" +
	"\trejection\x18\x05 \x01(\x0e2!.cozy.worker.v1.RegisterRejectionR\trejection\"\x91\x04\n" +
	"\tDirective\x12\x1d\n" +
	"\n" +
	"session_id\x18\x01 \x01(\tR\tsessionId\x121\n" +
	"\x14executor_incarnation\x18\x02 \x01(\x04R\x13executorIncarnation\x12\x1a\n" +
	"\brevision\x18\x03 \x01(\x04R\brevision\x121\n" +
	"\aposture\x18\x04 \x01(\x0e2\x17.cozy.worker.v1.PostureR\aposture\x12>\n" +
	"\fdrain_policy\x18\x05 \x01(\v2\x1b.cozy.worker.v1.DrainPolicyR\vdrainPolicy\x12D\n" +
	"\x0eartifact_grant\x18\x06 \x01(\v2\x1d.cozy.worker.v1.ArtifactGrantR\rartifactGrant\x12:\n" +
	"\n" +
	"credential\x18\a \x01(\v2\x1a.cozy.worker.v1.CredentialR\n" +
	"credential\x12\x1d\n" +
	"\n" +
	"wire_minor\x18\n" +
	" \x01(\rR\twireMinor\x12<\n" +
	"\aserving\x18\v \x01(\v2 .cozy.worker.v1.ServingDirectiveH\x00R\aserving\x120\n" +
	"\x03job\x18\f \x01(\v2\x1c.cozy.worker.v1.JobDirectiveH\x00R\x03jobB\x06\n" +
	"\x04modeJ\x04\b\b\x10\tJ\x04\b\t\x10\n" +
	"\"\x81\x01\n" +
	"\x10ServingDirective\x12.\n" +
	"\x13endpoint_release_id\x18\x01 \x01(\tR\x11endpointReleaseId\x12=\n" +
	"\x1bentrypoint_binding_plan_ids\x18\x02 \x03(\tR\x18entrypointBindingPlanIds\"\xbd\x02\n" +
	"\fJobDirective\x12\x19\n" +
	"\bbuild_id\x18\x01 \x01(\tR\abuildId\x12*\n" +
	"\x11job_descriptor_id\x18\x02 \x01(\tR\x0fjobDescriptorId\x12A\n" +
	"\rresource_caps\x18\x03 \x01(\v2\x1c.cozy.worker.v1.ResourceCapsR\fresourceCaps\x12V\n" +
	"\x14publication_contract\x18\x04 \x01(\v2#.cozy.worker.v1.PublicationContractR\x13publicationContract\x12.\n" +
	"\x13reclaim_on_terminal\x18\x05 \x01(\bR\x11reclaimOnTerminal\x12\x1b\n" +
	"\tgpu_count\x18\x06 \x01(\rR\bgpuCount\"\xa1\x05\n" +
	"\x06Report\x12\x1d\n" +
	"\n" +
	"session_id\x18\x01 \x01(\tR\tsessionId\x121\n" +
	"\x14executor_incarnation\x18\x02 \x01(\x04R\x13executorIncarnation\x12)\n" +
	"\x10applied_revision\x18\x03 \x01(\x04R\x0fappliedRevision\x12>\n" +
	"\fintake_state\x18\x04 \x01(\x0e2\x1b.cozy.worker.v1.IntakeStateR\vintakeState\x12)\n" +
	"\x10credential_epoch\x18\x05 \x01(\x04R\x0fcredentialEpoch\x12'\n" +
	"\x0freadiness_epoch\x18\x06 \x01(\x04R\x0ereadinessEpoch\x12F\n" +
	"\x0factive_attempts\x18\t \x03(\v2\x1d.cozy.worker.v1.ActiveAttemptR\x0eactiveAttempts\x12-\n" +
	"\x06faults\x18\n" +
	" \x03(\v2\x15.cozy.worker.v1.FaultR\x06faults\x129\n" +
	"\bactivity\x18\v \x03(\v2\x1d.cozy.worker.v1.ActivityEventR\bactivity\x12,\n" +
	"\x12applied_wire_minor\x18\f \x01(\rR\x10appliedWireMinor\x12L\n" +
	"\x10serving_capacity\x18\r \x01(\v2\x1f.cozy.worker.v1.ServingCapacityH\x00R\x0fservingCapacity\x12@\n" +
	"\fjob_capacity\x18\x0e \x01(\v2\x1b.cozy.worker.v1.JobCapacityH\x00R\vjobCapacityB\n" +
	"\n" +
	"\bcapacityJ\x04\b\a\x10\bJ\x04\b\b\x10\t\"g\n" +
	"\rActivityEvent\x12\x10\n" +
	"\x03seq\x18\x01 \x01(\x04R\x03seq\x12\x12\n" +
	"\x04kind\x18\x02 \x01(\tR\x04kind\x12\x12\n" +
	"\x04step\x18\x03 \x01(\tR\x04step\x12\x1c\n" +
	"\n" +
	"at_unix_ms\x18\x04 \x01(\x04R\batUnixMs\"\xae\x02\n" +
	"\fStartAttempt\x12\x1d\n" +
	"\n" +
	"session_id\x18\x01 \x01(\tR\tsessionId\x121\n" +
	"\x14executor_incarnation\x18\x02 \x01(\x04R\x13executorIncarnation\x12\x1d\n" +
	"\n" +
	"request_id\x18\x03 \x01(\tR\trequestId\x12\x18\n" +
	"\aattempt\x18\x04 \x01(\x04R\aattempt\x12(\n" +
	"\x10exec_spec_digest\x18\x06 \x01(\fR\x0eexecSpecDigest\x123\n" +
	"\x05grant\x18\a \x01(\v2\x1d.cozy.worker.v1.DeliveryGrantR\x05grant\x12.\n" +
	"\x13exec_spec_canonical\x18\b \x01(\fR\x11execSpecCanonicalJ\x04\b\x05\x10\x06\"\xdb\x02\n" +
	"\x0fAttemptAccepted\x12\x1d\n" +
	"\n" +
	"session_id\x18\x01 \x01(\tR\tsessionId\x121\n" +
	"\x14executor_incarnation\x18\x02 \x01(\x04R\x13executorIncarnation\x12\x1d\n" +
	"\n" +
	"request_id\x18\x03 \x01(\tR\trequestId\x12\x18\n" +
	"\aattempt\x18\x04 \x01(\x04R\aattempt\x12(\n" +
	"\x10exec_spec_digest\x18\x05 \x01(\fR\x0eexecSpecDigest\x12\x1f\n" +
	"\vplan_digest\x18\x06 \x01(\fR\n" +
	"planDigest\x12:\n" +
	"\x19model_construction_digest\x18\a \x01(\fR\x17modelConstructionDigest\x126\n" +
	"\x04plan\x18\b \x01(\v2\".cozy.worker.v1.AttemptPlanSummaryR\x04plan\"\xaa\x02\n" +
	"\x12AttemptPlanSummary\x12\x1a\n" +
	"\bdelivery\x18\x01 \x01(\tR\bdelivery\x12(\n" +
	"\x0fmaterialization\x18\x02 \x01(\tR\x0fmaterialization\x12#\n" +
	"\rcompute_dtype\x18\x03 \x01(\tR\fcomputeDtype\x12\x1c\n" +
	"\tplacement\x18\x04 \x01(\tR\tplacement\x12.\n" +
	"\x13reserved_vram_bytes\x18\x05 \x01(\x04R\x11reservedVramBytes\x12.\n" +
	"\x13reserved_host_bytes\x18\x06 \x01(\x04R\x11reservedHostBytes\x12+\n" +
	"\x11quantified_choice\x18\a \x01(\tR\x10quantifiedChoice\"\x95\x02\n" +
	"\rCancelAttempt\x12\x1d\n" +
	"\n" +
	"session_id\x18\x01 \x01(\tR\tsessionId\x121\n" +
	"\x14executor_incarnation\x18\x02 \x01(\x04R\x13executorIncarnation\x12\x1d\n" +
	"\n" +
	"request_id\x18\x03 \x01(\tR\trequestId\x12\x18\n" +
	"\aattempt\x18\x04 \x01(\x04R\aattempt\x124\n" +
	"\x06reason\x18\x05 \x01(\x0e2\x1c.cozy.worker.v1.CancelReasonR\x06reason\x12\x19\n" +
	"\bgrace_ms\x18\x06 \x01(\x04R\agraceMs\x12(\n" +
	"\x10exec_spec_digest\x18\a \x01(\fR\x0eexecSpecDigest\"\xe9\x02\n" +
	"\x0fAttemptTerminal\x12\x1d\n" +
	"\n" +
	"session_id\x18\x01 \x01(\tR\tsessionId\x121\n" +
	"\x14executor_incarnation\x18\x02 \x01(\x04R\x13executorIncarnation\x12\x1d\n" +
	"\n" +
	"request_id\x18\x03 \x01(\tR\trequestId\x12\x18\n" +
	"\aattempt\x18\x04 \x01(\x04R\aattempt\x12(\n" +
	"\x10exec_spec_digest\x18\x05 \x01(\fR\x0eexecSpecDigest\x12\x1f\n" +
	"\vterminal_id\x18\f \x01(\tR\n" +
	"terminalId\x12'\n" +
	"\x0fterminal_digest\x18\r \x01(\fR\x0eterminalDigest\x12-\n" +
	"\x12terminal_canonical\x18\x0f \x01(\fR\x11terminalCanonicalJ\x04\b\x06\x10\aJ\x04\b\a\x10\bJ\x04\b\b\x10\tJ\x04\b\t\x10\n" +
	"J\x04\b\n" +
	"\x10\vJ\x04\b\v\x10\fJ\x04\b\x0e\x10\x0f\"\x82\x04\n" +
	"\fTerminalBody\x12\x1d\n" +
	"\n" +
	"request_id\x18\x01 \x01(\tR\trequestId\x12\x18\n" +
	"\aattempt\x18\x02 \x01(\x04R\aattempt\x12(\n" +
	"\x10exec_spec_digest\x18\x03 \x01(\fR\x0eexecSpecDigest\x126\n" +
	"\x06status\x18\x04 \x01(\x0e2\x1e.cozy.worker.v1.TerminalStatusR\x06status\x12G\n" +
	"\x0foutput_manifest\x18\x05 \x01(\v2\x1e.cozy.worker.v1.OutputManifestR\x0eoutputManifest\x128\n" +
	"\ametrics\x18\x06 \x01(\v2\x1e.cozy.worker.v1.AttemptMetricsR\ametrics\x12D\n" +
	"\rtriage_bundle\x18\a \x01(\v2\x1f.cozy.worker.v1.TriageBundleRefR\ftriageBundle\x12!\n" +
	"\fsafe_message\x18\b \x01(\tR\vsafeMessage\x123\n" +
	"\x05cause\x18\t \x01(\v2\x1d.cozy.worker.v1.TerminalCauseR\x05cause\x126\n" +
	"\x06result\x18\n" +
	" \x01(\v2\x1e.cozy.worker.v1.ResultEnvelopeR\x06result\"\xc9\x02\n" +
	"\x0eResultEnvelope\x120\n" +
	"\x14result_schema_digest\x18\x01 \x01(\fR\x12resultSchemaDigest\x12#\n" +
	"\rinline_result\x18\x02 \x01(\fR\finlineResult\x12<\n" +
	"\vresult_blob\x18\x03 \x01(\v2\x1b.cozy.worker.v1.OutputEntryR\n" +
	"resultBlob\x12?\n" +
	"\vadjustments\x18\x04 \x03(\v2\x1d.cozy.worker.v1.AdjustmentRowR\vadjustments\x12%\n" +
	"\x0echeckpoint_ref\x18\x05 \x01(\tR\rcheckpointRef\x12:\n" +
	"\badapters\x18\x06 \x03(\v2\x1e.cozy.worker.v1.AppliedAdapterR\badapters\"u\n" +
	"\rAdjustmentRow\x12\x14\n" +
	"\x05field\x18\x01 \x01(\tR\x05field\x12\x1c\n" +
	"\trequested\x18\x02 \x01(\tR\trequested\x12\x18\n" +
	"\aapplied\x18\x03 \x01(\tR\aapplied\x12\x16\n" +
	"\x06reason\x18\x04 \x01(\tR\x06reason\"<\n" +
	"\x0eAppliedAdapter\x12\x10\n" +
	"\x03ref\x18\x01 \x01(\tR\x03ref\x12\x12\n" +
	"\x04kind\x18\x03 \x01(\tR\x04kindJ\x04\b\x02\x10\x03\"\xcc\x01\n" +
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
	"\x0eevidence_class\x18\x06 \x01(\tR\revidenceClass\"\x8c\x02\n" +
	"\vTerminalAck\x12\x1d\n" +
	"\n" +
	"session_id\x18\x01 \x01(\tR\tsessionId\x121\n" +
	"\x14executor_incarnation\x18\x02 \x01(\x04R\x13executorIncarnation\x12\x1d\n" +
	"\n" +
	"request_id\x18\x03 \x01(\tR\trequestId\x12\x18\n" +
	"\aattempt\x18\x04 \x01(\x04R\aattempt\x12(\n" +
	"\x10exec_spec_digest\x18\x05 \x01(\fR\x0eexecSpecDigest\x12\x1f\n" +
	"\vterminal_id\x18\x06 \x01(\tR\n" +
	"terminalId\x12'\n" +
	"\x0fterminal_digest\x18\a \x01(\fR\x0eterminalDigest\"\xd9\x02\n" +
	"\x14JobCheckpointRequest\x12\x1d\n" +
	"\n" +
	"session_id\x18\x01 \x01(\tR\tsessionId\x121\n" +
	"\x14executor_incarnation\x18\x02 \x01(\x04R\x13executorIncarnation\x12\x1d\n" +
	"\n" +
	"request_id\x18\x03 \x01(\tR\trequestId\x12\x18\n" +
	"\aattempt\x18\x04 \x01(\x04R\aattempt\x12#\n" +
	"\roperation_key\x18\x05 \x01(\tR\foperationKey\x12\x1f\n" +
	"\vlogical_key\x18\x06 \x01(\tR\n" +
	"logicalKey\x12%\n" +
	"\x0econtent_digest\x18\a \x01(\fR\rcontentDigest\x12\x10\n" +
	"\x03seq\x18\b \x01(\x04R\x03seq\x127\n" +
	"\bartifact\x18\t \x01(\v2\x1b.cozy.worker.v1.OutputEntryR\bartifact\"\xa1\x03\n" +
	"\x14JobCheckpointReceipt\x12\x1d\n" +
	"\n" +
	"session_id\x18\x01 \x01(\tR\tsessionId\x121\n" +
	"\x14executor_incarnation\x18\x02 \x01(\x04R\x13executorIncarnation\x12\x1d\n" +
	"\n" +
	"request_id\x18\x03 \x01(\tR\trequestId\x12\x18\n" +
	"\aattempt\x18\x04 \x01(\x04R\aattempt\x12#\n" +
	"\roperation_key\x18\x05 \x01(\tR\foperationKey\x12\x1f\n" +
	"\vlogical_key\x18\x06 \x01(\tR\n" +
	"logicalKey\x12%\n" +
	"\x0econtent_digest\x18\a \x01(\fR\rcontentDigest\x12\x1d\n" +
	"\n" +
	"receipt_id\x18\b \x01(\tR\treceiptId\x12;\n" +
	"\aoutcome\x18\t \x01(\x0e2!.cozy.worker.v1.CheckpointOutcomeR\aoutcome\x125\n" +
	"\x05fault\x18\n" +
	" \x01(\v2\x1f.cozy.worker.v1.CheckpointFaultR\x05fault\"\x9a\x01\n" +
	"\x0fCheckpointFault\x127\n" +
	"\x04code\x18\x01 \x01(\x0e2#.cozy.worker.v1.CheckpointFaultCodeR\x04code\x12\x16\n" +
	"\x06detail\x18\x02 \x01(\tR\x06detail\x126\n" +
	"\x17recorded_content_digest\x18\x03 \x01(\fR\x15recordedContentDigest\"\x88\x02\n" +
	"\x10JobCheckpointAck\x12\x1d\n" +
	"\n" +
	"session_id\x18\x01 \x01(\tR\tsessionId\x121\n" +
	"\x14executor_incarnation\x18\x02 \x01(\x04R\x13executorIncarnation\x12\x1d\n" +
	"\n" +
	"request_id\x18\x03 \x01(\tR\trequestId\x12\x18\n" +
	"\aattempt\x18\x04 \x01(\x04R\aattempt\x12#\n" +
	"\roperation_key\x18\x05 \x01(\tR\foperationKey\x12\x1d\n" +
	"\n" +
	"receipt_id\x18\x06 \x01(\tR\treceiptId\x12%\n" +
	"\x0econtent_digest\x18\a \x01(\fR\rcontentDigest\"\xe5\x01\n" +
	"\x0fAttemptProgress\x12\x1d\n" +
	"\n" +
	"session_id\x18\x01 \x01(\tR\tsessionId\x121\n" +
	"\x14executor_incarnation\x18\x02 \x01(\x04R\x13executorIncarnation\x12\x1d\n" +
	"\n" +
	"request_id\x18\x03 \x01(\tR\trequestId\x12\x18\n" +
	"\aattempt\x18\x04 \x01(\x04R\aattempt\x12\x10\n" +
	"\x03seq\x18\x05 \x01(\x04R\x03seq\x12!\n" +
	"\fcontent_type\x18\x06 \x01(\tR\vcontentType\x12\x12\n" +
	"\x04data\x18\a \x01(\fR\x04data\"\xfe\x01\n" +
	"\rPurgeArtifact\x12\x1d\n" +
	"\n" +
	"session_id\x18\x01 \x01(\tR\tsessionId\x121\n" +
	"\x14executor_incarnation\x18\x02 \x01(\x04R\x13executorIncarnation\x12\x19\n" +
	"\bpurge_id\x18\x03 \x01(\tR\apurgeId\x12*\n" +
	"\x11purge_body_digest\x18\x04 \x01(\fR\x0fpurgeBodyDigest\x120\n" +
	"\x14purge_body_canonical\x18\v \x01(\fR\x12purgeBodyCanonicalJ\x04\b\x05\x10\x06J\x04\b\x06\x10\aJ\x04\b\a\x10\bJ\x04\b\b\x10\tJ\x04\b\t\x10\n" +
	"J\x04\b\n" +
	"\x10\v\"\x98\x02\n" +
	"\tPurgeBody\x12\x1b\n" +
	"\ttenant_id\x18\x01 \x01(\tR\btenantId\x12;\n" +
	"\bsubjects\x18\x02 \x03(\v2\x1f.cozy.worker.v1.ArtifactSubjectR\bsubjects\x121\n" +
	"\x14tombstone_generation\x18\x03 \x01(\x04R\x13tombstoneGeneration\x12,\n" +
	"\x12target_instance_id\x18\x04 \x01(\tR\x10targetInstanceId\x12+\n" +
	"\x11target_generation\x18\x05 \x01(\x04R\x10targetGeneration\x12#\n" +
	"\rdeadline_unix\x18\x06 \x01(\x04R\fdeadlineUnix\"\xb7\x03\n" +
	"\x13PurgeArtifactReport\x12\x1d\n" +
	"\n" +
	"session_id\x18\x01 \x01(\tR\tsessionId\x121\n" +
	"\x14executor_incarnation\x18\x02 \x01(\x04R\x13executorIncarnation\x12\x19\n" +
	"\bpurge_id\x18\x03 \x01(\tR\apurgeId\x12*\n" +
	"\x11purge_body_digest\x18\x04 \x01(\fR\x0fpurgeBodyDigest\x12@\n" +
	"\x1capplied_tombstone_generation\x18\b \x01(\x04R\x1aappliedTombstoneGeneration\x129\n" +
	"\apending\x18\t \x01(\v2\x1d.cozy.worker.v1.PendingLeasesH\x00R\apending\x129\n" +
	"\x06absent\x18\n" +
	" \x01(\v2\x1f.cozy.worker.v1.AbsenceEvidenceH\x00R\x06absent\x122\n" +
	"\x05fault\x18\v \x01(\v2\x1a.cozy.worker.v1.PurgeFaultH\x00R\x05faultB\t\n" +
	"\aoutcomeJ\x04\b\x05\x10\x06J\x04\b\x06\x10\aJ\x04\b\a\x10\b\"\xbd\x02\n" +
	"\rExecutionSpec\x12.\n" +
	"\x13endpoint_release_id\x18\x01 \x01(\tR\x11endpointReleaseId\x12!\n" +
	"\fimage_digest\x18\x02 \x01(\tR\vimageDigest\x12#\n" +
	"\rconfig_digest\x18\x03 \x01(\tR\fconfigDigest\x12(\n" +
	"\x10deadline_unix_ms\x18\x06 \x01(\x04R\x0edeadlineUnixMs\x12@\n" +
	"\aserving\x18\a \x01(\v2$.cozy.worker.v1.ServingExecutionSpecH\x00R\aserving\x124\n" +
	"\x03job\x18\b \x01(\v2 .cozy.worker.v1.JobExecutionSpecH\x00R\x03jobB\x06\n" +
	"\x04specJ\x04\b\x04\x10\x05J\x04\b\x05\x10\x06\"\x81\x01\n" +
	"\x14ServingExecutionSpec\x12;\n" +
	"\x1aentrypoint_binding_plan_id\x18\x01 \x01(\tR\x17entrypointBindingPlanId\x12,\n" +
	"\x12attempt_binding_id\x18\x02 \x01(\tR\x10attemptBindingId\"\xb1\x01\n" +
	"\x10JobExecutionSpec\x12\x19\n" +
	"\bbuild_id\x18\x01 \x01(\tR\abuildId\x12*\n" +
	"\x11job_descriptor_id\x18\x02 \x01(\tR\x0fjobDescriptorId\x12V\n" +
	"\x14publication_contract\x18\x03 \x01(\v2#.cozy.worker.v1.PublicationContractR\x13publicationContract\"\x8b\x02\n" +
	"\rDeliveryGrant\x12:\n" +
	"\n" +
	"credential\x18\x01 \x01(\v2\x1a.cozy.worker.v1.CredentialR\n" +
	"credential\x12\"\n" +
	"\rfile_base_url\x18\x02 \x01(\tR\vfileBaseUrl\x12&\n" +
	"\x0fexpires_at_unix\x18\x03 \x01(\x04R\rexpiresAtUnix\x125\n" +
	"\x06inputs\x18\x04 \x03(\v2\x1d.cozy.worker.v1.InputLocationR\x06inputs\x12;\n" +
	"\aoutputs\x18\x05 \x03(\v2!.cozy.worker.v1.OutputDestinationR\aoutputs\"\x9f\x01\n" +
	"\rInputLocation\x12\x16\n" +
	"\x06digest\x18\x01 \x01(\fR\x06digest\x12\x10\n" +
	"\x03url\x18\x02 \x01(\tR\x03url\x12\x16\n" +
	"\x06length\x18\x03 \x01(\x04R\x06length\x12\x19\n" +
	"\binput_id\x18\x04 \x01(\tR\ainputId\x12\x14\n" +
	"\x05order\x18\x05 \x01(\rR\x05order\x12\x1b\n" +
	"\tkind_mime\x18\x06 \x01(\tR\bkindMime\"_\n" +
	"\x11OutputDestination\x12\x1b\n" +
	"\toutput_id\x18\x01 \x01(\tR\boutputId\x12\x10\n" +
	"\x03url\x18\x02 \x01(\tR\x03url\x12\x1b\n" +
	"\tmax_bytes\x18\x03 \x01(\x04R\bmaxBytes\"\x8f\x01\n" +
	"\n" +
	"Credential\x12\x16\n" +
	"\x06issuer\x18\x01 \x01(\tR\x06issuer\x12\x15\n" +
	"\x06key_id\x18\x02 \x01(\tR\x05keyId\x12\x14\n" +
	"\x05epoch\x18\x03 \x01(\x04R\x05epoch\x12&\n" +
	"\x0fexpires_at_unix\x18\x04 \x01(\x04R\rexpiresAtUnix\x12\x14\n" +
	"\x05token\x18\x05 \x01(\fR\x05token\"\x98\x01\n" +
	"\rArtifactGrant\x12\"\n" +
	"\rfile_base_url\x18\x01 \x01(\tR\vfileBaseUrl\x12&\n" +
	"\x0fexpires_at_unix\x18\x02 \x01(\x04R\rexpiresAtUnix\x12;\n" +
	"\bsubjects\x18\x03 \x03(\v2\x1f.cozy.worker.v1.ArtifactSubjectR\bsubjects\"Z\n" +
	"\x0fArtifactSubject\x12/\n" +
	"\x04kind\x18\x01 \x01(\x0e2\x1b.cozy.worker.v1.SubjectKindR\x04kind\x12\x16\n" +
	"\x06digest\x18\x02 \x01(\fR\x06digest\"g\n" +
	"\vDrainPolicy\x12\x1f\n" +
	"\vdeadline_ms\x18\x01 \x01(\x04R\n" +
	"deadlineMs\x127\n" +
	"\x18force_cancel_at_deadline\x18\x02 \x01(\bR\x15forceCancelAtDeadline\"\xa1\x01\n" +
	"\fResourceCaps\x12!\n" +
	"\fgpu_required\x18\x01 \x01(\bR\vgpuRequired\x12$\n" +
	"\x0emax_vram_bytes\x18\x02 \x01(\x04R\fmaxVramBytes\x12\"\n" +
	"\rmax_rss_bytes\x18\x03 \x01(\x04R\vmaxRssBytes\x12$\n" +
	"\x0emax_disk_bytes\x18\x04 \x01(\x04R\fmaxDiskBytes\"f\n" +
	"\x13PublicationContract\x124\n" +
	"\aoutputs\x18\x01 \x03(\v2\x1a.cozy.worker.v1.OutputSpecR\aoutputs\x12\x19\n" +
	"\bgrant_id\x18\x02 \x01(\tR\agrantId\"c\n" +
	"\n" +
	"OutputSpec\x12\x1b\n" +
	"\toutput_id\x18\x01 \x01(\tR\boutputId\x12\x1b\n" +
	"\tmime_type\x18\x02 \x01(\tR\bmimeType\x12\x1b\n" +
	"\tmax_bytes\x18\x03 \x01(\x04R\bmaxBytes\"\xc9\x05\n" +
	"\x0fWorkerResources\x12\x1b\n" +
	"\tgpu_count\x18\x01 \x01(\rR\bgpuCount\x12\x19\n" +
	"\bgpu_name\x18\x02 \x01(\tR\agpuName\x12\x15\n" +
	"\x06gpu_sm\x18\x03 \x01(\rR\x05gpuSm\x12(\n" +
	"\x10vram_total_bytes\x18\x04 \x01(\x04R\x0evramTotalBytes\x12%\n" +
	"\x0edriver_version\x18\x05 \x01(\tR\rdriverVersion\x12!\n" +
	"\fcuda_version\x18\x06 \x01(\tR\vcudaVersion\x12#\n" +
	"\rtorch_version\x18\a \x01(\tR\ftorchVersion\x12\x1a\n" +
	"\bplatform\x18\b \x01(\tR\bplatform\x12/\n" +
	"\x14host_ram_total_bytes\x18\t \x01(\x04R\x11hostRamTotalBytes\x12\x1d\n" +
	"\n" +
	"vcpu_count\x18\n" +
	" \x01(\rR\tvcpuCount\x12(\n" +
	"\x10disk_total_bytes\x18\v \x01(\x04R\x0ediskTotalBytes\x122\n" +
	"\x15pinned_capacity_bytes\x18\f \x01(\x04R\x13pinnedCapacityBytes\x12&\n" +
	"\x0fpower_cap_watts\x18\r \x01(\rR\rpowerCapWatts\x12\x1f\n" +
	"\vmig_profile\x18\x0e \x01(\tR\n" +
	"migProfile\x12\"\n" +
	"\finterconnect\x18\x0f \x01(\tR\finterconnect\x12\x1f\n" +
	"\vpeer_access\x18\x10 \x01(\bR\n" +
	"peerAccess\x12*\n" +
	"\x11h2d_gbps_measured\x18\x11 \x01(\rR\x0fh2dGbpsMeasured\x12*\n" +
	"\x11d2h_gbps_measured\x18\x12 \x01(\rR\x0fd2hGbpsMeasured\x12\x1e\n" +
	"\n" +
	"unreadable\x18\x13 \x03(\tR\n" +
	"unreadable\"\xd3\x01\n" +
	"\x0fServingCapacity\x12H\n" +
	"!ready_entrypoint_binding_plan_ids\x18\x01 \x03(\tR\x1dreadyEntrypointBindingPlanIds\x12N\n" +
	"$loadable_entrypoint_binding_plan_ids\x18\x02 \x03(\tR loadableEntrypointBindingPlanIds\x12&\n" +
	"\x0ffree_vram_bytes\x18\x03 \x01(\x04R\rfreeVramBytes\"\xaa\x01\n" +
	"\vJobCapacity\x12$\n" +
	"\x0ejobs_in_flight\x18\x01 \x01(\rR\fjobsInFlight\x12%\n" +
	"\x0ejobs_available\x18\x02 \x01(\rR\rjobsAvailable\x12&\n" +
	"\x0ffree_vram_bytes\x18\x03 \x01(\x04R\rfreeVramBytes\x12&\n" +
	"\x0ffree_disk_bytes\x18\x04 \x01(\x04R\rfreeDiskBytes\"\xd7\x01\n" +
	"\rActiveAttempt\x12\x1d\n" +
	"\n" +
	"request_id\x18\x01 \x01(\tR\trequestId\x12\x18\n" +
	"\aattempt\x18\x02 \x01(\x04R\aattempt\x12/\n" +
	"\x04kind\x18\x03 \x01(\x0e2\x1b.cozy.worker.v1.AttemptKindR\x04kind\x122\n" +
	"\x05state\x18\x04 \x01(\x0e2\x1c.cozy.worker.v1.AttemptStateR\x05state\x12(\n" +
	"\x10exec_spec_digest\x18\x05 \x01(\fR\x0eexecSpecDigest\"\x80\x01\n" +
	"\x05Fault\x12-\n" +
	"\x04kind\x18\x01 \x01(\x0e2\x19.cozy.worker.v1.FaultKindR\x04kind\x12\x18\n" +
	"\asubject\x18\x02 \x01(\tR\asubject\x12\x16\n" +
	"\x06reason\x18\x03 \x01(\tR\x06reason\x12\x16\n" +
	"\x06detail\x18\x04 \x01(\tR\x06detail\"n\n" +
	"\x0eOutputManifest\x12\x1f\n" +
	"\vmanifest_id\x18\x01 \x01(\tR\n" +
	"manifestId\x125\n" +
	"\aoutputs\x18\x03 \x03(\v2\x1b.cozy.worker.v1.OutputEntryR\aoutputsJ\x04\b\x02\x10\x03\"w\n" +
	"\vOutputEntry\x12\x1b\n" +
	"\toutput_id\x18\x01 \x01(\tR\boutputId\x12\x16\n" +
	"\x06digest\x18\x02 \x01(\fR\x06digest\x12\x16\n" +
	"\x06length\x18\x03 \x01(\x04R\x06length\x12\x1b\n" +
	"\tmime_type\x18\x04 \x01(\tR\bmimeType\"\xc0\x03\n" +
	"\x0eAttemptMetrics\x12\x1d\n" +
	"\n" +
	"runtime_ms\x18\x01 \x01(\x04R\truntimeMs\x12\x19\n" +
	"\bqueue_ms\x18\x02 \x01(\x04R\aqueueMs\x12&\n" +
	"\x0fpeak_vram_bytes\x18\x03 \x01(\x04R\rpeakVramBytes\x12'\n" +
	"\x10rss_at_end_bytes\x18\x04 \x01(\x04R\rrssAtEndBytes\x12!\n" +
	"\foutput_count\x18\x05 \x01(\rR\voutputCount\x12!\n" +
	"\finput_tokens\x18\x06 \x01(\x04R\vinputTokens\x12#\n" +
	"\routput_tokens\x18\a \x01(\x04R\foutputTokens\x12&\n" +
	"\x0fdevice_lease_ms\x18\b \x01(\x04R\rdeviceLeaseMs\x12\x1b\n" +
	"\tgpu_count\x18\t \x01(\rR\bgpuCount\x12\x1d\n" +
	"\n" +
	"handler_ms\x18\n" +
	" \x01(\x04R\thandlerMs\x12'\n" +
	"\x0ffinalization_ms\x18\v \x01(\x04R\x0efinalizationMs\x12+\n" +
	"\x11unverified_fields\x18\f \x03(\tR\x10unverifiedFields\"\x80\x01\n" +
	"\x0fTriageBundleRef\x12\x1d\n" +
	"\n" +
	"subject_id\x18\x01 \x01(\tR\tsubjectId\x120\n" +
	"\x14write_receipt_digest\x18\x04 \x01(\fR\x12writeReceiptDigest\x12\x16\n" +
	"\x06length\x18\x03 \x01(\x04R\x06lengthJ\x04\b\x02\x10\x03\",\n" +
	"\rPendingLeases\x12\x1b\n" +
	"\tlease_ids\x18\x01 \x03(\tR\bleaseIds\"[\n" +
	"\x0fAbsenceEvidence\x12\x1f\n" +
	"\vevidence_id\x18\x01 \x01(\tR\n" +
	"evidenceId\x12'\n" +
	"\x0fevidence_digest\x18\x02 \x01(\fR\x0eevidenceDigest\"X\n" +
	"\n" +
	"PurgeFault\x122\n" +
	"\x04code\x18\x01 \x01(\x0e2\x1e.cozy.worker.v1.PurgeFaultCodeR\x04code\x12\x16\n" +
	"\x06detail\x18\x02 \x01(\tR\x06detail*O\n" +
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
	"\x16CANCEL_REASON_DEADLINE\x10\x05*\xb0\x02\n" +
	"\x11RegisterRejection\x12\"\n" +
	"\x1eREGISTER_REJECTION_UNSPECIFIED\x10\x00\x12&\n" +
	"\"REGISTER_REJECTION_UNAUTHENTICATED\x10\x01\x12(\n" +
	"$REGISTER_REJECTION_SESSION_COLLISION\x10\x02\x12$\n" +
	" REGISTER_REJECTION_STALE_SESSION\x10\x03\x12)\n" +
	"%REGISTER_REJECTION_WORKER_ID_MISMATCH\x10\x04\x12*\n" +
	"&REGISTER_REJECTION_RELEASE_ID_MISMATCH\x10\x05\x12(\n" +
	"$REGISTER_REJECTION_UNSUPPORTED_MINOR\x10\x06*\x82\x03\n" +
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
	"*\xf2\x01\n" +
	"\vSubjectKind\x12\x1c\n" +
	"\x18SUBJECT_KIND_UNSPECIFIED\x10\x00\x12\x17\n" +
	"\x13SUBJECT_KIND_OBJECT\x10\x01\x12\x19\n" +
	"\x15SUBJECT_KIND_SNAPSHOT\x10\x02\x12\x1e\n" +
	"\x1aSUBJECT_KIND_TENSOR_HEADER\x10\x03\x12\x1b\n" +
	"\x17SUBJECT_KIND_CHECKPOINT\x10\x04\x12!\n" +
	"\x1dSUBJECT_KIND_ADAPTER_ARTIFACT\x10\x05\x12\x17\n" +
	"\x13SUBJECT_KIND_LAYOUT\x10\x06\x12\x18\n" +
	"\x14SUBJECT_KIND_RELEASE\x10\a*\x86\x03\n" +
	"\x0ePurgeFaultCode\x12 \n" +
	"\x1cPURGE_FAULT_CODE_UNSPECIFIED\x10\x00\x12/\n" +
	"+PURGE_FAULT_CODE_STALE_TOMBSTONE_GENERATION\x10\x01\x12,\n" +
	"(PURGE_FAULT_CODE_STALE_WORKER_GENERATION\x10\x02\x12(\n" +
	"$PURGE_FAULT_CODE_CHANGED_BODY_REPLAY\x10\x03\x12)\n" +
	"%PURGE_FAULT_CODE_CROSS_TENANT_SUBJECT\x10\x04\x12+\n" +
	"'PURGE_FAULT_CODE_LIVE_UNRELATED_BINDING\x10\x05\x12#\n" +
	"\x1fPURGE_FAULT_CODE_ERASURE_FAILED\x10\x06\x12$\n" +
	" PURGE_FAULT_CODE_UNKNOWN_SUBJECT\x10\a\x12&\n" +
	"\"PURGE_FAULT_CODE_DEADLINE_EXCEEDED\x10\b*\x9e\x02\n" +
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
	"$CHECKPOINT_FAULT_CODE_QUOTA_EXCEEDED\x10\x052\xf0\x01\n" +
	"\x06Worker\x12P\n" +
	"\aControl\x12\x1d.cozy.worker.v1.WorkerMessage\x1a\".cozy.worker.v1.CoordinatorMessage(\x010\x01\x12K\n" +
	"\x0eStreamProgress\x12\x1f.cozy.worker.v1.AttemptProgress\x1a\x16.google.protobuf.Empty(\x01\x12G\n" +
	"\x05Purge\x12\x1c.cozy.worker.v1.PurgeMessage\x1a\x1c.cozy.worker.v1.PurgeMessage(\x010\x01B4\n" +
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

var file_cozy_worker_v1_worker_proto_enumTypes = make([]protoimpl.EnumInfo, 15)
var file_cozy_worker_v1_worker_proto_msgTypes = make([]protoimpl.MessageInfo, 56)
var file_cozy_worker_v1_worker_proto_goTypes = []any{
	(Posture)(0),                 // 0: cozy.worker.v1.Posture
	(IntakeState)(0),             // 1: cozy.worker.v1.IntakeState
	(AttemptKind)(0),             // 2: cozy.worker.v1.AttemptKind
	(AttemptState)(0),            // 3: cozy.worker.v1.AttemptState
	(TerminalStatus)(0),          // 4: cozy.worker.v1.TerminalStatus
	(CauseCode)(0),               // 5: cozy.worker.v1.CauseCode
	(CauseOrigin)(0),             // 6: cozy.worker.v1.CauseOrigin
	(CancelReason)(0),            // 7: cozy.worker.v1.CancelReason
	(RegisterRejection)(0),       // 8: cozy.worker.v1.RegisterRejection
	(FaultKind)(0),               // 9: cozy.worker.v1.FaultKind
	(SubjectKind)(0),             // 10: cozy.worker.v1.SubjectKind
	(PurgeFaultCode)(0),          // 11: cozy.worker.v1.PurgeFaultCode
	(BootFailureReason)(0),       // 12: cozy.worker.v1.BootFailureReason
	(CheckpointOutcome)(0),       // 13: cozy.worker.v1.CheckpointOutcome
	(CheckpointFaultCode)(0),     // 14: cozy.worker.v1.CheckpointFaultCode
	(*WorkerMessage)(nil),        // 15: cozy.worker.v1.WorkerMessage
	(*CoordinatorMessage)(nil),   // 16: cozy.worker.v1.CoordinatorMessage
	(*PurgeMessage)(nil),         // 17: cozy.worker.v1.PurgeMessage
	(*Register)(nil),             // 18: cozy.worker.v1.Register
	(*BootFailure)(nil),          // 19: cozy.worker.v1.BootFailure
	(*RegisterAck)(nil),          // 20: cozy.worker.v1.RegisterAck
	(*Directive)(nil),            // 21: cozy.worker.v1.Directive
	(*ServingDirective)(nil),     // 22: cozy.worker.v1.ServingDirective
	(*JobDirective)(nil),         // 23: cozy.worker.v1.JobDirective
	(*Report)(nil),               // 24: cozy.worker.v1.Report
	(*ActivityEvent)(nil),        // 25: cozy.worker.v1.ActivityEvent
	(*StartAttempt)(nil),         // 26: cozy.worker.v1.StartAttempt
	(*AttemptAccepted)(nil),      // 27: cozy.worker.v1.AttemptAccepted
	(*AttemptPlanSummary)(nil),   // 28: cozy.worker.v1.AttemptPlanSummary
	(*CancelAttempt)(nil),        // 29: cozy.worker.v1.CancelAttempt
	(*AttemptTerminal)(nil),      // 30: cozy.worker.v1.AttemptTerminal
	(*TerminalBody)(nil),         // 31: cozy.worker.v1.TerminalBody
	(*ResultEnvelope)(nil),       // 32: cozy.worker.v1.ResultEnvelope
	(*AdjustmentRow)(nil),        // 33: cozy.worker.v1.AdjustmentRow
	(*AppliedAdapter)(nil),       // 34: cozy.worker.v1.AppliedAdapter
	(*TerminalCause)(nil),        // 35: cozy.worker.v1.TerminalCause
	(*ResourceShortfall)(nil),    // 36: cozy.worker.v1.ResourceShortfall
	(*TerminalAck)(nil),          // 37: cozy.worker.v1.TerminalAck
	(*JobCheckpointRequest)(nil), // 38: cozy.worker.v1.JobCheckpointRequest
	(*JobCheckpointReceipt)(nil), // 39: cozy.worker.v1.JobCheckpointReceipt
	(*CheckpointFault)(nil),      // 40: cozy.worker.v1.CheckpointFault
	(*JobCheckpointAck)(nil),     // 41: cozy.worker.v1.JobCheckpointAck
	(*AttemptProgress)(nil),      // 42: cozy.worker.v1.AttemptProgress
	(*PurgeArtifact)(nil),        // 43: cozy.worker.v1.PurgeArtifact
	(*PurgeBody)(nil),            // 44: cozy.worker.v1.PurgeBody
	(*PurgeArtifactReport)(nil),  // 45: cozy.worker.v1.PurgeArtifactReport
	(*ExecutionSpec)(nil),        // 46: cozy.worker.v1.ExecutionSpec
	(*ServingExecutionSpec)(nil), // 47: cozy.worker.v1.ServingExecutionSpec
	(*JobExecutionSpec)(nil),     // 48: cozy.worker.v1.JobExecutionSpec
	(*DeliveryGrant)(nil),        // 49: cozy.worker.v1.DeliveryGrant
	(*InputLocation)(nil),        // 50: cozy.worker.v1.InputLocation
	(*OutputDestination)(nil),    // 51: cozy.worker.v1.OutputDestination
	(*Credential)(nil),           // 52: cozy.worker.v1.Credential
	(*ArtifactGrant)(nil),        // 53: cozy.worker.v1.ArtifactGrant
	(*ArtifactSubject)(nil),      // 54: cozy.worker.v1.ArtifactSubject
	(*DrainPolicy)(nil),          // 55: cozy.worker.v1.DrainPolicy
	(*ResourceCaps)(nil),         // 56: cozy.worker.v1.ResourceCaps
	(*PublicationContract)(nil),  // 57: cozy.worker.v1.PublicationContract
	(*OutputSpec)(nil),           // 58: cozy.worker.v1.OutputSpec
	(*WorkerResources)(nil),      // 59: cozy.worker.v1.WorkerResources
	(*ServingCapacity)(nil),      // 60: cozy.worker.v1.ServingCapacity
	(*JobCapacity)(nil),          // 61: cozy.worker.v1.JobCapacity
	(*ActiveAttempt)(nil),        // 62: cozy.worker.v1.ActiveAttempt
	(*Fault)(nil),                // 63: cozy.worker.v1.Fault
	(*OutputManifest)(nil),       // 64: cozy.worker.v1.OutputManifest
	(*OutputEntry)(nil),          // 65: cozy.worker.v1.OutputEntry
	(*AttemptMetrics)(nil),       // 66: cozy.worker.v1.AttemptMetrics
	(*TriageBundleRef)(nil),      // 67: cozy.worker.v1.TriageBundleRef
	(*PendingLeases)(nil),        // 68: cozy.worker.v1.PendingLeases
	(*AbsenceEvidence)(nil),      // 69: cozy.worker.v1.AbsenceEvidence
	(*PurgeFault)(nil),           // 70: cozy.worker.v1.PurgeFault
	(*emptypb.Empty)(nil),        // 71: google.protobuf.Empty
}
var file_cozy_worker_v1_worker_proto_depIdxs = []int32{
	18, // 0: cozy.worker.v1.WorkerMessage.register:type_name -> cozy.worker.v1.Register
	24, // 1: cozy.worker.v1.WorkerMessage.report:type_name -> cozy.worker.v1.Report
	27, // 2: cozy.worker.v1.WorkerMessage.attempt_accepted:type_name -> cozy.worker.v1.AttemptAccepted
	30, // 3: cozy.worker.v1.WorkerMessage.attempt_terminal:type_name -> cozy.worker.v1.AttemptTerminal
	19, // 4: cozy.worker.v1.WorkerMessage.boot_failure:type_name -> cozy.worker.v1.BootFailure
	38, // 5: cozy.worker.v1.WorkerMessage.checkpoint_request:type_name -> cozy.worker.v1.JobCheckpointRequest
	41, // 6: cozy.worker.v1.WorkerMessage.checkpoint_ack:type_name -> cozy.worker.v1.JobCheckpointAck
	20, // 7: cozy.worker.v1.CoordinatorMessage.register_ack:type_name -> cozy.worker.v1.RegisterAck
	21, // 8: cozy.worker.v1.CoordinatorMessage.directive:type_name -> cozy.worker.v1.Directive
	26, // 9: cozy.worker.v1.CoordinatorMessage.start_attempt:type_name -> cozy.worker.v1.StartAttempt
	29, // 10: cozy.worker.v1.CoordinatorMessage.cancel_attempt:type_name -> cozy.worker.v1.CancelAttempt
	37, // 11: cozy.worker.v1.CoordinatorMessage.terminal_ack:type_name -> cozy.worker.v1.TerminalAck
	39, // 12: cozy.worker.v1.CoordinatorMessage.checkpoint_receipt:type_name -> cozy.worker.v1.JobCheckpointReceipt
	43, // 13: cozy.worker.v1.PurgeMessage.purge_artifact:type_name -> cozy.worker.v1.PurgeArtifact
	45, // 14: cozy.worker.v1.PurgeMessage.purge_artifact_report:type_name -> cozy.worker.v1.PurgeArtifactReport
	59, // 15: cozy.worker.v1.Register.resources:type_name -> cozy.worker.v1.WorkerResources
	62, // 16: cozy.worker.v1.Register.recovered_attempts:type_name -> cozy.worker.v1.ActiveAttempt
	12, // 17: cozy.worker.v1.BootFailure.reason:type_name -> cozy.worker.v1.BootFailureReason
	59, // 18: cozy.worker.v1.BootFailure.resources:type_name -> cozy.worker.v1.WorkerResources
	8,  // 19: cozy.worker.v1.RegisterAck.rejection:type_name -> cozy.worker.v1.RegisterRejection
	0,  // 20: cozy.worker.v1.Directive.posture:type_name -> cozy.worker.v1.Posture
	55, // 21: cozy.worker.v1.Directive.drain_policy:type_name -> cozy.worker.v1.DrainPolicy
	53, // 22: cozy.worker.v1.Directive.artifact_grant:type_name -> cozy.worker.v1.ArtifactGrant
	52, // 23: cozy.worker.v1.Directive.credential:type_name -> cozy.worker.v1.Credential
	22, // 24: cozy.worker.v1.Directive.serving:type_name -> cozy.worker.v1.ServingDirective
	23, // 25: cozy.worker.v1.Directive.job:type_name -> cozy.worker.v1.JobDirective
	56, // 26: cozy.worker.v1.JobDirective.resource_caps:type_name -> cozy.worker.v1.ResourceCaps
	57, // 27: cozy.worker.v1.JobDirective.publication_contract:type_name -> cozy.worker.v1.PublicationContract
	1,  // 28: cozy.worker.v1.Report.intake_state:type_name -> cozy.worker.v1.IntakeState
	62, // 29: cozy.worker.v1.Report.active_attempts:type_name -> cozy.worker.v1.ActiveAttempt
	63, // 30: cozy.worker.v1.Report.faults:type_name -> cozy.worker.v1.Fault
	25, // 31: cozy.worker.v1.Report.activity:type_name -> cozy.worker.v1.ActivityEvent
	60, // 32: cozy.worker.v1.Report.serving_capacity:type_name -> cozy.worker.v1.ServingCapacity
	61, // 33: cozy.worker.v1.Report.job_capacity:type_name -> cozy.worker.v1.JobCapacity
	49, // 34: cozy.worker.v1.StartAttempt.grant:type_name -> cozy.worker.v1.DeliveryGrant
	28, // 35: cozy.worker.v1.AttemptAccepted.plan:type_name -> cozy.worker.v1.AttemptPlanSummary
	7,  // 36: cozy.worker.v1.CancelAttempt.reason:type_name -> cozy.worker.v1.CancelReason
	4,  // 37: cozy.worker.v1.TerminalBody.status:type_name -> cozy.worker.v1.TerminalStatus
	64, // 38: cozy.worker.v1.TerminalBody.output_manifest:type_name -> cozy.worker.v1.OutputManifest
	66, // 39: cozy.worker.v1.TerminalBody.metrics:type_name -> cozy.worker.v1.AttemptMetrics
	67, // 40: cozy.worker.v1.TerminalBody.triage_bundle:type_name -> cozy.worker.v1.TriageBundleRef
	35, // 41: cozy.worker.v1.TerminalBody.cause:type_name -> cozy.worker.v1.TerminalCause
	32, // 42: cozy.worker.v1.TerminalBody.result:type_name -> cozy.worker.v1.ResultEnvelope
	65, // 43: cozy.worker.v1.ResultEnvelope.result_blob:type_name -> cozy.worker.v1.OutputEntry
	33, // 44: cozy.worker.v1.ResultEnvelope.adjustments:type_name -> cozy.worker.v1.AdjustmentRow
	34, // 45: cozy.worker.v1.ResultEnvelope.adapters:type_name -> cozy.worker.v1.AppliedAdapter
	5,  // 46: cozy.worker.v1.TerminalCause.code:type_name -> cozy.worker.v1.CauseCode
	6,  // 47: cozy.worker.v1.TerminalCause.origin:type_name -> cozy.worker.v1.CauseOrigin
	36, // 48: cozy.worker.v1.TerminalCause.shortfall:type_name -> cozy.worker.v1.ResourceShortfall
	65, // 49: cozy.worker.v1.JobCheckpointRequest.artifact:type_name -> cozy.worker.v1.OutputEntry
	13, // 50: cozy.worker.v1.JobCheckpointReceipt.outcome:type_name -> cozy.worker.v1.CheckpointOutcome
	40, // 51: cozy.worker.v1.JobCheckpointReceipt.fault:type_name -> cozy.worker.v1.CheckpointFault
	14, // 52: cozy.worker.v1.CheckpointFault.code:type_name -> cozy.worker.v1.CheckpointFaultCode
	54, // 53: cozy.worker.v1.PurgeBody.subjects:type_name -> cozy.worker.v1.ArtifactSubject
	68, // 54: cozy.worker.v1.PurgeArtifactReport.pending:type_name -> cozy.worker.v1.PendingLeases
	69, // 55: cozy.worker.v1.PurgeArtifactReport.absent:type_name -> cozy.worker.v1.AbsenceEvidence
	70, // 56: cozy.worker.v1.PurgeArtifactReport.fault:type_name -> cozy.worker.v1.PurgeFault
	47, // 57: cozy.worker.v1.ExecutionSpec.serving:type_name -> cozy.worker.v1.ServingExecutionSpec
	48, // 58: cozy.worker.v1.ExecutionSpec.job:type_name -> cozy.worker.v1.JobExecutionSpec
	57, // 59: cozy.worker.v1.JobExecutionSpec.publication_contract:type_name -> cozy.worker.v1.PublicationContract
	52, // 60: cozy.worker.v1.DeliveryGrant.credential:type_name -> cozy.worker.v1.Credential
	50, // 61: cozy.worker.v1.DeliveryGrant.inputs:type_name -> cozy.worker.v1.InputLocation
	51, // 62: cozy.worker.v1.DeliveryGrant.outputs:type_name -> cozy.worker.v1.OutputDestination
	54, // 63: cozy.worker.v1.ArtifactGrant.subjects:type_name -> cozy.worker.v1.ArtifactSubject
	10, // 64: cozy.worker.v1.ArtifactSubject.kind:type_name -> cozy.worker.v1.SubjectKind
	58, // 65: cozy.worker.v1.PublicationContract.outputs:type_name -> cozy.worker.v1.OutputSpec
	2,  // 66: cozy.worker.v1.ActiveAttempt.kind:type_name -> cozy.worker.v1.AttemptKind
	3,  // 67: cozy.worker.v1.ActiveAttempt.state:type_name -> cozy.worker.v1.AttemptState
	9,  // 68: cozy.worker.v1.Fault.kind:type_name -> cozy.worker.v1.FaultKind
	65, // 69: cozy.worker.v1.OutputManifest.outputs:type_name -> cozy.worker.v1.OutputEntry
	11, // 70: cozy.worker.v1.PurgeFault.code:type_name -> cozy.worker.v1.PurgeFaultCode
	15, // 71: cozy.worker.v1.Worker.Control:input_type -> cozy.worker.v1.WorkerMessage
	42, // 72: cozy.worker.v1.Worker.StreamProgress:input_type -> cozy.worker.v1.AttemptProgress
	17, // 73: cozy.worker.v1.Worker.Purge:input_type -> cozy.worker.v1.PurgeMessage
	16, // 74: cozy.worker.v1.Worker.Control:output_type -> cozy.worker.v1.CoordinatorMessage
	71, // 75: cozy.worker.v1.Worker.StreamProgress:output_type -> google.protobuf.Empty
	17, // 76: cozy.worker.v1.Worker.Purge:output_type -> cozy.worker.v1.PurgeMessage
	74, // [74:77] is the sub-list for method output_type
	71, // [71:74] is the sub-list for method input_type
	71, // [71:71] is the sub-list for extension type_name
	71, // [71:71] is the sub-list for extension extendee
	0,  // [0:71] is the sub-list for field type_name
}

func init() { file_cozy_worker_v1_worker_proto_init() }
func file_cozy_worker_v1_worker_proto_init() {
	if File_cozy_worker_v1_worker_proto != nil {
		return
	}
	file_cozy_worker_v1_worker_proto_msgTypes[0].OneofWrappers = []any{
		(*WorkerMessage_Register)(nil),
		(*WorkerMessage_Report)(nil),
		(*WorkerMessage_AttemptAccepted)(nil),
		(*WorkerMessage_AttemptTerminal)(nil),
		(*WorkerMessage_BootFailure)(nil),
		(*WorkerMessage_CheckpointRequest)(nil),
		(*WorkerMessage_CheckpointAck)(nil),
	}
	file_cozy_worker_v1_worker_proto_msgTypes[1].OneofWrappers = []any{
		(*CoordinatorMessage_RegisterAck)(nil),
		(*CoordinatorMessage_Directive)(nil),
		(*CoordinatorMessage_StartAttempt)(nil),
		(*CoordinatorMessage_CancelAttempt)(nil),
		(*CoordinatorMessage_TerminalAck)(nil),
		(*CoordinatorMessage_CheckpointReceipt)(nil),
	}
	file_cozy_worker_v1_worker_proto_msgTypes[2].OneofWrappers = []any{
		(*PurgeMessage_PurgeArtifact)(nil),
		(*PurgeMessage_PurgeArtifactReport)(nil),
	}
	file_cozy_worker_v1_worker_proto_msgTypes[6].OneofWrappers = []any{
		(*Directive_Serving)(nil),
		(*Directive_Job)(nil),
	}
	file_cozy_worker_v1_worker_proto_msgTypes[9].OneofWrappers = []any{
		(*Report_ServingCapacity)(nil),
		(*Report_JobCapacity)(nil),
	}
	file_cozy_worker_v1_worker_proto_msgTypes[30].OneofWrappers = []any{
		(*PurgeArtifactReport_Pending)(nil),
		(*PurgeArtifactReport_Absent)(nil),
		(*PurgeArtifactReport_Fault)(nil),
	}
	file_cozy_worker_v1_worker_proto_msgTypes[31].OneofWrappers = []any{
		(*ExecutionSpec_Serving)(nil),
		(*ExecutionSpec_Job)(nil),
	}
	type x struct{}
	out := protoimpl.TypeBuilder{
		File: protoimpl.DescBuilder{
			GoPackagePath: reflect.TypeOf(x{}).PkgPath(),
			RawDescriptor: unsafe.Slice(unsafe.StringData(file_cozy_worker_v1_worker_proto_rawDesc), len(file_cozy_worker_v1_worker_proto_rawDesc)),
			NumEnums:      15,
			NumMessages:   56,
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
