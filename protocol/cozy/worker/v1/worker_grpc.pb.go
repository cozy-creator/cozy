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

// Code generated by protoc-gen-go-grpc. DO NOT EDIT.
// versions:
// - protoc-gen-go-grpc v1.6.0
// - protoc             v6.31.1
// source: cozy/worker/v1/worker.proto

package workerprotov1

import (
	context "context"
	grpc "google.golang.org/grpc"
	codes "google.golang.org/grpc/codes"
	status "google.golang.org/grpc/status"
	emptypb "google.golang.org/protobuf/types/known/emptypb"
)

// This is a compile-time assertion to ensure that this generated file
// is compatible with the grpc package it is being compiled against.
// Requires gRPC-Go v1.64.0 or later.
const _ = grpc.SupportPackageIsVersion9

const (
	Worker_Control_FullMethodName        = "/cozy.worker.v1.Worker/Control"
	Worker_StreamProgress_FullMethodName = "/cozy.worker.v1.Worker/StreamProgress"
	Worker_Purge_FullMethodName          = "/cozy.worker.v1.Worker/Purge"
)

// WorkerClient is the client API for Worker service.
//
// For semantics around ctx use and closing/ending streaming RPCs, please refer to https://pkg.go.dev/google.golang.org/grpc/?tab=doc#ClientConn.NewStream.
//
// The worker (the cozy-runtime supervisor) is the gRPC CLIENT; the hub (tensorhub, or
// cozy-creator local-first) is the gRPC SERVER. All RPCs multiplex on one HTTP/2 connection;
// their durability attributes differ.
type WorkerClient interface {
	// RPC 1 - durable bidi control: one bidirectional stream, one causal order, one fence.
	// Durable messages are NEVER shed to backpressure. Terminal authority lives only here.
	Control(ctx context.Context, opts ...grpc.CallOption) (grpc.BidiStreamingClient[WorkerMessage, CoordinatorMessage], error)
	// RPC 2 - bounded LOSSY progress (worker->hub). Sequence-numbered, gaps visible, oldest
	// shed first on overflow, no terminal authority. Saturating it can never block Control.
	StreamProgress(ctx context.Context, opts ...grpc.CallOption) (grpc.ClientStreamingClient[AttemptProgress, emptypb.Empty], error)
	// RPC 3 - RESERVED, post-M9 (cr-013/tfs-014). Independent durable purge exchange, NOT
	// nested in Directive, so a full-replace deployment change can never forget a
	// confidentiality obligation. Schema and numbers are frozen now; the launch conformance
	// surface contains no Purge case.
	Purge(ctx context.Context, opts ...grpc.CallOption) (grpc.BidiStreamingClient[PurgeMessage, PurgeMessage], error)
}

type workerClient struct {
	cc grpc.ClientConnInterface
}

func NewWorkerClient(cc grpc.ClientConnInterface) WorkerClient {
	return &workerClient{cc}
}

func (c *workerClient) Control(ctx context.Context, opts ...grpc.CallOption) (grpc.BidiStreamingClient[WorkerMessage, CoordinatorMessage], error) {
	cOpts := append([]grpc.CallOption{grpc.StaticMethod()}, opts...)
	stream, err := c.cc.NewStream(ctx, &Worker_ServiceDesc.Streams[0], Worker_Control_FullMethodName, cOpts...)
	if err != nil {
		return nil, err
	}
	x := &grpc.GenericClientStream[WorkerMessage, CoordinatorMessage]{ClientStream: stream}
	return x, nil
}

// This type alias is provided for backwards compatibility with existing code that references the prior non-generic stream type by name.
type Worker_ControlClient = grpc.BidiStreamingClient[WorkerMessage, CoordinatorMessage]

func (c *workerClient) StreamProgress(ctx context.Context, opts ...grpc.CallOption) (grpc.ClientStreamingClient[AttemptProgress, emptypb.Empty], error) {
	cOpts := append([]grpc.CallOption{grpc.StaticMethod()}, opts...)
	stream, err := c.cc.NewStream(ctx, &Worker_ServiceDesc.Streams[1], Worker_StreamProgress_FullMethodName, cOpts...)
	if err != nil {
		return nil, err
	}
	x := &grpc.GenericClientStream[AttemptProgress, emptypb.Empty]{ClientStream: stream}
	return x, nil
}

// This type alias is provided for backwards compatibility with existing code that references the prior non-generic stream type by name.
type Worker_StreamProgressClient = grpc.ClientStreamingClient[AttemptProgress, emptypb.Empty]

func (c *workerClient) Purge(ctx context.Context, opts ...grpc.CallOption) (grpc.BidiStreamingClient[PurgeMessage, PurgeMessage], error) {
	cOpts := append([]grpc.CallOption{grpc.StaticMethod()}, opts...)
	stream, err := c.cc.NewStream(ctx, &Worker_ServiceDesc.Streams[2], Worker_Purge_FullMethodName, cOpts...)
	if err != nil {
		return nil, err
	}
	x := &grpc.GenericClientStream[PurgeMessage, PurgeMessage]{ClientStream: stream}
	return x, nil
}

// This type alias is provided for backwards compatibility with existing code that references the prior non-generic stream type by name.
type Worker_PurgeClient = grpc.BidiStreamingClient[PurgeMessage, PurgeMessage]

// WorkerServer is the server API for Worker service.
// All implementations must embed UnimplementedWorkerServer
// for forward compatibility.
//
// The worker (the cozy-runtime supervisor) is the gRPC CLIENT; the hub (tensorhub, or
// cozy-creator local-first) is the gRPC SERVER. All RPCs multiplex on one HTTP/2 connection;
// their durability attributes differ.
type WorkerServer interface {
	// RPC 1 - durable bidi control: one bidirectional stream, one causal order, one fence.
	// Durable messages are NEVER shed to backpressure. Terminal authority lives only here.
	Control(grpc.BidiStreamingServer[WorkerMessage, CoordinatorMessage]) error
	// RPC 2 - bounded LOSSY progress (worker->hub). Sequence-numbered, gaps visible, oldest
	// shed first on overflow, no terminal authority. Saturating it can never block Control.
	StreamProgress(grpc.ClientStreamingServer[AttemptProgress, emptypb.Empty]) error
	// RPC 3 - RESERVED, post-M9 (cr-013/tfs-014). Independent durable purge exchange, NOT
	// nested in Directive, so a full-replace deployment change can never forget a
	// confidentiality obligation. Schema and numbers are frozen now; the launch conformance
	// surface contains no Purge case.
	Purge(grpc.BidiStreamingServer[PurgeMessage, PurgeMessage]) error
	mustEmbedUnimplementedWorkerServer()
}

// UnimplementedWorkerServer must be embedded to have
// forward compatible implementations.
//
// NOTE: this should be embedded by value instead of pointer to avoid a nil
// pointer dereference when methods are called.
type UnimplementedWorkerServer struct{}

func (UnimplementedWorkerServer) Control(grpc.BidiStreamingServer[WorkerMessage, CoordinatorMessage]) error {
	return status.Error(codes.Unimplemented, "method Control not implemented")
}
func (UnimplementedWorkerServer) StreamProgress(grpc.ClientStreamingServer[AttemptProgress, emptypb.Empty]) error {
	return status.Error(codes.Unimplemented, "method StreamProgress not implemented")
}
func (UnimplementedWorkerServer) Purge(grpc.BidiStreamingServer[PurgeMessage, PurgeMessage]) error {
	return status.Error(codes.Unimplemented, "method Purge not implemented")
}
func (UnimplementedWorkerServer) mustEmbedUnimplementedWorkerServer() {}
func (UnimplementedWorkerServer) testEmbeddedByValue()                {}

// UnsafeWorkerServer may be embedded to opt out of forward compatibility for this service.
// Use of this interface is not recommended, as added methods to WorkerServer will
// result in compilation errors.
type UnsafeWorkerServer interface {
	mustEmbedUnimplementedWorkerServer()
}

func RegisterWorkerServer(s grpc.ServiceRegistrar, srv WorkerServer) {
	// If the following call panics, it indicates UnimplementedWorkerServer was
	// embedded by pointer and is nil.  This will cause panics if an
	// unimplemented method is ever invoked, so we test this at initialization
	// time to prevent it from happening at runtime later due to I/O.
	if t, ok := srv.(interface{ testEmbeddedByValue() }); ok {
		t.testEmbeddedByValue()
	}
	s.RegisterService(&Worker_ServiceDesc, srv)
}

func _Worker_Control_Handler(srv interface{}, stream grpc.ServerStream) error {
	return srv.(WorkerServer).Control(&grpc.GenericServerStream[WorkerMessage, CoordinatorMessage]{ServerStream: stream})
}

// This type alias is provided for backwards compatibility with existing code that references the prior non-generic stream type by name.
type Worker_ControlServer = grpc.BidiStreamingServer[WorkerMessage, CoordinatorMessage]

func _Worker_StreamProgress_Handler(srv interface{}, stream grpc.ServerStream) error {
	return srv.(WorkerServer).StreamProgress(&grpc.GenericServerStream[AttemptProgress, emptypb.Empty]{ServerStream: stream})
}

// This type alias is provided for backwards compatibility with existing code that references the prior non-generic stream type by name.
type Worker_StreamProgressServer = grpc.ClientStreamingServer[AttemptProgress, emptypb.Empty]

func _Worker_Purge_Handler(srv interface{}, stream grpc.ServerStream) error {
	return srv.(WorkerServer).Purge(&grpc.GenericServerStream[PurgeMessage, PurgeMessage]{ServerStream: stream})
}

// This type alias is provided for backwards compatibility with existing code that references the prior non-generic stream type by name.
type Worker_PurgeServer = grpc.BidiStreamingServer[PurgeMessage, PurgeMessage]

// Worker_ServiceDesc is the grpc.ServiceDesc for Worker service.
// It's only intended for direct use with grpc.RegisterService,
// and not to be introspected or modified (even as a copy)
var Worker_ServiceDesc = grpc.ServiceDesc{
	ServiceName: "cozy.worker.v1.Worker",
	HandlerType: (*WorkerServer)(nil),
	Methods:     []grpc.MethodDesc{},
	Streams: []grpc.StreamDesc{
		{
			StreamName:    "Control",
			Handler:       _Worker_Control_Handler,
			ServerStreams: true,
			ClientStreams: true,
		},
		{
			StreamName:    "StreamProgress",
			Handler:       _Worker_StreamProgress_Handler,
			ClientStreams: true,
		},
		{
			StreamName:    "Purge",
			Handler:       _Worker_Purge_Handler,
			ServerStreams: true,
			ClientStreams: true,
		},
	},
	Metadata: "cozy/worker/v1/worker.proto",
}
