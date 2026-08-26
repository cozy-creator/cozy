// cozy.worker.v1 — the RecordOwner<->worker wire contract. Schema only; no behavior.
//
// Authority: tracker-v2/worker-protocol/01-wire-schema.md (schema laws), 02-lifecycle.md
// (behavior), 03-versioning-conformance.md (versioning), rev-2-dynamic-serving.md (THIS
// revision). Issue: tracker/tensorhub/th-024.
//
// ORIENTATION: the WORKER (cozy-runtime's torch-free machine control process) HOSTS this
// service; the RECORDOWNER (cozy-creator's embedded orchestrator, a pod's PodRecordOwner, or a
// tensorhub orchestrator-shard via th-007) DIALS it. The orientation is IDENTICAL local and
// remote; only channel establishment differs: a Unix socket (Windows: loopback) co-resident,
// TLS over a private network or authenticated overlay remotely. A worker control port is NEVER
// required to be publicly exposed for topology symmetry.
//
// REV-2, THE DYNAMIC-SERVING REV (decisions #471-#475, #480-#483, #485b/c, #486, #487/#489,
// #507d). BREAKING IN PLACE on the unreleased `cozy.worker.v1` (#480g): field numbers are
// reserved at their sites, deleted messages keep their numbers reserved, both reference halves
// and the conformance corpus regenerate together. No shim speaks both shapes; no rename ships
// an alias. This is the fifth in-place revision (5b07b79 -> 732763b -> 8d90461 -> c6dbc12 ->
// this) and it renumbers freely, per the 2026-08-24 precedent.
//
// VERSIONING (03 §1, §2 R1-R8): the MAJOR is the package path (`cozy.worker.v1`); the MINOR is
// `wire_minor`, a linear-train number declared at Claim/ClaimAck — never a negotiation. R7 IS AN
// AUTHORING RULE ONLY: `reserved` numbers are compiler-enforced tombstones against REUSE;
// ordinary proto3 decoders do not refuse them on the wire and no runtime polices them.
//
// IDENTITY IS CANONICAL BYTES, PROTOBUF IS TRANSPORT: no digest is ever computed over
// protobuf-marshaled bytes. A meaning-fencing digest is the SHA-256 of a DOCUMENT's exact
// canonical bytes (RFC 8785-shaped canonical JSON under tfs-013's writer). Where the fence
// requires byte agreement the canonical bytes travel IN the message as a `bytes` field instead
// of a structured body; the receiver recomputes sha256 over the resident bytes BEFORE parsing a
// single field. DIGEST CLASSES: (a) document identity over canonical JSON bytes; (b) content
// digest over opaque artifact bytes. On the wire a class-(a)/(b) digest is raw 32-byte `bytes`;
// inside a canonical document the same digest is spelled `"sha256:<64 lowercase hex>"`.
// BYTES NAMING LAW: a `bytes` field named `digest`/`*_digest` IS a 32-byte SHA-256; every other
// `bytes` field is an opaque payload (base64 in documents).
//
// DOCUMENT VERSIONS. A digest-fenced document is NOT additively versioned: an unknown key
// REFUSES, and a new key is a new document version. The canonical `format` tag is the message's
// full name plus its document version. Every document here is at /1 except
// `cozy.worker.v1.AttemptOutcomeBody/2`, which carries the `execution_started` bit added by
// #480c on top of the frozen wire's TerminalBody/1 lineage (#481).
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
// key for one hosted assignment. The desired set maps placement_id -> PlacementSpec, a fully
// resolved IMMUTABLE document: the worker never resolves a mutable release id, so two workers
// handed the same set converge to the same bytes or fault typed. LAUNCH ENFORCES
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
// serving axis + dispatchable_plan_ids). Per-placement `readiness_epoch` is DELETED, not
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
// (th-007) stay reservations. `ArtifactGrant`/`ArtifactSubject` are UN-RESERVED by this rev:
// materialization happens before any attempt exists, so it now has a writer (§3).

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
)

// This is a compile-time assertion to ensure that this generated file
// is compatible with the grpc package it is being compiled against.
// Requires gRPC-Go v1.64.0 or later.
const _ = grpc.SupportPackageIsVersion9

const (
	WorkerControl_Control_FullMethodName       = "/cozy.worker.v1.WorkerControl/Control"
	WorkerControl_WatchProgress_FullMethodName = "/cozy.worker.v1.WorkerControl/WatchProgress"
)

// WorkerControlClient is the client API for WorkerControl service.
//
// For semantics around ctx use and closing/ending streaming RPCs, please refer to https://pkg.go.dev/google.golang.org/grpc/?tab=doc#ClientConn.NewStream.
//
// The worker is the gRPC SERVER; the RecordOwner is the CLIENT.
type WorkerControlClient interface {
	// Durable control: the RecordOwner's request stream carries RecordOwnerFrames, the worker's
	// response stream carries WorkerFrames. Durable messages are never shed to backpressure.
	// Outcome authority lives only here. gRPC orders each DIRECTION independently — there is no
	// cross-direction causal order; causality is expressed by revisions, snapshot ids, and acks.
	Control(ctx context.Context, opts ...grpc.CallOption) (grpc.BidiStreamingClient[RecordOwnerFrame, WorkerFrame], error)
	// Bounded LOSSY progress, worker->RecordOwner server-streaming. Sequence-numbered, gaps
	// visible, oldest shed first on overflow, no outcome authority. Where the contract promises
	// that progress saturation cannot block control, the RecordOwner opens WatchProgress on a
	// PHYSICALLY SEPARATE HTTP/2 connection — separate streams on one TCP connection do not
	// prove it.
	WatchProgress(ctx context.Context, in *ProgressOpen, opts ...grpc.CallOption) (grpc.ServerStreamingClient[AttemptProgress], error)
}

type workerControlClient struct {
	cc grpc.ClientConnInterface
}

func NewWorkerControlClient(cc grpc.ClientConnInterface) WorkerControlClient {
	return &workerControlClient{cc}
}

func (c *workerControlClient) Control(ctx context.Context, opts ...grpc.CallOption) (grpc.BidiStreamingClient[RecordOwnerFrame, WorkerFrame], error) {
	cOpts := append([]grpc.CallOption{grpc.StaticMethod()}, opts...)
	stream, err := c.cc.NewStream(ctx, &WorkerControl_ServiceDesc.Streams[0], WorkerControl_Control_FullMethodName, cOpts...)
	if err != nil {
		return nil, err
	}
	x := &grpc.GenericClientStream[RecordOwnerFrame, WorkerFrame]{ClientStream: stream}
	return x, nil
}

// This type alias is provided for backwards compatibility with existing code that references the prior non-generic stream type by name.
type WorkerControl_ControlClient = grpc.BidiStreamingClient[RecordOwnerFrame, WorkerFrame]

func (c *workerControlClient) WatchProgress(ctx context.Context, in *ProgressOpen, opts ...grpc.CallOption) (grpc.ServerStreamingClient[AttemptProgress], error) {
	cOpts := append([]grpc.CallOption{grpc.StaticMethod()}, opts...)
	stream, err := c.cc.NewStream(ctx, &WorkerControl_ServiceDesc.Streams[1], WorkerControl_WatchProgress_FullMethodName, cOpts...)
	if err != nil {
		return nil, err
	}
	x := &grpc.GenericClientStream[ProgressOpen, AttemptProgress]{ClientStream: stream}
	if err := x.ClientStream.SendMsg(in); err != nil {
		return nil, err
	}
	if err := x.ClientStream.CloseSend(); err != nil {
		return nil, err
	}
	return x, nil
}

// This type alias is provided for backwards compatibility with existing code that references the prior non-generic stream type by name.
type WorkerControl_WatchProgressClient = grpc.ServerStreamingClient[AttemptProgress]

// WorkerControlServer is the server API for WorkerControl service.
// All implementations must embed UnimplementedWorkerControlServer
// for forward compatibility.
//
// The worker is the gRPC SERVER; the RecordOwner is the CLIENT.
type WorkerControlServer interface {
	// Durable control: the RecordOwner's request stream carries RecordOwnerFrames, the worker's
	// response stream carries WorkerFrames. Durable messages are never shed to backpressure.
	// Outcome authority lives only here. gRPC orders each DIRECTION independently — there is no
	// cross-direction causal order; causality is expressed by revisions, snapshot ids, and acks.
	Control(grpc.BidiStreamingServer[RecordOwnerFrame, WorkerFrame]) error
	// Bounded LOSSY progress, worker->RecordOwner server-streaming. Sequence-numbered, gaps
	// visible, oldest shed first on overflow, no outcome authority. Where the contract promises
	// that progress saturation cannot block control, the RecordOwner opens WatchProgress on a
	// PHYSICALLY SEPARATE HTTP/2 connection — separate streams on one TCP connection do not
	// prove it.
	WatchProgress(*ProgressOpen, grpc.ServerStreamingServer[AttemptProgress]) error
	mustEmbedUnimplementedWorkerControlServer()
}

// UnimplementedWorkerControlServer must be embedded to have
// forward compatible implementations.
//
// NOTE: this should be embedded by value instead of pointer to avoid a nil
// pointer dereference when methods are called.
type UnimplementedWorkerControlServer struct{}

func (UnimplementedWorkerControlServer) Control(grpc.BidiStreamingServer[RecordOwnerFrame, WorkerFrame]) error {
	return status.Error(codes.Unimplemented, "method Control not implemented")
}
func (UnimplementedWorkerControlServer) WatchProgress(*ProgressOpen, grpc.ServerStreamingServer[AttemptProgress]) error {
	return status.Error(codes.Unimplemented, "method WatchProgress not implemented")
}
func (UnimplementedWorkerControlServer) mustEmbedUnimplementedWorkerControlServer() {}
func (UnimplementedWorkerControlServer) testEmbeddedByValue()                       {}

// UnsafeWorkerControlServer may be embedded to opt out of forward compatibility for this service.
// Use of this interface is not recommended, as added methods to WorkerControlServer will
// result in compilation errors.
type UnsafeWorkerControlServer interface {
	mustEmbedUnimplementedWorkerControlServer()
}

func RegisterWorkerControlServer(s grpc.ServiceRegistrar, srv WorkerControlServer) {
	// If the following call panics, it indicates UnimplementedWorkerControlServer was
	// embedded by pointer and is nil.  This will cause panics if an
	// unimplemented method is ever invoked, so we test this at initialization
	// time to prevent it from happening at runtime later due to I/O.
	if t, ok := srv.(interface{ testEmbeddedByValue() }); ok {
		t.testEmbeddedByValue()
	}
	s.RegisterService(&WorkerControl_ServiceDesc, srv)
}

func _WorkerControl_Control_Handler(srv interface{}, stream grpc.ServerStream) error {
	return srv.(WorkerControlServer).Control(&grpc.GenericServerStream[RecordOwnerFrame, WorkerFrame]{ServerStream: stream})
}

// This type alias is provided for backwards compatibility with existing code that references the prior non-generic stream type by name.
type WorkerControl_ControlServer = grpc.BidiStreamingServer[RecordOwnerFrame, WorkerFrame]

func _WorkerControl_WatchProgress_Handler(srv interface{}, stream grpc.ServerStream) error {
	m := new(ProgressOpen)
	if err := stream.RecvMsg(m); err != nil {
		return err
	}
	return srv.(WorkerControlServer).WatchProgress(m, &grpc.GenericServerStream[ProgressOpen, AttemptProgress]{ServerStream: stream})
}

// This type alias is provided for backwards compatibility with existing code that references the prior non-generic stream type by name.
type WorkerControl_WatchProgressServer = grpc.ServerStreamingServer[AttemptProgress]

// WorkerControl_ServiceDesc is the grpc.ServiceDesc for WorkerControl service.
// It's only intended for direct use with grpc.RegisterService,
// and not to be introspected or modified (even as a copy)
var WorkerControl_ServiceDesc = grpc.ServiceDesc{
	ServiceName: "cozy.worker.v1.WorkerControl",
	HandlerType: (*WorkerControlServer)(nil),
	Methods:     []grpc.MethodDesc{},
	Streams: []grpc.StreamDesc{
		{
			StreamName:    "Control",
			Handler:       _WorkerControl_Control_Handler,
			ServerStreams: true,
			ClientStreams: true,
		},
		{
			StreamName:    "WatchProgress",
			Handler:       _WorkerControl_WatchProgress_Handler,
			ServerStreams: true,
		},
	},
	Metadata: "cozy/worker/v1/worker.proto",
}
