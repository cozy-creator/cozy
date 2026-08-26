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
// The worker supervisor is the gRPC SERVER; the record-plane owner is the CLIENT.
type WorkerControlClient interface {
	// Durable control: the owner's request stream carries OwnerFrames, the worker's response
	// stream carries WorkerFrames. Durable messages are never shed to backpressure. Terminal
	// authority lives only here. gRPC orders each DIRECTION independently — there is no cross-
	// direction causal order; causality is expressed by revisions, snapshot ids, and acks.
	Control(ctx context.Context, opts ...grpc.CallOption) (grpc.BidiStreamingClient[OwnerFrame, WorkerFrame], error)
	// Bounded LOSSY progress, worker->owner server-streaming. Sequence-numbered, gaps visible,
	// oldest shed first on overflow, no terminal authority. Where the contract promises that
	// progress saturation cannot block control, the owner opens WatchProgress on a PHYSICALLY
	// SEPARATE HTTP/2 connection — separate streams on one TCP connection do not prove it.
	WatchProgress(ctx context.Context, in *ProgressOpen, opts ...grpc.CallOption) (grpc.ServerStreamingClient[AttemptProgress], error)
}

type workerControlClient struct {
	cc grpc.ClientConnInterface
}

func NewWorkerControlClient(cc grpc.ClientConnInterface) WorkerControlClient {
	return &workerControlClient{cc}
}

func (c *workerControlClient) Control(ctx context.Context, opts ...grpc.CallOption) (grpc.BidiStreamingClient[OwnerFrame, WorkerFrame], error) {
	cOpts := append([]grpc.CallOption{grpc.StaticMethod()}, opts...)
	stream, err := c.cc.NewStream(ctx, &WorkerControl_ServiceDesc.Streams[0], WorkerControl_Control_FullMethodName, cOpts...)
	if err != nil {
		return nil, err
	}
	x := &grpc.GenericClientStream[OwnerFrame, WorkerFrame]{ClientStream: stream}
	return x, nil
}

// This type alias is provided for backwards compatibility with existing code that references the prior non-generic stream type by name.
type WorkerControl_ControlClient = grpc.BidiStreamingClient[OwnerFrame, WorkerFrame]

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
// The worker supervisor is the gRPC SERVER; the record-plane owner is the CLIENT.
type WorkerControlServer interface {
	// Durable control: the owner's request stream carries OwnerFrames, the worker's response
	// stream carries WorkerFrames. Durable messages are never shed to backpressure. Terminal
	// authority lives only here. gRPC orders each DIRECTION independently — there is no cross-
	// direction causal order; causality is expressed by revisions, snapshot ids, and acks.
	Control(grpc.BidiStreamingServer[OwnerFrame, WorkerFrame]) error
	// Bounded LOSSY progress, worker->owner server-streaming. Sequence-numbered, gaps visible,
	// oldest shed first on overflow, no terminal authority. Where the contract promises that
	// progress saturation cannot block control, the owner opens WatchProgress on a PHYSICALLY
	// SEPARATE HTTP/2 connection — separate streams on one TCP connection do not prove it.
	WatchProgress(*ProgressOpen, grpc.ServerStreamingServer[AttemptProgress]) error
	mustEmbedUnimplementedWorkerControlServer()
}

// UnimplementedWorkerControlServer must be embedded to have
// forward compatible implementations.
//
// NOTE: this should be embedded by value instead of pointer to avoid a nil
// pointer dereference when methods are called.
type UnimplementedWorkerControlServer struct{}

func (UnimplementedWorkerControlServer) Control(grpc.BidiStreamingServer[OwnerFrame, WorkerFrame]) error {
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
	return srv.(WorkerControlServer).Control(&grpc.GenericServerStream[OwnerFrame, WorkerFrame]{ServerStream: stream})
}

// This type alias is provided for backwards compatibility with existing code that references the prior non-generic stream type by name.
type WorkerControl_ControlServer = grpc.BidiStreamingServer[OwnerFrame, WorkerFrame]

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
