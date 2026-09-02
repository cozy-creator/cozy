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

// Code generated by protoc-gen-go-grpc. DO NOT EDIT.
// versions:
// - protoc-gen-go-grpc v1.6.0
// - protoc             v7.35.1
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

const (
	RuntimePreparation_PreparePackageSet_FullMethodName       = "/cozy.worker.v1.RuntimePreparation/PreparePackageSet"
	RuntimePreparation_PrepareModelSource_FullMethodName      = "/cozy.worker.v1.RuntimePreparation/PrepareModelSource"
	RuntimePreparation_PrepareLocalPackage_FullMethodName     = "/cozy.worker.v1.RuntimePreparation/PrepareLocalPackage"
	RuntimePreparation_PreparePrivatePlacement_FullMethodName = "/cozy.worker.v1.RuntimePreparation/PreparePrivatePlacement"
)

// RuntimePreparationClient is the client API for RuntimePreparation service.
//
// For semantics around ctx use and closing/ending streaming RPCs, please refer to https://pkg.go.dev/google.golang.org/grpc/?tab=doc#ClientConn.NewStream.
//
// Loopback-only pod-supervisor -> Runtime preparation seam. Pod-supervisor never registers this
// service on its external listener. Runtime receives verified local files and logical refs only:
// no Tensorhub origin, presigned URL, delegation signature, or worker TLS credential.
type RuntimePreparationClient interface {
	PreparePackageSet(ctx context.Context, in *PreparePackageSetRequest, opts ...grpc.CallOption) (*PreparePackageSetResult, error)
	PrepareModelSource(ctx context.Context, in *PrepareModelSourceRequest, opts ...grpc.CallOption) (*PrepareModelSourceResult, error)
	PrepareLocalPackage(ctx context.Context, in *PrepareLocalPackageRequest, opts ...grpc.CallOption) (*PreparePackageSetResult, error)
	PreparePrivatePlacement(ctx context.Context, in *PreparePrivatePlacementRequest, opts ...grpc.CallOption) (*PreparePackageSetResult, error)
}

type runtimePreparationClient struct {
	cc grpc.ClientConnInterface
}

func NewRuntimePreparationClient(cc grpc.ClientConnInterface) RuntimePreparationClient {
	return &runtimePreparationClient{cc}
}

func (c *runtimePreparationClient) PreparePackageSet(ctx context.Context, in *PreparePackageSetRequest, opts ...grpc.CallOption) (*PreparePackageSetResult, error) {
	cOpts := append([]grpc.CallOption{grpc.StaticMethod()}, opts...)
	out := new(PreparePackageSetResult)
	err := c.cc.Invoke(ctx, RuntimePreparation_PreparePackageSet_FullMethodName, in, out, cOpts...)
	if err != nil {
		return nil, err
	}
	return out, nil
}

func (c *runtimePreparationClient) PrepareModelSource(ctx context.Context, in *PrepareModelSourceRequest, opts ...grpc.CallOption) (*PrepareModelSourceResult, error) {
	cOpts := append([]grpc.CallOption{grpc.StaticMethod()}, opts...)
	out := new(PrepareModelSourceResult)
	err := c.cc.Invoke(ctx, RuntimePreparation_PrepareModelSource_FullMethodName, in, out, cOpts...)
	if err != nil {
		return nil, err
	}
	return out, nil
}

func (c *runtimePreparationClient) PrepareLocalPackage(ctx context.Context, in *PrepareLocalPackageRequest, opts ...grpc.CallOption) (*PreparePackageSetResult, error) {
	cOpts := append([]grpc.CallOption{grpc.StaticMethod()}, opts...)
	out := new(PreparePackageSetResult)
	err := c.cc.Invoke(ctx, RuntimePreparation_PrepareLocalPackage_FullMethodName, in, out, cOpts...)
	if err != nil {
		return nil, err
	}
	return out, nil
}

func (c *runtimePreparationClient) PreparePrivatePlacement(ctx context.Context, in *PreparePrivatePlacementRequest, opts ...grpc.CallOption) (*PreparePackageSetResult, error) {
	cOpts := append([]grpc.CallOption{grpc.StaticMethod()}, opts...)
	out := new(PreparePackageSetResult)
	err := c.cc.Invoke(ctx, RuntimePreparation_PreparePrivatePlacement_FullMethodName, in, out, cOpts...)
	if err != nil {
		return nil, err
	}
	return out, nil
}

// RuntimePreparationServer is the server API for RuntimePreparation service.
// All implementations must embed UnimplementedRuntimePreparationServer
// for forward compatibility.
//
// Loopback-only pod-supervisor -> Runtime preparation seam. Pod-supervisor never registers this
// service on its external listener. Runtime receives verified local files and logical refs only:
// no Tensorhub origin, presigned URL, delegation signature, or worker TLS credential.
type RuntimePreparationServer interface {
	PreparePackageSet(context.Context, *PreparePackageSetRequest) (*PreparePackageSetResult, error)
	PrepareModelSource(context.Context, *PrepareModelSourceRequest) (*PrepareModelSourceResult, error)
	PrepareLocalPackage(context.Context, *PrepareLocalPackageRequest) (*PreparePackageSetResult, error)
	PreparePrivatePlacement(context.Context, *PreparePrivatePlacementRequest) (*PreparePackageSetResult, error)
	mustEmbedUnimplementedRuntimePreparationServer()
}

// UnimplementedRuntimePreparationServer must be embedded to have
// forward compatible implementations.
//
// NOTE: this should be embedded by value instead of pointer to avoid a nil
// pointer dereference when methods are called.
type UnimplementedRuntimePreparationServer struct{}

func (UnimplementedRuntimePreparationServer) PreparePackageSet(context.Context, *PreparePackageSetRequest) (*PreparePackageSetResult, error) {
	return nil, status.Error(codes.Unimplemented, "method PreparePackageSet not implemented")
}
func (UnimplementedRuntimePreparationServer) PrepareModelSource(context.Context, *PrepareModelSourceRequest) (*PrepareModelSourceResult, error) {
	return nil, status.Error(codes.Unimplemented, "method PrepareModelSource not implemented")
}
func (UnimplementedRuntimePreparationServer) PrepareLocalPackage(context.Context, *PrepareLocalPackageRequest) (*PreparePackageSetResult, error) {
	return nil, status.Error(codes.Unimplemented, "method PrepareLocalPackage not implemented")
}
func (UnimplementedRuntimePreparationServer) PreparePrivatePlacement(context.Context, *PreparePrivatePlacementRequest) (*PreparePackageSetResult, error) {
	return nil, status.Error(codes.Unimplemented, "method PreparePrivatePlacement not implemented")
}
func (UnimplementedRuntimePreparationServer) mustEmbedUnimplementedRuntimePreparationServer() {}
func (UnimplementedRuntimePreparationServer) testEmbeddedByValue()                            {}

// UnsafeRuntimePreparationServer may be embedded to opt out of forward compatibility for this service.
// Use of this interface is not recommended, as added methods to RuntimePreparationServer will
// result in compilation errors.
type UnsafeRuntimePreparationServer interface {
	mustEmbedUnimplementedRuntimePreparationServer()
}

func RegisterRuntimePreparationServer(s grpc.ServiceRegistrar, srv RuntimePreparationServer) {
	// If the following call panics, it indicates UnimplementedRuntimePreparationServer was
	// embedded by pointer and is nil.  This will cause panics if an
	// unimplemented method is ever invoked, so we test this at initialization
	// time to prevent it from happening at runtime later due to I/O.
	if t, ok := srv.(interface{ testEmbeddedByValue() }); ok {
		t.testEmbeddedByValue()
	}
	s.RegisterService(&RuntimePreparation_ServiceDesc, srv)
}

func _RuntimePreparation_PreparePackageSet_Handler(srv interface{}, ctx context.Context, dec func(interface{}) error, interceptor grpc.UnaryServerInterceptor) (interface{}, error) {
	in := new(PreparePackageSetRequest)
	if err := dec(in); err != nil {
		return nil, err
	}
	if interceptor == nil {
		return srv.(RuntimePreparationServer).PreparePackageSet(ctx, in)
	}
	info := &grpc.UnaryServerInfo{
		Server:     srv,
		FullMethod: RuntimePreparation_PreparePackageSet_FullMethodName,
	}
	handler := func(ctx context.Context, req interface{}) (interface{}, error) {
		return srv.(RuntimePreparationServer).PreparePackageSet(ctx, req.(*PreparePackageSetRequest))
	}
	return interceptor(ctx, in, info, handler)
}

func _RuntimePreparation_PrepareModelSource_Handler(srv interface{}, ctx context.Context, dec func(interface{}) error, interceptor grpc.UnaryServerInterceptor) (interface{}, error) {
	in := new(PrepareModelSourceRequest)
	if err := dec(in); err != nil {
		return nil, err
	}
	if interceptor == nil {
		return srv.(RuntimePreparationServer).PrepareModelSource(ctx, in)
	}
	info := &grpc.UnaryServerInfo{
		Server:     srv,
		FullMethod: RuntimePreparation_PrepareModelSource_FullMethodName,
	}
	handler := func(ctx context.Context, req interface{}) (interface{}, error) {
		return srv.(RuntimePreparationServer).PrepareModelSource(ctx, req.(*PrepareModelSourceRequest))
	}
	return interceptor(ctx, in, info, handler)
}

func _RuntimePreparation_PrepareLocalPackage_Handler(srv interface{}, ctx context.Context, dec func(interface{}) error, interceptor grpc.UnaryServerInterceptor) (interface{}, error) {
	in := new(PrepareLocalPackageRequest)
	if err := dec(in); err != nil {
		return nil, err
	}
	if interceptor == nil {
		return srv.(RuntimePreparationServer).PrepareLocalPackage(ctx, in)
	}
	info := &grpc.UnaryServerInfo{
		Server:     srv,
		FullMethod: RuntimePreparation_PrepareLocalPackage_FullMethodName,
	}
	handler := func(ctx context.Context, req interface{}) (interface{}, error) {
		return srv.(RuntimePreparationServer).PrepareLocalPackage(ctx, req.(*PrepareLocalPackageRequest))
	}
	return interceptor(ctx, in, info, handler)
}

func _RuntimePreparation_PreparePrivatePlacement_Handler(srv interface{}, ctx context.Context, dec func(interface{}) error, interceptor grpc.UnaryServerInterceptor) (interface{}, error) {
	in := new(PreparePrivatePlacementRequest)
	if err := dec(in); err != nil {
		return nil, err
	}
	if interceptor == nil {
		return srv.(RuntimePreparationServer).PreparePrivatePlacement(ctx, in)
	}
	info := &grpc.UnaryServerInfo{
		Server:     srv,
		FullMethod: RuntimePreparation_PreparePrivatePlacement_FullMethodName,
	}
	handler := func(ctx context.Context, req interface{}) (interface{}, error) {
		return srv.(RuntimePreparationServer).PreparePrivatePlacement(ctx, req.(*PreparePrivatePlacementRequest))
	}
	return interceptor(ctx, in, info, handler)
}

// RuntimePreparation_ServiceDesc is the grpc.ServiceDesc for RuntimePreparation service.
// It's only intended for direct use with grpc.RegisterService,
// and not to be introspected or modified (even as a copy)
var RuntimePreparation_ServiceDesc = grpc.ServiceDesc{
	ServiceName: "cozy.worker.v1.RuntimePreparation",
	HandlerType: (*RuntimePreparationServer)(nil),
	Methods: []grpc.MethodDesc{
		{
			MethodName: "PreparePackageSet",
			Handler:    _RuntimePreparation_PreparePackageSet_Handler,
		},
		{
			MethodName: "PrepareModelSource",
			Handler:    _RuntimePreparation_PrepareModelSource_Handler,
		},
		{
			MethodName: "PrepareLocalPackage",
			Handler:    _RuntimePreparation_PrepareLocalPackage_Handler,
		},
		{
			MethodName: "PreparePrivatePlacement",
			Handler:    _RuntimePreparation_PreparePrivatePlacement_Handler,
		},
	},
	Streams:  []grpc.StreamDesc{},
	Metadata: "cozy/worker/v1/worker.proto",
}

const (
	RuntimeWeights_Exchange_FullMethodName = "/cozy.worker.v1.RuntimeWeights/Exchange"
	RuntimeWeights_Upload_FullMethodName   = "/cozy.worker.v1.RuntimeWeights/Upload"
)

// RuntimeWeightsClient is the client API for RuntimeWeights service.
//
// For semantics around ctx use and closing/ending streaming RPCs, please refer to https://pkg.go.dev/google.golang.org/grpc/?tab=doc#ClientConn.NewStream.
//
// Loopback-only Runtime-hosted weights seam. pod-supervisor is its only client and dials it on
// the same loopback listener as WorkerControl and RuntimePreparation; it is never registered
// on the external listener. It carries the supervisor<->Runtime privates that were
// direction-annotated WorkerControl frames before minor 21.
type RuntimeWeightsClient interface {
	// Runtime -> host intents and receipts, host -> Runtime acks, on ONE stream the supervisor
	// opens per accepted control session. The host commits its ledger before each ack, and
	// forwards a recorded receipt to the RecordOwner on WorkerControl BEFORE acking it, so the
	// owner sees the receipt ahead of the attempt outcome the Runtime can only send after.
	Exchange(ctx context.Context, opts ...grpc.CallOption) (grpc.BidiStreamingClient[WeightsHostAck, WeightsHostEvent], error)
	// One validated upload grant in, one terminal answer out. Runtime streams the bytes it holds.
	Upload(ctx context.Context, in *WeightsUploadRequest, opts ...grpc.CallOption) (*WeightsUploadResult, error)
}

type runtimeWeightsClient struct {
	cc grpc.ClientConnInterface
}

func NewRuntimeWeightsClient(cc grpc.ClientConnInterface) RuntimeWeightsClient {
	return &runtimeWeightsClient{cc}
}

func (c *runtimeWeightsClient) Exchange(ctx context.Context, opts ...grpc.CallOption) (grpc.BidiStreamingClient[WeightsHostAck, WeightsHostEvent], error) {
	cOpts := append([]grpc.CallOption{grpc.StaticMethod()}, opts...)
	stream, err := c.cc.NewStream(ctx, &RuntimeWeights_ServiceDesc.Streams[0], RuntimeWeights_Exchange_FullMethodName, cOpts...)
	if err != nil {
		return nil, err
	}
	x := &grpc.GenericClientStream[WeightsHostAck, WeightsHostEvent]{ClientStream: stream}
	return x, nil
}

// This type alias is provided for backwards compatibility with existing code that references the prior non-generic stream type by name.
type RuntimeWeights_ExchangeClient = grpc.BidiStreamingClient[WeightsHostAck, WeightsHostEvent]

func (c *runtimeWeightsClient) Upload(ctx context.Context, in *WeightsUploadRequest, opts ...grpc.CallOption) (*WeightsUploadResult, error) {
	cOpts := append([]grpc.CallOption{grpc.StaticMethod()}, opts...)
	out := new(WeightsUploadResult)
	err := c.cc.Invoke(ctx, RuntimeWeights_Upload_FullMethodName, in, out, cOpts...)
	if err != nil {
		return nil, err
	}
	return out, nil
}

// RuntimeWeightsServer is the server API for RuntimeWeights service.
// All implementations must embed UnimplementedRuntimeWeightsServer
// for forward compatibility.
//
// Loopback-only Runtime-hosted weights seam. pod-supervisor is its only client and dials it on
// the same loopback listener as WorkerControl and RuntimePreparation; it is never registered
// on the external listener. It carries the supervisor<->Runtime privates that were
// direction-annotated WorkerControl frames before minor 21.
type RuntimeWeightsServer interface {
	// Runtime -> host intents and receipts, host -> Runtime acks, on ONE stream the supervisor
	// opens per accepted control session. The host commits its ledger before each ack, and
	// forwards a recorded receipt to the RecordOwner on WorkerControl BEFORE acking it, so the
	// owner sees the receipt ahead of the attempt outcome the Runtime can only send after.
	Exchange(grpc.BidiStreamingServer[WeightsHostAck, WeightsHostEvent]) error
	// One validated upload grant in, one terminal answer out. Runtime streams the bytes it holds.
	Upload(context.Context, *WeightsUploadRequest) (*WeightsUploadResult, error)
	mustEmbedUnimplementedRuntimeWeightsServer()
}

// UnimplementedRuntimeWeightsServer must be embedded to have
// forward compatible implementations.
//
// NOTE: this should be embedded by value instead of pointer to avoid a nil
// pointer dereference when methods are called.
type UnimplementedRuntimeWeightsServer struct{}

func (UnimplementedRuntimeWeightsServer) Exchange(grpc.BidiStreamingServer[WeightsHostAck, WeightsHostEvent]) error {
	return status.Error(codes.Unimplemented, "method Exchange not implemented")
}
func (UnimplementedRuntimeWeightsServer) Upload(context.Context, *WeightsUploadRequest) (*WeightsUploadResult, error) {
	return nil, status.Error(codes.Unimplemented, "method Upload not implemented")
}
func (UnimplementedRuntimeWeightsServer) mustEmbedUnimplementedRuntimeWeightsServer() {}
func (UnimplementedRuntimeWeightsServer) testEmbeddedByValue()                        {}

// UnsafeRuntimeWeightsServer may be embedded to opt out of forward compatibility for this service.
// Use of this interface is not recommended, as added methods to RuntimeWeightsServer will
// result in compilation errors.
type UnsafeRuntimeWeightsServer interface {
	mustEmbedUnimplementedRuntimeWeightsServer()
}

func RegisterRuntimeWeightsServer(s grpc.ServiceRegistrar, srv RuntimeWeightsServer) {
	// If the following call panics, it indicates UnimplementedRuntimeWeightsServer was
	// embedded by pointer and is nil.  This will cause panics if an
	// unimplemented method is ever invoked, so we test this at initialization
	// time to prevent it from happening at runtime later due to I/O.
	if t, ok := srv.(interface{ testEmbeddedByValue() }); ok {
		t.testEmbeddedByValue()
	}
	s.RegisterService(&RuntimeWeights_ServiceDesc, srv)
}

func _RuntimeWeights_Exchange_Handler(srv interface{}, stream grpc.ServerStream) error {
	return srv.(RuntimeWeightsServer).Exchange(&grpc.GenericServerStream[WeightsHostAck, WeightsHostEvent]{ServerStream: stream})
}

// This type alias is provided for backwards compatibility with existing code that references the prior non-generic stream type by name.
type RuntimeWeights_ExchangeServer = grpc.BidiStreamingServer[WeightsHostAck, WeightsHostEvent]

func _RuntimeWeights_Upload_Handler(srv interface{}, ctx context.Context, dec func(interface{}) error, interceptor grpc.UnaryServerInterceptor) (interface{}, error) {
	in := new(WeightsUploadRequest)
	if err := dec(in); err != nil {
		return nil, err
	}
	if interceptor == nil {
		return srv.(RuntimeWeightsServer).Upload(ctx, in)
	}
	info := &grpc.UnaryServerInfo{
		Server:     srv,
		FullMethod: RuntimeWeights_Upload_FullMethodName,
	}
	handler := func(ctx context.Context, req interface{}) (interface{}, error) {
		return srv.(RuntimeWeightsServer).Upload(ctx, req.(*WeightsUploadRequest))
	}
	return interceptor(ctx, in, info, handler)
}

// RuntimeWeights_ServiceDesc is the grpc.ServiceDesc for RuntimeWeights service.
// It's only intended for direct use with grpc.RegisterService,
// and not to be introspected or modified (even as a copy)
var RuntimeWeights_ServiceDesc = grpc.ServiceDesc{
	ServiceName: "cozy.worker.v1.RuntimeWeights",
	HandlerType: (*RuntimeWeightsServer)(nil),
	Methods: []grpc.MethodDesc{
		{
			MethodName: "Upload",
			Handler:    _RuntimeWeights_Upload_Handler,
		},
	},
	Streams: []grpc.StreamDesc{
		{
			StreamName:    "Exchange",
			Handler:       _RuntimeWeights_Exchange_Handler,
			ServerStreams: true,
			ClientStreams: true,
		},
	},
	Metadata: "cozy/worker/v1/worker.proto",
}

const (
	PodHost_PreparePackageSet_FullMethodName       = "/cozy.worker.v1.PodHost/PreparePackageSet"
	PodHost_PrepareLocalPackage_FullMethodName     = "/cozy.worker.v1.PodHost/PrepareLocalPackage"
	PodHost_PreparePrivatePlacement_FullMethodName = "/cozy.worker.v1.PodHost/PreparePrivatePlacement"
	PodHost_ModelSourceFile_FullMethodName         = "/cozy.worker.v1.PodHost/ModelSourceFile"
	PodHost_ModelSourcePrepare_FullMethodName      = "/cozy.worker.v1.PodHost/ModelSourcePrepare"
	PodHost_LocalPackageFetch_FullMethodName       = "/cozy.worker.v1.PodHost/LocalPackageFetch"
	PodHost_LocalPackageAbort_FullMethodName       = "/cozy.worker.v1.PodHost/LocalPackageAbort"
	PodHost_WeightsTransfer_FullMethodName         = "/cozy.worker.v1.PodHost/WeightsTransfer"
)

// PodHostClient is the client API for PodHost service.
//
// For semantics around ctx use and closing/ending streaming RPCs, please refer to https://pkg.go.dev/google.golang.org/grpc/?tab=doc#ClientConn.NewStream.
//
// ---------------------------------------------------------------------------
// PodHost -- the RecordOwner <-> pod HOST lane. A second service on the pod's external
// listener, served by pod-supervisor beside WorkerControl. It carries what the host does on the
// owner's behalf and nothing that is control: the owner asks the host to PREPARE (download,
// verify, obtain the PlacementSet bytes from the Runtime's loopback preparation), then sends
// those exact bytes ITSELF as DesiredWorkerState{placement_set} on WorkerControl. Remote and
// local are the same three steps -- prepare, placement_set bytes, desired_state -- and a
// multi-GiB materialization never sits on the control stream's read loop, where before 21 it
// head-of-line blocked every offer, ack, and cancel behind it.
//
// AUTHORITY: every call opens with the owner's Claim -- the same message and the same Ed25519
// proof over ClaimProof/1 the owner presents on WorkerControl -- which the host verifies against
// the same control key the Runtime verifies it against. A call whose record_owner_epoch is
// older than the highest Claim the host has forwarded is refused. A host call is scoped to the
// worker lifetime the proof names, never to one control stream: control_stream_epoch is 0 on
// the Claim and on every request carried here, and nonzero refuses. The request's
// record_owner_epoch and worker_boot_id MUST equal the Claim's.
//
// REPLAY: each lane is durable in the host ledger and idempotent on its identity fields. A
// reconnecting owner re-issues the exact call and receives the ledger's answer (the same
// terminal status, or REPLAYED) or attaches to the operation still in flight. Progress is the
// owner's observed signal; no lane needs a wall-clock guess.
//
// Locally PodHost is not a service: the daemon calls the same host functions in-process.
// ---------------------------------------------------------------------------
type PodHostClient interface {
	PreparePackageSet(ctx context.Context, in *PreparePackageSetCall, opts ...grpc.CallOption) (grpc.ServerStreamingClient[PrepareEvent], error)
	PrepareLocalPackage(ctx context.Context, in *PrepareLocalPackageCall, opts ...grpc.CallOption) (grpc.ServerStreamingClient[PrepareEvent], error)
	PreparePrivatePlacement(ctx context.Context, in *PreparePrivatePlacementCall, opts ...grpc.CallOption) (grpc.ServerStreamingClient[PrepareEvent], error)
	ModelSourceFile(ctx context.Context, in *ModelSourceFileCall, opts ...grpc.CallOption) (grpc.ServerStreamingClient[ModelSourceFileStatus], error)
	ModelSourcePrepare(ctx context.Context, in *ModelSourcePrepareCall, opts ...grpc.CallOption) (*ModelSourcePrepared, error)
	LocalPackageFetch(ctx context.Context, in *LocalPackageFetchCall, opts ...grpc.CallOption) (grpc.ServerStreamingClient[LocalPackageFileStatus], error)
	LocalPackageAbort(ctx context.Context, in *LocalPackageAbortCall, opts ...grpc.CallOption) (*LocalPackageAbortStatus, error)
	WeightsTransfer(ctx context.Context, in *WeightsTransferCall, opts ...grpc.CallOption) (grpc.ServerStreamingClient[WeightsTransferStatus], error)
}

type podHostClient struct {
	cc grpc.ClientConnInterface
}

func NewPodHostClient(cc grpc.ClientConnInterface) PodHostClient {
	return &podHostClient{cc}
}

func (c *podHostClient) PreparePackageSet(ctx context.Context, in *PreparePackageSetCall, opts ...grpc.CallOption) (grpc.ServerStreamingClient[PrepareEvent], error) {
	cOpts := append([]grpc.CallOption{grpc.StaticMethod()}, opts...)
	stream, err := c.cc.NewStream(ctx, &PodHost_ServiceDesc.Streams[0], PodHost_PreparePackageSet_FullMethodName, cOpts...)
	if err != nil {
		return nil, err
	}
	x := &grpc.GenericClientStream[PreparePackageSetCall, PrepareEvent]{ClientStream: stream}
	if err := x.ClientStream.SendMsg(in); err != nil {
		return nil, err
	}
	if err := x.ClientStream.CloseSend(); err != nil {
		return nil, err
	}
	return x, nil
}

// This type alias is provided for backwards compatibility with existing code that references the prior non-generic stream type by name.
type PodHost_PreparePackageSetClient = grpc.ServerStreamingClient[PrepareEvent]

func (c *podHostClient) PrepareLocalPackage(ctx context.Context, in *PrepareLocalPackageCall, opts ...grpc.CallOption) (grpc.ServerStreamingClient[PrepareEvent], error) {
	cOpts := append([]grpc.CallOption{grpc.StaticMethod()}, opts...)
	stream, err := c.cc.NewStream(ctx, &PodHost_ServiceDesc.Streams[1], PodHost_PrepareLocalPackage_FullMethodName, cOpts...)
	if err != nil {
		return nil, err
	}
	x := &grpc.GenericClientStream[PrepareLocalPackageCall, PrepareEvent]{ClientStream: stream}
	if err := x.ClientStream.SendMsg(in); err != nil {
		return nil, err
	}
	if err := x.ClientStream.CloseSend(); err != nil {
		return nil, err
	}
	return x, nil
}

// This type alias is provided for backwards compatibility with existing code that references the prior non-generic stream type by name.
type PodHost_PrepareLocalPackageClient = grpc.ServerStreamingClient[PrepareEvent]

func (c *podHostClient) PreparePrivatePlacement(ctx context.Context, in *PreparePrivatePlacementCall, opts ...grpc.CallOption) (grpc.ServerStreamingClient[PrepareEvent], error) {
	cOpts := append([]grpc.CallOption{grpc.StaticMethod()}, opts...)
	stream, err := c.cc.NewStream(ctx, &PodHost_ServiceDesc.Streams[2], PodHost_PreparePrivatePlacement_FullMethodName, cOpts...)
	if err != nil {
		return nil, err
	}
	x := &grpc.GenericClientStream[PreparePrivatePlacementCall, PrepareEvent]{ClientStream: stream}
	if err := x.ClientStream.SendMsg(in); err != nil {
		return nil, err
	}
	if err := x.ClientStream.CloseSend(); err != nil {
		return nil, err
	}
	return x, nil
}

// This type alias is provided for backwards compatibility with existing code that references the prior non-generic stream type by name.
type PodHost_PreparePrivatePlacementClient = grpc.ServerStreamingClient[PrepareEvent]

func (c *podHostClient) ModelSourceFile(ctx context.Context, in *ModelSourceFileCall, opts ...grpc.CallOption) (grpc.ServerStreamingClient[ModelSourceFileStatus], error) {
	cOpts := append([]grpc.CallOption{grpc.StaticMethod()}, opts...)
	stream, err := c.cc.NewStream(ctx, &PodHost_ServiceDesc.Streams[3], PodHost_ModelSourceFile_FullMethodName, cOpts...)
	if err != nil {
		return nil, err
	}
	x := &grpc.GenericClientStream[ModelSourceFileCall, ModelSourceFileStatus]{ClientStream: stream}
	if err := x.ClientStream.SendMsg(in); err != nil {
		return nil, err
	}
	if err := x.ClientStream.CloseSend(); err != nil {
		return nil, err
	}
	return x, nil
}

// This type alias is provided for backwards compatibility with existing code that references the prior non-generic stream type by name.
type PodHost_ModelSourceFileClient = grpc.ServerStreamingClient[ModelSourceFileStatus]

func (c *podHostClient) ModelSourcePrepare(ctx context.Context, in *ModelSourcePrepareCall, opts ...grpc.CallOption) (*ModelSourcePrepared, error) {
	cOpts := append([]grpc.CallOption{grpc.StaticMethod()}, opts...)
	out := new(ModelSourcePrepared)
	err := c.cc.Invoke(ctx, PodHost_ModelSourcePrepare_FullMethodName, in, out, cOpts...)
	if err != nil {
		return nil, err
	}
	return out, nil
}

func (c *podHostClient) LocalPackageFetch(ctx context.Context, in *LocalPackageFetchCall, opts ...grpc.CallOption) (grpc.ServerStreamingClient[LocalPackageFileStatus], error) {
	cOpts := append([]grpc.CallOption{grpc.StaticMethod()}, opts...)
	stream, err := c.cc.NewStream(ctx, &PodHost_ServiceDesc.Streams[4], PodHost_LocalPackageFetch_FullMethodName, cOpts...)
	if err != nil {
		return nil, err
	}
	x := &grpc.GenericClientStream[LocalPackageFetchCall, LocalPackageFileStatus]{ClientStream: stream}
	if err := x.ClientStream.SendMsg(in); err != nil {
		return nil, err
	}
	if err := x.ClientStream.CloseSend(); err != nil {
		return nil, err
	}
	return x, nil
}

// This type alias is provided for backwards compatibility with existing code that references the prior non-generic stream type by name.
type PodHost_LocalPackageFetchClient = grpc.ServerStreamingClient[LocalPackageFileStatus]

func (c *podHostClient) LocalPackageAbort(ctx context.Context, in *LocalPackageAbortCall, opts ...grpc.CallOption) (*LocalPackageAbortStatus, error) {
	cOpts := append([]grpc.CallOption{grpc.StaticMethod()}, opts...)
	out := new(LocalPackageAbortStatus)
	err := c.cc.Invoke(ctx, PodHost_LocalPackageAbort_FullMethodName, in, out, cOpts...)
	if err != nil {
		return nil, err
	}
	return out, nil
}

func (c *podHostClient) WeightsTransfer(ctx context.Context, in *WeightsTransferCall, opts ...grpc.CallOption) (grpc.ServerStreamingClient[WeightsTransferStatus], error) {
	cOpts := append([]grpc.CallOption{grpc.StaticMethod()}, opts...)
	stream, err := c.cc.NewStream(ctx, &PodHost_ServiceDesc.Streams[5], PodHost_WeightsTransfer_FullMethodName, cOpts...)
	if err != nil {
		return nil, err
	}
	x := &grpc.GenericClientStream[WeightsTransferCall, WeightsTransferStatus]{ClientStream: stream}
	if err := x.ClientStream.SendMsg(in); err != nil {
		return nil, err
	}
	if err := x.ClientStream.CloseSend(); err != nil {
		return nil, err
	}
	return x, nil
}

// This type alias is provided for backwards compatibility with existing code that references the prior non-generic stream type by name.
type PodHost_WeightsTransferClient = grpc.ServerStreamingClient[WeightsTransferStatus]

// PodHostServer is the server API for PodHost service.
// All implementations must embed UnimplementedPodHostServer
// for forward compatibility.
//
// ---------------------------------------------------------------------------
// PodHost -- the RecordOwner <-> pod HOST lane. A second service on the pod's external
// listener, served by pod-supervisor beside WorkerControl. It carries what the host does on the
// owner's behalf and nothing that is control: the owner asks the host to PREPARE (download,
// verify, obtain the PlacementSet bytes from the Runtime's loopback preparation), then sends
// those exact bytes ITSELF as DesiredWorkerState{placement_set} on WorkerControl. Remote and
// local are the same three steps -- prepare, placement_set bytes, desired_state -- and a
// multi-GiB materialization never sits on the control stream's read loop, where before 21 it
// head-of-line blocked every offer, ack, and cancel behind it.
//
// AUTHORITY: every call opens with the owner's Claim -- the same message and the same Ed25519
// proof over ClaimProof/1 the owner presents on WorkerControl -- which the host verifies against
// the same control key the Runtime verifies it against. A call whose record_owner_epoch is
// older than the highest Claim the host has forwarded is refused. A host call is scoped to the
// worker lifetime the proof names, never to one control stream: control_stream_epoch is 0 on
// the Claim and on every request carried here, and nonzero refuses. The request's
// record_owner_epoch and worker_boot_id MUST equal the Claim's.
//
// REPLAY: each lane is durable in the host ledger and idempotent on its identity fields. A
// reconnecting owner re-issues the exact call and receives the ledger's answer (the same
// terminal status, or REPLAYED) or attaches to the operation still in flight. Progress is the
// owner's observed signal; no lane needs a wall-clock guess.
//
// Locally PodHost is not a service: the daemon calls the same host functions in-process.
// ---------------------------------------------------------------------------
type PodHostServer interface {
	PreparePackageSet(*PreparePackageSetCall, grpc.ServerStreamingServer[PrepareEvent]) error
	PrepareLocalPackage(*PrepareLocalPackageCall, grpc.ServerStreamingServer[PrepareEvent]) error
	PreparePrivatePlacement(*PreparePrivatePlacementCall, grpc.ServerStreamingServer[PrepareEvent]) error
	ModelSourceFile(*ModelSourceFileCall, grpc.ServerStreamingServer[ModelSourceFileStatus]) error
	ModelSourcePrepare(context.Context, *ModelSourcePrepareCall) (*ModelSourcePrepared, error)
	LocalPackageFetch(*LocalPackageFetchCall, grpc.ServerStreamingServer[LocalPackageFileStatus]) error
	LocalPackageAbort(context.Context, *LocalPackageAbortCall) (*LocalPackageAbortStatus, error)
	WeightsTransfer(*WeightsTransferCall, grpc.ServerStreamingServer[WeightsTransferStatus]) error
	mustEmbedUnimplementedPodHostServer()
}

// UnimplementedPodHostServer must be embedded to have
// forward compatible implementations.
//
// NOTE: this should be embedded by value instead of pointer to avoid a nil
// pointer dereference when methods are called.
type UnimplementedPodHostServer struct{}

func (UnimplementedPodHostServer) PreparePackageSet(*PreparePackageSetCall, grpc.ServerStreamingServer[PrepareEvent]) error {
	return status.Error(codes.Unimplemented, "method PreparePackageSet not implemented")
}
func (UnimplementedPodHostServer) PrepareLocalPackage(*PrepareLocalPackageCall, grpc.ServerStreamingServer[PrepareEvent]) error {
	return status.Error(codes.Unimplemented, "method PrepareLocalPackage not implemented")
}
func (UnimplementedPodHostServer) PreparePrivatePlacement(*PreparePrivatePlacementCall, grpc.ServerStreamingServer[PrepareEvent]) error {
	return status.Error(codes.Unimplemented, "method PreparePrivatePlacement not implemented")
}
func (UnimplementedPodHostServer) ModelSourceFile(*ModelSourceFileCall, grpc.ServerStreamingServer[ModelSourceFileStatus]) error {
	return status.Error(codes.Unimplemented, "method ModelSourceFile not implemented")
}
func (UnimplementedPodHostServer) ModelSourcePrepare(context.Context, *ModelSourcePrepareCall) (*ModelSourcePrepared, error) {
	return nil, status.Error(codes.Unimplemented, "method ModelSourcePrepare not implemented")
}
func (UnimplementedPodHostServer) LocalPackageFetch(*LocalPackageFetchCall, grpc.ServerStreamingServer[LocalPackageFileStatus]) error {
	return status.Error(codes.Unimplemented, "method LocalPackageFetch not implemented")
}
func (UnimplementedPodHostServer) LocalPackageAbort(context.Context, *LocalPackageAbortCall) (*LocalPackageAbortStatus, error) {
	return nil, status.Error(codes.Unimplemented, "method LocalPackageAbort not implemented")
}
func (UnimplementedPodHostServer) WeightsTransfer(*WeightsTransferCall, grpc.ServerStreamingServer[WeightsTransferStatus]) error {
	return status.Error(codes.Unimplemented, "method WeightsTransfer not implemented")
}
func (UnimplementedPodHostServer) mustEmbedUnimplementedPodHostServer() {}
func (UnimplementedPodHostServer) testEmbeddedByValue()                 {}

// UnsafePodHostServer may be embedded to opt out of forward compatibility for this service.
// Use of this interface is not recommended, as added methods to PodHostServer will
// result in compilation errors.
type UnsafePodHostServer interface {
	mustEmbedUnimplementedPodHostServer()
}

func RegisterPodHostServer(s grpc.ServiceRegistrar, srv PodHostServer) {
	// If the following call panics, it indicates UnimplementedPodHostServer was
	// embedded by pointer and is nil.  This will cause panics if an
	// unimplemented method is ever invoked, so we test this at initialization
	// time to prevent it from happening at runtime later due to I/O.
	if t, ok := srv.(interface{ testEmbeddedByValue() }); ok {
		t.testEmbeddedByValue()
	}
	s.RegisterService(&PodHost_ServiceDesc, srv)
}

func _PodHost_PreparePackageSet_Handler(srv interface{}, stream grpc.ServerStream) error {
	m := new(PreparePackageSetCall)
	if err := stream.RecvMsg(m); err != nil {
		return err
	}
	return srv.(PodHostServer).PreparePackageSet(m, &grpc.GenericServerStream[PreparePackageSetCall, PrepareEvent]{ServerStream: stream})
}

// This type alias is provided for backwards compatibility with existing code that references the prior non-generic stream type by name.
type PodHost_PreparePackageSetServer = grpc.ServerStreamingServer[PrepareEvent]

func _PodHost_PrepareLocalPackage_Handler(srv interface{}, stream grpc.ServerStream) error {
	m := new(PrepareLocalPackageCall)
	if err := stream.RecvMsg(m); err != nil {
		return err
	}
	return srv.(PodHostServer).PrepareLocalPackage(m, &grpc.GenericServerStream[PrepareLocalPackageCall, PrepareEvent]{ServerStream: stream})
}

// This type alias is provided for backwards compatibility with existing code that references the prior non-generic stream type by name.
type PodHost_PrepareLocalPackageServer = grpc.ServerStreamingServer[PrepareEvent]

func _PodHost_PreparePrivatePlacement_Handler(srv interface{}, stream grpc.ServerStream) error {
	m := new(PreparePrivatePlacementCall)
	if err := stream.RecvMsg(m); err != nil {
		return err
	}
	return srv.(PodHostServer).PreparePrivatePlacement(m, &grpc.GenericServerStream[PreparePrivatePlacementCall, PrepareEvent]{ServerStream: stream})
}

// This type alias is provided for backwards compatibility with existing code that references the prior non-generic stream type by name.
type PodHost_PreparePrivatePlacementServer = grpc.ServerStreamingServer[PrepareEvent]

func _PodHost_ModelSourceFile_Handler(srv interface{}, stream grpc.ServerStream) error {
	m := new(ModelSourceFileCall)
	if err := stream.RecvMsg(m); err != nil {
		return err
	}
	return srv.(PodHostServer).ModelSourceFile(m, &grpc.GenericServerStream[ModelSourceFileCall, ModelSourceFileStatus]{ServerStream: stream})
}

// This type alias is provided for backwards compatibility with existing code that references the prior non-generic stream type by name.
type PodHost_ModelSourceFileServer = grpc.ServerStreamingServer[ModelSourceFileStatus]

func _PodHost_ModelSourcePrepare_Handler(srv interface{}, ctx context.Context, dec func(interface{}) error, interceptor grpc.UnaryServerInterceptor) (interface{}, error) {
	in := new(ModelSourcePrepareCall)
	if err := dec(in); err != nil {
		return nil, err
	}
	if interceptor == nil {
		return srv.(PodHostServer).ModelSourcePrepare(ctx, in)
	}
	info := &grpc.UnaryServerInfo{
		Server:     srv,
		FullMethod: PodHost_ModelSourcePrepare_FullMethodName,
	}
	handler := func(ctx context.Context, req interface{}) (interface{}, error) {
		return srv.(PodHostServer).ModelSourcePrepare(ctx, req.(*ModelSourcePrepareCall))
	}
	return interceptor(ctx, in, info, handler)
}

func _PodHost_LocalPackageFetch_Handler(srv interface{}, stream grpc.ServerStream) error {
	m := new(LocalPackageFetchCall)
	if err := stream.RecvMsg(m); err != nil {
		return err
	}
	return srv.(PodHostServer).LocalPackageFetch(m, &grpc.GenericServerStream[LocalPackageFetchCall, LocalPackageFileStatus]{ServerStream: stream})
}

// This type alias is provided for backwards compatibility with existing code that references the prior non-generic stream type by name.
type PodHost_LocalPackageFetchServer = grpc.ServerStreamingServer[LocalPackageFileStatus]

func _PodHost_LocalPackageAbort_Handler(srv interface{}, ctx context.Context, dec func(interface{}) error, interceptor grpc.UnaryServerInterceptor) (interface{}, error) {
	in := new(LocalPackageAbortCall)
	if err := dec(in); err != nil {
		return nil, err
	}
	if interceptor == nil {
		return srv.(PodHostServer).LocalPackageAbort(ctx, in)
	}
	info := &grpc.UnaryServerInfo{
		Server:     srv,
		FullMethod: PodHost_LocalPackageAbort_FullMethodName,
	}
	handler := func(ctx context.Context, req interface{}) (interface{}, error) {
		return srv.(PodHostServer).LocalPackageAbort(ctx, req.(*LocalPackageAbortCall))
	}
	return interceptor(ctx, in, info, handler)
}

func _PodHost_WeightsTransfer_Handler(srv interface{}, stream grpc.ServerStream) error {
	m := new(WeightsTransferCall)
	if err := stream.RecvMsg(m); err != nil {
		return err
	}
	return srv.(PodHostServer).WeightsTransfer(m, &grpc.GenericServerStream[WeightsTransferCall, WeightsTransferStatus]{ServerStream: stream})
}

// This type alias is provided for backwards compatibility with existing code that references the prior non-generic stream type by name.
type PodHost_WeightsTransferServer = grpc.ServerStreamingServer[WeightsTransferStatus]

// PodHost_ServiceDesc is the grpc.ServiceDesc for PodHost service.
// It's only intended for direct use with grpc.RegisterService,
// and not to be introspected or modified (even as a copy)
var PodHost_ServiceDesc = grpc.ServiceDesc{
	ServiceName: "cozy.worker.v1.PodHost",
	HandlerType: (*PodHostServer)(nil),
	Methods: []grpc.MethodDesc{
		{
			MethodName: "ModelSourcePrepare",
			Handler:    _PodHost_ModelSourcePrepare_Handler,
		},
		{
			MethodName: "LocalPackageAbort",
			Handler:    _PodHost_LocalPackageAbort_Handler,
		},
	},
	Streams: []grpc.StreamDesc{
		{
			StreamName:    "PreparePackageSet",
			Handler:       _PodHost_PreparePackageSet_Handler,
			ServerStreams: true,
		},
		{
			StreamName:    "PrepareLocalPackage",
			Handler:       _PodHost_PrepareLocalPackage_Handler,
			ServerStreams: true,
		},
		{
			StreamName:    "PreparePrivatePlacement",
			Handler:       _PodHost_PreparePrivatePlacement_Handler,
			ServerStreams: true,
		},
		{
			StreamName:    "ModelSourceFile",
			Handler:       _PodHost_ModelSourceFile_Handler,
			ServerStreams: true,
		},
		{
			StreamName:    "LocalPackageFetch",
			Handler:       _PodHost_LocalPackageFetch_Handler,
			ServerStreams: true,
		},
		{
			StreamName:    "WeightsTransfer",
			Handler:       _PodHost_WeightsTransfer_Handler,
			ServerStreams: true,
		},
	},
	Metadata: "cozy/worker/v1/worker.proto",
}
