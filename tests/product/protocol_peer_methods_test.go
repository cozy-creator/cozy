package producttest

import (
	"context"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
)

// The shared machine RPCs stay explicitly unsupported on these protocol fixtures.
// Each peer embeds both services, so promoted default methods would be ambiguous.
func (p *idleHoldPeer) GetMachineExecutionWorkspace(ctx context.Context, request *pb.MachineExecutionWorkspaceQuery) (*pb.MachineExecutionWorkspace, error) {
	return p.UnimplementedWorkerControlServer.GetMachineExecutionWorkspace(ctx, request)
}
func (p *idleHoldPeer) SubmitMachineExecution(ctx context.Context, request *pb.MachineExecutionSubmit) (*pb.MachineExecutionReceipt, error) {
	return p.UnimplementedWorkerControlServer.SubmitMachineExecution(ctx, request)
}
func (p *idleHoldPeer) GetMachineExecution(ctx context.Context, request *pb.MachineExecutionQuery) (*pb.MachineExecutionState, error) {
	return p.UnimplementedWorkerControlServer.GetMachineExecution(ctx, request)
}
func (p *idleHoldPeer) ListMachineExecutionEvents(ctx context.Context, request *pb.MachineExecutionEventsQuery) (*pb.MachineExecutionEventPage, error) {
	return p.UnimplementedWorkerControlServer.ListMachineExecutionEvents(ctx, request)
}
func (p *idleHoldPeer) ControlMachineExecution(ctx context.Context, request *pb.MachineExecutionControl) (*pb.MachineExecutionState, error) {
	return p.UnimplementedWorkerControlServer.ControlMachineExecution(ctx, request)
}
func (p *idleHoldPeer) CollectMachineExecution(ctx context.Context, request *pb.MachineExecutionCollect) (*pb.AttemptOutcome, error) {
	return p.UnimplementedWorkerControlServer.CollectMachineExecution(ctx, request)
}
func (p *idleHoldPeer) AcknowledgeMachineExecutionCollection(ctx context.Context, request *pb.MachineExecutionCollectionAck) (*pb.MachineExecutionState, error) {
	return p.UnimplementedWorkerControlServer.AcknowledgeMachineExecutionCollection(ctx, request)
}
func (p *fakePod) GetMachineExecutionWorkspace(ctx context.Context, request *pb.MachineExecutionWorkspaceQuery) (*pb.MachineExecutionWorkspace, error) {
	return p.UnimplementedWorkerControlServer.GetMachineExecutionWorkspace(ctx, request)
}
func (p *fakePod) SubmitMachineExecution(ctx context.Context, request *pb.MachineExecutionSubmit) (*pb.MachineExecutionReceipt, error) {
	return p.UnimplementedWorkerControlServer.SubmitMachineExecution(ctx, request)
}
func (p *fakePod) GetMachineExecution(ctx context.Context, request *pb.MachineExecutionQuery) (*pb.MachineExecutionState, error) {
	return p.UnimplementedWorkerControlServer.GetMachineExecution(ctx, request)
}
func (p *fakePod) ListMachineExecutionEvents(ctx context.Context, request *pb.MachineExecutionEventsQuery) (*pb.MachineExecutionEventPage, error) {
	return p.UnimplementedWorkerControlServer.ListMachineExecutionEvents(ctx, request)
}
func (p *fakePod) ControlMachineExecution(ctx context.Context, request *pb.MachineExecutionControl) (*pb.MachineExecutionState, error) {
	return p.UnimplementedWorkerControlServer.ControlMachineExecution(ctx, request)
}
func (p *fakePod) CollectMachineExecution(ctx context.Context, request *pb.MachineExecutionCollect) (*pb.AttemptOutcome, error) {
	return p.UnimplementedWorkerControlServer.CollectMachineExecution(ctx, request)
}
func (p *fakePod) AcknowledgeMachineExecutionCollection(ctx context.Context, request *pb.MachineExecutionCollectionAck) (*pb.MachineExecutionState, error) {
	return p.UnimplementedWorkerControlServer.AcknowledgeMachineExecutionCollection(ctx, request)
}
func (p *standInPod) GetMachineExecutionWorkspace(ctx context.Context, request *pb.MachineExecutionWorkspaceQuery) (*pb.MachineExecutionWorkspace, error) {
	return p.UnimplementedWorkerControlServer.GetMachineExecutionWorkspace(ctx, request)
}
func (p *standInPod) SubmitMachineExecution(ctx context.Context, request *pb.MachineExecutionSubmit) (*pb.MachineExecutionReceipt, error) {
	return p.UnimplementedWorkerControlServer.SubmitMachineExecution(ctx, request)
}
func (p *standInPod) GetMachineExecution(ctx context.Context, request *pb.MachineExecutionQuery) (*pb.MachineExecutionState, error) {
	return p.UnimplementedWorkerControlServer.GetMachineExecution(ctx, request)
}
func (p *standInPod) ListMachineExecutionEvents(ctx context.Context, request *pb.MachineExecutionEventsQuery) (*pb.MachineExecutionEventPage, error) {
	return p.UnimplementedWorkerControlServer.ListMachineExecutionEvents(ctx, request)
}
func (p *standInPod) ControlMachineExecution(ctx context.Context, request *pb.MachineExecutionControl) (*pb.MachineExecutionState, error) {
	return p.UnimplementedWorkerControlServer.ControlMachineExecution(ctx, request)
}
func (p *standInPod) CollectMachineExecution(ctx context.Context, request *pb.MachineExecutionCollect) (*pb.AttemptOutcome, error) {
	return p.UnimplementedWorkerControlServer.CollectMachineExecution(ctx, request)
}
func (p *standInPod) AcknowledgeMachineExecutionCollection(ctx context.Context, request *pb.MachineExecutionCollectionAck) (*pb.MachineExecutionState, error) {
	return p.UnimplementedWorkerControlServer.AcknowledgeMachineExecutionCollection(ctx, request)
}
