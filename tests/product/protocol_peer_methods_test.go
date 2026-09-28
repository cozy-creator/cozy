package producttest

import (
	"context"

	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
	"google.golang.org/grpc"
	"google.golang.org/protobuf/proto"
)

func (p *fakePod) GetMachineExecutionWorkspace(ctx context.Context, request *pb.MachineExecutionWorkspaceQuery) (*pb.MachineExecutionWorkspace, error) {
	machine, ok := p.machine.(machineSubmitPeer)
	if !ok {
		return p.UnimplementedWorkerControlServer.GetMachineExecutionWorkspace(ctx, request)
	}
	workspace, err := machine.GetMachineExecutionWorkspace(ctx, request)
	if err == nil {
		workspace.RunOutputLog = true // every fake machine is a wire-65 Runtime
	}
	if err == nil && request.Describe != nil && workspace.DescribedRelease == nil {
		// The machine reads a release at its own Hub; this pod's Hub is the fixture's.
		p.mu.Lock()
		described := p.releases[request.Describe.Package]
		p.mu.Unlock()
		if described != nil && (request.Describe.Release == "" || request.Describe.Release == described.Release) {
			workspace.DescribedRelease = proto.Clone(described).(*pb.DescribedRelease)
		}
	}
	return workspace, err
}

// machineTriagePeer reads a retained triage bundle for its owner through the Host.
type machineTriagePeer interface {
	ReadMachineExecutionTriage(context.Context, *pb.MachineExecutionTriageQuery) (*pb.MachineExecutionTriage, error)
}

func (p *fakePod) ReadMachineExecutionTriage(ctx context.Context, request *pb.MachineExecutionTriageQuery) (*pb.MachineExecutionTriage, error) {
	if machine, ok := p.machine.(machineTriagePeer); ok {
		return machine.ReadMachineExecutionTriage(ctx, request)
	}
	return p.UnimplementedPodHostServer.ReadMachineExecutionTriage(ctx, request)
}

func (p *fakePod) SubmitMachineExecution(ctx context.Context, request *pb.MachineExecutionSubmit) (*pb.MachineExecutionReceipt, error) {
	if machine, ok := p.machine.(machineSubmitPeer); ok {
		return machine.SubmitMachineExecution(ctx, request)
	}
	return p.UnimplementedWorkerControlServer.SubmitMachineExecution(ctx, request)
}
func (p *fakePod) GetMachineExecution(ctx context.Context, request *pb.MachineExecutionQuery) (*pb.MachineExecutionState, error) {
	if p.machine != nil {
		return p.machine.GetMachineExecution(ctx, request)
	}
	return p.UnimplementedWorkerControlServer.GetMachineExecution(ctx, request)
}
func (p *fakePod) ListMachineExecutionEvents(ctx context.Context, request *pb.MachineExecutionEventsQuery) (*pb.MachineExecutionEventPage, error) {
	if p.machine != nil {
		return p.machine.ListMachineExecutionEvents(ctx, request)
	}
	return p.UnimplementedWorkerControlServer.ListMachineExecutionEvents(ctx, request)
}
func (p *fakePod) ControlMachineExecution(ctx context.Context, request *pb.MachineExecutionControl) (*pb.MachineExecutionState, error) {
	if machine, ok := p.machine.(machineControlPeer); ok {
		return machine.ControlMachineExecution(ctx, request)
	}
	return p.UnimplementedWorkerControlServer.ControlMachineExecution(ctx, request)
}
func (p *fakePod) AcknowledgeMachineExecutionCollection(ctx context.Context, request *pb.MachineExecutionCollectionAck) (*pb.MachineExecutionState, error) {
	if p.machine != nil {
		return p.machine.AcknowledgeMachineExecutionCollection(ctx, request)
	}
	return p.UnimplementedWorkerControlServer.AcknowledgeMachineExecutionCollection(ctx, request)
}

type inputTreeImporter interface {
	ImportInputTree(grpc.ClientStreamingServer[pb.InputTreeImportFrame, pb.NativeByteRetentionResult]) error
}

func (p *fakePod) ImportInputTree(stream grpc.ClientStreamingServer[pb.InputTreeImportFrame, pb.NativeByteRetentionResult]) error {
	if importer, ok := p.machine.(inputTreeImporter); ok {
		return importer.ImportInputTree(stream)
	}
	return p.UnimplementedPodHostServer.ImportInputTree(stream)
}
