package machines

import (
	"context"

	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/hub"
	"github.com/cozy-creator/cozy/internal/machineendpoint"
	"github.com/cozy-creator/cozy/internal/rental"
	"github.com/cozy-creator/cozy/internal/workertls"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
)

// DialEndpoint uses the controller's existing authority and ordinary machine transport.
// The public descriptor selects a target; it supplies no credential or Hub authority.
func (r *Resolver) DialEndpoint(ctx context.Context, ep machineendpoint.Endpoint) (*Machine, *exit.Error) {
	return r.dialEndpointAt(ctx, ep, "")
}
func (r *Resolver) dialEndpointAt(ctx context.Context, ep machineendpoint.Endpoint, origin string) (*Machine, *exit.Error) {
	if err := ep.Validate(); err != nil {
		return nil, exit.New(exit.Validation, "invalid machine endpoint: %s", err)
	}
	key, problem := r.Host.ExistingOwner()
	if problem != nil {
		return nil, problem
	}
	pin, err := workertls.ParsePin([]byte(ep.CertificatePEM))
	if err != nil {
		return nil, exit.New(exit.Credential, "invalid machine TLS leaf: %s", err)
	}
	m := &Machine{Name: ep.Name(), owned: true}
	if problem = m.dial(ctx, target{name: ep.Name(), addr: ep.Address, workerID: ep.WorkerID, bootID: ep.WorkerBootID, pin: pin, key: key}, true); problem != nil {
		return nil, problem
	}
	if problem = m.ValidateNewWork(); problem != nil {
		m.Close()
		return nil, problem
	}
	workspace, err := m.Host.GetMachineExecutionWorkspace(ctx, &pb.MachineExecutionWorkspaceQuery{Claim: m.Claim})
	if err != nil {
		m.Close()
		return nil, Transport(err)
	}
	if workspace.ExecutionWorkspaceId != ep.WorkspaceID || workspace.WorkerId != ep.WorkerID || workspace.WorkerBootId != ep.WorkerBootID {
		m.Close()
		return nil, exit.New(exit.Conflict, "explicit machine endpoint no longer names its frozen workspace or boot")
	}
	m.Seen.Workspace.Store(workspace)
	if origin != "" {
		var account *hub.Client
		if r.Hub != nil {
			account = r.Hub(origin)
		}
		cache := r.Host.endpointCache(ep.Name())
		target := accessTarget{addr: ep.Address, worker: ep.WorkerID, leaf: pin.DER(),
			pin:   func() (*workertls.Pin, *exit.Error) { return pin, nil },
			owner: func() (rental.CreatorIdentity, *exit.Error) { return key, nil }}
		resumeAccessCleanup(ctx, cache, target)
		reads, problem := attachAccess(ctx, cache, target, origin, account)
		if problem != nil {
			m.Close()
			return nil, problem
		}
		m.Hub, m.hub = reads, account
	}
	return m, nil
}
