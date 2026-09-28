package host

import (
	"context"
	"crypto/ed25519"

	"github.com/cozy-creator/cozy/internal/host/outputs"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

var _ outputs.Source = (*Machine)(nil)

// Keys is the authorized key set and a channel closed when it changes.
func (m *Machine) Keys() ([]ed25519.PublicKey, <-chan struct{}) {
	m.claims.mu.Lock()
	defer m.claims.mu.Unlock()
	return append([]ed25519.PublicKey(nil), m.claims.authorized...), m.claims.changed
}

// request names run n's execution, asking the Runtime by number.
func (m *Machine) request(ctx context.Context, run uint64) (string, error) {
	if run == 0 {
		return "", outputs.ErrNotFound
	}
	conn, err := m.runtime(ctx)
	if err != nil {
		return "", err
	}
	list, err := pb.NewWorkerControlClient(conn).ListMachineExecutions(ctx, &pb.MachineExecutionListQuery{
		Claim: m.claims.claim, AfterNumber: run - 1, Limit: 1})
	switch {
	case status.Code(err) == codes.Unimplemented:
		return "", outputs.ErrUpdateRequired
	case err != nil:
		return "", err
	case len(list.Executions) == 0 || list.Executions[0].Number != run:
		return "", outputs.ErrNotFound
	}
	return list.Executions[0].RequestId, nil
}

// Entries and Open read run outputs through the shared fold (internal/runoutputs); until it
// lands they answer ErrUpdateRequired after resolving the run.
func (m *Machine) Entries(ctx context.Context, run uint64, _ uint64) ([]outputs.Entry, error) {
	if _, err := m.request(ctx, run); err != nil {
		return nil, err
	}
	return nil, outputs.ErrUpdateRequired
}

func (m *Machine) Open(uint64, string, int) (outputs.Snapshot, error) {
	return outputs.Snapshot{}, outputs.ErrUpdateRequired
}
