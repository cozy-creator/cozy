package cli

import (
	"context"
	"sync"
	"time"

	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/hub"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
)

// ownerReads paces the owner's finalization reads per publication. A read that settles
// nothing (Hub still finalizing, or no answer at all) is asked again after a doubling
// delay; Runtime's own publication retries back off the same way.
type ownerReads struct {
	mu   sync.Mutex
	next map[string]ownerRead
}

type ownerRead struct {
	at    time.Time
	delay time.Duration
}

const maxOwnerReadDelay = 60 * time.Second

func (o *ownerReads) due(publication string) bool {
	o.mu.Lock()
	defer o.mu.Unlock()
	return !time.Now().Before(o.next[publication].at)
}

func (o *ownerReads) later(publication string) {
	o.mu.Lock()
	defer o.mu.Unlock()
	if o.next == nil {
		o.next = map[string]ownerRead{}
	}
	delay := min(max(2*o.next[publication].delay, time.Second), maxOwnerReadDelay)
	o.next[publication] = ownerRead{at: time.Now().Add(delay), delay: delay}
}

func (o *ownerReads) done(publication string) {
	o.mu.Lock()
	defer o.mu.Unlock()
	delete(o.next, publication)
}

// reconcilePublications settles sent publications the machine can no longer read: its
// publication authorization expired or was revoked. The owner's credential on the run's own
// Hub reads each finalization, and Runtime settles from exactly that answer; Creator never
// interprets it. Until then the run shows "awaiting owner reconciliation".
func (m *machineRuns) reconcilePublications(ctx context.Context, request string, hubOrigin string,
	connection *machineConnection, query *pb.MachineExecutionQuery,
) *exit.Error {
	awaiting, problem := m.store.MachinePublicationsAwaitingOwner(request)
	if problem != nil || len(awaiting) == 0 {
		return problem
	}
	owner := client(m.context.forHub(hubOrigin))
	for _, publication := range awaiting {
		if !m.ownerReads.due(publication.Publication) {
			continue
		}
		ref, problem := hub.ParseRef(publication.Destination)
		if problem != nil {
			return problem
		}
		status, body, problem := owner.ModelFinalizationAnswer(ctx, ref, publication.Publication)
		if problem == nil {
			_, err := connection.Host.ControlMachineExecution(ctx, &pb.MachineExecutionControl{
				Execution: query, CommandId: "reconcile-" + publication.Publication,
				Action: pb.MachineExecutionAction_MACHINE_EXECUTION_ACTION_RECONCILE_PUBLICATION,
				Publication: &pb.MachinePublicationReconciliation{
					CallIndex: publication.CallIndex, HttpStatus: uint32(status), Finalization: body},
			})
			if err == nil {
				m.ownerReads.done(publication.Publication)
				continue
			}
		}
		// Hub is still finalizing, or neither Hub nor the machine answered: wait.
		m.ownerReads.later(publication.Publication)
	}
	return nil
}
