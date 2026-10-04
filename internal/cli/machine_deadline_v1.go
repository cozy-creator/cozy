package cli

import (
	"fmt"
	"strings"
	"time"

	"github.com/cozy-creator/cozy/internal/records"
)

// deadlineActor names a cancel that `--timeout` caused. A run canceled by it ends "deadline".
const deadlineActor = "cozy run --timeout"

func deadlineCancel(actor string) bool { return strings.HasPrefix(actor, deadlineActor) }

// enforceDeadlineV1 cancels a cozy.machine.v1 run when its request deadline passes, as a
// worker.v1 machine does from its spec: the daemon records the cancel as `cozy run cancel`
// would, attributed to the deadline, and Control delivers it. A rented run never bills past
// its owner's deadline, whether or not a client still waits. The answer stops the timer.
func (m *machineRuns) enforceDeadlineV1(request records.Request) func() {
	timeout, deadline, problem := m.store.RequestExecutionTiming(request.ID)
	if problem != nil || deadline == 0 {
		return func() {}
	}
	timer := time.AfterFunc(time.Until(time.UnixMilli(int64(deadline))), func() {
		current, problem := m.store.RequestRow(request.ID)
		if problem != nil || current == nil || records.Settled(current.State) {
			return
		}
		actor := fmt.Sprintf("%s %s", deadlineActor, time.Duration(timeout)*time.Millisecond)
		if _, problem := m.store.RequestMachineCancellation(request.ID, actor); problem != nil {
			fmt.Fprintf(m.context.Out, "machine execution %s: its --timeout passed and the cancel was not recorded: %s\n", request.ID, problem.Message)
			return
		}
		_ = m.Control(m.ctx, *current, "cancel")
	})
	return func() { timer.Stop() }
}
