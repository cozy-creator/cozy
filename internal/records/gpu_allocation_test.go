package records

import (
	"path/filepath"
	"strings"
	"testing"

	"github.com/cozy-creator/cozy/internal/exit"
	v1 "github.com/cozy-creator/cozy/protocol/cozy/machine/v1"
)

// acceptRuns submits ids in order, each accepted by its machine as run numbers[id].
func acceptRuns(t *testing.T, s *Store, numbers map[string]uint64, ids ...string) {
	t.Helper()
	for _, id := range ids {
		for _, step := range []func() *exit.Error{
			func() *exit.Error {
				_, _, problem := s.Submit(Request{ID: id, IdemKey: id, Package: "local/test", Entrypoint: "main", Kind: "call",
					Payload: []byte(`{}`), BodyDigest: "sha256:" + strings.Repeat("1", 64), MachineExecutionObserver: true})
				return problem
			},
			func() *exit.Error { return s.LinkMachineExecution(id, "machine") },
			func() *exit.Error {
				return s.AcceptRunV1(id, "machine", &v1.RunState{Id: id, Number: numbers[id], State: "running", Attempt: 1})
			},
		} {
			if problem := step(); problem != nil {
				t.Fatal(problem.Message)
			}
		}
	}
}

// A machine's allocation events become the run's GPU wait (behind the holder) and then its
// allocation state: held on its GPUs, then released.
func TestMachineGPUAllocationEventsSayWaitingHeldReleased(t *testing.T) {
	s, problem := Open(filepath.Join(t.TempDir(), "records.sqlite"))
	if problem != nil {
		t.Fatal(problem)
	}
	defer s.Close()
	acceptRuns(t, s, map[string]uint64{"holder": 1, "run": 1}, "holder", "run")
	gpu := func(seq uint64, allocation *v1.GpuAllocation) {
		event := &v1.RunEvent{Sequence: seq, AtMs: 1, Event: &v1.RunEvent_Gpu{Gpu: allocation}}
		if problem := s.ObserveRunV1("run", event, nil); problem != nil {
			t.Fatal(problem.Message)
		}
	}
	state := func() *GPUAllocation {
		allocation, problem := s.MachineGPUAllocation("run")
		if problem != nil {
			t.Fatal(problem.Message)
		}
		return allocation
	}
	if state() != nil {
		t.Fatal("a run with no GPU step has no allocation")
	}
	gpu(1, &v1.GpuAllocation{State: "wait", Width: 4, Behind: "holder"})
	wait, problem := s.MachineGPUWait("run")
	if problem != nil || wait == nil || wait.Width != 4 || len(wait.BlockedBy) != 1 || wait.BlockedBy[0] != "holder" {
		t.Fatalf("the wait reads %+v (%v)", wait, problem)
	}
	gpu(2, &v1.GpuAllocation{State: "grant", Width: 4, Gpus: []uint32{0, 1, 2, 3}})
	if wait, _ := s.MachineGPUWait("run"); wait != nil {
		t.Fatalf("a granted allocation still waits: %+v", wait)
	}
	if got := state(); got.State != "held" || len(got.GPUs) != 4 || got.GPUs[3] != 3 {
		t.Fatalf("the allocation reads %+v", got)
	}
	gpu(3, &v1.GpuAllocation{State: "release", Width: 4, Gpus: []uint32{0, 1, 2, 3}})
	if got := state(); got.State != "released" || len(got.GPUs) != 0 {
		t.Fatalf("the released allocation reads %+v", got)
	}
}

// Two runs raced to the machine: it accepted the later-numbered one first, and its GPU queue
// follows that order. The run that took GPUs while the other was not ready yields them to it.
func TestMachineAcceptanceOrdersTheGPUQueueAndAYieldNamesItsRun(t *testing.T) {
	s, problem := Open(filepath.Join(t.TempDir(), "records.sqlite"))
	if problem != nil {
		t.Fatal(problem)
	}
	defer s.Close()
	acceptRuns(t, s, map[string]uint64{"first": 8, "second": 7}, "first", "second")
	first, _ := s.MachineRunNumber("first")
	second, _ := s.MachineRunNumber("second")
	if first != 8 || second != 7 {
		t.Fatalf("machine numbers %d and %d", first, second)
	}
	if _, accepted := s.MachineRunNumber("unknown"); accepted {
		t.Fatal("an unaccepted run has a machine number")
	}
	for seq, allocation := range []*v1.GpuAllocation{
		{State: "grant", Width: 2, Gpus: []uint32{0, 1}},
		{State: "release", Width: 2, Gpus: []uint32{0, 1}, Behind: "second"},
	} {
		event := &v1.RunEvent{Sequence: uint64(seq + 1), AtMs: 1, Event: &v1.RunEvent_Gpu{Gpu: allocation}}
		if problem := s.ObserveRunV1("first", event, nil); problem != nil {
			t.Fatal(problem.Message)
		}
	}
	got, problem := s.MachineGPUAllocation("first")
	if problem != nil || got == nil || got.State != "yielded" || got.To != 2 || len(got.GPUs) != 0 {
		t.Fatalf("the yielded allocation reads %+v (%v)", got, problem)
	}
}
