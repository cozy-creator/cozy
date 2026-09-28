package producttest

import (
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/cozy-creator/cozy/internal/records"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
)

// hostIdleClock is a Host's own idle release: it releases once its granted deadline
// passes without a renewal. Its Runtime reports nothing here, as after the Runtime
// restart on denken, so a keepalive is the only renewal it gets.
type hostIdleClock struct {
	mu       sync.Mutex
	window   time.Duration
	deadline time.Time
	renewals []time.Time
	lapsed   time.Time
}

func (h *hostIdleClock) keepalive(request *pb.KeepRentalAliveRequest) (*pb.KeepRentalAliveResult, error) {
	h.mu.Lock()
	defer h.mu.Unlock()
	now := time.Now()
	if !h.deadline.IsZero() && now.After(h.deadline) && h.lapsed.IsZero() {
		h.lapsed = h.deadline
	}
	h.deadline = now.Add(h.window)
	h.renewals = append(h.renewals, now)
	return &pb.KeepRentalAliveResult{RequestId: request.RequestId, WorkerId: request.Claim.WorkerId, WorkerBootId: request.Claim.WorkerBootId,
		AcknowledgedAtUnixMs: now.UnixMilli(), IdleDeadlineUnixMs: h.deadline.UnixMilli()}, nil
}

// start sets the deadline the Host holds when work arrives it does not itself observe.
func (h *hostIdleClock) start() {
	h.mu.Lock()
	defer h.mu.Unlock()
	if h.deadline.IsZero() {
		h.deadline = time.Now().Add(h.window)
	}
}

func (h *hostIdleClock) state() (int, time.Time, time.Time) {
	h.mu.Lock()
	defer h.mu.Unlock()
	lapsed := h.lapsed
	if lapsed.IsZero() && !h.deadline.IsZero() && time.Now().After(h.deadline) {
		lapsed = h.deadline
	}
	return len(h.renewals), h.deadline, lapsed
}

// Accepted work executing on a rental holds it against the Host's idle release too, after
// a failed Runtime update and whatever the Host's own report says. On denken (fp8 run
// 1413, mxfp8 run 1414) the Host released at its initial deadline while both ran; this
// daemon counted them and held only its own clock. Once the work ends, renewals stop.
func TestAcceptedExecutionHoldsTheRentalAgainstTheHostIdleRelease(t *testing.T) {
	pod := newMaintenancePod(t)
	h := newLadderHub(t)
	h.bind(goodLadder())
	machine := &runtimeMachine{}
	host := &hostIdleClock{window: 8 * time.Second}
	worker := &fakePod{machine: machine, keepalive: host.keepalive}
	root, layout := rentedLadderHome(t, h, worker, nil)
	serveMaintenance(t, layout, worker.controlKey, func(key string, value any) {
		h.mu.Lock()
		h.rentals[podRental][key] = value
		h.mu.Unlock()
	})
	startDaemonProcess(t, root, pod.path())
	runtimeWheel, tensorfsWheel := localRuntimePair(t)

	pod.sftpMode(t, "drop")
	if code, out := cozyWithin(t, root, 5*time.Minute, "rental", "update", "tessa", "--runtime-wheel", runtimeWheel, "--tensorfs-wheel", tensorfsWheel); code == 0 {
		t.Fatalf("the update was expected to fail in transfer [exit %d]: %s", code, out)
	}
	const key = "held-by-its-execution"
	if code, out := cozyWithin(t, root, time.Minute, "run", ladderPackage+"/generate", "steps=1", "--rental=tessa", "--json", "--idempotency-key", key); code != 0 {
		t.Fatalf("the run was refused [exit %d]: %s", code, out)
	}
	eventually(t, root, "the machine accepting the run", func() bool { return machine.submitted() != nil })
	host.start()

	// Several of the Host's windows pass while the run executes; none may lapse. The sweep
	// samples every 2 s, so the window is wide enough for its half-window renewal.
	for end := time.Now().Add(3 * host.window); time.Now().Before(end); time.Sleep(200 * time.Millisecond) {
		if _, _, lapsed := host.state(); !lapsed.IsZero() {
			t.Fatalf("the Host's idle deadline %s lapsed while the accepted run executed:\n%s", lapsed.Format(time.RFC3339Nano),
				tail(filepath.Join(root, "daemon.log")))
		}
	}
	if renewals, _, _ := host.state(); renewals < 3 {
		t.Fatalf("the executing run renewed the Host deadline only %d time(s) over three windows", renewals)
	}
	if _, list := cozyWithin(t, root, time.Minute, "rental", "list"); !strings.Contains(list, "tessa") {
		t.Fatalf("the rental was released while its run executed:\n%s", list)
	}

	// The work ends: nothing renews the Host's clock any longer.
	machine.finish()
	store, problem := records.Open(layout.DB)
	fatal(t, problem)
	defer store.Close()
	eventually(t, root, "the run settling", func() bool {
		row, problem := store.RequestByIdempotencyKey(key)
		return problem == nil && row != nil && row.State == "succeeded"
	})
	settled, _, _ := host.state()
	time.Sleep(2 * host.window)
	if renewals, _, _ := host.state(); renewals > settled+1 {
		t.Fatalf("the Host deadline was renewed %d more time(s) after the run ended", renewals-settled)
	}
}
