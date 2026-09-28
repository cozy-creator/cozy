package producttest

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"syscall"
	"testing"
	"time"

	"github.com/cozy-creator/cozy/internal/records"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

// hostIdleClock is a Host's own idle release: it releases once its granted deadline
// passes without a renewal. Its Runtime reports nothing here, as after the Runtime
// restart on denken, so a keepalive is the only renewal it gets. Like the Host, it
// answers a replayed request id with the original receipt and no renewal, and refuses
// a keepalive at or after the deadline.
type hostIdleClock struct {
	mu       sync.Mutex
	window   time.Duration
	deadline time.Time
	renewals []time.Time
	receipts map[string]*pb.KeepRentalAliveResult
	replays  int
	early    int
	lapsed   time.Time
}

func (h *hostIdleClock) keepalive(request *pb.KeepRentalAliveRequest) (*pb.KeepRentalAliveResult, error) {
	h.mu.Lock()
	defer h.mu.Unlock()
	now := time.Now()
	if receipt, replayed := h.receipts[request.RequestId]; replayed {
		h.replays++
		return receipt, nil
	}
	if !h.deadline.IsZero() && !now.Before(h.deadline) {
		if h.lapsed.IsZero() {
			h.lapsed = h.deadline
		}
		return nil, status.Error(codes.FailedPrecondition, "rental idle release is already due or committed")
	}
	if n := len(h.renewals); n > 0 && now.Sub(h.renewals[n-1]) < h.window/2 {
		h.early++
	}
	h.deadline = now.Add(h.window)
	h.renewals = append(h.renewals, now)
	receipt := &pb.KeepRentalAliveResult{RequestId: request.RequestId, WorkerId: request.Claim.WorkerId, WorkerBootId: request.Claim.WorkerBootId,
		AcknowledgedAtUnixMs: now.UnixMilli(), IdleDeadlineUnixMs: h.deadline.UnixMilli()}
	if h.receipts == nil {
		h.receipts = map[string]*pb.KeepRentalAliveResult{}
	}
	h.receipts[request.RequestId] = receipt
	return receipt, nil
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
	startDaemonProcess(t, root, pod.path(t, root))
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
	// Each renewal is a fresh request, sent once half the granted window has passed.
	host.mu.Lock()
	replays, early := host.replays, host.early
	host.mu.Unlock()
	if replays != 0 || early != 0 {
		t.Fatalf("renewals replayed %d request id(s) and came %d time(s) before half the window", replays, early)
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

// Accepted work keeps holding its rental across a daemon restart: the new daemon rebuilds
// what is live from its records, keeps it off its own idle clock, and keeps renewing the
// Host's deadline (hakufu, runs 1431 and 1432 submitted before a daemon restart).
func TestAcceptedExecutionHoldsTheRentalAcrossADaemonRestart(t *testing.T) {
	h := newLadderHub(t)
	h.bind(goodLadder())
	machine := &runtimeMachine{}
	host := &hostIdleClock{window: 20 * time.Second}
	root, layout := rentedLadderHome(t, h, &fakePod{machine: machine, keepalive: host.keepalive}, nil)
	first := startDaemonProcess(t, root)
	const key = "held-across-a-restart"
	if code, out := cozyWithin(t, root, time.Minute, "run", ladderPackage+"/generate", "steps=1", "--rental=tessa", "--json", "--idempotency-key", key); code != 0 {
		t.Fatalf("the run was refused [exit %d]: %s", code, out)
	}
	eventually(t, root, "the machine accepting the run", func() bool { return machine.submitted() != nil })
	host.start()
	eventually(t, root, "the first renewal", func() bool {
		renewals, _, _ := host.state()
		return renewals >= 1
	})
	// A release time the log announced before the run arrived is withdrawn once it runs.
	eventually(t, root, "the idle announcement withdrawn", func() bool {
		log, err := os.ReadFile(filepath.Join(root, "daemon.log"))
		return err == nil && (!strings.Contains(string(log), "(tessa) idle since") ||
			strings.Contains(string(log), "(tessa) has work again; no idle release is scheduled"))
	})

	must(t, first.cmd.Process.Signal(syscall.SIGTERM))
	<-first.exited
	before, _, _ := host.state()
	startDaemonProcess(t, root)

	for end := time.Now().Add(3 * host.window); time.Now().Before(end); time.Sleep(200 * time.Millisecond) {
		if _, _, lapsed := host.state(); !lapsed.IsZero() {
			t.Fatalf("the Host's idle deadline %s lapsed across the daemon restart:\n%s", lapsed.Format(time.RFC3339Nano),
				tail(filepath.Join(root, "daemon.log")))
		}
	}
	if renewals, _, _ := host.state(); renewals < before+2 {
		t.Fatalf("the restarted daemon renewed the Host deadline %d time(s) over three windows", renewals-before)
	}
	host.mu.Lock()
	replays, early := host.replays, host.early
	host.mu.Unlock()
	if replays != 0 || early != 0 {
		t.Fatalf("renewals replayed %d request id(s) and came %d time(s) before half the window", replays, early)
	}
	var list struct {
		Rentals []struct {
			Machine   string `json:"machine"`
			Running   int    `json:"running"`
			IdleSince string `json:"idle_since_at"`
		} `json:"rentals"`
	}
	_, raw := cozyWithin(t, root, time.Minute, "rental", "list", "--json")
	if json.Unmarshal([]byte(raw), &list) != nil || len(list.Rentals) != 1 || list.Rentals[0].Running != 1 || list.Rentals[0].IdleSince != "" {
		t.Fatalf("the restarted daemon does not hold the rental for its executing run:\n%s", raw)
	}

	machine.finish()
	store, problem := records.Open(layout.DB)
	fatal(t, problem)
	defer store.Close()
	eventually(t, root, "the run settling", func() bool {
		row, problem := store.RequestByIdempotencyKey(key)
		return problem == nil && row != nil && row.State == "succeeded"
	})
}

// A run whose execution Runtime ends failed holds nothing. On nemuri (run 1474) a crashed
// lane left the worker FAILED with the root open, renewed forever and with no reason shown;
// Runtime now ends a FAILED worker's executions failed, with the worker's reason. The run
// fails with that cause and message, `cozy run show` says so, and the Host's idle deadline
// is no longer renewed.
func TestAFailedExecutionStopsHoldingTheRentalAndShowsItsReason(t *testing.T) {
	h := newLadderHub(t)
	h.bind(goodLadder())
	bundle, err := os.ReadFile(filepath.Join("testdata", "execution_evidence", "runtime-bundle.json"))
	must(t, err)
	machine := &runtimeMachine{triage: bundle}
	host := &hostIdleClock{window: 8 * time.Second}
	root, layout := rentedLadderHome(t, h, &fakePod{machine: machine, keepalive: host.keepalive}, nil)
	startDaemonProcess(t, root)
	const key = "failed-worker-execution"
	if code, out := cozyWithin(t, root, time.Minute, "run", ladderPackage+"/generate", "steps=1", "--rental=tessa", "--json", "--idempotency-key", key); code != 0 {
		t.Fatalf("the run was refused [exit %d]: %s", code, out)
	}
	eventually(t, root, "the machine accepting the run", func() bool { return machine.submitted() != nil })
	host.start()
	eventually(t, root, "the first renewal", func() bool {
		renewals, _, _ := host.state()
		return renewals >= 1
	})
	const reason = "worker FAILED: watchdog lane crashed finishing a child result: KeyError: 'child-3'"
	machine.mu.Lock()
	machine.failure, machine.code = reason, pb.CauseCode_CAUSE_CODE_LOCAL_SAFETY
	machine.mu.Unlock()
	machine.finish()
	store, problem := records.Open(layout.DB)
	fatal(t, problem)
	defer store.Close()
	var row *records.Request
	eventually(t, root, "the run failing", func() bool {
		row, problem = store.RequestByIdempotencyKey(key)
		fatal(t, problem)
		link, problem := store.MachineExecution(row.ID)
		fatal(t, problem)
		return row.State == "failed" && link.Collected
	})
	if _, show := cozyWithin(t, root, time.Minute, "run", "show", row.ID); !strings.Contains(show, "LOCAL_SAFETY") || !strings.Contains(show, reason) {
		t.Fatalf("run show does not give the failure's cause and message:\n%s", show)
	}
	settled, _, _ := host.state()
	time.Sleep(2 * host.window)
	if renewals, _, _ := host.state(); renewals > settled+1 {
		t.Fatalf("the Host deadline was renewed %d more time(s) after the run failed", renewals-settled)
	}
}
