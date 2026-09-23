package producttest

import (
	"crypto/ed25519"
	"crypto/rand"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"sync/atomic"
	"testing"

	"github.com/cozy-creator/cozy/internal/cli"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/inputasset"
	"github.com/cozy-creator/cozy/internal/orchestrator"
	"github.com/cozy-creator/cozy/internal/records"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
)

// Hold the peer's preparation answer while another observation takes ownership of
// the request. Its late refusal must not undo execution or destroy retained input.
// Even an aborted dispatch advances the ordinal and supersedes the old preparation;
// its new selection may fail later, but the old waiter cannot settle it.
func TestLatePreparationFailureRespectsAttemptOwnership(t *testing.T) {
	for _, phase := range []string{"queued", "dispatch_aborted", "accepted", "terminal", "closed"} {
		t.Run(phase, func(t *testing.T) {
			public, private, err := ed25519.GenerateKey(rand.Reader)
			must(t, err)
			entered, resume := make(chan struct{}), make(chan struct{})
			var resumeOnce sync.Once
			unblock := func() { resumeOnce.Do(func() { close(resume) }) }
			pod := &fakePod{controlKey: public, serve: true, jobReady: true}
			pod.onJobReady = func(frame *pb.WorkerFrame, send func(*pb.WorkerFrame) error) error {
				close(entered)
				<-resume
				state := frame.GetObservedState()
				revision := state.AcceptedDesiredStateRevision
				state.AcceptedDesiredStateRevision = 0
				state.ConvergedRevision = 0
				state.AdmissionState = pb.AdmissionState_ADMISSION_STATE_CLOSED
				state.AvailableAttemptSlots = 0
				state.JobCapacity = nil
				state.Faults = []*pb.Fault{{Kind: pb.FaultKind_FAULT_KIND_CONFIG_REFUSED,
					Subject: fmt.Sprintf("revision %d", revision), Reason: "late_preparation_refusal", Detail: "held until the attempt observation committed"}}
				return send(frame)
			}
			connection, _ := startFakePod(t, t.TempDir(), pod)
			var releases atomic.Int64
			o := hostOwner(t, "late-preparation-"+phase, rentalWiring(connection, private), func(options *orchestrator.Options) {
				options.ModelTransfers = cli.NewModelTransferOwner(options.Cfg, options.Store, options.Log, nil)
				options.ReleaseManagedRental = func(string) (string, *exit.Error) {
					releases.Add(1)
					return "", nil
				}
			})
			defer unblock()
			asset := filepath.Join(t.TempDir(), "reference.txt")
			must(t, os.WriteFile(asset, []byte("retained input bytes"), 0600))
			sub := outputPublicationSubmission()
			binding, problem := inputasset.Bind(records.AssetBinding{FieldPath: "reference", LocalPath: asset}, 1024)
			fatal(t, problem)
			sub.Assets = []records.AssetBinding{binding}
			id, _, problem := o.c.Submit(sub)
			fatal(t, problem)
			waitUntil(t, "peer waiting to report job preparation", func() bool {
				select {
				case <-entered:
					return true
				default:
					return false
				}
			})
			row, problem := o.store.RequestRow(id)
			fatal(t, problem)
			if row.State != "submitted" && row.State != "queued" {
				t.Fatalf("waiter did not start queued: %s", row.State)
			}
			instance, _, _, problem := o.c.EnsureRental(podRental)
			fatal(t, problem)
			ordinal := int64(0)
			digest := "sha256:" + strings.Repeat("b", 64)
			if phase != "queued" {
				ordinal, problem = o.store.Dispatch(records.Attempt{RequestID: id, SessionID: podBootID,
					InstanceID: instance, InvocationDigest: digest, InvocationCanonical: []byte("{}")})
				fatal(t, problem)
				if phase == "dispatch_aborted" {
					fatal(t, o.store.AbortDispatch(id, ordinal, podBootID, "no offer crossed"))
				} else {
					fatal(t, o.store.OfferDispatch(id, ordinal, podBootID))
					fatal(t, o.store.Accepted(id, ordinal, podBootID))
					if phase != "accepted" {
						_, problem = o.store.AcceptTerminal(records.Terminal{RequestID: id, Attempt: ordinal,
							SessionID: podBootID, InvocationDigest: digest, TerminalID: "out-preparation-race",
							TerminalDigest: "sha256:" + strings.Repeat("c", 64), Status: "SUCCEEDED", Cause: "COMPLETED",
							RequestState: "finalizing"})
						fatal(t, problem)
						fatal(t, o.store.BeginModelTransferFinalization(id))
						if phase == "closed" {
							fatal(t, o.store.Closed(id, ordinal))
						}
					}
				}
			}
			before, problem := o.store.RequestRow(id)
			fatal(t, problem)
			unblock()
			completion := "preparation selection changed; discarding its result"
			if phase == "queued" {
				completion = "FAILED before any offer"
			}
			waitUntil(t, "late preparation waiter answered", func() bool {
				body, _ := os.ReadFile(filepath.Join(o.root, "orchestrator.log"))
				return strings.Contains(string(body), id+" "+completion)
			})
			after, problem := o.store.RequestRow(id)
			fatal(t, problem)
			transfer, problem := o.store.ModelTransferOf(id)
			fatal(t, problem)
			events, problem := o.store.EventsAfter(id, 0, 100)
			fatal(t, problem)
			failures := 0
			for _, event := range events {
				if event.Type == "request.failed" {
					failures++
				}
			}
			if phase == "queued" {
				if after.State != "failed" || transfer.State != "failed" || failures != 1 || transfer.ErrorCode != "placement_config_refused" {
					t.Fatalf("preoffer failure not atomic: request=%s transfer=%+v failures=%d", after.State, transfer, failures)
				}
				waitUntil(t, "failed preparation released provider without removing original", func() bool {
					_, err := os.Stat(row.Assets[0].LocalPath)
					return releases.Load() == 1 && err == nil
				})
			} else {
				expectedTransfer := "pending"
				if phase == "terminal" || phase == "closed" {
					expectedTransfer = "finalizing"
				}
				if after.State != before.State || transfer.State != expectedTransfer || failures != 0 || releases.Load() != 0 {
					t.Fatalf("late waiter overwrote attempt ownership: before=%s after=%s transfer=%s failures=%d releases=%d", before.State, after.State, transfer.State, failures, releases.Load())
				}
				if _, err := os.Stat(row.Assets[0].LocalPath); err != nil {
					t.Fatalf("retained input removed: %v", err)
				}
				workers, problem := o.store.LiveWorkers()
				fatal(t, problem)
				live := false
				for _, worker := range workers {
					if worker.InstanceID == instance {
						live = true
					}
				}
				if !live {
					t.Fatalf("late waiter stopped retained worker: %+v", workers)
				}
				attempt := attemptRow(t, o.store, id, ordinal)
				if attempt.State != phase {
					t.Fatalf("attempt became %s, expected %s", attempt.State, phase)
				}
			}
		})
	}
}

// Aborted dispatch history is not an open execution. A preparation captured after
// that abort can still fail the request, while the preceding selection cannot.
func TestQueuedPreparationFailureAfterAbortedDispatch(t *testing.T) {
	store, problem := records.Open(filepath.Join(t.TempDir(), "creator.sqlite"))
	fatal(t, problem)
	defer store.Close()
	const id = "job-preparation-after-abort"
	fatal(t, store.SpawnWorker(records.WorkerProcess{InstanceID: "worker-preparation-after-abort",
		Package: "proof/producer", WorkerID: "worker", Devices: []string{"cpu"}}))
	original, _, problem := store.Submit(records.Request{ID: id, IdemKey: id,
		Package: "proof/producer", Entrypoint: "quantize", Kind: "job", Payload: []byte("{}"),
		BodyDigest: "sha256:" + strings.Repeat("a", 64), ModelTransfer: outputPublicationSubmission().ModelTransfer})
	fatal(t, problem)
	ordinal, problem := store.Dispatch(records.Attempt{RequestID: id, SessionID: podBootID,
		InstanceID: "worker-preparation-after-abort", InvocationDigest: "sha256:" + strings.Repeat("b", 64),
		InvocationCanonical: []byte("{}")})
	fatal(t, problem)
	fatal(t, store.AbortDispatch(id, ordinal, podBootID, "no offer crossed"))
	current, problem := store.RequestRow(id)
	fatal(t, problem)
	payload := map[string]any{"error_type": "placement_config_refused", "error": "current preparation refused"}
	applied, problem := store.FailQueuedPreparation(original, payload)
	fatal(t, problem)
	if applied {
		t.Fatal("pre-abort preparation settled the newer ordinal")
	}
	applied, problem = store.FailQueuedPreparation(*current, payload)
	fatal(t, problem)
	if !applied {
		t.Fatal("current preparation could not settle an aborted dispatch")
	}
	row, problem := store.RequestRow(id)
	fatal(t, problem)
	transfer, problem := store.ModelTransferOf(id)
	fatal(t, problem)
	events, problem := store.EventsAfter(id, 0, 100)
	fatal(t, problem)
	if row.State != "failed" || row.Ordinal != ordinal || transfer.State != "failed" || transfer.ErrorCode != "placement_config_refused" || len(events) != 1 || events[0].Type != "request.failed" {
		t.Fatalf("current failure not atomic: request=%+v transfer=%+v events=%+v", row, transfer, events)
	}
}

func TestQueuedFailureKeepsOrdinaryFinalization(t *testing.T) {
	store, problem := records.Open(filepath.Join(t.TempDir(), "creator.sqlite"))
	fatal(t, problem)
	defer store.Close()
	const id = "job-finalizing-zero-open-attempts"
	_, _, problem = store.Submit(records.Request{ID: id, IdemKey: id, Package: "proof/producer", Entrypoint: "quantize",
		Payload: []byte("{}"), BodyDigest: "sha256:" + strings.Repeat("a", 64), Kind: "job"})
	fatal(t, problem)
	fatal(t, store.SettleRequest(id, "finalizing"))
	applied, problem := store.FailQueuedRequest(id, map[string]any{"error": "late preparation"})
	fatal(t, problem)
	if applied {
		t.Fatal("zero open attempts let queued failure overwrite finalizing request")
	}
}

// A failed event encoding must roll back both lifecycle rows, so status and watch
// cannot disagree after the database transaction returns an error.
func TestQueuedFailureRollsBackRequestAndTransfer(t *testing.T) {
	store, problem := records.Open(filepath.Join(t.TempDir(), "creator.sqlite"))
	fatal(t, problem)
	defer store.Close()
	const id = "job-preparation-rollback"
	_, _, problem = store.Submit(records.Request{ID: id, IdemKey: id, Package: "proof/producer", Entrypoint: "quantize",
		Payload: []byte("{}"), BodyDigest: "sha256:" + strings.Repeat("a", 64), Kind: "job",
		ModelTransfer: outputPublicationSubmission().ModelTransfer})
	fatal(t, problem)
	payload := map[string]any{"error_type": "preparation_refused", "error": "typed detail", "unencodable": func() {}}
	applied, problem := store.FailQueuedRequest(id, payload)
	if problem == nil || applied {
		t.Fatal("unencodable terminal unexpectedly committed")
	}
	row, problem := store.RequestRow(id)
	fatal(t, problem)
	transfer, problem := store.ModelTransferOf(id)
	fatal(t, problem)
	events, problem := store.EventsAfter(id, 0, 100)
	fatal(t, problem)
	if row.State != "submitted" || transfer.State != "pending" || transfer.ErrorCode != "" || len(events) != 0 {
		t.Fatalf("partial failure committed: request=%s transfer=%+v events=%v", row.State, transfer, events)
	}
}
