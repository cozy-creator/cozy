package producttest

import (
	"bytes"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/cozy-creator/cozy/internal/config"
	"github.com/cozy-creator/cozy/internal/orchestrator"
	"github.com/cozy-creator/cozy/internal/records"
)

// A retained completed job pod is still billable but cannot become fresh capacity
// merely because its last Hub readiness row survived the job's container exit.
func TestSpentJobRentalIsPreservedWhileUnstartedWorkReplans(t *testing.T) {
	for _, mode := range []struct {
		custody string
		pinned  bool
	}{
		{"managed", false}, {"managed", true},
		{"manual-terminal", false}, {"manual-terminal", true},
		{"manual-unpublished", false}, {"manual-unpublished", true},
	} {
		custody, pinned := mode.custody, mode.pinned
		t.Run(fmt.Sprintf("%s/pinned_before_restart_%t", custody, pinned), func(t *testing.T) {
			root := t.TempDir()
			port := reservePort(t)
			origin := fmt.Sprintf("http://127.0.0.1:%d", port)
			must(t, os.WriteFile(filepath.Join(root, config.FileName), []byte("tensorhub_url: "+origin+"\ntensorhub_token: rental-idle-test\nrentals:\n  max_hourly_spend_usd: 1\n  idle_release_s: 0\ndaemon:\n  idle_shutdown_s: 0\n"), 0600))
			peer := newFakeRentalHub(t, port)
			peer.setSKUs(map[string]any{"name": "cpu", "accelerator_model": "CPU", "price_usd_micros_per_hour": 100000, "base_worker_profile": "python3.12-cpu-linux-x86"})
			store, problem := records.Open(filepath.Join(root, "creator.sqlite"))
			fatal(t, problem)
			defer store.Close()
			const old = "job-spent-producer"
			const retained = "rental-spent"
			const next = "job-unstarted-source"
			var publication *records.ModelTransferIntent
			if custody == "manual-unpublished" {
				publication = &records.ModelTransferIntent{Kind: "model-upload", Destination: "proof/model",
					Outputs: []records.ModelTransferOutput{{Name: "model"}}}
			}
			_, _, problem = store.Submit(records.Request{ID: old, IdemKey: old, Kind: "job", Package: "proof/producer", Entrypoint: "convert", Payload: []byte("{}"), BodyDigest: "sha256:" + strings.Repeat("a", 64), Rental: true, Worker: retained, ModelTransfer: publication})
			fatal(t, problem)
			fatal(t, store.SpawnWorker(records.WorkerProcess{InstanceID: "ins-spent", Package: "proof/producer", WorkerID: "remote", Devices: []string{"cpu"}}))
			digest := "sha256:" + strings.Repeat("b", 64)
			ordinal, problem := store.Dispatch(records.Attempt{RequestID: old, SessionID: "boot-spent", InstanceID: "ins-spent", InvocationDigest: digest, InvocationCanonical: []byte("{}")})
			fatal(t, problem)
			fatal(t, store.OfferDispatch(old, ordinal, "boot-spent"))
			fatal(t, store.Accepted(old, ordinal, "boot-spent"))
			_, problem = store.AcceptTerminal(records.Terminal{RequestID: old, Attempt: ordinal, SessionID: "boot-spent", InvocationDigest: digest, TerminalID: "out-spent", TerminalDigest: "sha256:" + strings.Repeat("c", 64), Status: "SUCCEEDED", Cause: "COMPLETED"})
			fatal(t, problem)
			attemptState := "terminal"
			if custody != "manual-terminal" {
				fatal(t, store.Closed(old, ordinal))
				attemptState = "closed"
			}
			if publication != nil {
				fatal(t, store.FailModelTransfer(old, "proof.upload_failed", "retained output"))
			}
			fatal(t, store.SettleRequest(old, "failed")) // old publication failure, successful native execution
			peer.add(retained, "bus")
			managedRequest := ""
			if custody == "managed" {
				managedRequest = old
			}
			fatal(t, store.RecordRental(records.Rental{ID: retained, MachineName: "bus", SKU: "cpu", AcceleratorModel: "CPU", HourlyRateUSDMicros: 100000, ManagedRequestID: managedRequest, State: "ready", Hub: origin, Address: "127.0.0.1:1", CertPath: filepath.Join(root, "retained.pem")}))
			kept := filepath.Join(root, "retained-output-evidence")
			raw := []byte("unbanked output custody is not disposable capacity")
			must(t, os.WriteFile(kept, raw, 0600))
			worker := ""
			if pinned {
				worker = retained
			}
			replacementRequest(t, store, next, worker)
			before, problem := store.RequestRow(next)
			fatal(t, problem)
			var creates atomic.Int64
			peer.rent = func(body map[string]any) map[string]any {
				creates.Add(1)
				return map[string]any{"rental_id": "pr-fresh", "name": body["name"], "state": "acquiring", "requested_accelerator_model": "CPU", "hourly_rate_usd_micros": 100000}
			}
			startDaemonProcess(t, root)
			defer peer.setState("pr-fresh", "failed", "fixture_finished")
			until := time.Now().Add(20 * time.Second)
			for creates.Load() == 0 {
				if time.Now().After(until) {
					t.Fatalf("spent rental blocked fresh capacity; %s", tail(filepath.Join(root, "daemon.log")))
				}
				time.Sleep(20 * time.Millisecond)
			}
			after, problem := store.RequestRow(next)
			fatal(t, problem)
			a, _ := json.Marshal(before.ModelTransfer)
			b, _ := json.Marshal(after.ModelTransfer)
			if after.ID != next || after.Worker == retained || after.State == "failed" || after.Requeues != 0 || after.BodyDigest != before.BodyDigest || !bytes.Equal(a, b) {
				t.Fatalf("unstarted replan changed its request: %+v", after)
			}
			attempts, problem := store.Attempts(next)
			fatal(t, problem)
			if len(attempts) != 0 {
				t.Fatal("replan invented a producer attempt")
			}
			oldAttempt := attemptRow(t, store, old, ordinal)
			oldRequest, problem := store.RequestRow(old)
			fatal(t, problem)
			if oldAttempt.State != attemptState || oldAttempt.TerminalStatus != "SUCCEEDED" || oldRequest.State != "failed" {
				t.Fatal("old producer history changed")
			}
			row, problem := store.RentalRow(retained)
			fatal(t, problem)
			if row == nil || row.State != "ready" || row.ManagedRequestID != managedRequest || row.HourlyRateUSDMicros != 100000 || peer.releases(retained) != 0 {
				t.Fatal("retained rental was released or removed from billing")
			}
			contents, err := os.ReadFile(kept)
			must(t, err)
			if !bytes.Equal(contents, raw) {
				t.Fatal("retained files changed")
			}
			time.Sleep(200 * time.Millisecond)
			if creates.Load() != 1 {
				t.Fatalf("one queued request caused %d creates", creates.Load())
			}
			if custody == "manual-terminal" {
				fatal(t, store.Closed(old, ordinal))
				blocked, problem := store.RentalHasRetainedJob(retained)
				fatal(t, problem)
				if blocked {
					t.Fatal("acknowledging the old outcome permanently retired a manual rental")
				}
			}
		})
	}
}

func TestRentalPurposeKeepsManualAndOwedCustody(t *testing.T) {
	store, problem := records.Open(filepath.Join(t.TempDir(), "creator.sqlite"))
	fatal(t, problem)
	defer store.Close()
	row := records.Rental{ID: "rental-purpose"}
	spent, problem := orchestrator.RentalSpent(store, row)
	fatal(t, problem)
	if spent {
		t.Fatal("a manual rental inherited managed-job retirement")
	}
	replacementRequest(t, store, "job-purpose", row.ID)
	row.ManagedRequestID = "job-purpose"
	for _, state := range []string{"submitted", "finalizing"} {
		fatal(t, store.SettleRequest("job-purpose", state))
		spent, problem = orchestrator.RentalSpent(store, row)
		fatal(t, problem)
		if spent {
			t.Fatalf("%s source/publication custody was called spent", state)
		}
	}
}
