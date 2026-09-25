package producttest

import (
	"bytes"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/config"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/hub"
	"github.com/cozy-creator/cozy/internal/records"
)

func replacementRequest(t *testing.T, store *records.Store, id, worker string) {
	t.Helper()
	_, _, problem := store.Submit(records.Request{ID: id, IdemKey: id, BodyDigest: "sha256:" + strings.Repeat("7", 64),
		Package: "proof/source-producer", Release: "1", Entrypoint: "convert", Kind: "job", Rental: true, Worker: worker, Payload: []byte("{}"), Outputs: "[]", WeightsOutputs: "[]",
		ModelTransfer: &records.ModelTransferIntent{Kind: "model-upload", Destination: "proof/model", Source: "hf://proof/source@" + strings.Repeat("1", 40), SourceSelection: "sha256:" + strings.Repeat("2", 64), SourceProfiles: map[string]string{"input": "source-profile"}, SourceFiles: []records.ModelTransferSourceFile{{Member: "index.json", SHA256: strings.Repeat("3", 64), Length: 2, Header: []byte("{}")}}, Outputs: []records.ModelTransferOutput{{Name: "model"}}}})
	fatal(t, problem)
}
func replacementAuthor(machine string) ([]byte, string, *exit.Error) {
	raw, problem := hub.RentalRequestBytes(machine, "cpu", 1, strings.Repeat("1", 64), "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA", hub.DeclaredWorkload{SourceBytes: 2}, nil)
	if problem != nil {
		return nil, "", problem
	}
	digest, _ := canonical.Spell(canonical.Digest(raw))
	return raw, digest, nil
}

func TestReleasedManagedRentalGetsOneConcurrentReplacement(t *testing.T) {
	path := filepath.Join(t.TempDir(), "creator.sqlite")
	store, problem := records.Open(path)
	fatal(t, problem)
	defer store.Close()
	const request = "job-replacement-cas"
	replacementRequest(t, store, request, "")
	key, problem := store.ManagedRentalOperationKey(request)
	fatal(t, problem)
	first, _, problem := store.BeginRentalOperation(records.RentalOperation{Key: key, Hub: "http://hub.example", HourlyRateUSDMicros: 100000, ManagedRequestID: request}, replacementAuthor)
	fatal(t, problem)
	for _, state := range []string{"acquiring", "attached", "release_requested"} {
		fatal(t, store.AdvanceRentalOperation(first.Key, "pr-old", state))
		same, problem := store.ManagedRentalOperationKey(request)
		fatal(t, problem)
		if same != first.Key {
			t.Fatal("an unreleased paid obligation selected another create key")
		}
	}
	fatal(t, store.AdvanceRentalOperation(first.Key, "pr-old", "released"))
	raced, _, problem := store.BeginRentalOperation(records.RentalOperation{Key: first.Key, Hub: "http://hub.example", HourlyRateUSDMicros: 100000, ManagedRequestID: request}, replacementAuthor)
	if problem == nil || problem.Code != exit.Unavailable || raced.Key != "" {
		t.Fatal("a selector raced by release replayed the settled key before the paid call")
	}

	next, problem := store.ManagedRentalOperationKey(request)
	fatal(t, problem)
	if next == first.Key {
		t.Fatal("released rental reused its immutable paid operation")
	}
	var authors atomic.Int64
	var wg sync.WaitGroup
	errors := make(chan string, 8)
	for range 8 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			other, p := records.Open(path)
			if p != nil {
				errors <- p.Message
				return
			}
			defer other.Close()
			chosen, p := other.ManagedRentalOperationKey(request)
			if p != nil {
				errors <- p.Message
				return
			}
			op, _, p := other.BeginRentalOperation(records.RentalOperation{Key: chosen, Hub: "http://hub.example", HourlyRateUSDMicros: 100000, ManagedRequestID: request}, func(machine string) ([]byte, string, *exit.Error) { authors.Add(1); return replacementAuthor(machine) })
			if p != nil {
				errors <- p.Message
			} else if op.Key != next {
				errors <- "concurrent replacement changed key"
			}
		}()
	}
	wg.Wait()
	close(errors)
	for err := range errors {
		t.Error(err)
	}
	if authors.Load() != 1 {
		t.Fatalf("replacement body was authored %d times, want one durable operation", authors.Load())
	}
	stale, _, problem := store.BeginRentalOperation(records.RentalOperation{Key: next + "-other", Hub: "http://hub.example", HourlyRateUSDMicros: 100000, ManagedRequestID: request}, replacementAuthor)
	if problem == nil || problem.Code != exit.Unavailable || stale.Key != "" {
		t.Fatal("stale selector created another managed acquisition")
	}
	saved, problem := store.RentalOperation(first.Key)
	fatal(t, problem)
	if saved.State != "released" || !bytes.Equal(saved.RequestBody, first.RequestBody) {
		t.Fatal("replacement overwrote the old paid obligation")
	}
}

func TestLostOnlyRentalReacquiresForTheSameSourceJob(t *testing.T) {
	root := t.TempDir()
	port := reservePort(t)
	origin := fmt.Sprintf("http://127.0.0.1:%d", port)
	must(t, os.WriteFile(filepath.Join(root, config.FileName), []byte("tensorhub_url: "+origin+"\ntensorhub_token: rental-idle-test\ndaemon:\n  idle_shutdown_s: 0\n"), 0600))
	peer := newFakeRentalHub(t, port)
	peer.packageReleases = map[string]any{"proof/source-producer@1": rentalReleaseFacts()}
	peer.setSKUs(map[string]any{"name": "cpu", "accelerator_model": "CPU", "accelerator_count": 1, "price_usd_micros_per_hour": 100000, "base_worker_profile": "python3.12-cpu-linux-x86"})
	store, problem := records.Open(filepath.Join(root, "creator.sqlite"))
	fatal(t, problem)
	defer store.Close()
	const request = "job-source-replacement"
	replacementRequest(t, store, request, "rental-lost-only")
	before, problem := store.RequestRow(request)
	fatal(t, problem)
	original, _, problem := store.BeginRentalOperation(records.RentalOperation{Key: "managed-rental-" + request, Hub: origin, HourlyRateUSDMicros: 100000, ManagedRequestID: request}, replacementAuthor)
	fatal(t, problem)
	fatal(t, store.AdvanceRentalOperation(original.Key, "rental-lost-only", "attached"))
	var oldBody struct{ Name string }
	must(t, json.Unmarshal(original.RequestBody, &oldBody))
	peer.add("rental-lost-only", oldBody.Name)
	peer.setState("rental-lost-only", "released", "")
	fatal(t, store.RecordRental(records.Rental{AcceleratorCount: 1, ID: "rental-lost-only", MachineName: oldBody.Name, SKU: "cpu", AcceleratorModel: "CPU", HourlyRateUSDMicros: 100000, ManagedRequestID: request, State: "ready", Hub: origin}))
	var creates atomic.Int64
	peer.rent = func(body map[string]any) map[string]any {
		creates.Add(1)
		return map[string]any{"rental_id": "pr-replacement-only", "name": body["name"], "state": "acquiring", "requested_accelerator_model": "CPU", "accelerator_count": 1, "hourly_rate_usd_micros": 100000}
	}
	startDaemonProcess(t, root)
	defer peer.setState("pr-replacement-only", "failed", "fixture_finished")
	deadline := time.After(20 * time.Second)
	for creates.Load() == 0 {
		select {
		case <-deadline:
			t.Fatalf("original source job did not create replacement; %s", tail(filepath.Join(root, "daemon.log")))
		case <-time.After(20 * time.Millisecond):
		}
	}
	next, problem := store.ManagedRentalOperationKey(request)
	fatal(t, problem)
	op, problem := store.RentalOperation(next)
	fatal(t, problem)
	if op == nil || op.Key == original.Key || op.ManagedRequestID != request {
		t.Fatalf("replacement lost request lineage: %+v", op)
	}
	// A queued source job has not reached any byte/preparation/producer boundary;
	// this arm proves real owner recovery and paid-create identity, not native custody.
	after, problem := store.RequestRow(request)
	fatal(t, problem)
	a, _ := json.Marshal(before.ModelTransfer)
	b, _ := json.Marshal(after.ModelTransfer)
	if after.ID != request || after.State == "failed" || after.BodyDigest != before.BodyDigest || !bytes.Equal(a, b) {
		t.Fatalf("replacement changed or failed the original source request: %+v", after)
	}
	attempts, problem := store.Attempts(request)
	fatal(t, problem)
	if len(attempts) != 0 {
		t.Fatal("provider acquisition invented a producer attempt")
	}
	time.Sleep(200 * time.Millisecond)
	if creates.Load() != 1 {
		t.Fatalf("one lost rental created %d replacements", creates.Load())
	}
}
