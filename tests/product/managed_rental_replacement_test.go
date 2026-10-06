package producttest

import (
	"bytes"
	"path/filepath"
	"strings"
	"sync"
	"sync/atomic"
	"testing"

	"github.com/cozy-creator/cozy/internal/canonical"
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
	raw, problem := hub.RentalRequestBytes(machine, "cpu", 1, strings.Repeat("1", 64), "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA", hub.DeclaredWorkload{SourceBytes: 2}, nil, "", "")
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
