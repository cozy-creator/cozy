package producttest

import (
	"errors"
	"path/filepath"
	"sort"
	"strings"
	"testing"

	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/hub"
	"github.com/cozy-creator/cozy/internal/records"
	"github.com/cozy-creator/cozy/internal/rentalid"
)

// TestRentalMachineNames is the owner's ruling as behaviour (2026-09-02): a rented machine
// is named by ONE memorable word, unique only among this owner's live rentals — Tensorhub's
// identity for it is the `pr-…` id — and a word returns to the draw once its rental is
// released. The draw is the real store's: the word is reserved inside the transaction that
// records the paid operation, against every rental row and unsettled operation it holds.
func TestRentalMachineNames(t *testing.T) {
	words := rentalid.Words()
	if len(words) < 300 || !sort.StringsAreSorted(words) {
		t.Fatalf("the vocabulary holds %d words", len(words))
	}
	for i, word := range words {
		if !rentalid.ValidMachineName(word) || strings.Contains(word, "-") {
			t.Fatalf("%q is not one plain machine word", word)
		}
		for _, other := range words[i+1:] {
			if other == word || oneLetterApart(word, other) {
				t.Fatalf("%q and %q would be confused when typed from memory", word, other)
			}
		}
	}

	store, problem := records.Open(filepath.Join(t.TempDir(), "records.db"))
	fatal(t, problem)
	defer store.Close()
	const hubURL, fleetCap = "https://hub.invalid", int64(1_000_000_000_000)
	begin := func(key string) string {
		t.Helper()
		op, replay, problem := store.BeginRentalOperation(records.RentalOperation{
			Key: key, Hub: hubURL, Reason: "cozy rental new cpu", HourlyRateUSDMicros: 100_000,
		}, fleetCap, func(machineName string) ([]byte, string, *exit.Error) {
			body, problem := hub.RentalRequestBytes(machineName, "cpu", strings.Repeat("ab", 32),
				"AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA")
			return body, "sha256:" + strings.Repeat("cd", 32), problem
		})
		fatal(t, problem)
		if replay {
			t.Fatalf("operation %s replayed on first use", key)
		}
		request, problem := hub.ParseRentalRequestBytes(op.RequestBody)
		fatal(t, problem)
		if at := sort.SearchStrings(words, request.Name); at >= len(words) || words[at] != request.Name {
			t.Fatalf("operation %s was named %q, not a vocabulary word", key, request.Name)
		}
		return request.Name
	}
	rent := func(id, key string) string {
		t.Helper()
		word := begin(key)
		fatal(t, store.AdvanceRentalOperation(key, id, "ready"))
		fatal(t, store.RecordRental(records.Rental{
			ID: id, MachineName: word, SKU: "cpu", AcceleratorModel: "CPU", HourlyRateUSDMicros: 100_000,
			State: "ready", Hub: hubURL, Address: "127.0.0.1:1", CertPath: id + ".pem",
		}))
		return word
	}

	// Two live rentals never share a word, and an acquisition that has not yet produced its
	// rental row already holds its word against the next one.
	first := rent("pr-first", "op-first")
	second := rent("pr-second", "op-second")
	pending := begin("op-pending")
	if first == second || first == pending || second == pending {
		t.Fatalf("live rentals share a word: %s %s %s", first, second, pending)
	}
	taken, problem := store.MachineNamesInUse()
	fatal(t, problem)
	if !taken[first] || !taken[second] || !taken[pending] || len(taken) != 3 {
		t.Fatalf("names in use = %v", taken)
	}
	if row, problem := store.RentalByMachine(first); problem != nil || row == nil || row.ID != "pr-first" {
		t.Fatalf("`cozy rental end %s` would not find pr-first: %+v %v", first, row, problem)
	}

	// Releasing the first rental frees its word, and only its word.
	if forgotten, problem := store.ForgetRental("pr-first"); problem != nil || !forgotten {
		t.Fatalf("forget: %v %v", forgotten, problem)
	}
	taken, problem = store.MachineNamesInUse()
	fatal(t, problem)
	if taken[first] || !taken[second] || !taken[pending] {
		t.Fatalf("names in use after release = %v", taken)
	}

	// With every other word held by a live rental, the released word is what the next
	// acquisition is named — the draw is over free words, not over history.
	for _, word := range words {
		if word == first || taken[word] {
			continue
		}
		fatal(t, store.RecordRental(records.Rental{
			ID: "pr-" + word, MachineName: word, SKU: "cpu", AcceleratorModel: "CPU",
			HourlyRateUSDMicros: 100_000, State: "ready", Hub: hubURL, Address: "127.0.0.1:1",
			CertPath: word + ".pem",
		}))
	}
	if reused := begin("op-reuse"); reused != first {
		t.Fatalf("the released word %s was not reused; got %s", first, reused)
	}
	_, _, problem = store.BeginRentalOperation(records.RentalOperation{
		Key: "op-exhausted", Hub: hubURL, Reason: "cozy rental new cpu", HourlyRateUSDMicros: 100_000,
	}, fleetCap, func(string) ([]byte, string, *exit.Error) {
		t.Fatal("a request was authored with no free word")
		return nil, "", nil
	})
	if problem == nil || problem.Name != "rental.machine_names_exhausted" {
		t.Fatalf("an exhausted vocabulary did not refuse: %v", problem)
	}
	all := map[string]bool{}
	for _, word := range words {
		all[word] = true
	}
	if _, err := rentalid.NewMachineName(all); !errors.Is(err, rentalid.ErrNoFreeMachineName) {
		t.Fatalf("draw over an exhausted vocabulary: %v", err)
	}
}

func oneLetterApart(a, b string) bool {
	if len(a) == len(b) {
		differ := 0
		for i := range a {
			if a[i] != b[i] {
				differ++
			}
		}
		return differ == 1
	}
	if len(a) > len(b) {
		a, b = b, a
	}
	if len(b)-len(a) != 1 {
		return false
	}
	i := 0
	for i < len(a) && a[i] == b[i] {
		i++
	}
	return a[i:] == b[i+1:]
}
