package producttest

import (
	"errors"
	"os"
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
	if len(words) < 5000 || !sort.StringsAreSorted(words) {
		t.Fatalf("the vocabulary holds %d words", len(words))
	}
	// One substitution shares a wildcard key; one insertion deletes to the other word.
	present, wildcard := map[string]bool{}, map[string]string{}
	for _, word := range words {
		if !rentalid.ValidMachineName(word) || strings.Contains(word, "-") {
			t.Fatalf("%q is not one plain machine word", word)
		}
		if present[word] {
			t.Fatalf("%q appears twice", word)
		}
		present[word] = true
	}
	for _, word := range words {
		for i := range word {
			if present[word[:i]+word[i+1:]] {
				t.Fatalf("%q and %q would be confused when typed from memory", word, word[:i]+word[i+1:])
			}
			key := word[:i] + "*" + word[i+1:]
			if other, seen := wildcard[key]; seen {
				t.Fatalf("%q and %q would be confused when typed from memory", other, word)
			}
			wildcard[key] = word
		}
	}

	// cl-098's red arm: the vocabulary is anime given names MINUS the top-100
	// most-popular tier. Every banned name in the generated drop list — and the
	// Goku/Naruto/Luffy household names by name — must be unmintable forever.
	raw, err := os.ReadFile(filepath.Join("testdata", "machine-words-dropped.txt"))
	if err != nil {
		t.Fatalf("the generated drop list is part of the product: %v", err)
	}
	dropped := map[string]bool{}
	for _, line := range strings.Split(string(raw), "\n") {
		if line = strings.TrimSpace(line); line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		dropped[line] = true
		if at := sort.SearchStrings(words, line); at < len(words) && words[at] == line {
			t.Fatalf("%q is a dropped top-100 name, yet a rental could be named by it", line)
		}
	}
	if len(dropped) < 50 {
		t.Fatalf("the drop list holds %d names; the top-100 ban has gone missing", len(dropped))
	}
	for _, famous := range []string{"gokuu", "naruto", "luffy", "lelouch", "levi"} {
		if !dropped[famous] {
			t.Fatalf("%q is not on the drop list; the top-100 tier is wrong", famous)
		}
	}
	// The household spellings too: "goku" is one letter from the banned "gokuu",
	// and the generator refuses any word one letter from a banned name.
	for _, famous := range []string{"goku", "gokuu", "naruto", "luffy", "lelouch", "levi", "zoro"} {
		if at := sort.SearchStrings(words, famous); at < len(words) && words[at] == famous {
			t.Fatalf("a rental could be named %q", famous)
		}
	}

	// Exhausting the vocabulary records one rental per word; tmpfs keeps those thousands
	// of commits from being thousands of disk syncs.
	dir := t.TempDir()
	if shm, err := os.MkdirTemp("/dev/shm", "cozy-names-"); err == nil {
		dir = shm
		t.Cleanup(func() { _ = os.RemoveAll(shm) })
	}
	store, problem := records.Open(filepath.Join(dir, "creator.sqlite"))
	fatal(t, problem)
	defer store.Close()
	const hubURL, fleetCap = "https://hub.invalid", int64(1_000_000_000_000)
	begin := func(key string) string {
		t.Helper()
		op, replay, problem := store.BeginRentalOperation(records.RentalOperation{
			Key: key, Hub: hubURL, Reason: "cozy rental new cpu", HourlyRateUSDMicros: 100_000,
		}, func(machineName string) ([]byte, string, *exit.Error) {
			body, problem := hub.RentalRequestBytes(machineName, "cpu", 1, strings.Repeat("ab", 32),
				"AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA", hub.DeclaredWorkload{}, nil)
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
		fatal(t, store.RecordRental(records.Rental{AcceleratorCount: 1,
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
		fatal(t, store.RecordRental(records.Rental{AcceleratorCount: 1,
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
	}, func(string) ([]byte, string, *exit.Error) {
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
