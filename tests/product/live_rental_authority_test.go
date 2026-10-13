package producttest

import (
	"encoding/json"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"

	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/home"
	"github.com/cozy-creator/cozy/internal/hub"
	"github.com/cozy-creator/cozy/internal/reclaim"
	"github.com/cozy-creator/cozy/internal/records"
)

func TestLiveRentalSelectorsIgnoreHistoryAndMissingProjection(t *testing.T) {
	const current, old, other = "pr-live-host", "pr-old-host", "pr-other-host"
	var lock sync.Mutex
	broken, collision := false, false
	wire := map[string]any{"rental_id": current, "name": "shichika", "machine_name": "katsuie",
		"machine_aliases": []string{"adolf"}, "state": "acquiring", "requested_accelerator_model": "CPU",
		"accelerator_count": 1, "hourly_rate_usd_micros": 100000, "worker_address": "127.0.0.1:1",
		"worker_id": "worker-current", "worker_boot_id": "boot-current"}
	root, postsLock, posts, _, _ := runModelCatalog(t, func(mux *http.ServeMux, _ *hub.PackageReleaseDetail) {
		mux.HandleFunc("GET /v1/rentals", func(w http.ResponseWriter, r *http.Request) {
			lock.Lock()
			defer lock.Unlock()
			if broken {
				http.Error(w, "Hub unavailable", 503)
				return
			}
			rows := []any{map[string]any{"rental_id": old, "name": "cooler", "machine_name": "adolf", "state": "failed", "accelerator_count": 1}, wire}
			if collision {
				rows = append(rows, map[string]any{"rental_id": other, "name": "other", "machine_name": "katsuie", "machine_aliases": []string{"adolf"}, "state": "ready", "accelerator_count": 1})
			}
			json.NewEncoder(w).Encode(map[string]any{"rentals": rows})
		})
		mux.HandleFunc("GET /v1/rentals/{id}", func(w http.ResponseWriter, r *http.Request) {
			lock.Lock()
			defer lock.Unlock()
			if broken {
				http.Error(w, "Hub unavailable", 503)
				return
			}
			if r.PathValue("id") == current {
				json.NewEncoder(w).Encode(wire)
				return
			}
			if r.PathValue("id") == old {
				json.NewEncoder(w).Encode(map[string]any{"rental_id": old, "name": "cooler", "machine_name": "adolf", "state": "failed", "accelerator_count": 1})
				return
			}
			http.NotFound(w, r)
		})
	})
	origin := configuredHub(t, root)
	installedHere(t, root, origin, "proof/quantize", "1.0.0")
	layout, problem := home.Open(root)
	fatal(t, problem)
	key := adoptCreatorIdentity(t, layout, current)
	cert, err := os.ReadFile(standInCertificate(t, root))
	must(t, err)
	lock.Lock()
	wire["creator_public_key"], wire["cert_pem"] = key.PublicKey(), string(cert)
	lock.Unlock()
	store, problem := records.Open(layout.DB)
	fatal(t, problem)
	defer store.Close()
	fatal(t, store.RecordRental(records.Rental{ID: old, MachineName: "cooler", StableMachineName: "adolf", State: "failed", Hub: origin, AcceleratorModel: "CPU", AcceleratorCount: 1, HourlyRateUSDMicros: 100000}))
	if code, out := runCozy(t, root, "rental", "show", "AdOlF", "--json"); code != 0 || !strings.Contains(out, "katsuie") || !strings.Contains(out, current) {
		t.Fatalf("live show used history/missing local row [%d]: %s", code, out)
	}
	lock.Lock()
	wire["state"] = "ready"
	lock.Unlock()
	args := []string{"run", "proof/quantize/quantize", "steps=7", "model.dits=proof/source@1.0.0/bf16", "model.shared=proof/source@1.0.0/bf16", "--json", "--full"}
	for _, name := range []string{"katsuie", "adolf", "shichika", current} {
		_, problem := store.ForgetRental(current)
		fatal(t, problem) // Retained key stays; projection is disposable.
		request, _, out := submitRun(t, root, "live-selector-"+name, append(append([]string{}, args...), "--rental="+name)...)
		if request == nil || request.RequestedRental != current {
			t.Fatalf("%s did not select current Hub identity: %+v %s", name, request, out)
		}
	}
	lock.Lock()
	collision = true
	lock.Unlock()
	if code, out := runCozy(t, root, append(append([]string{}, args...), "--rental=adolf")...); code == 0 || !strings.Contains(out, "rental.machine_name_ambiguous") {
		t.Fatalf("two live hosts were not ambiguous [%d]: %s", code, out)
	}
	request, _, out := submitRun(t, root, "live-selector-exact", append(append([]string{}, args...), "--rental="+current)...)
	if request == nil || request.RequestedRental != current {
		t.Fatalf("immutable ID became ambiguous: %+v %s", request, out)
	}
	lock.Lock()
	broken = true
	lock.Unlock()
	// Verbs that read or change the rental itself need its Hub. Work sent to it does not: its
	// dispatch reads the Hub when it dials the machine.
	for _, command := range [][]string{
		{"rental", "show", "katsuie", "--json"},
		{"rental", "update", "katsuie", "--runtime-version=0.23.25", "--json"},
		{"rental", "end", "katsuie", "--json"},
	} {
		if code, out := runCozy(t, root, command...); code == 0 || strings.Contains(out, `"state":"ended"`) {
			t.Fatalf("Hub failure became local truth [%d]: %v %s", code, command, out)
		}
	}
	postsLock.Lock()
	defer postsLock.Unlock()
	if len(*posts) != 0 {
		t.Fatal("selection bought replacement capacity")
	}
}

func TestRentalObservationReplacesEndpointAndOldBootUpdateHold(t *testing.T) {
	layout, problem := home.Open(t.TempDir())
	fatal(t, problem)
	store, problem := records.Open(layout.DB)
	fatal(t, problem)
	defer store.Close()
	row := records.Rental{ID: "pr-replaced", MachineName: "retained-alias", StableMachineName: "old-name", State: "ready", Hub: "https://hub.invalid", AcceleratorModel: "H100", AcceleratorCount: 1, HourlyRateUSDMicros: 100000, Address: "old.invalid:1", ExpectedWorkerID: "old-worker", ExpectedWorkerBootID: "old-boot"}
	fatal(t, store.RecordRental(row))
	update, problem := store.BeginRuntimeUpdate(row.ID, row.ExpectedWorkerBootID, []byte(`{"runtime_version":"0.23.25"}`))
	fatal(t, problem)
	update.State = "updating"
	fatal(t, store.SaveRuntimeUpdate(*update))
	if problem := store.RuntimeUpdateHold(row.ID); problem == nil {
		t.Fatal("old update did not hold its own boot")
	}
	row.Address, row.ExpectedWorkerID, row.ExpectedWorkerBootID, row.StableMachineName = "new.invalid:2", "new-worker", "new-boot", "new-name"
	fatal(t, store.RecordRental(row)) // Acquisition replay must accept a Hub replacement.
	if problem := store.RuntimeUpdateHold(row.ID); problem != nil {
		t.Fatalf("old incarnation still blocked replacement: %v", problem)
	}
	next, problem := store.BeginRuntimeUpdate(row.ID, row.ExpectedWorkerBootID, update.Selection)
	fatal(t, problem)
	if next.ID == update.ID {
		t.Fatal("different boot reused the old native update identity")
	}
	update.State = "failed"
	if problem := store.SaveRuntimeUpdate(*update); problem == nil {
		t.Fatal("stale updater overwrote replacement operation")
	}
	row.State, row.Failure.Code = "degraded", "readiness.provider_stopped"
	fatal(t, store.ObserveRental(row))
	stopped, problem := store.RuntimeUpdate(row.ID)
	fatal(t, problem)
	if stopped.Active() || !strings.Contains(stopped.Error, "provider reports this rental stopped") {
		t.Fatalf("provider stop retained update hold: %+v", stopped)
	}
	row.State, row.Failure.Code = "ready", ""
	fatal(t, store.ObserveRental(row))
	stopped, problem = store.RuntimeUpdate(row.ID)
	fatal(t, problem)
	if stopped.Active() {
		t.Fatal("observed resume restarted a canceled software change")
	}
	next, problem = store.BeginRuntimeUpdate(row.ID, row.ExpectedWorkerBootID, update.Selection)
	fatal(t, problem) // A new explicit update remains possible after resume.
	row.State = "released"
	fatal(t, store.ObserveRental(row))
	settled, problem := store.RuntimeUpdate(row.ID)
	fatal(t, problem)
	if settled.Active() || !strings.Contains(settled.Error, "Hub confirmed") {
		t.Fatalf("gone rental kept update active: %+v", settled)
	}
	next.State = "updating"
	if problem := store.SaveRuntimeUpdate(*next); problem == nil {
		t.Fatal("late observer resurrected ended rental update")
	}
}

func TestRentalSecretSweepRequiresAffirmativeHubAbsence(t *testing.T) {
	layout, problem := home.Open(t.TempDir())
	fatal(t, problem)
	store, problem := records.Open(layout.DB)
	fatal(t, problem)
	defer store.Close()
	const id = "pr-live-without-projection"
	adoptCreatorIdentity(t, layout, id)
	pending := filepath.Join(layout.Rentals, "pending-unknown.creator.pem")
	must(t, os.WriteFile(pending, []byte("retained pending custody"), 0600))
	for _, read := range []func(string) (bool, *exit.Error){
		func(string) (bool, *exit.Error) { return false, nil },
		func(string) (bool, *exit.Error) { return false, exit.New(exit.Unavailable, "Hub unavailable") },
		func(string) (bool, *exit.Error) { return false, exit.New(exit.NotFound, "unknown authority") },
	} {
		_, _ = reclaim.RentalSecrets(layout, store, read)
		if _, err := os.Stat(layout.RentalCreatorIdentity(id)); err != nil {
			t.Fatalf("lost live key custody: %s", err)
		}
	}
	_, problem = reclaim.RentalSecrets(layout, store, func(string) (bool, *exit.Error) { return true, nil })
	fatal(t, problem)
	if _, err := os.Stat(layout.RentalCreatorIdentity(id)); !os.IsNotExist(err) {
		t.Fatal("confirmed terminal key was retained")
	}
	if _, err := os.Stat(pending); err != nil {
		t.Fatal("unknown pending custody was deleted")
	}
}

func TestHistoricalAttachedAcquisitionsDoNotBecomeFleetRows(t *testing.T) {
	root, origin, stand := rentalEndRoot(t, "historical-attached")
	stand.publishListing()
	store, problem := records.Open(filepath.Join(root, "creator.sqlite"))
	fatal(t, problem)
	defer store.Close()
	for _, id := range []string{"sep-first", "sep-second", "sep-third", "oct-first", "oct-second", "oct-third"} {
		_, _, problem := store.BeginRentalOperation(records.RentalOperation{Key: id, Hub: origin, HourlyRateUSDMicros: 100000}, func(name string) ([]byte, string, *exit.Error) {
			return []byte(`{"name":"` + name + `","sku":"cpu"}`), id, nil
		})
		fatal(t, problem)
		fatal(t, store.AdvanceRentalOperation(id, "pr-"+id, "attached"))
	}
	code, out := runCozy(t, root, "rental", "list", "--json", "--full")
	var report struct {
		Rentals  []any `json:"rentals"`
		Machines int   `json:"machines_running"`
		Pending  int   `json:"unattached_rental_operations"`
	}
	if code != 0 || json.Unmarshal([]byte(out), &report) != nil || len(report.Rentals) != 0 || report.Machines != 0 || report.Pending != 0 {
		t.Fatalf("historical receipts masqueraded as live fleet: %d %s", code, out)
	}
	ops, problem := store.RentalOperations()
	fatal(t, problem)
	if len(ops) != 6 {
		t.Fatal("fleet listing removed durable acquisition history")
	}
}
