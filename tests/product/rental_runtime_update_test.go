package producttest

import (
	"context"
	"database/sql"
	"encoding/json"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
	"google.golang.org/grpc/metadata"
	"path/filepath"
	"strings"
	"testing"

	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/orchestrator"
	"github.com/cozy-creator/cozy/internal/records"
	"github.com/cozy-creator/cozy/internal/rental"
)

func TestRentalRuntimeUpdateJournalKeepsDispatchClosedAcrossRestart(t *testing.T) {
	f := developmentFixtureAt(t)
	f.attach(t, "127.0.0.1:1")
	update, problem := f.store.BeginRuntimeUpdate(f.rentalID, f.peer.bootID, "")
	fatal(t, problem)
	update.Selection = []byte(`{"wheels":[]}`)
	update.State = "updating"
	fatal(t, f.store.SaveRuntimeUpdate(*update))
	f.store.Close()
	reopened, problem := records.Open(f.layout.DB)
	fatal(t, problem)
	defer reopened.Close()
	c, problem := orchestrator.Open(orchestrator.Options{Store: reopened, Layout: f.layout, Rentals: rental.Resolver(f.layout, reopened)})
	fatal(t, problem)
	defer c.Close(orchestrator.StopGrace)
	if _, problem := c.UseRental(f.rentalID); problem == nil || problem.ErrName() != "rental.maintenance" {
		t.Fatalf("unfinished update did not retain dispatch hold: %v", problem)
	}
	other, problem := c.UseRental("another-rental")
	fatal(t, problem)
	other()
	row, problem := reopened.RuntimeUpdate(f.rentalID)
	fatal(t, problem)
	if row.ID != update.ID || row.State != "updating" {
		t.Fatalf("update was not durable: %+v", row)
	}
	row.State = "failed"
	row.Error = "candidate rolled back"
	fatal(t, reopened.SaveRuntimeUpdate(*row))
	release, problem := c.UseRental(f.rentalID)
	fatal(t, problem)
	release()
	release()
	row.ID = "stale-update"
	if problem := reopened.SaveRuntimeUpdate(*row); problem == nil {
		t.Fatal("stale updater overwrote the current operation")
	}
}

func TestRentalMaintenanceRefusesActiveTransportAndIsolatesOtherRentals(t *testing.T) {
	f := developmentFixtureAt(t)
	f.attach(t, "127.0.0.1:1")
	c, problem := orchestrator.Open(orchestrator.Options{Store: f.store, Layout: f.layout, Rentals: rental.Resolver(f.layout, f.store)})
	fatal(t, problem)
	defer c.Close(orchestrator.StopGrace)
	release, problem := c.UseRental(f.rentalID)
	fatal(t, problem)
	called := false
	updater := func(context.Context, *orchestrator.WorkerConnection) *exit.Error { called = true; return nil }
	if problem := c.MaintainRental(context.Background(), f.rentalID, updater); problem == nil || problem.ErrName() != "rental.maintenance_busy" || called {
		t.Fatalf("active transport was interrupted: %v", problem)
	}
	release()
	fatal(t, c.MaintainRental(context.Background(), f.rentalID, func(context.Context, *orchestrator.WorkerConnection) *exit.Error {
		if _, problem := c.UseRental(f.rentalID); problem == nil {
			t.Fatal("the target accepted transport during maintenance")
		}
		other, problem := c.UseRental("another-rental")
		fatal(t, problem)
		other()
		return nil
	}))
	use, problem := c.UseRental(f.rentalID)
	fatal(t, problem)
	use()
}

func TestRuntimeUpdateMigrationFrom41PreservesRental(t *testing.T) {
	f := developmentFixtureAt(t)
	f.attach(t, "127.0.0.1:1")
	f.store.Close()
	db, err := sql.Open("sqlite", filepath.Join(f.layout.Root, "creator.sqlite"))
	must(t, err)
	_, err = db.Exec("DROP TABLE rental_idle; DROP TABLE rental_runtime_updates")
	must(t, err)
	_, err = db.Exec("PRAGMA user_version=41")
	must(t, err)
	must(t, db.Close())
	if old, problem := records.Open(f.layout.DB); problem == nil {
		old.Close()
		t.Fatal("an ordinary reader migrated the daemon-owned database")
	}
	store, problem := records.OpenForDaemon(f.layout.DB, "")
	fatal(t, problem)
	defer store.Close()
	row, problem := store.RentalRow(f.rentalID)
	fatal(t, problem)
	if row == nil || row.ExpectedWorkerBootID != f.peer.bootID {
		t.Fatalf("migration lost the pinned rental: %+v", row)
	}
	_, problem = store.BeginRuntimeUpdate(f.rentalID, f.peer.bootID, "")
	fatal(t, problem)
}

func TestRentalDependencyVerdictIsSharedByRootAndServingPreparation(t *testing.T) {
	fields := map[string]string{"package": "paul/minimax-h3", "distribution": "cozy-runtime", "required": "cozy-runtime>=0.18.4,<1", "installed": "0.18.2"} //cozy:allow dependency metadata fixture, not a binary invocation
	raw, err := json.Marshal(fields)
	must(t, err)
	event := &pb.PrepareEvent{SafeCode: "package_runtime_incompatible", SafeDetail: string(raw)}
	fromEvent := orchestrator.RuntimeRequirementEvent(event)
	fromTrailer := orchestrator.RuntimeRequirementTrailer(metadata.Pairs(
		"cozy-requirement-package", fields["package"], "cozy-requirement-distribution", fields["distribution"],
		"cozy-requirement-required", fields["required"], "cozy-requirement-installed", fields["installed"]))
	for _, problem := range []*exit.Error{fromEvent, fromTrailer} {
		if problem == nil || problem.ErrName() != "machine_execution.runtime_requirement" ||
			!strings.Contains(problem.Message, "cozy-runtime 0.18.2") || !strings.Contains(problem.Message, "requires cozy-runtime>=0.18.4,<1") {
			t.Fatalf("lost actionable dependency facts: %v", problem)
		}
	}
	event.SafeDetail = `{"package":"x"}`
	if orchestrator.RuntimeRequirementEvent(event) != nil {
		t.Fatal("incomplete error became update authority")
	}
	fields["distribution"] = "torch"
	raw, err = json.Marshal(fields)
	must(t, err)
	event.SafeCode, event.SafeDetail = "package_sdk_incompatible", string(raw)
	if problem := orchestrator.RuntimeRequirementEvent(event); problem == nil || problem.ErrName() != "machine_execution.package_requirement" {
		t.Fatalf("non-updatable distribution became Runtime update authority: %v", problem)
	}
}
