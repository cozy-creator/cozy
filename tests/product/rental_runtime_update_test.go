package producttest

import (
	"context"
	"database/sql"
	"path/filepath"
	"testing"

	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/launch"
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

func TestRuntimeRequirementUsesActualPairAndPreservesAuthoredRange(t *testing.T) {
	for _, requirement := range []string{"cozy-runtime>=0.18.3,<1", "cozy-runtime[media]>=0.18.3,<1"} {
		mismatch := launch.RuntimeRequirementMismatch([]string{requirement, "torch>=2.13"}, "0.18.2", "0.3.43")
		if mismatch == nil || mismatch.Distribution != "cozy-runtime" || mismatch.Installed != "0.18.2" || mismatch.Required != requirement {
			t.Fatalf("lost compatibility facts: %+v", mismatch)
		}
		if launch.RuntimeRequirementMismatch([]string{requirement}, "0.18.3", "0.3.43") != nil {
			t.Fatal("compatible actual Runtime was refused")
		}
	}
	if mismatch := launch.RuntimeRequirementMismatch([]string{"tensorfs>=0.3.44"}, "0.18.3", "0.3.43"); mismatch == nil || mismatch.Distribution != "tensorfs" {
		t.Fatalf("TensorFS floor ignored: %+v", mismatch)
	}
}

func TestRuntimeUpdateMigrationFrom41PreservesRental(t *testing.T) {
	f := developmentFixtureAt(t)
	f.attach(t, "127.0.0.1:1")
	f.store.Close()
	db, err := sql.Open("sqlite", filepath.Join(f.layout.Root, "creator.sqlite"))
	must(t, err)
	_, err = db.Exec("DROP TABLE rental_runtime_updates")
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
