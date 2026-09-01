package producttest

import (
	"crypto/sha256"
	"database/sql"
	"encoding/hex"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/cozy-creator/cozy/internal/records"
)

func TestRecordsSchemasSixThroughEightMigrateWithoutDroppingDurableRows(t *testing.T) {
	for _, version := range []int{6, 7, 8} {
		t.Run(fmt.Sprintf("v%d", version), func(t *testing.T) {
			root := t.TempDir()
			database := filepath.Join(root, "records.db")
			generation := recoverablePackageGeneration(t, root, "0123456789abcdef", "proof/migrate")
			store, problem := records.Open(database)
			fatal(t, problem)
			_, problem = store.Activate(generation)
			fatal(t, problem)
			rental := records.Rental{ID: "rental-migrate", MachineName: "migrate-pod",
				SKU: "cpu-proof", AcceleratorModel: "CPU", HourlyRateUSDMicros: 125_000,
				Address: "127.0.0.1:9443", CertPath: "/proof/rental.crt", State: "ready",
				Hub: "https://tensorhub.example", RentedAt: "2026-08-31T00:00:00Z",
				MediaAddress: "127.0.0.1:9444", ExpectedWorkerID: "worker-proof",
				ExpectedWorkerBootID: "boot-proof"}
			fatal(t, store.RecordRental(rental))
			_, _, problem = store.BeginModelProduction(records.ModelProductionOperation{
				ID: "modelupload-pre-account", PlanDigest: "sha256:" + strings.Repeat("c", 64),
				Plan: []byte(`{}`),
			})
			fatal(t, problem)
			store.Close()
			db, err := sql.Open("sqlite", database)
			must(t, err)
			_, err = db.Exec(`INSERT INTO requests
				(id,idem_key,body_digest,package,entrypoint,plan_id,payload,state,created_at,install_id)
				VALUES('req-migrate','idem-migrate','body-migrate','proof/migrate','render','plan-migrate',
				x'7b7d','queued','2026-08-31T00:00:00Z',?)`, generation.ID)
			must(t, err)
			db.Close()

			stampRecordsVersion(t, database, version)

			store, problem = records.Open(database)
			fatal(t, problem)
			defer store.Close()
			pin, installed, problem := store.ActivePackage("proof/migrate")
			fatal(t, problem)
			if pin == nil || installed == nil || installed.ID != generation.ID ||
				installed.PackageDescriptor != generation.PackageDescriptor {
				t.Fatalf("schema migration dropped or changed package inventory: %#v %#v", pin, installed)
			}
			request, problem := store.RequestRow("req-migrate")
			fatal(t, problem)
			if request == nil || request.InstallID != generation.ID || request.RentalRequired {
				t.Fatalf("schema migration dropped or changed request row: %#v", request)
			}
			migratedRental, problem := store.RentalRow(rental.ID)
			fatal(t, problem)
			if migratedRental == nil || *migratedRental != rental {
				t.Fatalf("schema migration dropped or changed rental row: %#v", migratedRental)
			}
			productions, problem := store.ModelProductions("any", 50)
			fatal(t, problem)
			if len(productions) != 0 {
				t.Fatalf("schema migration retained pre-account model productions: %+v", productions)
			}
		})
	}
}

func TestExplicitPackageInventoryRecoveryUsesExactRowsAndFiles(t *testing.T) {
	root := t.TempDir()
	generation := recoverablePackageGeneration(t, root, "fedcba9876543210", "proof/marco")
	sourcePath := filepath.Join(root, "records.schema8.backup")
	source, problem := records.Open(sourcePath)
	fatal(t, problem)
	_, problem = source.Activate(generation)
	fatal(t, problem)
	source.Close()
	stampRecordsVersion(t, sourcePath, 8)

	current := recoverablePackageGeneration(t, root, "0011223344556677", "proof/current")
	destination, problem := records.Open(filepath.Join(root, "records.db"))
	fatal(t, problem)
	_, problem = destination.Activate(current)
	fatal(t, problem)
	destination.Close()

	code, out := runCozy(t, root, "package", "recover", sourcePath, "--json")
	if code != 0 || !strings.Contains(out, `"status":"recovered"`) ||
		!strings.Contains(out, `"generations":1`) || !strings.Contains(out, `"models_changed":false`) {
		t.Fatalf("explicit package inventory recovery [exit %d]\n%s", code, out)
	}
	destination, problem = records.Open(filepath.Join(root, "records.db"))
	fatal(t, problem)
	_, installed, problem := destination.ActivePackage("proof/marco")
	fatal(t, problem)
	if installed == nil || installed.ID != generation.ID ||
		installed.PlacementSetDigest != generation.PlacementSetDigest {
		t.Fatalf("recovered package identity changed: %#v", installed)
	}
	_, kept, problem := destination.ActivePackage(current.Package)
	fatal(t, problem)
	if kept == nil || kept.ID != current.ID {
		t.Fatalf("recovery changed the existing package inventory: %#v", kept)
	}
	destination.Close()
	code, out = runCozy(t, root, "package", "recover", sourcePath, "--json")
	if code != 0 || !strings.Contains(out, `"generations":0`) {
		t.Fatalf("identical recovery was not idempotent [exit %d]\n%s", code, out)
	}
}

func TestPackageInventoryRecoveryRefusesPinCollisionAtomically(t *testing.T) {
	root := t.TempDir()
	current := recoverablePackageGeneration(t, root, "1111222233334444", "proof/collision")
	destination, problem := records.Open(filepath.Join(root, "records.db"))
	fatal(t, problem)
	_, problem = destination.Activate(current)
	fatal(t, problem)
	destination.Close()

	backup := recoverablePackageGeneration(t, root, "aaaabbbbccccdddd", "proof/collision")
	sourcePath := filepath.Join(root, "records.schema8.backup")
	source, problem := records.Open(sourcePath)
	fatal(t, problem)
	_, problem = source.Activate(backup)
	fatal(t, problem)
	source.Close()
	stampRecordsVersion(t, sourcePath, 8)

	code, out := runCozy(t, root, "package", "recover", sourcePath, "--json")
	if code != 1 || !strings.Contains(out, `"code":"package_pin_recovery_collision"`) {
		t.Fatalf("pin collision did not refuse recovery [exit %d]\n%s", code, out)
	}
	destination, problem = records.Open(filepath.Join(root, "records.db"))
	fatal(t, problem)
	defer destination.Close()
	inserted, problem := destination.Install(backup.ID)
	fatal(t, problem)
	if inserted != nil {
		t.Fatalf("refused recovery partially inserted generation: %#v", inserted)
	}
}

func stampRecordsVersion(t *testing.T, path string, version int) {
	t.Helper()
	db, err := sql.Open("sqlite", path)
	must(t, err)
	if version < 8 {
		_, err = db.Exec(`DROP INDEX rentals_machine_name; ALTER TABLE rentals RENAME TO rentals_current`)
		must(t, err)
		_, err = db.Exec(testPriorRentalsDDL)
		must(t, err)
		_, err = db.Exec(`INSERT INTO rentals
		(id,machine_name,sku,accelerator_model,hourly_rate_usd_micros,managed_request_id,address,
		 cert_path,state,hub,rented_at,media_address,expected_worker_id,expected_worker_boot_id)
		 SELECT id,machine_name,sku,accelerator_model,hourly_rate_usd_micros,managed_request_id,address,
		 cert_path,state,hub,rented_at,media_address,expected_worker_id,expected_worker_boot_id
		 FROM rentals_current; DROP TABLE rentals_current`)
		must(t, err)
		_, err = db.Exec(testPriorRentalIndexDDL)
		must(t, err)
	}
	if version == 6 {
		_, err = db.Exec(`ALTER TABLE requests DROP COLUMN rental_required`)
		must(t, err)
	}
	_, err = db.Exec(`PRAGMA user_version=` + fmt.Sprint(version))
	must(t, err)
	db.Close()
}

const testPriorRentalsDDL = `CREATE TABLE IF NOT EXISTS rentals (
  id                TEXT PRIMARY KEY,
  machine_name      TEXT NOT NULL DEFAULT '',
  sku               TEXT NOT NULL DEFAULT '',
  accelerator_model TEXT NOT NULL,
  hourly_rate_usd_micros INTEGER NOT NULL,
  managed_request_id TEXT NOT NULL DEFAULT '',
  address           TEXT NOT NULL,
  cert_path         TEXT NOT NULL,
  state             TEXT NOT NULL,
  hub               TEXT NOT NULL,
  rented_at         TEXT NOT NULL,
  media_address     TEXT NOT NULL DEFAULT '',
  expected_worker_id         TEXT NOT NULL DEFAULT '',
  expected_worker_boot_id    TEXT NOT NULL DEFAULT ''
, wheelhouse_manifest_digest TEXT NOT NULL DEFAULT '')`

const testPriorRentalIndexDDL = `
CREATE UNIQUE INDEX IF NOT EXISTS rentals_machine_name
  ON rentals(machine_name) WHERE machine_name<>''`

func recoverablePackageGeneration(t *testing.T, root, id, pkg string) records.PackageInstall {
	t.Helper()
	dir := filepath.Join(root, "generations", id)
	descriptor := []byte(`{"application":"proof:app","entrypoints":[],"format":"cozy.package.descriptor/1","jobs":[],"model_productions":[]}`)
	placement := []byte(`{"format":"cozy.worker.v1.PlacementSet/1","placements":[]}`)
	descriptorDigest := testSHA256(descriptor)
	placementDigest := testSHA256(placement)
	must(t, os.MkdirAll(filepath.Join(dir, "documents"), 0o700))
	must(t, os.MkdirAll(filepath.Join(dir, "artifact-cache"), 0o700))
	must(t, os.WriteFile(filepath.Join(dir, "documents", "descriptor.json"), descriptor, 0o600))
	must(t, os.WriteFile(filepath.Join(dir, "artifact-cache",
		strings.TrimPrefix(placementDigest, "sha256:")), placement, 0o600))
	return records.PackageInstall{ID: id, Package: pkg, Major: 1, Version: "1.0.0",
		SourceKind: "tensorhub", SourceRef: pkg + "@1.0.0",
		SourceDigest: "sha256:" + strings.Repeat("a", 64), Verified: true,
		Dir: dir, Python: "/usr/bin/python3", Runtime: "/usr/bin/python3",
		ProjectDir: filepath.Join(dir, "source"), UV: "proof", LockDigest: "sha256:" + strings.Repeat("b", 64),
		Platform: "linux", Packages: 1, Closure: "proof==1.0.0",
		PackageDescriptor: descriptorDigest, PlacementSetDigest: placementDigest}
}

func testSHA256(raw []byte) string {
	digest := sha256.Sum256(raw)
	return "sha256:" + hex.EncodeToString(digest[:])
}
