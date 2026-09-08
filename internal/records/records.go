// Package records is the ONE local lifecycle authority: package installs and the
// one active pin per package, as rows in ONE local SQLite database
// (cozy-creator.md "Records"). There is no state.json and no second lifecycle store;
// any JSON output is a derived read.
//
// Seam for cl-001: the Cozy daemon adopts THIS package as its lifecycle store and
// adds its own tables (worker sessions, requests, attempts, outputs) to the same
// database. Nothing here assumes a CLI caller; Open takes a path.
//
// Driver: modernc.org/sqlite — SQLite itself, transpiled to Go. The store uses no
// engine-specific feature (this is one open of one local file), so the driver is a
// DISTRIBUTION decision, and pure Go is the whole answer: `cozy` builds and
// cross-compiles with CGO_ENABLED=0 for every platform Go targets. The file format is
// unchanged — plain `SQLite format 3`, which is what the previous embedded-libSQL
// driver wrote too, so a records.db an older binary created opens here as it is.
package records

import (
	"crypto/sha256"
	"database/sql"
	"encoding/hex"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"github.com/cozy-creator/cozy/internal/exit"
	_ "modernc.org/sqlite"
)

// PackageInstall is one immutable install: a materialized environment plus the exact files
// that produced it. Rows are never updated — a rebuild is a NEW install. It is deliberately
// not a fencing counter: the wire spends `epoch` on the executor and admission fences, the
// local install is neither, and one word for three things is how they drift (#484).
type PackageInstall struct {
	ID                 string
	Package            string // org/name
	Major              int
	Version            string
	SourceKind         string // "tensorhub" | "local"
	SourceRef          string
	SourceDigest       string
	Verified           bool // false = an explicit local/development install, not published custody
	Dir                string
	Python             string
	Runtime            string // exact cozy-runtime binary; empty on older source installs derives from venv
	ProjectDir         string // exact installed project root; empty on source installs derives from source kind
	UV                 string
	LockDigest         string
	Platform           string
	Extra              string // the CUDA-extra pick ("" = none declared or no accelerator)
	Packages           int
	Closure            string // one "name==version" per line
	PackageInterface   string // exact digest of the install-private Runtime-derived PackageInterface
	PlacementSetDigest string // exact Hub-selected PlacementSet/1 stored in the artifact cache
	BytesExcl          int64
	BytesShared        int64
	CreatedAt          string
}

// Pin is the package's one active install. Major remains recorded for package metadata
// and removal output; activating any release replaces the package's previous pin.
type Pin struct {
	Package     string
	Major       int
	InstallID   string // names a PackageInstall row's id
	ActivatedAt string
}

type Store struct{ db *sql.DB }

const schemaVersion = 34

const installsDDL = `
CREATE TABLE IF NOT EXISTS installs (
  id            TEXT PRIMARY KEY,
  package      TEXT    NOT NULL,
  major         INTEGER NOT NULL,
  version       TEXT    NOT NULL,
  source_kind   TEXT    NOT NULL,
  source_ref    TEXT    NOT NULL,
  source_digest TEXT    NOT NULL,
  verified      INTEGER NOT NULL,
  dir           TEXT    NOT NULL,
  python        TEXT    NOT NULL,
  runtime       TEXT    NOT NULL DEFAULT '',
  project_dir   TEXT    NOT NULL DEFAULT '',
  uv            TEXT    NOT NULL,
  lock_digest   TEXT    NOT NULL,
  platform      TEXT    NOT NULL,
  extra         TEXT    NOT NULL,
  packages      INTEGER NOT NULL,
  closure       TEXT    NOT NULL,
  package_interface TEXT    NOT NULL,
  placement_set_digest TEXT  NOT NULL DEFAULT '',
  bytes_excl    INTEGER NOT NULL,
  bytes_shared  INTEGER NOT NULL,
  created_at    TEXT    NOT NULL
)`

const pinsDDL = `
CREATE TABLE IF NOT EXISTS pins (
  package     TEXT    NOT NULL,
  major        INTEGER NOT NULL,
  install_id   TEXT    NOT NULL REFERENCES installs(id),
  activated_at TEXT    NOT NULL,
  PRIMARY KEY (package)
)`

// schema is the only records shape this pre-launch build accepts.
var schema = append([]string{installsDDL, pinsDDL, childBindingsDDL}, append(orchestratorSchema,
	append(modelTransferSchema, append(eventSchema, append(rentalSchema, packageEventSchema...)...)...)...)...)

func init() {
	schema = append(schema, weightsRetentionsDDL, operationLookupsDDL, nativeCallsDDL, nativeArtifactRetentionsDDL, byteOutputsDDL, childArgumentsDDL, activeChildRequestIndex, activeNativeCallIndex)
}

// pragmas ride the DSN rather than being executed after the open, because a pragma is a
// property of a CONNECTION and database/sql may discard and redial one at any moment: a
// re-dialled connection with foreign_keys OFF would silently accept the delete Forget
// exists to refuse. The driver replays them on every connection it opens.
//
//	busy_timeout  transient lock contention waits instead of failing immediately
//	foreign_keys  the pin -> install reference is enforced, not decorative
//	txlock        writers reserve the lock before reading, so two processes cannot both
//	              read and then fail immediately while upgrading a deferred transaction
const pragmas = "?_pragma=busy_timeout(5000)&_pragma=foreign_keys(1)&_txlock=immediate"

func Open(path string) (*Store, *exit.Error) {
	return open(path, false, "")
}

// OpenForDaemon is the only schema-migrating entrance. The daemon holds
// home.Layout.Daemon before calling it and keeps that lock for its lifetime, so
// a newly installed CLI cannot rewrite the database under an older live daemon.
// `triageDir` is the retired pre-cl-116 bundle directory: schema 22 folds each
// referenced file into its attempt row, and the caller deletes the directory once
// this open has returned.
func OpenForDaemon(path, triageDir string) (*Store, *exit.Error) {
	return open(path, true, triageDir)
}

func open(path string, migratePrior bool, triageDir string) (*Store, *exit.Error) {
	// No `file:` prefix: the driver hands an unprefixed name to SQLite verbatim, so a
	// local root containing `%` or `#` stays a path instead of becoming a URI to decode.
	db, err := sql.Open("sqlite", path+pragmas)
	if err != nil {
		return nil, exit.Internalf("cannot open the local records database %s: %s", path, err)
	}
	// ONE writer: the lifecycle authority is a single-writer store, so serializing every
	// statement on one connection is the honest shape rather than a tuning choice.
	db.SetMaxOpenConns(1)
	if err := db.Ping(); err != nil {
		db.Close()
		return nil, exit.Internalf("cannot open the local records database %s: %s", path, err)
	}
	version, err := databaseVersion(db)
	if err != nil {
		db.Close()
		return nil, exit.Internalf("cannot read the records schema version in %s: %s", path, err)
	}
	if version == 0 {
		if e := initialize(db, path); e != nil {
			db.Close()
			return nil, e
		}
	} else if version >= 6 && version < schemaVersion {
		if !migratePrior {
			db.Close()
			return nil, exit.Named(exit.Conflict, "records_schema_upgrade_required",
				"records database has schema %d; this Creator requires schema %d",
				version, schemaVersion).
				WithRemedy("stop the active Cozy daemon, then run `cozy up` so the new daemon can migrate it")
		}
		if e := migrate(db, path, version, triageDir); e != nil {
			db.Close()
			return nil, e
		}
	} else if version != schemaVersion {
		db.Close()
		return nil, schemaReset(path,
			"records database has user_version %d, not exact version %d", version, schemaVersion)
	}
	if e := verifySchema(db, path); e != nil {
		db.Close()
		return nil, e
	}
	// The database retains request payloads, rental facts and triage bundles; 0600 is
	// the same boundary the daemon record carries, and the WAL/SHM siblings SQLite
	// creates next inherit these bits (cl-116). Windows mode bits are a fiction; the
	// user-profile ACL scopes the root there.
	if err := os.Chmod(path, 0o600); err != nil {
		db.Close()
		return nil, exit.Internalf("cannot protect the records database %s: %s", path, err)
	}
	if _, err := db.Exec(`PRAGMA journal_mode=WAL`); err != nil {
		db.Close()
		return nil, exit.Internalf("cannot enable WAL for %s: %s", path, err)
	}
	return &Store{db: db}, nil
}

// Schemas 6 through 22 migrate in place. Schema 11 replaces authored GPU counts with
// the one derived accelerator-class fact; schema 12 gives the install table and its two
// foreign keys the one word the row actually names; schema 13 records when a rental was
// first seen ready; schema 14 drops the output export's pre-execution payload hash — a
// file is named by its own content digest now; schema 15 spells the request's editable
// revision columns `local_package_*` (proto-027: one word for a local package); schema 16
// adds the package_events table (cl-097); schema 17 records the rental's machine word on
// the request (cl-107) — existing rows joinable to a surviving rental get their word,
// unjoinable history stays blank; schema 18 removes checkpoint evidence from
// model-transfer outputs; schema 19 removes the duplicate request config digest now owned
// by the CozyTensors header; schema 20 renames the installed package interface and removes
// the obsolete aggregate package revision digest. Package, request, event, export, and rental rows survive;
// schema 21 retains Tensorhub's sanitized terminal rental boot failure; schema 22 moves
// each attempt's verified triage bundle bytes INTO the attempt row (cl-116), retiring the
// triage file directory and its orphan class; schema 33 records the rental's WIDTH
// (cl-179), the count of accelerators the paid pod delivers.
// only schema 9's superseded special
// model-production subsystem is dropped.
// Schema 10 creates empty request-attached transfer sidecars because older rows cannot be
// translated into ordinary request identity safely.
func migrate(db *sql.DB, path string, sourceVersion int, triageDir string) *exit.Error {
	if e := verifyPriorSchema(db, path, sourceVersion); e != nil {
		return e
	}
	// Rebuild the one changed table so sqlite_master matches the closed current schema.
	// Foreign-key rewriting on ALTER TABLE must be disabled during the swap; integrity is
	// checked again before Open returns the store.
	if _, err := db.Exec(`PRAGMA foreign_keys=OFF`); err != nil {
		return exit.Internalf("cannot suspend foreign-key checks while migrating %s: %s", path, err)
	}
	defer db.Exec(`PRAGMA foreign_keys=ON`)
	if _, err := db.Exec(`PRAGMA legacy_alter_table=ON`); err != nil {
		return exit.Internalf("cannot fence table rename while migrating %s: %s", path, err)
	}
	defer db.Exec(`PRAGMA legacy_alter_table=OFF`)
	tx, err := db.Begin()
	if err != nil {
		return exit.Internalf("cannot begin records schema migration in %s: %s", path, err)
	}
	defer tx.Rollback()
	version, err := databaseVersion(tx)
	if err != nil {
		return exit.Internalf("cannot confirm records schema in %s: %s", path, err)
	}
	if version == schemaVersion {
		return commitMigration(tx, path)
	}
	if version != sourceVersion {
		return schemaReset(path, "records database changed to user_version %d while migrating", version)
	}
	// Each rebuild runs only from a schema that still has the prior shape: the install
	// rename is schema 12's, the rental rebuild ends at 13, the export table's payload
	// hash goes at 14, package events arrive at 16, and the request rebuild ends at 17
	// (the recorded machine word), which is also where its rows get their backfill.
	if sourceVersion < 20 {
		if e := migrateInstalls(tx, path, sourceVersion); e != nil {
			return e
		}
	}
	if sourceVersion < 33 {
		if e := migrateRentals(tx, path, sourceVersion); e != nil {
			return e
		}
	}
	if sourceVersion < 14 {
		if e := migrateOutputExports(tx, path); e != nil {
			return e
		}
	}
	if sourceVersion < 34 {
		if e := migrateRequests(tx, path, sourceVersion); e != nil {
			return e
		}
	}
	if sourceVersion < 34 {
		if _, err := tx.Exec(childRequestIndex); err != nil {
			return exit.Internalf("cannot restore child call admission index in %s: %s", path, err)
		}
		if _, err := tx.Exec(weightsRetentionsDDL); err != nil {
			return exit.Internalf("cannot create artifact retention ownership in %s: %s", path, err)
		}
	}
	if sourceVersion < 16 {
		for _, statement := range packageEventSchema {
			if _, err := tx.Exec(statement); err != nil {
				return exit.Internalf("cannot create package events while migrating %s: %s", path, err)
			}
		}
	}
	if sourceVersion >= 10 && sourceVersion < 18 {
		if e := migrateModelTransferOutputs(tx, path); e != nil {
			return e
		}
	}
	if sourceVersion < 10 {
		for _, table := range []string{"model_production_objects", "model_production_artifacts",
			"model_production_steps", "model_production_sources", "model_production_source_files",
			"model_productions"} {
			if _, err := tx.Exec(`DROP TABLE ` + table); err != nil {
				return exit.Internalf("cannot retire schema-9 %s while migrating %s: %s", table, path, err)
			}
		}
		for _, statement := range modelTransferSchema {
			if _, err := tx.Exec(statement); err != nil {
				return exit.Internalf("cannot create model transfer sidecars while migrating %s: %s", path, err)
			}
		}
	}
	if sourceVersion < 22 {
		if e := migrateAttemptTriage(tx, path, triageDir); e != nil {
			return e
		}
	}
	if sourceVersion < 23 {
		if sourceVersion >= 10 {
			for _, statement := range []string{
				`ALTER TABLE request_model_transfers RENAME TO request_model_transfers_prior`,
				modelTransferSchema[0],
				`INSERT INTO request_model_transfers(request_id,intent,state,models,checkpoints,error_code,safe_error,updated_at)
				 SELECT request_id,intent,state,models,checkpoints,error_code,safe_error,updated_at FROM request_model_transfers_prior`,
				`DROP TABLE request_model_transfers_prior`,
				`ALTER TABLE request_model_transfer_files RENAME TO request_model_transfer_files_prior`,
				modelTransferSchema[1],
				`INSERT INTO request_model_transfer_files(request_id,member,object_id,length,capability_revision,state,transferred,safe_code,safe_detail)
				 SELECT request_id,member,object_id,length,capability_revision,state,transferred,safe_code,safe_detail FROM request_model_transfer_files_prior`,
				`DROP TABLE request_model_transfer_files_prior`,
			} {
				if _, err := tx.Exec(statement); err != nil {
					return exit.Internalf("cannot bind source status to worker boot while migrating %s: %s", path, err)
				}
			}
		}
		for _, statement := range []string{modelCheckpointSchema, modelCheckpointPublicationSchema} {
			if _, err := tx.Exec(statement); err != nil {
				return exit.Internalf("cannot create source checkpoint progress while migrating %s: %s", path, err)
			}
		}
	}
	if sourceVersion == 23 {
		for _, statement := range []string{
			`ALTER TABLE request_model_transfers RENAME TO request_model_transfers_prior`,
			modelTransferSchema[0],
			`INSERT INTO request_model_transfers(request_id,intent,state,models,checkpoints,error_code,safe_error,updated_at,models_worker_boot_id)
			 SELECT request_id,intent,state,models,checkpoints,error_code,safe_error,updated_at,models_worker_boot_id FROM request_model_transfers_prior`,
			`DROP TABLE request_model_transfers_prior`,
		} {
			if _, err := tx.Exec(statement); err != nil {
				return exit.Internalf("cannot add publication cancellation intent while migrating %s: %s", path, err)
			}
		}
	}
	if sourceVersion >= 23 && sourceVersion < 25 {
		if e := migrateCheckpoints(tx, path); e != nil {
			return e
		}
	}
	if sourceVersion < 27 {
		for _, statement := range []string{childBindingsDDL, childRequestIndex} {
			if _, err := tx.Exec(statement); err != nil {
				return exit.Internalf("cannot create private child ownership in %s: %s", path, err)
			}
		}
	}
	if sourceVersion < 29 {
		if _, err := tx.Exec(operationLookupsDDL); err != nil {
			return exit.Internalf("cannot create pending operation lookups in %s: %s", path, err)
		}
	}
	if sourceVersion < 30 {
		if _, err := tx.Exec(nativeCallsDDL); err != nil {
			return exit.Internalf("cannot create native call admission in %s: %s", path, err)
		}
	}
	if sourceVersion < 31 {
		if _, err := tx.Exec(nativeArtifactRetentionsDDL); err != nil {
			return exit.Internalf("cannot create native artifact custody in %s: %s", path, err)
		}
	}

	if sourceVersion >= 30 && sourceVersion < 32 {
		columns := strings.Replace(nativeCallColumns, ",cancel_requested", "", 1)
		for _, statement := range []string{
			`ALTER TABLE native_calls RENAME TO native_calls_prior`, nativeCallsDDL,
			`INSERT INTO native_calls(` + columns + `) SELECT ` + columns + ` FROM native_calls_prior`,
			`DROP TABLE native_calls_prior`,
		} {
			if _, err := tx.Exec(statement); err != nil {
				return exit.Internalf("cannot preserve effect cancellation in %s: %s", path, err)
			}
		}
	}

	if sourceVersion < 33 {

		if _, err := tx.Exec(byteOutputsDDL); err != nil {
			return exit.Internalf("cannot add byte output custody: %s", err)
		}

		if sourceVersion >= 31 {
			columns := strings.TrimPrefix(nativeArtifactCols, "artifact_kind,producer_attempt,producer_output_id,content_bytes,")
			for _, statement := range []string{`ALTER TABLE native_artifact_retentions RENAME TO native_artifact_retentions_prior`, nativeArtifactRetentionsDDL, `INSERT INTO native_artifact_retentions(` + columns + `) SELECT ` + columns + ` FROM native_artifact_retentions_prior`, `DROP TABLE native_artifact_retentions_prior`} {
				if _, err := tx.Exec(statement); err != nil {
					return exit.Internalf("cannot preserve native custody migration: %s", err)
				}
			}
		}
	}

	if sourceVersion < 34 {
		if sourceVersion >= 32 {
			for _, statement := range []string{`ALTER TABLE native_calls RENAME TO native_calls_prior34`, nativeCallsDDL,
				`INSERT INTO native_calls(` + nativeCallColumns + `) SELECT ` + nativeCallColumns + ` FROM native_calls_prior34`,
				`DROP TABLE native_calls_prior34`} {
				if _, err := tx.Exec(statement); err != nil {
					return exit.Internalf("cannot preserve native call indices: %s", err)
				}
			}
		}
		if _, err := tx.Exec(childArgumentsDDL); err != nil {
			return exit.Internalf("cannot add serving child arguments: %s", err)
		}
		for _, statement := range []string{activeChildRequestIndex, activeNativeCallIndex} {
			if _, err := tx.Exec(statement); err != nil {
				return exit.Internalf("cannot index active child calls: %s", err)
			}
		}
	}
	if _, err := tx.Exec(fmt.Sprintf("PRAGMA user_version=%d", schemaVersion)); err != nil {
		return exit.Internalf("cannot stamp records migration in %s: %s", path, err)
	}
	if e := commitMigration(tx, path); e != nil {
		return e
	}
	if _, err := db.Exec(`PRAGMA legacy_alter_table=OFF`); err != nil {
		return exit.Internalf("cannot restore rename policy after migrating %s: %s", path, err)
	}
	if _, err := db.Exec(`PRAGMA foreign_keys=ON`); err != nil {
		return exit.Internalf("cannot restore foreign keys after migrating %s: %s", path, err)
	}
	rows, err := db.Query(`PRAGMA foreign_key_check`)
	if err != nil {
		return exit.Internalf("cannot verify migrated foreign keys in %s: %s", path, err)
	}
	defer rows.Close()
	if rows.Next() {
		return exit.Named(exit.Conflict, "records.migration_foreign_key_failed",
			"records migration in %s left an invalid foreign-key reference", path)
	}
	return nil
}

// migrateInstalls carries schema 11's `install_generations` table and the two foreign keys
// that spelled it `generation` onto their one name. The three tables are REBUILT rather than
// ALTERed: SQLite's own RENAME rewrites every referencing DDL with the new name QUOTED, and
// verifySchema compares the stored text to the authored text character for character. Rows
// move by column name, so the copy states what it preserves.
func migrateInstalls(tx *sql.Tx, path string, sourceVersion int) *exit.Error {
	priorTable, priorInstallID := "installs", "install_id"
	if sourceVersion < 12 {
		priorTable, priorInstallID = "install_generations", "generation"
	}
	priorInstallCols := strings.Replace(installCols(""), "package_interface", "package_descriptor", 1)
	for _, statement := range []string{
		`DROP INDEX worker_session`,
		`ALTER TABLE ` + priorTable + ` RENAME TO installs_prior`,
		`ALTER TABLE pins RENAME TO pins_prior`,
		`ALTER TABLE worker_processes RENAME TO worker_processes_prior`,
		installsDDL,
		`INSERT INTO installs(` + installCols("") + `) SELECT ` + installCols("") +
			` FROM installs_prior`,
		pinsDDL,
		`INSERT INTO pins(package,major,install_id,activated_at)
			SELECT package,major,` + priorInstallID + `,activated_at FROM pins_prior`,
		workerProcessesDDL,
		`INSERT INTO worker_processes(` + workerProcessCols + `)
			SELECT instance_id,package,` + priorInstallID + `,worker_id,
			  devices,pid,birth,session_id,state,opened_at,closed_at FROM worker_processes_prior`,
		workerSessionIndex,
		`DROP TABLE worker_processes_prior`,
		`DROP TABLE pins_prior`,
		`DROP TABLE installs_prior`,
	} {
		if strings.HasPrefix(statement, `INSERT INTO installs(`) {
			statement = `INSERT INTO installs(` + installCols("") + `) SELECT ` + priorInstallCols +
				` FROM installs_prior`
		}
		if _, err := tx.Exec(statement); err != nil {
			return exit.Internalf("cannot rename the install table while migrating %s: %s", path, err)
		}
	}
	return nil
}

// migrateRentals rebuilds the rentals table on every migration: schema 13 added ready_at,
// schema 21 the sanitized boot failure, schema 33 the rental's WIDTH, and every earlier
// shape carries the released column list without them. A rental that was already ready
// when the schema moved gets no ready_at; its idle clock starts at its next settlement,
// never at a guess. Every rental that predates schema 33 was bought one card wide — no
// wider product could be expressed, let alone attached — so the width backfills to 1,
// which is a fact about those rows and not a default standing in for an unknown.
func migrateRentals(tx *sql.Tx, path string, sourceVersion int) *exit.Error {
	if _, err := tx.Exec(`DROP INDEX rentals_machine_name`); err != nil {
		return exit.Internalf("cannot stage rental index while migrating %s: %s", path, err)
	}
	if _, err := tx.Exec(`ALTER TABLE rentals RENAME TO rentals_prior`); err != nil {
		return exit.Internalf("cannot stage rental rows while migrating %s: %s", path, err)
	}
	if _, err := tx.Exec(rentalsDDL); err != nil {
		return exit.Internalf("cannot create current rentals table while migrating %s: %s", path, err)
	}
	columns := rentalColsPriorThirtyThree
	switch {
	case sourceVersion < 13:
		columns = rentalColsPriorThirteen
	case sourceVersion < 21:
		columns = rentalColsPriorTwentyOne
	}
	if _, err := tx.Exec(`INSERT INTO rentals(` + columns + `) SELECT ` +
		columns + ` FROM rentals_prior`); err != nil {
		return exit.Internalf("cannot preserve rental rows while migrating %s: %s", path, err)
	}
	if _, err := tx.Exec(`UPDATE rentals SET accelerator_count=1 WHERE accelerator_count=0`); err != nil {
		return exit.Internalf("cannot record the width of retained rentals in %s: %s", path, err)
	}
	if _, err := tx.Exec(`DROP TABLE rentals_prior`); err != nil {
		return exit.Internalf("cannot finish rental migration in %s: %s", path, err)
	}
	if _, err := tx.Exec(rentalSchema[3]); err != nil {
		return exit.Internalf("cannot restore rental index while migrating %s: %s", path, err)
	}
	return nil
}

// migrateOutputExports rebuilds the export table without its payload hash. Rows carry
// over: a settled row keeps its published paths, an owed one is retried under the
// content-digest naming and lands on the same bytes.
func migrateOutputExports(tx *sql.Tx, path string) *exit.Error {
	if _, err := tx.Exec(`ALTER TABLE request_output_exports RENAME TO request_output_exports_prior`); err != nil {
		return exit.Internalf("cannot stage output export rows while migrating %s: %s", path, err)
	}
	if _, err := tx.Exec(outputExportSchema); err != nil {
		return exit.Internalf("cannot create current output export table while migrating %s: %s", path, err)
	}
	columns := `request_id,directory,outputs,state,attempts,error_code,safe_error,published_paths,updated_at`
	if _, err := tx.Exec(`INSERT INTO request_output_exports(` + columns + `) SELECT ` +
		columns + ` FROM request_output_exports_prior`); err != nil {
		return exit.Internalf("cannot preserve output export rows while migrating %s: %s", path, err)
	}
	if _, err := tx.Exec(`DROP TABLE request_output_exports_prior`); err != nil {
		return exit.Internalf("cannot finish output export migration in %s: %s", path, err)
	}
	return nil
}

// migrateAttemptTriage rebuilds the attempts table with the bundle IN the row
// (schema 22): `triage_path` becomes `triage_bundle`, and every referenced file that
// still verifies against its recorded digest and length is folded in. A file that is
// missing or no longer matches imports as an empty bundle — the row keeps the subject
// and digest as the record of what was lost. The caller deletes the retired directory
// after this open returns.
func migrateAttemptTriage(tx *sql.Tx, path, triageDir string) *exit.Error {
	const cols = `request_id,attempt,attempt_key,instance_id,session_id,invocation_digest,
	  invocation,weights_outputs,state,plan_digest,construction,plan_summary,terminal_id,
	  terminal_digest,terminal_status,terminal_cause,safe_message,triage_subject,
	  triage_digest,triage_length,terminal_body,dispatched_at,accepted_at,closed_at,media_cleaned`
	for _, statement := range []string{
		`ALTER TABLE attempts RENAME TO attempts_prior`,
		attemptsDDL,
		`INSERT INTO attempts(` + cols + `) SELECT ` + cols + ` FROM attempts_prior`,
	} {
		if _, err := tx.Exec(statement); err != nil {
			return exit.Internalf("cannot rebuild the attempts table while migrating %s: %s", path, err)
		}
	}
	type kept struct {
		key     string
		subject string
		digest  string
		length  int64
	}
	var rows []kept
	if triageDir != "" {
		read, err := tx.Query(`SELECT attempt_key, triage_subject, triage_digest, triage_length
			FROM attempts_prior WHERE triage_path != ''`)
		if err != nil {
			return exit.Internalf("cannot read kept triage references while migrating %s: %s", path, err)
		}
		for read.Next() {
			var r kept
			if err := read.Scan(&r.key, &r.subject, &r.digest, &r.length); err != nil {
				read.Close()
				return exit.Internalf("cannot read a kept triage reference while migrating %s: %s", path, err)
			}
			rows = append(rows, r)
		}
		read.Close()
	}
	for _, r := range rows {
		if r.subject == "" || filepath.Base(r.subject) != r.subject ||
			r.length <= 0 || r.length > maxTriageBundle {
			continue
		}
		data, err := os.ReadFile(filepath.Join(triageDir, r.subject+".json"))
		if err != nil || int64(len(data)) != r.length {
			continue
		}
		sum := sha256.Sum256(data)
		if "sha256:"+hex.EncodeToString(sum[:]) != r.digest {
			continue
		}
		if _, err := tx.Exec(`UPDATE attempts SET triage_bundle=? WHERE attempt_key=?`,
			data, r.key); err != nil {
			return exit.Internalf("cannot import the triage bundle of %s while migrating %s: %s", r.key, path, err)
		}
	}
	if _, err := tx.Exec(`DROP TABLE attempts_prior`); err != nil {
		return exit.Internalf("cannot finish the attempts migration in %s: %s", path, err)
	}
	return nil
}

// maxTriageBundle mirrors the media plane's bound: the protocol caps one bundle at 1 MiB,
// so anything larger on disk was never a bundle this product kept.
const maxTriageBundle = 1 << 20

// migrateModelTransferOutputs drops the checkpoint evidence payload. The Manifest/header is
// the checkpoint authority; a transfer row keeps only the exact receipt and object inventory.
func migrateModelTransferOutputs(tx *sql.Tx, path string) *exit.Error {
	for _, statement := range []string{
		`ALTER TABLE request_model_transfer_outputs RENAME TO request_model_transfer_outputs_prior`,
		modelTransferSchema[2],
		`INSERT INTO request_model_transfer_outputs(
		  request_id,output_slot,manifest_id,manifest_length,attempt,invocation_digest,
		  transaction_id,receipt_digest,receipt,final_id)
		 SELECT request_id,output_slot,manifest_id,manifest_length,attempt,invocation_digest,
		  transaction_id,receipt_digest,receipt,final_id
		 FROM request_model_transfer_outputs_prior`,
		`DROP TABLE request_model_transfer_outputs_prior`,
	} {
		if _, err := tx.Exec(statement); err != nil {
			return exit.Internalf("cannot remove model-transfer evidence while migrating %s: %s", path, err)
		}
	}
	return nil
}

// migrateRequests rebuilds the requests table on every migration: before schema 11 to derive
// the accelerator-class fact from the retired GPU count, at schema 12 because the row's own
// DDL text names the install table it references, at schema 15 because the editable
// revision columns are spelled `local_package_*`, and at schema 17 to record the rental's
// machine word — backfilled from a surviving rental row, blank for unjoinable history.
func migrateRequests(tx *sql.Tx, path string, sourceVersion int) *exit.Error {
	if _, err := tx.Exec(`ALTER TABLE requests RENAME TO requests_prior`); err != nil {
		return exit.Internalf("cannot stage request rows while migrating %s: %s", path, err)
	}
	if _, err := tx.Exec(requestsDDL); err != nil {
		return exit.Internalf("cannot create current requests table while migrating %s: %s", path, err)
	}
	destinationColumns := `id,idem_key,body_digest,package,entrypoint,plan_id,package_release,
		local_package_digest,local_package_uploaded_boot_id,
		environment_digest,payload,outputs,state,ordinal,requeues,created_at,kind,
		needs_accelerator,org,trees,worker,machine,rental,rental_required,install_id,assets,models,weights_outputs`
	rentalRequired := "rental_required"
	if sourceVersion == 6 {
		rentalRequired = "0"
	}
	needsAccelerator := "needs_accelerator"
	if sourceVersion < 11 {
		needsAccelerator = `CASE WHEN job_gpu_count>0 OR (kind!='job' AND models!='[]') THEN 1 ELSE 0 END`
	}
	localPackage := `private_package_digest,private_package_uploaded_boot_id`
	if sourceVersion >= 15 {
		localPackage = `local_package_digest,local_package_uploaded_boot_id`
	}
	machine := `COALESCE((SELECT machine_name FROM rentals WHERE rentals.id=requests_prior.worker),'')`
	if sourceVersion >= 17 {
		machine = "machine"
	}
	selectColumns := `id,idem_key,body_digest,package,entrypoint,plan_id,package_release,
		` + localPackage + `,
		environment_digest,payload,outputs,state,ordinal,requeues,created_at,kind,` +
		needsAccelerator + `,
		org,trees,worker,` + machine + `,rental,` + rentalRequired +
		`,install_id,assets,models,weights_outputs`
	if sourceVersion >= 26 {
		retained := ",retain_work,retry_of,reuse_scope,control_revision"
		destinationColumns += retained
		selectColumns += retained
	}
	if sourceVersion >= 27 {
		child := ",parent_request_id,parent_call_index,child_intent_digest,child_target_digest,child_reusable,reused_from,orchestration_directive"
		destinationColumns += child
		selectColumns += child
	}
	if sourceVersion >= 28 {
		destinationColumns += ",child_artifacts"
		selectColumns += ",child_artifacts"
	}
	if sourceVersion >= 33 {
		destinationColumns += ",capture"
		selectColumns += ",capture"
	}
	if _, err := tx.Exec(`INSERT INTO requests(` + destinationColumns + `) SELECT ` + selectColumns +
		` FROM requests_prior`); err != nil {
		return exit.Internalf("cannot preserve request rows while migrating %s: %s", path, err)
	}
	if _, err := tx.Exec(`DROP TABLE requests_prior`); err != nil {
		return exit.Internalf("cannot finish request migration in %s: %s", path, err)
	}
	return nil
}

const rentalsDDLPrior = `
CREATE TABLE IF NOT EXISTS rentals (
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

// priorStatements is the released DDL of one earlier schema, derived from the current one
// so a released shape is never a second copy that can drift from it.
func priorStatements(version int) []string {
	priorRequestsNineteen := strings.Replace(requestsDDL,
		"  package_release TEXT NOT NULL DEFAULT '',\n",
		"  package_release TEXT NOT NULL DEFAULT '',\n  package_revision_digest TEXT NOT NULL DEFAULT '',\n", 1)
	priorRequestsEighteen := strings.Replace(priorRequestsNineteen,
		"  environment_digest TEXT NOT NULL DEFAULT '',\n",
		"  environment_digest TEXT NOT NULL DEFAULT '',\n  config_digest TEXT NOT NULL DEFAULT '',\n", 1)
	// Every schema before 17 carried the requests row without its recorded machine word.
	priorRequestsSixteen := strings.Replace(priorRequestsEighteen,
		"  machine      TEXT    NOT NULL DEFAULT '',\n", "", 1)
	priorRequests := strings.Replace(priorRequestsSixteen,
		"  needs_accelerator INTEGER NOT NULL DEFAULT 0,\n",
		"  job_gpu_count INTEGER NOT NULL DEFAULT 0,\n", 1)
	priorRequestsSix := strings.Replace(priorRequests,
		"  rental_required INTEGER NOT NULL DEFAULT 0,\n", "", 1)
	// Schema 33 gave the rentals row its width; every earlier shape is this one without it.
	priorRentalsThirtyTwo := strings.Replace(rentalsDDL,
		"  accelerator_count INTEGER NOT NULL DEFAULT 0,\n", "", 1)
	priorRentalsTwenty := strings.Replace(priorRentalsThirtyTwo,
		"  ready_at          TEXT NOT NULL DEFAULT '',\n  failure_code                 TEXT NOT NULL DEFAULT '',\n  failure_image_digest         TEXT NOT NULL DEFAULT '',\n  failure_provider             TEXT NOT NULL DEFAULT '',\n  failure_provider_resource_id TEXT NOT NULL DEFAULT '',\n  failure_provider_host_id     TEXT NOT NULL DEFAULT '',\n  failure_provider_state       TEXT NOT NULL DEFAULT '',\n  failure_container_state      TEXT NOT NULL DEFAULT ''\n",
		"  ready_at          TEXT NOT NULL DEFAULT ''\n", 1)
	priorRentals := strings.Replace(priorRentalsTwenty,
		"  expected_worker_boot_id    TEXT NOT NULL DEFAULT '',\n  ready_at          TEXT NOT NULL DEFAULT ''\n",
		"  expected_worker_boot_id    TEXT NOT NULL DEFAULT ''\n", 1)
	priorOutputExports := strings.Replace(outputExportSchema,
		"  directory        TEXT    NOT NULL,\n",
		"  directory        TEXT    NOT NULL,\n  payload_hash     TEXT    NOT NULL,\n", 1)
	priorRequests = priorLocalPackageNames(priorRequests)
	priorRequestsSix = priorLocalPackageNames(priorRequestsSix)
	priorRequestsFourteen := priorLocalPackageNames(priorRequestsSixteen)
	priorInstalls := strings.Replace(installsDDL,
		"  package_interface TEXT    NOT NULL,\n", "  package_descriptor TEXT    NOT NULL,\n", 1)
	priorWorkerProcesses := strings.Replace(workerProcessesDDL,
		"  install_id      TEXT    REFERENCES installs(id),\n",
		"  install_id      TEXT    REFERENCES installs(id),\n  package_revision_digest      TEXT    NOT NULL,\n", 1)
	statements := make([]string, 0, len(schema)+len(schemaNineModelProduction))
	for _, statement := range schema {
		if version < 34 && (statement == childArgumentsDDL || statement == activeChildRequestIndex || statement == activeNativeCallIndex) {
			continue
		}
		if version < 33 && statement == byteOutputsDDL {
			continue
		}

		if version < 29 && statement == operationLookupsDDL {
			continue
		}
		if version < 31 && statement == nativeArtifactRetentionsDDL {
			continue
		}
		if version < 30 && statement == nativeCallsDDL {
			continue
		}
		if version < 28 && statement == weightsRetentionsDDL {
			continue
		}
		if version < 27 && (statement == childBindingsDDL || statement == childRequestIndex) {
			continue
		}
		if version < 10 && containsStatement(modelTransferSchema, statement) ||
			version < 16 && containsStatement(packageEventSchema, statement) ||
			version < 23 && (statement == modelCheckpointSchema || statement == modelCheckpointPublicationSchema) {
			continue
		}
		if version < 33 && statement == nativeArtifactRetentionsDDL {
			statement = strings.Replace(statement, " artifact_kind TEXT NOT NULL DEFAULT 'derived' CHECK(artifact_kind IN ('derived','tree')),\n producer_attempt INTEGER NOT NULL DEFAULT 0, producer_output_id TEXT NOT NULL DEFAULT '',content_bytes INTEGER NOT NULL DEFAULT 0,\n", "", 1)
		}
		if version < 32 && statement == nativeCallsDDL {
			statement = strings.Replace(statement, " cancel_requested INTEGER NOT NULL DEFAULT 0 CHECK(cancel_requested IN (0,1)),\n", "", 1)
		}
		statements = append(statements, statement)
	}
	if version < 10 {
		statements = append(statements, schemaNineModelProduction...)
	}
	for index, stmt := range statements {
		transferStatement := stmt == modelTransferSchema[0]
		requestStatement := stmt == requestsDDL
		switch {
		case stmt == requestsDDL && version == 6:
			stmt = priorRequestsSix
		case stmt == requestsDDL && version < 11:
			stmt = priorRequests
		case stmt == requestsDDL && version < 15:
			stmt = priorRequestsFourteen
		case stmt == requestsDDL && version < 17:
			stmt = priorRequestsSixteen
		case stmt == requestsDDL && version < 19:
			stmt = priorRequestsEighteen
		case stmt == requestsDDL && version < 20:
			stmt = priorRequestsNineteen
		case stmt == installsDDL && version < 20:
			stmt = priorInstalls
		case stmt == workerProcessesDDL && version < 20:
			stmt = priorWorkerProcesses
		case stmt == rentalsDDL && version < 8:
			stmt = rentalsDDLPrior
		case stmt == rentalsDDL && version < 13:
			stmt = priorRentals
		case stmt == rentalsDDL && version < 21:
			stmt = priorRentalsTwenty
		case stmt == rentalsDDL && version < 33:
			stmt = priorRentalsThirtyTwo
		case stmt == attemptsDDL && version < 22:
			stmt = strings.Replace(stmt,
				"  triage_bundle    BLOB    NOT NULL DEFAULT x'',\n",
				"  triage_path      TEXT    NOT NULL DEFAULT '',\n", 1)
		case stmt == outputExportSchema && version < 14:
			stmt = priorOutputExports
		case stmt == modelTransferSchema[0] && version < 23:
			stmt = strings.Replace(stmt, "  models_worker_boot_id TEXT NOT NULL DEFAULT '',\n", "", 1)
		case stmt == modelTransferSchema[1] && version < 23:
			stmt = strings.Replace(stmt, "  worker_boot_id      TEXT NOT NULL DEFAULT '',\n", "", 1)
		case stmt == modelTransferSchema[2] && version < 18:
			stmt = strings.Replace(stmt,
				"  manifest_length  INTEGER NOT NULL CHECK(manifest_length>0),\n",
				"  manifest_length  INTEGER NOT NULL CHECK(manifest_length>0),\n  evidence         BLOB NOT NULL,\n", 1)
		}
		if requestStatement && version < 27 {
			stmt = strings.Replace(stmt, ",\n  parent_request_id TEXT NOT NULL DEFAULT '',\n  parent_call_index INTEGER NOT NULL DEFAULT -1 CHECK(parent_call_index>=-1 AND parent_call_index<4294967296),\n  child_intent_digest TEXT NOT NULL DEFAULT '',\n  child_target_digest TEXT NOT NULL DEFAULT '',\n  child_reusable INTEGER NOT NULL DEFAULT 0 CHECK(child_reusable IN (0,1)),\n  reused_from TEXT NOT NULL DEFAULT '',\n  orchestration_directive BLOB NOT NULL DEFAULT x''", "", 1)
		}
		if requestStatement && version < 28 {
			stmt = strings.Replace(stmt, ",\n  child_artifacts INTEGER NOT NULL DEFAULT 0 CHECK(child_artifacts IN (0,1))", "", 1)
		}
		if requestStatement && version < 26 {
			stmt = strings.Replace(stmt, ",\n  retain_work INTEGER NOT NULL DEFAULT 0 CHECK(retain_work IN (0,1)),\n  retry_of TEXT NOT NULL DEFAULT '',\n  reuse_scope TEXT NOT NULL DEFAULT '',\n  control_revision INTEGER NOT NULL DEFAULT 0 CHECK(control_revision>=0)", "", 1)
		}
		if transferStatement && version < 24 {
			stmt = strings.Replace(stmt, ",'canceling'", "", 1)
		}
		if version < 25 {
			if stmt == modelCheckpointSchema {
				stmt = priorCheckpointSchema()
			}
			if stmt == modelCheckpointPublicationSchema {
				stmt = strings.Replace(stmt, "request_model_checkpoint_publications", "request_model_source_publications", 1)
			}
		}
		if version < 12 {
			stmt = priorInstallNames(stmt)
		}
		if requestStatement && version < 33 {
			stmt = strings.Replace(stmt, "  capture      TEXT    NOT NULL DEFAULT '',\n", "", 1)
		}
		if version < 34 {
			stmt = strings.ReplaceAll(stmt, "call_index<4294967296", "call_index<32")
		}
		statements[index] = stmt
	}
	return statements
}

func priorSchema(version int) ([]string, error) {
	db, err := sql.Open("sqlite", ":memory:"+pragmas)
	if err != nil {
		return nil, err
	}
	defer db.Close()
	for _, stmt := range priorStatements(version) {
		if _, err := db.Exec(stmt); err != nil {
			return nil, err
		}
	}
	return schemaSnapshot(db)
}

// priorInstallNames restores the pre-12 spelling of the install table and the two foreign
// keys that named it. Both retired names are exactly as long as the ones that replaced them,
// so the released DDL text — which verifyPriorSchema compares character for character — comes
// back by substitution instead of by keeping a second copy of three tables.
func priorInstallNames(stmt string) string {
	stmt = strings.Replace(stmt, "EXISTS installs (", "EXISTS install_generations (", 1)
	stmt = strings.ReplaceAll(stmt, "REFERENCES installs(id)", "REFERENCES install_generations(id)")
	stmt = strings.Replace(stmt,
		"  install_id   TEXT    NOT NULL REFERENCES", "  generation   TEXT    NOT NULL REFERENCES", 1)
	stmt = strings.Replace(stmt,
		"  install_id      TEXT    REFERENCES", "  generation      TEXT    REFERENCES", 1)
	return stmt
}

// priorLocalPackageNames restores the pre-15 spelling of the request's two editable
// revision columns, which said `private_package_*` before proto-027 settled on one word.
func priorLocalPackageNames(stmt string) string {
	stmt = strings.Replace(stmt, "  local_package_digest TEXT", "  private_package_digest TEXT", 1)
	stmt = strings.Replace(stmt, "  local_package_uploaded_boot_id TEXT", "  private_package_uploaded_boot_id TEXT", 1)
	return stmt
}

func containsStatement(statements []string, wanted string) bool {
	for _, statement := range statements {
		if statement == wanted {
			return true
		}
	}
	return false
}

func verifyPriorSchema(db *sql.DB, path string, version int) *exit.Error {
	want, err := priorSchema(version)
	if err != nil {
		return exit.Internalf("cannot derive schema-%d records shape: %s", version, err)
	}
	got, err := schemaSnapshot(db)
	if err != nil {
		return exit.Internalf("cannot inspect schema-%d records in %s: %s", version, path, err)
	}
	if strings.Join(got, "\n") != strings.Join(want, "\n") {
		return schemaReset(path, "records database claims schema %d but is not its exact released shape", version)
	}
	return nil
}

func commitMigration(tx *sql.Tx, path string) *exit.Error {
	if err := tx.Commit(); err != nil {
		return exit.Internalf("cannot commit records schema migration in %s: %s", path, err)
	}
	return nil
}

func databaseVersion(db interface{ QueryRow(string, ...any) *sql.Row }) (int, error) {
	var version int
	err := db.QueryRow(`PRAGMA user_version`).Scan(&version)
	return version, err
}

func initialize(db *sql.DB, path string) *exit.Error {
	tx, err := db.Begin()
	if err != nil {
		return exit.Internalf("cannot begin records initialization in %s: %s", path, err)
	}
	defer tx.Rollback()
	version, err := databaseVersion(tx)
	if err != nil {
		return exit.Internalf("cannot re-read records version in %s: %s", path, err)
	}
	if version == schemaVersion {
		if err := tx.Commit(); err != nil {
			return exit.Internalf("cannot adopt concurrent records initialization in %s: %s", path, err)
		}
		return nil
	}
	if version != 0 {
		return schemaReset(path, "records database changed to user_version %d while opening", version)
	}
	var objects int
	if err := tx.QueryRow(`SELECT COUNT(*) FROM sqlite_master
		WHERE name NOT LIKE 'sqlite_%'`).Scan(&objects); err != nil {
		return exit.Internalf("cannot inspect unversioned records database %s: %s", path, err)
	}
	if objects != 0 {
		return schemaReset(path, "unversioned records database contains %d schema objects", objects)
	}
	for _, stmt := range schema {
		if _, err := tx.Exec(stmt); err != nil {
			return exit.Internalf("cannot initialize records schema in %s: %s", path, err)
		}
	}
	if _, err := tx.Exec(fmt.Sprintf("PRAGMA user_version=%d", schemaVersion)); err != nil {
		return exit.Internalf("cannot stamp records schema in %s: %s", path, err)
	}
	if err := tx.Commit(); err != nil {
		return exit.Internalf("cannot commit records initialization in %s: %s", path, err)
	}
	return nil
}

func schemaReset(path, format string, args ...any) *exit.Error {
	return exit.Named(exit.Conflict, "records.schema_reset_required", format, args...).
		WithRemedy("stop Cozy, move %s aside, and start again to create the current records database", path)
}

type schemaReader interface {
	Query(string, ...any) (*sql.Rows, error)
}

func schemaSnapshot(db schemaReader) ([]string, error) {
	rows, err := db.Query(`SELECT type,name,COALESCE(sql,'') FROM sqlite_master
		WHERE name NOT LIKE 'sqlite_%' ORDER BY type,name`)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	var out []string
	for rows.Next() {
		var kind, name, ddl string
		if err := rows.Scan(&kind, &name, &ddl); err != nil {
			return nil, err
		}
		out = append(out, kind+"\x00"+name+"\x00"+ddl)
	}
	return out, rows.Err()
}

var expectedSchema struct {
	sync.Once
	rows []string
	err  error
}

func currentSchema() ([]string, error) {
	expectedSchema.Do(func() {
		db, err := sql.Open("sqlite", ":memory:"+pragmas)
		if err != nil {
			expectedSchema.err = err
			return
		}
		defer db.Close()
		for _, stmt := range schema {
			if _, err := db.Exec(stmt); err != nil {
				expectedSchema.err = err
				return
			}
		}
		expectedSchema.rows, expectedSchema.err = schemaSnapshot(db)
	})
	return expectedSchema.rows, expectedSchema.err
}

func verifySchema(db *sql.DB, path string) *exit.Error {
	want, err := currentSchema()
	if err != nil {
		return exit.Internalf("cannot derive current records schema: %s", err)
	}
	got, err := schemaSnapshot(db)
	if err != nil {
		return exit.Internalf("cannot inspect records schema in %s: %s", path, err)
	}
	if strings.Join(got, "\n") != strings.Join(want, "\n") {
		return schemaReset(path, "records database schema is not the exact current shape")
	}
	return nil
}

func (s *Store) Close() { _ = s.db.Close() }

var installFields = []string{
	"id", "package", "major", "version", "source_kind", "source_ref", "source_digest",
	"verified", "dir", "python", "runtime", "project_dir", "uv", "lock_digest", "platform", "extra",
	"packages", "closure", "package_interface", "placement_set_digest", "bytes_excl", "bytes_shared", "created_at",
}

// installCols is the select list, optionally table-qualified for a join.
func installCols(alias string) string {
	out := make([]string, len(installFields))
	for i, f := range installFields {
		out[i] = alias + f
	}
	return strings.Join(out, ",")
}

func placeholders() string {
	return strings.TrimSuffix(strings.Repeat("?,", len(installFields)), ",")
}

func scanInstall(rows interface{ Scan(...any) error }) (PackageInstall, error) {
	var inst PackageInstall
	var verified int
	err := rows.Scan(&inst.ID, &inst.Package, &inst.Major, &inst.Version, &inst.SourceKind, &inst.SourceRef,
		&inst.SourceDigest, &verified, &inst.Dir, &inst.Python, &inst.Runtime, &inst.ProjectDir, &inst.UV, &inst.LockDigest, &inst.Platform,
		&inst.Extra, &inst.Packages, &inst.Closure, &inst.PackageInterface, &inst.PlacementSetDigest,
		&inst.BytesExcl, &inst.BytesShared, &inst.CreatedAt)
	inst.Verified = verified == 1
	return inst, err
}

// Activate is THE install transaction: the install row and the pin swap commit
// together or not at all. A crash before Commit leaves the previous pin — and the
// previous install's venv — exactly as it was.
func (s *Store) Activate(inst PackageInstall) (superseded string, e *exit.Error) {
	return s.recordInstall(inst, true)
}

// RecordInstall retains an immutable invocation snapshot without replacing the
// user's editable package pin. The accepting request owns its lifetime.
func (s *Store) RecordInstall(inst PackageInstall) *exit.Error {
	_, problem := s.recordInstall(inst, false)
	return problem
}

func (s *Store) recordInstall(inst PackageInstall, activate bool) (string, *exit.Error) {
	tx, err := s.db.Begin()
	if err != nil {
		return "", exit.Internalf("cannot begin the activation transaction: %s", err)
	}
	defer tx.Rollback()

	var prior string
	err = tx.QueryRow(`SELECT install_id FROM pins WHERE package=?
		ORDER BY activated_at DESC LIMIT 1`, inst.Package).Scan(&prior)
	if err != nil && !errors.Is(err, sql.ErrNoRows) {
		return "", exit.Internalf("cannot read the current pin: %s", err)
	}

	inst.CreatedAt = time.Now().UTC().Format(time.RFC3339)
	verified := 0
	if inst.Verified {
		verified = 1
	}
	if _, err := tx.Exec(`INSERT INTO installs(`+installCols("")+`)
		VALUES(`+placeholders()+`)`,
		inst.ID, inst.Package, inst.Major, inst.Version, inst.SourceKind, inst.SourceRef, inst.SourceDigest,
		verified, inst.Dir, inst.Python, inst.Runtime, inst.ProjectDir, inst.UV, inst.LockDigest, inst.Platform, inst.Extra,
		inst.Packages, inst.Closure, inst.PackageInterface, inst.PlacementSetDigest,
		inst.BytesExcl, inst.BytesShared, inst.CreatedAt); err != nil {
		return "", exit.Internalf("cannot insert install %s: %s", inst.ID, err)
	}
	if activate {
		if _, err := tx.Exec(`DELETE FROM pins WHERE package=?`, inst.Package); err != nil {
			return "", exit.Internalf("cannot replace the active pin for %s: %s", inst.Package, err)
		}
		if _, err := tx.Exec(`INSERT INTO pins(package,major,install_id,activated_at)
			VALUES(?,?,?,?)`,
			inst.Package, inst.Major, inst.ID, inst.CreatedAt); err != nil {
			return "", exit.Internalf("cannot activate the pin for %s@v%d: %s", inst.Package, inst.Major, err)
		}
	} else {
		prior = ""
	}
	if err := tx.Commit(); err != nil {
		return "", exit.New(exit.Conflict,
			"the activation transaction did not commit for %s@v%d: %s", inst.Package, inst.Major, err).
			WithRemedy("the previous pin is untouched; re-run the install")
	}
	return prior, nil
}

// ActivePackage returns the package's one active install.
func (s *Store) ActivePackage(pkg string) (*Pin, *PackageInstall, *exit.Error) {
	var p Pin
	err := s.db.QueryRow(`SELECT package,major,install_id,activated_at FROM pins
		WHERE package=? ORDER BY activated_at DESC LIMIT 1`, pkg).
		Scan(&p.Package, &p.Major, &p.InstallID, &p.ActivatedAt)
	if errors.Is(err, sql.ErrNoRows) {
		return nil, nil, nil
	}
	if err != nil {
		return nil, nil, exit.Internalf("cannot read the pin for %s: %s", pkg, err)
	}
	inst, e := s.Install(p.InstallID)
	return &p, inst, e
}

// ActivePin returns the package pin only when it has the requested major. Removal uses
// this after listing the active pin; installation uses ActivePackage.
func (s *Store) ActivePin(pkg string, major int) (*Pin, *PackageInstall, *exit.Error) {
	var p Pin
	err := s.db.QueryRow(`SELECT package,major,install_id,activated_at FROM pins
		WHERE package=? AND major=?`, pkg, major).
		Scan(&p.Package, &p.Major, &p.InstallID, &p.ActivatedAt)
	if errors.Is(err, sql.ErrNoRows) {
		return nil, nil, nil
	}
	if err != nil {
		return nil, nil, exit.Internalf("cannot read the pin for %s@v%d: %s", pkg, major, err)
	}
	inst, e := s.Install(p.InstallID)
	return &p, inst, e
}

func (s *Store) Install(id string) (*PackageInstall, *exit.Error) {
	inst, err := scanInstall(s.db.QueryRow(`SELECT `+installCols("")+` FROM installs WHERE id=?`, id))
	if errors.Is(err, sql.ErrNoRows) {
		return nil, nil
	}
	if err != nil {
		return nil, exit.Internalf("cannot read install %s: %s", id, err)
	}
	return &inst, nil
}

// Installed is every active package pin joined to its install, package ordered.
// This is what `cozy package list` reads — records only, never a walk of the filesystem.
func (s *Store) Installed() ([]PackageInstall, *exit.Error) {
	rows, err := s.db.Query(`SELECT ` + installCols("i.") + `
		FROM installs i JOIN pins p ON p.install_id = i.id
		ORDER BY i.package, i.major`)
	if err != nil {
		return nil, exit.Internalf("cannot list installed packages: %s", err)
	}
	defer rows.Close()
	var out []PackageInstall
	for rows.Next() {
		inst, err := scanInstall(rows)
		if err != nil {
			return nil, exit.Internalf("cannot read an install record: %s", err)
		}
		out = append(out, inst)
	}
	return out, nil
}

// Unreferenced is every install no pin points at — gc's reclaim set.
func (s *Store) Unreferenced() ([]PackageInstall, *exit.Error) {
	rows, err := s.db.Query(`SELECT ` + installCols("i.") + `
		FROM installs i WHERE i.id NOT IN (SELECT install_id FROM pins)
		AND NOT EXISTS(SELECT 1 FROM requests WHERE install_id=i.id AND (state IN (` + activeRequestStates + `) OR (retain_work=1 AND state='succeeded' AND child_artifacts=1)))
		AND NOT EXISTS(SELECT 1 FROM private_child_bindings WHERE child_install_id=i.id)
		ORDER BY i.created_at`)
	if err != nil {
		return nil, exit.Internalf("cannot list unreferenced installs: %s", err)
	}
	defer rows.Close()
	var out []PackageInstall
	for rows.Next() {
		inst, err := scanInstall(rows)
		if err != nil {
			return nil, exit.Internalf("cannot read an install record: %s", err)
		}
		out = append(out, inst)
	}
	return out, nil
}

// Pins returns the package's active pin. The slice shape remains useful to old local
// databases long enough for the next activation to collapse any historical duplicates.
func (s *Store) Pins(pkg string) ([]Pin, *exit.Error) {
	rows, err := s.db.Query(`SELECT package,major,install_id,activated_at FROM pins
		WHERE package=? ORDER BY major`, pkg)
	if err != nil {
		return nil, exit.Internalf("cannot list pins for %s: %s", pkg, err)
	}
	defer rows.Close()
	var out []Pin
	for rows.Next() {
		var p Pin
		if err := rows.Scan(&p.Package, &p.Major, &p.InstallID, &p.ActivatedAt); err != nil {
			return nil, exit.Internalf("cannot read a pin: %s", err)
		}
		out = append(out, p)
	}
	return out, nil
}

// Unpin drops one pin. The install row survives as unreferenced until gc — `rm`
// removes the install, gc reclaims the bytes.
func (s *Store) Unpin(pkg string, major int) *exit.Error {
	if _, err := s.db.Exec(`DELETE FROM pins WHERE package=?`, pkg); err != nil {
		return exit.Internalf("cannot remove the pin for %s@v%d: %s", pkg, major, err)
	}
	return nil
}

// ForgetIfUnreferenced atomically claims one install for GC. The row goes before
// filesystem deletion, so a failure leaves an ordinary orphan the next GC can retry.
func (s *Store) ForgetIfUnreferenced(id string) (bool, *exit.Error) {
	tx, err := s.db.Begin()
	if err != nil {
		return false, exit.New(exit.Conflict, "cannot begin install %s gc: %s", id, err)
	}
	defer tx.Rollback()
	// Check ownership before clearing historical references. A rejected GC must
	// not sever the result schema from a completed child still owned by its parent.
	var held bool
	if err := tx.QueryRow(`SELECT EXISTS(SELECT 1 FROM pins WHERE install_id=?)
		OR EXISTS(SELECT 1 FROM requests WHERE install_id=? AND (state IN (`+activeRequestStates+`) OR (retain_work=1 AND state='succeeded' AND child_artifacts=1)))
		OR EXISTS(SELECT 1 FROM private_child_bindings WHERE child_install_id=?)
		OR EXISTS(SELECT 1 FROM worker_processes WHERE install_id=? AND state!='closed')`, id, id, id, id).Scan(&held); err != nil {
		return false, exit.New(exit.Conflict, "cannot inspect install %s gc ownership: %s", id, err)
	}
	if held {
		return false, nil
	}
	if _, err := tx.Exec(`UPDATE requests SET install_id=NULL WHERE install_id=?
		AND state NOT IN (`+activeRequestStates+`)`, id); err != nil {
		return false, exit.New(exit.Conflict,
			"cannot release terminal requests from install %s: %s", id, err)
	}
	if _, err := tx.Exec(`UPDATE worker_processes SET install_id=NULL WHERE install_id=?
		AND state='closed'`, id); err != nil {
		return false, exit.New(exit.Conflict, "cannot release closed workers from install %s: %s", id, err)
	}
	result, err := tx.Exec(`DELETE FROM installs WHERE id=?
		AND NOT EXISTS (SELECT 1 FROM pins WHERE install_id=?)
		AND NOT EXISTS (SELECT 1 FROM requests WHERE install_id=?
		  AND state IN (`+activeRequestStates+`))
		AND NOT EXISTS (SELECT 1 FROM private_child_bindings WHERE child_install_id=?)
		AND NOT EXISTS (SELECT 1 FROM worker_processes WHERE install_id=? AND state!='closed')`,
		id, id, id, id, id)
	if err != nil {
		return false, exit.New(exit.Conflict, "cannot claim install %s for gc: %s", id, err)
	}
	n, _ := result.RowsAffected()
	if err := tx.Commit(); err != nil {
		return false, exit.New(exit.Conflict, "cannot commit install %s gc: %s", id, err)
	}
	return n == 1, nil
}
