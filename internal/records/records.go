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
	"database/sql"
	"errors"
	"strings"
	"sync"
	"time"

	"github.com/cozy-creator/cozy/internal/exit"
	_ "modernc.org/sqlite"
)

// PackageInstall is one immutable install: a materialized environment plus the evidence
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
	PackageDescriptor  string // exact digest of the install-private Runtime-derived descriptor
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

const schemaVersion = 15

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
  package_descriptor TEXT    NOT NULL,
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
var schema = append([]string{installsDDL, pinsDDL}, append(orchestratorSchema,
	append(modelTransferSchema, append(eventSchema, rentalSchema...)...)...)...)

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
		if e := migrate(db, path, version); e != nil {
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
	if _, err := db.Exec(`PRAGMA journal_mode=WAL`); err != nil {
		db.Close()
		return nil, exit.Internalf("cannot enable WAL for %s: %s", path, err)
	}
	return &Store{db: db}, nil
}

// Schemas 6 through 14 migrate in place. Schema 11 replaces authored GPU counts with
// the one derived accelerator-class fact; schema 12 gives the install table and its two
// foreign keys the one word the row actually names; schema 13 records when a rental was
// first seen ready; schema 14 drops the output export's pre-execution payload hash — a
// file is named by its own content digest now; schema 15 spells the request's editable
// revision columns `local_package_*` (proto-027: one word for a local package). Package,
// request, event, export, and rental rows survive; only schema 9's superseded special
// model-production subsystem is dropped.
// Schema 10 creates empty request-attached transfer sidecars because older rows cannot be
// translated into ordinary request identity safely.
func migrate(db *sql.DB, path string, sourceVersion int) *exit.Error {
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
	// hash goes at 14, and the request rebuild ends at 15 (its column spelling). A 14 → 15
	// migration touches requests alone.
	if sourceVersion < 12 {
		if e := migrateInstalls(tx, path); e != nil {
			return e
		}
	}
	if sourceVersion < 13 {
		if e := migrateRentals(tx, path); e != nil {
			return e
		}
	}
	if sourceVersion < 14 {
		if e := migrateOutputExports(tx, path); e != nil {
			return e
		}
	}
	if e := migrateRequests(tx, path, sourceVersion); e != nil {
		return e
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
	if _, err := tx.Exec(`PRAGMA user_version=15`); err != nil {
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
func migrateInstalls(tx *sql.Tx, path string) *exit.Error {
	for _, statement := range []string{
		`DROP INDEX worker_session`,
		`ALTER TABLE install_generations RENAME TO installs_prior`,
		`ALTER TABLE pins RENAME TO pins_prior`,
		`ALTER TABLE worker_processes RENAME TO worker_processes_prior`,
		installsDDL,
		`INSERT INTO installs(` + installCols("") + `) SELECT ` + installCols("") +
			` FROM installs_prior`,
		pinsDDL,
		`INSERT INTO pins(package,major,install_id,activated_at)
			SELECT package,major,generation,activated_at FROM pins_prior`,
		workerProcessesDDL,
		`INSERT INTO worker_processes(` + workerProcessCols + `)
			SELECT instance_id,package,generation,package_revision_digest,worker_id,
			  devices,pid,birth,session_id,state,opened_at,closed_at FROM worker_processes_prior`,
		workerSessionIndex,
		`DROP TABLE worker_processes_prior`,
		`DROP TABLE pins_prior`,
		`DROP TABLE installs_prior`,
	} {
		if _, err := tx.Exec(statement); err != nil {
			return exit.Internalf("cannot rename the install table while migrating %s: %s", path, err)
		}
	}
	return nil
}

// migrateRentals rebuilds the rentals table on every migration: schema 13 added ready_at,
// and every earlier shape carries the released column list without it. A rental that was
// already ready when the schema moved gets no ready_at; its idle clock starts at its next
// settlement, never at a guess.
func migrateRentals(tx *sql.Tx, path string) *exit.Error {
	if _, err := tx.Exec(`DROP INDEX rentals_machine_name`); err != nil {
		return exit.Internalf("cannot stage rental index while migrating %s: %s", path, err)
	}
	if _, err := tx.Exec(`ALTER TABLE rentals RENAME TO rentals_prior`); err != nil {
		return exit.Internalf("cannot stage rental rows while migrating %s: %s", path, err)
	}
	if _, err := tx.Exec(rentalsDDL); err != nil {
		return exit.Internalf("cannot create current rentals table while migrating %s: %s", path, err)
	}
	if _, err := tx.Exec(`INSERT INTO rentals(` + rentalColsPriorThirteen + `) SELECT ` +
		rentalColsPriorThirteen + ` FROM rentals_prior`); err != nil {
		return exit.Internalf("cannot preserve rental rows while migrating %s: %s", path, err)
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

// migrateRequests rebuilds the requests table on every migration: before schema 11 to derive
// the accelerator-class fact from the retired GPU count, at schema 12 because the row's own
// DDL text names the install table it references, and at schema 15 because the editable
// revision columns are spelled `local_package_*`.
func migrateRequests(tx *sql.Tx, path string, sourceVersion int) *exit.Error {
	if _, err := tx.Exec(`ALTER TABLE requests RENAME TO requests_prior`); err != nil {
		return exit.Internalf("cannot stage request rows while migrating %s: %s", path, err)
	}
	if _, err := tx.Exec(requestsDDL); err != nil {
		return exit.Internalf("cannot create current requests table while migrating %s: %s", path, err)
	}
	destinationColumns := `id,idem_key,body_digest,package,entrypoint,plan_id,package_release,
		package_revision_digest,local_package_digest,local_package_uploaded_boot_id,
		environment_digest,config_digest,payload,outputs,state,ordinal,requeues,created_at,kind,
		needs_accelerator,org,trees,worker,rental,rental_required,install_id,assets,models,weights_outputs`
	rentalRequired := "rental_required"
	if sourceVersion == 6 {
		rentalRequired = "0"
	}
	needsAccelerator := "needs_accelerator"
	if sourceVersion < 11 {
		needsAccelerator = `CASE WHEN job_gpu_count>0 OR (kind!='job' AND models!='[]') THEN 1 ELSE 0 END`
	}
	selectColumns := `id,idem_key,body_digest,package,entrypoint,plan_id,package_release,
		package_revision_digest,private_package_digest,private_package_uploaded_boot_id,
		environment_digest,config_digest,payload,outputs,state,ordinal,requeues,created_at,kind,` +
		needsAccelerator + `,
		org,trees,worker,rental,` + rentalRequired + `,install_id,assets,models,weights_outputs`
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
	priorRequests := strings.Replace(requestsDDL,
		"  needs_accelerator INTEGER NOT NULL DEFAULT 0,\n",
		"  job_gpu_count INTEGER NOT NULL DEFAULT 0,\n", 1)
	priorRequestsSix := strings.Replace(priorRequests,
		"  rental_required INTEGER NOT NULL DEFAULT 0,\n", "", 1)
	priorRentals := strings.Replace(rentalsDDL,
		"  expected_worker_boot_id    TEXT NOT NULL DEFAULT '',\n  ready_at          TEXT NOT NULL DEFAULT ''\n",
		"  expected_worker_boot_id    TEXT NOT NULL DEFAULT ''\n", 1)
	priorOutputExports := strings.Replace(outputExportSchema,
		"  directory        TEXT    NOT NULL,\n",
		"  directory        TEXT    NOT NULL,\n  payload_hash     TEXT    NOT NULL,\n", 1)
	priorRequests = priorLocalPackageNames(priorRequests)
	priorRequestsSix = priorLocalPackageNames(priorRequestsSix)
	priorRequestsFourteen := priorLocalPackageNames(requestsDDL)
	statements := make([]string, 0, len(schema)+len(schemaNineModelProduction))
	for _, statement := range schema {
		if version >= 10 || !containsStatement(modelTransferSchema, statement) {
			statements = append(statements, statement)
		}
	}
	if version < 10 {
		statements = append(statements, schemaNineModelProduction...)
	}
	for index, stmt := range statements {
		switch {
		case stmt == requestsDDL && version == 6:
			stmt = priorRequestsSix
		case stmt == requestsDDL && version < 11:
			stmt = priorRequests
		case stmt == requestsDDL && version < 15:
			stmt = priorRequestsFourteen
		case stmt == rentalsDDL && version < 8:
			stmt = rentalsDDLPrior
		case stmt == rentalsDDL && version < 13:
			stmt = priorRentals
		case stmt == outputExportSchema && version < 14:
			stmt = priorOutputExports
		}
		if version < 12 {
			stmt = priorInstallNames(stmt)
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
	if _, err := tx.Exec(`PRAGMA user_version=15`); err != nil {
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
	"packages", "closure", "package_descriptor", "placement_set_digest", "bytes_excl", "bytes_shared", "created_at",
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
		&inst.Extra, &inst.Packages, &inst.Closure, &inst.PackageDescriptor, &inst.PlacementSetDigest,
		&inst.BytesExcl, &inst.BytesShared, &inst.CreatedAt)
	inst.Verified = verified == 1
	return inst, err
}

// Activate is THE install transaction: the install row and the pin swap commit
// together or not at all. A crash before Commit leaves the previous pin — and the
// previous install's venv — exactly as it was.
func (s *Store) Activate(inst PackageInstall) (superseded string, e *exit.Error) {
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
		inst.Packages, inst.Closure, inst.PackageDescriptor, inst.PlacementSetDigest,
		inst.BytesExcl, inst.BytesShared, inst.CreatedAt); err != nil {
		return "", exit.Internalf("cannot insert install %s: %s", inst.ID, err)
	}
	if _, err := tx.Exec(`DELETE FROM pins WHERE package=?`, inst.Package); err != nil {
		return "", exit.Internalf("cannot replace the active pin for %s: %s", inst.Package, err)
	}
	if _, err := tx.Exec(`INSERT INTO pins(package,major,install_id,activated_at)
		VALUES(?,?,?,?)`,
		inst.Package, inst.Major, inst.ID, inst.CreatedAt); err != nil {
		return "", exit.Internalf("cannot activate the pin for %s@v%d: %s", inst.Package, inst.Major, err)
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
	if _, err := tx.Exec(`UPDATE requests SET install_id=NULL WHERE install_id=?
		AND state NOT IN ('submitted','queued','dispatching','requeue_pending')`, id); err != nil {
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
		  AND state IN ('submitted','queued','dispatching','requeue_pending'))
		AND NOT EXISTS (SELECT 1 FROM worker_processes WHERE install_id=? AND state!='closed')`,
		id, id, id, id)
	if err != nil {
		return false, exit.New(exit.Conflict, "cannot claim install %s for gc: %s", id, err)
	}
	n, _ := result.RowsAffected()
	if err := tx.Commit(); err != nil {
		return false, exit.New(exit.Conflict, "cannot commit install %s gc: %s", id, err)
	}
	return n == 1, nil
}
