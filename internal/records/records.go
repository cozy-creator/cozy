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
// cross-compiles with CGO_ENABLED=0 for every platform Go targets.
package records

import (
	"database/sql"
	"errors"
	"fmt"
	"os"
	"strings"
	"sync"
	"time"

	"github.com/cozy-creator/cozy/internal/exit"
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
	SourceKind         string // "tensorhub" | "local" | "wheel" (private immutable dependency)
	SourceRef          string
	Verified           bool // false = an explicit local/development install, not published custody
	Dir                string
	Python             string
	Runtime            string // exact cozy-runtime binary; empty on older source installs derives from venv
	ProjectDir         string // exact installed project root; empty on source installs derives from source kind
	UV                 string
	Platform           string
	Extra              string // the CUDA-extra pick ("" = none declared or no accelerator)
	Packages           int
	Closure            string // one "name==version" per line
	PlacementSetDigest string // exact Hub-selected PlacementSet/1 stored in the artifact cache
	BytesExcl          int64
	BytesShared        int64
	CreatedAt          string
	// Hub is the Tensorhub a published install came from; "" for a local install.
	Hub string
}

// Pin is the package's one active install. Major remains recorded for package metadata
// and removal output; activating any release replaces the package's previous pin.
type Pin struct {
	Package     string
	Major       int
	InstallID   string // names a PackageInstall row's id
	ActivatedAt string
}

type Store struct {
	db        *sql.DB
	telemetry telemetryV1
}

const schemaVersion = 49

const installsDDL = `
CREATE TABLE IF NOT EXISTS installs (
  id            TEXT PRIMARY KEY,
  package      TEXT    NOT NULL,
  major         INTEGER NOT NULL,
  version       TEXT    NOT NULL,
  source_kind   TEXT    NOT NULL,
  source_ref    TEXT    NOT NULL,
  verified      INTEGER NOT NULL,
  dir           TEXT    NOT NULL,
  python        TEXT    NOT NULL,
  runtime       TEXT    NOT NULL DEFAULT '',
  project_dir   TEXT    NOT NULL DEFAULT '',
  uv            TEXT    NOT NULL,
  platform      TEXT    NOT NULL,
  extra         TEXT    NOT NULL,
  packages      INTEGER NOT NULL,
  closure       TEXT    NOT NULL,
  placement_set_digest TEXT  NOT NULL DEFAULT '',
  bytes_excl    INTEGER NOT NULL,
  bytes_shared  INTEGER NOT NULL,
  created_at    TEXT    NOT NULL,
  hub           TEXT    NOT NULL DEFAULT ''
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
	schema = append(schema, successfulWorkDDL, weightsRetentionsDDL, operationLookupsDDL, nativeCallsDDL, nativeArtifactRetentionsDDL, byteOutputsDDL, nativeByteOutputIndex, childArgumentsDDL, activeChildRequestIndex, activeNativeCallIndex, servingPlacementsDDL, operationContextsDDL)
	schema = append(schema, machineExecutionSchema...)
	schema = append(schema, ownerMemoIndex)
	schema = append(schema, rentalInstallsDDL, rentalInstallsIndex, runtimeUpdatesDDL, rentalIdleDDL, bindingRevisionDDL,
		deviceMemoryMeasurementsDDL, deviceMemoryMeasurementsIndex)
	schema = append(schema, obligationIndexes...)
}

// pragmas ride the DSN rather than being executed after the open, because a pragma is a
// property of a CONNECTION and database/sql may discard and redial one at any moment: a
// re-dialled connection with foreign_keys OFF would silently accept the delete Forget
// exists to refuse. The driver replays them on every connection it opens.
//
//	busy_timeout  one round of waiting for another writer (driverName waits out the rest)
//	foreign_keys  the pin -> install reference is enforced, not decorative
//	txlock        writers reserve the lock before reading, so two processes cannot both
//	              read and then fail immediately while upgrading a deferred transaction
const pragmas = "?_pragma=busy_timeout(100)&_pragma=foreign_keys(1)&_txlock=immediate"

func Open(path string) (*Store, *exit.Error) {
	return open(path)
}

// OpenForDaemon is Open plus the daemon's startup correction of retained-work projections.
func OpenForDaemon(path string) (*Store, *exit.Error) {
	store, problem := open(path)
	if problem != nil {
		return nil, problem
	}
	if problem := store.ReconcileLostRetainedWork(); problem != nil {
		store.Close()
		return nil, problem
	}
	return store, nil
}

// open accepts an empty database, the current schema, or a newer one. The current schema
// evolves additively: verifySchema adds any table, column or index the authored DDL has and
// the database lacks. An older numbered schema is refused; this build carries no migrations.
func open(path string) (*Store, *exit.Error) {
	// No `file:` prefix: the driver hands an unprefixed name to SQLite verbatim, so a
	// local root containing `%` or `#` stays a path instead of becoming a URI to decode.
	db, err := sql.Open(driverName, path+pragmas)
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
	switch {
	case version == 0:
		if e := initialize(db, path); e != nil {
			db.Close()
			return nil, e
		}
	case version < schemaVersion:
		db.Close()
		return nil, exit.Named(exit.Conflict, "records_schema_unsupported",
			"records database %s has schema %d; this Creator reads schema %d and does not migrate older records",
			path, version, schemaVersion).
			WithRemedy("move %s aside and run the command again; a new database is created", path)
	case version > schemaVersion:
		// A newer Creator wrote this database. This build reads and writes only the tables
		// and columns it knows, and it changes nothing else: no re-stamp, no added or
		// dropped table or column, no rebuilt row.
		if e := verifyNewerSchema(db, path, version); e != nil {
			db.Close()
			return nil, e
		}
		schemaNotice.Do(func() {
			fmt.Fprintf(os.Stderr, "records written by a newer Creator (v%d); running in compatibility mode, upgrade for full features\n", version)
		})
	}
	if version <= schemaVersion {
		if e := verifySchema(db, path); e != nil {
			db.Close()
			return nil, e
		}
		if err := renameLifecycleEvents(db); err != nil {
			db.Close()
			return nil, exit.Internalf("cannot rename lifecycle events in %s: %s", path, err)
		}
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
		return schemaChanged(path, version)
	}
	required, err := requiredCurrentShape()
	if err != nil {
		return exit.Internalf("cannot derive current records schema: %s", err)
	}
	if _, err := conform(tx, required); err != nil {
		return exit.Internalf("cannot initialize records schema in %s: %s", path, err)
	}
	if _, err := tx.Exec(fmt.Sprintf("PRAGMA user_version=%d", schemaVersion)); err != nil {
		return exit.Internalf("cannot stamp records schema in %s: %s", path, err)
	}
	if err := tx.Commit(); err != nil {
		return exit.Internalf("cannot commit records initialization in %s: %s", path, err)
	}
	return nil
}

// schemaNotice names a newer or incompletely indexed database once per process.
var schemaNotice sync.Once

// verifyNewerSchema admits a newer Creator's database when every table and column this build
// requires is present with a compatible type. It never changes the database.
func verifyNewerSchema(db *sql.DB, path string, version int) *exit.Error {
	required, err := requiredCurrentShape()
	if err != nil {
		return exit.Internalf("cannot derive current records schema: %s", err)
	}
	lacking, err := missing(db, required)
	if err == nil {
		var mismatched []string
		mismatched, err = incompatible(db, required)
		lacking = append(lacking, mismatched...)
	}
	if err != nil {
		return exit.Internalf("cannot inspect records schema in %s: %s", path, err)
	}
	if len(lacking) > 0 {
		return exit.Named(exit.Conflict, "records_schema_newer",
			"records written by a newer Creator (v%d) lack what this build requires: %s", version, strings.Join(lacking, ", ")).
			WithRemedy("upgrade Cozy Creator to a version that supports schema %d; keep %s in place", version, path)
	}
	return nil
}

// schemaChanged answers a database another process re-stamped while this one opened it.
func schemaChanged(path string, version int) *exit.Error {
	return exit.Named(exit.Conflict, "records.schema_changed_concurrently",
		"records database %s changed to schema %d while it was being opened", path, version).
		WithRemedy("run the command again")
}

// verifySchema creates every required table, column and index the database lacks: without an
// index a query scans every event ever recorded. An index its rows refuse is named once and
// skipped. Extra columns, indexes and tables and differences in DDL text are accepted.
func verifySchema(db *sql.DB, path string) *exit.Error {
	required, err := requiredCurrentShape()
	if err != nil {
		return exit.Internalf("cannot derive current records schema: %s", err)
	}
	unbuilt, err := conform(db, required)
	if err != nil {
		return exit.Named(exit.Conflict, "records.schema_incomplete",
			"records database %s is incomplete and cannot be completed: %s", path, err).
			WithRemedy("keep %s in place; its rows are intact — run the Cozy Creator build that wrote it", path)
	}
	if len(unbuilt) > 0 {
		schemaNotice.Do(func() {
			fmt.Fprintf(os.Stderr, "records database %s lacks %s; commands run without it, more slowly\n", path, strings.Join(unbuilt, ", "))
		})
	}
	return nil
}

func (s *Store) Close() {
	s.stopTelemetry()
	_ = s.db.Close()
}

var installFields = []string{
	"id", "package", "major", "version", "source_kind", "source_ref",
	"verified", "dir", "python", "runtime", "project_dir", "uv", "platform", "extra",
	"packages", "closure", "placement_set_digest", "bytes_excl", "bytes_shared", "created_at",
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
	return scanInstallFields(rows, false)
}

// installHubCols is installCols plus the install's hub, for readers of the current
// schema. Package recovery reads a backup through installCols alone.
func installHubCols(alias string) string { return installCols(alias) + "," + alias + "hub" }

func scanInstallHub(rows interface{ Scan(...any) error }) (PackageInstall, error) {
	return scanInstallFields(rows, true)
}

func scanInstallFields(rows interface{ Scan(...any) error }, withHub bool) (PackageInstall, error) {
	var inst PackageInstall
	var verified int
	targets := []any{&inst.ID, &inst.Package, &inst.Major, &inst.Version, &inst.SourceKind, &inst.SourceRef,
		&verified, &inst.Dir, &inst.Python, &inst.Runtime, &inst.ProjectDir, &inst.UV, &inst.Platform,
		&inst.Extra, &inst.Packages, &inst.Closure, &inst.PlacementSetDigest,
		&inst.BytesExcl, &inst.BytesShared, &inst.CreatedAt}
	if withHub {
		targets = append(targets, &inst.Hub)
	}
	err := rows.Scan(targets...)
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
	if _, err := tx.Exec(`INSERT INTO installs(`+installHubCols("")+`)
		VALUES(`+placeholders()+`,?)`,
		inst.ID, inst.Package, inst.Major, inst.Version, inst.SourceKind, inst.SourceRef,
		verified, inst.Dir, inst.Python, inst.Runtime, inst.ProjectDir, inst.UV, inst.Platform, inst.Extra,
		inst.Packages, inst.Closure, inst.PlacementSetDigest,
		inst.BytesExcl, inst.BytesShared, inst.CreatedAt, strings.TrimRight(inst.Hub, "/")); err != nil {
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
	inst, err := scanInstallHub(s.db.QueryRow(`SELECT `+installHubCols("")+` FROM installs WHERE id=?`, id))
	if errors.Is(err, sql.ErrNoRows) {
		return nil, nil
	}
	if err != nil {
		return nil, exit.Internalf("cannot read install %s: %s", id, err)
	}
	return &inst, nil
}

// SourceEnvironments returns recorded same-package candidates for capture-time
// metadata reuse. The installer still compares their frozen dependency inputs;
// an install ID or version alone never makes an environment reusable. Capture
// holds the install writer, so GC cannot remove an unreferenced candidate while
// its metadata is read. Completed history need not keep a caller alive.
func (s *Store) SourceEnvironments(pkg, version string) ([]PackageInstall, *exit.Error) {
	rows, err := s.db.Query(`SELECT `+installCols("i.")+` FROM installs i
		WHERE i.package=? AND i.version=? AND i.source_kind='local'
		ORDER BY i.created_at DESC,i.id DESC`, pkg, version)
	if err != nil {
		return nil, exit.Internalf("cannot read retained source environments: %s", err)
	}
	defer rows.Close()
	var out []PackageInstall
	for rows.Next() {
		inst, err := scanInstall(rows)
		if err != nil {
			return nil, exit.Internalf("cannot read retained source environment: %s", err)
		}
		out = append(out, inst)
	}
	if err := rows.Err(); err != nil {
		return nil, exit.Internalf("cannot finish retained source environments: %s", err)
	}
	return out, nil
}

// Installed is every active package pin joined to its install, package ordered.
// This is what `cozy package list` reads — records only, never a walk of the filesystem.
func (s *Store) Installed() ([]PackageInstall, *exit.Error) {
	rows, err := s.db.Query(`SELECT ` + installHubCols("i.") + `
		FROM installs i JOIN pins p ON p.install_id = i.id
		ORDER BY i.package, i.major`)
	if err != nil {
		return nil, exit.Internalf("cannot list installed packages: %s", err)
	}
	defer rows.Close()
	var out []PackageInstall
	for rows.Next() {
		inst, err := scanInstallHub(rows)
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
		-- Keep the newest local environment for each exact package/version. Remote
		-- capture may reuse that compatible metadata environment after a daemon
		-- restart; unlike a pin, this retention is bounded to one candidate per
		-- source revision and older candidates remain ordinary GC work.
		AND NOT (i.source_kind='local' AND NOT EXISTS(
			SELECT 1 FROM installs newer
			WHERE newer.source_kind='local' AND newer.package=i.package AND newer.version=i.version
			AND (newer.created_at > i.created_at OR (newer.created_at = i.created_at AND newer.id > i.id))
		))
		AND NOT EXISTS(SELECT 1 FROM requests WHERE install_id=i.id AND (state IN (` + activeRequestStates + `) OR (retain_work=1 AND state='succeeded' AND child_artifacts=1)))
		AND NOT EXISTS(SELECT 1 FROM private_child_bindings WHERE child_install_id=i.id AND parent_install_id!=i.id)
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

// Pins returns the package's active pin, if any.
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
		OR EXISTS(SELECT 1 FROM private_child_bindings WHERE child_install_id=? AND parent_install_id!=child_install_id)
		OR EXISTS(SELECT 1 FROM worker_processes WHERE install_id=? AND state!='closed')
		OR EXISTS(SELECT 1 FROM installs candidate
			WHERE candidate.id=? AND candidate.source_kind='local'
			AND NOT EXISTS(SELECT 1 FROM installs newer
				WHERE newer.source_kind='local' AND newer.package=candidate.package AND newer.version=candidate.version
				AND (newer.created_at > candidate.created_at OR (newer.created_at = candidate.created_at AND newer.id > candidate.id))))`, id, id, id, id, id).Scan(&held); err != nil {
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
		AND NOT EXISTS (SELECT 1 FROM private_child_bindings WHERE child_install_id=? AND parent_install_id!=child_install_id)
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
