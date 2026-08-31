// Package records is the ONE local lifecycle authority: install generations and the
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
// that produced it. Rows are never updated — a rebuild is a NEW install. Was `Generation`,
// which the wire spends on executor and admission generations; the local install is not
// one of those, and one word for three fences is how they drift (#484).
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
	LinkMode           string // "hardlink" | "copy" (cross-mount degradation)
	Packages           int
	Closure            string // one "name==version" per line
	PackageDescriptor  string // exact digest of the generation-private Runtime-derived descriptor
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
	InstallID   string // was `Generation` (#484): it names an PackageInstall row's id
	ActivatedAt string
}

type Store struct{ db *sql.DB }

const schemaVersion = 2

// schema is the only records shape this pre-launch build accepts.
var schema = append([]string{`
CREATE TABLE IF NOT EXISTS install_generations (
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
  link_mode     TEXT    NOT NULL,
  packages      INTEGER NOT NULL,
  closure       TEXT    NOT NULL,
  package_descriptor TEXT    NOT NULL,
  placement_set_digest TEXT  NOT NULL DEFAULT '',
  bytes_excl    INTEGER NOT NULL,
  bytes_shared  INTEGER NOT NULL,
  created_at    TEXT    NOT NULL
)`, `
CREATE TABLE IF NOT EXISTS pins (
  package     TEXT    NOT NULL,
  major        INTEGER NOT NULL,
  generation   TEXT    NOT NULL REFERENCES install_generations(id),
  activated_at TEXT    NOT NULL,
  PRIMARY KEY (package)
)`}, append(modelProductionSchema,
	append(orchestratorSchema, append(eventSchema, rentalSchema...)...)...)...)

// pragmas ride the DSN rather than being executed after the open, because a pragma is a
// property of a CONNECTION and database/sql may discard and redial one at any moment: a
// re-dialled connection with foreign_keys OFF would silently accept the delete Forget
// exists to refuse. The driver replays them on every connection it opens.
//
//	busy_timeout  transient lock contention waits instead of failing immediately
//	foreign_keys  the pin -> generation reference is enforced, not decorative
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
	} else if version == 1 {
		if e := migrateOutputExports(db, path); e != nil {
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
	if _, err := tx.Exec(`PRAGMA user_version=2`); err != nil {
		return exit.Internalf("cannot stamp records schema in %s: %s", path, err)
	}
	if err := tx.Commit(); err != nil {
		return exit.Internalf("cannot commit records initialization in %s: %s", path, err)
	}
	return nil
}

// migrateOutputExports is the one pre-launch records migration with a real durability
// consumer: existing runs and rentals survive the addition of daemon-owned --out work.
// The new object is created from the exact same DDL currentSchema uses, so strict schema
// readback remains meaningful without rewriting any prior table.
func migrateOutputExports(db *sql.DB, path string) *exit.Error {
	tx, err := db.Begin()
	if err != nil {
		return exit.Internalf("cannot begin records migration in %s: %s", path, err)
	}
	defer tx.Rollback()
	version, err := databaseVersion(tx)
	if err != nil {
		return exit.Internalf("cannot re-read records version in %s: %s", path, err)
	}
	if version == schemaVersion {
		return nil
	}
	if version != 1 {
		return schemaReset(path, "records database changed to user_version %d while migrating", version)
	}
	want, err := schemaWithoutOutputExports()
	if err != nil {
		return exit.Internalf("cannot derive version 1 records schema: %s", err)
	}
	got, err := schemaSnapshot(tx)
	if err != nil {
		return exit.Internalf("cannot inspect version 1 records schema in %s: %s", path, err)
	}
	if strings.Join(got, "\n") != strings.Join(want, "\n") {
		return schemaReset(path, "records database schema is not the exact version 1 shape")
	}
	if _, err := tx.Exec(outputExportSchema); err != nil {
		return exit.Internalf("cannot add durable output exports in %s: %s", path, err)
	}
	if _, err := tx.Exec(`PRAGMA user_version=2`); err != nil {
		return exit.Internalf("cannot stamp records migration in %s: %s", path, err)
	}
	if err := tx.Commit(); err != nil {
		return exit.Internalf("cannot commit records migration in %s: %s", path, err)
	}
	return nil
}

func schemaWithoutOutputExports() ([]string, error) {
	db, err := sql.Open("sqlite", ":memory:"+pragmas)
	if err != nil {
		return nil, err
	}
	defer db.Close()
	for _, stmt := range schema {
		if stmt == outputExportSchema {
			continue
		}
		if _, err := db.Exec(stmt); err != nil {
			return nil, err
		}
	}
	return schemaSnapshot(db)
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

var genFields = []string{
	"id", "package", "major", "version", "source_kind", "source_ref", "source_digest",
	"verified", "dir", "python", "runtime", "project_dir", "uv", "lock_digest", "platform", "extra", "link_mode",
	"packages", "closure", "package_descriptor", "placement_set_digest", "bytes_excl", "bytes_shared", "created_at",
}

// genCols is the select list, optionally table-qualified for a join.
func genCols(alias string) string {
	out := make([]string, len(genFields))
	for i, f := range genFields {
		out[i] = alias + f
	}
	return strings.Join(out, ",")
}

func placeholders() string {
	return strings.TrimSuffix(strings.Repeat("?,", len(genFields)), ",")
}

func scanGen(rows interface{ Scan(...any) error }) (PackageInstall, error) {
	var g PackageInstall
	var verified int
	err := rows.Scan(&g.ID, &g.Package, &g.Major, &g.Version, &g.SourceKind, &g.SourceRef,
		&g.SourceDigest, &verified, &g.Dir, &g.Python, &g.Runtime, &g.ProjectDir, &g.UV, &g.LockDigest, &g.Platform,
		&g.Extra, &g.LinkMode, &g.Packages, &g.Closure, &g.PackageDescriptor, &g.PlacementSetDigest,
		&g.BytesExcl, &g.BytesShared, &g.CreatedAt)
	g.Verified = verified == 1
	return g, err
}

// Activate is THE install transaction: the generation row and the pin swap commit
// together or not at all. A crash before Commit leaves the previous pin — and the
// previous generation's venv — exactly as it was.
func (s *Store) Activate(g PackageInstall) (superseded string, e *exit.Error) {
	tx, err := s.db.Begin()
	if err != nil {
		return "", exit.Internalf("cannot begin the activation transaction: %s", err)
	}
	defer tx.Rollback()

	var prior string
	err = tx.QueryRow(`SELECT generation FROM pins WHERE package=?
		ORDER BY activated_at DESC LIMIT 1`, g.Package).Scan(&prior)
	if err != nil && !errors.Is(err, sql.ErrNoRows) {
		return "", exit.Internalf("cannot read the current pin: %s", err)
	}

	g.CreatedAt = time.Now().UTC().Format(time.RFC3339)
	verified := 0
	if g.Verified {
		verified = 1
	}
	if _, err := tx.Exec(`INSERT INTO install_generations(`+genCols("")+`)
		VALUES(`+placeholders()+`)`,
		g.ID, g.Package, g.Major, g.Version, g.SourceKind, g.SourceRef, g.SourceDigest,
		verified, g.Dir, g.Python, g.Runtime, g.ProjectDir, g.UV, g.LockDigest, g.Platform, g.Extra, g.LinkMode,
		g.Packages, g.Closure, g.PackageDescriptor, g.PlacementSetDigest,
		g.BytesExcl, g.BytesShared, g.CreatedAt); err != nil {
		return "", exit.Internalf("cannot insert generation %s: %s", g.ID, err)
	}
	if _, err := tx.Exec(`DELETE FROM pins WHERE package=?`, g.Package); err != nil {
		return "", exit.Internalf("cannot replace the active pin for %s: %s", g.Package, err)
	}
	if _, err := tx.Exec(`INSERT INTO pins(package,major,generation,activated_at)
		VALUES(?,?,?,?)`,
		g.Package, g.Major, g.ID, g.CreatedAt); err != nil {
		return "", exit.Internalf("cannot activate the pin for %s@v%d: %s", g.Package, g.Major, err)
	}
	if err := tx.Commit(); err != nil {
		return "", exit.New(exit.Conflict,
			"the activation transaction did not commit for %s@v%d: %s", g.Package, g.Major, err).
			WithRemedy("the previous pin is untouched; re-run the install")
	}
	return prior, nil
}

// ActivePackage returns the package's one active install.
func (s *Store) ActivePackage(pkg string) (*Pin, *PackageInstall, *exit.Error) {
	var p Pin
	err := s.db.QueryRow(`SELECT package,major,generation,activated_at FROM pins
		WHERE package=? ORDER BY activated_at DESC LIMIT 1`, pkg).
		Scan(&p.Package, &p.Major, &p.InstallID, &p.ActivatedAt)
	if errors.Is(err, sql.ErrNoRows) {
		return nil, nil, nil
	}
	if err != nil {
		return nil, nil, exit.Internalf("cannot read the pin for %s: %s", pkg, err)
	}
	g, e := s.Install(p.InstallID)
	return &p, g, e
}

// ActivePin returns the package pin only when it has the requested major. Removal uses
// this after listing the active pin; installation uses ActivePackage.
func (s *Store) ActivePin(pkg string, major int) (*Pin, *PackageInstall, *exit.Error) {
	var p Pin
	err := s.db.QueryRow(`SELECT package,major,generation,activated_at FROM pins
		WHERE package=? AND major=?`, pkg, major).
		Scan(&p.Package, &p.Major, &p.InstallID, &p.ActivatedAt)
	if errors.Is(err, sql.ErrNoRows) {
		return nil, nil, nil
	}
	if err != nil {
		return nil, nil, exit.Internalf("cannot read the pin for %s@v%d: %s", pkg, major, err)
	}
	g, e := s.Install(p.InstallID)
	return &p, g, e
}

func (s *Store) Install(id string) (*PackageInstall, *exit.Error) {
	g, err := scanGen(s.db.QueryRow(`SELECT `+genCols("")+` FROM install_generations WHERE id=?`, id))
	if errors.Is(err, sql.ErrNoRows) {
		return nil, nil
	}
	if err != nil {
		return nil, exit.Internalf("cannot read generation %s: %s", id, err)
	}
	return &g, nil
}

// Installed is every active package pin joined to its generation, package ordered.
// This is what `cozy package list` reads — records only, never a walk of the filesystem.
func (s *Store) Installed() ([]PackageInstall, *exit.Error) {
	rows, err := s.db.Query(`SELECT ` + genCols("g.") + `
		FROM install_generations g JOIN pins p ON p.generation = g.id
		ORDER BY g.package, g.major`)
	if err != nil {
		return nil, exit.Internalf("cannot list installed packages: %s", err)
	}
	defer rows.Close()
	var out []PackageInstall
	for rows.Next() {
		g, err := scanGen(rows)
		if err != nil {
			return nil, exit.Internalf("cannot read an install record: %s", err)
		}
		out = append(out, g)
	}
	return out, nil
}

// Unreferenced is every generation no pin points at — gc's reclaim set.
func (s *Store) Unreferenced() ([]PackageInstall, *exit.Error) {
	rows, err := s.db.Query(`SELECT ` + genCols("g.") + `
		FROM install_generations g WHERE g.id NOT IN (SELECT generation FROM pins)
		ORDER BY g.created_at`)
	if err != nil {
		return nil, exit.Internalf("cannot list unreferenced generations: %s", err)
	}
	defer rows.Close()
	var out []PackageInstall
	for rows.Next() {
		g, err := scanGen(rows)
		if err != nil {
			return nil, exit.Internalf("cannot read a generation record: %s", err)
		}
		out = append(out, g)
	}
	return out, nil
}

// Pins returns the package's active pin. The slice shape remains useful to old local
// databases long enough for the next activation to collapse any historical duplicates.
func (s *Store) Pins(pkg string) ([]Pin, *exit.Error) {
	rows, err := s.db.Query(`SELECT package,major,generation,activated_at FROM pins
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

// Unpin drops one pin. The generation row survives as unreferenced until gc — `rm`
// removes the install, gc reclaims the bytes.
func (s *Store) Unpin(pkg string, major int) *exit.Error {
	if _, err := s.db.Exec(`DELETE FROM pins WHERE package=?`, pkg); err != nil {
		return exit.Internalf("cannot remove the pin for %s@v%d: %s", pkg, major, err)
	}
	return nil
}

// ForgetIfUnreferenced atomically claims one generation for GC. The row goes before
// filesystem deletion, so a failure leaves an ordinary orphan the next GC can retry.
func (s *Store) ForgetIfUnreferenced(id string) (bool, *exit.Error) {
	tx, err := s.db.Begin()
	if err != nil {
		return false, exit.New(exit.Conflict, "cannot begin generation %s gc: %s", id, err)
	}
	defer tx.Rollback()
	if _, err := tx.Exec(`UPDATE requests SET install_id=NULL WHERE install_id=?
		AND state NOT IN ('submitted','queued','dispatching','requeue_pending')`, id); err != nil {
		return false, exit.New(exit.Conflict,
			"cannot release terminal requests from generation %s: %s", id, err)
	}
	if _, err := tx.Exec(`UPDATE worker_processes SET generation=NULL WHERE generation=?
		AND state='closed'`, id); err != nil {
		return false, exit.New(exit.Conflict, "cannot release closed workers from generation %s: %s", id, err)
	}
	result, err := tx.Exec(`DELETE FROM install_generations WHERE id=?
		AND NOT EXISTS (SELECT 1 FROM pins WHERE generation=?)
		AND NOT EXISTS (SELECT 1 FROM requests WHERE install_id=?
		  AND state IN ('submitted','queued','dispatching','requeue_pending'))
		AND NOT EXISTS (SELECT 1 FROM worker_processes WHERE generation=? AND state!='closed')`,
		id, id, id, id)
	if err != nil {
		return false, exit.New(exit.Conflict, "cannot claim generation %s for gc: %s", id, err)
	}
	n, _ := result.RowsAffected()
	if err := tx.Commit(); err != nil {
		return false, exit.New(exit.Conflict, "cannot commit generation %s gc: %s", id, err)
	}
	return n == 1, nil
}
