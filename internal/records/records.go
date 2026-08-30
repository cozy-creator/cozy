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
	SelectionProfile   string // Hub-selected base-worker compatibility profile
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

// schema is applied one statement at a time, so a refusal names the one table that
// refused rather than the whole script.
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
  selection_profile TEXT     NOT NULL DEFAULT '',
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
)`}, append(orchestratorSchema, append(eventSchema, rentalSchema...)...)...)

// renames is the pre-launch domain hardcut expressed as a database migration instead of
// a demand that users retain the executable that wrote an older local root. SQLite keeps
// row values, constraints, indexes and foreign-key references intact when it renames a
// column. Every identifier here is a source constant, never caller input.
var renames = []struct{ table, from, to string }{
	{"install_generations", "endpoint", "package"},
	{"install_generations", "descriptor", "package_descriptor"},
	{"pins", "endpoint", "package"},
	{"worker_processes", "endpoint", "package"},
	{"worker_processes", "release_id", "package_revision_digest"},
	{"worker_processes", "package_release_id", "package_revision_digest"},
	{"placement_acquisition_observations", "endpoint_started_ns", "package_started_ns"},
	{"placement_acquisition_observations", "endpoint_ended_ns", "package_ended_ns"},
	{"placement_acquisition_observations", "endpoint_downloaded_bytes", "package_downloaded_bytes"},
	{"placement_acquisition_observations", "endpoint_reused_bytes", "package_reused_bytes"},
	{"requests", "endpoint", "package"},
	{"attempts", "exec_spec_digest", "invocation_digest"},
	{"attempts", "exec_spec", "invocation"},
	{"rentals", "endpoint_ref", "package_ref"},
}

// pragmas ride the DSN rather than being executed after the open, because a pragma is a
// property of a CONNECTION and database/sql may discard and redial one at any moment: a
// re-dialled connection with foreign_keys OFF would silently accept the delete Forget
// exists to refuse. The driver replays them on every connection it opens.
//
//	busy_timeout  transient lock contention waits instead of failing immediately
//	foreign_keys  the pin -> generation reference is enforced, not decorative
//	journal_mode  WAL keeps reads independent of the single writer; it is persistent, so
//	              an older root converts on its first open here
const pragmas = "?_pragma=busy_timeout(5000)&_pragma=foreign_keys(1)&_pragma=journal_mode(WAL)"

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
	for _, stmt := range schema {
		if _, err := db.Exec(stmt); err != nil {
			db.Close()
			return nil, exit.Internalf("cannot apply the records schema to %s: %s", path, err)
		}
	}
	if e := migrateColumnRenames(db, path); e != nil {
		db.Close()
		return nil, e
	}
	if e := migrateRentalSchema(db, path); e != nil {
		db.Close()
		return nil, e
	}
	// A column added to a table an older root already created. `CREATE TABLE IF NOT
	// EXISTS` is a no-op on that root, so the new column would never appear; adding it
	// here is the whole migration story a pre-launch single-writer store needs. A
	// duplicate-column answer is the statement already having been applied.
	for _, stmt := range widen {
		if _, err := db.Exec(stmt); err != nil && !strings.Contains(err.Error(), "duplicate column") {
			db.Close()
			return nil, exit.Internalf("cannot widen the records schema in %s: %s", path, err)
		}
	}
	for _, stmt := range normalize {
		if _, err := db.Exec(stmt); err != nil {
			db.Close()
			return nil, exit.Internalf("cannot normalize lifecycle state in %s: %s", path, err)
		}
	}
	// A table whose IDENTITY changed. `CREATE TABLE IF NOT EXISTS` above left an older
	// root's shape in place and no `ALTER TABLE` can move a primary key, so the rows move
	// to a new table instead — in ONE transaction, so a kill mid-rebuild leaves the old
	// shape whole and the next open retries it.
	for _, r := range rebuild {
		var ddl string
		err := db.QueryRow(`SELECT COALESCE(sql,'') FROM sqlite_master
			WHERE type='table' AND name=?`, r.table).Scan(&ddl)
		if err != nil || !strings.Contains(ddl, r.stale) {
			continue
		}
		tx, err := db.Begin()
		if err != nil {
			db.Close()
			return nil, exit.Internalf("cannot begin the %s rebuild in %s: %s", r.table, path, err)
		}
		for _, stmt := range r.steps {
			if _, err := tx.Exec(stmt); err != nil {
				tx.Rollback()
				db.Close()
				return nil, exit.Internalf("cannot rebuild %s in %s: %s", r.table, path, err)
			}
		}
		if err := tx.Commit(); err != nil {
			db.Close()
			return nil, exit.Internalf("the %s rebuild did not commit in %s: %s", r.table, path, err)
		}
	}
	return &Store{db: db}, nil
}

func migrateColumnRenames(db *sql.DB, path string) *exit.Error {
	known := map[string]map[string]bool{}
	for _, rename := range renames {
		columns := known[rename.table]
		if columns == nil {
			var err error
			columns, err = tableColumns(db, rename.table)
			if err != nil {
				return exit.Internalf("cannot inspect %s in %s: %s", rename.table, path, err)
			}
			known[rename.table] = columns
		}
		if !columns[rename.from] || columns[rename.to] {
			continue
		}
		if _, err := db.Exec(`ALTER TABLE ` + rename.table + ` RENAME COLUMN ` +
			rename.from + ` TO ` + rename.to); err != nil {
			return exit.Internalf("cannot rename %s.%s to %s in %s: %s",
				rename.table, rename.from, rename.to, path, err)
		}
		delete(columns, rename.from)
		columns[rename.to] = true
	}
	return nil
}

func tableColumns(db *sql.DB, table string) (map[string]bool, error) {
	rows, err := db.Query(`PRAGMA table_info(` + table + `)`)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	columns := map[string]bool{}
	for rows.Next() {
		var id, notNull, primaryKey int
		var name, columnType string
		var defaultValue sql.NullString
		if err := rows.Scan(&id, &name, &columnType, &notNull, &defaultValue, &primaryKey); err != nil {
			return nil, err
		}
		columns[name] = true
	}
	return columns, rows.Err()
}

func (s *Store) Close() { _ = s.db.Close() }

var genFields = []string{
	"id", "package", "major", "version", "source_kind", "source_ref", "source_digest",
	"verified", "dir", "python", "runtime", "project_dir", "uv", "lock_digest", "platform", "extra", "link_mode",
	"packages", "closure", "package_descriptor", "selection_profile", "placement_set_digest", "bytes_excl", "bytes_shared", "created_at",
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
		&g.Extra, &g.LinkMode, &g.Packages, &g.Closure, &g.PackageDescriptor, &g.SelectionProfile, &g.PlacementSetDigest,
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
		g.Packages, g.Closure, g.PackageDescriptor, g.SelectionProfile, g.PlacementSetDigest,
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

// KnownIDs is every generation id on record — the set gc compares the generations
// directory against to find orphans a pre-activation crash left behind.
func (s *Store) KnownIDs() (map[string]bool, *exit.Error) {
	rows, err := s.db.Query(`SELECT id FROM install_generations`)
	if err != nil {
		return nil, exit.Internalf("cannot list generation ids: %s", err)
	}
	defer rows.Close()
	out := map[string]bool{}
	for rows.Next() {
		var id string
		if err := rows.Scan(&id); err != nil {
			return nil, exit.Internalf("cannot read a generation id: %s", err)
		}
		out[id] = true
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
		// Older roots gained install_id through a pre-launch NOT NULL widen and carry no
		// foreign key on that column. Empty is their historical spelling of no install;
		// fresh roots use nullable FK-backed identity.
		if _, fallback := tx.Exec(`UPDATE requests SET install_id='' WHERE install_id=?
			AND state NOT IN ('submitted','queued','dispatching','requeue_pending')`, id); fallback != nil {
			return false, exit.New(exit.Conflict,
				"cannot release terminal requests from generation %s: %s", id, err)
		}
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
