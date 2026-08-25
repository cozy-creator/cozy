// Package records is the ONE local lifecycle authority: install generations and the
// active pin per (endpoint, major), as rows in ONE local Turso/libSQL database
// (cozy-creator.md "Records" — Turso, never vanilla SQLite). There is no state.json
// and no second lifecycle store; any JSON output is a derived read.
//
// Seam for cl-001: the LocalService adopts THIS package as its lifecycle store and
// adds its own tables (worker sessions, requests, attempts, outputs) to the same
// database. Nothing here assumes a CLI caller; Open takes a path.
//
// Driver (settles cozy-creator.md's OPEN ENGINEERING QUESTION for the local-file
// case): github.com/tursodatabase/go-libsql, the embedded libSQL driver. It is
// CGO-based, so `cozy` builds with CGO_ENABLED=1; qualifying that per platform for
// the static-binary requirement is cl-013's distribution work.
package records

import (
	"database/sql"
	"errors"
	"strings"
	"time"

	"github.com/cozy-creator/cozy-creator-v2/internal/exit"
	_ "github.com/tursodatabase/go-libsql"
)

// Generation is one immutable install: a built environment plus the evidence that
// produced it. Rows are never updated — a rebuild is a NEW generation.
type Generation struct {
	ID           string
	Endpoint     string // org/name
	Major        int
	Version      string
	SourceKind   string // "archive" | "dir"
	SourceRef    string
	SourceDigest string
	Verified     bool // false = installed through the --allow-unsigned development door
	Dir          string
	Python       string
	UV           string
	LockDigest   string
	Platform     string
	Extra        string // the CUDA-extra pick ("" = none declared or no accelerator)
	LinkMode     string // "hardlink" | "copy" (cross-mount degradation)
	Packages     int
	Closure      string // one "name==version" per line
	Descriptor   string // "pending-cr-003" until the runtime's describe exists
	BytesExcl    int64
	BytesShared  int64
	CreatedAt    string
}

// Pin is the active generation for one (endpoint, major). Two majors of one
// endpoint coexist because the key is the pair.
type Pin struct {
	Endpoint    string
	Major       int
	Generation  string
	ActivatedAt string
}

type Store struct{ db *sql.DB }

// schema is applied one statement at a time: the driver executes a single
// statement per call.
var schema = []string{`
CREATE TABLE IF NOT EXISTS install_generations (
  id            TEXT PRIMARY KEY,
  endpoint      TEXT    NOT NULL,
  major         INTEGER NOT NULL,
  version       TEXT    NOT NULL,
  source_kind   TEXT    NOT NULL,
  source_ref    TEXT    NOT NULL,
  source_digest TEXT    NOT NULL,
  verified      INTEGER NOT NULL,
  dir           TEXT    NOT NULL,
  python        TEXT    NOT NULL,
  uv            TEXT    NOT NULL,
  lock_digest   TEXT    NOT NULL,
  platform      TEXT    NOT NULL,
  extra         TEXT    NOT NULL,
  link_mode     TEXT    NOT NULL,
  packages      INTEGER NOT NULL,
  closure       TEXT    NOT NULL,
  descriptor    TEXT    NOT NULL,
  bytes_excl    INTEGER NOT NULL,
  bytes_shared  INTEGER NOT NULL,
  created_at    TEXT    NOT NULL
)`, `
CREATE TABLE IF NOT EXISTS pins (
  endpoint     TEXT    NOT NULL,
  major        INTEGER NOT NULL,
  generation   TEXT    NOT NULL REFERENCES install_generations(id),
  activated_at TEXT    NOT NULL,
  PRIMARY KEY (endpoint, major)
)`}

func Open(path string) (*Store, *exit.Error) {
	db, err := sql.Open("libsql", "file:"+path)
	if err != nil {
		return nil, exit.Internalf("cannot open the local records database %s: %s", path, err)
	}
	if _, err := db.Exec("PRAGMA foreign_keys=ON"); err != nil {
		db.Close()
		return nil, exit.Internalf("cannot enable foreign keys on %s: %s", path, err)
	}
	for _, stmt := range schema {
		if _, err := db.Exec(stmt); err != nil {
			db.Close()
			return nil, exit.Internalf("cannot apply the records schema to %s: %s", path, err)
		}
	}
	return &Store{db: db}, nil
}

func (s *Store) Close() { _ = s.db.Close() }

var genFields = []string{
	"id", "endpoint", "major", "version", "source_kind", "source_ref", "source_digest",
	"verified", "dir", "python", "uv", "lock_digest", "platform", "extra", "link_mode",
	"packages", "closure", "descriptor", "bytes_excl", "bytes_shared", "created_at",
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

func scanGen(rows interface{ Scan(...any) error }) (Generation, error) {
	var g Generation
	var verified int
	err := rows.Scan(&g.ID, &g.Endpoint, &g.Major, &g.Version, &g.SourceKind, &g.SourceRef,
		&g.SourceDigest, &verified, &g.Dir, &g.Python, &g.UV, &g.LockDigest, &g.Platform,
		&g.Extra, &g.LinkMode, &g.Packages, &g.Closure, &g.Descriptor,
		&g.BytesExcl, &g.BytesShared, &g.CreatedAt)
	g.Verified = verified == 1
	return g, err
}

// Activate is THE install transaction: the generation row and the pin swap commit
// together or not at all. A crash before Commit leaves the previous pin — and the
// previous generation's venv — exactly as it was.
func (s *Store) Activate(g Generation) (superseded string, e *exit.Error) {
	tx, err := s.db.Begin()
	if err != nil {
		return "", exit.Internalf("cannot begin the activation transaction: %s", err)
	}
	defer tx.Rollback()

	var prior string
	err = tx.QueryRow(`SELECT generation FROM pins WHERE endpoint=? AND major=?`,
		g.Endpoint, g.Major).Scan(&prior)
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
		g.ID, g.Endpoint, g.Major, g.Version, g.SourceKind, g.SourceRef, g.SourceDigest,
		verified, g.Dir, g.Python, g.UV, g.LockDigest, g.Platform, g.Extra, g.LinkMode,
		g.Packages, g.Closure, g.Descriptor, g.BytesExcl, g.BytesShared, g.CreatedAt); err != nil {
		return "", exit.Internalf("cannot insert generation %s: %s", g.ID, err)
	}
	if _, err := tx.Exec(`INSERT INTO pins(endpoint,major,generation,activated_at)
		VALUES(?,?,?,?) ON CONFLICT(endpoint,major) DO UPDATE SET generation=excluded.generation,
		activated_at=excluded.activated_at`,
		g.Endpoint, g.Major, g.ID, g.CreatedAt); err != nil {
		return "", exit.Internalf("cannot activate the pin for %s@v%d: %s", g.Endpoint, g.Major, err)
	}
	if err := tx.Commit(); err != nil {
		return "", exit.New(exit.Conflict,
			"the activation transaction did not commit for %s@v%d: %s", g.Endpoint, g.Major, err).
			WithRemedy("the previous pin is untouched; re-run the install")
	}
	return prior, nil
}

// ActivePin returns the pinned generation for one (endpoint, major).
func (s *Store) ActivePin(endpoint string, major int) (*Pin, *Generation, *exit.Error) {
	var p Pin
	err := s.db.QueryRow(`SELECT endpoint,major,generation,activated_at FROM pins
		WHERE endpoint=? AND major=?`, endpoint, major).
		Scan(&p.Endpoint, &p.Major, &p.Generation, &p.ActivatedAt)
	if errors.Is(err, sql.ErrNoRows) {
		return nil, nil, nil
	}
	if err != nil {
		return nil, nil, exit.Internalf("cannot read the pin for %s@v%d: %s", endpoint, major, err)
	}
	g, e := s.Generation(p.Generation)
	return &p, g, e
}

func (s *Store) Generation(id string) (*Generation, *exit.Error) {
	g, err := scanGen(s.db.QueryRow(`SELECT `+genCols("")+` FROM install_generations WHERE id=?`, id))
	if errors.Is(err, sql.ErrNoRows) {
		return nil, nil
	}
	if err != nil {
		return nil, exit.Internalf("cannot read generation %s: %s", id, err)
	}
	return &g, nil
}

// Installed is every active pin joined to its generation, endpoint-major ordered.
// This is what `cozy ls` reads — records only, never a walk of the filesystem.
func (s *Store) Installed() ([]Generation, *exit.Error) {
	rows, err := s.db.Query(`SELECT ` + genCols("g.") + `
		FROM install_generations g JOIN pins p ON p.generation = g.id
		ORDER BY g.endpoint, g.major`)
	if err != nil {
		return nil, exit.Internalf("cannot list installed endpoints: %s", err)
	}
	defer rows.Close()
	var out []Generation
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
func (s *Store) Unreferenced() ([]Generation, *exit.Error) {
	rows, err := s.db.Query(`SELECT ` + genCols("g.") + `
		FROM install_generations g WHERE g.id NOT IN (SELECT generation FROM pins)
		ORDER BY g.created_at`)
	if err != nil {
		return nil, exit.Internalf("cannot list unreferenced generations: %s", err)
	}
	defer rows.Close()
	var out []Generation
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

// Pins of one endpoint across every major (rm without an explicit major).
func (s *Store) Pins(endpoint string) ([]Pin, *exit.Error) {
	rows, err := s.db.Query(`SELECT endpoint,major,generation,activated_at FROM pins
		WHERE endpoint=? ORDER BY major`, endpoint)
	if err != nil {
		return nil, exit.Internalf("cannot list pins for %s: %s", endpoint, err)
	}
	defer rows.Close()
	var out []Pin
	for rows.Next() {
		var p Pin
		if err := rows.Scan(&p.Endpoint, &p.Major, &p.Generation, &p.ActivatedAt); err != nil {
			return nil, exit.Internalf("cannot read a pin: %s", err)
		}
		out = append(out, p)
	}
	return out, nil
}

// Unpin drops one pin. The generation row survives as unreferenced until gc — `rm`
// removes the install, gc reclaims the bytes.
func (s *Store) Unpin(endpoint string, major int) *exit.Error {
	if _, err := s.db.Exec(`DELETE FROM pins WHERE endpoint=? AND major=?`, endpoint, major); err != nil {
		return exit.Internalf("cannot remove the pin for %s@v%d: %s", endpoint, major, err)
	}
	return nil
}

// Forget drops an unreferenced generation row. The foreign key refuses while a pin
// still points at it, so a live install can never be forgotten by accident.
func (s *Store) Forget(id string) *exit.Error {
	if _, err := s.db.Exec(`DELETE FROM install_generations WHERE id=?`, id); err != nil {
		return exit.New(exit.Conflict, "cannot forget generation %s: %s", id, err).
			WithRemedy("a pin still references it — `cozy rm` the install first")
	}
	return nil
}
