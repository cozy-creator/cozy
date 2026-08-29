// Package records is the ONE local lifecycle authority: install generations and the
// active pin per (endpoint, major), as rows in ONE local SQLite database
// (cozy-creator.md "Records"). There is no state.json and no second lifecycle store;
// any JSON output is a derived read.
//
// Seam for cl-001: the local controller adopts THIS package as its lifecycle store and
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

	"github.com/cozy-creator/cozy-creator/internal/exit"
	_ "modernc.org/sqlite"
)

// EndpointInstall is one immutable install: a materialized environment plus the evidence
// that produced it. Rows are never updated — a rebuild is a NEW install. Was `Generation`,
// which the wire spends on executor and admission generations; the local install is not
// one of those, and one word for three fences is how they drift (#484).
type EndpointInstall struct {
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
	Runtime      string // exact cozy-runtime binary; empty on older source installs derives from venv
	ProjectDir   string // exact installed project root; empty on source installs derives from source kind
	UV           string
	LockDigest   string
	Platform     string
	Extra        string // the CUDA-extra pick ("" = none declared or no accelerator)
	LinkMode     string // "hardlink" | "copy" (cross-mount degradation)
	Packages     int
	Closure      string // one "name==version" per line
	Descriptor   string // exact descriptor_digest, checked in this generation's own venv
	BytesExcl    int64
	BytesShared  int64
	CreatedAt    string
}

// Pin is the active install for one (endpoint, major). Two majors of one endpoint coexist
// because the key is the pair.
type Pin struct {
	Endpoint    string
	Major       int
	InstallID   string // was `Generation` (#484): it names an EndpointInstall row's id
	ActivatedAt string
}

type Store struct{ db *sql.DB }

// schema is applied one statement at a time, so a refusal names the one table that
// refused rather than the whole script.
var schema = append([]string{`
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
  runtime       TEXT    NOT NULL DEFAULT '',
  project_dir   TEXT    NOT NULL DEFAULT '',
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
)`, `
CREATE TABLE IF NOT EXISTS managed_profile_installs (
  install_id                    TEXT PRIMARY KEY REFERENCES install_generations(id) ON DELETE CASCADE,
  release_id                   TEXT NOT NULL,
  profile                      TEXT NOT NULL,
  candidate_id                 TEXT NOT NULL,
  base_realization_digest      TEXT NOT NULL,
  wheelhouse_manifest_digest   TEXT NOT NULL,
  environment_spec_digest      TEXT NOT NULL,
  endpoint_bundle_digest       TEXT NOT NULL,
  resolved_wheel_set_digest    TEXT NOT NULL,
  resolution_lock_digest       TEXT NOT NULL,
  installed_receipt_digest     TEXT NOT NULL,
  installed_receipt_length     INTEGER NOT NULL,
  host_evidence_digest         TEXT NOT NULL,
  native_evidence_digest       TEXT NOT NULL DEFAULT '',
  lease_id                     TEXT NOT NULL,
  lease_expires_at             TEXT NOT NULL,
  recorded_at                  TEXT NOT NULL
)`}, append(orchestratorSchema, append(eventSchema, rentalSchema...)...)...)

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
	if e := CheckRentalSchema(db, path); e != nil {
		db.Close()
		return nil, e
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

func (s *Store) Close() { _ = s.db.Close() }

var genFields = []string{
	"id", "endpoint", "major", "version", "source_kind", "source_ref", "source_digest",
	"verified", "dir", "python", "runtime", "project_dir", "uv", "lock_digest", "platform", "extra", "link_mode",
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

func scanGen(rows interface{ Scan(...any) error }) (EndpointInstall, error) {
	var g EndpointInstall
	var verified int
	err := rows.Scan(&g.ID, &g.Endpoint, &g.Major, &g.Version, &g.SourceKind, &g.SourceRef,
		&g.SourceDigest, &verified, &g.Dir, &g.Python, &g.Runtime, &g.ProjectDir, &g.UV, &g.LockDigest, &g.Platform,
		&g.Extra, &g.LinkMode, &g.Packages, &g.Closure, &g.Descriptor,
		&g.BytesExcl, &g.BytesShared, &g.CreatedAt)
	g.Verified = verified == 1
	return g, err
}

// Activate is THE install transaction: the generation row and the pin swap commit
// together or not at all. A crash before Commit leaves the previous pin — and the
// previous generation's venv — exactly as it was.
func (s *Store) Activate(g EndpointInstall) (superseded string, e *exit.Error) {
	return s.activate(g, nil)
}

type ManagedProfileInstall struct {
	InstallID                string
	ReleaseID                string
	Profile                  string
	CandidateID              string
	BaseRealizationDigest    string
	WheelhouseManifestDigest string
	EnvironmentSpecDigest    string
	EndpointBundleDigest     string
	ResolvedWheelSetDigest   string
	ResolutionLockDigest     string
	InstalledReceiptDigest   string
	InstalledReceiptLength   int64
	HostEvidenceDigest       string
	NativeEvidenceDigest     string
	LeaseID                  string
	LeaseExpiresAt           string
	RecordedAt               string
}

// ActivateManaged commits the ordinary control install/pin and its independent
// profile realization, receipt, host evidence, and lease facts in one transaction.
func (s *Store) ActivateManaged(g EndpointInstall, facts ManagedProfileInstall) (string, *exit.Error) {
	facts.InstallID = g.ID
	for name, value := range map[string]string{
		"release": facts.ReleaseID, "profile": facts.Profile, "candidate": facts.CandidateID,
		"base realization": facts.BaseRealizationDigest, "wheelhouse": facts.WheelhouseManifestDigest,
		"environment": facts.EnvironmentSpecDigest, "bundle": facts.EndpointBundleDigest,
		"resolved wheels": facts.ResolvedWheelSetDigest, "resolution lock": facts.ResolutionLockDigest,
		"receipt": facts.InstalledReceiptDigest, "host evidence": facts.HostEvidenceDigest,
		"lease": facts.LeaseID, "lease expiry": facts.LeaseExpiresAt,
	} {
		if strings.TrimSpace(value) == "" {
			return "", exit.Internalf("managed install has no %s fact", name)
		}
	}
	if facts.InstalledReceiptLength <= 0 {
		return "", exit.Internalf("managed install has no positive receipt length")
	}
	return s.activate(g, &facts)
}

func (s *Store) activate(g EndpointInstall, managed *ManagedProfileInstall) (superseded string, e *exit.Error) {
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
		verified, g.Dir, g.Python, g.Runtime, g.ProjectDir, g.UV, g.LockDigest, g.Platform, g.Extra, g.LinkMode,
		g.Packages, g.Closure, g.Descriptor, g.BytesExcl, g.BytesShared, g.CreatedAt); err != nil {
		return "", exit.Internalf("cannot insert generation %s: %s", g.ID, err)
	}
	if managed != nil {
		managed.RecordedAt = g.CreatedAt
		if _, err := tx.Exec(`INSERT INTO managed_profile_installs(
			install_id,release_id,profile,candidate_id,base_realization_digest,
			wheelhouse_manifest_digest,environment_spec_digest,
			endpoint_bundle_digest,resolved_wheel_set_digest,resolution_lock_digest,
			installed_receipt_digest,installed_receipt_length,host_evidence_digest,
			native_evidence_digest,lease_id,lease_expires_at,recorded_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)`,
			managed.InstallID, managed.ReleaseID, managed.Profile, managed.CandidateID,
			managed.BaseRealizationDigest,
			managed.WheelhouseManifestDigest, managed.EnvironmentSpecDigest,
			managed.EndpointBundleDigest, managed.ResolvedWheelSetDigest,
			managed.ResolutionLockDigest, managed.InstalledReceiptDigest,
			managed.InstalledReceiptLength, managed.HostEvidenceDigest,
			managed.NativeEvidenceDigest,
			managed.LeaseID, managed.LeaseExpiresAt, managed.RecordedAt); err != nil {
			return "", exit.Internalf("cannot insert managed profile facts for %s: %s", g.ID, err)
		}
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

func (s *Store) ManagedInstall(installID string) (*ManagedProfileInstall, *exit.Error) {
	var out ManagedProfileInstall
	err := s.db.QueryRow(`SELECT install_id,release_id,profile,candidate_id,
		base_realization_digest,wheelhouse_manifest_digest,
		environment_spec_digest,endpoint_bundle_digest,resolved_wheel_set_digest,
		resolution_lock_digest,installed_receipt_digest,installed_receipt_length,
		host_evidence_digest,native_evidence_digest,lease_id,lease_expires_at,recorded_at
		FROM managed_profile_installs WHERE install_id=?`, installID).Scan(
		&out.InstallID, &out.ReleaseID, &out.Profile, &out.CandidateID,
		&out.BaseRealizationDigest, &out.WheelhouseManifestDigest,
		&out.EnvironmentSpecDigest, &out.EndpointBundleDigest, &out.ResolvedWheelSetDigest,
		&out.ResolutionLockDigest, &out.InstalledReceiptDigest, &out.InstalledReceiptLength,
		&out.HostEvidenceDigest, &out.NativeEvidenceDigest, &out.LeaseID, &out.LeaseExpiresAt, &out.RecordedAt)
	if errors.Is(err, sql.ErrNoRows) {
		return nil, nil
	}
	if err != nil {
		return nil, exit.Internalf("cannot read managed profile facts for %s: %s", installID, err)
	}
	return &out, nil
}

// ActivePin returns the pinned generation for one (endpoint, major).
func (s *Store) ActivePin(endpoint string, major int) (*Pin, *EndpointInstall, *exit.Error) {
	var p Pin
	err := s.db.QueryRow(`SELECT endpoint,major,generation,activated_at FROM pins
		WHERE endpoint=? AND major=?`, endpoint, major).
		Scan(&p.Endpoint, &p.Major, &p.InstallID, &p.ActivatedAt)
	if errors.Is(err, sql.ErrNoRows) {
		return nil, nil, nil
	}
	if err != nil {
		return nil, nil, exit.Internalf("cannot read the pin for %s@v%d: %s", endpoint, major, err)
	}
	g, e := s.Install(p.InstallID)
	return &p, g, e
}

func (s *Store) Install(id string) (*EndpointInstall, *exit.Error) {
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
// This is what `cozy endpoint list` reads — records only, never a walk of the filesystem.
func (s *Store) Installed() ([]EndpointInstall, *exit.Error) {
	rows, err := s.db.Query(`SELECT ` + genCols("g.") + `
		FROM install_generations g JOIN pins p ON p.generation = g.id
		ORDER BY g.endpoint, g.major`)
	if err != nil {
		return nil, exit.Internalf("cannot list installed endpoints: %s", err)
	}
	defer rows.Close()
	var out []EndpointInstall
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
func (s *Store) Unreferenced() ([]EndpointInstall, *exit.Error) {
	rows, err := s.db.Query(`SELECT ` + genCols("g.") + `
		FROM install_generations g WHERE g.id NOT IN (SELECT generation FROM pins)
		ORDER BY g.created_at`)
	if err != nil {
		return nil, exit.Internalf("cannot list unreferenced generations: %s", err)
	}
	defer rows.Close()
	var out []EndpointInstall
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
		if err := rows.Scan(&p.Endpoint, &p.Major, &p.InstallID, &p.ActivatedAt); err != nil {
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

// ForgetIfUnreferenced atomically claims one generation for GC. The row goes before
// filesystem deletion, so a failure leaves an ordinary orphan the next GC can retry.
func (s *Store) ForgetIfUnreferenced(id string) (bool, *exit.Error) {
	result, err := s.db.Exec(`DELETE FROM install_generations WHERE id=?
		AND NOT EXISTS (SELECT 1 FROM pins WHERE generation=?)
		AND NOT EXISTS (SELECT 1 FROM requests WHERE install_id=?
		  AND state IN ('submitted','queued','dispatching','requeue_pending'))
		AND NOT EXISTS (SELECT 1 FROM worker_processes WHERE generation=? AND state!='closed')`,
		id, id, id, id)
	if err != nil {
		return false, exit.New(exit.Conflict, "cannot claim generation %s for gc: %s", id, err)
	}
	n, _ := result.RowsAffected()
	return n == 1, nil
}
