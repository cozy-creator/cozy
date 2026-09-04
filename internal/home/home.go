// Package home is the ONE place that knows the local layout. It derives paths from a
// root it is GIVEN — the environment is read once, in internal/config, and COZY_HOME
// carries a value, never a decision (cozy-runtime-cli.md): it moves the root, nothing else.
package home

import (
	"crypto/sha256"
	"database/sql"
	"encoding/hex"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"time"

	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/flock"
	_ "modernc.org/sqlite"
)

// Layout is the resolved set of paths every cl-009 verb works against. Every
// top-level entry is one of: durable Creator authority, user-owned input/output,
// protected credential, immutable install, bounded log, runtime lock, or typed live
// work (proto-030's home diet, carried by cl-116). Everything else was deleted or
// nested under its owner; nothing here may reintroduce a retired root.
type Layout struct {
	Root     string
	DB       string // creator.sqlite — the one local SQLite lifecycle database
	Installs string // one immutable directory per package install
	Lock     string // installs/.lock — the install-filesystem single-writer flock
	// Daemon is the Cozy daemon's lifetime ownership/address record (cl-001). The live
	// owner holds its kernel lock and publishes the client address, worker socket and
	// the per-launch CLI token under it, mode 0600. There is no separate client.cred.
	Daemon  string
	Workers string // per-worker roots: journal, logs, staged binding plans (live only)
	// Outputs is the user-facing store: `<org>-<package>/<digest>.<ext>`, one file per
	// result file, named by its own content digest so a regenerated file lands on itself.
	// User-owned result files are never automatically removed.
	Outputs string
	// Inputs is the content-addressed request input store, created on demand. cl-116's
	// end state narrows it to uploaded/streamed bodies only — bytes without a stable
	// caller path; the zero-copy local-CLI-file arm is still open and Stage currently
	// copies those too.
	Inputs string
	// Tmp is typed in-flight work: a model transfer's request-named staging and the
	// model-acquisition flocks under `tmp/locks`. Verb-lifetime exchange scratch uses OS
	// temp; the whole root disappears when nothing is in flight.
	Tmp string
	// Publications is the job publication plane (cl-004): one directory per job scratch
	// repo, reclaimed when its request settles without a committed publication.
	Publications string
	// Rentals holds one rented pod's SECRET MATERIAL: its media bearer (0600) and the
	// certificate this client pins when it dials (cl-015). Secrets whose rental is
	// proven absent are erased by the boot sweep.
	Rentals string
	// LocalPackages holds exact ephemeral wheel revisions for rented local-package
	// commands. It is staging the daemon alone writes, never a catalog.
	LocalPackages string
	// Log is the Cozy daemon's own log, bounded by rotation on an observed size
	// (internal/daemon.OpenLog). Its one rotated predecessor is Log + ".1".
	Log string
}

func Open(root string) (Layout, *exit.Error) {
	if root == "" {
		return Layout{}, exit.Internalf("the local root is unset: internal/config.Load did not run")
	}
	// The records database retains request payloads and path-free creative prompts; the
	// daemon lock carries the CLI token and rental secrets live under rentals/, so this
	// is an OS-private user root. Protecting the directory also protects SQLite's
	// lazily-created WAL/SHM siblings.
	if err := os.MkdirAll(root, 0o700); err != nil {
		return Layout{}, exit.Internalf("cannot create the private local root %s: %s", root, err)
	}
	if err := os.Chmod(root, 0o700); err != nil {
		return Layout{}, exit.Internalf("cannot protect the private local root %s: %s", root, err)
	}
	l := Layout{
		Root:     root,
		DB:       filepath.Join(root, "creator.sqlite"),
		Installs: filepath.Join(root, "installs"),
		Daemon:   filepath.Join(root, "daemon.lock"),
		Workers:  filepath.Join(root, "workers"),
		Outputs:  filepath.Join(root, "outputs"),
		Inputs:   filepath.Join(root, "inputs"),
		Tmp:      filepath.Join(root, "tmp"),
	}
	l.Lock = filepath.Join(l.Installs, ".lock")
	l.Publications = filepath.Join(root, "publications")
	l.Rentals = filepath.Join(root, "rentals")
	l.LocalPackages = filepath.Join(root, "local-packages")
	l.Log = filepath.Join(root, "daemon.log")
	// A prior root's records.db is the same database under its retired name. The rename
	// runs here — cheap, idempotent, and before any open — so no second code path ever
	// reads the old spelling. WAL/SHM siblings move with it or not at all: a database
	// whose main file moved without its WAL would silently lose committed pages.
	if e := renameRecords(root, l.Daemon, l.DB); e != nil {
		return Layout{}, e
	}
	// Only the two roots every verb touches exist up front. Everything else is created
	// by its writer at the moment work exists, and reclaimed when the work dies.
	for _, dir := range []string{l.Installs, l.Outputs} {
		if err := os.MkdirAll(dir, 0o755); err != nil {
			return Layout{}, exit.Internalf("cannot create %s: %s", dir, err)
		}
	}
	return l, nil
}

// renameRecords moves a pre-cl-116 records.db (and its WAL/SHM siblings) onto the
// creator.sqlite spelling. It refuses when both databases exist — two lifecycle
// authorities in one root is a state no rename may silently pick a winner for — and it
// refuses under a LIVE daemon: an old-build daemon still holds records.db open, and
// renaming its WAL out from under it would split committed pages from their database.
// The daemon liveness lock is held across the renames so no daemon can start mid-move.
func renameRecords(root, daemonLock, db string) *exit.Error {
	prior := filepath.Join(root, "records.db")
	if _, err := os.Lstat(prior); err != nil {
		return nil
	}
	if _, err := os.Lstat(db); err == nil {
		// BOTH SPELLINGS EXIST, AND THAT IS USUALLY NOT AN AMBIGUITY (cl-134). An old
		// build run once inside an already-migrated root recreates records.db, runs its
		// own migrations into it, and writes no row. Every current binary then refused
		// the whole root, and the refusal named a decision — "remove the one that is not
		// the lifecycle authority" — that it gave the reader no evidence to make. The
		// answer needed a sqlite shell to see. Observed twice on 2026-09-04.
		//
		// A database carrying no rows in any table carries no lifecycle, and that is
		// decidable rather than a judgement: a real pre-cl-116 authority has installs,
		// rentals or runs in it. So the empty one is sidelined and the root opens.
		holds, why := priorStoreHoldsRecords(prior)
		if !holds && why == "" {
			stamp := time.Now().UTC().Format("20060102T150405Z")
			for _, suffix := range []string{"", "-wal", "-shm"} {
				from := prior + suffix
				if _, err := os.Lstat(from); err != nil {
					continue
				}
				if err := os.Rename(from, prior+".superseded-"+stamp+suffix); err != nil {
					return exit.Internalf("cannot set aside %s: %s", from, err)
				}
			}
			fmt.Fprintf(os.Stderr,
				"cozy: %s held an empty records.db beside creator.sqlite — an older build "+
					"recreated it and wrote nothing. Set aside as records.db.superseded-%s; "+
					"creator.sqlite is the lifecycle authority and is untouched.\n", root, stamp)
			return nil
		}
		detail := "it carries lifecycle rows"
		if why != "" {
			detail = why
		}
		return exit.Named(exit.Conflict, "records_ambiguous",
			"%s holds both records.db and creator.sqlite, and the records.db is not "+
				"obviously stale (%s); this build only ever writes creator.sqlite, so move "+
				"records.db aside if it predates the migration", root, detail).
			WithRemedy("inspect both, then move the one you do not want aside: `mv %s %s.aside`",
				prior, prior)
	}
	f, err := os.OpenFile(daemonLock, os.O_RDWR|os.O_CREATE, 0o600)
	if err != nil {
		return exit.Internalf("cannot take the daemon lock to migrate %s: %s", root, err)
	}
	defer f.Close()
	if err := flock.Exclusive(f); err != nil {
		return exit.Named(exit.Conflict, "records_migration_blocked",
			"a running Cozy daemon still owns %s under its old records.db name", root).
			WithRemedy("stop it with `cozy down`, then retry; the new daemon migrates the database at start").
			WithNext("cozy down")
	}
	defer flock.Release(f)
	for _, suffix := range []string{"", "-wal", "-shm"} {
		from, to := prior+suffix, db+suffix
		if _, err := os.Lstat(from); err != nil {
			continue
		}
		if err := os.Rename(from, to); err != nil {
			return exit.Internalf("cannot move %s to %s: %s", from, to, err)
		}
	}
	return nil
}

// priorStoreHoldsRecords answers whether a records.db beside creator.sqlite is a real
// pre-cl-116 lifecycle authority or a scratch file a stale binary created after the
// migration already ran. It returns (holds, why): `why` is non-empty when the question
// could not be ANSWERED, which is treated exactly like "holds" — this decides whether to
// move a database, so it is conservative in one direction only.
//
// The file is opened read-only and immutable, so a live old-build daemon still holding it
// is never disturbed. `immutable=1` skips locking and therefore IGNORES the WAL, so a
// non-empty WAL is checked first and answered as "holds": rows committed only there would
// otherwise read as an empty database, which is the one wrong answer that loses data.
func priorStoreHoldsRecords(path string) (holds bool, why string) {
	if info, err := os.Lstat(path + "-wal"); err == nil && info.Size() > 0 {
		return true, "it has an unmerged write-ahead log"
	}
	db, err := sql.Open("sqlite", "file:"+path+"?mode=ro&immutable=1")
	if err != nil {
		return true, "it could not be opened to answer"
	}
	defer db.Close()
	rows, err := db.Query(
		"SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")
	if err != nil {
		return true, "its table list could not be read"
	}
	var tables []string
	for rows.Next() {
		var name string
		if err := rows.Scan(&name); err != nil {
			rows.Close()
			return true, "its table list could not be read"
		}
		tables = append(tables, name)
	}
	if err := rows.Err(); err != nil {
		rows.Close()
		return true, "its table list could not be read"
	}
	rows.Close()
	for _, table := range tables {
		var any int
		q := fmt.Sprintf(`SELECT EXISTS(SELECT 1 FROM "%s")`, strings.ReplaceAll(table, `"`, `""`))
		if err := db.QueryRow(q).Scan(&any); err != nil {
			return true, "table " + table + " could not be counted"
		}
		if any == 1 {
			return true, "table " + table + " carries rows"
		}
	}
	return false, ""
}

// InputAsset resolves one verified sha256 digest into its private immutable staging
// path. Callers validate the digest before reaching this method; keeping the spelling
// here prevents each transport from inventing a layout.
func (l Layout) InputAsset(digest string) string {
	return filepath.Join(l.Inputs, strings.TrimPrefix(digest, "sha256:"))
}

// RentalMediaToken is one rental's provisioned media bearer, mode 0600. It is deliberately
// NOT a row: a credential in the records database would be readable by every reader of
// that database, and the whole point of the 0600 handoff is that it is not.
func (l Layout) RentalMediaToken(id string) string {
	return filepath.Join(l.Rentals, id+".media-token")
}

// RentalCreatorIdentity is one rental's Ed25519 private key. The worker receives only
// the public key and verifies the signed ClaimProof.
func (l Layout) RentalCreatorIdentity(id string) string {
	return filepath.Join(l.Rentals, id+".creator.pem")
}

// PendingRentalMediaToken is the Creator-minted media bearer before the hub has answered with a
// rental id. The caller's operation key may contain path separators, so only its digest
// becomes a filename. The operation row keeps the unhashed key needed on the wire.
func (l Layout) PendingRentalMediaToken(operationKey string) string {
	sum := sha256.Sum256([]byte(operationKey))
	return filepath.Join(l.Rentals, "pending-"+hex.EncodeToString(sum[:])+".media-token")
}

func (l Layout) PendingRentalCreatorIdentity(operationKey string) string {
	sum := sha256.Sum256([]byte(operationKey))
	return filepath.Join(l.Rentals, "pending-"+hex.EncodeToString(sum[:])+".creator.pem")
}

// RentalCert is the worker certificate this client PINS for one rental. A certificate is
// public — trusting exactly this PEM and no CA is what makes the pin a pin (#445).
func (l Layout) RentalCert(id string) string { return filepath.Join(l.Rentals, id+".pem") }

// AcquisitionLocks is the model-acquisition flock directory: one lock file per exact
// Manifest, electing the one live mover across processes. Lock files are ephemeral
// coordination, created on demand; the empty directory is pruned at boot.
func (l Layout) AcquisitionLocks() string { return filepath.Join(l.Tmp, "locks") }

// InstallDir is where one install's source tree and venv live. The directory is replaced
// wholesale, never mutated. "generations" was the wrong name for the FOLDER -- in a product
// that generates media, a top-level `generations/` reads as the output namespace, which is
// `outputs/`.
func (l Layout) InstallDir(id string) string { return filepath.Join(l.Installs, id) }

// WorkerDir is one worker session's own root: its journal, its log, and the binding
// plan records the runtime resolves out of its COZY_HOME.
func (l Layout) WorkerDir(session string) string { return filepath.Join(l.Workers, session) }

// ScratchRepo is the job SCRATCH repo one request publishes into:
// `<org>/_job-<request-id>`. The underscore is deliberately outside the public repo
// grammar, so no user ref can ever squat a job's scratch, and promotion OUT of it is an
// explicit later act (jobs.md; cr-009's `LocalCoordinator.scratch_repo`).
func ScratchRepo(org, requestID string) string {
	return org + "/_job-" + requestID
}

// PublicationRoot is the one directory a job of this request may be granted to write
// into. Every OutputDestination the orchestrator mints resolves under it, and the fence
// that proves so is `orchestrator.publicationDest`.
func (l Layout) PublicationRoot(org, requestID string) string {
	return filepath.Join(l.Publications, org, "_job-"+requestID)
}

// PublicationStage is where ONE attempt of a job is granted to write. It is under the
// publication root but is not the addressable publication: `.staging` is outside the
// output-id namespace (an output id is a declared result FIELD PATH and cannot begin with
// a dot), so a committed bundle and an attempt in flight never share a name.
func (l Layout) PublicationStage(org, requestID string, attempt uint64) string {
	return filepath.Join(l.PublicationRoot(org, requestID), ".staging", "a"+itoa(attempt))
}

// PackageOutputs is where one package's result files are kept: `outputs/<org>-<name>`.
// The org/name separator becomes a dash so a package is one directory, never a tree.
func (l Layout) PackageOutputs(pkg string) string {
	return filepath.Join(l.Outputs, strings.ReplaceAll(pkg, "/", "-"))
}

func itoa(n uint64) string {
	if n == 0 {
		return "0"
	}
	var buf [20]byte
	i := len(buf)
	for n > 0 {
		i--
		buf[i] = byte('0' + n%10)
		n /= 10
	}
	return string(buf[i:])
}

// VenvPython is the interpreter inside one venv — `bin/python` where venvs have a bin,
// `Scripts\python.exe` on Windows. The venv's executable layout is a PLATFORM fact and
// this is its one spelling (#449); a path built by hand with "bin" is the bug class this
// replaces.
func VenvPython(venvDir string) string { return VenvTool(venvDir, "python") }

// VenvTool is a console script inside one venv (`cozy-runtime`, `uv`-installed
// entrypoints), spelled for the platform the way VenvPython is.
func VenvTool(venvDir, name string) string {
	if runtime.GOOS == "windows" {
		return filepath.Join(venvDir, "Scripts", name+".exe")
	}
	return filepath.Join(venvDir, "bin", name)
}
