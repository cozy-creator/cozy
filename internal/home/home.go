// Package home is the ONE place that knows the local layout. It derives paths from a
// root it is GIVEN — the environment is read once, in internal/config, and COZY_HOME
// carries a value, never a decision (cozy-runtime-cli.md): it moves the root, nothing else.
package home

import (
	"crypto/sha256"
	"encoding/hex"
	"os"
	"path/filepath"
	"runtime"
	"strings"

	"github.com/cozy-creator/cozy/internal/exit"
)

// Layout is the resolved set of paths every cl-009 verb works against.
type Layout struct {
	Root     string
	DB       string // the one local SQLite lifecycle database
	Installs string // one immutable directory per package install
	Lock     string // the single-writer flock file
	Daemon   string // the Cozy daemon's liveness lock (cl-001; held, never read)
	Workers  string // per-worker roots: journal, logs, staged binding plans
	// Attempts is the orchestrator's INTERNAL working area: one directory per attempt
	// holding the staged payload and the destinations the delivery grant names. It is
	// reclaimed once the attempt is closed and its result files have moved out.
	Attempts string
	// Outputs is the user-facing store: `<org>-<package>/<digest>.<ext>`, one file per
	// result file, named by its own content digest so a regenerated file lands on itself.
	Outputs string
	// Inputs is the immutable, content-addressed staging area for caller-owned assets.
	// Request rows point here so requeue never depends on the submitting CLI or its
	// original path still existing.
	Inputs string
	// Uploads is the private content-addressed store for bytes received through the
	// localhost upload API. Browser/API callers name opaque ids, never paths.
	Uploads string
	// CAS is the shared local tensorfs store: the one place canonical bytes live on
	// this host. cozy-creator names it and never writes into it — every byte crosses
	// through the tfs binary (cl-012).
	CAS string
	// Transfer is scratch for bytes in flight. It is not a journal: the CAS's own
	// verification records are (a transfer resumes from what is VERIFIED, never from
	// what happens to be lying in a temp directory). Every entry is claimed through
	// internal/scratch by the live process using it; the daemon's start-time sweep
	// removes what nobody holds. Only `locks/` is permanent.
	Transfer string
	// Triage holds WorkerTriageBundles copied out of worker roots (cl-006). A worker
	// root does not outlive its worker — a one-shot run deletes it — so the bundle
	// worth keeping is kept HERE, by the client that wanted it.
	Triage string
	// Publications is the DURABLE PUBLICATION PLANE (cl-004): one directory per job
	// scratch repo, and the only place a job is ever granted to write. It is
	// deliberately NOT under Outputs or Workers — a bounded job is reclaimed at its
	// terminal, and a bundle that lived inside what reclaim removes would be destroyed
	// by the very act that ends the job that produced it.
	Publications string
	// Client is the CLI's local credential file, mode 0600. A credential never rides
	// argv (cl-011's rule), so the handoff is an OS-protected file the Cozy daemon
	// writes and its own CLI reads.
	Client string
	// Rentals holds one rented pod's SECRET MATERIAL: its media bearer (0600)
	// and the certificate this client pins when it dials (cl-015). The rental's facts —
	// address, pod id, state — are rows in the one records authority; only what must not
	// be readable by another user on this host lives out here as files.
	Rentals string
	// LocalPackages holds exact ephemeral wheel revisions for rented local-package commands.
	// It is staging the daemon alone writes, never a catalog or mutable checkout.
	LocalPackages string
	// Log is the Cozy daemon's own log: the orchestrator's words and the process's
	// banner, bounded by rotation on an observed size (internal/daemon.OpenLog). Its one
	// rotated predecessor is Log + ".1". Read with `cozy daemon log`.
	Log string
}

func Open(root string) (Layout, *exit.Error) {
	if root == "" {
		return Layout{}, exit.Internalf("the local root is unset: internal/config.Load did not run")
	}
	// The records database retains request payloads and path-free creative prompts; the
	// client credential and rental secrets already make this an OS-private user root.
	// Protecting the directory also protects SQLite's lazily-created WAL/SHM siblings.
	if err := os.MkdirAll(root, 0o700); err != nil {
		return Layout{}, exit.Internalf("cannot create the private local root %s: %s", root, err)
	}
	if err := os.Chmod(root, 0o700); err != nil {
		return Layout{}, exit.Internalf("cannot protect the private local root %s: %s", root, err)
	}
	l := Layout{
		Root:     root,
		DB:       filepath.Join(root, "records.db"),
		Installs: filepath.Join(root, "installs"),
		Lock:     filepath.Join(root, "writer.lock"),
		Daemon:   filepath.Join(root, "daemon.lock"),
		Workers:  filepath.Join(root, "workers"),
		Attempts: filepath.Join(root, "attempts"),
		Outputs:  filepath.Join(root, "outputs"),
		Inputs:   filepath.Join(root, "inputs"),
		Uploads:  filepath.Join(root, "uploads", "sha256"),
		CAS:      filepath.Join(root, "cas"),
		Transfer: filepath.Join(root, "transfer"),
		Triage:   filepath.Join(root, "triage"),
		Client:   filepath.Join(root, "client.cred"),
	}
	l.Publications = filepath.Join(root, "publications")
	l.Rentals = filepath.Join(root, "rentals")
	l.LocalPackages = filepath.Join(root, "local-packages")
	l.Log = filepath.Join(root, "daemon.log")
	// The revision store was `private-packages/` before the vocabulary hard-cut (proto-027).
	// A root written by that build keeps what it holds: the directory moves, once, and the
	// local-package sweep retires any revision written under the old document format.
	if prior := filepath.Join(root, "private-packages"); dirExists(prior) && !dirExists(l.LocalPackages) {
		if err := os.Rename(prior, l.LocalPackages); err != nil {
			return Layout{}, exit.Internalf("cannot move %s to %s: %s", prior, l.LocalPackages, err)
		}
	}
	for _, dir := range []string{l.Installs, l.Workers, l.Attempts, l.Outputs, l.Triage, l.Publications} {
		if err := os.MkdirAll(dir, 0o755); err != nil {
			return Layout{}, exit.Internalf("cannot create %s: %s", dir, err)
		}
	}
	for _, dir := range []string{l.Inputs, l.Uploads, l.LocalPackages} {
		if err := os.MkdirAll(dir, 0o700); err != nil {
			return Layout{}, exit.Internalf("cannot create %s: %s", dir, err)
		}
	}
	return l, nil
}

func dirExists(path string) bool {
	info, err := os.Lstat(path)
	return err == nil && info.IsDir()
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

// InstallDir is where one install's source tree and venv live. The directory is replaced
// wholesale, never mutated. "generations" was the wrong name for the FOLDER -- in a product
// that generates media, a top-level `generations/` reads as the output namespace, which is
// `outputs/`.
func (l Layout) InstallDir(id string) string { return filepath.Join(l.Installs, id) }

// WorkerDir is one worker session's own root: its journal, its log, and the binding
// plan records the runtime resolves out of its COZY_HOME.
func (l Layout) WorkerDir(session string) string { return filepath.Join(l.Workers, session) }

// TriageFile is where one kept bundle lives, named by its OPAQUE subject and by
// nothing a client can shape.
func (l Layout) TriageFile(subject string) string {
	return filepath.Join(l.Triage, subject+".json")
}

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

// AttemptDir is one attempt's working directory. The orchestrator grants exactly this
// directory and nothing above it; the runtime writes under the grant and never learns
// who may read it. Nothing durable lives here: results move to PackageOutputs (or the
// caller's --out) and the directory is reclaimed.
func (l Layout) AttemptDir(requestID string, attempt uint64) string {
	return filepath.Join(l.RequestAttempts(requestID), itoa(attempt))
}

// RequestAttempts is the parent of one request's attempt directories.
func (l Layout) RequestAttempts(requestID string) string {
	return filepath.Join(l.Attempts, requestID)
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
