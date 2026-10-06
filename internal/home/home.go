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
	// Daemon is the Calcifer's lifetime ownership/address record (cl-001). The live
	// owner holds its kernel lock and publishes the client address, worker socket and
	// the per-launch CLI token under it, mode 0600. There is no separate client.cred.
	Daemon  string
	Workers string // per-worker roots: journal, logs, staged binding plans (live only)
	// Outputs is the user-facing store: `<org>-<package>/<digest>.<ext>`, one file per
	// result file, named by its own content digest so a regenerated file lands on itself.
	// User-owned result files are never automatically removed.
	Outputs string
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
	// Machine is this computer as a machine (proto-062): `root/` is laid out exactly as a
	// pod's `/` for the same pod-supervisor Host, beside the controller's own grant for it
	// (registration, owner key, launch record).
	Machine string
	// Log is the Calcifer's own log, bounded by rotation on an observed size
	// (internal/calcifer.OpenLog). Its one rotated predecessor is Log + ".1".
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
	l := Paths(root)
	// Only the two roots every verb touches exist up front. Everything else is created
	// by its writer at the moment work exists, and reclaimed when the work dies.
	for _, dir := range []string{l.Installs, l.Outputs} {
		if err := os.MkdirAll(dir, 0o755); err != nil {
			return Layout{}, exit.Internalf("cannot create %s: %s", dir, err)
		}
	}
	return l, nil
}

// Paths derives the layout from root without touching the filesystem. Only readers that
// must not create anything (shell completion) use it directly; everything else goes
// through Open.
func Paths(root string) Layout {
	l := Layout{
		Root:     root,
		DB:       filepath.Join(root, "creator.sqlite"),
		Installs: filepath.Join(root, "installs"),
		Daemon:   filepath.Join(root, "daemon.lock"),
		Workers:  filepath.Join(root, "workers"),
		Outputs:  filepath.Join(root, "outputs"),
		Tmp:      filepath.Join(root, "tmp"),
	}
	l.Lock = filepath.Join(l.Installs, ".lock")
	l.Publications = filepath.Join(root, "publications")
	l.Rentals = filepath.Join(root, "rentals")
	l.LocalPackages = filepath.Join(root, "local-packages")
	l.Machine = filepath.Join(root, "machine")
	l.Log = filepath.Join(root, "daemon.log")
	return l
}

// DependencyCache is disposable immutable payload storage. Captured generations
// retain independent hardlinks, so deleting cache entries never invalidates them.
func (l Layout) DependencyCache() string {
	return filepath.Join(l.LocalPackages, "dependency-objects")
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
