// Package home is the ONE place that knows the local layout. It derives paths from a
// root it is GIVEN — the environment is read once, in internal/config, and COZY_HOME
// carries a value, never a decision (cozy-runtime-cli.md): it moves the root, nothing else.
package home

import (
	"os"
	"path/filepath"
	"runtime"

	"github.com/cozy-creator/cozy-creator-v2/internal/exit"
)

// Layout is the resolved set of paths every cl-009 verb works against.
type Layout struct {
	Root        string
	DB          string // the one local SQLite lifecycle database
	Generations string // one immutable directory per install generation
	Lock        string // the single-writer flock file
	Service     string // the LocalService's liveness lock (cl-001; held, never read)
	Workers     string // per-worker roots: journal, logs, staged binding plans
	Outputs     string // the local output namespace the orchestrator grants into
	// CAS is the shared local tensorfs store: the one place canonical bytes live on
	// this host. cozy-creator names it and never writes into it — every byte crosses
	// through the tfs binary (cl-012).
	CAS string
	// Transfer is scratch for bytes in flight. It is not a journal: the CAS's own
	// verification records are (a transfer resumes from what is VERIFIED, never from
	// what happens to be lying in a temp directory).
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
	// argv (cl-011's rule), so the handoff is an OS-protected file the LocalService
	// writes and its own CLI reads.
	Client string
	// Rentals holds one rented pod's SECRET MATERIAL: its provisioned owner token (0600)
	// and the certificate this client pins when it dials (cl-015). The rental's facts —
	// address, pod id, state — are rows in the one records authority; only what must not
	// be readable by another user on this host lives out here as files.
	Rentals string
}

func Open(root string) (Layout, *exit.Error) {
	if root == "" {
		return Layout{}, exit.Internalf("the local root is unset: internal/config.Load did not run")
	}
	l := Layout{
		Root:        root,
		DB:          filepath.Join(root, "records.db"),
		Generations: filepath.Join(root, "generations"),
		Lock:        filepath.Join(root, "writer.lock"),
		Service:     filepath.Join(root, "service.lock"),
		Workers:     filepath.Join(root, "workers"),
		Outputs:     filepath.Join(root, "outputs"),
		CAS:         filepath.Join(root, "cas"),
		Transfer:    filepath.Join(root, "transfer"),
		Triage:      filepath.Join(root, "triage"),
		Client:      filepath.Join(root, "client.cred"),
	}
	l.Publications = filepath.Join(root, "publications")
	l.Rentals = filepath.Join(root, "rentals")
	for _, dir := range []string{l.Generations, l.Workers, l.Outputs, l.Triage, l.Publications} {
		if err := os.MkdirAll(dir, 0o755); err != nil {
			return Layout{}, exit.Internalf("cannot create %s: %s", dir, err)
		}
	}
	return l, nil
}

// RentalToken is one rental's provisioned owner token, mode 0600. It is deliberately
// NOT a row: a credential in the records database would be readable by every reader of
// that database, and the whole point of the 0600 handoff is that it is not.
func (l Layout) RentalToken(id string) string { return filepath.Join(l.Rentals, id+".token") }

// RentalCert is the worker certificate this client PINS for one rental. A certificate is
// public — trusting exactly this PEM and no CA is what makes the pin a pin (#445).
func (l Layout) RentalCert(id string) string { return filepath.Join(l.Rentals, id+".pem") }

// GenerationDir is where one generation's source tree and venv live.
func (l Layout) GenerationDir(id string) string { return filepath.Join(l.Generations, id) }

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

// AttemptDir is the local output namespace for one attempt. The orchestrator grants
// exactly this directory and nothing above it; the runtime writes under the grant and
// never learns who may read it.
func (l Layout) AttemptDir(requestID string, attempt uint64) string {
	return filepath.Join(l.Outputs, requestID, "a"+itoa(attempt))
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
