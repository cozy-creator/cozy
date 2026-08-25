// Package home is the ONE place that knows the local layout. It derives paths from a
// root it is GIVEN — the environment is read once, in internal/config, and COZY_HOME
// carries a value, never a decision (cozy-runtime-cli.md): it moves the root, nothing else.
package home

import (
	"os"
	"path/filepath"

	"github.com/cozy-creator/cozy-creator-v2/internal/exit"
)

// Layout is the resolved set of paths every cl-009 verb works against.
type Layout struct {
	Root        string
	DB          string // the one local libSQL lifecycle database
	Generations string // one immutable directory per install generation
	Lock        string // the single-writer flock file
	Service     string // the LocalService's liveness lock (cl-001; held, never read)
	Workers     string // per-worker roots: journal, logs, staged binding plans
	Outputs     string // the local output namespace the coordinator grants into
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
	for _, dir := range []string{l.Generations, l.Workers, l.Outputs, l.Triage, l.Publications} {
		if err := os.MkdirAll(dir, 0o755); err != nil {
			return Layout{}, exit.Internalf("cannot create %s: %s", dir, err)
		}
	}
	return l, nil
}

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
// into. Every OutputDestination the coordinator mints resolves under it, and the fence
// that proves so is `coord.publicationDest`.
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

// AttemptDir is the local output namespace for one attempt. The coordinator grants
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
