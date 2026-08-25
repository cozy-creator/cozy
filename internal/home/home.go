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
	for _, dir := range []string{l.Generations, l.Workers, l.Outputs, l.Triage} {
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
