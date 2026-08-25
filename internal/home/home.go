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
	}
	for _, dir := range []string{l.Generations, l.Workers, l.Outputs} {
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
