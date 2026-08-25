// Package home is the ONE place that knows the local layout. COZY_HOME carries a
// value, never a decision (cozy-runtime-cli.md): it moves the root, nothing else.
package home

import (
	"os"
	"path/filepath"

	"github.com/cozy-creator/cozy-creator-v2/internal/exit"
)

// root is $COZY_HOME, or ~/.cozy.
func root() (string, *exit.Error) {
	if v := os.Getenv("COZY_HOME"); v != "" {
		abs, err := filepath.Abs(v)
		if err != nil {
			return "", exit.Internalf("COZY_HOME %q is not resolvable: %s", v, err)
		}
		return abs, nil
	}
	h, err := os.UserHomeDir()
	if err != nil {
		return "", exit.Internalf("no home directory and COZY_HOME is unset: %s", err)
	}
	return filepath.Join(h, ".cozy"), nil
}

// Layout is the resolved set of paths every cl-009 verb works against.
type Layout struct {
	Root        string
	DB          string // the one local libSQL lifecycle database
	Generations string // one immutable directory per install generation
	Lock        string // the single-writer flock file
}

func Open() (Layout, *exit.Error) {
	r, e := root()
	if e != nil {
		return Layout{}, e
	}
	l := Layout{
		Root:        r,
		DB:          filepath.Join(r, "records.db"),
		Generations: filepath.Join(r, "generations"),
		Lock:        filepath.Join(r, "writer.lock"),
	}
	if err := os.MkdirAll(l.Generations, 0o755); err != nil {
		return Layout{}, exit.Internalf("cannot create %s: %s", l.Generations, err)
	}
	return l, nil
}

// GenerationDir is where one generation's source tree and venv live.
func (l Layout) GenerationDir(id string) string { return filepath.Join(l.Generations, id) }
