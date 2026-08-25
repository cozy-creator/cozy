package install

import (
	"os"
	"path/filepath"
	"syscall"

	"github.com/cozy-creator/cozy-creator-v2/internal/exit"
	"github.com/cozy-creator/cozy-creator-v2/internal/home"
	"github.com/cozy-creator/cozy-creator-v2/internal/records"
)

// Writer is the single-writer flock every mutating verb takes. One writer per
// local records database: a second concurrent mutation is exit 13, never a race.
type Writer struct{ f *os.File }

func Lock(l home.Layout) (*Writer, *exit.Error) {
	f, err := os.OpenFile(l.Lock, os.O_CREATE|os.O_RDWR, 0o644)
	if err != nil {
		return nil, exit.Internalf("cannot open the writer lock %s: %s", l.Lock, err)
	}
	if err := syscall.Flock(int(f.Fd()), syscall.LOCK_EX|syscall.LOCK_NB); err != nil {
		f.Close()
		return nil, exit.New(exit.Conflict, "another cozy writer holds %s", l.Lock).
			WithRemedy("one writer per local records database; wait for it to finish").
			WithNext("cozy ls")
	}
	return &Writer{f: f}, nil
}

func (w *Writer) Unlock() {
	_ = syscall.Flock(int(w.f.Fd()), syscall.LOCK_UN)
	_ = w.f.Close()
}

// Reclaimable is one item in the gc plan.
type Reclaimable struct {
	Kind     string // "generation" — a row no pin references; "orphan" — a directory with no row
	ID       string
	Endpoint string
	Version  string
	Reason   string
	Bytes    int64
}

// Plan is what `cozy gc` prints without --yes: a read, exit 0. It reclaims only
// generations nothing references — a pinned generation is never in the plan.
func Plan(l home.Layout, st *records.Store) ([]Reclaimable, *exit.Error) {
	var out []Reclaimable
	unref, e := st.Unreferenced()
	if e != nil {
		return nil, e
	}
	for _, g := range unref {
		out = append(out, Reclaimable{
			Kind: "generation", ID: g.ID, Endpoint: g.Endpoint, Version: g.Version,
			Reason: "no pin references it", Bytes: g.BytesExcl,
		})
	}
	known, e := st.KnownIDs()
	if e != nil {
		return nil, e
	}
	entries, err := os.ReadDir(l.Generations)
	if err != nil {
		return out, nil
	}
	for _, d := range entries {
		if !d.IsDir() || known[d.Name()] {
			continue
		}
		dir := l.GenerationDir(d.Name())
		excl, shared := Disk(dir)
		out = append(out, Reclaimable{
			Kind: "orphan", ID: d.Name(), Endpoint: "-", Version: "-",
			Reason: "a pre-activation crash left it with no record", Bytes: excl + shared,
		})
	}
	return out, nil
}

// Collect executes the plan: directories go, then the rows. A pinned generation's
// foreign key refuses the row delete, so an active install can never be collected.
func Collect(l home.Layout, st *records.Store, plan []Reclaimable) (int64, *exit.Error) {
	var freed int64
	for _, r := range plan {
		if err := os.RemoveAll(filepath.Join(l.Generations, r.ID)); err != nil {
			return freed, exit.Internalf("cannot remove generation directory %s: %s", r.ID, err)
		}
		if r.Kind == "generation" {
			if e := st.Forget(r.ID); e != nil {
				return freed, e
			}
		}
		freed += r.Bytes
	}
	return freed, nil
}
