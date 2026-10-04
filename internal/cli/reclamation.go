package cli

import (
	"fmt"
	"strings"

	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/home"
	"github.com/cozy-creator/cozy/internal/records"
	"github.com/cozy-creator/cozy/internal/tfs"
)

// reclamationFence is what a reclamation pass waits for beyond TensorFS's own holds: local
// requests still moving unnamed bytes into the store, and retained stops whose retry may
// resume an ingest session with no live writer. Every row is a run `cozy run list` shows
// and `cozy run cancel` settles.
type reclamationFence struct {
	writers  []records.Request
	retained []records.Request
}

func readReclamationFence(layout home.Layout) (reclamationFence, *exit.Error) {
	store, problem := records.Open(layout.DB)
	if problem != nil {
		return reclamationFence{}, problem
	}
	defer store.Close()
	writers, problem := store.LocalStoreWriters()
	if problem != nil {
		return reclamationFence{}, problem
	}
	retained, problem := store.LocalRetainedStops()
	if problem != nil {
		return reclamationFence{}, problem
	}
	return reclamationFence{writers: writers, retained: retained}, nil
}

// busy refuses a pass beside a request that may be moving bytes the store has not named.
func (f reclamationFence) busy() *exit.Error {
	if len(f.writers) == 0 {
		return nil
	}
	first := f.writers[0]
	return exit.New(exit.Conflict, "%s may still be moving bytes into the store", runsPhrase(f.writers)).
		WithRemedy("let it settle, or cancel it, then reclaim").
		WithNext("cozy run cancel " + runReference(first.Number, first.ID))
}

// collect runs one pass: ingest sessions whose writer is gone are abandoned only when no
// retained run may resume them.
func (f reclamationFence) collect(tool *tfs.Tool) (tfs.GCReport, []string, *exit.Error) {
	report, problem := tool.GC(len(f.retained) == 0)
	if problem != nil || len(f.retained) == 0 || report.KeptObjects == 0 {
		return report, nil, problem
	}
	return report, []string{"ingest sessions kept for retained " + runsPhrase(f.retained) +
		"; cancel them to release their bytes"}, nil
}

func runsPhrase(rows []records.Request) string {
	const shown = 5
	refs := make([]string, 0, shown)
	for i, row := range rows {
		if i == shown {
			break
		}
		refs = append(refs, fmt.Sprintf("%s (%s)", runReference(row.Number, row.ID),
			records.PublicRunStatus(row.State, false)))
	}
	phrase := "run " + strings.Join(refs, ", ")
	if len(rows) > 1 {
		phrase = "runs " + strings.Join(refs, ", ")
	}
	if len(rows) > shown {
		phrase += fmt.Sprintf(" and %d more", len(rows)-shown)
	}
	return phrase
}
