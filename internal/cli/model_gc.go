package cli

import (
	"fmt"
	"strings"

	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/home"
	"github.com/cozy-creator/cozy/internal/output"
	"github.com/cozy-creator/cozy/internal/records"
	"github.com/cozy-creator/cozy/internal/tfs"
)

// `cozy model gc`: reclaim what no local model names. `model remove` already does this
// as part of removing; this is the verb for a store that carries unreferenced bytes for
// any other reason — a crashed download, an abandoned ingest — and the pass the daemon
// runs on maintenance.gc_cron. The decision is TensorFS's, from its filesystem census.
func handleModelGC(ctx *Context) *exit.Error {
	layout, problem := home.Open(ctx.Cfg.Home)
	if problem != nil {
		return problem
	}
	store, problem := records.Open(layout.DB)
	if problem != nil {
		return problem
	}
	active, problem := store.ActiveRequests()
	store.Close()
	if problem != nil {
		return problem
	}
	tool, _, problem := localTensorFS(ctx)
	if problem != nil {
		return problem
	}
	report, problem := tool.GC(len(active) == 0)
	if problem != nil {
		return problem
	}
	record := output.Record{Fields: reclaimFields(report)}
	if report.KeptObjects > 0 && len(active) > 0 {
		record.Notes = []string{fmt.Sprintf("%d active request(s): ingest sessions were not abandoned", len(active))}
	}
	return emit(ctx, record)
}

// reclaimed is the `reclaimed` field: a program gets the pass, a person gets the bytes.
type reclaimed tfs.GCReport

func (r reclaimed) Human() string { return output.Bytes(r.ReclaimedBytes) }

// kept is what an open ingest session still names, said only when there is some.
type kept tfs.GCReport

func (k kept) Human() string {
	if k.KeptObjects == 0 {
		return ""
	}
	return fmt.Sprintf("%s held by ingest session %s", output.Bytes(k.KeptBytes),
		strings.Join(k.Sessions, ", "))
}

func reclaimFields(report tfs.GCReport) []output.Field {
	fields := []output.Field{{K: "reclaimed", V: reclaimed(report)}}
	if report.KeptObjects > 0 {
		fields = append(fields, output.Field{K: "kept", V: kept(report)})
	}
	return fields
}
