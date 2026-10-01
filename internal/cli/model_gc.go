package cli

import (
	"fmt"
	"strings"

	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/home"
	"github.com/cozy-creator/cozy/internal/output"
	"github.com/cozy-creator/cozy/internal/tfs"
)

// `cozy model gc`: reclaim what no local model names. `model remove` already does this
// as part of removing; this is the verb for a store that carries unreferenced bytes for
// any other reason — a crashed download, an abandoned ingest — and the pass the daemon
// runs on maintenance.gc_cron. The decision is TensorFS's, from its filesystem census;
// only a local request still moving bytes in defers it (reclamationFence).
func handleModelGC(ctx *Context) *exit.Error {
	if name := strings.TrimSpace(ctx.Inv.Value("--rental")); name != "" {
		return rentalModelGC(ctx, name)
	}
	layout, problem := home.Open(ctx.Cfg.Home)
	if problem != nil {
		return problem
	}
	fence, problem := readReclamationFence(layout)
	if problem != nil {
		return problem
	}
	if problem := fence.busy(); problem != nil {
		return problem
	}
	tool, _, problem := localTensorFS(ctx)
	if problem != nil {
		return problem
	}
	report, notes, problem := fence.collect(tool)
	if problem != nil {
		return problem
	}
	return emit(ctx, output.Record{Fields: reclaimFields(report), Notes: notes})
}

// rentalModelGC reclaims a rental's store through its machine: the Runtime drops cached
// operation results nothing uses, then collects what no model or live work references,
// deferring while bytes are still moving in (store_busy).
func rentalModelGC(ctx *Context, name string) *exit.Error {
	_, store, problem := rentalStores(ctx)
	if problem != nil {
		return problem
	}
	row, problem := store.RentalByMachine(name)
	store.Close()
	if problem != nil {
		return problem
	}
	if row == nil {
		return exit.New(exit.NotFound, "no rental %q on this host", name)
	}
	client, problem := dial(ctx)
	if problem != nil {
		return problem
	}
	result, problem := client.PruneRental(row.ID)
	if problem != nil {
		return problem
	}
	record := compactRecord([]output.Field{{K: "rental", V: row.MachineName},
		{K: "reclaimed", V: reclaimed(tfs.GCReport{ReclaimedBytes: int64(result.ReclaimedBytes)})},
		{K: "cached_results_removed", V: result.RemovedEntries}, {K: "store_busy", V: result.StoreBusy}},
		"rental", "reclaimed", "cached_results_removed", "store_busy")
	if result.StoreBusy {
		record.Notes = []string{"collection deferred: the machine is still moving bytes into its store; run it again once that settles"}
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
