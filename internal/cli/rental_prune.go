package cli

import (
	"strings"

	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/output"
)

func handleRentalPrune(ctx *Context) *exit.Error {
	_, store, problem := rentalStores(ctx)
	if problem != nil {
		return problem
	}
	defer store.Close()
	row, problem := store.RentalByMachine(strings.TrimSpace(ctx.Inv.Args[0]))
	if problem != nil {
		return problem
	}
	if row == nil {
		return exit.New(exit.NotFound, "no rental %q on this host", ctx.Inv.Args[0])
	}
	client, problem := dial(ctx)
	if problem != nil {
		return problem
	}
	result, problem := client.PruneRental(row.ID)
	if problem != nil {
		return problem
	}
	return emit(ctx, compactRecord([]output.Field{{K: "rental", V: result.Rental}, {K: "removed_entries", V: result.RemovedEntries}, {K: "reclaimed_bytes", V: result.ReclaimedBytes}, {K: "store_busy", V: result.StoreBusy}}, "rental", "removed_entries", "reclaimed_bytes", "store_busy"))
}
