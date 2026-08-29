package cli

import (
	"encoding/json"
	"fmt"
	"io"
	"strings"
	"time"

	localapi "github.com/cozy-creator/cozy/internal/client"
	"github.com/cozy-creator/cozy/internal/daemon"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/home"
	"github.com/cozy-creator/cozy/internal/hostgpu"
	"github.com/cozy-creator/cozy/internal/hub"
	"github.com/cozy-creator/cozy/internal/orchestrator"
	"github.com/cozy-creator/cozy/internal/output"
	"github.com/cozy-creator/cozy/internal/records"
	"github.com/cozy-creator/cozy/internal/rental"
)

func handleUnload(ctx *Context) *exit.Error {
	state := daemon.Probe(ctx.Cfg)
	if !state.Up {
		return emit(ctx, output.Record{Fields: []output.Field{
			{K: "stopped", V: 0}, {K: "released", V: "0B"}, {K: "changed", V: false},
		}})
	}
	ctx.Daemon = state
	client, problem := dial(ctx)
	if problem != nil {
		return problem
	}
	result, problem := client.Unload()
	if problem != nil {
		return problem
	}
	list := output.List{
		Name: "workers", Fields: []string{"package", "devices"},
		AllFields: []string{"instance", "package", "release", "devices"},
	}
	for _, worker := range result.Stopped {
		list.Rows = append(list.Rows, map[string]string{
			"instance": worker.InstanceID, "package": worker.Package,
			"release": worker.PackageReleaseID, "devices": strings.Join(worker.Devices, ","),
		})
	}
	list.Aggregates = []output.Field{{K: "changed", V: result.Count > 0}}
	return emit(ctx, list)
}

func handleUp(ctx *Context) *exit.Error {
	state, changed, problem := ensureDaemon(ctx)
	if problem != nil {
		return problem
	}
	inventory := hostgpu.Probe(ctx.Cfg)
	summaries := make([]string, 0, len(inventory.GPUs))
	for _, gpu := range inventory.GPUs {
		cuda := "driver CUDA unknown"
		if gpu.DriverCUDAVersion != "" {
			cuda = "driver CUDA " + gpu.DriverCUDAVersion
		}
		summaries = append(summaries, fmt.Sprintf(
			"GPU %d · %s · %.1f / %.1f GiB free · driver %s · %s · %s",
			gpu.Index, gpu.Model, float64(gpu.VRAMFreeBytes)/(1<<30),
			float64(gpu.VRAMTotalBytes)/(1<<30), gpu.DriverVersion, cuda, gpu.SM))
	}
	fields := []output.Field{
		{K: "daemon", V: "running"}, {K: "url", V: "http://" + state.Addr + "/"},
		{K: "api", V: "http://" + state.Addr}, {K: "pid", V: state.PID},
		{K: "since", V: state.Since}, {K: "gpus", V: summaries},
		{K: "gpu_count", V: len(inventory.GPUs)}, {K: "gpu_details", V: inventory.GPUs},
		{K: "changed", V: changed},
	}
	record := compactRecord(fields, "url", "gpus", "changed")
	if inventory.Diagnostic != "" {
		record.Notes = append(record.Notes, inventory.Diagnostic)
	} else if len(inventory.GPUs) == 0 {
		record.Notes = append(record.Notes, "no NVIDIA GPUs detected")
	}
	return emit(ctx, record)
}

func handleDown(ctx *Context) *exit.Error {
	all := ctx.Inv.Bool("--all")
	state := daemon.Probe(ctx.Cfg)
	if !state.Up {
		blockers, problem := offlineDownBlockers(ctx)
		if problem != nil {
			return problem
		}
		if len(blockers) == 0 {
			return emit(ctx, output.Record{Fields: []output.Field{
				{K: "daemon", V: "stopped"}, {K: "changed", V: false},
			}})
		}
		if !all {
			return exit.Named(exit.Conflict, "active_work",
				"daemon shutdown refused: active %s", strings.Join(blockers, ", ")).
				WithRemedy("cancel/end the named work, or use explicit `cozy down --all`")
		}
		state, _, problem = ensureDaemon(ctx)
		if problem != nil {
			return problem
		}
	}
	ctx.Daemon = state
	client, problem := dial(ctx)
	if problem != nil {
		return problem
	}
	if !all {
		result, problem := client.Down(false)
		if problem != nil {
			return problem
		}
		if !result.ShuttingDown {
			return exit.Internalf("safe down returned without a shutdown decision")
		}
		return finishDaemonDown(ctx, nil)
	}
	return downAll(ctx, client)
}

func downAll(ctx *Context, client *localapi.Client) *exit.Error {
	canceled := map[string]bool{}
	ended := map[string]bool{}
	for {
		result, problem := client.Down(true)
		if problem != nil {
			return problem.WithRemedy("partial teardown stopped; the daemon remains alive for reconciliation")
		}
		for _, identity := range result.CancellationRequested {
			canceled[identity.ID] = true
		}
		for _, identity := range result.Rentals {
			switch identity.Kind {
			case "rental":
				if problem := endRentalSilently(ctx, identity.ID); problem != nil {
					return problem.WithRemedy("rental termination was not confirmed; the daemon remains alive")
				}
				ended[identity.ID] = true
			case "rental_operation":
				id, problem := resolveRentalOperation(ctx, identity.ID)
				if problem != nil {
					return problem.WithRemedy("the paid operation remains recorded and the daemon remains alive")
				}
				if id != "" {
					if problem := endRentalSilently(ctx, id); problem != nil {
						return problem.WithRemedy("rental termination was not confirmed; the daemon remains alive")
					}
					ended[id] = true
				}
			}
		}
		if result.ShuttingDown {
			return finishDaemonDown(ctx, []output.Field{
				{K: "canceled", V: len(canceled)}, {K: "rentals_ended", V: len(ended)},
			})
		}
		// Local cancellations settle through their durable worker terminals. Rental
		// work above is synchronous through confirmed provider absence.
		time.Sleep(100 * time.Millisecond)
	}
}

func endRentalSilently(ctx *Context, id string) *exit.Error {
	row, problem := rentalRow(ctx, id)
	if problem != nil {
		return problem
	}
	sub := *ctx
	sub.Out = io.Discard
	sub.Inv = &Invocation{Args: []string{id}, Bools: map[string]bool{},
		Values: map[string][]string{}, Mode: ctx.Mode()}
	if row != nil {
		sub.Cfg.HubURL = row.Hub
		sub.Cfg.HubURLSource = "rental record"
	}
	return handleRentRelease(&sub)
}

func resolveRentalOperation(ctx *Context, key string) (string, *exit.Error) {
	layout, problem := home.Open(ctx.Cfg.Home)
	if problem != nil {
		return "", problem
	}
	store, problem := records.Open(layout.DB)
	if problem != nil {
		return "", problem
	}
	defer store.Close()
	operation, problem := store.RentalOperation(key)
	if problem != nil || operation == nil {
		return "", problem
	}
	if operation.RentalID != "" {
		return operation.RentalID, nil
	}
	var request hub.RentalRequest
	if err := json.Unmarshal(operation.RequestBody, &request); err != nil {
		return "", exit.Internalf("stored rental operation %s has an unreadable request: %s", key, err)
	}
	sub := *ctx
	sub.Cfg.HubURL = operation.Hub
	sub.Cfg.HubURLSource = "rental operation"
	hubClient := client(&sub)
	hctx, cancel := hub.LongContext()
	seen, problem := hubClient.Rent(hctx, operation.RequestBody, operation.Reason, operation.Key)
	cancel()
	if problem != nil {
		if problem.Code == exit.Credential || problem.Code == exit.Validation ||
			problem.Code == exit.NotFound || problem.Code == exit.Conflict {
			if advanced := store.AdvanceRentalOperation(key, "", "rejected"); advanced != nil {
				return "", advanced
			}
			rental.ForgetPending(layout, key)
			return "", nil
		}
		return "", problem
	}
	if problem := store.AdvanceRentalOperation(key, seen.ID, seen.State); problem != nil {
		return "", problem
	}
	if problem := store.RecordRental(records.Rental{
		ID: seen.ID, PackageRef: request.PackageRef, AcceleratorModel: seen.AcceleratorModel,
		State: seen.State, Hub: operation.Hub,
	}); problem != nil {
		return "", problem
	}
	return seen.ID, nil
}

func rentalRow(ctx *Context, id string) (*records.Rental, *exit.Error) {
	layout, problem := home.Open(ctx.Cfg.Home)
	if problem != nil {
		return nil, problem
	}
	store, problem := records.Open(layout.DB)
	if problem != nil {
		return nil, problem
	}
	defer store.Close()
	return store.RentalRow(id)
}

func offlineDownBlockers(ctx *Context) ([]string, *exit.Error) {
	layout, problem := home.Open(ctx.Cfg.Home)
	if problem != nil {
		return nil, problem
	}
	store, problem := records.Open(layout.DB)
	if problem != nil {
		return nil, problem
	}
	defer store.Close()
	var blockers []string
	requests, problem := store.ActiveRequests()
	if problem != nil {
		return nil, problem
	}
	for _, request := range requests {
		kind := "invocation"
		if request.IsJob() {
			kind = "job"
		}
		blockers = append(blockers, fmt.Sprintf("%s %s (%s)", kind, request.ID, request.State))
	}
	rentals, problem := store.Rentals()
	if problem != nil {
		return nil, problem
	}
	for _, rental := range rentals {
		blockers = append(blockers, fmt.Sprintf("rental %s (%s)", rental.ID, rental.State))
	}
	operations, problem := store.ActiveRentalOperations()
	if problem != nil {
		return nil, problem
	}
	for _, operation := range operations {
		if operation.RentalID == "" {
			blockers = append(blockers, fmt.Sprintf("rental operation %s (%s)", operation.Key, operation.State))
		}
	}
	return blockers, nil
}

func finishDaemonDown(ctx *Context, extra []output.Field) *exit.Error {
	deadline := time.Now().Add(2 * orchestrator.StopGrace)
	for time.Now().Before(deadline) {
		if !daemon.Probe(ctx.Cfg).Up {
			fields := []output.Field{{K: "daemon", V: "stopped"}, {K: "changed", V: true}}
			fields = append(fields, extra...)
			return emit(ctx, output.Record{Fields: fields})
		}
		time.Sleep(50 * time.Millisecond)
	}
	return exit.New(exit.Conflict, "the daemon accepted down but did not finish stopping").
		WithRemedy("the daemon remains responsible for its workers; no forced kill was performed")
}
