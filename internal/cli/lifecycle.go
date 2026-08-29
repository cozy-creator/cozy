package cli

import (
	"encoding/json"
	"fmt"
	"io"
	"strings"
	"time"

	localapi "github.com/cozy-creator/cozy-creator/internal/client"
	"github.com/cozy-creator/cozy-creator/internal/exit"
	"github.com/cozy-creator/cozy-creator/internal/home"
	"github.com/cozy-creator/cozy-creator/internal/hub"
	"github.com/cozy-creator/cozy-creator/internal/orchestrator"
	"github.com/cozy-creator/cozy-creator/internal/output"
	"github.com/cozy-creator/cozy-creator/internal/records"
	"github.com/cozy-creator/cozy-creator/internal/rental"
	"github.com/cozy-creator/cozy-creator/internal/service"
)

func handleUnload(ctx *Context) *exit.Error {
	state := service.Probe(ctx.Cfg)
	if !state.Up {
		return emit(ctx, output.Record{Fields: []output.Field{
			{K: "stopped", V: 0}, {K: "released", V: "0B"}, {K: "changed", V: false},
		}})
	}
	ctx.Service = state
	client, problem := dial(ctx)
	if problem != nil {
		return problem
	}
	result, problem := client.Unload()
	if problem != nil {
		return problem
	}
	list := output.List{
		Name: "workers", Fields: []string{"endpoint", "devices"},
		AllFields: []string{"instance", "endpoint", "release", "devices"},
	}
	for _, worker := range result.Stopped {
		list.Rows = append(list.Rows, map[string]string{
			"instance": worker.InstanceID, "endpoint": worker.Endpoint,
			"release": worker.ReleaseID, "devices": strings.Join(worker.Devices, ","),
		})
	}
	list.Aggregates = []output.Field{{K: "changed", V: result.Count > 0}}
	return emit(ctx, list)
}

func handleUp(ctx *Context) *exit.Error {
	state, changed, problem := ensureController(ctx)
	if problem != nil {
		return problem
	}
	fields := []output.Field{
		{K: "controller", V: "running"}, {K: "url", V: "http://" + state.Addr + "/"},
		{K: "api", V: "http://" + state.Addr}, {K: "pid", V: state.PID},
		{K: "since", V: state.Since}, {K: "changed", V: changed},
	}
	return emit(ctx, compactRecord(fields, "url", "changed"))
}

func handleDown(ctx *Context) *exit.Error {
	all := ctx.Inv.Bool("--all")
	state := service.Probe(ctx.Cfg)
	if !state.Up {
		blockers, problem := offlineDownBlockers(ctx)
		if problem != nil {
			return problem
		}
		if len(blockers) == 0 {
			return emit(ctx, output.Record{Fields: []output.Field{
				{K: "controller", V: "stopped"}, {K: "changed", V: false},
			}})
		}
		if !all {
			return exit.Named(exit.Conflict, "active_work",
				"controller down refused: active %s", strings.Join(blockers, ", ")).
				WithRemedy("cancel/end the named work, or use explicit `cozy down --all`")
		}
		state, _, problem = ensureController(ctx)
		if problem != nil {
			return problem
		}
	}
	ctx.Service = state
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
		return finishControllerDown(ctx, nil)
	}
	return downAll(ctx, client)
}

func downAll(ctx *Context, client *localapi.Client) *exit.Error {
	canceled := map[string]bool{}
	ended := map[string]bool{}
	for {
		result, problem := client.Down(true)
		if problem != nil {
			return problem.WithRemedy("partial teardown stopped; the controller remains alive for reconciliation")
		}
		for _, identity := range result.CancellationRequested {
			canceled[identity.ID] = true
		}
		for _, identity := range result.Rentals {
			switch identity.Kind {
			case "rental":
				if problem := endRentalSilently(ctx, identity.ID); problem != nil {
					return problem.WithRemedy("rental termination was not confirmed; the controller remains alive")
				}
				ended[identity.ID] = true
			case "rental_operation":
				id, problem := resolveRentalOperation(ctx, identity.ID)
				if problem != nil {
					return problem.WithRemedy("the paid operation remains recorded and the controller remains alive")
				}
				if id != "" {
					if problem := endRentalSilently(ctx, id); problem != nil {
						return problem.WithRemedy("rental termination was not confirmed; the controller remains alive")
					}
					ended[id] = true
				}
			}
		}
		if result.ShuttingDown {
			return finishControllerDown(ctx, []output.Field{
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
		ID: seen.ID, EndpointRef: request.EndpointRef, AcceleratorModel: request.AcceleratorModel,
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

func finishControllerDown(ctx *Context, extra []output.Field) *exit.Error {
	deadline := time.Now().Add(2 * orchestrator.StopGrace)
	for time.Now().Before(deadline) {
		if !service.Probe(ctx.Cfg).Up {
			fields := []output.Field{{K: "controller", V: "stopped"}, {K: "changed", V: true}}
			fields = append(fields, extra...)
			return emit(ctx, output.Record{Fields: fields})
		}
		time.Sleep(50 * time.Millisecond)
	}
	return exit.New(exit.Conflict, "the controller accepted down but did not finish stopping").
		WithRemedy("the controller remains responsible for its workers; no forced kill was performed")
}
