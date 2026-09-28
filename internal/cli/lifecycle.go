package cli

import (
	"fmt"
	"io"
	"time"

	localapi "github.com/cozy-creator/cozy/internal/client"
	"github.com/cozy-creator/cozy/internal/daemon"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/home"
	"github.com/cozy-creator/cozy/internal/hostgpu"
	"github.com/cozy-creator/cozy/internal/hostruntime"
	"github.com/cozy-creator/cozy/internal/orchestrator"
	"github.com/cozy-creator/cozy/internal/output"
	"github.com/cozy-creator/cozy/internal/records"
	"github.com/cozy-creator/cozy/internal/rental"
)

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
	// Rentals and the Hub need no host Runtime; only local runs refuse without a usable one.
	if _, problem := hostruntime.Path(ctx.Cfg.Tool()); problem != nil && problem.ErrName() != "host_runtime_missing" {
		record.Notes = append(record.Notes, fmt.Sprintf("local runs refuse with %s until the host Runtime is upgraded: %s; %s",
			problem.ErrName(), problem.Message, problem.Remedy))
	}
	return emit(ctx, record)
}

func handleDown(ctx *Context) *exit.Error {
	all := ctx.Inv.Bool("--all")
	force := ctx.Inv.Bool("--force")
	state := daemon.Probe(ctx.Cfg)
	if !state.Up {
		if !all {
			return emit(ctx, output.Record{Fields: []output.Field{{K: "daemon", V: "stopped"}, {K: "changed", V: false}}})
		}
		blockers, problem := offlineDownBlockers(ctx)
		if problem != nil {
			return problem
		}
		if len(blockers) == 0 {
			return emit(ctx, output.Record{Fields: []output.Field{
				{K: "daemon", V: "stopped"}, {K: "changed", V: false},
			}})
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
		result, problem := client.Down(false, force)
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

// downAll is the GUARANTEED teardown (owner ruling 2026-09-04): cancel every request, end
// every rental, and close. It is the verb a person reaches for because reconciliation is
// not working, so no single failure inside it may stop the rest — the previous version
// returned on the first problem with "the daemon remains alive for reconciliation", which
// is plain `down`'s job and left an operator with one wedged request unable to stop their
// daemon by any documented means.
// downAll is the GUARANTEED teardown (owner ruling 2026-09-04): cancel every request, end
// every rental, and close. It is the verb a person reaches for because reconciliation is
// not working, so no single failure inside it may stop the rest — the previous version
// returned on the first problem with "the daemon remains alive for reconciliation", which
// is plain `down`'s job and left an operator with one wedged request unable to stop their
// daemon by any documented means.
func downAll(ctx *Context, client *localapi.Client) *exit.Error {
	canceled := map[string]bool{}
	ended := map[string]bool{}
	var refused []string
	note := func(what, why string) {
		line := what + ": " + why
		for _, seen := range refused {
			if seen == line {
				return
			}
		}
		refused = append(refused, line)
	}
	for {
		result, problem := client.Down(true, false)
		if problem != nil {
			// The daemon itself refused or is unreachable. Nothing further can be asked of
			// it here, so report honestly rather than pretending a teardown happened.
			return problem.WithRemedy(
				"the daemon did not accept the teardown; `cozy down --all` again, or stop the process")
		}
		for _, identity := range result.CancellationRequested {
			canceled[identity.ID] = true
		}
		for _, line := range result.Refused {
			note("request", line)
		}
		for _, identity := range result.Rentals {
			switch identity.Kind {
			case "rental":
				if problem := endRentalSilently(ctx, identity.ID); problem != nil {
					note("rental "+identity.ID, problem.Message)
					continue
				}
				ended[identity.ID] = true
			case "rental_operation":
				id, problem := resolveRentalOperation(ctx, identity.ID)
				if problem != nil {
					note("rental operation "+identity.ID, problem.Message)
					continue
				}
				if id == "" {
					// The hub proved this ask created nothing; there is no pod to end.
					ended[identity.ID] = true
					continue
				}
				if problem := endRentalSilently(ctx, id); problem != nil {
					note("rental "+id, problem.Message)
					continue
				}
				// Keyed under BOTH names: the warning loop below walks the daemon's
				// identities, which spell an operation by its key, and a pod that was
				// ended must not be reported as still billing.
				ended[id], ended[identity.ID] = true, true
			}
		}
		if result.ShuttingDown {
			cancelledLabel := "cancelled"
			if ctx.Mode().JSON {
				cancelledLabel = "canceled"
			}
			fields := []output.Field{
				{K: cancelledLabel, V: len(canceled)}, {K: "rentals_ended", V: len(ended)},
			}
			// SAY WHAT WAS DESTROYED, and what survived. This cancels the owner's in-flight
			// work by design, and a paid pod that could not be ended is still billing after
			// this process exits — silence about either is the wrong kind of tidy.
			for _, identity := range result.Active {
				note("request "+identity.ID, "still "+identity.State+" at shutdown")
			}
			for _, identity := range result.Rentals {
				if !ended[identity.ID] {
					note(identity.Kind+" "+identity.ID,
						"NOT ended; it is still running and still billing")
				}
			}
			if len(refused) > 0 {
				ctx.exitCode = 1
				fields = append(fields, output.Field{K: "not_closed_cleanly", V: len(refused)})
				for _, line := range refused {
					fmt.Fprintf(ctx.Err, "  %s\n", line)
				}
			}
			return finishDaemonDown(ctx, fields)
		}
		// Local cancellations settle through their durable worker terminals. Rental
		// work above is synchronous through confirmed provider absence.
		time.Sleep(100 * time.Millisecond)
	}
}

// endRentalSilently releases one rental for `cozy down --all`. The hub it asks is the one
// the rental was BOUGHT from — recorded on the rental row, or on the paid ask when the
// pod was never attached — because a 404 from any other hub says nothing about the pod.
func endRentalSilently(ctx *Context, id string) *exit.Error {
	authority, problem := rentalAuthority(ctx, id)
	if problem != nil {
		return problem
	}
	sub := *ctx.forHub(authority)
	sub.teardown = true
	sub.Out = io.Discard
	sub.Inv = &Invocation{Args: []string{id}, Bools: map[string]bool{},
		Values: map[string][]string{}, Mode: ctx.Mode()}
	return handleRentRelease(&sub)
}

func rentalAuthority(ctx *Context, id string) (string, *exit.Error) {
	layout, problem := home.Open(ctx.Cfg.Home)
	if problem != nil {
		return "", problem
	}
	store, problem := records.Open(layout.DB)
	if problem != nil {
		return "", problem
	}
	defer store.Close()
	known, problem := rental.Resolve(store, id)
	if problem != nil {
		return "", problem
	}
	return known.Hub, nil
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
	return learnRentalIdentity(ctx, layout, store, operation)
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
	obligations, problem := store.Obligations()
	if problem != nil {
		return nil, problem
	}
	var blockers []string
	for _, o := range obligations {
		switch o.Kind {
		case "invocation", "job", "rental", "rental_operation":
			blockers = append(blockers, o.String())
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
