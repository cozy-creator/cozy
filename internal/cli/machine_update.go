package cli

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"os/signal"
	"path/filepath"
	"strings"
	"syscall"

	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/home"
	"github.com/cozy-creator/cozy/internal/machineendpoint"
	"github.com/cozy-creator/cozy/internal/machines"
	"github.com/cozy-creator/cozy/internal/output"
	"github.com/cozy-creator/cozy/internal/records"
	"github.com/cozy-creator/cozy/internal/scratch"
	pb "github.com/cozy-creator/cozy/protocol/cozy/machine/v1"
)

// An explicit endpoint is a public selector, not a rental enrollment. The normal
// resolver selects the existing owner key (or the matching rental's own key).
func endpointMaintenance(ctx *Context, call context.Context) (*machines.V1, *pb.StatusFrame, *exit.Error) {
	ep, err := machineendpoint.Read(ctx.Inv.Value("--machine-endpoint-file"))
	if err != nil {
		return nil, nil, exit.New(exit.Validation, "invalid explicit machine endpoint: %s", err)
	}
	layout, problem := home.Open(ctx.Cfg.Home)
	if problem != nil {
		return nil, nil, problem
	}
	store, problem := records.Open(layout.DB)
	if problem != nil {
		return nil, nil, problem
	}
	defer store.Close()
	resolver := &machines.Resolver{Host: machines.NewHost(layout.Machine, ctx.Cfg.TensorFSRoot, nil), EndpointRental: rentalEndpoint(layout, store)}
	machine, problem := resolver.DialEndpointV1(*ep)
	if problem != nil {
		return nil, nil, problem
	}
	frame, err := machine.Status(call)
	if err != nil {
		machine.Client.Close()
		return nil, nil, machines.Transport(err)
	}
	if frame.GetWorkerId() != ep.WorkerID || frame.GetBootId() != ep.WorkerBootID {
		machine.Client.Close()
		return nil, nil, exit.New(exit.Conflict, "the endpoint's worker or boot identity changed; use its current pinned endpoint")
	}
	return machine, frame, nil
}

func showEndpoint(ctx *Context) *exit.Error {
	call, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()
	machine, frame, problem := endpointMaintenance(ctx, call)
	if problem != nil {
		return problem
	}
	defer machine.Client.Close()
	fields := []output.Field{{K: "machine", V: machine.Name}, {K: "worker_id", V: machine.WorkerID}, {K: "worker_boot_id", V: machine.BootID}}
	fields = append(fields, machineStatusFields(statusOf(frame), !ctx.Mode().Human || ctx.Mode().JSON)...)
	return emit(ctx, output.Record{Fields: fields})
}

func handleEndpointUpdate(ctx *Context) *exit.Error {
	id := strings.TrimSpace(ctx.Inv.Value("--idempotency-key"))
	if id == "" || len(id) > 200 {
		return exit.Usagef("--idempotency-key must contain 1 to 200 bytes")
	}
	observe := ctx.Inv.Bool("--observe")
	selection := runtimeUpdateSelection{RuntimeVersion: ctx.Inv.Value("--runtime-version"), TensorFSVersion: ctx.Inv.Value("--tensorfs-version")}
	runtimeWheel, tensorfsWheel := ctx.Inv.Value("--runtime-wheel"), ctx.Inv.Value("--tensorfs-wheel")
	named := runtimeWheel != "" || tensorfsWheel != "" || selection.RuntimeVersion != "" || selection.TensorFSVersion != ""
	if observe && (named || ctx.Inv.Bool("--keep-agent")) {
		return exit.Usagef("--observe takes the endpoint and update id only; it submits no software selection")
	}
	if !observe && !named {
		return exit.Usagef("name the software to update, or use --observe to follow an accepted update")
	}
	if runtimeWheel != "" && selection.RuntimeVersion != "" || tensorfsWheel != "" && selection.TensorFSVersion != "" {
		return exit.Usagef("name a wheel or a version for each member, not both")
	}
	call, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop() // Disconnect only: the accepted update belongs to the machine.
	machine, _, problem := endpointMaintenance(ctx, call)
	if problem != nil {
		return problem
	}
	defer machine.Client.Close()
	var snapshots []*scratch.Dir
	defer func() {
		for _, snapshot := range snapshots {
			snapshot.Release()
		}
	}()
	for _, member := range []struct {
		path string
		into **runtimeUpdateWheel
	}{
		{runtimeWheel, &selection.LocalRuntime}, {tensorfsWheel, &selection.LocalTensorFS},
	} {
		if member.path == "" {
			continue
		}
		absolute, err := filepath.Abs(member.path)
		if err != nil {
			return exit.New(exit.Validation, "cannot resolve update wheel: %s", err)
		}
		wheel, snapshot, problem := freezeRuntimeWheel(home.Paths(ctx.Cfg.Home).Tmp, absolute)
		if problem != nil {
			return problem
		}
		*member.into = wheel
		snapshots = append(snapshots, snapshot)
	}
	last := ""
	progress := func(event *pb.RunEvent) {
		stage := event.GetProgress().GetStage()
		if stage == "" || stage == last {
			return
		}
		last = stage
		if ctx.Mode().JSON {
			_ = json.NewEncoder(ctx.Err).Encode(map[string]string{"update_id": id, "stage": stage})
		} else {
			fmt.Fprintf(ctx.Err, "update %s: %s\n", id, stage)
		}
	}
	var outcome *pb.Outcome
	var err error
	if observe {
		outcome, err = machine.ObserveUpdate(call, id, progress)
	} else {
		agent := "bundled"
		if ctx.Inv.Bool("--keep-agent") {
			agent = "explicit"
		}
		cohort, problem := runtimeUpdateCohort(call, machine.Client, selection, agent)
		if problem != nil {
			return problem
		}
		outcome, err = machine.Update(call, id, cohort, progress)
	}
	if err != nil {
		problem := machines.Transport(err)
		if call.Err() == nil && problem.Code != exit.Unavailable {
			return problem
		}
		return exit.Named(exit.Unavailable, "machine.update_observation_lost", "update %q may continue on the machine: %s", id, machines.Transport(err).Message).
			WithRemedy("repeat with the same endpoint and idempotency key plus --observe; omit software-selection flags")
	}
	if outcome.GetStatus() != "succeeded" {
		return exit.Named(exit.Failed, "machine.runtime_update_failed", "update %q: %s", id, either(outcome.GetReason().GetMessage(), outcome.GetReason().GetCode()))
	}
	var result runtimeUpdateResult
	if json.Unmarshal(outcome.GetResult(), &result) != nil || result.To.Runtime == "" || result.To.TensorFS == "" {
		return exit.Named(exit.Structural, "machine.update_result_unreadable", "update %q succeeded, but its software result cannot be read; inspect the endpoint with cozy machine show", id)
	}
	return emit(ctx, output.Record{Fields: []output.Field{
		{K: "machine", V: machine.Name}, {K: "update_id", V: id}, {K: "status", V: "ready"},
		{K: "runtime", V: result.To.Runtime}, {K: "tensorfs", V: result.To.TensorFS},
		{K: "previous_runtime", V: result.From.Runtime},
	}})
}
