package cli

import (
	"context"
	"fmt"
	"os"
	"os/exec"
	"os/signal"
	"path/filepath"
	"strings"
	"syscall"

	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/home"
	"github.com/cozy-creator/cozy/internal/machines"
	"github.com/cozy-creator/cozy/internal/output"
	"github.com/cozy-creator/cozy/internal/records"
	"github.com/cozy-creator/cozy/internal/userunit"
)

func localMachineHost(ctx *Context) (*machines.Host, *exit.Error) {
	layout, problem := home.Open(ctx.Cfg.Home)
	if problem != nil {
		return nil, problem
	}
	host := machines.NewHost(layout.Machine, ctx.Cfg.TensorFSRoot, ctx.Cfg.Child())
	host.GPUBudget = ctx.Cfg.MachineGPUBudget
	host.WebRTCPort = ctx.Cfg.MachineWebRTCPort
	return host, nil
}

func handleMachineInstall(ctx *Context) *exit.Error {
	host, problem := localMachineHost(ctx)
	if problem != nil {
		return problem
	}
	source := machines.Source{}
	for flag, target := range map[string]*string{"--host": &source.Host, "--runtime-wheel": &source.RuntimeWheel, "--tensorfs-wheel": &source.TensorFSWheel} {
		value := ctx.Inv.Value(flag)
		if value == "" {
			continue
		}
		absolute, err := filepath.Abs(value)
		if err != nil {
			return exit.Usagef("cannot resolve %s: %s", value, err)
		}
		*target = absolute
	}
	// The Runtime wheel's bundled machine is the default; --host names another, which Install
	// accepts by the API it serves.
	uv, err := exec.LookPath("uv")
	if err != nil {
		return exit.Named(exit.Structural, "machine.uv_missing", "uv builds the machine's package environments and is not on PATH")
	}
	installCtx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()
	if host.PredatesAPI(installCtx) {
		// The install replaces this machine and its run journal: what it still holds settles first.
		if problem := localRunsSettled(ctx); problem != nil {
			return problem
		}
	}
	installed, problem := host.Install(installCtx, source, uv)
	if problem != nil {
		return problem
	}
	fields := []output.Field{
		{K: "root", V: host.Root()}, {K: "host", V: installed.Host.Name}, {K: "host_sha256", V: installed.Host.SHA256},
		{K: "runtime", V: installed.Runtime.Name}, {K: "tensorfs", V: installed.TensorFS.Name},
	}
	notes := []string{"the installed machine remains available for local runs", "the next local run launches this machine"}
	if replaced := installed.Replaced; replaced != nil {
		fields = append(fields, output.Field{K: "replaced", V: replaced})
		notes = []string{
			"replaced the older machine; the new one started and answered as ready",
			"kept: your downloaded models and the package cache (" + strings.Join(replaced.Kept, ", ") + ")",
			fmt.Sprintf("removed the old machine's software, package environments and run history (%.1f GiB freed); each package is set up again on its first run", float64(replaced.FreedBytes)/(1<<30)),
		}
	}
	return emit(ctx, output.Record{Fields: fields, Notes: notes})
}

// localRunsSettled refuses while this computer's machine holds unsettled runs.
func localRunsSettled(ctx *Context) *exit.Error {
	layout := home.Paths(ctx.Cfg.Home)
	if _, err := os.Stat(layout.DB); err != nil {
		return nil
	}
	store, problem := records.Open(layout.DB)
	if problem != nil {
		return problem
	}
	defer store.Close()
	live, problem := store.MachineLiveRuns(machines.Local)
	if problem != nil || len(live) == 0 {
		return problem
	}
	return exit.Named(exit.Conflict, "machine.holds_runs", "this computer's machine still holds %s; replacing it would lose that work", runsPhrase(live)).
		WithRemedy("let them finish, or settle each with `cozy run cancel --abandon`, then install again").
		WithNext("cozy run cancel --abandon " + runReference(live[0].Number, live[0].ID))
}

func handleMachineShow(ctx *Context) *exit.Error {
	host, problem := localMachineHost(ctx)
	if problem != nil {
		return problem
	}
	status, problem := host.Status()
	if problem != nil {
		return problem
	}
	fields := []output.Field{{K: "root", V: host.Root()}, {K: "machine", V: status.MachineID},
		{K: "running", V: status.Running}, {K: "pid", V: status.PID}}
	if userunit.Available() {
		fields = append(fields, output.Field{K: "unit", V: host.Unit()})
	}
	if status.Installed != nil {
		fields = append(fields, output.Field{K: "last_install", V: status.Installed})
	}
	var notes []string
	if host.PredatesAPI(context.Background()) {
		notes = append(notes, "this machine is an older kind this cozy cannot run; `cozy machine install` replaces it and keeps your downloaded models")
	} else if status.Running && !status.Recorded {
		notes = append(notes, "its launch record is missing; `cozy machine start` or the next local run adopts it")
	} else if status.Running {
		readCtx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
		defer stop()
		live, problem := host.ReadStatus(readCtx)
		if problem != nil {
			notes = append(notes, "live status could not be observed: "+problem.Message)
		} else if live != nil {
			fields = append(fields, machineStatusFields(statusOf(live), !ctx.Mode().Human || ctx.Mode().JSON)...)
		}
	}
	return emit(ctx, output.Record{Fields: fields, Notes: notes})
}

func handleMachineStart(ctx *Context) *exit.Error {
	host, problem := localMachineHost(ctx)
	if problem != nil {
		return problem
	}
	startCtx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()
	if _, problem := host.Start(startCtx); problem != nil {
		return problem
	}
	return handleMachineShow(ctx)
}

func handleMachineStop(ctx *Context) *exit.Error {
	host, problem := localMachineHost(ctx)
	if problem != nil {
		return problem
	}
	if problem := host.Stop(context.Background()); problem != nil {
		return problem
	}
	return emit(ctx, output.Record{Fields: []output.Field{{K: "stopped", V: true}}})
}

// runsPhrase names up to five runs with their states.
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
