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

	"github.com/cozy-creator/cozy/internal/api"
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
	if source.RuntimeWheel == "" {
		// Unnamed software is the Hub's target; with no target the install names nothing and is refused.
		if target, problem := client(ctx).Software(installCtx); problem == nil {
			source.RuntimeVersion, source.TensorFSVersion = target.Runtime, target.TensorFS
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
		// The install stands either way; what the records could not say is said here.
		switch uncollected, problem := replacedMachineRuns(ctx); {
		case problem != nil:
			notes = append(notes, "the older machine's ended runs could not be marked as gone with it: "+problem.Message)
		case len(uncollected) > 0:
			notes = append(notes, "completed but not fully collected; any output not already saved went with the older machine: "+runsPhrase(uncollected))
		}
	}
	return emit(ctx, output.Record{Fields: fields, Notes: notes})
}

// localRuns asks this computer's machine's runs of the records; an absent database has none.
func localRuns(ctx *Context, ask func(*records.Store) ([]records.Request, *exit.Error)) ([]records.Request, *exit.Error) {
	layout := home.Paths(ctx.Cfg.Home)
	if _, err := os.Stat(layout.DB); err != nil {
		return nil, nil
	}
	store, problem := records.Open(layout.DB)
	if problem != nil {
		return nil, problem
	}
	defer store.Close()
	return ask(store)
}

// localRunsSettled refuses while this computer's machine holds runs that have not ended. Ended
// runs do not hold it: only the machine's own record of them goes.
func localRunsSettled(ctx *Context) *exit.Error {
	live, problem := localRuns(ctx, func(store *records.Store) ([]records.Request, *exit.Error) {
		return store.MachineLiveRuns(machines.Local)
	})
	if problem != nil || len(live) == 0 {
		return problem
	}
	return exit.Named(exit.Conflict, "machine.holds_runs", "this computer's machine still holds %s; replacing it would lose that work", runsPhrase(live)).
		WithRemedy("let them finish, or settle each with `cozy run cancel --abandon`, then install again").
		WithNext("cozy run cancel --abandon " + runReference(live[0].Number, live[0].ID))
}

// replacedMachineRuns records that the replaced machine's own record of its ended runs went
// with it, so nothing waits on them, and answers the completed runs never collected from it.
func replacedMachineRuns(ctx *Context) ([]records.Request, *exit.Error) {
	return localRuns(ctx, func(store *records.Store) ([]records.Request, *exit.Error) {
		uncollected, problem := store.MachineUncollectedRuns(machines.Local)
		if problem != nil {
			return nil, problem
		}
		return uncollected, store.LoseMachine(machines.Local, "this computer's machine was replaced; the older machine's own record of this run went with it")
	})
}

func handleMachineShow(ctx *Context) *exit.Error {
	if ctx.Inv.Value("--machine-endpoint-file") != "" {
		return showEndpoint(ctx)
	}
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
	launch, problem := host.Start(startCtx, client(ctx))
	if problem != nil {
		return problem
	}
	if launch.Kept != "" {
		fmt.Fprintln(ctx.Err, "machine: "+launch.Kept)
	}
	return handleMachineShow(ctx)
}

// startForCancel starts this computer's stopped machine for a canceled run still waiting on
// it, as `cozy machine start` does: the daemon then delivers the cancel and the run settles
// as its machine says (canceled, or completed if it finished first). A machine that cannot
// start leaves the run canceling, and the caller is told why and what to do next.
func startForCancel(ctx *Context, number int64, id, status string, view *api.MachineExecutionView) *exit.Error {
	if status != "canceling" || view == nil || view.Machine != machines.Local || view.AbandonedLocally {
		return nil
	}
	host, problem := localMachineHost(ctx)
	if problem != nil {
		return problem
	}
	if current, problem := host.Status(); problem == nil && current.Running {
		return nil
	}
	startCtx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()
	if _, problem := host.Start(startCtx, client(ctx)); problem != nil {
		remedy := problem.Remedy
		if remedy == "" {
			remedy = "`cozy machine start` shows why it cannot start; the cancel is delivered as soon as the machine runs"
		}
		return exit.Named(problem.Code, "run.cancel_waits_for_machine",
			"run %s stays canceling: this computer's machine could not start to take the cancel: %s", runReference(number, id), problem.Message).
			WithRemedy("%s", remedy).WithNext("cozy machine start", "cozy run cancel "+runReference(number, id)+" --await")
	}
	return nil
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
