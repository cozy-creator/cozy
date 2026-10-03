package cli

import (
	"context"
	"os"
	"os/exec"
	"os/signal"
	"path/filepath"
	"syscall"

	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/home"
	"github.com/cozy-creator/cozy/internal/machines"
	"github.com/cozy-creator/cozy/internal/output"
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
	// The Runtime wheel supplies the default agent; --host deliberately pins a copy, which
	// Install accepts by its own identity and capabilities.
	source.Pinned = source.Host != ""
	uv, err := exec.LookPath("uv")
	if err != nil {
		return exit.Named(exit.Structural, "machine.uv_missing", "uv builds the machine's Python environment and is not on PATH")
	}
	installCtx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()
	installed, problem := host.Install(installCtx, source, uv)
	if problem != nil {
		return problem
	}
	fields := []output.Field{
		{K: "root", V: host.Root()}, {K: "host", V: installed.Host.Name}, {K: "host_sha256", V: installed.Host.SHA256},
		{K: "runtime", V: installed.Runtime.Name}, {K: "tensorfs", V: installed.TensorFS.Name},
	}
	notes := []string{"the installed machine remains available for local runs"}
	if installed.Pending != nil {
		fields = append(fields, output.Field{K: "update", V: installed.Pending})
		notes = append(notes, "the Runtime candidate is prepared; activation waits for current machine work to drain")
	} else {
		notes = append(notes, "the next local run launches this machine")
	}
	return emit(ctx, output.Record{Fields: fields, Notes: notes})
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
	if status.Running && !status.Recorded {
		notes = append(notes, "its launch record is missing; `cozy machine start` or the next local run adopts it")
	} else if status.Running {
		readCtx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
		defer stop()
		live, problem := host.ReadSoftware(readCtx)
		if problem != nil {
			notes = append(notes, "live software could not be observed: "+problem.Message)
		} else if live != nil {
			fields = append(fields, output.Field{K: "runtime", V: live.Runtime}, output.Field{K: "tensorfs", V: live.TensorFS},
				output.Field{K: "agent", V: live.Agent}, output.Field{K: "bootstrap", V: live.Bootstrap}, output.Field{K: "phase", V: live.Phase})
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
