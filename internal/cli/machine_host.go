package cli

import (
	"cmp"
	"context"
	"os/exec"
	"path/filepath"

	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/home"
	"github.com/cozy-creator/cozy/internal/machines"
	"github.com/cozy-creator/cozy/internal/output"
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
	// The machine agent is released independently from the CLI.
	if source.Pinned = source.Host != ""; !source.Pinned {
		source.Host, problem = host.PublishedAgent(context.Background())
		if problem != nil {
			return problem
		}
	} else if module := machines.HostModule(source.Host); module != machines.AgentModule {
		return exit.Usagef("--host %s is not a cozy-machine binary (%s)", source.Host, cmp.Or(module, "not a Go program")).
			WithRemedy("omit --host to install the published machine agent")
	}
	uv, err := exec.LookPath("uv")
	if err != nil {
		return exit.Named(exit.Structural, "machine.uv_missing", "uv builds the machine's Python environment and is not on PATH")
	}
	installed, problem := host.Install(context.Background(), source, uv)
	if problem != nil {
		return problem
	}
	return emit(ctx, output.Record{Fields: []output.Field{
		{K: "root", V: host.Root()}, {K: "host", V: installed.Host.Name}, {K: "host_sha256", V: installed.Host.SHA256},
		{K: "runtime", V: installed.Runtime.Name}, {K: "tensorfs", V: installed.TensorFS.Name},
	}, Notes: []string{"the next local run launches this machine"}})
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
	if status.Installed != nil {
		fields = append(fields, output.Field{K: "host", V: status.Installed.Host.Name},
			output.Field{K: "runtime", V: status.Installed.Runtime.Name}, output.Field{K: "tensorfs", V: status.Installed.TensorFS.Name})
	}
	return emit(ctx, output.Record{Fields: fields})
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
