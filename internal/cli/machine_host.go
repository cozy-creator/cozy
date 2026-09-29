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
	// The machine's Host is this cozy binary; --host pins another cozy build, for development.
	if source.Pinned = source.Host != ""; !source.Pinned {
		self, err := machines.Self()
		if err != nil {
			return exit.Internalf("cannot locate the cozy binary: %s", err)
		}
		source.Host = self
	} else if module := machines.HostModule(source.Host); module != machines.CozyModule {
		return exit.Usagef("--host %s is not a cozy binary (%s): the machine's Host is cozy itself", source.Host, cmp.Or(module, "not a Go program")).
			WithRemedy("omit --host to install this cozy")
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
	fields := []output.Field{{K: "root", V: host.Root()}, {K: "machine", V: status.MachineID}, {K: "hub", V: status.Hub},
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
