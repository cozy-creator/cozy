package cli

import (
	"context"
	"encoding/json"
	"strings"

	machinepb "github.com/cozy-creator/cozy/protocol/cozy/machine/v1"

	"github.com/cozy-creator/cozy/internal/api"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/machineendpoint"
	"github.com/cozy-creator/cozy/internal/machines"
	"github.com/cozy-creator/cozy/internal/records"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

// ValidateEndpoint proves target TLS, controller authority and workspace before admission.
func (m *machineRuns) ValidateEndpoint(ctx context.Context, ep *machineendpoint.Endpoint) *exit.Error {
	if ep == nil {
		return exit.New(exit.Validation, "explicit machine endpoint is absent")
	}
	// A cozy.machine.v1 machine answers Status to a key it authorizes: that is the check.
	v1, problem := m.machines.DialEndpointV1(*ep)
	if problem != nil {
		return problem
	}
	frame, err := v1.Status(ctx)
	v1.Close()
	switch {
	case err == nil && frame.WorkerId != ep.WorkerID:
		return exit.New(exit.Conflict, "the explicit endpoint is machine %s, not %s", frame.WorkerId, ep.WorkerID)
	case err == nil:
		return nil
	case status.Code(err) == codes.Unauthenticated || status.Code(err) == codes.PermissionDenied:
		return exit.Named(exit.Credential, "machine.endpoint_unauthorized", "the machine does not authorize this host's key: %s", status.Convert(err).Message())
	}
	return machines.Transport(err)
}

// Prewarm makes one installation on its machine without running anything: `cozy package
// install`, `cozy model download` and `cozy model upload`, as a warm run.
func (m *machineRuns) Prewarm(ctx context.Context, row records.RentalInstall, report func(machines.InstallProgress)) (json.RawMessage, *exit.Error) {
	return m.prewarmV1(ctx, row, report)
}

func (m *machineRuns) Forget(machine string) { m.machines.Forget(machine) }

// Status reads one machine's picture over cozy.machine.v1 as its owner.
func (m *machineRuns) Status(ctx context.Context, machine string) (api.MachineStatus, *exit.Error) {
	connection, problem := m.machines.DialV1(ctx, machine, "reading its status")
	if problem != nil {
		return api.MachineStatus{}, problem
	}
	defer connection.Close()
	frame, err := connection.Status(ctx)
	if err != nil {
		return api.MachineStatus{}, machines.Transport(err)
	}
	return statusOf(frame), nil
}

// machineLogs maps the log names clients use to the logs machines keep: the name
// cozy.machine.v1 Read takes, and worker.v1's.
var machineLogs = map[string]struct{ name string }{"tensorfs": {"tensorfs-transport"}}

// MachineLog reads one log a machine keeps, with Read.
func (m *machineRuns) MachineLog(ctx context.Context, machine, log string, tailBytes uint64) (api.MachineLog, *exit.Error) {
	kept, ok := machineLogs[log]
	if !ok {
		return api.MachineLog{}, exit.Named(exit.NotFound, "machine.log_unknown", "machines keep no log %q", log)
	}
	out := api.MachineLog{Log: log}
	var text strings.Builder
	v1, problem := m.machines.DialV1(ctx, machine, "reading its logs")
	if problem != nil {
		return api.MachineLog{}, problem
	}
	_, _, err := v1.ReadLog(ctx, kept.name, tailBytes, &text)
	v1.Close()
	if err != nil {
		return api.MachineLog{}, machines.Transport(err)
	}
	out.Text = text.String()
	return out, nil
}

// statusOf is a machine's Status frame as the daemon's API answers it.
func statusOf(frame *machinepb.StatusFrame) api.MachineStatus {
	out := api.MachineStatus{WorkerID: frame.GetWorkerId(), BootID: frame.GetBootId(), Agent: frame.GetVersion(),
		Phase: frame.GetPhase(), Runtime: frame.GetRuntime(), TensorFS: frame.GetTensorfs(),
		Capabilities: frame.GetCapabilities(), IdleDeadlineUnixMS: frame.GetIdleDeadlineUnixMs(),
		GPUs: []api.MachineGPU{}, Runs: []api.MachineRun{}, Environments: []api.MachineEnvironment{},
		DiskTotalBytes: frame.GetDisk().GetTotalBytes(), DiskFreeBytes: frame.GetDisk().GetFreeBytes()}
	for _, gpu := range frame.GetGpus() {
		out.GPUs = append(out.GPUs, api.MachineGPU{Index: gpu.GetIndex(), Name: gpu.GetName(), MemoryBytes: gpu.GetMemoryBytes(), Driver: gpu.GetDriver()})
	}
	for _, run := range frame.GetRuns() {
		out.Runs = append(out.Runs, api.MachineRun{ID: run.GetId(), Number: run.GetNumber(), State: run.GetState()})
	}
	for _, env := range frame.GetEnvironments() {
		out.Environments = append(out.Environments, api.MachineEnvironment{Installation: env.GetInstallation(), Package: env.GetPackage(),
			Release: env.GetRelease(), Level: env.GetLevel()})
	}
	for _, item := range frame.GetWarm() {
		level := strings.ToLower(strings.TrimPrefix(item.GetLevel().String(), "WARM_LEVEL_"))
		out.Warm = append(out.Warm, api.MachineWarmMember{Package: either(item.GetRelease().GetPackage(), item.GetInstallation()),
			Release: item.GetRelease().GetRelease(), Entrypoint: item.GetEntrypoint(), Level: level, Holds: item.GetHolds(), HeldBack: item.GetHeldBack()})
	}
	return out
}
