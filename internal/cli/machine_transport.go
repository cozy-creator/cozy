package cli

import (
	"context"
	"maps"
	"net/url"
	"slices"
	"strings"
	"sync"

	machinepb "github.com/cozy-creator/cozy/protocol/cozy/machine/v1"

	"github.com/cozy-creator/cozy/internal/api"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/hub"
	"github.com/cozy-creator/cozy/internal/machineendpoint"
	"github.com/cozy-creator/cozy/internal/machines"
	"github.com/cozy-creator/cozy/internal/machinev1"
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

// Status reads one machine's picture over cozy.machine.v1 as its owner.
func (m *machineRuns) Status(ctx context.Context, machine string) (api.MachineStatus, *exit.Error) {
	var connection *machines.V1
	var problem *exit.Error
	if observed, ok := m.fleet.listedReady(machine); ok {
		connection, problem = m.machines.DialObserved(observed, "reading its status")
	} else {
		connection, problem = m.machines.DialV1(ctx, machine, "reading its status")
	}
	if problem != nil {
		return api.MachineStatus{}, problem
	}
	defer connection.Close()
	return m.readStatus(ctx, connection, machine)
}

// readStatus is one Status read on an open connection.
func (m *machineRuns) readStatus(ctx context.Context, connection *machines.V1, machine string) (api.MachineStatus, *exit.Error) {
	frame, err := connection.Status(ctx)
	if err != nil {
		return api.MachineStatus{}, machines.Transport(err)
	}
	status := statusOf(frame)
	// Each installation's hub as this computer names it: a named hub at that origin or at the
	// origin it gives machines to read it at, else the rental's own hub, which a rental reads at
	// the address the Hub gives it, not this computer's URL of it.
	known := append([]string{m.context.Cfg.HubURL}, slices.Collect(maps.Values(m.context.Cfg.Hubs))...)
	hubs := map[string]string{}
	for _, url := range known {
		hubs[originKey(url)] = url
	}
	for _, url := range known {
		if reads := originKey(client(m.context.forHub(url)).KeptOrigin()); reads != "" && hubs[reads] == "" {
			hubs[reads] = url
		}
	}
	if row, _ := m.store.RentalRow(machine); row != nil && row.Hub != "" {
		for _, own := range frame.GetHubs() {
			hubs[originKey(own.GetOrigin())] = row.Hub
		}
	}
	for i, env := range status.Environments {
		if url := hubs[originKey(env.Hub)]; url != "" {
			status.Environments[i].Hub = url
		}
	}
	return status, nil
}

// machineRead is one machine's Status as a listing read it, or why it could not.
type machineRead struct {
	status  api.MachineStatus
	problem *exit.Error
}

// statuses reads each ready rental's machine, at most eight at once, at the identity its Hub
// listed (DialV1 would ask the Hub again for each).
func (m *machineRuns) statuses(ctx context.Context, ready []hub.Rental) map[string]machineRead {
	out := make(map[string]machineRead, len(ready))
	var mu sync.Mutex
	var wait sync.WaitGroup
	slots := make(chan struct{}, 8)
	for _, observed := range ready {
		wait.Go(func() {
			slots <- struct{}{}
			defer func() { <-slots }()
			var read machineRead
			connection, problem := m.machines.DialObserved(observed, "reading its status")
			if read.problem = problem; problem == nil {
				read.status, read.problem = m.readStatus(ctx, connection, observed.ID)
				connection.Close()
			}
			mu.Lock()
			out[observed.ID] = read
			mu.Unlock()
		})
	}
	wait.Wait()
	return out
}

// originKey is scheme://host:port, lowercase with the default port spelled, as a machine keys
// a Hub origin; "" for anything else.
func originKey(origin string) string {
	parsed, err := url.Parse(strings.TrimRight(strings.TrimSpace(origin), "/"))
	if err != nil || parsed.Host == "" || parsed.Path != "" {
		return ""
	}
	port := parsed.Port()
	switch {
	case port != "":
	case parsed.Scheme == "https":
		port = "443"
	case parsed.Scheme == "http":
		port = "80"
	default:
		return ""
	}
	return strings.ToLower(parsed.Scheme) + "://" + strings.ToLower(parsed.Hostname()) + ":" + port
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
		Capabilities: frame.GetCapabilities(), IdleDeadlineUnixMS: machinev1.IdleDeadline(frame),
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
			Release: env.GetRelease(), Level: env.GetLevel(), Hub: env.GetHub()})
	}
	for _, item := range frame.GetWarm() {
		level := strings.ToLower(strings.TrimPrefix(item.GetLevel().String(), "WARM_LEVEL_"))
		out.Warm = append(out.Warm, api.MachineWarmMember{Package: either(item.GetRelease().GetPackage(), item.GetInstallation()),
			Release: item.GetRelease().GetRelease(), Entrypoint: item.GetEntrypoint(), Level: level, Holds: item.GetHolds(), HeldBack: item.GetHeldBack()})
	}
	return out
}
