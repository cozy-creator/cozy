package cli

import (
	"bytes"
	"context"
	"crypto/rand"
	_ "embed"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"net"
	"os"
	"os/exec"
	"path/filepath"
	"slices"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/cozy-creator/cozy/internal/api"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/hostruntime"
	"github.com/cozy-creator/cozy/internal/orchestrator"
	"github.com/cozy-creator/cozy/internal/output"
	"github.com/cozy-creator/cozy/internal/records"
	"github.com/cozy-creator/cozy/internal/rental"
	"github.com/cozy-creator/cozy/internal/scratch"
)

//go:embed runtime_update_transport.py
var runtimeUpdateTransport string

//go:embed runtime_update_probe.py
var runtimeUpdateProbe string

//go:embed runtime_update_stage.py
var runtimeUpdateStager string

type rentalRuntimeUpdates struct {
	machines *machineRuns
	running  sync.Map
}

type runtimeUpdateWheel struct {
	Path     string `json:"path,omitempty"`
	Version  string `json:"version"`
	Filename string `json:"filename"`
	URL      string `json:"url"`
	Digest   string `json:"digest"`
	Length   int64  `json:"length"`
}

type runtimeUpdateTarget struct {
	RuntimeUpdate struct {
		Runtime  runtimeUpdateWheel `json:"runtime"`
		TensorFS runtimeUpdateWheel `json:"tensorfs"`
	} `json:"runtime_update"`
}

type runtimeUpdateSelection struct {
	LocalRuntime  *runtimeUpdateWheel `json:"local_runtime,omitempty"`
	LocalTensorFS *runtimeUpdateWheel `json:"local_tensorfs,omitempty"`
	// Published versions to install instead of local wheels; a machine that updates itself
	// fetches them (rental_runtime_update_native.go), and records its operation in Native.
	RuntimeVersion  string        `json:"runtime_version,omitempty"`
	TensorFSVersion string        `json:"tensorfs_version,omitempty"`
	Native          *nativeUpdate `json:"native,omitempty"`
	Target        runtimeUpdateTarget `json:"target"`
	Directory     string              `json:"directory"`
	Stage         string              `json:"stage"`
	Host          string              `json:"host"`
	SSHArguments  []string            `json:"ssh_arguments"`
	SFTPArguments []string            `json:"sftp_arguments"`
	Selection     json.RawMessage     `json:"selection,omitempty"`
}

func (u *rentalRuntimeUpdates) Start(id string, options api.RuntimeUpdateRequest) (*records.RuntimeUpdate, *exit.Error) {
	return u.start(id, options.RuntimeWheel, options.TensorFSWheel, options.RuntimeVersion, options.TensorFSVersion)
}

func (u *rentalRuntimeUpdates) start(id, wheelPath, tensorfsPath, runtimeVersion, tensorfsVersion string) (*records.RuntimeUpdate, *exit.Error) {
	if tensorfsPath != "" && wheelPath == "" {
		return nil, exit.New(exit.Validation, "--tensorfs-wheel requires --runtime-wheel")
	}
	if wheelPath != "" && runtimeVersion != "" || tensorfsPath != "" && tensorfsVersion != "" {
		return nil, exit.New(exit.Validation, "name a wheel or a version, not both")
	}
	m := u.machines
	row, problem := m.store.RentalRow(id)
	if problem != nil {
		return nil, problem
	}
	if row == nil || row.State != "ready" {
		return nil, exit.New(exit.Conflict, "this rental is not ready; no machine was purchased or changed")
	}
	current, problem := m.store.RuntimeUpdate(id)
	if problem != nil {
		return nil, problem
	}
	var candidate *runtimeUpdateWheel
	var snapshot *scratch.Dir
	if wheelPath != "" {
		candidate, snapshot, problem = freezeRuntimeWheel(m.layout.Tmp, wheelPath)
		if problem != nil {
			return nil, problem
		}
	}
	if snapshot != nil {
		defer snapshot.Release()
	}
	var tensorfs *runtimeUpdateWheel
	var tensorfsSnapshot *scratch.Dir
	if tensorfsPath != "" {
		tensorfs, tensorfsSnapshot, problem = freezeRuntimeWheel(m.layout.Tmp, tensorfsPath)
		if problem != nil {
			return nil, problem
		}
		defer tensorfsSnapshot.Release()
	}
	if current != nil && current.Active() && candidate != nil {
		var saved runtimeUpdateSelection
		if json.Unmarshal(current.Selection, &saved) != nil || !sameUpdateWheel(saved.LocalRuntime, candidate) || !sameUpdateWheel(saved.LocalTensorFS, tensorfs) {
			return nil, exit.New(exit.Conflict, "this rental already has a different frozen Runtime update; resume it without wheel flags")
		}
	}
	if current == nil || !current.Active() {
		var selection json.RawMessage
		if candidate != nil || runtimeVersion != "" || tensorfsVersion != "" {
			selection, _ = json.Marshal(runtimeUpdateSelection{LocalRuntime: candidate, LocalTensorFS: tensorfs,
				RuntimeVersion: runtimeVersion, TensorFSVersion: tensorfsVersion})
		}
		current, problem = m.store.BeginRuntimeUpdate(id, row.ExpectedWorkerBootID, "", selection)
		if problem == nil && snapshot != nil {
			snapshot.Detach()
			if tensorfsSnapshot != nil {
				tensorfsSnapshot.Detach()
			}
		}
	}
	if problem == nil && current.State == "unusable" {
		// The owner resumes the same recorded operation; its error says what to re-check.
		current.State = "reconciling"
		problem = m.store.SaveRuntimeUpdate(*current)
	}
	if problem == nil {
		u.run(*current)
	}
	return current, problem
}

func sameUpdateWheel(a, b *runtimeUpdateWheel) bool {
	if a == nil || b == nil {
		return a == nil && b == nil
	}
	return a.Digest == b.Digest && a.Filename == b.Filename && a.Length == b.Length
}

func (u *rentalRuntimeUpdates) Resume() {
	rows, problem := u.machines.store.ActiveRuntimeUpdates()
	if problem != nil {
		fmt.Fprintln(u.machines.context.Out, problem.Message)
		return
	}
	for _, row := range rows {
		if row.InProgress() {
			u.run(row)
		}
	}
}

func (u *rentalRuntimeUpdates) run(row records.RuntimeUpdate) {
	if _, exists := u.running.LoadOrStore(row.RentalID, true); exists {
		return
	}
	go func() {
		defer func() {
			u.running.Delete(row.RentalID)
			current, problem := u.machines.store.RuntimeUpdate(row.RentalID)
			if problem == nil && current != nil && current.Active() && current.ID != row.ID {
				u.run(*current)
			}
		}()
		m := u.machines
		problem := m.fleet.owner.MaintainRental(m.ctx, row.RentalID, func(ctx context.Context, identity *orchestrator.WorkerConnection) *exit.Error {
			if identity.WorkerBootID != row.BootID {
				return exit.New(exit.Conflict, "the rental's worker boot changed before maintenance")
			}
			if row.RequestID != "" {
				if problem := m.store.AppendEvent(row.RequestID, "machine.runtime_update_attempted", 0, map[string]any{"rental": row.RentalID, "update": row.ID}); problem != nil {
					return problem
				}
			}
			return u.update(ctx, &row, identity)
		})
		if problem != nil {
			row.Error = problem.Message
			switch {
			case row.State != "updating" && row.State != "reconciling":
				row.State = "failed" // nothing was sent to the worker
			case m.ctx.Err() != nil:
				row.State = "reconciling" // the next daemon resumes it
			case problem.ErrName() == "rental.ended":
				row.State = "failed"
			default:
				// The worker may be part-way through: it takes no work until its owner acts.
				row.State = "unusable"
			}
		}
		if problem := m.store.SaveRuntimeUpdate(row); problem != nil {
			fmt.Fprintln(m.context.Out, problem.Message)
		}
		m.fleet.owner.WakeQueue()
	}()
}

func (u *rentalRuntimeUpdates) connectionSelection(ctx context.Context, row records.RuntimeUpdate) (runtimeUpdateSelection, *exit.Error) {
	m := u.machines
	var selection runtimeUpdateSelection
	remote, problem := client(m.fleet.atRental(row.RentalID)).Rental(ctx, row.RentalID)
	if problem != nil {
		return selection, problem
	}
	if !remote.Development {
		return selection, exit.Named(exit.Conflict, "rental.maintenance_unavailable",
			"%s was rented without SSH maintenance and its machine predates in-place Runtime updates; nothing was changed", row.RentalID).
			WithRemedy("rent a replacement on a current image")
	}
	if !remote.Ready() || remote.WorkerBootID != row.BootID {
		return selection, exit.New(exit.Conflict, "this rental has no compatible private maintenance endpoint")
	}
	host, port, err := net.SplitHostPort(remote.SSHAddress)
	number, portErr := strconv.Atoi(port)
	if err != nil || net.ParseIP(host) == nil || portErr != nil || number < 1 || number > 65535 {
		return selection, exit.New(exit.Unavailable, "the rental has no mapped SSH maintenance endpoint")
	}
	key := m.context.Cfg.RentalsSSHPublicKey
	if key == "" {
		key = filepath.Join(m.context.Cfg.Home, "auth", "rental-ssh.pub")
	}
	if strings.HasPrefix(key, "~/") {
		home, err := os.UserHomeDir()
		if err != nil {
			return selection, exit.New(exit.Credential, "cannot resolve rental SSH identity")
		}
		key = filepath.Join(home, strings.TrimPrefix(key, "~/"))
	}
	if !filepath.IsAbs(key) {
		key = filepath.Join(m.context.Cfg.Home, key)
	}
	key = strings.TrimSuffix(key, ".pub")
	if info, err := os.Stat(key); err != nil || !info.Mode().IsRegular() {
		return selection, exit.New(exit.Credential, "the rental SSH private key is unavailable; restore the key matching its creation public key")
	}
	directory := filepath.Join(m.layout.Tmp, "runtime-updates", row.ID)
	if err := os.MkdirAll(directory, 0700); err != nil {
		return selection, exit.Internalf("cannot create update staging: %s", err)
	}
	knownHosts := filepath.Join(m.layout.Rentals, row.RentalID, "ssh-known-hosts")
	if err := os.MkdirAll(filepath.Dir(knownHosts), 0700); err != nil {
		return selection, exit.Internalf("cannot retain the rental SSH host identity: %s", err)
	}
	common := []string{"-oBatchMode=yes", "-oStrictHostKeyChecking=accept-new", "-oUserKnownHostsFile=" + knownHosts, "-oIdentitiesOnly=yes", "-i", key}
	var entropy [16]byte
	if _, err := rand.Read(entropy[:]); err != nil {
		return selection, exit.Internalf("cannot name Runtime update staging: %s", err)
	}
	selection = runtimeUpdateSelection{Directory: directory, Stage: hex.EncodeToString(entropy[:]), Host: "root@" + host}
	selection.SSHArguments = append(append([]string{}, common...), "-p", port, selection.Host)
	selection.SFTPArguments = append(append([]string{}, common...), "-P", port)
	return selection, nil
}

func (u *rentalRuntimeUpdates) transport(ctx context.Context, selection runtimeUpdateSelection, action string) (json.RawMessage, *exit.Error) {
	input, _ := json.Marshal(selection)
	var fields map[string]any
	_ = json.Unmarshal(input, &fields)
	fields["action"] = action
	// The probe is fixed first-party source, kept in its own file so the same
	// strict type check covers code executed on the worker as well as the client.
	fields["probe"] = runtimeUpdateProbe
	fields["stager"] = runtimeUpdateStager
	input, _ = json.Marshal(fields)
	// This helper executes trusted maintenance code, not the package's Python.
	python, problem := hostruntime.EnsurePython(ctx, ">=3.12", "")
	if problem != nil {
		return nil, problem
	}
	command := exec.CommandContext(ctx, "uv", "run", "--isolated", "--no-project", "--no-config", "--python", python.Executable, "--no-python-downloads", "--with", "packaging==26.2", "python", "-I", "-c", runtimeUpdateTransport)
	command.Env = u.machines.context.Cfg.Tool()
	command.Stdin = bytes.NewReader(input) //cozy:stdin-value bounded maintenance metadata, never user package code
	log, err := os.OpenFile(filepath.Join(selection.Directory, "transport.log"), os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0600)
	if err != nil {
		return nil, exit.Internalf("cannot retain update diagnostics: %s", err)
	}
	defer log.Close()
	command.Stderr = log
	answer, err := command.Output()
	if len(answer) > 1<<20 {
		return nil, exit.New(exit.Structural, "Runtime update response exceeded its bound")
	}
	if err != nil {
		_, _ = log.Write(answer)
		var reply struct {
			Error  string `json:"error"`
			Unsent bool   `json:"unsent"`
		}
		if json.Unmarshal(answer, &reply) == nil && reply.Unsent {
			return nil, exit.Named(exit.Failed, "rental.runtime_update_unsent", "the update never reached the worker: %s; diagnostics: %s", reply.Error, log.Name())
		}
		return nil, exit.Named(exit.Failed, "rental.runtime_update_failed", "Runtime update could not %s; diagnostics: %s", action, log.Name())
	}
	if !json.Valid(answer) {
		return nil, exit.New(exit.Structural, "Runtime update returned invalid metadata")
	}
	return answer, nil
}

func (u *rentalRuntimeUpdates) update(ctx context.Context, row *records.RuntimeUpdate, identity *orchestrator.WorkerConnection) *exit.Error {
	var selection runtimeUpdateSelection
	if len(row.Selection) > 0 && json.Unmarshal(row.Selection, &selection) != nil {
		return exit.New(exit.Structural, "recorded Runtime update selection is unreadable")
	}
	if selection.Native != nil || len(selection.Selection) == 0 {
		// A machine that updates itself is asked to; only one that does not is reached over SSH.
		machine, problem := u.maintenance(identity)
		if problem != nil {
			return problem
		}
		native := selection.Native != nil
		if !native {
			state, problem := machine.state(ctx)
			if problem != nil {
				return problem
			}
			if native = state != nil; native {
				if problem := u.updateNative(ctx, row, &selection, machine); problem != nil {
					return problem
				}
			}
		}
		if native {
			return u.followNative(ctx, row, identity, machine)
		}
	}
	if selection.RuntimeVersion != "" || selection.TensorFSVersion != "" {
		return exit.New(exit.Validation, "--runtime-version needs a machine that updates its own Runtime; this one takes --runtime-wheel")
	}
	if len(row.Selection) > 0 {
		if row.State == "updating" || row.State == "reconciling" {
			if row.Error != "" {
				if _, problem := u.transport(ctx, selection, "resume"); unsent(problem) {
					return unchanged(row, selection, problem)
				} else if problem != nil {
					return problem
				}
				row.Error = ""
				if problem := u.machines.store.SaveRuntimeUpdate(*row); problem != nil {
					return problem
				}
			}
			return u.reconcile(ctx, row, identity, selection)
		}
	}
	if len(selection.Selection) == 0 {
		candidate, tensorfs := selection.LocalRuntime, selection.LocalTensorFS
		var problem *exit.Error
		selection, problem = u.connectionSelection(ctx, *row)
		if problem != nil {
			return problem
		}
		selection.LocalRuntime, selection.LocalTensorFS = candidate, tensorfs
		selection.Selection, problem = u.transport(ctx, selection, "plan")
		if problem != nil {
			return problem
		}
		var planned struct {
			Unchanged   bool                `json:"unchanged"`
			Target      runtimeUpdateTarget `json:"target"`
			RuntimeWire *rental.RuntimeWire `json:"runtime_wire"`
		}
		if json.Unmarshal(selection.Selection, &planned) != nil || planned.Target.RuntimeUpdate.Runtime.Version == "" || planned.Target.RuntimeUpdate.TensorFS.Version == "" {
			return exit.New(exit.Structural, "Runtime update did not resolve a verified wheel pair")
		}
		selection.Target = planned.Target
		row.Selection, _ = json.Marshal(selection)
		if planned.Unchanged {
			row.State = "succeeded"
			row.Result = selection.Selection
			return nil
		}
		info, problem := orchestrator.RentalProtocolInfo(ctx, identity)
		if problem != nil {
			return problem
		}
		if problem := rental.RuntimeUpdateHost(planned.Target.RuntimeUpdate.Runtime.Version, planned.RuntimeWire, info.WireMinor); problem != nil {
			return problem
		}
		if problem := u.machines.store.SaveRuntimeUpdate(*row); problem != nil {
			return problem
		}
	}
	control, problem := orchestrator.DialMaintenanceControl(ctx, identity, rental.ClaimProof(u.machines.layout), u.machines.store)
	if problem != nil {
		return problem
	}
	defer control.Close()
	row.State = "updating"
	if problem := u.machines.store.SaveRuntimeUpdate(*row); problem != nil {
		return problem
	}
	_, problem = u.transport(ctx, selection, "apply")
	_ = control.Close()
	if unsent(problem) {
		return unchanged(row, selection, problem)
	}
	// Once apply may have reached the guardian, only its durable operation
	// journal may decide completion. SSH loss is not an update failure.
	row.State = "reconciling"
	row.Error = ""
	if saveProblem := u.machines.store.SaveRuntimeUpdate(*row); saveProblem != nil {
		return saveProblem
	}
	return u.reconcile(ctx, row, identity, selection)

}

func handleRentalUpdate(ctx *Context) *exit.Error {
	adoptRentalHub(ctx, ctx.Inv.Args[0])
	_, store, problem := rentalStores(ctx)
	if problem != nil {
		return problem
	}
	defer store.Close()
	row, problem := store.RentalByMachine(strings.TrimSpace(ctx.Inv.Args[0]))
	if problem != nil {
		return problem
	}
	if row == nil {
		return exit.New(exit.NotFound, "no rental %q on this host", ctx.Inv.Args[0])
	}
	c, problem := dial(ctx)
	if problem != nil {
		return problem
	}
	wheelPath := ctx.Inv.Value("--runtime-wheel")
	if wheelPath != "" {
		var err error
		wheelPath, err = filepath.Abs(wheelPath)
		if err != nil {
			return exit.New(exit.Validation, "cannot resolve local Runtime wheel: %s", err)
		}
	}
	tensorfsPath := ctx.Inv.Value("--tensorfs-wheel")
	if tensorfsPath != "" {
		if wheelPath == "" {
			return exit.New(exit.Validation, "--tensorfs-wheel requires --runtime-wheel")
		}
		var err error
		tensorfsPath, err = filepath.Abs(tensorfsPath)
		if err != nil {
			return exit.New(exit.Validation, "cannot resolve local TensorFS wheel: %s", err)
		}
	}
	result, problem := c.UpdateRentalRuntime(row.ID, api.RuntimeUpdateRequest{RuntimeWheel: wheelPath, TensorFSWheel: tensorfsPath,
		RuntimeVersion: ctx.Inv.Value("--runtime-version"), TensorFSVersion: ctx.Inv.Value("--tensorfs-version")})
	if problem != nil {
		return problem
	}
	lastState := ""
	for result.InProgress() {
		if !ctx.Mode().JSON && lastState != result.State {
			messages := map[string]string{"preparing": "Checking Runtime and published updates", "updating": "Updating Runtime", "reconciling": "Checking the worker after an interrupted update"}
			if message := messages[result.State]; message != "" {
				fmt.Fprintf(ctx.Err, "%s: %s...\n", row.MachineName, message)
			}
			lastState = result.State
		}
		time.Sleep(time.Second)
		result, problem = c.RentalRuntimeUpdate(row.ID)
		if problem != nil {
			return problem
		}
	}
	if result.State == "unusable" {
		return result.Unusable(row.MachineName)
	}
	if result.State == "failed" {
		return exit.Named(exit.Failed, "rental.runtime_update_failed", "%s: %s", row.MachineName, result.Error)
	}
	var actual runtimeObservation
	_ = json.Unmarshal(result.Result, &actual)
	var selection runtimeUpdateSelection
	var previous runtimeObservation
	_ = json.Unmarshal(result.Selection, &selection)
	_ = json.Unmarshal(selection.Selection, &previous)
	var state struct {
		Unchanged bool `json:"unchanged"`
		Update    struct {
			From struct {
				Runtime string `json:"runtime"`
			} `json:"from"`
		} `json:"update"`
	}
	_ = json.Unmarshal(result.Result, &state)
	if previous.Observed.Runtime.Distribution == "" {
		previous.Observed.Runtime.Distribution = state.Update.From.Runtime // the machine updated itself
	}
	status := "ready"
	if state.Unchanged {
		status = "already current"
	}
	return emit(ctx, compactRecord([]output.Field{
		{K: "machine", V: row.MachineName}, {K: "runtime", V: actual.Observed.Runtime.Distribution},
		{K: "tensorfs", V: actual.Observed.TensorFS}, {K: "status", V: status},
		{K: "previous_runtime", V: previous.Observed.Runtime.Distribution}, {K: "update_id", V: result.ID},
		{K: "details", V: result.Result}}, "machine", "runtime", "tensorfs", "status"))
}

type runtimeUpdateStatus struct {
	Update struct {
		Operation      string   `json:"operation"`
		Expected       []string `json:"expected"`
		State          string   `json:"state"`
		Phase          string   `json:"phase"`
		UpdaterRunning *bool    `json:"updater_running"`
		Error          string   `json:"error"`
	} `json:"update"`
}

func (u *rentalRuntimeUpdates) reconcile(ctx context.Context, row *records.RuntimeUpdate, identity *orchestrator.WorkerConnection, selection runtimeUpdateSelection) *exit.Error {
	expected := []string{strings.TrimPrefix(selection.Target.RuntimeUpdate.Runtime.Digest, "sha256:"), strings.TrimPrefix(selection.Target.RuntimeUpdate.TensorFS.Digest, "sha256:")}
	slices.Sort(expected)
	resumed := false
	for ctx.Err() == nil {
		raw, problem := u.transport(ctx, selection, "status")
		if problem != nil {
			// An unreachable worker is asked again only while its rental lasts.
			if rented, readProblem := u.machines.store.RentalRow(row.RentalID); readProblem != nil {
				return readProblem
			} else if rented == nil || rented.State != "ready" {
				return exit.Named(exit.Conflict, "rental.ended", "the rental ended during its Runtime update")
			}
		}
		if problem == nil {
			var status runtimeUpdateStatus
			var actual runtimeObservation
			if json.Unmarshal(raw, &status) != nil || json.Unmarshal(raw, &actual) != nil {
				return exit.New(exit.Structural, "maintenance reconciliation returned invalid operation metadata")
			}
			if status.Update.State == "missing" {
				if resumed {
					return exit.New(exit.Conflict, "worker did not acknowledge the recorded update; maintenance remains closed, retry cozy rental update")
				}
				resumed = true
				// The enqueue response may have been lost before acceptance. Replaying
				// this same immutable stage is idempotent, including while running.
				if _, problem := u.transport(ctx, selection, "apply"); unsent(problem) {
					return unchanged(row, selection, problem)
				} else if problem != nil {
					return problem
				}
			} else {
				slices.Sort(status.Update.Expected)
				if status.Update.Operation != selection.Stage || !slices.Equal(status.Update.Expected, expected) {
					return exit.New(exit.Conflict, "worker update journal does not match this recorded operation; maintenance remains closed")
				}
				switch status.Update.State {
				case "queued", "running":
					if status.Update.UpdaterRunning != nil && !*status.Update.UpdaterRunning && status.Update.Phase != "recovery_required" {
						if resumed {
							return exit.New(exit.Conflict, "the worker update guardian did not start; maintenance remains closed, retry cozy rental update")
						}
						resumed = true
						if _, problem := u.transport(ctx, selection, "resume"); problem != nil {
							return problem
						}
					}
					if status.Update.Phase == "recovery_required" {
						return exit.New(exit.Conflict, "the worker update needs recovery: %s; run cozy rental update again to resume the same recorded operation", status.Update.Error)
					}
				case "succeeded", "rolled_back", "refused":
					if actual.Observed.Updating {
						break
					}
					control, problem := orchestrator.DialMaintenanceControl(ctx, identity, rental.ClaimProof(u.machines.layout), u.machines.store)
					if problem != nil {
						return problem
					}
					_ = control.Close()
					// The worker boot is the same; its Runtime is not. The next call claims
					// it again and reads the capabilities it now has.
					u.machines.machines.Forget(row.RentalID)
					// The worker finished and answered maintenance control, so it is healthy.
					// The pair it reports is recorded as observed; a spelling or version it
					// chose differently never keeps a paid rental closed.
					if status.Update.State == "succeeded" {
						row.State, row.Error = "succeeded", ""
					} else {
						row.State = "failed"
						row.Error = fmt.Sprintf("the update was refused or rolled back; the worker runs Runtime %s / TensorFS %s",
							actual.Observed.Runtime.Distribution, actual.Observed.TensorFS)
						if status.Update.Error != "" {
							row.Error += ": " + status.Update.Error
						}
					}
					row.Result = raw
					return nil
				default:
					return exit.New(exit.Structural, "worker reported an unknown update state; maintenance remains closed")
				}
			}
		}
		select {
		case <-ctx.Done():
			return exit.New(exit.Canceled, "Runtime update observation interrupted")
		case <-time.After(time.Second):
		}
	}
	return exit.New(exit.Canceled, "Runtime update observation interrupted")
}

func unsent(problem *exit.Error) bool {
	return problem != nil && problem.ErrName() == "rental.runtime_update_unsent"
}

// unchanged ends an update whose apply never reached the worker: it still runs, and
// serves, the Runtime it had.
func unchanged(row *records.RuntimeUpdate, selection runtimeUpdateSelection, problem *exit.Error) *exit.Error {
	var before runtimeObservation
	_ = json.Unmarshal(selection.Selection, &before)
	row.State, row.Error = "failed", problem.Message
	if runtime := before.Observed.Runtime.Distribution; runtime != "" {
		row.Error += "; the worker keeps serving Runtime " + runtime
	}
	return nil
}
