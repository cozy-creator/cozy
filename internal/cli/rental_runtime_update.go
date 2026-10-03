package cli

import (
	"context"
	"encoding/json"
	"fmt"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"github.com/cozy-creator/cozy/internal/api"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/orchestrator"
	"github.com/cozy-creator/cozy/internal/output"
	"github.com/cozy-creator/cozy/internal/records"
	"github.com/cozy-creator/cozy/internal/scratch"
)

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

type runtimeUpdateSelection struct {
	LocalRuntime  *runtimeUpdateWheel `json:"local_runtime,omitempty"`
	LocalTensorFS *runtimeUpdateWheel `json:"local_tensorfs,omitempty"`
	// Published versions to install instead of local wheels; the machine fetches them.
	RuntimeVersion  string `json:"runtime_version,omitempty"`
	TensorFSVersion string `json:"tensorfs_version,omitempty"`
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
			if m.ctx.Err() != nil && row.State != "preparing" {
				row.State = "reconciling" // the next daemon attaches to the same update
			} else {
				// The machine refused it, or rolled it back: it serves its previous software.
				row.State = "failed"
			}
		}
		if problem := m.store.SaveRuntimeUpdate(row); problem != nil {
			fmt.Fprintln(m.context.Out, problem.Message)
		}
		m.fleet.owner.WakeQueue()
	}()
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
	var previous runtimeObservation
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
