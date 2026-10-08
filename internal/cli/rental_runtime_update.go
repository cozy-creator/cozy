package cli

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"github.com/cozy-creator/cozy/internal/api"
	localapi "github.com/cozy-creator/cozy/internal/client"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/hub"
	"github.com/cozy-creator/cozy/internal/machines"
	"github.com/cozy-creator/cozy/internal/machinev1"
	"github.com/cozy-creator/cozy/internal/orchestrator"
	"github.com/cozy-creator/cozy/internal/output"
	"github.com/cozy-creator/cozy/internal/records"
	"github.com/cozy-creator/cozy/internal/scratch"
	"github.com/cozy-creator/cozy/internal/workertls"
	pb "github.com/cozy-creator/cozy/protocol/cozy/machine/v1"
)

// rentalRuntimeUpdates changes rentals' software with Run kind: update. Each rental's latest
// update is recorded against its boot; until it ends it holds the rental, so work waits and
// lands on the new software. The machine waits for idleness and rolls back a candidate that
// never proves ready, so an update ends succeeded or failed and the rental serves either way.
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

// runtimeUpdateSelection is what an update installs: local wheels or exact published
// versions. Target is a boot's own follow: the versions the rental's Hub names when the update
// runs. Nothing named installs nothing.
type runtimeUpdateSelection struct {
	LocalRuntime    *runtimeUpdateWheel `json:"local_runtime,omitempty"`
	LocalTensorFS   *runtimeUpdateWheel `json:"local_tensorfs,omitempty"`
	RuntimeVersion  string              `json:"runtime_version,omitempty"`
	TensorFSVersion string              `json:"tensorfs_version,omitempty"`
	Target          bool                `json:"target,omitempty"`
}

type softwarePair struct {
	Runtime  string `json:"runtime"`
	TensorFS string `json:"tensorfs"`
}

// runtimeUpdateResult is a finished update's record. Unchanged: nothing was installed, and
// Note says why.
type runtimeUpdateResult struct {
	From      softwarePair `json:"from"`
	To        softwarePair `json:"to"`
	Unchanged bool         `json:"unchanged,omitempty"`
	Note      string       `json:"note,omitempty"`
}

// Boot brings a rental's newly attached worker boot to the Hub's target software before it
// takes work. Only a new boot is followed, so an explicit update stands until the pod boots
// again.
func (u *rentalRuntimeUpdates) Boot(id string) {
	if _, problem := u.follow(id); problem != nil {
		fmt.Fprintf(u.machines.context.Out, "rental %s keeps its software: %s\n", id, problem.Message)
	}
}

// follow starts (or joins) the update of the rental's boot to its Hub's target software.
func (u *rentalRuntimeUpdates) follow(id string) (*records.RuntimeUpdate, *exit.Error) {
	return u.begin(id, api.RuntimeUpdateRequest{}, true)
}

// Start is an update a client asked for. It installs only what it names: an empty request,
// as a client older than the Hub's target sends, changes nothing.
func (u *rentalRuntimeUpdates) Start(id string, options api.RuntimeUpdateRequest) (*records.RuntimeUpdate, *exit.Error) {
	return u.begin(id, options, false)
}

func (u *rentalRuntimeUpdates) begin(id string, options api.RuntimeUpdateRequest, target bool) (*records.RuntimeUpdate, *exit.Error) {
	if options.TensorFSWheel != "" && options.RuntimeWheel == "" {
		return nil, exit.New(exit.Validation, "--tensorfs-wheel requires --runtime-wheel")
	}
	if options.RuntimeWheel != "" && options.RuntimeVersion != "" || options.TensorFSWheel != "" && options.TensorFSVersion != "" {
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
	if current != nil && current.Active() && current.BootID == row.ExpectedWorkerBootID {
		if options != (api.RuntimeUpdateRequest{}) || target && !strings.Contains(string(current.Selection), `"target":true`) {
			return nil, exit.Named(exit.Unavailable, "rental.maintenance", "this rental is already updating its software; wait for it to end")
		}
		u.run(*current) // the update in progress is the one asked for
		return current, nil
	}
	selection := runtimeUpdateSelection{RuntimeVersion: options.RuntimeVersion, TensorFSVersion: options.TensorFSVersion, Target: target}
	var snapshots []*scratch.Dir
	defer func() {
		for _, snapshot := range snapshots {
			snapshot.Release()
		}
	}()
	for _, local := range []struct {
		path string
		into **runtimeUpdateWheel
	}{{options.RuntimeWheel, &selection.LocalRuntime}, {options.TensorFSWheel, &selection.LocalTensorFS}} {
		if local.path == "" {
			continue
		}
		wheel, snapshot, problem := freezeRuntimeWheel(m.layout.Tmp, local.path)
		if problem != nil {
			return nil, problem
		}
		*local.into, snapshots = wheel, append(snapshots, snapshot)
	}
	raw, _ := json.Marshal(selection)
	current, problem = m.store.BeginRuntimeUpdate(id, row.ExpectedWorkerBootID, raw)
	if problem != nil {
		return nil, problem
	}
	for _, snapshot := range snapshots {
		snapshot.Detach() // the record owns the frozen wheels now
	}
	snapshots = nil
	u.run(*current)
	return current, nil
}

func (u *rentalRuntimeUpdates) Resume() {
	rows, problem := u.machines.store.ActiveRuntimeUpdates()
	if problem != nil {
		fmt.Fprintln(u.machines.context.Out, problem.Message)
		return
	}
	for _, row := range rows {
		u.run(row)
	}
}

func (u *rentalRuntimeUpdates) run(row records.RuntimeUpdate) {
	if _, exists := u.running.LoadOrStore(row.RentalID, true); exists {
		return
	}
	go func() {
		m := u.machines
		defer func() {
			u.running.Delete(row.RentalID)
			current, problem := m.store.RuntimeUpdate(row.RentalID)
			if problem == nil && current != nil && current.Active() && current.ID != row.ID {
				u.run(*current)
			}
		}()
		problem := m.fleet.owner.MaintainRental(m.ctx, row.RentalID, func(ctx context.Context, identity *orchestrator.WorkerConnection) *exit.Error {
			if identity.WorkerBootID != row.BootID {
				return exit.New(exit.Conflict, "the rental's worker boot changed before its update")
			}
			return u.update(ctx, &row, identity)
		})
		if problem != nil {
			row.Error = problem.Message
			if m.ctx.Err() != nil && row.State != "preparing" {
				row.State = "reconciling" // the next daemon attaches to the same update
			} else {
				row.State = "failed" // refused or rolled back: it serves its previous software
			}
		}
		if problem := m.store.SaveRuntimeUpdate(row); problem != nil {
			fmt.Fprintln(m.context.Out, problem.Message)
		}
		m.fleet.owner.WakeQueue()
		if m.fleet.installs != nil {
			m.fleet.installs.Wake()
		}
	}()
}

// update sends the recorded selection to the rental's machine as Run kind: update and follows
// it to its outcome. The run's id is this update's id, so a daemon that resumes the update
// attaches to the same run; local wheels go up first with Write. Published versions the
// machine already runs are not installed again.
func (u *rentalRuntimeUpdates) update(ctx context.Context, row *records.RuntimeUpdate, identity *orchestrator.WorkerConnection) *exit.Error {
	var selection runtimeUpdateSelection
	if len(row.Selection) > 0 && json.Unmarshal(row.Selection, &selection) != nil {
		return exit.New(exit.Structural, "recorded software update selection is unreadable")
	}
	if selection == (runtimeUpdateSelection{}) {
		row.State, row.Error = "succeeded", ""
		row.Result, _ = json.Marshal(runtimeUpdateResult{Unchanged: true, Note: "the update named no software"})
		return nil
	}
	pin, err := workertls.LoadPin(identity.CACert)
	if err != nil {
		return exit.New(exit.Credential, "the machine's TLS identity cannot be read")
	}
	key, problem := u.machines.machines.RentalKey(identity.RentalID)
	if problem != nil {
		return problem
	}
	machine, err := machinev1.Dial(identity.Addr, pin.TLSConfig(), identity.WorkerID, key.Signer())
	if err != nil {
		return machines.Transport(err)
	}
	defer machine.Close()
	frame, err := machine.Status(ctx)
	if err != nil {
		return machines.Transport(err)
	}
	result := runtimeUpdateResult{From: softwarePair{frame.GetRuntime(), frame.GetTensorfs()}}
	finish := func(note string) *exit.Error {
		result.To, result.Unchanged, result.Note = result.From, true, note
		row.State, row.Error = "succeeded", ""
		row.Result, _ = json.Marshal(result)
		return nil
	}
	cohort, problem := runtimeUpdateCohort(ctx, machine, selection, "bundled")
	if problem != nil {
		return problem
	}
	if selection.Target {
		target, problem := client(u.machines.fleet.atRental(row.RentalID)).Software(ctx)
		if problem != nil {
			return problem
		}
		if target.Runtime == "" {
			return finish("the Hub names no target software")
		}
		cohort.Runtime, cohort.TensorFS = &machinev1.Member{Version: target.Runtime}, &machinev1.Member{Version: target.TensorFS}
		// Recorded as exact versions: a daemon that resumes this update installs what was resolved.
		selection.RuntimeVersion, selection.TensorFSVersion = target.Runtime, target.TensorFS
		row.Selection, _ = json.Marshal(selection)
	}
	if published(cohort.Runtime, result.From.Runtime) && published(cohort.TensorFS, result.From.TensorFS) {
		return finish("it already runs this software")
	}
	row.State = "updating"
	if problem := u.machines.store.SaveRuntimeUpdate(*row); problem != nil {
		return problem
	}
	var saveProblem *exit.Error
	outcome, err := machine.Update(ctx, row.ID, cohort, func(event *pb.RunEvent) {
		if event.GetProgress().GetStage() == "waiting_activation" && row.State != "waiting_activation" {
			row.State = "waiting_activation"
			saveProblem = u.machines.store.SaveRuntimeUpdate(*row)
		}
	})
	if err != nil {
		return machines.Transport(err)
	}
	if saveProblem != nil {
		return saveProblem
	}
	var outcomeResult struct{ From, To softwarePair }
	_ = json.Unmarshal(outcome.GetResult(), &outcomeResult)
	result.To = outcomeResult.To
	row.State, row.Error = "succeeded", ""
	if outcome.GetStatus() != "succeeded" {
		result.To = result.From
		row.State, row.Error = "failed", outcome.GetReason().GetMessage()
		if row.Error == "" {
			row.Error = outcome.GetReason().GetCode()
		}
	}
	row.Result, _ = json.Marshal(result)
	return nil
}

// runtimeUpdateCohort is shared by rental maintenance and explicit owned endpoints.
// A missing member remains nil: the machine preserves its current wheel.
func runtimeUpdateCohort(ctx context.Context, machine *machinev1.Client, selection runtimeUpdateSelection, agent string) (machinev1.Cohort, *exit.Error) {
	member := func(local *runtimeUpdateWheel, version string) (*machinev1.Member, *exit.Error) {
		if local == nil {
			if version == "" {
				return nil, nil
			}
			return &machinev1.Member{Version: version}, nil
		}
		file, err := os.Open(local.Path)
		if err != nil {
			return nil, exit.New(exit.Conflict, "the frozen update wheel %s is gone: %s", local.Filename, err)
		}
		defer file.Close()
		if err := machine.Write(ctx, local.Digest, uint64(local.Length), file); err != nil {
			return nil, machines.Transport(err)
		}
		return &machinev1.Member{Wheel: local.Filename, Digest: local.Digest, Length: uint64(local.Length)}, nil
	}
	cohort := machinev1.Cohort{Agent: agent}
	var problem *exit.Error
	if cohort.Runtime, problem = member(selection.LocalRuntime, selection.RuntimeVersion); problem != nil {
		return cohort, problem
	}
	cohort.TensorFS, problem = member(selection.LocalTensorFS, selection.TensorFSVersion)
	return cohort, problem
}

// published answers whether a cohort member leaves the running version as it is.
func published(member *machinev1.Member, running string) bool {
	return member == nil || member.Wheel == "" && member.Version == running
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
	request := api.RuntimeUpdateRequest{RuntimeVersion: ctx.Inv.Value("--runtime-version"), TensorFSVersion: ctx.Inv.Value("--tensorfs-version")}
	for flag, into := range map[string]*string{"--runtime-wheel": &request.RuntimeWheel, "--tensorfs-wheel": &request.TensorFSWheel} {
		if path := ctx.Inv.Value(flag); path != "" {
			absolute, err := filepath.Abs(path)
			if err != nil {
				return exit.New(exit.Validation, "cannot resolve %s: %s", path, err)
			}
			*into = absolute
		}
	}
	if request == (api.RuntimeUpdateRequest{}) {
		// Nothing named is the Hub's target, named here: a daemon older than this cozy reads an
		// empty request as the newest published pair.
		hctx, cancel := hub.Context()
		target, problem := client(ctx).Software(hctx)
		cancel()
		if problem != nil {
			return problem
		}
		if target.Runtime == "" {
			return exit.Named(exit.Validation, "rental.no_target_software", "the Hub names no target software for %s", row.MachineName).
				WithRemedy("name it: cozy rental update %s --runtime-version <version> --tensorfs-version <version>", row.MachineName)
		}
		request.RuntimeVersion, request.TensorFSVersion = target.Runtime, target.TensorFS
	}
	c, problem := dial(ctx)
	if problem != nil {
		return problem
	}
	update, problem := c.UpdateRentalRuntime(row.ID, request)
	if problem == nil {
		update, problem = awaitRentalUpdate(ctx, c, row.MachineName, update)
	}
	if problem != nil {
		return problem
	}
	if update.State == "failed" {
		return exit.Named(exit.Failed, "rental.runtime_update_failed", "%s: %s", row.MachineName, update.Error)
	}
	var result runtimeUpdateResult
	_ = json.Unmarshal(update.Result, &result)
	status := "ready"
	if result.Unchanged {
		status = "unchanged: " + result.Note
	}
	return emit(ctx, compactRecord([]output.Field{
		{K: "machine", V: row.MachineName}, {K: "runtime", V: result.To.Runtime}, {K: "tensorfs", V: result.To.TensorFS},
		{K: "status", V: status}, {K: "previous_runtime", V: result.From.Runtime}, {K: "update_id", V: update.ID},
	}, "machine", "runtime", "tensorfs", "status"))
}

// awaitRentalUpdate follows a rental's software update to its end.
func awaitRentalUpdate(ctx *Context, c *localapi.Client, machine string, update api.RuntimeUpdate) (api.RuntimeUpdate, *exit.Error) {
	said := ""
	for update.Active() {
		if !ctx.Mode().JSON && said != update.State {
			messages := map[string]string{"preparing": "checking its software", "updating": "updating its software",
				"waiting_activation": "waiting for its work to finish", "reconciling": "following an interrupted update"}
			if message := messages[update.State]; message != "" {
				fmt.Fprintf(ctx.Err, "%s: %s...\n", machine, message)
			}
			said = update.State
		}
		time.Sleep(time.Second)
		var problem *exit.Error
		if update, problem = c.RentalRuntimeUpdate(update.RentalID); problem != nil {
			return update, problem
		}
	}
	return update, nil
}

// bootSoftware is a new rental's software once the Hub's target is in place, for `rental new`
// to show: the fields, or the note saying why it kept its image's.
// The versions are named, never left empty: a daemon older than this cozy reads an empty
// request as the newest published pair, and a Hub naming no target asks for nothing.
func bootSoftware(ctx *Context, rentalID, machine string) ([]output.Field, string) {
	hctx, cancel := hub.Context()
	target, problem := client(ctx).Software(hctx)
	cancel()
	if problem != nil {
		return nil, "it keeps its image's software: the Hub's target could not be read: " + problem.Message
	}
	if target.Runtime == "" {
		return nil, ""
	}
	c, problem := dial(ctx)
	if problem != nil {
		return nil, "it keeps its image's software: " + problem.Message
	}
	update, problem := c.UpdateRentalRuntime(rentalID, api.RuntimeUpdateRequest{RuntimeVersion: target.Runtime, TensorFSVersion: target.TensorFS})
	if problem == nil {
		update, problem = awaitRentalUpdate(ctx, c, machine, update)
	}
	switch {
	case problem != nil:
		return nil, "it keeps its image's software: " + problem.Message
	case update.State == "failed":
		return nil, "it keeps its image's software: " + update.Error
	}
	var result runtimeUpdateResult
	_ = json.Unmarshal(update.Result, &result)
	return []output.Field{{K: "runtime", V: result.To.Runtime}, {K: "tensorfs", V: result.To.TensorFS}}, ""
}
