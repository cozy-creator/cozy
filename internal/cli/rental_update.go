package cli

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"regexp"
	"slices"
	"strings"
	"time"

	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/hub"
	"github.com/cozy-creator/cozy/internal/machines"
	"github.com/cozy-creator/cozy/internal/machinev1"
	"github.com/cozy-creator/cozy/internal/output"
	"github.com/cozy-creator/cozy/internal/records"
	"github.com/cozy-creator/cozy/internal/wheel"
	v1 "github.com/cozy-creator/cozy/protocol/cozy/machine/v1"
)

// A rental's software update is an operation like any other (records.SoftwareUpdate): a
// numbered run its machine takes as Run kind: update, listed, watched and canceled as runs
// are. The machine stages the pair, waits for the work it admitted, restarts on the pair in
// place and rolls back a candidate that never proves ready; a cancel ends it before it
// activates. The rental's unsent work waits for it (stepV1), so it lands on the new software.

type softwarePair struct {
	Runtime  string `json:"runtime"`
	TensorFS string `json:"tensorfs"`
}

// softwareResult is a finished update's result; From equal to To is unchanged.
type softwareResult struct{ From, To softwarePair }

var wheelName = regexp.MustCompile(`^[A-Za-z0-9_.+-]+\.whl$`)

// measureWheel is a local wheel as an update names it, read where it is: it is written to the
// machine as measured, or the machine refuses it.
func measureWheel(source string) (*records.Wheel, *exit.Error) {
	const limit = 128 << 20
	path, err := filepath.Abs(source)
	if err != nil || !wheelName.MatchString(filepath.Base(path)) {
		return nil, exit.New(exit.Validation, "an update wheel is a local .whl file, not %s", source)
	}
	file, err := os.Open(path)
	if err != nil {
		return nil, exit.New(exit.Validation, "cannot read update wheel: %s", err)
	}
	defer file.Close()
	info, err := file.Stat()
	if err != nil || !info.Mode().IsRegular() || info.Size() <= 0 || info.Size() > limit {
		return nil, exit.New(exit.Validation, "an update wheel is a regular file of at most 128 MiB")
	}
	digest := sha256.New()
	if length, err := io.Copy(digest, file); err != nil || length != info.Size() {
		return nil, exit.New(exit.Validation, "update wheel %s changed while it was measured", path)
	}
	return &records.Wheel{Path: path, Digest: "sha256:" + hex.EncodeToString(digest.Sum(nil)), Length: info.Size()}, nil
}

// runtimeUpdateCohort is the software an update sends: each local wheel written to the machine
// first, each version named; a member left out keeps the machine's own.
func runtimeUpdateCohort(ctx context.Context, machine *machinev1.Client, software records.SoftwareUpdate, agent string) (machinev1.Cohort, *exit.Error) {
	member := func(local *records.Wheel, version string) (*machinev1.Member, *exit.Error) {
		if local == nil {
			if version == "" {
				return nil, nil
			}
			return &machinev1.Member{Version: version}, nil
		}
		file, err := os.Open(local.Path)
		if err != nil {
			return nil, exit.New(exit.Conflict, "the update wheel %s is gone: %s", local.Path, err)
		}
		defer file.Close()
		if err := machine.Write(ctx, local.Digest, uint64(local.Length), file); err != nil {
			return nil, machines.Transport(err)
		}
		return &machinev1.Member{Wheel: filepath.Base(local.Path), Digest: local.Digest, Length: uint64(local.Length)}, nil
	}
	cohort := machinev1.Cohort{Agent: agent}
	var problem *exit.Error
	if cohort.Runtime, problem = member(software.RuntimeWheel, software.RuntimeVersion); problem != nil {
		return cohort, problem
	}
	cohort.TensorFS, problem = member(software.TensorFSWheel, software.TensorFSVersion)
	return cohort, problem
}

// updateSpecV1 is a software update as its machine takes it. Target is read from the rental's
// Hub now: with none, or unpinned and older than the machine runs, the machine is sent its own
// pair, which it answers unchanged.
func (m *machineRuns) updateSpecV1(ctx context.Context, machine *machines.V1, frame *v1.StatusFrame, software records.SoftwareUpdate) (*v1.RunSpec, *exit.Error) {
	if !slices.Contains(frame.GetCapabilities(), "update/1") {
		return nil, exit.Named(exit.Structural, "machine.operation_unsupported", "%s cannot update its software in place", machine.Name)
	}
	if software.Target {
		target, problem := client(m.fleet.atRental(machine.Name)).Software(ctx)
		if problem != nil {
			return nil, problem
		}
		software = records.SoftwareUpdate{RuntimeVersion: frame.GetRuntime(), TensorFSVersion: frame.GetTensorfs()}
		if target.Runtime != "" && (target.Pinned || !wheel.NewerVersion(frame.GetRuntime(), target.Runtime)) {
			software.RuntimeVersion, software.TensorFSVersion = target.Runtime, target.TensorFS
		}
	}
	cohort, problem := runtimeUpdateCohort(ctx, machine.Client, software, "bundled")
	if problem != nil {
		return nil, problem
	}
	spec, err := cohort.Spec()
	if err != nil {
		return nil, exit.New(exit.Validation, "%s", err)
	}
	return spec, nil
}

// followTarget asks a rental to take its Hub's target software: a new boot does, before its
// work (fleet.boot). The same ask while one is unsettled is that run.
func (m *machineRuns) followTarget(rental string) (records.Request, *exit.Error) {
	if m.submit == nil {
		return records.Request{}, exit.Unavailablef("Calcifer is still starting")
	}
	request, _, problem := m.submit(rental, records.Operation{Software: &records.SoftwareUpdate{Target: true}})
	return request, problem
}

// followTargetV1 updates a machine to the Hub's target software and answers "" once it runs
// other software, else the next step its owner takes.
func (m *machineRuns) followTargetV1(machine, origin string) string {
	if machines.IsLocal(machine) {
		changed, kept := m.machines.Host.FollowTarget(m.ctx, client(m.context.forHub(origin)))
		if changed {
			return ""
		}
		return either(kept, "this computer's machine has no installed software to update") +
			"; `cozy machine install` takes the Hub's target software, or name newer wheels with --runtime-wheel/--tensorfs-wheel"
	}
	name := machine
	if row, _ := m.store.RentalRow(machine); row != nil && row.MachineName != "" {
		name = row.MachineName
	}
	update, problem := m.followTarget(machine)
	for problem == nil && !records.Settled(update.State) && m.ctx.Err() == nil {
		select {
		case <-m.ctx.Done():
		case <-time.After(time.Second):
		}
		var row *records.Request
		if row, problem = m.store.RequestByReference(update.ID); problem == nil && row == nil {
			problem = exit.New(exit.Conflict, "its update run is gone")
		} else if problem == nil {
			update = *row
		}
	}
	var result softwareResult
	if link, _ := m.store.MachineExecution(update.ID); link != nil {
		if outcome := records.RunV1Outcome(link); outcome != nil {
			_ = json.Unmarshal(outcome.Result, &result)
		}
	}
	switch {
	case problem != nil:
		return fmt.Sprintf("it could not take the Hub's target software (%s); `cozy rental update %s` retries", problem.Message, name)
	case update.State != "succeeded":
		return fmt.Sprintf("its update to the Hub's target software (run %d) ended %s; `cozy rental update %s` retries, or `cozy rental end %s` and rent a new machine",
			update.Number, update.State, name, name)
	case result.From != result.To:
		return ""
	}
	return fmt.Sprintf("%s runs Runtime %s (it already runs this software, the Hub's target); update it to a release that serves this run: `cozy rental update %s --runtime-version <version> --tensorfs-version <version>`",
		name, result.From.Runtime, name)
}

func handleRentalUpdate(ctx *Context) *exit.Error {
	_, store, problem := rentalStores(ctx)
	if problem != nil {
		return problem
	}
	defer store.Close()
	software := records.SoftwareUpdate{RuntimeVersion: ctx.Inv.Value("--runtime-version"), TensorFSVersion: ctx.Inv.Value("--tensorfs-version")}
	for flag, into := range map[string]**records.Wheel{"--runtime-wheel": &software.RuntimeWheel, "--tensorfs-wheel": &software.TensorFSWheel} {
		if path := ctx.Inv.Value(flag); path != "" {
			if *into, problem = measureWheel(path); problem != nil {
				return problem
			}
		}
	}
	name := strings.TrimSpace(ctx.Inv.Args[0])
	live, problem := resolveLiveRental(ctx, store, name)
	if problem != nil {
		return problem
	}
	ctx = ctx.forHub(live.Hub)
	if software == (records.SoftwareUpdate{}) {
		// Nothing named is the Hub's target, named exactly.
		call, cancel := hub.Context()
		target, problem := client(ctx).Software(call)
		cancel()
		if problem != nil {
			return problem
		}
		if target.Runtime == "" {
			return exit.Named(exit.Validation, "rental.no_target_software", "the Hub names no target software for %s", name).
				WithRemedy("name it: cozy rental update %s --runtime-version <version> --tensorfs-version <version>", name)
		}
		software.RuntimeVersion, software.TensorFSVersion = target.Runtime, target.TensorFS
	}
	return runOperation(ctx, name, records.Operation{Software: &software}, !ctx.Inv.Bool("--no-wait"))
}

// bootSoftware is a new rental's software once the Hub's pin is in place, for `rental new` to
// show: the boot's update run, followed, or the note saying why it kept its image's software.
// Unpinned, it runs its image's.
func bootSoftware(ctx *Context, rentalID string) ([]output.Field, string) {
	call, cancel := hub.Context()
	target, problem := client(ctx).Software(call)
	cancel()
	if problem != nil {
		return nil, "it keeps its image's software: the Hub's target could not be read: " + problem.Message
	}
	if !target.Pinned {
		return nil, ""
	}
	c, problem := dial(ctx)
	if problem != nil {
		return nil, "it keeps its image's software: " + problem.Message
	}
	handle, problem := c.SubmitOperation(rentalID, records.Operation{Software: &records.SoftwareUpdate{Target: true}})
	if problem != nil {
		return nil, "it keeps its image's software: " + problem.Message
	}
	life, detached, problem := followOperation(ctx, c, handle)
	switch {
	case problem != nil:
		return nil, "it keeps its image's software: " + problem.Message
	case detached || life.Status != "completed":
		return nil, fmt.Sprintf("it keeps its image's software: its update, run %d, is %s", life.Number, life.Status)
	}
	var result softwareResult
	raw, _ := json.Marshal(life.Result)
	_ = json.Unmarshal(raw, &result)
	return []output.Field{{K: "runtime", V: result.To.Runtime}, {K: "tensorfs", V: result.To.TensorFS}}, ""
}
