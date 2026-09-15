package cli

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"

	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/hub"
	"github.com/cozy-creator/cozy/internal/launch"
	"github.com/cozy-creator/cozy/internal/packagepublish"
	"github.com/cozy-creator/cozy/internal/records"
	"github.com/cozy-creator/cozy/internal/rental"
	"github.com/cozy-creator/cozy/internal/wheel"
)

type runtimeObservation struct {
	Observed struct {
		Updating bool `json:"update_in_progress"`
		Durable  bool `json:"durable_updates"`
		Runtime  struct {
			Distribution string `json:"distribution"`
			Guarded      bool   `json:"supports_guarded_restart"`
		} `json:"runtime"`
		TensorFS string `json:"tensorfs"`
		Python   string `json:"python"`
	} `json:"observed"`
	Target       hub.RuntimeUpdateTarget `json:"-"`
	FreshRequest string                  `json:"-"`
}

func (u *rentalRuntimeUpdates) preflight(ctx context.Context, request records.Request, machine string) *exit.Error {
	if machine == "local" {
		return nil
	}
	current, problem := u.machines.store.RuntimeUpdate(machine)
	if problem != nil {
		return problem
	}
	if current != nil && current.Active() {
		return exit.Named(exit.Unavailable, "rental.maintenance", "this rental is updating its Runtime; this request remains queued")
	}
	row, problem := u.machines.store.RentalRow(machine)
	if problem != nil {
		return problem
	}
	if row == nil {
		return exit.New(exit.NotFound, "the requested rental is no longer attached")
	}
	if !records.RentalReadyState(row.State) {
		return nil // The existing rental lifecycle owns recovery and replacement.
	}
	if cached, ok := u.observed.Load(machine); ok && cached.(runtimeObservation).Target.WorkerBootID != row.ExpectedWorkerBootID {
		u.observed.Delete(machine)
	}
	var observed runtimeObservation
	if cached, ok := u.observed.Load(machine); ok {
		observed = cached.(runtimeObservation)
	} else {
		row, problem := u.machines.store.RentalRow(machine)
		if problem != nil {
			return problem
		}
		if row == nil {
			return exit.New(exit.NotFound, "the requested rental is no longer attached")
		}
		remote, problem := client(u.machines.context).Rental(ctx, machine)
		if problem != nil {
			// This is an advance observation, not a new availability dependency.
			// Older/offline Hubs may lack these facts; Runtime still checks the
			// actual installed dependency boundary before package import.
			if problem.Code == exit.NotFound || problem.Code == exit.Unavailable || problem.Code == exit.Deadline {
				return nil
			}
			return problem
		}
		if !remote.Ready() {
			return nil
		}
		if !remote.Development {
			raw, problem := client(u.machines.context).RentalImageInventory(ctx, machine)
			if problem != nil {
				if problem.Code == exit.NotFound || problem.Code == exit.Unavailable || problem.Code == exit.Deadline {
					return nil
				}
				return problem
			}
			inventory, err := rental.ImageInventory(raw)
			if err != nil {
				return exit.New(exit.Structural, "the immutable worker image inventory is invalid")
			}
			observed.Observed.Python = inventory.Python
			for _, distribution := range inventory.Distributions {
				if distribution.Distribution == "cozy-runtime" { //cozy:allow distribution metadata, not a binary invocation
					observed.Observed.Runtime.Distribution = distribution.Version
				}
				if distribution.Distribution == "tensorfs" {
					observed.Observed.TensorFS = distribution.Version
				}
			}
			observed.Target.WorkerBootID = row.ExpectedWorkerBootID
		} else {
			selection, problem := u.connectionSelection(ctx, records.RuntimeUpdate{ID: "probe-" + machine, RentalID: machine, BootID: row.ExpectedWorkerBootID})
			if problem != nil {
				// Maintenance access is optional for ordinary execution. Runtime
				// still validates its actual dependency boundary before import.
				return nil
			}
			raw, problem := u.transport(ctx, selection, "inspect")
			if problem != nil {
				return nil
			}
			if json.Unmarshal(raw, &observed) == nil && observed.Observed.Updating {
				return exit.Named(exit.Unavailable, "rental.maintenance", "this rental is applying a Runtime update; preparation will resume afterward")
			}
			if json.Unmarshal(raw, &observed) != nil || observed.Observed.Runtime.Distribution == "" || observed.Observed.Python == "" {
				return exit.New(exit.Structural, "worker did not report its actual Runtime/interpreter versions")
			}
			observed.Target.WorkerBootID = row.ExpectedWorkerBootID
		}

		observed.FreshRequest = request.ID
		u.observed.Store(machine, observed)
	}
	requirements, problem := u.requirements(ctx, request, observed.Observed.Python)
	if problem != nil {
		return problem
	}
	mismatch := launch.RuntimeRequirementMismatch(requirements, observed.Observed.Runtime.Distribution, observed.Observed.TensorFS)
	if mismatch == nil {
		return nil
	}
	if observed.FreshRequest != request.ID {
		u.observed.Delete(machine)
		return u.preflight(ctx, request, machine)
	}
	message := fmt.Sprintf("%s has %s %s; %s@%s requires %s", row.MachineName, mismatch.Distribution, mismatch.Installed, request.Package, request.Release, mismatch.Required)
	if !observed.Observed.Runtime.Guarded || !observed.Observed.Durable {
		return exit.Named(exit.Structural, "rental.runtime_incompatible", "%s", message+". This older image cannot update safely; use a current private worker image")
	}
	target, problem := client(u.machines.context).RentalRuntimeUpdateTarget(ctx, machine)
	if problem != nil {
		return exit.Named(exit.Structural, "rental.runtime_incompatible", "%s. No approved compatible update is available: %s", message, problem.Message)
	}
	if target.WorkerBootID != row.ExpectedWorkerBootID {
		return exit.New(exit.Conflict, "approved update refers to another worker boot")
	}
	observed.Target = target
	if launch.RuntimeRequirementMismatch(requirements, observed.Target.RuntimeUpdate.Runtime.Version, observed.Target.RuntimeUpdate.TensorFS.Version) != nil {
		return exit.Named(exit.Structural, "rental.runtime_incompatible", "%s", message+". The approved worker update does not satisfy this package")
	}
	current, problem = u.machines.store.RuntimeUpdate(machine)
	if problem != nil {
		return problem
	}
	if current != nil && current.Active() {
		return exit.Named(exit.Unavailable, "rental.maintenance", "%s", message+"; waiting for this rental's Runtime update")
	}
	attempted, problem := u.machines.store.RuntimeUpdateAttempted(request.ID)
	if problem != nil {
		return problem
	}
	if attempted {
		return exit.Named(exit.Structural, "rental.runtime_incompatible", "%s", message+" after its automatic update attempt; inspect `cozy rental update "+row.MachineName+"`")
	}
	if _, problem := u.startForRequest(machine, request.ID); problem != nil {
		return problem
	}
	return exit.Named(exit.Unavailable, "rental.maintenance", "%s", message+"; applying one compatible update before retrying this request")
}

func (u *rentalRuntimeUpdates) requirements(ctx context.Context, request records.Request, python string) ([]string, *exit.Error) {
	key := request.Package + "@" + request.Release + "#" + request.LocalPackageDigest + "|" + python
	if cached, ok := u.requirementFacts.Load(key); ok {
		return cached.([]string), nil
	}
	result, problem := u.readRequirements(ctx, request, python)
	if problem == nil {
		result, problem = packagepublish.EvaluateRequirements(ctx, result, python)
	}
	if problem == nil {
		u.requirementFacts.Store(key, result)
	}
	return result, problem
}

func (u *rentalRuntimeUpdates) readRequirements(ctx context.Context, request records.Request, python string) ([]string, *exit.Error) {
	if request.LocalPackageDigest == "" {
		ref, problem := hub.ParseRef(request.Package)
		if problem != nil {
			return nil, problem
		}
		detail, problem := client(u.machines.context).PackageRelease(ctx, ref, request.Release)
		if problem != nil {
			return nil, problem
		}
		requirements, _, problem := detail.Constraints()
		return requirements, problem
	}
	revision, problem := u.machines.resolver.LocalRevision(request.InstallID, request.LocalPackageDigest)
	if problem != nil {
		return nil, problem
	}
	var paths []string
	project := ""
	for _, file := range revision.Files {
		if !strings.HasSuffix(file.Filename, ".whl") {
			continue
		}
		paths = append(paths, file.Path)
		if file.Kind == "project" {
			raw, problem := wheel.Metadata(file.Path)
			if problem != nil {
				return nil, problem
			}
			name, _, problem := wheel.MetadataIdentity(raw)
			if problem != nil {
				return nil, problem
			}
			project = name
		}
	}
	if project == "" {
		return nil, exit.New(exit.Structural, "captured script has no project wheel metadata")
	}
	selection, problem := packagepublish.ActiveWheelRequirements(ctx, project, nil, paths, python)
	if problem != nil {
		return nil, problem
	}
	return selection.ImageRequirements(), nil
}

// A typed pre-import rejection invalidates one boot's cached SDK facts exactly
// once for this request. Its queued identity and unoffered attempt are preserved.
func (u *rentalRuntimeUpdates) reobserve(request records.Request, machine string, problem *exit.Error) *exit.Error {
	if problem == nil || problem.ErrName() != "machine_execution.runtime_requirement" {
		return problem
	}
	seen, readProblem := u.machines.store.RequestHasEvent(request.ID, "machine.runtime_requirement_reobserved")
	if readProblem != nil {
		return readProblem
	}
	if seen {
		if row, readProblem := u.machines.store.RentalRow(machine); readProblem == nil && row != nil {
			return exit.Named(problem.Code, problem.ErrName(), "%s: %s", row.MachineName, problem.Message)
		}
		return problem
	}
	if readProblem := u.machines.store.AppendEvent(request.ID, "machine.runtime_requirement_reobserved", 0, map[string]any{"rental": machine, "requirement": problem.Message}); readProblem != nil {
		return readProblem
	}
	u.observed.Delete(machine)
	return exit.Named(exit.Unavailable, "rental.runtime_reobserve", "worker SDK requirements changed; checking the actual Runtime before retrying this request")
}
