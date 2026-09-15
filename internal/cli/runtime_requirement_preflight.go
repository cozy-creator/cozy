package cli

import (
	"context"
	"encoding/json"
	"fmt"

	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/hub"
	"github.com/cozy-creator/cozy/internal/launch"
	"github.com/cozy-creator/cozy/internal/packagepublish"
	"github.com/cozy-creator/cozy/internal/records"
	"github.com/cozy-creator/cozy/internal/rental"
)

type runtimeObservation struct {
	Observed struct {
		Updating bool `json:"update_in_progress"`
		Runtime  struct {
			Distribution string `json:"distribution"`
			Guarded      bool   `json:"supports_guarded_restart"`
		} `json:"runtime"`
		TensorFS string `json:"tensorfs"`
		Python   string `json:"python"`
	} `json:"observed"`
	Target hub.RuntimeUpdateTarget `json:"-"`
}

func (u *rentalRuntimeUpdates) preflight(ctx context.Context, request records.Request, machine string) *exit.Error {
	if machine == "local" || request.Release == "" {
		return nil
	}
	ref, problem := hub.ParseRef(request.Package)
	if problem != nil {
		return problem
	}
	detail, problem := client(u.machines.context).PackageRelease(ctx, ref, request.Release)
	if problem != nil {
		return problem
	}
	requirements, _, problem := detail.Constraints()
	if problem != nil {
		return problem
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
			return problem
		}
		if !remote.Development {
			raw, problem := client(u.machines.context).RentalImageInventory(ctx, machine)
			if problem != nil {
				return problem
			}
			inventory, err := rental.ImageInventory(raw)
			if err != nil {
				return exit.New(exit.Structural, "the immutable worker image inventory is invalid")
			}
			observed.Observed.Python = inventory.Python
			for _, distribution := range inventory.Distributions {
				if distribution.Distribution == "cozy-runtime" {
					observed.Observed.Runtime.Distribution = distribution.Version
				}
				if distribution.Distribution == "tensorfs" {
					observed.Observed.TensorFS = distribution.Version
				}
			}
			observed.Target.WorkerBootID = row.ExpectedWorkerBootID
		} else {
			selection, problem := u.selection(ctx, records.RuntimeUpdate{ID: "probe-" + machine, RentalID: machine, BootID: row.ExpectedWorkerBootID})
			if problem != nil {
				return problem
			}
			raw, problem := u.transport(ctx, selection, "inspect")
			if problem != nil {
				return problem
			}
			if json.Unmarshal(raw, &observed) != nil || observed.Observed.Runtime.Distribution == "" || observed.Observed.Python == "" {
				return exit.New(exit.Structural, "worker did not report its actual Runtime/interpreter versions")
			}
			observed.Target = selection.Target
		}

		u.observed.Store(machine, observed)
	}
	requirements, problem = packagepublish.EvaluateRequirements(ctx, requirements, observed.Observed.Python)
	if problem != nil {
		return problem
	}
	mismatch := launch.RuntimeRequirementMismatch(requirements, observed.Observed.Runtime.Distribution, observed.Observed.TensorFS)
	if mismatch == nil {
		return nil
	}
	row, problem := u.machines.store.RentalRow(machine)
	if problem != nil {
		return problem
	}
	if row == nil {
		return exit.New(exit.NotFound, "the requested rental is no longer attached")
	}
	message := fmt.Sprintf("%s has %s %s; %s@%s requires %s", row.MachineName, mismatch.Distribution, mismatch.Installed, request.Package, request.Release, mismatch.Required)
	if !observed.Observed.Runtime.Guarded {
		return exit.Named(exit.Structural, "rental.runtime_incompatible", "%s", message+". This older image cannot update safely; use a current private worker image")
	}
	if launch.RuntimeRequirementMismatch(requirements, observed.Target.RuntimeUpdate.Runtime.Version, observed.Target.RuntimeUpdate.TensorFS.Version) != nil {
		return exit.Named(exit.Structural, "rental.runtime_incompatible", "%s", message+". The approved worker update does not satisfy this package")
	}
	current, problem := u.machines.store.RuntimeUpdate(machine)
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
