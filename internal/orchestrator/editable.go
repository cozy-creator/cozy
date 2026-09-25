package orchestrator

// The editable refresh's view of the fleet (cl-097). A refreshed install is only warm
// where a worker already holds the package: those are the workers a run would have
// re-prepared next, so the watcher re-prepares them first, over the same calls a run makes.

import (
	"sort"

	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/localpackage"
)

// LocalHolder is one local serving worker hosting a package under one install.
type LocalHolder struct {
	InstanceID, InstallID string
	Models                []ModelRef
}

// RentalHolder is one attached rental hosting the package under its pinned slot, with
// every placement it serves for it: the function each was resolved for and its selection.
type RentalHolder struct {
	RentalID, InstanceID string
	Placements           []DesiredPlacement
}

// PackageHolders names the live workers that hold a placement of pkg right now.
func (c *Orchestrator) PackageHolders(pkg string) ([]LocalHolder, []RentalHolder) {
	c.mu.Lock()
	defer c.mu.Unlock()
	var locals []LocalHolder
	var rentals []RentalHolder
	for _, w := range c.workers {
		if w.exited || w.stopping || w.spec.IsJob() {
			continue
		}
		if w.spec.Connection == nil {
			if w.spec.Placement.Package == pkg {
				locals = append(locals, LocalHolder{InstanceID: w.instanceID,
					InstallID: w.spec.Placement.InstallID,
					Models:    append([]ModelRef(nil), w.spec.Placement.Models...)})
			}
			continue
		}
		rentalID := w.spec.Connection.RentalID
		slot := pinnedPackage(pkg, rentalID)
		holder := RentalHolder{RentalID: rentalID, InstanceID: w.instanceID}
		// One row per placement: every entrypoint a report made routable shares it, and
		// the editable refresh re-prepares a placement once.
		first := map[string]DesiredPlacement{}
		for _, placement := range w.remotePlacements {
			if placement.Package != slot {
				continue
			}
			id := placement.PlacementIDValue
			if held, seen := first[id]; !seen || placement.Entrypoints[0].Name < held.Entrypoints[0].Name {
				first[id] = placement
			}
		}
		for _, placement := range first {
			holder.Placements = append(holder.Placements, placement)
		}
		if len(holder.Placements) > 0 {
			sort.Slice(holder.Placements, func(i, j int) bool {
				return holder.Placements[i].Entrypoints[0].Name < holder.Placements[j].Entrypoints[0].Name
			})
			rentals = append(rentals, holder)
		}
	}
	sort.Slice(locals, func(i, j int) bool { return locals[i].InstanceID < locals[j].InstanceID })
	sort.Slice(rentals, func(i, j int) bool { return rentals[i].RentalID < rentals[j].RentalID })
	return locals, rentals
}

// PrepareRentalRevision is resolveFor's rental arm for an editable request, run before any
// request asks: the sealed revision crosses through PodHost, its models rebind, and the
// worker's placement for the function is recorded so the next run finds it staged.
func (c *Orchestrator) PrepareRentalRevision(rentalID, operationID string,
	revision localpackage.Revision, logical LogicalPackage,
) (DesiredPlacement, *exit.Error) {
	instance, _, _, problem := c.EnsureRental(rentalID)
	if problem != nil {
		return DesiredPlacement{}, problem
	}
	if problem := c.ConvergeLocalPackage(instance, operationID, revision, "", nil); problem != nil {
		return DesiredPlacement{}, problem
	}
	if len(logical.Models) > 0 {
		models := downloadModelRefs(logical.Models)
		if len(models) != len(logical.Models) {
			return DesiredPlacement{}, exit.Named(exit.Validation,
				"private_placement_model_unpublished",
				"private serving requires exact published model releases")
		}
		if problem := c.ConvergeUnpublishedPlacement(instance, operationID, revision.Digest,
			models); problem != nil {
			return DesiredPlacement{}, problem
		}
	}
	spec, _, problem := c.ensureLogicalPackageReady(instance, rentalID, logical, false)
	if problem != nil {
		return DesiredPlacement{}, problem
	}
	return spec.Placement, nil
}
