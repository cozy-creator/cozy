package orchestrator

// The editable refresh's view of the fleet (cl-097). A refreshed install is only warm
// where a worker already holds the package: those are the workers a run would have
// re-prepared next, so the watcher re-prepares them first, over the same calls a run makes.

import "sort"

// LocalHolder is one local serving worker hosting a package under one install.
type LocalHolder struct {
	InstanceID, InstallID string
	Models                []ModelRef
}

// PackageHolders names the live local workers that hold a placement of pkg right now. A
// rental holds none: each rented run submits its own captured revision to Runtime.
func (c *Orchestrator) PackageHolders(pkg string) []LocalHolder {
	c.mu.Lock()
	defer c.mu.Unlock()
	var locals []LocalHolder
	for _, w := range c.workers {
		if w.exited || w.stopping || w.spec.IsJob() || w.spec.Connection != nil {
			continue
		}
		if w.spec.Placement.Package == pkg {
			locals = append(locals, LocalHolder{InstanceID: w.instanceID,
				InstallID: w.spec.Placement.InstallID,
				Models:    append([]ModelRef(nil), w.spec.Placement.Models...)})
		}
	}
	sort.Slice(locals, func(i, j int) bool { return locals[i].InstanceID < locals[j].InstanceID })
	return locals
}
