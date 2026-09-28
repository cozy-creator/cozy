package orchestrator

// The editable refresh's view of the fleet (cl-097). A refreshed install is only warm
// where a worker already holds the package: those are the workers a run would have
// re-prepared next, so the watcher re-prepares them first, over the same calls a run makes.

// LocalHolder is one local serving worker hosting a package under one install.
type LocalHolder struct {
	InstanceID, InstallID string
	Models                []ModelRef
}
