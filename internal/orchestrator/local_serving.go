package orchestrator

// LocalServingPreparation names package metadata retained by the install. It is
// launch configuration, not a guessed PlacementSet or an invocation binding.
type LocalServingPreparation struct {
	PythonVersion      string   `json:"python_version"`
	PythonRequires     string   `json:"python_requires"`
	Published          bool     `json:"published"`
	Application        string   `json:"application"`
	ModelSlotPaths     []string `json:"model_slot_paths"`
	PackageInterface   []byte   `json:"package_interface"`
	LockedRequirements string   `json:"locked_requirements"`
}
