package launch

// MachineResultDrift is how a machine's result differs from the captured result schema.
// The machine is an independently upgraded peer: a field it returns that the package does
// not declare is ignored, and a declared output it does not return, or returns unusably,
// fails alone while the rest of the result is read.
type MachineResultDrift struct {
	Ignored []string          // undeclared result field paths
	Failed  map[string]string // declared top-level output -> why it cannot be used
}
