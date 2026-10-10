package cli

import (
	"fmt"
	"io"
	"sync/atomic"
	"time"
)

// daemonStartup is what a started daemon owes its records: recovery of the work a previous
// daemon left, and the sweeps of what crashed writers left behind. It runs after the API
// serves, one step after another, and gates no request: a step waiting on another writer
// or a slow disk waits alone. Each step logs what it did and how long it took, and the idle
// exit waits for the last one.
type daemonStartup struct {
	log  io.Writer
	step atomic.Pointer[string]
}

type startupStep struct {
	name string
	run  func() string
}

func (s *daemonStartup) run(quit <-chan struct{}, steps []startupStep) {
	defer s.step.Store(nil)
	for _, step := range steps {
		select {
		case <-quit:
			return
		default:
		}
		s.step.Store(&step.name)
		began := time.Now()
		note := step.run()
		fmt.Fprintf(s.log, "startup %s: %s (%s)\n", step.name, note, time.Since(began).Round(time.Millisecond))
	}
}

// holding names the step still running, for the idle exit.
func (s *daemonStartup) holding() []string {
	if step := s.step.Load(); step != nil {
		return []string{"startup " + *step}
	}
	return nil
}
