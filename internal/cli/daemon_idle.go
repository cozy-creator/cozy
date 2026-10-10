package cli

import (
	"fmt"
	"io"
	"sort"
	"strings"
	"time"

	"github.com/cozy-creator/cozy/internal/api"
	"github.com/cozy-creator/cozy/internal/daemon"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/orchestrator"
	"github.com/cozy-creator/cozy/internal/records"
)

// The daemon's own reason to leave. The owner's ruling: it may stop only once there is
// nothing left for it to manage — no rental it owns, no request or attempt it owes, no
// transfer, export, launch or teardown in flight, no client stream attached — and only
// once that has stayed true for a while. Work at rest (a paused run, a finished one keeping
// its results) is not managed: the next command's daemon resumes or serves it. User interaction is not the signal; the absence
// of a CLI is not the signal. The next command brings the daemon back (ensureDaemon).
//
// The duration is a DEBOUNCE over an observed fact, never the decision. `managed` decides;
// daemon.idle_shutdown_s says how long its answer must stay empty; the sample cadence is
// only how often it is asked.

const idleSampleCadence = time.Second

type idleWatch struct {
	debounce time.Duration
	store    *records.Store
	owner    *orchestrator.Orchestrator
	server   *api.Server
	startup  *daemonStartup
	log      io.Writer
}

// managed is the whole predicate: the durable obligations the records authority lists
// joined with the in-memory transitions the orchestrator holds.
func (w idleWatch) managed() ([]string, *exit.Error) {
	obligations, problem := w.store.Obligations()
	if problem != nil {
		return nil, problem
	}
	held, problem := w.owner.Managing()
	if problem != nil {
		return nil, problem
	}
	for _, o := range obligations {
		if !o.AtRest() {
			held = append(held, o.String())
		}
	}
	held = append(held, w.startup.holding()...)
	sort.Strings(held)
	return held, nil
}

// run samples until the predicate has answered "nothing" for the whole debounce, then asks
// the API server to stop under its admission gate. It returns once the daemon is stopping,
// or when quit closes because a signal or `cozy down` got there first.
func (w idleWatch) run(quit <-chan struct{}) {
	tick := time.NewTicker(idleSampleCadence)
	defer tick.Stop()
	var idleSince, heldSince time.Time
	lastHeld, lastProblem := "", ""
	for {
		select {
		case <-quit:
			return
		case <-tick.C:
		}
		held, problem := w.managed()
		if problem != nil {
			if problem.Message != lastProblem {
				fmt.Fprintf(w.log, "idle exit deferred: cannot read what the daemon manages: %s\n", problem.Message)
			}
			lastProblem = problem.Message
			idleSince = time.Time{}
			continue
		}
		lastProblem = ""
		if len(held) > 0 {
			idleSince = time.Time{}
			summary := strings.Join(held, ", ")
			if summary != lastHeld {
				lastHeld, heldSince = summary, time.Now()
				continue
			}
			// Said once per unchanged set, and only after the debounce it would otherwise
			// have exited under: a rental whose release keeps failing shows up here.
			if !heldSince.IsZero() && time.Since(heldSince) >= w.debounce {
				fmt.Fprintf(w.log, "idle exit held for %s by: %s\n", w.debounce, summary)
				heldSince = time.Time{}
			}
			continue
		}
		lastHeld = ""
		if idleSince.IsZero() {
			idleSince = time.Now()
			continue
		}
		if time.Since(idleSince) < w.debounce {
			continue
		}
		held, problem = w.server.StopUnlessManaging(w.managed)
		if problem != nil {
			fmt.Fprintf(w.log, "idle exit deferred: %s\n", problem.Message)
			idleSince = time.Time{}
			continue
		}
		if len(held) > 0 {
			idleSince = time.Time{}
			continue
		}
		fmt.Fprintf(w.log, "nothing to manage for %s; stopping (daemon.idle_shutdown_s=%d)\n",
			w.debounce, int64(w.debounce/time.Second))
		return
	}
}

// The daemon's OTHER reason to leave, and the one no configuration turns off: it no
// longer owns the root it published itself under. `daemon.idle_shutdown_s` is the knob
// for "nothing to manage"; there is deliberately no knob for this, because a daemon whose
// claim is gone cannot be found by a client, cannot be told to stop, and cannot write a
// record anyone will read. Staying alive is not a safety property there — it is a process
// nobody can reach and nobody can kill by any documented means, which is exactly how 143
// unreachable daemons once accumulated over three days on one development host.
//
// The signal is an observed fact about the filesystem, never elapsed time: `Claimed`
// compares the file this process holds against the path it published it at. Samples are
// required to agree `claimLostSamples` times running only so that a single unlucky read
// cannot end a daemon.
const claimLostSamples = 3

type claimWatch struct {
	held    *daemon.Held
	root    string
	managed func() ([]string, *exit.Error)
	stop    func()
	log     io.Writer
}

func (w claimWatch) run(quit <-chan struct{}) {
	tick := time.NewTicker(idleSampleCadence)
	defer tick.Stop()
	lost := 0
	for {
		select {
		case <-quit:
			return
		case <-tick.C:
		}
		if w.held.Claimed() {
			lost = 0
			continue
		}
		if lost++; lost < claimLostSamples {
			continue
		}
		// SAY WHAT IS BEING ABANDONED. The log lives under the root that just went, so
		// these words may reach no one — which is the honest shape of the situation and
		// not a reason to stay: the records naming this work went with the root.
		if held, problem := w.managed(); problem != nil {
			fmt.Fprintf(w.log, "%s no longer carries this daemon's claim; stopping (what it manages is unreadable: %s)\n",
				w.root, problem.Message)
		} else if len(held) > 0 {
			fmt.Fprintf(w.log, "%s no longer carries this daemon's claim; stopping and ABANDONING: %s\n",
				w.root, strings.Join(held, ", "))
		} else {
			fmt.Fprintf(w.log, "%s no longer carries this daemon's claim; stopping\n", w.root)
		}
		w.stop()
		return
	}
}
