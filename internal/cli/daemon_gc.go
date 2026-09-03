package cli

import (
	"fmt"
	"io"
	"strings"
	"time"

	"github.com/cozy-creator/cozy/internal/config"
	"github.com/cozy-creator/cozy/internal/home"
	"github.com/cozy-creator/cozy/internal/output"
	"github.com/cozy-creator/cozy/internal/tfs"
	"github.com/robfig/cron/v3"
)

// The store's scheduled reclamation. Owner ruling (2026-09-02): "garbage-collection for
// repo-CAS (both local and on tensorhub) should run on a cron job." The schedule is a
// cadence — when a pass runs — never the decision: what is reclaimed is TensorFS's, from
// its filesystem census, and a live writer or read lease refuses the pass by name. A pass
// runs only while the daemon manages nothing (the idle predicate): a download's admitted
// objects are unnamed until its commit, and an ingest of its own sits between `ingest run`
// and `ingest install` with no live writer — a sweep beside either could take bytes the
// daemon is still moving in.
type gcCron struct {
	schedule cron.Schedule
	cfg      config.Config
	layout   home.Layout
	idle     idleWatch
	log      io.Writer
}

func newGCCron(cfg config.Config, layout home.Layout, idle idleWatch, log io.Writer) (gcCron, bool) {
	if cfg.MaintenanceGCCron == "" {
		return gcCron{}, false
	}
	schedule, err := cron.ParseStandard(cfg.MaintenanceGCCron)
	if err != nil {
		// config.Load validated the expression; a disagreement here is a bug, not a run.
		fmt.Fprintf(log, "gc: maintenance.gc_cron %q refused: %s\n", cfg.MaintenanceGCCron, err)
		return gcCron{}, false
	}
	return gcCron{schedule: schedule, cfg: cfg, layout: layout, idle: idle, log: log}, true
}

func (g gcCron) run(quit <-chan struct{}) {
	for {
		// The wait until the next scheduled minute; nothing is killed when it fires.
		timer := time.NewTimer(time.Until(g.schedule.Next(time.Now())))
		select {
		case <-quit:
			timer.Stop()
			return
		case <-timer.C:
		}
		g.once()
	}
}

func (g gcCron) once() {
	held, problem := g.idle.managed()
	if problem != nil {
		fmt.Fprintf(g.log, "gc: deferred: %s\n", problem.Message)
		return
	}
	if len(held) > 0 {
		fmt.Fprintf(g.log, "gc: deferred: the daemon manages %s\n", strings.Join(held, ", "))
		return
	}
	tool, problem := tfs.Open(g.cfg)
	if problem != nil {
		fmt.Fprintf(g.log, "gc: deferred: %s\n", problem.Message)
		return
	}
	report, problem := tool.GC(true)
	if problem != nil {
		fmt.Fprintf(g.log, "gc: deferred: %s\n", problem.Message)
		return
	}
	line := fmt.Sprintf("gc: reclaimed %s in %d objects", output.Bytes(report.ReclaimedBytes),
		report.ReclaimedBlobs+report.ReclaimedManifests)
	if report.KeptObjects > 0 {
		line += fmt.Sprintf(", kept %s held by ingest session %s", output.Bytes(report.KeptBytes),
			strings.Join(report.Sessions, ", "))
	}
	fmt.Fprintln(g.log, line)
}
