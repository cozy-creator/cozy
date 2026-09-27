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
// its filesystem census, and a live writer or read lease refuses the pass by name. The
// reclamationFence adds what TensorFS cannot see: a local request still moving unnamed
// bytes in defers the pass, and a retained run keeps the ingest sessions it may resume.
type gcCron struct {
	schedule cron.Schedule
	cfg      config.Config
	layout   home.Layout
	log      io.Writer
}

func newGCCron(cfg config.Config, layout home.Layout, log io.Writer) (gcCron, bool) {
	if cfg.MaintenanceGCCron == "" {
		return gcCron{}, false
	}
	schedule, err := cron.ParseStandard(cfg.MaintenanceGCCron)
	if err != nil {
		// config.Load validated the expression; a disagreement here is a bug, not a run.
		fmt.Fprintf(log, "gc: maintenance.gc_cron %q refused: %s\n", cfg.MaintenanceGCCron, err)
		return gcCron{}, false
	}
	return gcCron{schedule: schedule, cfg: cfg, layout: layout, log: log}, true
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
	fence, problem := readReclamationFence(g.layout)
	if problem == nil {
		problem = fence.busy()
	}
	if problem != nil {
		fmt.Fprintf(g.log, "gc: deferred: %s\n", problem.Message)
		return
	}
	tool, problem := tfs.Open(g.cfg)
	if problem != nil {
		fmt.Fprintf(g.log, "gc: deferred: %s\n", problem.Message)
		return
	}
	report, notes, problem := fence.collect(tool)
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
	for _, note := range notes {
		line += "; " + note
	}
	fmt.Fprintln(g.log, line)
}
