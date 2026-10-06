package cli

import (
	"encoding/json"
	"fmt"
	"slices"
	"strings"
	"time"

	"github.com/cozy-creator/cozy/internal/api"
	localapi "github.com/cozy-creator/cozy/internal/client"
	"github.com/cozy-creator/cozy/internal/orchestrator"
	"github.com/cozy-creator/cozy/internal/output"
)

// liveFrame bounds redraws to ten a second: a burst of events (a replayed history, a fast
// step loop) becomes one frame.
const liveFrame = 100 * time.Millisecond

// liveDoneRows is how many finished stages the live region keeps; the settled block lists all.
const liveDoneRows = 4

// liveView is a terminal's in-place display of one run: a header, the finished stages with
// their durations, the active stages with their bars, and the whole run's fraction. Events
// change the model and schedule a frame; a one-second clock keeps elapsed times moving
// while nothing arrives. Settling replaces the region with a stable block, so scrollback
// keeps one clean record of the run.
type liveView struct {
	run, target, machine, status string
	executing                    bool
	ended                        time.Time
	calls                        map[string]*liveStage
	phase                        *liveStage   // the preparation lane: one wait or phase at a time
	scopes                       []*liveStage // execution stages, in the order they started
	done                         []liveDone
	overall                      float64
	hasOverall                   bool
	eta                          time.Duration
	hasETA                       bool
	held                         map[string][]any // this run's GPU grants: call key -> ordinals
	waits                        []gpuWait
	stalls                       map[string]*stall // steps held for their callee's weights

	shown   []string // rows on screen
	cells   []int    // their widths, to erase them after a resize
	timer   *time.Timer
	due     time.Time
	settled bool
}

// liveStage is one open stage. An execution stage is the first part of a Runtime stage
// path ("Segment 2 of 9" in "Segment 2 of 9 / denoise"); the rest is its current leaf.
type liveStage struct {
	key, label, leaf string
	call             *callPhaseEvent
	scope            bool
	announced        bool           // reported at the root, not only as a child's scope
	whole            bool           // a child's own scope whose count completed: it is over
	event            localapi.Event // a phase's latest observation
	started, last    time.Time

	fraction       float64
	hasFraction    bool
	counted, bytes bool
	current, total int64
	rate           float64
	hasRate        bool
	stepSeconds    float64
	steps          int
	position       float64
}

// gpuWait is one of this run's calls waiting for GPUs; ownRun says only this run holds them.
type gpuWait struct {
	key                  string
	width                int
	ownRun, behindOthers bool
	since                time.Time
}

// stall names the model preparation a legacy step awaits. These partial wait spans
// cannot account for the call's other preparation or queue time.
type stall struct {
	count      int
	since      time.Time
	entrypoint string
}

type liveDone struct {
	stage  *liveStage
	label  string
	took   time.Duration
	failed bool
}

func (p *RunProgress) liveMode() bool {
	mode := p.ctx.Mode()
	return mode.Live && !mode.Full && !mode.JSON && !p.rawJSON
}

// Describe names the run in the live display's header. A display with no run (a rental's
// acquisition) has no header.
func (p *RunProgress) Describe(run, target, machine, status string) {
	p.mu.Lock()
	defer p.mu.Unlock()
	v := &p.view
	v.run, v.target = run, target
	if machine != "" {
		v.machine = machine
	}
	if v.status == "" {
		v.status = runStatus(publicObservedStatus(status))
	}
}

func (p *RunProgress) describeRun(life api.Lifecycle) {
	p.Describe(runReference(life.Number, life.RequestID), life.Package+"/"+life.Function, life.Machine, life.Status)
}

func (p *RunProgress) describeJob(state api.JobState) {
	machine := ""
	if state.MachineExecution != nil {
		machine = state.MachineExecution.Machine
	}
	p.Describe(runReference(state.Number, state.JobID), state.Package+"/"+state.Function, machine, state.Status)
}

// Say prints a message during the watch: above the live region, or verbatim elsewhere.
func (p *RunProgress) Say(text string) {
	p.mu.Lock()
	defer p.mu.Unlock()
	if p.liveMode() && !p.view.settled {
		p.notice(localapi.Event{}, strings.Trim(text, "\n"))
		return
	}
	fmt.Fprint(p.ctx.Err, text)
}

// Detach ends this watcher's display, never the run: the region settles as it stands and
// text follows it. Later events are not drawn.
func (p *RunProgress) Detach(text string) {
	p.mu.Lock()
	defer p.mu.Unlock()
	if !p.liveMode() {
		fmt.Fprint(p.ctx.Err, text)
		return
	}
	p.settle("", p.now())
	fmt.Fprint(p.ctx.Err, strings.TrimLeft(text, "\n"))
}

// onLive folds one event into the view and schedules a frame.
func (p *RunProgress) onLive(e localapi.Event) {
	v := &p.view
	if v.settled {
		return
	}
	kind := strings.TrimPrefix(e.Type, "request.")
	at := eventTime(e)
	fields, _ := e.Payload["value"].(map[string]any)
	switch kind {
	case "phase":
		name, _ := fields["phase"].(string)
		if name == "" || v.executing {
			return // replayed preparation cannot replace execution
		}
		machine, _ := fields["machine"].(string)
		if machine != "" && v.machine == "" {
			v.machine = machine
		}
		if bootOf(fields) != nil {
			// One stage for the whole boot, however many phases and attempts it passes.
			v.enter("boot:"+machine, e, at, "queued")
			break
		}
		v.enter("phase:"+name, e, at, "preparing")
	case "preparing":
		// The machine's account of a preparation still under way; a timed record is over.
		detail, _ := e.Payload["detail"].(string)
		if _, timed := e.Payload["ms"]; timed || detail == "" || e.Payload["stage"] != "machine" || v.executing {
			return
		}
		v.enter("preparing", e, at, "preparing")
	case "queued", "parked":
		// A queue heartbeat says less than the live preparation phase.
		cause, _ := e.Payload["wait"].(string)
		capacity := cause == orchestrator.WaitSlotBusy || cause == orchestrator.WaitQueueAhead
		if v.executing || v.phase != nil && v.phase.observed() && !capacity &&
			!(cause == orchestrator.WaitRental && v.phase.key == "phase:rental") {
			return
		}
		v.enter("wait", e, at, "queued")
	case "run.in_progress":
		if v.executing {
			return
		}
		v.enter("starting", e, at, "running")
	case "progress":
		p.liveProgress(fields, at)
	case "machine.call.phase":
		p.liveCallPhase(e)
	case "machine.call":
		p.finishLiveCall(e)
	case "run.completed":
		p.settle("completed", at)
		return
	case "run.failed", "run.canceled":
		p.settle(strings.TrimPrefix(kind, "run."), at)
		return
	case "machine.gpu.grant", "machine.gpu.release", "machine.gpu.wait":
		v.gpu(e, at)
	case "log":
		fields, _ := e.Payload["fields"].(map[string]any)
		switch e.Payload["name"] {
		case "model-prefetch":
			// A prefetch narrates its download on the log; one that counts bytes is drawn.
			if fields["unit"] != "bytes" {
				return
			}
			p.liveProgress(fields, at)
		case "model wait":
			v.stall(fields, at)
		default:
			return
		}
	case "attempt_failed", "requeued":
		v.retireAll(at, true)
		v.executing, v.status = false, "retrying"
		p.notice(e, progressLine(e, false))
	default:
		return
	}
	p.wake(liveFrame)
}

func (p *RunProgress) liveProgress(fields map[string]any, at time.Time) {
	v := &p.view
	facts, _ := p.observe(fields)
	if facts.hasOverall {
		v.overall, v.hasOverall = facts.overallFraction, true
		v.eta, v.hasETA = facts.overallRemaining, facts.hasOverallETA
	}
	v.executing, v.status = true, "running"
	if v.phase != nil {
		v.retire(v.phase, at, false)
		v.phase = nil
	}
	name, _ := fields["stage"].(string)
	name = strings.TrimSpace(name)
	if name == "" || strings.HasPrefix(name, "Waiting for GPU (") {
		return // Runtime's words for a gpu.wait, which this view reads typed
	}
	scope, _, nested := strings.Cut(name, " / ")
	key := scope
	var s *liveStage
	request, _ := fields["call_request"].(string)
	if request != "" {
		key = "call:" + request
		s = v.calls[request]
		if s != nil && s.call != nil {
			if attempt, known := number(fields["call_attempt"]); known && int64(attempt) != s.call.Attempt {
				return
			}
		}
	} else {
		for _, candidate := range v.calls {
			if candidate.call != nil && candidate.call.Label == scope {
				if s != nil {
					return
				} // a label cannot identify two calls, including a late completed one
				s = candidate
			}
		}
	}
	if s == nil {
		s = v.open(key)
	}
	if s != nil && s.call != nil && s.call.Phase == "terminal" {
		return
	}
	if s == nil {
		if !nested {
			v.supersede(at)
		}
		s = &liveStage{key: key, label: stageLabel(scope), scope: true, started: at, position: -1}
		v.scopes = append(v.scopes, s)
	} else if s.announced && !nested {
		v.sequenced(s)
	}
	if request != "" {
		if v.calls == nil {
			v.calls = map[string]*liveStage{}
		}
		v.calls[request] = s
	}
	leaf := ""
	if nested {
		leaf = strings.TrimPrefix(stageLabel(name), s.label+" · ")
		if s.call != nil {
			if _, nestedLeaf, found := strings.Cut(leaf, " · "+s.label+" · "); found {
				leaf = nestedLeaf
			}
		}
	}
	s.announced = s.announced || !nested
	if leaf != s.leaf {
		*s = liveStage{key: s.key, label: s.label, leaf: leaf, scope: true, announced: s.announced,
			started: s.started, position: -1, call: s.call}
	}
	s.measure(fields)
	s.last = at
	// A child's own scope (a weights download) has nothing left once its count is whole;
	// a later count under its name is another download.
	if !s.announced && s.counted && s.current == s.total {
		s.whole = true
		v.retire(s, at, false)
	}
}

// gpu follows this run's device grants and GPU waits. A wait whose blockers are all this
// run's own execution is not behind another run.
func (v *liveView) gpu(e localapi.Event, at time.Time) {
	key, _ := e.Payload["key"].(string)
	v.waits = slices.DeleteFunc(v.waits, func(w gpuWait) bool { return w.key == key })
	switch e.Type {
	case "machine.gpu.grant":
		if v.held == nil {
			v.held = map[string][]any{}
		}
		v.held[key], _ = e.Payload["ordinals"].([]any)
	case "machine.gpu.release":
		delete(v.held, key)
	default:
		width, _ := number(e.Payload["width"])
		blocked, _ := e.Payload["blocked_by"].([]any)
		wait := gpuWait{key: key, width: int(width), ownRun: len(blocked) > 0, since: at}
		for _, root := range blocked {
			if root != e.RequestID {
				wait.ownRun, wait.behindOthers = false, true
			}
		}
		v.waits = append(v.waits, wait)
	}
}

// open is the open stage named key, reopening one closed too early: a stage that reports
// again was still running.
func (v *liveView) open(key string) *liveStage {
	for _, s := range v.scopes {
		if s.key == key {
			return s
		}
	}
	for i := len(v.done) - 1; i >= 0; i-- {
		if s := v.done[i].stage; s != nil && s.key == key && !v.done[i].failed && !s.whole {
			v.done = slices.Delete(v.done, i, i+1)
			v.scopes = append(v.scopes, s)
			slices.SortStableFunc(v.scopes, func(a, b *liveStage) int { return a.started.Compare(b.started) })
			return s
		}
	}
	return nil
}

// supersede closes the stages a new root stage follows. Stages a root starts within one
// frame of each other may be a fan-out of concurrent calls and stay open until sequenced;
// a child's own scope (a download) runs beside the root's stages until its count is whole.
func (v *liveView) supersede(at time.Time) {
	v.close(at, false, func(s *liveStage) bool { return at.Sub(s.started) <= liveFrame || !s.announced })
}

// stall follows a step held for its callee's weights (Runtime's `model wait` records).
func (v *liveView) stall(fields map[string]any, at time.Time) {
	step, _ := fields["step"].(string)
	if step == "" {
		return
	}
	if v.stalls == nil {
		v.stalls = map[string]*stall{}
	}
	held := v.stalls[step]
	switch fields["event"] {
	case "start":
		if held == nil {
			held = &stall{since: at}
			v.stalls[step] = held
		}
		held.count++
		held.entrypoint, _ = fields["entrypoint"].(string)
	case "end":
		if held == nil {
			return
		}
		if held.count--; held.count <= 0 {
			delete(v.stalls, step)
		}
	}
}

// ran uses the producer's execution measurement, or a legacy stage's wall time.
func (v *liveView) ran(s *liveStage, at time.Time) time.Duration {
	if s.call != nil {
		elapsed, _ := s.call.duration("running", at)
		return elapsed
	}
	return max(at.Sub(s.started), 0)
}

// download is a child's own scope downloading its weights, read as the download itself.
func (s *liveStage) download() (string, bool) {
	rest, ok := strings.CutPrefix(s.leaf, "Downloading ")
	if s.announced || !s.bytes || !ok {
		return "", false
	}
	return "Downloading " + s.label + " " + rest, true
}

// sequenced closes the stages that started just before s and have been silent since: s
// reporting again at the root shows they were a quick sequence. A report nested under s
// makes it a parent step, which a fan-out's siblings run beside (run 1560's references).
func (v *liveView) sequenced(s *liveStage) {
	for _, other := range slices.Clone(v.scopes) {
		if other.announced && other.started.Before(s.started) && s.started.Sub(other.started) <= liveFrame &&
			other.last.Before(s.started) {
			v.retire(other, s.started, false)
		}
	}
}

// close retires the open stages spare does not keep, as the run moves on at at: the stage
// that reported last hands over then; any other ended with its own last report.
func (v *liveView) close(at time.Time, failed bool, spare func(*liveStage) bool) {
	var latest time.Time
	for _, s := range v.scopes {
		if s.last.After(latest) {
			latest = s.last
		}
	}
	for _, s := range slices.Clone(v.scopes) {
		if s.call != nil || spare != nil && spare(s) {
			continue
		}
		end := s.last
		if !s.last.Before(latest) {
			end = at
		}
		v.retire(s, end, failed)
	}
}

func (v *liveView) retireAll(at time.Time, failed bool) {
	if v.phase != nil {
		v.retire(v.phase, at, failed)
		v.phase = nil
	}
	v.close(at, failed, nil)
	for _, s := range slices.Clone(v.scopes) {
		// A run ending without this call's terminal record gives no final call
		// measurement. Do not charge the missing interval to execution.
		if s.call != nil {
			s.call.Phase, s.call.AtUnixMS = "terminal", at.UnixMilli()
			s.call.callTiming = callTiming{}
		}
		v.retire(s, at, failed)
	}
}

func (v *liveView) retire(s *liveStage, end time.Time, failed bool) {
	v.scopes = slices.DeleteFunc(v.scopes, func(open *liveStage) bool { return open == s })
	label := s.label
	if download, ok := s.download(); ok {
		label = download
	} else if (failed || !s.announced) && s.leaf != "" {
		label += " · " + s.leaf // where it failed, or all a child's own scope did
	}
	if s.bytes && s.total > 0 {
		label += " · " + output.Bytes(s.total)
	} else if fields, ok := s.event.Payload["value"].(map[string]any); ok {
		if total, ok := number(fields["total_bytes"]); ok && total > 0 {
			label += " · " + output.Bytes(int64(total))
		}
	}
	stage := s
	if !s.scope {
		stage = nil
	}
	v.done = append(v.done, liveDone{stage: stage, label: label, took: v.ran(s, end), failed: failed})
}

// observed is a stage a producer measured, which a queue heartbeat never replaces.
func (s *liveStage) observed() bool {
	return strings.HasPrefix(s.key, "phase:") || strings.HasPrefix(s.key, "boot:")
}

// enter makes key the preparation lane's current stage, closing a different one now. A
// rental's boot takes over the queue wait it explains rather than closing it.
func (v *liveView) enter(key string, e localapi.Event, at time.Time, status string) {
	fields, _ := e.Payload["value"].(map[string]any)
	started := at
	if elapsed, ok := number(fields["elapsed_ms"]); ok && elapsed > 0 {
		started = at.Add(-time.Duration(elapsed) * time.Millisecond)
	}
	if v.phase != nil && v.phase.key != key {
		if strings.HasPrefix(key, "boot:") && v.phase.key == "wait" {
			started = v.phase.started
		} else {
			v.retire(v.phase, at, false)
		}
		v.phase = nil
	}
	if v.phase == nil {
		if boot := bootOf(fields); boot != nil && !boot.StartedAt.IsZero() && boot.StartedAt.Before(started) {
			started = boot.StartedAt
		}
		v.phase = &liveStage{key: key, started: started}
	}
	v.phase.event, v.phase.last, v.status = e, at, status
	if machine, ok := strings.CutPrefix(key, "boot:"); ok {
		w := describeBoot(machine, bootOf(fields), at)
		v.phase.label = joinParts(w.subject(machine), w.where)
		return
	}
	switch key {
	case "wait":
		v.phase.label = strings.TrimSpace(HumanWaitLine(e.Payload))
	case "preparing":
		v.phase.label, _ = e.Payload["detail"].(string)
	case "starting":
		v.phase.label = "starting"
	default:
		name, _ := fields["phase"].(string)
		if name == orchestrator.WaitSlotBusy || name == orchestrator.WaitQueueAhead {
			v.phase.label = strings.TrimSpace(HumanPhaseLine(fields))
			return
		}
		v.phase.label = strings.ReplaceAll(preparationLabel(name), "_", " ")
		if machine, _ := fields["machine"].(string); machine != "" {
			v.phase.label += " on " + machine
		}
	}
}

// measure reads one progress sample of the current leaf.
func (s *liveStage) measure(fields map[string]any) {
	fraction, hasFraction := number(fields["stage_fraction"])
	position, hasPosition := number(fields["position"])
	total, hasTotal := number(fields["total"])
	if hasPosition && hasTotal && position >= 0 && total > 0 && position <= total &&
		position == float64(int64(position)) && total == float64(int64(total)) {
		s.counted, s.current, s.total = true, int64(position), int64(total)
		s.bytes = fields["unit"] == "bytes"
		s.rate, s.hasRate = number(fields["rate"])
		s.hasRate = s.hasRate && s.rate >= 0
		if !hasFraction {
			fraction, hasFraction = position/total, true
		}
		if stepMS, ok := number(fields["step_ms"]); ok && stepMS > 0 && position > s.position {
			s.stepSeconds += stepMS / 1000
			s.steps++
			s.position = position
		}
	}
	if hasFraction && fraction >= 0 && fraction <= 1 {
		s.fraction, s.hasFraction = fraction, true
	}
}

// frame is the region's rows at a moment. Live, it fits height (0: unbounded) by keeping
// fewer finished stages, then fewer active rows; settled, it is the whole record.
func (p *RunProgress) frame(at time.Time, width, height int) []string {
	v := &p.view
	if v.settled && !v.ended.IsZero() {
		at = v.ended
	}
	var head []string
	if v.run != "" || v.target != "" {
		head = append(head, p.liveHeader(at))
	}
	var active []string
	if v.phase != nil {
		active = append(active, v.phase.phaseRows(at, width)...)
	}
	for _, s := range v.scopes {
		active = append(active, s.rows(v.ran(s, at), v.stalls[s.key], width, at)...)
	}
	for _, w := range v.waits {
		if !v.ended.IsZero() {
			break
		}
		held := map[any]bool{}
		for _, ordinals := range v.held {
			for _, ordinal := range ordinals {
				held[ordinal] = true
			}
		}
		detail := api.GPUWaitDetail(w.width, len(held), w.ownRun)
		if w.behindOthers {
			detail += ", behind another run"
		}
		active = append(active, "  ▸ waiting for GPU ("+detail+") · "+shortDuration(max(at.Sub(w.since), 0)))
	}
	if v.hasOverall && v.ended.IsZero() {
		row := "  overall " + meter(v.overall, width)
		if v.hasETA {
			row += " · ETA ~" + shortDuration(v.eta)
		}
		active = append(active, row)
	}
	done := v.done
	if !v.settled {
		done = done[max(0, len(done)-liveDoneRows):]
	}
	if height > 1 {
		// Finished stages give way first; "… N earlier" takes a row when any are hidden.
		room := height - 1 - len(head)
		if len(active)+len(done)+min(1, len(v.done)-len(done)) > room {
			done = done[len(done)-min(len(done), max(0, room-len(active)-1)):]
		}
		if used := len(done) + min(1, len(v.done)-len(done)); len(active) > room-used {
			visible := max(0, room-used-1)
			active = append(active[:visible], fmt.Sprintf("    … %d more rows", len(active)-visible))
		}
	}
	rows := head
	if earlier := len(v.done) - len(done); earlier > 0 {
		rows = append(rows, fmt.Sprintf("  … %d earlier", earlier))
	}
	return append(append(rows, doneRows(done, width)...), active...)
}

func (p *RunProgress) liveHeader(at time.Time) string {
	v := &p.view
	parts := []string{}
	if v.run != "" {
		parts = append(parts, "Run "+v.run)
	}
	for _, part := range []string{v.target, v.machine, v.status} {
		if part != "" {
			parts = append(parts, part)
		}
	}
	if !p.began.IsZero() && !at.Before(p.began) {
		parts = append(parts, shortDuration(at.Sub(p.began)))
	}
	return strings.Join(parts, " · ")
}

func doneRows(done []liveDone, width int) []string {
	column := 0
	for _, d := range done {
		column = max(column, textCells(d.label))
	}
	if width > 0 {
		column = min(column, max(8, width-16))
	}
	rows := make([]string, 0, len(done))
	for _, d := range done {
		mark := "✓"
		if d.failed {
			mark = "✗"
		}
		label := clampLine(d.label, column+1)
		timing := shortDuration(d.took)
		if d.stage != nil {
			timing = "wall " + timing
		}
		if d.stage != nil && d.stage.call != nil {
			timing = callTimingText(*d.stage.call, time.UnixMilli(d.stage.call.AtUnixMS), true)
		}
		rows = append(rows, "  "+mark+" "+label+strings.Repeat(" ", column-textCells(label))+"  "+timing)
	}
	return rows
}

// rows are an open stage's line and bar. A legacy step held for weights names its wait;
// producer phase measurements keep preparation and execution clocks separate.
func (s *liveStage) rows(ran time.Duration, held *stall, width int, at time.Time) []string {
	line := "  ▸ " + s.label
	if s.call != nil && s.call.Phase != "running" {
		return []string{line + " · " + callTimingText(*s.call, at, false)}
	}
	if held != nil && s.call == nil {
		return []string{line + " · waiting for " + strings.TrimSpace(held.entrypoint+" weights")}
	}
	if download, ok := s.download(); ok {
		line = "  ▸ " + download
	} else if s.leaf != "" {
		line += " · " + s.leaf
	}
	timing := "wall " + shortDuration(ran)
	if s.call != nil {
		timing = callTimingText(*s.call, at, false)
	}
	rows := []string{line + " · " + timing}
	if !s.hasFraction {
		return rows
	}
	row := "    " + meter(s.fraction, width)
	switch {
	case s.counted && s.bytes:
		row += "  " + bytesMoved(float64(s.current), float64(s.total), s.rate, s.hasRate, 0)
	case s.counted:
		row += fmt.Sprintf("  step %d/%d", s.current, s.total)
		if s.steps > 0 {
			perStep := s.stepSeconds / float64(s.steps)
			remaining := time.Duration(float64(s.total-s.current) * perStep * float64(time.Second))
			row += fmt.Sprintf(" · %.2fs/step avg · ETA ~%s", perStep, shortDuration(remaining))
		}
	}
	return append(rows, row)
}

func (s *liveStage) phaseRows(at time.Time, width int) []string {
	rows := []string{"  ▸ " + s.label + " · " + shortDuration(max(at.Sub(s.started), 0))}
	if machine, ok := strings.CutPrefix(s.key, "boot:"); ok {
		fields, _ := s.event.Payload["value"].(map[string]any)
		w := describeBoot(machine, bootOf(fields), at)
		if w.replan != "" {
			rows = append(rows, "    "+w.replan)
		}
		rows = append(rows, "    "+joinParts(w.stage, w.activity))
		return append(rows, rentalRows(fields)...)
	}
	if !strings.HasPrefix(s.key, "phase:") {
		return rows
	}
	fields := phaseFieldsAt(s.event, s.started, at)
	rows = append(rows, rentalRows(fields)...)
	rows = append(rows, transferRows("    ", fields, fields, width)...)
	models, _ := fields["models"].([]any)
	for _, value := range models {
		model, ok := value.(map[string]any)
		name, _ := model["model"].(string)
		if !ok || name == "" {
			continue
		}
		if lane, _ := model["lane"].(string); lane != "" {
			name += "/" + lane
		}
		rows = append(rows, "    "+name)
		rows = append(rows, transferRows("      ", model, fields, width)...)
	}
	return rows
}

// transferRows is a measured transfer's bar, bytes, rate and remaining time; an old
// sample keeps its bytes and says how old it is instead of a stale rate.
func transferRows(indent string, counts, phase map[string]any, width int) []string {
	moved, ok := number(counts["moved_bytes"])
	if !ok {
		return nil
	}
	total, _ := number(counts["total_bytes"])
	age, measured := number(counts["sample_age_ms"])
	if !measured {
		age, _ = number(phase["sample_age_ms"])
	}
	row := indent
	if total > 0 {
		row += meter(moved/total, width) + "  "
	}
	if old := time.Duration(age) * time.Millisecond; old > orchestrator.PreparationRateMaxAge {
		return []string{row + bytesMoved(moved, total, 0, false, 0) + " · last update " + shortDuration(old) + " ago"}
	}
	rate, hasRate := number(counts["rate_bytes_per_second"])
	remaining, _ := number(counts["remaining_ms"])
	return []string{row + bytesMoved(moved, total, rate, hasRate && rate > 0, time.Duration(remaining)*time.Millisecond)}
}

// bytesMoved is "moved / total · rate/s · ETA"; remaining is the producer's, else rate's.
func bytesMoved(moved, total, rate float64, hasRate bool, remaining time.Duration) string {
	text := output.Bytes(int64(moved))
	if total > 0 {
		text += " / " + output.Bytes(int64(total))
	}
	if !hasRate {
		return text
	}
	text += " · " + output.Bytes(int64(rate)) + "/s"
	if remaining <= 0 && rate > 0 && moved < total {
		remaining = time.Duration((total - moved) / rate * float64(time.Second))
	}
	if remaining > 0 {
		text += " · ETA ~" + shortDuration(remaining)
	}
	return text
}

// meter is a bar and its percentage, narrower on a narrow terminal.
func meter(fraction float64, width int) string {
	cells := 20
	if width > 0 && width < 80 {
		cells = max(6, cells-(80-width))
	}
	filled := min(max(int(fraction*float64(cells)+0.5), 0), cells)
	return strings.Repeat("█", filled) + strings.Repeat("░", cells-filled) + fmt.Sprintf(" %3.0f%%", fraction*100)
}

// wake schedules a frame after at most after; an earlier one already due stands.
func (p *RunProgress) wake(after time.Duration) {
	v := &p.view
	due := time.Now().Add(after)
	if !v.due.IsZero() && !v.due.After(due) {
		return
	}
	v.due = due
	if v.timer == nil {
		v.timer = time.AfterFunc(after, p.tick)
		return
	}
	v.timer.Reset(after)
}

// tick draws a frame, then keeps the clock moving while the display lives.
func (p *RunProgress) tick() {
	p.mu.Lock()
	defer p.mu.Unlock()
	v := &p.view
	v.due = time.Time{}
	if p.closed || v.settled {
		return
	}
	width, height := p.width(), terminalHeight(p.ctx.Err)
	if rows := p.frame(p.now(), width, height); !slices.Equal(rows, v.shown) {
		p.replace(rows, width, height)
	}
	p.wake(time.Second)
}

// settle replaces the live region with the run's stable record. A terminal closes every
// open stage at its recorded time; a watcher leaving early keeps them as they stand.
func (p *RunProgress) settle(status string, at time.Time) {
	v := &p.view
	if v.settled {
		return
	}
	if v.timer != nil {
		v.timer.Stop()
	}
	if status != "" {
		v.retireAll(at, status != "completed")
		v.status, v.ended = status, at
	}
	v.settled = true
	width := p.width()
	if rows := p.frame(at, width, 0); len(rows) > 0 || len(v.shown) > 0 {
		p.replace(rows, width, terminalHeight(p.ctx.Err))
	}
}

// notice prints one line above the live region, which is redrawn beneath it.
func (p *RunProgress) notice(_ localapi.Event, line string) {
	if line == "" {
		return
	}
	if !p.liveMode() || p.view.settled || len(p.view.shown) == 0 {
		fmt.Fprintln(p.ctx.Err, line)
		return
	}
	width, height := p.width(), terminalHeight(p.ctx.Err)
	shown := p.view.shown
	p.replace(nil, width, height)
	fmt.Fprintf(p.ctx.Err, "\r\033[K%s\r\n", clampLine(line, width))
	p.replace(shown, width, height)
}

// replace swaps the region on screen for rows in one write. The cursor rests at the start
// of the line below the region, so erasing is one move up and one clear, counting the
// rows a narrower terminal has since wrapped.
func (p *RunProgress) replace(rows []string, width, height int) {
	v := &p.view
	var b strings.Builder
	lines := 0
	for _, cells := range v.cells {
		lines++
		if width > 0 && cells > width {
			lines += (cells - 1) / width
		}
	}
	if height > 1 {
		lines = min(lines, height-1)
	}
	if lines > 0 {
		fmt.Fprintf(&b, "\r\033[%dA\033[J", lines)
	}
	v.cells = v.cells[:0]
	for _, row := range rows {
		row = clampLine(row, width)
		v.cells = append(v.cells, textCells(row))
		b.WriteString("\r\033[K" + p.paint(row) + "\r\n")
	}
	v.shown = rows
	fmt.Fprint(p.ctx.Err, b.String())
}

// paint styles a clamped row: the run's name, stage marks and the bar. NO_COLOR keeps none.
func (p *RunProgress) paint(row string) string {
	if !p.ctx.Mode().Color {
		return row
	}
	const reset = "\033[0m"
	if strings.HasPrefix(row, "Run ") {
		name, tail, found := strings.Cut(row, " · ")
		if found {
			tail = " · " + tail
		}
		return "\033[1m" + name + reset + tail
	}
	for _, mark := range []struct{ prefix, color string }{{"  ✓ ", "32"}, {"  ✗ ", "31"}, {"  ▸ ", "36"}} {
		if rest, ok := strings.CutPrefix(row, mark.prefix); ok {
			return "  \033[" + mark.color + "m" + strings.TrimSpace(mark.prefix) + reset + " " + rest
		}
	}
	start := strings.IndexAny(row, "█░")
	if start < 0 {
		return row
	}
	filled := start
	for strings.HasPrefix(row[filled:], "█") {
		filled += len("█")
	}
	empty := filled
	for strings.HasPrefix(row[empty:], "░") {
		empty += len("░")
	}
	return row[:start] + "\033[36m" + row[start:filled] + reset + "\033[2m" + row[filled:empty] + reset + row[empty:]
}

// Phase observations own a call's clock; labels are only its human presentation.
func (p *RunProgress) liveCallPhase(e localapi.Event) {
	phase, ok := readCallPhase(e)
	if !ok {
		return
	}
	p.applyCallPhase(phase, false)
}

func (p *RunProgress) applyCallPhase(phase callPhaseEvent, final bool) {
	v := &p.view
	if v.calls == nil {
		v.calls = map[string]*liveStage{}
	}
	s := v.calls[phase.Request]
	if s != nil {
		if prior := s.call; prior != nil {
			if final && phase.Attempt == prior.Attempt {
				// The settled call is authoritative, including absent measurements.
				// Its envelope clock may trail a producer's monotonic phase stamp.
				phase.AtUnixMS = max(phase.AtUnixMS, prior.AtUnixMS)
			} else if !phase.follows(prior.Attempt, prior.AtUnixMS, prior.Phase) {
				return
			}
		}
	} else {
		for _, candidate := range v.scopes {
			if candidate.key == phase.Label && candidate.call == nil {
				s = candidate
				break
			}
		}
		if s == nil {
			s = &liveStage{key: "call:" + phase.Request, label: stageLabel(phase.Label), scope: true, announced: true, position: -1}
			v.scopes = append(v.scopes, s)
		}
		v.calls[phase.Request] = s
	}
	v.executing, v.status = true, "running"
	if v.phase != nil {
		v.retire(v.phase, time.UnixMilli(phase.AtUnixMS), false)
		v.phase = nil
	}
	if phase.Label != "" {
		s.label = stageLabel(phase.Label)
	}
	if s.label == "" {
		s.label = phase.Export
	}
	if s.label == "" {
		s.label = phase.Request
	}
	if s.call != nil && phase.Attempt > s.call.Attempt {
		v.done = slices.DeleteFunc(v.done, func(done liveDone) bool { return done.stage == s })
		if !slices.Contains(v.scopes, s) {
			v.scopes = append(v.scopes, s)
		}
		s.leaf = ""
		s.hasFraction, s.counted, s.whole = false, false, false
	}
	s.call = &phase
	s.started = time.UnixMilli(phase.CalledUnixMS)
	s.last = time.UnixMilli(phase.AtUnixMS)
	if phase.Phase == "terminal" {
		s.whole = true
		for i := range v.done {
			if v.done[i].stage == s {
				v.done[i].took = v.ran(s, s.last)
				v.done[i].failed = phase.Status == "failed" || phase.Status == "canceled"
				return
			}
		}
		v.retire(s, s.last, phase.Status == "failed" || phase.Status == "canceled")
	}
}

func (p *RunProgress) finishLiveCall(e localapi.Event) {
	raw, err := json.Marshal(e.Payload)
	if err != nil {
		return
	}
	var call callEvent
	if json.Unmarshal(raw, &call) != nil || call.Request == "" || (!call.callTiming.present() && p.view.calls[call.Request] == nil) {
		return
	}
	phase := callPhaseEvent{Request: call.Request, Parent: call.Parent, Index: call.Index, Attempt: call.Attempt, Module: call.Module, Export: call.Export, Label: call.Label, Phase: "terminal", Status: call.Status, AtUnixMS: eventTime(e).UnixMilli(), CalledUnixMS: call.CalledUnixMS, callTiming: call.callTiming}
	p.applyCallPhase(phase, true)
}

func callTimingText(call callPhaseEvent, at time.Time, complete bool) string {
	phase, label := call.Phase, call.Phase
	switch phase {
	case "running", "terminal":
		phase, label = "running", "execution"
	case "finalizing", "paused":
		phase, label = "running", call.Phase+" · execution"
	}
	elapsed, known := call.duration(phase, at)
	text := label
	if known {
		text += " " + shortDuration(elapsed)
	} else {
		text += " —"
	}
	if complete {
		for _, part := range []struct{ phase, label string }{{"queued", "queued"}, {"preparing", "preparation"}} {
			if elapsed, ok := call.duration(part.phase, at); ok && elapsed > 0 {
				text += " · " + part.label + " " + shortDuration(elapsed)
			}
		}
		if call.CalledUnixMS > 0 {
			text += " · wall " + shortDuration(max(at.Sub(time.UnixMilli(call.CalledUnixMS)), 0))
		}
	}
	return text
}

// Frame is the region at a moment on a terminal width columns wide, unstyled: the settled
// block once the run ended or the watcher left. The product suite golden-tests it.
func (p *RunProgress) Frame(at time.Time, width int) []string {
	p.mu.Lock()
	defer p.mu.Unlock()
	rows := p.frame(at, width, 0)
	for i, row := range rows {
		rows[i] = clampLine(row, width)
	}
	return rows
}
