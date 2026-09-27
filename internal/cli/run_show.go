package cli

import (
	"encoding/json"
	"fmt"
	"io"
	"sort"
	"strings"
	"text/tabwriter"
	"time"

	"github.com/cozy-creator/cozy/internal/api"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/orchestrator"
	"github.com/cozy-creator/cozy/internal/output"
	"github.com/cozy-creator/cozy/internal/units"
)

// runReport is one run's execution evidence: cold setup apart from inference, the per-step
// series, and which ranks ran it. Every fact comes from the daemon's records (durable
// events and the kept triage bundle); JSON also carries both sources whole.
type runReport struct {
	Number      int64               `json:"number"`
	RequestID   string              `json:"request_id"`
	Status      string              `json:"status"`
	Target      string              `json:"target"`
	Machine     string              `json:"machine,omitempty"`
	CreatedAt   string              `json:"created_at"`
	QueuedMS    int64               `json:"queued_ms"`
	ExecutionMS int64               `json:"execution_ms"`
	WallMS      int64               `json:"wall_ms,omitempty"`
	Waiting     string              `json:"waiting,omitempty"`
	Stages      []reportStage       `json:"stages"`
	Steps       []reportSteps       `json:"steps,omitempty"`
	Degree      int                 `json:"degree,omitempty"`
	Ranks       []reportRank        `json:"ranks,omitempty"`
	Events      []api.EvidenceEvent `json:"events"`
	Triage      json.RawMessage     `json:"triage,omitempty"`
}

type reportStage struct {
	Name        string  `json:"name"`
	Kind        string  `json:"kind"` // setup, gpu, phase, inference or transfer
	StartUnixMS int64   `json:"start_unix_ms,omitempty"`
	MS          float64 `json:"ms"`
	Count       int     `json:"count,omitempty"`
	Bytes       int64   `json:"bytes,omitempty"`
	Detail      string  `json:"detail,omitempty"`
}

type reportSteps struct {
	Name       string       `json:"name"`
	Count      int          `json:"count"`
	TotalMS    float64      `json:"total_ms"`
	FirstMS    float64      `json:"first_ms"`
	RestMeanMS float64      `json:"rest_mean_ms"`
	MinMS      float64      `json:"min_ms"`
	MaxMS      float64      `json:"max_ms"`
	Series     [][2]float64 `json:"series,omitempty"` // [end_unix_ms, ms] per step
	Dropped    int          `json:"series_dropped,omitempty"`
}

// reportRank is Runtime's per-rank record, read tolerantly.
type reportRank struct {
	Rank      int    `json:"rank"`
	PID       int    `json:"pid"`
	Ordinal   int    `json:"ordinal"`
	UUID      string `json:"uuid"`
	StartUS   int64  `json:"start_us"`
	EndUS     int64  `json:"end_us"`
	Attention struct {
		Requested string `json:"requested"`
		Observed  string `json:"observed"`
		Impl      string `json:"impl"`
	} `json:"attention"`
}

type triageTrack struct {
	Count         int          `json:"count"`
	TotalMS       float64      `json:"total_ms"`
	MinMS         float64      `json:"min_ms"`
	MaxMS         float64      `json:"max_ms"`
	FirstMS       float64      `json:"first_ms"`
	StartedUnixMS int64        `json:"started_unix_ms"`
	Series        [][2]float64 `json:"series"`
	SeriesDropped int          `json:"series_dropped"`
}

type triageSetup struct {
	StartedUnixMS  int64              `json:"started_unix_ms"`
	PreparedUnixMS int64              `json:"prepared_unix_ms"`
	MS             float64            `json:"ms"`
	PrepareMS      float64            `json:"prepare_ms"`
	FillMS         float64            `json:"fill_ms"`
	WarmMS         float64            `json:"warm_ms"`
	Legs           map[string]float64 `json:"legs_ms"`
}

type triageEvidence struct {
	Measurements struct {
		Attribution struct {
			Stages map[string]triageTrack `json:"stages"`
			Steps  map[string]triageTrack `json:"steps"`
		} `json:"attribution"`
		Execution struct {
			Degree       int          `json:"degree"`
			Ranks        []reportRank `json:"ranks"`
			Executor     triageSetup  `json:"executor"`
			Construction triageSetup  `json:"construction"`
		} `json:"execution"`
	} `json:"measurements"`
}

func handleRunShow(ctx *Context) *exit.Error {
	client, problem := dial(ctx)
	if problem != nil {
		return problem
	}
	life, problem := client.Request(strings.TrimSpace(ctx.Inv.Args[0]))
	if problem != nil {
		return problem
	}
	evidence, problem := client.Evidence(life.RequestID)
	if problem != nil {
		return problem
	}
	return emit(ctx, buildRunReport(life, evidence))
}

func buildRunReport(life api.Lifecycle, evidence api.Evidence) runReport {
	report := runReport{Number: life.Number, RequestID: life.RequestID, Status: life.Status,
		Target: strings.Trim(life.Package+"/"+life.Function, "/"), Machine: life.Machine,
		CreatedAt: life.CreatedAt, QueuedMS: life.QueuedMS, ExecutionMS: life.ExecutionMS,
		Events: evidence.Events, Triage: evidence.Triage, Stages: []reportStage{}}
	if life.Phase == orchestrator.PhaseGPUWait || life.Phase == orchestrator.PhaseOwnerReconciliation {
		report.Waiting = PhaseCell(life)
	}
	created, _ := time.Parse(time.RFC3339Nano, life.CreatedAt)
	granted := map[string]int{}
	for _, event := range evidence.Events {
		switch event.Type {
		case "machine.gpu.grant":
			// Runtime's device lease for one call: the ordinals it granted, held until the
			// matching release.
			at, _ := time.Parse(time.RFC3339Nano, event.At)
			key, _ := event.Payload["key"].(string)
			granted[key] = len(report.Stages)
			report.Stages = append(report.Stages, reportStage{Name: "GPU " + key, Kind: "gpu",
				StartUnixMS: at.UnixMilli(), Detail: "ordinals " + fmt.Sprint(event.Payload["ordinals"])})
		case "machine.gpu.release":
			key, _ := event.Payload["key"].(string)
			if index, ok := granted[key]; ok {
				at, _ := time.Parse(time.RFC3339Nano, event.At)
				report.Stages[index].MS = float64(at.UnixMilli() - report.Stages[index].StartUnixMS)
			}
		case "request.preparing":
			report.Stages = append(report.Stages, preparingStage(event.Payload))
		case "request.log":
			if stage, ok := phaseStage(event.Payload); ok {
				report.Stages = append(report.Stages, stage)
			}
		case "request.outputs_fetched":
			outputs, _ := event.Payload["outputs"].([]any)
			moved := payloadInt(event.Payload["bytes"])
			report.Stages = append(report.Stages, reportStage{Name: "output transfer", Kind: "transfer",
				StartUnixMS: payloadInt(event.Payload["started_unix_ms"]), MS: float64(payloadInt(event.Payload["ms"])),
				Bytes: moved, Count: len(outputs), Detail: fmt.Sprintf("%d output(s), %s", len(outputs), units.Bytes(moved))})
		case "request.completed", "request.failed", "request.canceled":
			if ended, err := time.Parse(time.RFC3339Nano, event.At); err == nil && !created.IsZero() {
				report.WallMS = ended.Sub(created).Milliseconds()
			}
		}
	}
	var triage triageEvidence
	if len(evidence.Triage) > 0 && json.Unmarshal(evidence.Triage, &triage) == nil {
		execution := triage.Measurements.Execution
		report.Degree, report.Ranks = execution.Degree, execution.Ranks
		if boot := execution.Executor; boot.StartedUnixMS > 0 {
			report.Stages = append(report.Stages, setupStage("executor boot", boot.StartedUnixMS,
				boot.MS, created, topLegs(boot.Legs)))
		}
		if load := execution.Construction; load.PreparedUnixMS > 0 {
			report.Stages = append(report.Stages, setupStage("construction load/warm",
				load.PreparedUnixMS-int64(load.PrepareMS), load.PrepareMS, created,
				fmt.Sprintf("fill %s, warm %s", span(load.FillMS), span(load.WarmMS))))
		}
		for name, track := range triage.Measurements.Attribution.Stages {
			report.Stages = append(report.Stages, reportStage{Name: name, Kind: "inference",
				StartUnixMS: track.StartedUnixMS, MS: track.TotalMS, Count: track.Count})
		}
		for name, track := range triage.Measurements.Attribution.Steps {
			report.Steps = append(report.Steps, stepSummary(name, track))
		}
		sort.Slice(report.Steps, func(i, j int) bool { return report.Steps[i].Name < report.Steps[j].Name })
	}
	// Setup, GPU wait, then execution phases and inference, then transfer; within an
	// order, by start (unknown last).
	order := map[string]int{"setup": 0, "gpu": 1, "phase": 2, "inference": 2, "transfer": 3}
	sort.SliceStable(report.Stages, func(i, j int) bool {
		a, b := report.Stages[i], report.Stages[j]
		if order[a.Kind] != order[b.Kind] {
			return order[a.Kind] < order[b.Kind]
		}
		if (a.StartUnixMS == 0) != (b.StartUnixMS == 0) {
			return b.StartUnixMS == 0
		}
		if a.StartUnixMS != b.StartUnixMS {
			return a.StartUnixMS < b.StartUnixMS
		}
		return a.Name < b.Name
	})
	return report
}

func preparingStage(payload map[string]any) reportStage {
	stage, _ := payload["stage"].(string)
	name := map[string]string{"resolved": "resolve", "downloading": "download",
		"preparing": "package environment", "connect": "machine connection",
		"package_preparation": "package preparation", "model_defaults": "model defaults",
		"inputs": "input staging", "submit": "submission"}[stage]
	if name == "" {
		name = stage
	}
	detail, _ := payload["detail"].(string)
	row := reportStage{Name: name, Kind: "setup", StartUnixMS: payloadInt(payload["started_unix_ms"]),
		MS: float64(payloadInt(payload["ms"])), Detail: detail}
	if moved := payloadInt(payload["transferred_bytes"]); moved > 0 && stage == "downloading" {
		row.Bytes = moved
		row.Detail = units.Bytes(moved)
		if row.MS > 0 {
			row.Detail += fmt.Sprintf(" at %s/s", units.Bytes(int64(float64(moved)/(row.MS/1000))))
		}
		if origin, cached := payloadInt(payload["origin_bytes"]), payloadInt(payload["cached_bytes"]); origin+cached > 0 {
			row.Detail += fmt.Sprintf(" (origin %s, cached %s)", units.Bytes(origin), units.Bytes(cached))
		}
	}
	return row
}

// phaseStage reads one Runtime phase record: a named piece of execution work (a source
// download, a conversion, a checkpoint upload) with its elapsed time and, for byte work,
// what it moved and at what average rate.
func phaseStage(payload map[string]any) (reportStage, bool) {
	fields, _ := payload["fields"].(map[string]any)
	name, _ := fields["phase"].(string)
	elapsed, timed := number(fields["elapsed_ms"])
	if name == "" || !timed {
		return reportStage{}, false
	}
	start := payloadInt(fields["started_unix_ms"])
	if start == 0 {
		if at := payloadInt(payload["at_unix_ms"]); at > 0 {
			start = at - int64(elapsed)
		}
	}
	row := reportStage{Name: name, Kind: "phase", StartUnixMS: start, MS: elapsed}
	if total := payloadInt(fields["total_bytes"]); total > 0 {
		row.Bytes = payloadInt(fields["bytes"])
		row.Detail = units.Bytes(row.Bytes) + " of " + units.Bytes(total)
		if moved := payloadInt(fields["moved_bytes"]); moved != row.Bytes {
			row.Detail += ", " + units.Bytes(moved) + " moved"
		}
		if rate, ok := number(fields["rate_bytes_per_second"]); ok && rate > 0 {
			row.Detail += " at " + units.Bytes(int64(rate)) + "/s"
		}
	}
	if completed, ok := fields["completed"].(bool); ok && !completed {
		row.Detail = strings.TrimPrefix(row.Detail+"; did not complete", "; ")
	}
	return row, true
}

// setupStage marks setup that finished before the run was submitted as reused, not paid.
func setupStage(name string, start int64, ms float64, created time.Time, detail string) reportStage {
	if !created.IsZero() && start < created.UnixMilli() {
		detail = strings.TrimSuffix("reused, before this run; "+detail, "; ")
	}
	return reportStage{Name: name, Kind: "setup", StartUnixMS: start, MS: ms, Detail: detail}
}

func stepSummary(name string, track triageTrack) reportSteps {
	steps := reportSteps{Name: name, Count: track.Count, TotalMS: track.TotalMS,
		FirstMS: track.FirstMS, MinMS: track.MinMS, MaxMS: track.MaxMS, Series: track.Series,
		Dropped: track.SeriesDropped}
	if track.Count > 1 {
		steps.RestMeanMS = (track.TotalMS - track.FirstMS) / float64(track.Count-1)
	}
	return steps
}

// topLegs names the three most expensive legs; JSON keeps every one in the triage bundle.
func topLegs(legs map[string]float64) string {
	names := make([]string, 0, len(legs))
	for name := range legs {
		names = append(names, name)
	}
	sort.Slice(names, func(i, j int) bool { return legs[names[i]] > legs[names[j]] })
	parts := []string{}
	for _, name := range names[:min(3, len(names))] {
		parts = append(parts, name+" "+span(legs[name]))
	}
	return strings.Join(parts, ", ")
}

func payloadInt(value any) int64 {
	n, _ := number(value)
	return int64(n)
}

// span spells milliseconds as seconds, or minutes and seconds past a minute.
func span(ms float64) string {
	if ms < 1000 {
		return fmt.Sprintf("%.0fms", ms)
	}
	if ms >= 60_000 {
		return (time.Duration(ms) * time.Millisecond).Round(100 * time.Millisecond).String()
	}
	return fmt.Sprintf("%.1fs", ms/1000)
}

func (r runReport) Emit(w io.Writer, mode output.Mode) error {
	if !mode.Human || mode.JSON {
		// JSON whatever the machine format: the bundle's lists of nested objects are
		// outside what the TOON encoder round-trips.
		encoder := json.NewEncoder(w)
		encoder.SetEscapeHTML(false)
		return encoder.Encode(r)
	}
	created, _ := time.Parse(time.RFC3339Nano, r.CreatedAt)
	offset := func(unixMS int64) string {
		if unixMS == 0 || created.IsZero() {
			return "-"
		}
		if delta := float64(unixMS - created.UnixMilli()); delta >= 0 {
			return "+" + span(delta)
		}
		return "-" + span(-float64(unixMS-created.UnixMilli()))
	}
	fmt.Fprintf(w, "run %d %s  %s", r.Number, r.Status, r.Target)
	if r.Machine != "" {
		fmt.Fprintf(w, "  on %s", r.Machine)
	}
	if r.Waiting != "" {
		fmt.Fprintf(w, "\n%s", r.Waiting)
	}
	fmt.Fprintf(w, "\nqueued %s · execution %s", span(float64(r.QueuedMS)), span(float64(r.ExecutionMS)))
	if r.WallMS > 0 {
		fmt.Fprintf(w, " · wall %s", span(float64(r.WallMS)))
	}
	fmt.Fprintln(w)
	table := tabwriter.NewWriter(w, 0, 0, 2, ' ', 0)
	if len(r.Stages) > 0 {
		fmt.Fprintln(w)
		fmt.Fprintln(table, "STAGE\tKIND\tSTART\tTIME\tDETAIL")
		for _, stage := range r.Stages {
			detail := stage.Detail
			if stage.Kind == "inference" && stage.Count > 1 {
				detail = strings.TrimPrefix(detail+fmt.Sprintf(", %d×", stage.Count), ", ")
			}
			fmt.Fprintf(table, "%s\t%s\t%s\t%s\t%s\n", stage.Name, stage.Kind, offset(stage.StartUnixMS),
				span(stage.MS), detail)
		}
		table.Flush()
	}
	if len(r.Steps) > 0 {
		fmt.Fprintln(w)
	}
	for _, steps := range r.Steps {
		fmt.Fprintf(w, "steps %s: %d in %s", steps.Name, steps.Count, span(steps.TotalMS))
		if steps.Count > 1 {
			fmt.Fprintf(w, "; first %s, then mean %s (min %s, max %s)", span(steps.FirstMS),
				span(steps.RestMeanMS), span(steps.MinMS), span(steps.MaxMS))
		}
		fmt.Fprintln(w)
	}
	if len(r.Ranks) > 0 {
		fmt.Fprintf(w, "\nranks (degree %d)\n", r.Degree)
		fmt.Fprintln(table, "RANK\tGPU\tUUID\tPID\tSTART\tTIME\tATTENTION")
		for _, rank := range r.Ranks {
			gpu, start, took := "-", "-", "-"
			if rank.Ordinal >= 0 {
				gpu = fmt.Sprint(rank.Ordinal)
			}
			if rank.StartUS > 0 {
				start, took = offset(rank.StartUS/1000), span(float64(rank.EndUS-rank.StartUS)/1000)
			}
			fmt.Fprintf(table, "%d\t%s\t%s\t%d\t%s\t%s\t%s\n", rank.Rank, gpu,
				output.Elide(dash(rank.UUID), 17, mode.Full), rank.PID, start, took, rankAttention(rank))
		}
		table.Flush()
	}
	return nil
}

func dash(value string) string {
	if value == "" {
		return "-"
	}
	return value
}

func rankAttention(rank reportRank) string {
	seen := rank.Attention.Observed
	if seen == "" {
		seen = "unobserved"
	}
	if rank.Attention.Impl != "" {
		seen += " (" + rank.Attention.Impl + ")"
	}
	if rank.Attention.Requested != "" {
		return rank.Attention.Requested + " → " + seen
	}
	return seen
}
