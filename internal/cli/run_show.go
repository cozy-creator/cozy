package cli

import (
	"encoding/json"
	"fmt"
	"io"
	"slices"
	"sort"
	"strconv"
	"strings"
	"text/tabwriter"
	"time"

	"github.com/cozy-creator/cozy/internal/api"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/orchestrator"
	"github.com/cozy-creator/cozy/internal/output"
	"github.com/cozy-creator/cozy/internal/records"
	"github.com/cozy-creator/cozy/internal/units"
)

// runReport is one run's execution evidence: cold setup apart from inference, the per-step
// series, which GPUs ran it, and each child call it made. Every fact comes from the
// daemon's records (durable events and the kept triage bundle); JSON also carries both
// sources whole.
type runReport struct {
	Number    int64  `json:"number"`
	RequestID string `json:"request_id"`
	Status    string `json:"status"`
	Target    string `json:"target"`
	Machine   string `json:"machine,omitempty"`
	// Runtime is the cozy-runtime (and TensorFS) the run's own executor loaded.
	Runtime        string `json:"runtime,omitempty"`
	CreatedAt      string `json:"created_at"`
	QueuedMS       int64  `json:"queued_ms"`
	ExecutionMS    *int64 `json:"execution_ms"`
	ExecutionKnown bool   `json:"execution_known"`
	AttemptWallMS  int64  `json:"attempt_wall_ms,omitempty"`
	WallMS         int64  `json:"wall_ms,omitempty"`
	Waiting        string `json:"waiting,omitempty"`
	ErrorType      string `json:"error_type,omitempty"`
	Error          string `json:"error,omitempty"`
	// Warnings are what the run reported without failing: its admission's and its machine's.
	Warnings []records.Warning `json:"warnings,omitempty"`
	Stages   []reportStage     `json:"stages"`
	Steps    []reportSteps     `json:"steps,omitempty"`
	Degree   int               `json:"degree,omitempty"`
	GPUs     []reportGPU       `json:"gpus,omitempty"`
	Calls    []reportCall      `json:"calls,omitempty"`
	// Resolved is what the machine installed and which checkpoint each Model slot ran: the
	// run's reproducible identity, recorded by the machine that chose it.
	Resolved json.RawMessage     `json:"resolved,omitempty"`
	Events   []api.EvidenceEvent `json:"events"`
	Triage   json.RawMessage     `json:"triage,omitempty"`
	// Result is the run's inline result once this host holds it.
	Result any `json:"result,omitempty"`
	// Products are the run's output log: what it made, as it made it. A superseded one is
	// an earlier revision of a single output.
	// Output is the run's items at their current revision.
	Output []records.OutputItem `json:"output,omitempty"`
	// CollectionPending names why a finished result still waits on its machine.
	CollectionPending string `json:"collection_pending,omitempty"`
	CollectionError   string `json:"collection_error,omitempty"`
}

type reportStage struct {
	Name        string  `json:"name"`
	Kind        string  `json:"kind"` // setup, download, gpu, wait, phase, inference or transfer
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

// reportCall is one child call the run made: the function it ran under the author's
// label, the GPUs it held, its timeline and its per-step times. Runtime records a call
// on the run's journal when it settles; until then only its phases and grants show.
type reportCall struct {
	callTiming
	Phase       string `json:"phase,omitempty"`
	timingAt    int64
	Number      int           `json:"number"` // its place in the run's call order, from 1
	Request     string        `json:"request"`
	Parent      string        `json:"parent,omitempty"`
	Index       int           `json:"index"` // its index among its parent's calls
	Attempt     int64         `json:"attempt,omitempty"`
	Module      string        `json:"module,omitempty"`
	Function    string        `json:"function,omitempty"`
	Label       string        `json:"label,omitempty"`
	Status      string        `json:"status,omitempty"` // empty until Runtime records it settled
	Error       string        `json:"error,omitempty"`
	Runtime     string        `json:"runtime,omitempty"` // the SDK its executor loaded
	GPUs        []reportGPU   `json:"gpus,omitempty"`    // its grants' cards, then its release's records
	StartUnixMS int64         `json:"start_unix_ms,omitempty"`
	MS          float64       `json:"ms"`
	Stages      []reportStage `json:"stages"`
	Steps       []reportSteps `json:"steps,omitempty"`

	timed []reportStage // its latest record's attribution stages, beside Stages until sorted
}

// reportGPU is Runtime's record of one GPU an execution ran on, read tolerantly: `gpu` is
// the number nvidia-smi shows, -1 when unknown. A grant's row names only the card.
type reportGPU struct {
	GPU       int    `json:"gpu"`
	PID       int    `json:"pid,omitempty"`
	UUID      string `json:"uuid,omitempty"`
	Arch      string `json:"arch,omitempty"`
	StartUS   int64  `json:"start_us"`
	EndUS     int64  `json:"end_us"`
	Attention struct {
		Requested string         `json:"requested"`
		Observed  string         `json:"observed"`
		Impl      string         `json:"impl"`
		Kernels   []reportKernel `json:"kernels,omitempty"` // every kernel of its chains
	} `json:"attention"`
}

// legacyRank is a Runtime's `ranks` row, all a Runtime before `gpus` sends: its `ordinal`
// is the GPU's number. Drop once no deployed Runtime lacks `gpus`.
type legacyRank struct {
	reportGPU
	Ordinal int `json:"ordinal"`
}

// gpuRecords prefers Runtime's `gpus` and falls back to an older one's `ranks`.
func gpuRecords(gpus []reportGPU, ranks []legacyRank) []reportGPU {
	if len(gpus) > 0 {
		return gpus
	}
	for _, rank := range ranks {
		rank.GPU = rank.Ordinal
		gpus = append(gpus, rank.reportGPU)
	}
	return gpus
}

// reportKernel is one kernel of a GPU's attention chains: whether it served, and if not
// why (compiling, failed, absent, unsupported), with what compiling it cost this machine.
type reportKernel struct {
	Kernel    string   `json:"kernel"`
	State     string   `json:"state"`
	Served    bool     `json:"served"`
	Progress  *float64 `json:"progress,omitempty"`
	CompileMS float64  `json:"compile_ms,omitempty"`
	Detail    string   `json:"detail,omitempty"`
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
	// Bytes is what the span moved: an effect's upload.
	Bytes int64 `json:"bytes"`
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
			GPUs         []reportGPU  `json:"gpus"`
			Ranks        []legacyRank `json:"ranks"`
			Executor     triageSetup  `json:"executor"`
			Construction triageSetup  `json:"construction"`
		} `json:"execution"`
	} `json:"measurements"`
}

// The evidence events run show reads, each decoded once into its own type. A payload
// that does not decode is skipped: one odd record never hides the rest.

// gpuEvent is Runtime's grant or release of one call attempt's devices (`request#attempt`):
// `gpus` names each card, and a release's rows are its execution records. A Runtime before
// `gpus` sends only `ordinals` (the same numbers on a whole-pod worker) and `ranks`.
type gpuEvent struct {
	Key      string       `json:"key"`
	Ordinals []int        `json:"ordinals"`
	GPUs     []reportGPU  `json:"gpus"`
	Ranks    []legacyRank `json:"ranks"`
}

// cards are the GPUs the event names.
func (e gpuEvent) cards() []reportGPU {
	if len(e.GPUs) > 0 {
		return e.GPUs
	}
	cards := make([]reportGPU, len(e.Ordinals))
	for i, ordinal := range e.Ordinals {
		cards[i] = reportGPU{GPU: ordinal}
	}
	return cards
}

type preparingEvent struct {
	Stage            string `json:"stage"`
	Detail           string `json:"detail"`
	MS               int64  `json:"ms"`
	StartedUnixMS    int64  `json:"started_unix_ms"`
	TransferredBytes int64  `json:"transferred_bytes"`
	OriginBytes      int64  `json:"origin_bytes"`
	CachedBytes      int64  `json:"cached_bytes"`
}

// logEvent is one Runtime log record. A phase record (a source download, a conversion, a
// call's input check) names its phase and elapsed time, and its call when it is one's.
type logEvent struct {
	Name     string `json:"name"`
	AtUnixMS int64  `json:"at_unix_ms"`
	Fields   struct {
		Phase         string   `json:"phase"`
		ChildRequest  string   `json:"child_request"`
		ElapsedMS     *float64 `json:"elapsed_ms"`
		StartedUnixMS int64    `json:"started_unix_ms"`
		TotalBytes    int64    `json:"total_bytes"`
		Bytes         int64    `json:"bytes"`
		MovedBytes    int64    `json:"moved_bytes"`
		Rate          float64  `json:"rate_bytes_per_second"`
		Completed     *bool    `json:"completed"`
		Detail        string   `json:"detail"` // what it found: an executor start's legs, the kernels' compiles
		// Runtime's `model fetch` and `model wait` records: one model's pull, and a call held
		// for its callee's model preparation.
		Event      string  `json:"event"`
		Model      string  `json:"model"`
		Entrypoint string  `json:"entrypoint"`
		Prefetch   bool    `json:"prefetch"`
		Step       string  `json:"step"`
		Call       string  `json:"call"`
		WaitedMS   float64 `json:"waited_ms"`
	} `json:"fields"`
}

type outputsEvent struct {
	Outputs       []json.RawMessage `json:"outputs"`
	Bytes         int64             `json:"bytes"`
	StartedUnixMS int64             `json:"started_unix_ms"`
	MS            int64             `json:"ms"`
}

// publicationModule is the module of Runtime's effect calls (upload, publish, assessment).
const publicationModule = "cozy_runtime.author.publication"

// callEvent is Runtime's record of one settled call (machine_calls.CallRecord): a child
// call, or an effect such as a checkpoint upload.
type callEvent struct {
	callTiming
	Request      string                 `json:"request"`
	Parent       string                 `json:"parent"`
	Index        int                    `json:"index"`
	Attempt      int64                  `json:"attempt"`
	Module       string                 `json:"module"`
	Export       string                 `json:"export"`
	Label        string                 `json:"label"`
	Status       string                 `json:"status"`
	Error        string                 `json:"error"`
	CalledUnixMS int64                  `json:"called_unix_ms"`
	Stages       map[string]triageTrack `json:"stages"`
	Steps        map[string]triageTrack `json:"steps"`
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
	report := buildRunReport(life, evidence)
	if selector := strings.TrimSpace(ctx.Inv.Value("--call")); selector != "" {
		call, problem := report.call(selector)
		if problem != nil {
			return problem
		}
		return emit(ctx, callReport{run: report, reportCall: call})
	}
	return emit(ctx, report)
}

func buildRunReport(life api.Lifecycle, evidence api.Evidence) runReport {
	report := runReport{Number: life.Number, RequestID: life.RequestID, Status: life.Status,
		Target: strings.Trim(life.Package+"/"+life.Function, "/"), Machine: life.Machine,
		ErrorType: life.ErrorType, Error: life.Error,
		CreatedAt: life.CreatedAt, QueuedMS: life.QueuedMS,
		ExecutionKnown: life.ExecutionKnown || life.MachineExecution == nil, AttemptWallMS: life.AttemptWallMS,
		Events: evidence.Events, Triage: evidence.Triage, Stages: []reportStage{}}
	if report.ExecutionKnown {
		measured := life.ExecutionMS
		report.ExecutionMS = &measured
	}
	switch {
	case life.Phase == orchestrator.PhaseGPUWait || life.Phase == orchestrator.PhaseOwnerReconciliation:
		report.Waiting = PhaseCell(life)
	case life.Status == "queued" && life.RentalBoot != nil:
		report.Waiting = describeBoot(bootMachine(life), life.RentalBoot, time.Now()).line(bootMachine(life))
	case life.Status == "queued" && life.WaitReason != "":
		report.Waiting = "waiting: " + life.WaitReason
	}
	report.Result = life.Result
	report.Output = life.Output
	if view := life.MachineExecution; view != nil && !view.Collected && view.CollectionRefused != "" {
		report.CollectionPending, report.CollectionError = view.CollectionRefused, view.ObservationError
	}
	created, _ := time.Parse(time.RFC3339Nano, life.CreatedAt)
	calls := map[string]*reportCall{}
	call := func(request string) *reportCall {
		if calls[request] == nil {
			calls[request] = &reportCall{Request: request, Stages: []reportStage{}}
		}
		return calls[request]
	}
	type grant struct {
		stages *[]reportStage
		index  int
	}
	granted := map[string]grant{}        // an open GPU grant's stage, by its key
	phases := map[string][]reportStage{} // phase records naming a child request, by it
	fetched := false                     // the Runtime recorded each model's pull itself
	var started time.Time                // when the machine began the run's own execution
	for _, event := range evidence.Events {
		at, _ := time.Parse(time.RFC3339Nano, event.At)
		switch event.Type {
		case "run.in_progress":
			if started.IsZero() {
				started = at
			}
		case "machine.gpu.grant":
			// Runtime's device lease for one call attempt, held until the matching release.
			var lease gpuEvent
			if json.Unmarshal(event.Payload, &lease) != nil {
				continue
			}
			stage := reportStage{Name: "GPU", Kind: "gpu", StartUnixMS: at.UnixMilli(),
				Detail: gpuNames(lease.cards())}
			stages := &report.Stages
			if request, _, _ := strings.Cut(lease.Key, "#"); request != life.RequestID {
				c := call(request)
				c.GPUs, stages = seat(c.GPUs, lease.cards(), false), &c.Stages
			} else {
				stage.Name += " " + lease.Key
			}
			*stages = append(*stages, stage)
			granted[lease.Key] = grant{stages, len(*stages) - 1}
		case "machine.gpu.release":
			var release gpuEvent
			if json.Unmarshal(event.Payload, &release) != nil {
				continue
			}
			if open, ok := granted[release.Key]; ok {
				stage := &(*open.stages)[open.index]
				stage.MS = float64(at.UnixMilli() - stage.StartUnixMS)
			}
			if request, _, _ := strings.Cut(release.Key, "#"); request != life.RequestID {
				c := call(request)
				c.GPUs = seat(c.GPUs, gpuRecords(release.GPUs, release.Ranks), true)
			}
			delete(granted, release.Key)
		case "machine.call.phase":
			var phase callPhaseEvent
			if json.Unmarshal(event.Payload, &phase) != nil || phase.Request == "" || !phase.valid() {
				continue
			}
			c := call(phase.Request)
			if !phase.follows(c.Attempt, c.timingAt, c.Phase) {
				continue
			}
			c.Parent, c.Index, c.Attempt, c.Module, c.Function, c.Label = phase.Parent, phase.Index, phase.Attempt, phase.Module, phase.Export, phase.Label
			c.Phase, c.Status, c.callTiming, c.timingAt = phase.Phase, phase.Status, phase.callTiming, phase.AtUnixMS
			c.StartUnixMS, c.MS = phase.CalledUnixMS, float64(max(phase.AtUnixMS-phase.CalledUnixMS, 0))
		case "machine.call":
			var record callEvent
			if json.Unmarshal(event.Payload, &record) != nil || record.Request == "" {
				continue
			}
			c := call(record.Request)
			if record.Attempt < c.Attempt {
				continue
			}
			c.Parent, c.Index, c.Attempt, c.Module, c.Function = record.Parent, record.Index, record.Attempt, record.Module, record.Export
			c.Label, c.Status, c.Error = record.Label, record.Status, record.Error
			c.StartUnixMS = record.CalledUnixMS
			c.MS = float64(at.UnixMilli() - record.CalledUnixMS)
			c.callTiming = record.callTiming
			c.Phase, c.timingAt = "terminal", at.UnixMilli()
			c.timed = nil
			for name, track := range record.Stages {
				stage := reportStage{Name: name, Kind: "inference",
					StartUnixMS: track.StartedUnixMS, MS: track.TotalMS, Count: track.Count}
				if record.Module == publicationModule {
					stage.Kind = "phase" // an effect runs no model
				}
				if track.Bytes > 0 {
					// An effect's upload: what it moved, and how fast.
					stage.Kind, stage.Bytes = "transfer", track.Bytes
					stage.Detail = units.Bytes(track.Bytes)
					if track.TotalMS > 0 {
						stage.Detail += fmt.Sprintf(", %s/s", units.Bytes(int64(float64(track.Bytes)/(track.TotalMS/1000))))
					}
				}
				c.timed = append(c.timed, stage)
			}
			c.Steps = stepSummaries(record.Steps)
		case "machine.resolved":
			report.Resolved = event.Payload
		case "machine.executor":
			var loaded executorEvent
			if json.Unmarshal(event.Payload, &loaded) != nil || loaded.Runtime == "" {
				continue
			}
			if loaded.Request == life.RequestID || loaded.Request == "" {
				report.Runtime = loaded.String()
			} else {
				call(loaded.Request).Runtime = loaded.String()
			}
		case "machine.warning", "request.warning":
			var warning records.Warning
			if json.Unmarshal(event.Payload, &warning) == nil {
				report.Warnings = append(report.Warnings, warning)
			}
		case "request.preparing":
			var preparing preparingEvent
			if json.Unmarshal(event.Payload, &preparing) == nil {
				report.Stages = append(report.Stages, preparingStage(preparing))
			}
		case "request.log":
			var record logEvent
			if json.Unmarshal(event.Payload, &record) != nil {
				continue
			}
			switch {
			case record.Name == "model fetch" && record.Fields.Event == "end":
				fetched = true
				report.Stages = append(report.Stages, fetchStage(record))
			case record.Name == "model wait" && record.Fields.Event == "end" && record.Fields.Call != "":
				c := call(record.Fields.Call)
				c.Stages = append(c.Stages, reportStage{Name: "waiting for model weights", Kind: "wait",
					StartUnixMS: record.AtUnixMS - int64(record.Fields.WaitedMS), MS: record.Fields.WaitedMS,
					Detail: strings.TrimSpace(record.Fields.Entrypoint + " weights, in " + record.Fields.Step)})
			default:
				if stage, ok := phaseStage(record); ok {
					phases[record.Fields.ChildRequest] = append(phases[record.Fields.ChildRequest], stage)
				}
			}
		case "request.outputs_fetched":
			var fetched outputsEvent
			if json.Unmarshal(event.Payload, &fetched) != nil {
				continue
			}
			report.Stages = append(report.Stages, reportStage{Name: "output transfer", Kind: "transfer",
				StartUnixMS: fetched.StartedUnixMS, MS: float64(fetched.MS), Bytes: fetched.Bytes, Count: len(fetched.Outputs),
				Detail: fmt.Sprintf("%d output(s), %s", len(fetched.Outputs), units.Bytes(fetched.Bytes))})
		case "run.completed", "run.failed", "run.canceled":
			if !at.IsZero() && !created.IsZero() {
				report.WallMS = at.Sub(created).Milliseconds()
			}
		}
	}
	// A call's phases are its own; a shared preparation's (a model download several calls
	// wait on) and the run's are the run's. A Runtime that records each model's pull has
	// its download phases as those rows, with their bytes and rates.
	for child, stages := range phases {
		if fetched {
			stages = slices.DeleteFunc(stages, func(stage reportStage) bool { return stage.Name == "Downloading model weights" })
		}
		if c := calls[child]; c != nil && child != "" {
			c.Stages = append(c.Stages, stages...)
		} else {
			report.Stages = append(report.Stages, stages...)
		}
	}
	var triage triageEvidence
	if len(evidence.Triage) > 0 && json.Unmarshal(evidence.Triage, &triage) == nil {
		execution := triage.Measurements.Execution
		report.Degree, report.GPUs = execution.Degree, gpuRecords(execution.GPUs, execution.Ranks)
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
		report.Steps = stepSummaries(triage.Measurements.Attribution.Steps)
	}
	sortStages(report.Stages)
	for _, c := range calls {
		// A call's own timeline reads in the order it happened.
		c.Stages = append(c.Stages, c.timed...)
		sort.SliceStable(c.Stages, func(i, j int) bool { return c.Stages[i].StartUnixMS < c.Stages[j].StartUnixMS })
		if c.StartUnixMS == 0 && len(c.Stages) > 0 {
			// Settled before Runtime recorded calls, or still running: its first and last
			// recorded phase or grant bound it.
			c.StartUnixMS = c.Stages[0].StartUnixMS
			for _, stage := range c.Stages {
				c.MS = max(c.MS, float64(stage.StartUnixMS-c.StartUnixMS)+stage.MS)
			}
		}
		report.Calls = append(report.Calls, *c)
	}
	sort.Slice(report.Calls, func(i, j int) bool {
		a, b := report.Calls[i], report.Calls[j]
		if a.StartUnixMS != b.StartUnixMS {
			return a.StartUnixMS < b.StartUnixMS
		}
		return a.Request < b.Request
	})
	// The run's own execution is call 0: what it ran, on which GPUs, for how long. Its calls
	// keep their numbers from 1.
	if !started.IsZero() {
		status := life.Status
		if status == "completed" {
			status = "succeeded" // a call's status is Runtime's word, as its calls' rows say it
		}
		root := reportCall{Request: life.RequestID, Module: life.Package, Function: life.Function, Label: "this run",
			Status: status, GPUs: report.GPUs, StartUnixMS: started.UnixMilli(), MS: float64(life.ExecutionMS),
			Steps: report.Steps, Stages: []reportStage{}}
		for _, stage := range report.Stages {
			if stage.Kind == "inference" {
				root.Stages = append(root.Stages, stage)
			}
		}
		for i := range report.Calls {
			report.Calls[i].Number = i + 1
		}
		report.Calls = append([]reportCall{root}, report.Calls...)
		return report
	}
	for i := range report.Calls {
		report.Calls[i].Number = i + 1
	}
	return report
}

// sortStages orders setup, GPU wait, then execution phases and inference, then transfer;
// within an order, by start (unknown last).
func sortStages(stages []reportStage) {
	order := map[string]int{"setup": 0, "download": 1, "gpu": 1, "phase": 2, "inference": 2, "transfer": 3}
	sort.SliceStable(stages, func(i, j int) bool {
		a, b := stages[i], stages[j]
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
}

// seat puts rows into a call's GPUs by number, in number order: a grant's card is added
// once, and a release's execution record replaces its card's row.
func seat(held, rows []reportGPU, records bool) []reportGPU {
	for _, row := range rows {
		switch i := slices.IndexFunc(held, func(h reportGPU) bool { return h.GPU == row.GPU }); {
		case i < 0:
			held = append(held, row)
		case records:
			held[i] = row
		}
	}
	slices.SortFunc(held, func(a, b reportGPU) int { return a.GPU - b.GPU })
	return held
}

// call selects one of the run's calls by its number, its request id (with or without
// `call-`, or a unique prefix), its label, or its function when only one call ran it.
func (r runReport) call(selector string) (reportCall, *exit.Error) {
	if number, err := strconv.Atoi(strings.TrimPrefix(selector, "#")); err == nil {
		for _, c := range r.Calls {
			if c.Number == number {
				return c, nil
			}
		}
		return reportCall{}, exit.New(exit.NotFound, "run %s has %d call(s); there is no call %d",
			runReference(r.Number, r.RequestID), r.childCalls(), number)
	}
	id := "call-" + strings.TrimPrefix(selector, "call-")
	for _, match := range []func(reportCall) bool{
		func(c reportCall) bool { return c.Request == id },
		func(c reportCall) bool { return strings.EqualFold(c.Label, selector) },
		func(c reportCall) bool { return strings.HasPrefix(c.Request, id) },
		func(c reportCall) bool { return c.Function == selector },
	} {
		var found []reportCall
		for _, c := range r.Calls {
			if match(c) {
				found = append(found, c)
			}
		}
		if len(found) == 1 {
			return found[0], nil
		}
		if len(found) > 1 {
			numbers := make([]string, len(found))
			for i, c := range found {
				numbers[i] = strconv.Itoa(c.Number)
			}
			return reportCall{}, exit.Usagef("%q names %d calls of run %s (%s); pass its number",
				selector, len(found), runReference(r.Number, r.RequestID), strings.Join(numbers, ", "))
		}
	}
	return reportCall{}, exit.New(exit.NotFound, "run %s has no call %q; `cozy run show %s` lists its calls",
		runReference(r.Number, r.RequestID), selector, runReference(r.Number, r.RequestID))
}

func preparingStage(event preparingEvent) reportStage {
	name := map[string]string{"resolved": "resolve", "downloading": "download",
		"preparing": "package environment", "connect": "machine connection",
		"package_preparation": "package preparation", "machine": "machine preparation", "model_defaults": "model defaults",
		"inputs": "input staging", "submit": "submission"}[event.Stage]
	if name == "" {
		name = event.Stage
	}
	row := reportStage{Name: name, Kind: "setup", StartUnixMS: event.StartedUnixMS, MS: float64(event.MS), Detail: event.Detail}
	if moved := event.TransferredBytes; moved > 0 && event.Stage == "downloading" {
		row.Bytes = moved
		row.Detail = units.Bytes(moved)
		if row.MS > 0 {
			row.Detail += fmt.Sprintf(" at %s/s", units.Bytes(int64(float64(moved)/(row.MS/1000))))
		}
		if event.OriginBytes+event.CachedBytes > 0 {
			row.Detail += fmt.Sprintf(" (origin %s, cached %s)", units.Bytes(event.OriginBytes), units.Bytes(event.CachedBytes))
		}
	}
	return row
}

// phaseStage reads one Runtime phase record: a named piece of execution work (a source
// download, a conversion, a checkpoint upload) with its elapsed time and, for byte work,
// what it moved and at what average rate.
func phaseStage(record logEvent) (reportStage, bool) {
	fields := record.Fields
	if fields.Phase == "" || fields.ElapsedMS == nil {
		return reportStage{}, false
	}
	elapsed := *fields.ElapsedMS
	start := fields.StartedUnixMS
	if start == 0 && record.AtUnixMS > 0 {
		start = record.AtUnixMS - int64(elapsed)
	}
	row := reportStage{Name: fields.Phase, Kind: "phase", StartUnixMS: start, MS: elapsed}
	if fields.TotalBytes > 0 {
		row.Bytes = fields.Bytes
		row.Detail = units.Bytes(row.Bytes) + " of " + units.Bytes(fields.TotalBytes)
		if fields.MovedBytes != row.Bytes {
			row.Detail += ", " + units.Bytes(fields.MovedBytes) + " moved"
		}
		if fields.Rate > 0 {
			row.Detail += " at " + units.Bytes(int64(fields.Rate)) + "/s"
		}
	}
	if row.Detail == "" {
		row.Detail = fields.Detail
	}
	if fields.Completed != nil && !*fields.Completed {
		row.Detail = strings.TrimPrefix(row.Detail+"; did not complete", "; ")
	}
	return row, true
}

// fetchStage is one model's pull (Runtime's `model fetch` end record): its bytes, time and
// rate, and the step it held or that it was a prefetch.
func fetchStage(record logEvent) reportStage {
	fields := record.Fields
	elapsed := 0.0
	if fields.ElapsedMS != nil {
		elapsed = *fields.ElapsedMS
	}
	row := reportStage{Name: "download " + fields.Model, Kind: "download", StartUnixMS: fields.StartedUnixMS,
		MS: elapsed, Bytes: fields.Bytes}
	parts := []string{}
	if fields.Bytes > 0 {
		moved := units.Bytes(fields.Bytes)
		if fields.Rate > 0 {
			moved += " at " + units.Bytes(int64(fields.Rate)) + "/s"
		}
		parts = append(parts, moved)
	}
	switch {
	case fields.Prefetch:
		parts = append(parts, "prefetch for "+fields.Entrypoint)
	case fields.Step != "":
		parts = append(parts, "for "+fields.Entrypoint+", held "+fields.Step)
	}
	if fields.Completed != nil && !*fields.Completed {
		parts = append(parts, "did not complete")
	}
	row.Detail = strings.Join(parts, "; ")
	return row
}

// setupStage marks setup that finished before the run was submitted as reused, not paid.
func setupStage(name string, start int64, ms float64, created time.Time, detail string) reportStage {
	if !created.IsZero() && start < created.UnixMilli() {
		detail = strings.TrimSuffix("reused, before this run; "+detail, "; ")
	}
	return reportStage{Name: name, Kind: "setup", StartUnixMS: start, MS: ms, Detail: detail}
}

// stepSummaries are the step tracks by name.
func stepSummaries(tracks map[string]triageTrack) []reportSteps {
	var out []reportSteps
	for name, track := range tracks {
		steps := reportSteps{Name: name, Count: track.Count, TotalMS: track.TotalMS,
			FirstMS: track.FirstMS, MinMS: track.MinMS, MaxMS: track.MaxMS, Series: track.Series,
			Dropped: track.SeriesDropped}
		if track.Count > 1 {
			steps.RestMeanMS = (track.TotalMS - track.FirstMS) / float64(track.Count-1)
		}
		out = append(out, steps)
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Name < out[j].Name })
	return out
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

// offsets spells an instant as its offset from the run's creation.
func (r runReport) offsets() func(int64) string {
	created, _ := time.Parse(time.RFC3339Nano, r.CreatedAt)
	return func(unixMS int64) string {
		if unixMS == 0 || created.IsZero() {
			return "-"
		}
		if delta := float64(unixMS - created.UnixMilli()); delta >= 0 {
			return "+" + span(delta)
		}
		return "-" + span(-float64(unixMS-created.UnixMilli()))
	}
}

func (r runReport) Emit(w io.Writer, mode output.Mode) error {
	if !mode.Human || mode.JSON {
		return emitJSON(w, r)
	}
	offset := r.offsets()
	fmt.Fprintf(w, "run %d %s  %s", r.Number, r.Status, r.Target)
	if r.Machine != "" {
		fmt.Fprintf(w, "  on %s", r.Machine)
	}
	if r.Runtime != "" {
		fmt.Fprintf(w, "  %s", r.Runtime)
	}
	if r.Waiting != "" {
		fmt.Fprintf(w, "\n%s", r.Waiting)
	}
	if r.Error != "" {
		fmt.Fprintf(w, "\nerror %s: %s", r.ErrorType, r.Error)
	}
	for _, warning := range r.Warnings {
		if warning.Code != "" {
			fmt.Fprintf(w, "\nwarning %s: %s", warning.Code, warning.Message)
		} else {
			fmt.Fprintf(w, "\nwarning: %s", warning.Message)
		}
	}
	execution := "—"
	if r.ExecutionMS != nil {
		execution = span(float64(*r.ExecutionMS))
	}
	fmt.Fprintf(w, "\nqueued %s · execution %s", span(float64(r.QueuedMS)), execution)
	if r.WallMS > 0 {
		fmt.Fprintf(w, " · wall %s", span(float64(r.WallMS)))
	}
	fmt.Fprintln(w)
	if r.Result != nil {
		if raw, err := json.Marshal(r.Result); err == nil {
			fmt.Fprintf(w, "result %s\n", output.Elide(string(raw), 2000, mode.Full))
		}
	}
	if r.CollectionPending != "" {
		fmt.Fprintf(w, "collection pending: %s — %s\n", r.CollectionPending, r.CollectionError)
	}
	table := tabwriter.NewWriter(w, 0, 0, 2, ' ', 0)
	if len(r.Output) > 0 {
		fmt.Fprintln(w)
		fmt.Fprintln(table, "OUTPUT\tREV\tSIZE\tSTATUS\tFILE")
		for _, item := range r.Output {
			name := item.Name
			if item.Index > 0 {
				name = fmt.Sprintf("%s %d", item.Name, item.Index)
			}
			fmt.Fprintf(table, "%s\tr%d\t%s\t%s\t%s\n", name, item.Rev, output.Bytes(item.Length), item.Status, item.Path)
		}
		_ = table.Flush()
	}
	emitTimeline(w, table, r.Stages, r.Steps, offset, mode.Full)
	emitGPUs(w, table, r.GPUs, offset, mode.Full)
	if len(r.Calls) == 0 {
		return nil
	}
	fmt.Fprintf(w, "\ncalls (%d)\n", r.childCalls())
	fmt.Fprintln(table, "#\tCALL\tFUNCTION\tSTATUS\tGPUS\tSTART\tWALL\tSTEPS\tRUNTIME\tATTENTION")
	for _, c := range r.Calls {
		fmt.Fprintf(table, "%d\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n", c.Number, clip(c.name(), 40), dash(c.Function),
			dash(c.state()), dash(gpuList(c.GPUs)), offset(c.StartUnixMS), span(c.MS), dash(stepsCell(c.Steps)),
			dash(c.Runtime), dash(clip(served(c.GPUs), 40)))
	}
	table.Flush()
	for _, c := range r.Calls {
		if c.callTiming.present() {
			fmt.Fprintf(w, "  call %d: %s\n", c.Number, callTimingText(callPhaseEvent{Phase: "terminal", callTiming: c.callTiming}, time.Time{}, true))
		}
	}
	if !mode.Full {
		fmt.Fprintf(w, "\n`cozy run show %s --call <#>` shows one call's stages, steps, GPUs and attention kernels; --full shows every call's.\n",
			runReference(r.Number, r.RequestID))
		return nil
	}
	for _, c := range r.Calls {
		fmt.Fprintln(w)
		c.emit(w, table, r.childCalls(), offset, true)
	}
	return nil
}

// childCalls counts the calls the run made, not its own execution (call 0).
func (r runReport) childCalls() int {
	if len(r.Calls) > 0 && r.Calls[0].Number == 0 {
		return len(r.Calls) - 1
	}
	return len(r.Calls)
}

// callReport is one call of a run (`cozy run show <run> --call <call>`).
type callReport struct {
	run runReport
	reportCall
}

func (c callReport) Emit(w io.Writer, mode output.Mode) error {
	if !mode.Human || mode.JSON {
		return emitJSON(w, c.reportCall)
	}
	fmt.Fprintf(w, "run %d %s  %s", c.run.Number, c.run.Status, c.run.Target)
	if c.run.Machine != "" {
		fmt.Fprintf(w, "  on %s", c.run.Machine)
	}
	fmt.Fprintln(w)
	c.emit(w, tabwriter.NewWriter(w, 0, 0, 2, ' ', 0), c.run.childCalls(), c.run.offsets(), mode.Full)
	return nil
}

// emit prints one call: what it ran, its timeline on the run's clock, its steps and GPUs.
func (c reportCall) emit(w io.Writer, table *tabwriter.Writer, calls int, offset func(int64) string, full bool) {
	fmt.Fprintf(w, "call %d of %d  %s", c.Number, calls, c.name())
	if c.Function != "" {
		fmt.Fprintf(w, "  %s", c.Function)
	}
	fmt.Fprintf(w, "  %s  wall %s\n", dash(c.state()), span(c.MS))
	if c.callTiming.present() {
		fmt.Fprintln(w, callTimingText(callPhaseEvent{Phase: "terminal", callTiming: c.callTiming}, time.Time{}, true))
	}
	fmt.Fprintf(w, "request %s", c.Request)
	if c.Parent != "" {
		fmt.Fprintf(w, " · parent %s · index %d · attempt %d", c.Parent, c.Index, c.Attempt)
	}
	if len(c.GPUs) > 0 {
		fmt.Fprintf(w, " · GPUs %s", gpuList(c.GPUs))
	}
	if c.Runtime != "" {
		fmt.Fprintf(w, " · %s", c.Runtime)
	}
	fmt.Fprintln(w)
	if c.Error != "" {
		fmt.Fprintf(w, "error: %s\n", c.Error)
	}
	emitTimeline(w, table, c.Stages, c.Steps, offset, full)
	emitGPUs(w, table, c.GPUs, offset, full)
}

func (c reportCall) state() string {
	if c.Status != "" {
		return c.Status
	}
	return c.Phase
}

// name is the call's label, else its request id's first eight digits.
func (c reportCall) name() string {
	if c.Label != "" {
		return c.Label
	}
	return clip(c.Request, len("call-")+9)
}

func emitTimeline(w io.Writer, table *tabwriter.Writer, stages []reportStage, steps []reportSteps, offset func(int64) string, full bool) {
	if len(stages) > 0 {
		fmt.Fprintln(w)
		fmt.Fprintln(table, "STAGE\tKIND\tSTART\tTIME\tDETAIL")
		for _, stage := range stages {
			detail := stage.Detail
			if stage.Kind == "inference" && stage.Count > 1 {
				detail = strings.TrimPrefix(detail+fmt.Sprintf(", %d×", stage.Count), ", ")
			}
			if stage.Kind == "phase" {
				detail = output.Elide(detail, 120, full)
			}
			fmt.Fprintf(table, "%s\t%s\t%s\t%s\t%s\n", stage.Name, stage.Kind, offset(stage.StartUnixMS),
				span(stage.MS), detail)
		}
		table.Flush()
	}
	if len(steps) > 0 {
		fmt.Fprintln(w)
	}
	for _, track := range steps {
		fmt.Fprintf(w, "steps %s: %d in %s", track.Name, track.Count, span(track.TotalMS))
		if track.Count > 1 {
			fmt.Fprintf(w, "; first %s, then mean %s (min %s, max %s)", span(track.FirstMS),
				span(track.RestMeanMS), span(track.MinMS), span(track.MaxMS))
		}
		fmt.Fprintln(w)
	}
}

// emitGPUs prints the GPUs an execution ran on, by the number nvidia-smi shows. A card held
// with no process on it (a one-process call on a wider grant) and a CPU process are not.
func emitGPUs(w io.Writer, table *tabwriter.Writer, gpus []reportGPU, offset func(int64) string, full bool) {
	gpus = slices.DeleteFunc(slices.Clone(gpus), func(g reportGPU) bool {
		return g.PID <= 0 || g.GPU < 0 && g.UUID == ""
	})
	if len(gpus) == 0 {
		return
	}
	fmt.Fprintf(w, "\nGPUs (%d)\n", len(gpus))
	fmt.Fprintln(table, "GPU\tARCH\tUUID\tPID\tSTART\tTIME\tATTENTION")
	for _, gpu := range gpus {
		start, took := "-", "-"
		if gpu.StartUS > 0 {
			start, took = offset(gpu.StartUS/1000), span(float64(gpu.EndUS-gpu.StartUS)/1000)
		}
		fmt.Fprintf(table, "%s\t%s\t%s\t%d\t%s\t%s\t%s\n", gpu.number(), dash(gpu.Arch),
			output.Elide(dash(gpu.UUID), 17, full), gpu.PID, start, took, gpuAttention(gpu))
	}
	table.Flush()
	emitKernels(w, table, gpus, full)
}

// number is the GPU's number as nvidia-smi shows it, "-" when unknown.
func (g reportGPU) number() string {
	if g.GPU < 0 {
		return "-"
	}
	return strconv.Itoa(g.GPU)
}

// emitKernels prints each GPU's attention kernels: the one that served, and for every other
// kernel of its chains why it did not (still compiling, failed, absent, unsupported), with
// its compile time on this machine.
func emitKernels(w io.Writer, table *tabwriter.Writer, gpus []reportGPU, full bool) {
	header := false
	for _, gpu := range gpus {
		for _, kernel := range gpu.Attention.Kernels {
			if !header {
				fmt.Fprintln(w, "\nattention kernels")
				fmt.Fprintln(table, "GPU\tKERNEL\tSTATE\tCOMPILE\tDETAIL")
				header = true
			}
			state := kernel.State
			if kernel.Served {
				state = "served"
			} else if kernel.Progress != nil {
				state = fmt.Sprintf("%s (%.0f%%)", state, *kernel.Progress*100)
			}
			compile := "-"
			if kernel.CompileMS > 0 {
				compile = span(kernel.CompileMS)
			}
			fmt.Fprintf(table, "%s\t%s\t%s\t%s\t%s\n", gpu.number(), kernel.Kernel, state, compile,
				dash(output.Elide(kernel.Detail, 72, full)))
		}
	}
	table.Flush()
}

// emitJSON writes JSON whatever the machine format: the bundle's lists of nested objects
// are outside what the TOON encoder round-trips.
func emitJSON(w io.Writer, value any) error {
	encoder := json.NewEncoder(w)
	encoder.SetEscapeHTML(false)
	return encoder.Encode(value)
}

// stepsCell is a call's step tracks at a glance: count and mean step time.
func stepsCell(steps []reportSteps) string {
	parts := make([]string, 0, len(steps))
	for _, track := range steps {
		parts = append(parts, fmt.Sprintf("%s %d× %s", track.Name, track.Count, span(track.TotalMS/float64(max(track.Count, 1)))))
	}
	return strings.Join(parts, ", ")
}

// served names the attention kernels that served a call's GPUs, else what its GPUs observed.
func served(gpus []reportGPU) string {
	var names []string
	for _, gpu := range gpus {
		for _, kernel := range gpu.Attention.Kernels {
			if kernel.Served && !slices.Contains(names, kernel.Kernel) {
				names = append(names, kernel.Kernel)
			}
		}
	}
	if len(names) > 0 {
		return strings.Join(names, ",")
	}
	for _, gpu := range gpus {
		if seen := gpu.Attention.Observed; seen != "" && !slices.Contains(names, seen) {
			names = append(names, seen)
		}
	}
	return strings.Join(names, ",")
}

// gpuList spells GPU numbers in order, a contiguous run as its bounds: "0", "0-3", "0 2".
func gpuList(gpus []reportGPU) string {
	numbers := make([]int, len(gpus))
	for i, gpu := range gpus {
		numbers[i] = gpu.GPU
	}
	slices.Sort(numbers)
	if n := len(numbers); n > 2 && numbers[n-1]-numbers[0] == n-1 && numbers[0] >= 0 {
		return fmt.Sprintf("%d-%d", numbers[0], numbers[n-1])
	}
	parts := make([]string, len(gpus))
	for i, number := range numbers {
		parts[i] = reportGPU{GPU: number}.number()
	}
	return strings.Join(parts, " ")
}

// gpuNames is a grant's cards as a person reads them: "GPU 2", "GPUs 0-3".
func gpuNames(gpus []reportGPU) string {
	if len(gpus) == 1 {
		return "GPU " + gpus[0].number()
	}
	return "GPUs " + gpuList(gpus)
}

func clip(value string, limit int) string {
	if runes := []rune(value); len(runes) > limit {
		return string(runes[:limit-1]) + "…"
	}
	return value
}

func dash(value string) string {
	if value == "" {
		return "-"
	}
	return value
}

func gpuAttention(gpu reportGPU) string {
	seen := gpu.Attention.Observed
	if seen == "" {
		seen = "unobserved"
	}
	if gpu.Attention.Impl != "" {
		seen += " (" + gpu.Attention.Impl + ")"
	}
	if gpu.Attention.Requested != "" {
		return gpu.Attention.Requested + " → " + seen
	}
	return seen
}

// executorEvent is the SDK an attempt's executor loaded, as the Runtime recorded it.
type executorEvent struct {
	Request  string `json:"request"`
	Runtime  string `json:"runtime_version"`
	TensorFS string `json:"tensorfs_version"`
}

func (e executorEvent) String() string {
	if e.TensorFS == "" {
		return "cozy-runtime " + e.Runtime
	}
	return "cozy-runtime " + e.Runtime + " · tensorfs " + e.TensorFS
}
