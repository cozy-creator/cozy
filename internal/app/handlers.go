package app

import (
	"fmt"
	"runtime"
	"runtime/debug"
	"sort"
	"strings"

	"github.com/cozy-creator/cozy-creator/internal/api"
	"github.com/cozy-creator/cozy-creator/internal/exit"
	"github.com/cozy-creator/cozy-creator/internal/manifest"
	"github.com/cozy-creator/cozy-creator/internal/render"
	"github.com/cozy-creator/cozy-creator/internal/service"
	pb "github.com/cozy-creator/cozy-creator/protocol/cozy/worker/v1"
)

// Build identity. tag/commit may be stamped with -ldflags -X; otherwise the Go
// VCS stamp answers.
var (
	tag    = "0.0.0-dev"
	commit = ""
	// The package path is the major; the additive minor comes from the vendored binding.
	protocolVersion = fmt.Sprintf("cozy.worker.v1 (wire_minor %d)", pb.WireMinor)
	// The local client API contract this binary serves (cl-006). It is the SHARED
	// contract's core version, not a local build number: the same string is what
	// Tensorhub's host and the pod profile answer with when they serve the same core.
	contractVersion = api.ContractVersion
)

// handlers is the registry the manifest's Handler keys bind to. Adding one here
// without advertising it in the manifest is a startup refusal, and vice versa.
var handlers = map[string]Handler{
	"status":            handleStatus,
	"version":           handleVersion,
	"capabilities":      handleCapabilities,
	"commands":          handleCommands,
	"help":              handleHelp,
	"install":           handleInstall,
	"pack":              handlePack,
	"ls":                handleLs,
	"rm":                handleRm,
	"gc":                handleGC,
	"media.ls":          handleMediaLs,
	"up":                handleUp,
	"down":              handleDown,
	"endpoint.search":   handleEndpointSearch,
	"model.search":      handleModelSearch,
	"endpoint.show":     handleEndpointShow,
	"endpoint.create":   handleEndpointCreate,
	"model.show":        handleModelShow,
	"model.create":      handleModelCreate,
	"hub.status":        handleHubStatus,
	"hub.config":        handleHubConfig,
	"model.publish":     handleModelPublish,
	"model.download":    handleModelDownload,
	"start":             handleStart,
	"stop":              handleStop,
	"logs":              handleLogs,
	"run":               handleRun,
	"describe":          handleDescribe,
	"doctor":            handleDoctor,
	"fit":               handleFit,
	"job.submit":        handleJobSubmit,
	"job.status":        handleJobStatus,
	"job.ls":            handleJobLs,
	"job.follow":        handleJobFollow,
	"job.cancel":        handleJobCancel,
	"workflow.submit":   handleWorkflowSubmit,
	"workflow.status":   handleWorkflowStatus,
	"workflow.follow":   handleWorkflowFollow,
	"workflow.download": handleWorkflowDownload,
	"workflow.cancel":   handleWorkflowCancel,
	"video.compose":     handleVideoCompose,
	"video.submit":      handleVideoSubmit,
	"rent":              handleRent,
	"rent.ls":           handleRentLs,
	"rent.show":         handleRentShow,
	"rent.probe":        handleRentProbe,
	"rent.revise":       handleRentRevise,
	"rent.release":      handleRentRelease,
}

func handlerNames() []string {
	out := make([]string, 0, len(handlers))
	for k := range handlers {
		out = append(out, k)
	}
	sort.Strings(out)
	return out
}

// emit is the ONE place a handler's success document reaches the stream, so it is where
// AXI 9's disclosure is attached: the resolved row's manifest `Next` fills a document the
// handler left empty, and a handler that computed a state-dependent one keeps it.
func emit(ctx *Context, d render.Document) *exit.Error {
	if c := ctx.Inv.Cmd; c != nil && len(c.Next) > 0 {
		d = d.WithDefaultNext(c.Next)
	}
	if err := d.Emit(ctx.Out, ctx.Mode()); err != nil {
		return exit.As(err)
	}
	return nil
}

func handleStatus(ctx *Context) *exit.Error {
	st := service.Probe(ctx.Cfg)
	rec := render.Record{
		Kind: "status",
		Fields: []render.Field{
			{K: "home", V: ctx.Cfg.Home},
			{K: "service", V: map[bool]string{true: "up", false: "down"}[st.Up]},
			{K: "address", V: st.Addr},
		},
	}
	if !st.Up {
		rec.Next = []string{"cozy up -d"}
		rec.Fields = requestedStatusFields(rec.Fields, ctx.Mode().Fields)
		return emit(ctx, rec)
	}

	rec.Fields = append(rec.Fields,
		render.Field{K: "pid", V: st.PID},
		render.Field{K: "since", V: st.Since},
	)
	ctx.Service = st
	c, problem := dial(ctx)
	if problem != nil {
		rec.Notes = []string{"running-state details are unavailable: " + problem.Message}
		rec.Next = []string{"cozy down", "cozy up -d"}
		rec.Fields = requestedStatusFields(rec.Fields, ctx.Mode().Fields)
		return emit(ctx, rec)
	}
	doc, problem := c.Doctor()
	if problem != nil {
		rec.Notes = []string{"running-state details are unavailable: " + problem.Message}
		rec.Next = []string{"cozy doctor"}
		rec.Fields = requestedStatusFields(rec.Fields, ctx.Mode().Fields)
		return emit(ctx, rec)
	}
	counts := doc.Counts
	rec.Fields = append(rec.Fields,
		render.Field{K: "endpoints", V: counts["endpoints"]},
		render.Field{K: "workers", V: counts["workers"]},
		render.Field{K: "requests", V: counts["requests"]},
		render.Field{K: "active_requests", V: counts["active_requests"]},
		render.Field{K: "jobs", V: counts["jobs"]},
		render.Field{K: "active_jobs", V: counts["active_jobs"]},
		render.Field{K: "workflows", V: counts["workflows"]},
		render.Field{K: "active_workflows", V: counts["active_workflows"]},
	)
	switch {
	case counts["endpoints"] == 0:
		rec.Next = []string{"cozy install <org/endpoint>", "cozy endpoint search", "cozy model search"}
	case counts["resident_endpoints"] == 0:
		rec.Next = []string{"cozy start <org/endpoint>"}
	}
	return emit(ctx, rec)
}

// requestedStatusFields keeps --fields stable across service states without printing a
// wall of `unknown` values by default. An explicitly requested running-only fact is null
// while the service is down; a misspelling still reaches render's ordinary usage refusal.
func requestedStatusFields(fields []render.Field, requested []string) []render.Field {
	valid := map[string]bool{
		"home": true, "service": true, "address": true, "pid": true, "since": true,
		"endpoints": true, "workers": true, "requests": true, "active_requests": true,
		"jobs": true, "active_jobs": true, "workflows": true, "active_workflows": true,
	}
	have := map[string]bool{}
	for _, field := range fields {
		have[field.K] = true
	}
	for _, name := range requested {
		if valid[name] && !have[name] {
			fields = append(fields, render.Field{K: name, V: nil})
			have[name] = true
		}
	}
	return fields
}

func buildStamp() (string, bool) {
	rev, dirty := commit, false
	if info, ok := debug.ReadBuildInfo(); ok {
		for _, s := range info.Settings {
			switch s.Key {
			case "vcs.revision":
				if rev == "" {
					rev = s.Value
				}
			case "vcs.modified":
				dirty = s.Value == "true"
			}
		}
	}
	if rev == "" {
		rev = "unknown"
	}
	if len(rev) > 12 {
		rev = rev[:12]
	}
	return rev, dirty
}

func handleVersion(ctx *Context) *exit.Error {
	rev, dirty := buildStamp()
	return emit(ctx, render.Record{
		Kind: "version",
		Fields: []render.Field{
			{K: "tag", V: tag},
			{K: "commit", V: rev},
			{K: "dirty", V: dirty},
			{K: "protocol", V: protocolVersion},
			{K: "contract", V: contractVersion},
			{K: "go", V: runtime.Version()},
			{K: "platform", V: runtime.GOOS + "/" + runtime.GOARCH},
		},
	})
}

func handleCapabilities(ctx *Context) *exit.Error {
	st := service.Probe(ctx.Cfg)
	l := render.Lines{
		Kind:  "capabilities",
		Key:   "capabilities",
		Items: manifest.Capabilities(),
		Empty: "0 capabilities",
		Extra: []render.Field{{K: "service_up", V: st.Up}},
	}
	if !st.Up {
		l.Notes = []string{"LocalService tokens are unreadable while it is down; these are the binary's"}
		l.Next = []string{"cozy up"}
	}
	return emit(ctx, l)
}

func exitsText(codes []exit.Code) string {
	parts := make([]string, 0, len(codes))
	for _, c := range codes {
		parts = append(parts, fmt.Sprintf("%d %s", int(c), c.Name()))
	}
	return strings.Join(parts, " · ")
}

func flagsText(c *manifest.Command) string {
	parts := make([]string, 0, len(c.Flags))
	for _, f := range c.Flags {
		parts = append(parts, f.Name)
	}
	if len(parts) == 0 {
		return "-"
	}
	return strings.Join(parts, " ")
}

func handleCommands(ctx *Context) *exit.Error {
	prefix := ""
	if len(ctx.Inv.Args) == 1 {
		prefix = ctx.Inv.Args[0]
	}
	l := render.List{
		Kind:      "commands",
		Fields:    []string{"name", "status", "summary"},
		AllFields: []string{"name", "status", "group", "summary", "capability", "server", "destructive", "exits", "flags", "issue"},
		Empty:     "0 commands",
	}
	implemented, planned := 0, 0
	groups := map[string]bool{}
	for i := range manifest.Commands {
		c := &manifest.Commands[i]
		if prefix != "" && !strings.HasPrefix(c.Name(), prefix) {
			continue
		}
		status := string(c.Status)
		if c.Status == manifest.Planned {
			status = "planned"
			planned++
		} else {
			implemented++
		}
		groups[c.Group] = true
		l.Rows = append(l.Rows, map[string]string{
			"name":        c.Name(),
			"status":      status,
			"group":       c.Group,
			"summary":     c.Summary,
			"capability":  c.Capability,
			"server":      fmt.Sprintf("%t", c.NeedsServer),
			"destructive": fmt.Sprintf("%t", c.Destructive),
			"exits":       exitsText(c.Exits),
			"flags":       flagsText(c),
			"issue":       c.Issue,
		})
	}
	if len(l.Rows) == 0 {
		l.Empty = fmt.Sprintf("0 commands matching %q", prefix)
		l.Next = []string{"cozy commands"}
		return emit(ctx, l)
	}
	l.Aggregates = []render.Field{
		{K: "commands", V: len(l.Rows)},
		{K: "implemented", V: implemented},
		{K: "planned", V: planned},
		{K: "groups", V: len(groups)},
	}
	return emit(ctx, l)
}

func handleHelp(ctx *Context) *exit.Error {
	if len(ctx.Inv.Args) == 0 {
		return renderHelp(ctx, nil)
	}
	c, n := manifest.Lookup(ctx.Inv.Args)
	if c == nil || n != len(ctx.Inv.Args) {
		e := exit.New(exit.NotFound, "unknown command %q", strings.Join(ctx.Inv.Args, " "))
		if s := manifest.Suggest(ctx.Inv.Args[0]); len(s) > 0 {
			e.WithRemedy("did you mean: %s", strings.Join(s, ", "))
		}
		return e.WithNext("cozy commands")
	}
	return renderHelp(ctx, c)
}
