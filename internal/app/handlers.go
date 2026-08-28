package app

import (
	"fmt"
	"os"
	"runtime"
	"runtime/debug"
	"sort"
	"strings"

	"github.com/cozy-creator/cozy-creator/internal/api"
	"github.com/cozy-creator/cozy-creator/internal/exit"
	"github.com/cozy-creator/cozy-creator/internal/home"
	"github.com/cozy-creator/cozy-creator/internal/manifest"
	"github.com/cozy-creator/cozy-creator/internal/records"
	"github.com/cozy-creator/cozy-creator/internal/render"
	"github.com/cozy-creator/cozy-creator/internal/service"
)

// Build identity. tag/commit may be stamped with -ldflags -X; otherwise the Go
// VCS stamp answers.
var (
	tag    = "0.0.0-dev"
	commit = ""
	// The published worker protocol (worker-protocol-v2, th-024). Asserted here as a
	// constant; the binding link to a running worker arrives with cl-001.
	protocolVersion = "cozy.worker.v1 (wire_minor 0)"
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
	"up":                handleUp,
	"down":              handleDown,
	"search":            handleSearch,
	"repo.show":         handleRepoShow,
	"repo.create":       handleRepoCreate,
	"hub.status":        handleHubStatus,
	"hub.config":        handleHubConfig,
	"push":              handlePush,
	"pull":              handlePull,
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
			{K: "service", V: map[bool]string{true: "up", false: "down"}[st.Up]},
			{K: "address", V: st.Addr},
			{K: "endpoints", V: nil},
			{K: "workers", V: nil},
			{K: "jobs", V: nil},
			{K: "workflows", V: nil},
		},
	}
	if st.Up {
		// The live facts come from the ONE authority, read directly. A second reader of
		// the same rows is a read, never a second store.
		if l, e := home.Open(ctx.Cfg.Home); e == nil {
			if store, e := records.Open(l.DB); e == nil {
				defer store.Close()
				if counts, e := store.Counts(); e == nil {
					rec.Fields = []render.Field{
						{K: "service", V: "up"},
						{K: "address", V: st.Addr},
						{K: "socket", V: st.Socket},
						{K: "pid", V: st.PID},
						{K: "workers", V: counts["workers"]},
						{K: "requests", V: counts["requests"]},
						{K: "attempts", V: counts["attempts"]},
						{K: "live_attempts", V: counts["live"]},
						{K: "recovered_open", V: counts["recovered"]},
						{K: "outputs", V: counts["outputs"]},
						{K: "workflows", V: counts["workflows"]},
						{K: "live_workflows", V: counts["live_workflows"]},
					}
				}
			}
		}
		// The ENDPOINT and WORKER listings are the API's, because they are facts only the
		// running service holds (cl-010: `cozy status` is a client of /v1/local/*). A
		// credential that cannot be read is reported, never fatal — bare `cozy` is
		// content-first and answers with what it could learn.
		ctx.Service = st
		installed, resident := -1, 0
		if c, e := dial(ctx); e == nil {
			if rows, e := c.Endpoints(); e == nil {
				names := make([]string, 0, len(rows))
				for _, row := range rows {
					state := "cold"
					if row.Resident {
						state = "running"
						resident++
					}
					names = append(names, row.Endpoint+" ("+state+")")
				}
				installed = len(rows)
				rec.Fields = append(rec.Fields, render.Field{K: "endpoints", V: names})
			}
			if workers, e := c.Workers(); e == nil {
				live := make([]string, 0, len(workers))
				for _, w := range workers {
					live = append(live, w.Endpoint+" "+w.InstanceID+" "+w.Serving)
				}
				rec.Fields = append(rec.Fields, render.Field{K: "live_workers", V: live})
			}
		} else {
			rec.Notes = append(rec.Notes, "the local client credential is unreadable: "+e.Message)
		}
		rec.Notes = append(rec.Notes, st.Details)
		// The suggestion follows the STATE, not the verb: nothing installed and "run
		// something" is unactionable. Everything warm falls through to the row's own
		// manifest default. A listing that could not be read leaves it there too.
		switch {
		case installed == 0:
			rec.Next = []string{"cozy install <org/endpoint>", "cozy search"}
		case installed > 0 && resident == 0:
			rec.Next = []string{"cozy start <org/endpoint>"}
		}
	} else {
		rec.Notes = []string{st.Details,
			"installed endpoints, workers and jobs are readable only while the service runs"}
		rec.Next = []string{"cozy up"}
	}
	// AXI 10 — the HOME view names the tool before its live data. `cozy status` typed
	// explicitly is a poll of that data and stays exactly as it is.
	if ctx.Inv.Bare {
		rec.Fields = append(identity(), rec.Fields...)
	}
	return emit(ctx, rec)
}

// identity is AXI 10's identification block. The path is the RUNNING binary's own, not
// an env reading — several cozy builds coexist and the agent needs the one it just ran.
func identity() []render.Field {
	bin, err := os.Executable()
	if err != nil {
		bin = "cozy"
	}
	return []render.Field{
		{K: "bin", V: bin},
		{K: "description", V: manifest.Description},
		{K: "version", V: tag},
	}
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
