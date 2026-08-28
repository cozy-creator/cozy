// Package app wires the manifest to dispatch: startup self-check, parse, the
// shared gates, then one handler. AXI conventions: bare `cozy` is live status,
// -h/--help is concise and never mutates, and no code path reads a prompt.
package app

import (
	"fmt"
	"io"
	"strings"

	"github.com/cozy-creator/cozy-creator/internal/config"
	"github.com/cozy-creator/cozy-creator/internal/exit"
	"github.com/cozy-creator/cozy-creator/internal/manifest"
	"github.com/cozy-creator/cozy-creator/internal/render"
	"github.com/cozy-creator/cozy-creator/internal/service"
)

// Context is what a handler gets. Handlers never parse argv and never probe the
// service themselves — the gates did both.
type Context struct {
	Inv *Invocation
	Out io.Writer
	Err io.Writer
	// Cfg is the frozen typed value internal/config read ONCE at startup. No handler
	// reads the environment; the env-read fence proves it.
	Cfg     config.Config
	Service service.State
}

func (c *Context) Mode() render.Mode { return c.Inv.Mode }

type Handler func(*Context) *exit.Error

// Run is the whole CLI. It returns a shared-matrix exit code and never panics out.
func Run(args []string, stdout, stderr io.Writer) int {
	// AXI 10 — the version probe is the first statement of the process: agents run it at
	// every session start, so it pays for no self-check, no manifest parse and no
	// config.Load (which reads $COZY_HOME/config.yaml and .env off disk). It fires ONLY as
	// the sole argument, which is what keeps `cozy pack --version <x.y.z>` — a real
	// value-taking command flag — untouched. This is deliberately NOT a manifest.GlobalFlags
	// row: parse.findFlag checks globals before command flags, so a global --version would
	// silently swallow pack's. The manifest fence forbids anyone from adding one.
	if len(args) == 1 && (args[0] == "-v" || args[0] == "-V" || args[0] == "--version") {
		fmt.Fprintln(stdout, tag)
		return int(exit.OK)
	}

	// Startup self-check: the manifest and the handler registry must agree.
	if e := manifest.SelfCheck(handlerNames()); e != nil {
		render.EmitError(stdout, e, render.Mode{JSON: prescanJSON(args)})
		return int(exit.Internal)
	}

	inv, e := parse(args)
	if e != nil {
		render.EmitError(stdout, e, inv.Mode)
		return int(exit.Of(e))
	}

	// The ONE environment read of this process, before any handler runs.
	cfg, e := config.Load()
	if e != nil {
		render.EmitError(stdout, e, inv.Mode)
		return int(exit.Of(e))
	}

	ctx := &Context{Inv: inv, Out: stdout, Err: stderr, Cfg: cfg}

	// -h/--help short-circuits before every gate: help never mutates, never dials.
	if inv.Help {
		if e := renderHelp(ctx, inv.Cmd); e != nil {
			render.EmitError(stdout, e, inv.Mode)
			return int(exit.Of(e))
		}
		return int(exit.OK)
	}
	// Bare `cozy` is live status, never help.
	if inv.Bare {
		inv.Cmd, _ = manifest.Lookup([]string{"status"})
	}

	// Gates, in precedence order — see gate(). Arity is one of them, deliberately:
	// checked here rather than in parse so that -h and the planned-row answer come first.
	if e := gate(ctx); e != nil {
		render.EmitError(stdout, e, inv.Mode)
		return int(exit.Of(e))
	}

	h := handlers[inv.Cmd.Handler]
	if e := h(ctx); e != nil {
		render.EmitError(stdout, e, inv.Mode)
		return int(exit.Of(e))
	}
	return int(exit.OK)
}

// gate is the whole refusal precedence, in one place and in one order. -h/--help never
// reaches here — Run answers it first, because help outranks every refusal (AXI 10) and a
// gate that fired earlier would make `cozy job submit -h` a usage error, which it was.
// Below help: authorization, then what this build carries, then the shape of the line,
// then the world. The last one dials another process, so it is asked last.
func gate(ctx *Context) *exit.Error {
	c := ctx.Inv.Cmd

	// 7 — destructive without --yes. No prompt exists anywhere. First even for a line that
	// is also malformed: this refusal's next: already spells the corrected line, arguments
	// included, so following it fixes both facts at once — where an arity error would send
	// the caller back for a second refusal it never mentioned.
	if c.Destructive && !ctx.Inv.Bool("--yes") {
		e := exit.Confirmf("`cozy %s` is destructive and refuses without --yes", c.Name()).
			WithRemedy("re-run with --yes; there is no prompt")
		if c.Status == manifest.Planned {
			e.Message += " (not implemented in this build either — " + c.Issue + ")"
		}
		return e.WithNext("cozy " + c.Name() + " " + strings.TrimSpace(c.Args) + " --yes")
	}

	// 2 — advertised in the manifest, not carried by this build. Ahead of arity: a planned
	// row has no handler, so its MinArgs/MaxArgs describe a contract nothing here enforces,
	// and "you are missing an argument" is a worse answer than the issue it lands with.
	if c.Status != manifest.Implemented {
		return exit.Named(exit.Usage, "not_implemented",
			"`cozy %s` is not implemented in this build", c.Name()).
			WithRemedy("it lands with issue %s", c.Issue).
			WithNext("cozy commands", "cozy help "+c.Name())
	}

	// 2 — arity: a fact about the line itself, settled without reading disk or dialing.
	if e := checkArgs(ctx.Inv); e != nil {
		return e
	}

	// 9 — the LocalService is the only door to server-backed verbs. Last: it is the only
	// gate that touches another process, and "not running on this host right now" is the
	// least durable of the four — a malformed or unbuilt verb is refused without it.
	if c.NeedsServer {
		ctx.Service = service.Probe(ctx.Cfg)
		if !ctx.Service.Up {
			return ctx.Service.Unavailable()
		}
	}
	return nil
}
