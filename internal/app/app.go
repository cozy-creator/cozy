// Package app wires the manifest to dispatch: startup self-check, parse, the
// shared gates, then one handler. AXI conventions: bare `cozy` is live status,
// -h/--help is concise and never mutates, and no code path reads a prompt.
package app

import (
	"io"
	"strings"

	"github.com/cozy-creator/cozy-creator-v2/internal/config"
	"github.com/cozy-creator/cozy-creator-v2/internal/exit"
	"github.com/cozy-creator/cozy-creator-v2/internal/manifest"
	"github.com/cozy-creator/cozy-creator-v2/internal/render"
	"github.com/cozy-creator/cozy-creator-v2/internal/service"
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
	// Startup self-check: the manifest and the handler registry must agree.
	if e := manifest.SelfCheck(handlerNames()); e != nil {
		render.EmitError(stdout, stderr, e, render.Mode{JSON: prescanJSON(args)})
		return int(exit.Internal)
	}

	inv, e := parse(args)
	if e != nil {
		render.EmitError(stdout, stderr, e, inv.Mode)
		return int(exit.Of(e))
	}

	// The ONE environment read of this process, before any handler runs.
	cfg, e := config.Load()
	if e != nil {
		render.EmitError(stdout, stderr, e, inv.Mode)
		return int(exit.Of(e))
	}

	ctx := &Context{Inv: inv, Out: stdout, Err: stderr, Cfg: cfg}

	// -h/--help short-circuits before every gate: help never mutates, never dials.
	if inv.Help {
		if e := renderHelp(ctx, inv.Cmd); e != nil {
			render.EmitError(stdout, stderr, e, inv.Mode)
			return int(exit.Of(e))
		}
		return int(exit.OK)
	}
	// Bare `cozy` is live status, never help.
	if inv.Bare {
		inv.Cmd, _ = manifest.Lookup([]string{"status"})
	}

	// Gates, most-durable refusal first: a refusal that would still stand in a
	// fully implemented binary is reported before one that is only true of this build.
	if e := gate(ctx); e != nil {
		render.EmitError(stdout, stderr, e, inv.Mode)
		return int(exit.Of(e))
	}

	h := handlers[inv.Cmd.Handler]
	if e := h(ctx); e != nil {
		render.EmitError(stdout, stderr, e, inv.Mode)
		return int(exit.Of(e))
	}
	return int(exit.OK)
}

func gate(ctx *Context) *exit.Error {
	c := ctx.Inv.Cmd

	// 7 — destructive without --yes. No prompt exists anywhere.
	if c.Destructive && !ctx.Inv.Bool("--yes") {
		e := exit.Confirmf("`cozy %s` is destructive and refuses without --yes", c.Name()).
			WithRemedy("re-run with --yes; there is no prompt")
		if c.Status == manifest.Planned {
			e.Message += " (not implemented in this build either — " + c.Issue + ")"
		}
		return e.WithNext("cozy " + c.Name() + " " + strings.TrimSpace(c.Args) + " --yes")
	}

	// 9 — the LocalService is the only door to server-backed verbs.
	if c.NeedsServer {
		ctx.Service = service.Probe(ctx.Cfg)
		if !ctx.Service.Up {
			return ctx.Service.Unavailable()
		}
	}

	// 2 — advertised in the manifest, not carried by this build.
	if c.Status != manifest.Implemented {
		return exit.Named(exit.Usage, "not_implemented",
			"`cozy %s` is not implemented in this build", c.Name()).
			WithRemedy("it lands with issue %s", c.Issue).
			WithNext("cozy commands", "cozy help "+c.Name())
	}
	return nil
}
