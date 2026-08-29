package app

import (
	"fmt"
	"strings"

	"github.com/cozy-creator/cozy-creator/internal/exit"
	"github.com/cozy-creator/cozy-creator/internal/manifest"
)

// renderHelp writes concise help built from the manifest. It never mutates and
// never touches the network — it runs before every gate, including the probe.
func renderHelp(ctx *Context, c *manifest.Command) *exit.Error {
	if c == nil {
		return renderRootHelp(ctx)
	}
	w := ctx.Out
	fmt.Fprintf(w, "cozy %s — %s\n", c.Name(), c.Summary)
	fmt.Fprintf(w, "usage: %s\n", strings.TrimSpace("cozy "+c.Name()+" "+c.Args))
	if c.MinArgs > 0 {
		fmt.Fprintf(w, "required: %s (%d argument(s))\n", strings.TrimSpace(c.Args), c.MinArgs)
	}
	fmt.Fprintf(w, "group: %s\n", c.Group)
	if c.Status == manifest.Implemented {
		fmt.Fprintln(w, "status: implemented")
	} else {
		fmt.Fprintf(w, "status: planned (%s)\n", c.Issue)
	}
	if len(c.Flags) > 0 {
		fmt.Fprintln(w, "flags:")
		width := 0
		for _, f := range c.Flags {
			if n := len(flagSpelling(f)); n > width {
				width = n
			}
		}
		for _, f := range c.Flags {
			fmt.Fprintf(w, "  %-*s  %s\n", width, flagSpelling(f), flagSummary(f))
		}
	}
	if len(c.Exits) > 0 {
		fmt.Fprintf(w, "exits: %s\n", exitsText(c.Exits))
	}
	if c.Terminals {
		fmt.Fprintf(w, "terminal mapping: %s\n", exit.TerminalMapping())
	}
	if c.Destructive {
		fmt.Fprintln(w, "destructive: refuses without --yes (exit 7) — no prompt exists")
	}
	if c.PlanFirst {
		fmt.Fprintln(w, "plan-first: without --yes it prints the plan and exits 0; --yes executes it")
	}
	if c.NeedsServer {
		fmt.Fprintln(w, "requires: the LocalService (exit 9 while it is down)")
	}
	if c.Capability != "" {
		fmt.Fprintf(w, "capability: %s\n", c.Capability)
	}
	// AXI 10 — worked invocations, parameterized. An agent reading `--in <file>` still
	// has to guess the shape of a call; an example is the shape.
	if len(c.Examples) > 0 {
		fmt.Fprintln(w, "examples:")
		for _, e := range c.Examples {
			fmt.Fprintf(w, "  %s\n", e)
		}
	}
	fmt.Fprintln(w, "next: cozy commands")
	return nil
}

// flagSummary appends the value that applies when the flag is absent (AXI 10).
func flagSummary(f manifest.Flag) string {
	if f.Default == "" {
		return f.Summary
	}
	return fmt.Sprintf("%s (default %s)", f.Summary, f.Default)
}

func flagSpelling(f manifest.Flag) string {
	s := f.Name
	if f.Short != "" {
		s += "/" + f.Short
	}
	if f.Arg != "" {
		s += " " + f.Arg
	}
	return s
}

func renderRootHelp(ctx *Context) *exit.Error {
	w := ctx.Out
	fmt.Fprintln(w, "cozy — "+manifest.Description)
	fmt.Fprintln(w, "usage: cozy [global flags] <command> [args…]")
	fmt.Fprintln(w)
	planned := false
	for _, g := range manifest.Groups {
		var names []string
		for i := range manifest.Commands {
			c := &manifest.Commands[i]
			if c.Group != g {
				continue
			}
			n := c.Name()
			if c.Status == manifest.Planned {
				n += "*"
				planned = true
			}
			names = append(names, n)
		}
		if len(names) > 0 {
			fmt.Fprintf(w, "  %-11s %s\n", g, strings.Join(names, "  "))
		}
	}
	if planned {
		fmt.Fprintln(w, "\n  * planned — `cozy commands` names the owning issue")
	}
	fmt.Fprintln(w, "\nglobal flags:")
	// The version probe is a pre-parse fast path in Run, not a GlobalFlags row (a global
	// --version would shadow `cozy pack --version`), so root help is where it is advertised.
	const versionFlags = "-v/-V/--version"
	width := len(versionFlags)
	for _, f := range manifest.GlobalFlags {
		if n := len(flagSpelling(f)); n > width {
			width = n
		}
	}
	for _, f := range manifest.GlobalFlags {
		fmt.Fprintf(w, "  %-*s  %s\n", width, flagSpelling(f), flagSummary(f))
	}
	fmt.Fprintf(w, "  %-*s  %s\n", width, versionFlags,
		"print the bare version and exit 0; only as the sole argument (`cozy version` is the full record)")
	fmt.Fprintln(w, "\nexamples:")
	fmt.Fprintln(w, "  cozy status")
	fmt.Fprintln(w, "  cozy up -d")
	fmt.Fprintln(w, "  cozy endpoint search video")
	fmt.Fprintln(w, "  cozy install org/endpoint --from ./release.tar.gz --digest sha256:<hex>")
	fmt.Fprintln(w, "  cozy run org/endpoint/v1/generate \"a red bicycle\"")
	fmt.Fprintln(w, "next: cozy commands")
	fmt.Fprintln(w, "next: cozy help <command>")
	return nil
}
