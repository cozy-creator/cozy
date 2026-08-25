package app

import (
	"fmt"
	"strings"

	"github.com/cozy-creator/cozy-creator-v2/internal/exit"
	"github.com/cozy-creator/cozy-creator-v2/internal/manifest"
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
			fmt.Fprintf(w, "  %-*s  %s\n", width, flagSpelling(f), f.Summary)
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
	if c.NeedsServer {
		fmt.Fprintln(w, "requires: the LocalService (exit 9 while it is down)")
	}
	if c.Capability != "" {
		fmt.Fprintf(w, "capability: %s\n", c.Capability)
	}
	fmt.Fprintln(w, "next: cozy commands")
	return nil
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
	fmt.Fprintln(w, "cozy — local-first generative media: install endpoints, run them, publish releases")
	fmt.Fprintln(w, "usage: cozy [global flags] <command> [args…]   ·   bare `cozy` prints live status")
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
	width := 0
	for _, f := range manifest.GlobalFlags {
		if n := len(flagSpelling(f)); n > width {
			width = n
		}
	}
	for _, f := range manifest.GlobalFlags {
		fmt.Fprintf(w, "  %-*s  %s\n", width, flagSpelling(f), f.Summary)
	}
	fmt.Fprintln(w, "next: cozy commands --full")
	fmt.Fprintln(w, "next: cozy help <command>")
	return nil
}
