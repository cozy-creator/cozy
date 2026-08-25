package app

import (
	"strings"

	"github.com/cozy-creator/cozy-creator-v2/internal/exit"
	"github.com/cozy-creator/cozy-creator-v2/internal/manifest"
	"github.com/cozy-creator/cozy-creator-v2/internal/render"
)

// Invocation is one parsed command line. Flags are validated against the manifest,
// so an unknown flag is a typed usage refusal naming what the command accepts.
type Invocation struct {
	Cmd    *manifest.Command
	Args   []string
	Bools  map[string]bool
	Values map[string][]string
	Mode   render.Mode
	Help   bool
	Bare   bool
}

func (i *Invocation) Bool(name string) bool { return i.Bools[name] }

// Value is the last value given for a value-taking flag, or "".
func (i *Invocation) Value(name string) string {
	v := i.Values[name]
	if len(v) == 0 {
		return ""
	}
	return v[len(v)-1]
}

// prescanJSON finds --json before parsing so even a parse refusal renders in the
// mode the caller asked for.
func prescanJSON(args []string) bool {
	for _, a := range args {
		if a == "--json" {
			return true
		}
		if a == "--" {
			break
		}
	}
	return false
}

func findFlag(name string, cmd *manifest.Command) (manifest.Flag, bool) {
	sets := [][]manifest.Flag{manifest.GlobalFlags}
	if cmd != nil {
		sets = append(sets, cmd.Flags)
	}
	for _, set := range sets {
		for _, f := range set {
			if f.Name == name || (f.Short != "" && f.Short == name) {
				return f, true
			}
		}
	}
	return manifest.Flag{}, false
}

func acceptedFlags(cmd *manifest.Command) string {
	var names []string
	for _, f := range manifest.GlobalFlags {
		names = append(names, f.Name)
	}
	if cmd != nil {
		for _, f := range cmd.Flags {
			names = append(names, f.Name)
		}
	}
	return strings.Join(names, " ")
}

func parse(args []string) (*Invocation, *exit.Error) {
	inv := &Invocation{Bools: map[string]bool{}, Values: map[string][]string{}}
	inv.Mode.JSON = prescanJSON(args)

	var positional []string
	endOfFlags := false

	for i := 0; i < len(args); i++ {
		tok := args[i]
		switch {
		case !endOfFlags && tok == "--":
			endOfFlags = true
		case !endOfFlags && strings.HasPrefix(tok, "-") && tok != "-":
			name, inline, hasInline := strings.Cut(tok, "=")
			spec, ok := findFlag(name, inv.Cmd)
			if !ok {
				return inv, exit.Usagef("unknown flag %q", name).
					WithRemedy("accepted flags: %s", acceptedFlags(inv.Cmd)).
					WithNext(helpNext(inv.Cmd))
			}
			if !spec.TakesValue() {
				if hasInline {
					return inv, exit.Usagef("flag %s takes no value", spec.Name).
						WithNext(helpNext(inv.Cmd))
				}
				inv.Bools[spec.Name] = true
				continue
			}
			val := inline
			if !hasInline {
				if i+1 >= len(args) {
					return inv, exit.Usagef("flag %s needs a value %s", spec.Name, spec.Arg).
						WithNext(helpNext(inv.Cmd))
				}
				i++
				val = args[i]
			}
			inv.Values[spec.Name] = append(inv.Values[spec.Name], val)
		default:
			positional = append(positional, tok)
			if inv.Cmd == nil {
				if c, n := manifest.Lookup(positional); c != nil {
					inv.Cmd = c
					positional = positional[n:]
				} else if len(positional) >= 2 {
					return inv, unknownCommand(positional)
				}
			}
		}
	}

	if inv.Cmd == nil && len(positional) > 0 {
		return inv, unknownCommand(positional)
	}

	inv.Args = positional
	inv.Help = inv.Bools["--help"]
	inv.Mode.Full = inv.Bools["--full"]
	if f := inv.Values["--fields"]; len(f) > 0 {
		for _, part := range strings.Split(strings.Join(f, ","), ",") {
			if p := strings.TrimSpace(part); p != "" {
				inv.Mode.Fields = append(inv.Mode.Fields, p)
			}
		}
	}
	if inv.Cmd == nil {
		inv.Bare = !inv.Help
		return inv, nil
	}
	if err := checkArgs(inv); err != nil {
		return inv, err
	}
	return inv, nil
}

func checkArgs(inv *Invocation) *exit.Error {
	c := inv.Cmd
	if len(inv.Args) < c.MinArgs {
		return exit.Usagef("`cozy %s` needs %s", c.Name(), c.Args).
			WithRemedy("usage: cozy %s", strings.TrimSpace(c.Name()+" "+c.Args)).
			WithNext(helpNext(c))
	}
	if c.MaxArgs >= 0 && len(inv.Args) > c.MaxArgs {
		return exit.Usagef("`cozy %s` takes at most %d argument(s), got %d", c.Name(), c.MaxArgs, len(inv.Args)).
			WithRemedy("usage: cozy %s", strings.TrimSpace(c.Name()+" "+c.Args)).
			WithNext(helpNext(c))
	}
	return nil
}

func unknownCommand(words []string) *exit.Error {
	e := exit.Usagef("unknown command %q", strings.Join(words, " "))
	if s := manifest.Suggest(words[0]); len(s) > 0 {
		e.WithRemedy("did you mean: %s", strings.Join(s, ", "))
	} else {
		e.WithRemedy("`cozy commands` lists the whole surface")
	}
	return e.WithNext("cozy commands", "cozy -h")
}

func helpNext(c *manifest.Command) string {
	if c == nil {
		return "cozy -h"
	}
	return "cozy help " + c.Name()
}
