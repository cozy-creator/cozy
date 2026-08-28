package app

import (
	"fmt"
	"strings"

	"github.com/cozy-creator/cozy-creator/internal/exit"
	"github.com/cozy-creator/cozy-creator/internal/manifest"
	"github.com/cozy-creator/cozy-creator/internal/render"
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

// acceptedFlags names the RESOLVED command's own flags first and the globals second
// (AXI 6): "valid flags for `ls`" is the answer; a flat union of everything is a list the
// agent has to re-derive the scope of.
func acceptedFlags(cmd *manifest.Command) string {
	var global []string
	for _, f := range manifest.GlobalFlags {
		global = append(global, f.Name)
	}
	if cmd == nil {
		return strings.Join(global, " ") + " (global; the line names no command)"
	}
	var own []string
	for _, f := range cmd.Flags {
		own = append(own, f.Name)
	}
	if len(own) == 0 {
		return fmt.Sprintf("`cozy %s` takes no flags of its own · global: %s",
			cmd.Name(), strings.Join(global, " "))
	}
	return fmt.Sprintf("`cozy %s`: %s · global: %s",
		cmd.Name(), strings.Join(own, " "), strings.Join(global, " "))
}

// scope answers WHICH command a refusal is about. `cozy --stat ls` fails on token 1,
// before `ls` was ever read, so inv.Cmd is still nil and the refusal would name the
// global flags only — the wrong surface for the command the caller actually typed.
// This re-reads the whole line for the command word. It is BEST EFFORT and never feeds
// dispatch: an unknown flag's arity is unknowable, so it is assumed boolean, and a
// KNOWN global flag's value is skipped so `cozy --fields ls run` still resolves `run`.
func scope(inv *Invocation, args []string) *manifest.Command {
	if inv.Cmd != nil {
		return inv.Cmd
	}
	var words []string
	for i := 0; i < len(args); i++ {
		tok := args[i]
		if tok == "--" {
			break
		}
		if strings.HasPrefix(tok, "-") && tok != "-" {
			name, _, inline := strings.Cut(tok, "=")
			if f, ok := findFlag(name, nil); ok && f.TakesValue() && !inline {
				i++
			}
			continue
		}
		words = append(words, tok)
	}
	// Lookup already prefers the two-word row, so the held-verb rule needs no repeat here.
	for start := range words {
		if c, _ := manifest.Lookup(words[start:]); c != nil {
			return c
		}
	}
	return nil
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
				at := scope(inv, args)
				return inv, exit.Usagef("unknown flag %q for `cozy %s`", name, cmdName(at)).
					WithRemedy("accepted flags: %s", acceptedFlags(at)).
					WithNext(helpNext(at))
			}
			if !spec.TakesValue() {
				if hasInline {
					return inv, exit.Usagef("flag %s takes no value", spec.Name).
						WithNext(helpNext(scope(inv, args)))
				}
				inv.Bools[spec.Name] = true
				continue
			}
			val := inline
			if !hasInline {
				if i+1 >= len(args) {
					return inv, exit.Usagef("flag %s needs a value %s", spec.Name, spec.Arg).
						WithNext(helpNext(scope(inv, args)))
				}
				i++
				val = args[i]
			}
			inv.Values[spec.Name] = append(inv.Values[spec.Name], val)
		default:
			positional = append(positional, tok)
			if inv.Cmd != nil {
				continue
			}
			// LONGEST MATCH WINS. A one-word row whose word also starts a two-word row is
			// HELD for one more token: binding it the moment it matched made `rent ls`
			// unreachable and made `ls` an argument `rent` then refused.
			if len(positional) == 1 && manifest.Extendable(positional[0]) {
				continue
			}
			if c, n := manifest.Lookup(positional); c != nil {
				inv.Cmd = c
				positional = positional[n:]
			} else if len(positional) >= 2 {
				return inv, unknownCommand(positional)
			}
		}
	}

	// A held word with nothing after it is the one-word verb after all.
	if inv.Cmd == nil && len(positional) > 0 {
		c, n := manifest.Lookup(positional)
		if c == nil {
			return inv, unknownCommand(positional)
		}
		inv.Cmd, positional = c, positional[n:]
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
	}
	return inv, nil
}

// checkArgs is a GATE, not a parse step: it runs from app.gate, after -h has been
// answered and after a planned row has said which issue it lands with. Refusing here
// would put arity ahead of both.
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

func cmdName(c *manifest.Command) string {
	if c == nil {
		return "<command>"
	}
	return c.Name()
}
