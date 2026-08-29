// Package cli is Cozy's single Kong command grammar and application boundary.
package cli

import (
	"errors"
	"fmt"
	"io"
	"strconv"
	"strings"

	"github.com/alecthomas/kong"
	"github.com/cozy-creator/cozy-creator/internal/config"
	"github.com/cozy-creator/cozy-creator/internal/exit"
	"github.com/cozy-creator/cozy-creator/internal/output"
	"github.com/cozy-creator/cozy-creator/internal/service"
)

const description = "Local-first generative media: install endpoints, run them, and publish releases."

// Invocation is the small adapter retained mechanism handlers consume. Kong is
// the only parser; this value merely carries already-typed command values.
type Invocation struct {
	Args   []string
	Bools  map[string]bool
	Values map[string][]string
	Mode   output.Mode
}

func (i *Invocation) Bool(name string) bool { return i.Bools[name] }

func (i *Invocation) Value(name string) string {
	values := i.Values[name]
	if len(values) == 0 {
		return ""
	}
	return values[len(values)-1]
}

// Context is the retained mechanism boundary. Handlers receive typed values and
// frozen configuration; they never parse argv or read ambient configuration.
type Context struct {
	Inv     *Invocation
	Out     io.Writer
	Err     io.Writer
	Cfg     config.Config
	Service service.State
	Next    []string
}

func (c *Context) Mode() output.Mode { return c.Inv.Mode }

type handler func(*Context) *exit.Error

// Runtime is injected into the selected Kong command's Run method.
type Runtime struct {
	Cfg  config.Config
	Out  io.Writer
	Err  io.Writer
	Mode output.Mode
}

func (r *Runtime) call(h handler, args []string, flags map[string]bool,
	values map[string][]string, next []string, controller bool,
) error {
	ctx := &Context{
		Inv: &Invocation{
			Args: append([]string(nil), args...), Bools: flags,
			Values: values, Mode: r.Mode,
		},
		Out: r.Out, Err: r.Err, Cfg: r.Cfg, Next: next,
	}
	if controller {
		state, _, problem := ensureController(ctx)
		if problem != nil {
			return problem
		}
		ctx.Service = state
	}
	return h(ctx)
}

// Run parses and executes one public CLI invocation.
func Run(args []string, stdout, stderr io.Writer) int {
	if len(args) == 1 && (args[0] == "-v" || args[0] == "-V" || args[0] == "--version") {
		fmt.Fprintln(stdout, version())
		return 0
	}

	args = helpArgs(args)
	wantsJSON := jsonRequested(args)
	var grammar CLI
	helpExit := -1
	parser, err := kong.New(&grammar,
		kong.Name("cozy"),
		kong.Description(description),
		kong.Writers(stdout, stderr),
		kong.Exit(func(code int) { helpExit = code }),
		kong.ExplicitGroups([]kong.Group{
			{Key: "Resources", Title: "Resources"},
			{Key: "Work", Title: "Work"},
			{Key: "Lifecycle", Title: "Lifecycle"},
		}),
		kong.ConfigureHelp(kong.HelpOptions{Compact: true, FlagsLast: true, WrapUpperBound: 100}),
	)
	mode := output.Mode{JSON: wantsJSON}
	if err != nil {
		problem := output.NewError(output.Operational, "cli.grammar", err.Error())
		_ = output.EmitError(stdout, problem, mode)
		return output.ShellCode(problem)
	}

	parsed, err := parser.Parse(args)
	mode = output.Mode{JSON: wantsJSON || grammar.JSON, Full: grammar.Full, Fields: grammar.Fields}
	if helpExit >= 0 {
		return helpExit
	}
	if err != nil {
		problem := output.NewError(output.Usage, "cli.usage", err.Error()).
			WithNext(helpFor(parsed))
		_ = output.EmitError(stdout, problem, mode)
		return output.ShellCode(problem)
	}

	cfg, problem := config.Load()
	if problem != nil {
		out := output.NewError(output.Config, problem.ErrName(), problem.Message)
		if problem.Remedy != "" {
			out.WithRemedy(problem.Remedy)
		}
		out.WithNext(problem.Next...)
		_ = output.EmitError(stdout, out, mode)
		return output.ShellCode(out)
	}

	runtime := &Runtime{Cfg: cfg, Out: stdout, Err: stderr, Mode: mode}
	if err := parsed.Run(runtime); err != nil {
		problem := projectError(err)
		_ = output.EmitError(stdout, problem, mode)
		return output.ShellCode(problem)
	}
	return 0
}

func jsonRequested(args []string) bool {
	for _, arg := range args {
		if arg == "--" {
			return false
		}
		if arg == "--json" {
			return true
		}
	}
	return false
}

func helpArgs(args []string) []string {
	if len(args) == 0 {
		return []string{"--help"}
	}
	if args[0] != "help" {
		return args
	}
	if len(args) == 1 {
		return []string{"--help"}
	}
	out := append([]string(nil), args[1:]...)
	return append(out, "--help")
}

func helpFor(ctx *kong.Context) string {
	if ctx == nil || strings.TrimSpace(ctx.Command()) == "" {
		return "cozy help"
	}
	return "cozy help " + strings.ReplaceAll(ctx.Command(), " <", "")
}

func projectError(err error) *output.Error {
	var ready *output.Error
	if errors.As(err, &ready) {
		return ready
	}
	var problem *exit.Error
	if !errors.As(err, &problem) {
		return output.NewError(output.Operational, "internal", err.Error())
	}
	class := output.Operational
	if problem.Code == exit.Usage {
		class = output.Usage
	}
	out := output.NewError(class, problem.ErrName(), problem.Message)
	if problem.Remedy != "" {
		out.WithRemedy(problem.Remedy)
	}
	return out.WithNext(problem.Next...)
}

func bools(pairs ...any) map[string]bool {
	out := map[string]bool{}
	for i := 0; i+1 < len(pairs); i += 2 {
		name, nameOK := pairs[i].(string)
		value, valueOK := pairs[i+1].(bool)
		if nameOK && valueOK && value {
			out[name] = true
		}
	}
	return out
}

func values(pairs ...any) map[string][]string {
	out := map[string][]string{}
	for i := 0; i+1 < len(pairs); i += 2 {
		name, ok := pairs[i].(string)
		if !ok {
			continue
		}
		switch value := pairs[i+1].(type) {
		case string:
			if value != "" {
				out[name] = []string{value}
			}
		case []string:
			if len(value) > 0 {
				out[name] = append([]string(nil), value...)
			}
		}
	}
	return out
}

func intText(value int) string { return strconv.Itoa(value) }
