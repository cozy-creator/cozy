// Package cli is Cozy's single Kong command grammar and application boundary.
package cli

import (
	"bytes"
	"errors"
	"fmt"
	"io"
	"os"
	"strconv"
	"strings"
	"time"

	"github.com/alecthomas/kong"
	"github.com/cozy-creator/cozy/internal/accountauth"
	"github.com/cozy-creator/cozy/internal/calcifer"
	localapi "github.com/cozy-creator/cozy/internal/client"
	"github.com/cozy-creator/cozy/internal/config"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/machineendpoint"
	"github.com/cozy-creator/cozy/internal/output"
	"github.com/cozy-creator/cozy/internal/packagepublish"
	"github.com/cozy-creator/cozy/internal/records"
	"github.com/mattn/go-isatty"
)

const description = "Local-first generative media: install packages, run them, and publish releases."

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
	endpoint         *machineendpoint.Endpoint
	foregroundClient *localapi.Client
	exitCode         int       // a completed aggregate can report partial failures without a second document
	teardown         bool      // `down --all`: hand each rental to its hub once; never wait on one
	commandStarted   time.Time // includes capture and resolution before a request exists
	Inv              *Invocation
	Out              io.Writer
	Err              io.Writer
	Cfg              config.Config
	Daemon           calcifer.State
	AccountAuth      *accountauth.Manager
	namespace        packagepublish.NamespaceSource // the caller on Cfg's Tensorhub, asked once
	ingestBytes      int64                          // planned source bytes a native ingest declares for its rental
	warnings         *[]records.Warning             // this command's, shared by its scoped copies
}

// warn says one warning now on stderr for a person and in the command's JSON document for a
// program; the run's own event stream never repeats it.
func (c *Context) warn(warning records.Warning) {
	if c.warnings == nil {
		c.warnings = new([]records.Warning)
	}
	*c.warnings = append(*c.warnings, warning)
	if !c.Mode().JSON {
		fmt.Fprintf(c.Err, "warning: %s\n", warning.Message)
	}
}

func (c *Context) said() []records.Warning {
	if c.warnings == nil {
		return nil
	}
	return *c.warnings
}

// warned is whether this command already said an event's warning: its code or, uncoded, its text.
func (c *Context) warned(event map[string]any) bool {
	code, _ := event["code"].(string)
	message, _ := event["message"].(string)
	for _, said := range c.said() {
		if code != "" && said.Code == code || code == "" && said.Message == strings.TrimSpace(message) {
			return true
		}
	}
	return false
}

func (c *Context) Mode() output.Mode { return c.Inv.Mode }

// forHub is this context addressing one Tensorhub origin with that origin's own
// credential. Work on a record uses the record's hub; "" keeps the current one.
func (c *Context) forHub(origin string) *Context {
	scoped := c.Cfg.ForHub(origin)
	if scoped.HubURL == c.Cfg.HubURL {
		return c
	}
	sub := *c
	sub.Cfg, sub.AccountAuth = scoped, accountauth.New(scoped)
	return &sub
}

type handler func(*Context) *exit.Error

// Runtime is injected into the selected Kong command's Run method.
type Runtime struct {
	exitCode int
	Cfg      config.Config
	Out      io.Writer
	Err      io.Writer
	Mode     output.Mode
}

func (r *Runtime) call(h handler, args []string, flags map[string]bool,
	values map[string][]string, daemon bool,
) error {
	ctx := &Context{
		commandStarted: time.Now(),
		Inv: &Invocation{
			Args: append([]string(nil), args...), Bools: flags,
			Values: values, Mode: r.Mode,
		},
		Out: r.Out, Err: r.Err, Cfg: r.Cfg, AccountAuth: accountauth.New(r.Cfg),
		warnings: new([]records.Warning),
	}
	if daemon {
		state, _, problem := ensureCalcifer(ctx)
		if problem != nil {
			return problem
		}
		ctx.Daemon = state
	}
	problem := h(ctx)
	r.exitCode = ctx.exitCode
	if said := ctx.said(); problem != nil && len(said) > 0 {
		if problem.Details == nil {
			problem.Details = map[string]any{}
		}
		problem.Details["warnings"] = said
	}
	return problem
}

// Run parses and executes one public CLI invocation.
func Run(args []string, stdout, stderr io.Writer) int {
	if len(args) == 1 && (args[0] == "-v" || args[0] == "-V" || args[0] == "--version") {
		fmt.Fprintln(stdout, version())
		return 0
	}

	args = helpArgs(args)
	wantsJSON := jsonRequested(args)
	terminal := config.ReadTerminal()
	var grammar CLI
	helpExit := -1
	parser, err := kong.New(&grammar,
		kong.Name("cozy"),
		kong.Description(description),
		kong.Writers(stdout, stderr),
		kong.Exit(func(code int) { helpExit = code }),
		kong.Help(cozyHelp),
		kong.ExplicitGroups([]kong.Group{
			{Key: "Packages", Title: "Packages"},
			{Key: "Models", Title: "Models"},
			{Key: "Authentication", Title: "Authentication"},
			{Key: "Runs", Title: "Runs"},
			{Key: "Rentals", Title: "Rentals"},
			{Key: "Lifecycle", Title: "Lifecycle"},
		}),
		kong.ConfigureHelp(kong.HelpOptions{Compact: true, FlagsLast: true, WrapUpperBound: 100}),
	)
	if err == nil && len(args) > 0 && args[0] == completeVerb {
		return complete(parser.Model, args[1:], stdout)
	}
	mode := presentationMode(stdout, wantsJSON, terminal)
	if err != nil {
		problem := output.NewError(output.Operational, "cli.grammar", err.Error())
		_ = output.EmitError(stdout, problem, mode)
		return output.ShellCode(problem)
	}

	args = normalizeRunArgs(args, parser.Model.Node)
	parsed, err := parser.Parse(args)
	mode = presentationMode(stdout, wantsJSON || grammar.JSON, terminal)
	mode.Full, mode.Fields = grammar.Full, grammar.Fields
	if helpExit >= 0 {
		return helpExit
	}
	if err != nil {
		if strings.Contains(err.Error(), "--rental") && strings.Contains(err.Error(), "value") {
			err = fmt.Errorf("--rental requires an existing rental name or id; use --rental-only for automatic allocation: %w", err)
		}
		problem := output.NewError(output.Usage, "cli.usage", err.Error()).
			WithNext(helpFor(parsed, args))
		_ = output.EmitError(stdout, problem, mode)
		return output.ShellCode(problem)
	}

	cfg, problem := config.LoadForTensorhub(grammar.Tensorhub)
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
	return runtime.exitCode
}

func presentationMode(w io.Writer, json bool, terminal config.Terminal) output.Mode {
	mode := output.Mode{JSON: json, Human: !json}
	if file, ok := w.(*os.File); ok && !json {
		mode.TTY = isatty.IsTerminal(file.Fd()) || isatty.IsCygwinTerminal(file.Fd())
		mode.Live = mode.TTY && !terminal.Dumb
		mode.Color = mode.Live && !terminal.NoColor
	}
	return mode
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

// normalizeRunArgs keeps the variadic payload contiguous for Kong while allowing
// options between its terms. Kong metadata protects option values; the existing
// payload parser still owns values and model overrides.
func normalizeRunArgs(args []string, application *kong.Node) []string {
	var execute *kong.Node
	for _, command := range application.Children {
		if command.Name == "run" {
			execute = command.DefaultCmd
			break
		}
	}
	if execute == nil {
		return args
	}
	valueFlags := map[string]bool{}
	for node := execute; node != nil; node = node.Parent {
		for _, flag := range node.Flags {
			valueFlags["--"+flag.Name] = !flag.IsBool() && !flag.IsCounter()
			for _, alias := range flag.Aliases {
				valueFlags["--"+alias] = !flag.IsBool() && !flag.IsCounter()
			}
			if flag.Short != 0 {
				valueFlags["-"+string(flag.Short)] = !flag.IsBool() && !flag.IsCounter()
			}
		}
	}
	var payload []string
	remove := map[int]bool{}
	inRun, target := false, -1
	for i := 0; i < len(args); i++ {
		arg := args[i]
		if arg == "--" {
			if target < 0 {
				return args
			}
			for j := i; j < len(args); j++ {
				remove[j] = true
			}
			payload = append(payload, args[i+1:]...)
			break
		}
		name, _, inline := strings.Cut(arg, "=")
		if inRun && inline && strings.HasPrefix(name, "--model.") {
			payload = append(payload, strings.TrimPrefix(arg, "--"))
			remove[i] = true
			continue
		}
		if strings.HasPrefix(arg, "-") {
			if !inline && valueFlags[name] {
				if i+1 == len(args) {
					return args // Leave the missing option value for Kong to report.
				}
				i++
			}
			continue
		}
		if !inRun {
			if arg != "run" {
				return args
			}
			inRun = true
		} else if target < 0 {
			for _, command := range execute.Parent.Children {
				if command != execute && command.Name == arg {
					return args
				}
			}
			target = i
		} else {
			payload = append(payload, arg)
			remove[i] = true
		}
	}
	if target < 0 || len(payload) == 0 {
		return args
	}
	out := make([]string, 0, len(args))
	for i, arg := range args {
		if !remove[i] {
			out = append(out, arg)
		}
	}
	// A single literal tail also preserves dash-prefixed values supplied after --.
	return append(append(out, "--"), payload...)
}

func helpArgs(args []string) []string {
	if len(args) == 0 {
		return []string{"--help"}
	}
	if len(args) == 1 {
		switch args[0] {
		case "package", "model", "rental":
			return []string{args[0], "--help"}
		case "run":
			return []string{"run", "org/package/function", "--help"}
		}
	}
	if len(args) == 2 && args[0] == "run" && (args[1] == "--help" || args[1] == "-h") {
		return []string{"run", "org/package/function", args[1]}
	}
	if args[0] != "help" {
		return args
	}
	if len(args) == 1 {
		return []string{"--help"}
	}
	if len(args) == 2 && args[1] == "run" {
		return []string{"run", "org/package/function", "--help"}
	}
	out := append([]string(nil), args[1:]...)
	return append(out, "--help")
}

func cozyHelp(options kong.HelpOptions, ctx *kong.Context) error {
	var rendered bytes.Buffer
	stdout := ctx.Stdout
	ctx.Stdout = &rendered
	if err := kong.DefaultHelpPrinter(options, ctx); err != nil {
		ctx.Stdout = stdout
		return err
	}
	ctx.Stdout = stdout
	help := strings.ReplaceAll(rendered.String(), "cozy run execute", "cozy run")
	if strings.TrimSpace(ctx.Command()) == "" {
		help = strings.Replace(help, "\nRuns\n",
			"\nRuns\n  run <org/package/function> [input]    Run a package function on a local or rented machine.\n", 1)
	}
	if _, err := io.WriteString(stdout, help); err != nil {
		return err
	}
	return nil
}

func helpFor(ctx *kong.Context, args []string) string {
	if ctx == nil || strings.TrimSpace(ctx.Command()) == "" {
		if len(args) > 0 {
			switch args[0] {
			case "package", "model", "auth", "run", "rental":
				return "cozy help " + args[0]
			}
		}
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
	out.Details = problem.Details
	if problem.Cause != "" {
		if out.Details == nil {
			out.Details = map[string]any{}
		}
		if _, ok := out.Details["error_code"]; !ok {
			out.Details["error_code"] = problem.Cause
		}
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
