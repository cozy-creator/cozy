// Package manifest is the ONE declarative command surface. It is data: dispatch,
// help, capability tokens and the docs inventory all read this and nothing else.
// A row and its handler are bound by a string key, checked at startup.
package manifest

import (
	"fmt"
	"sort"
	"strings"

	"github.com/cozy-creator/cozy-creator-v2/internal/exit"
)

type Status string

const (
	Implemented Status = "implemented"
	Planned     Status = "planned"
)

type Flag struct {
	Name    string // long form, e.g. "--json"
	Short   string // optional, e.g. "-d"
	Arg     string // "" = boolean, else the value placeholder
	Summary string
}

func (f Flag) TakesValue() bool { return f.Arg != "" }

type Command struct {
	Path        []string // {"job","submit"}
	Group       string
	Summary     string
	Args        string // positional spec for help/usage
	MinArgs     int
	MaxArgs     int // -1 = unbounded
	Flags       []Flag
	Exits       []exit.Code
	Capability  string
	Destructive bool // refuses without --yes (exit 7)
	PlanFirst   bool // carries --yes but is NOT destructive: bare it prints the plan, exit 0
	NeedsServer bool // refuses with unavailable (exit 9) while the LocalService is down
	Terminals   bool // help renders the job terminal mapping
	Status      Status
	Issue       string // owning issue for a planned row
	Handler     string // handler key; empty iff Status == Planned
}

func (c *Command) Name() string { return strings.Join(c.Path, " ") }

// Groups is the display order of command groups.
var Groups = []string{"meta", "service", "endpoints", "invocation", "workflows", "jobs", "rentals", "catalog", "transfer", "account"}

// GlobalFlags apply to every command.
var GlobalFlags = []Flag{
	{Name: "--json", Summary: "emit the full typed result as one JSON document"},
	{Name: "--full", Summary: "widen columns and lift truncation"},
	{Name: "--fields", Arg: "<a,b,c>", Summary: "pick listing columns"},
	{Name: "--dir", Arg: "<path>", Summary: "project root (default cwd)"},
	{Name: "--help", Short: "-h", Summary: "concise help for this command; never mutates"},
}

// Lookup resolves the longest matching command path, returning the consumed word count.
func Lookup(words []string) (*Command, int) {
	for n := 2; n >= 1; n-- {
		if len(words) < n {
			continue
		}
		for i := range Commands {
			c := &Commands[i]
			if len(c.Path) != n {
				continue
			}
			match := true
			for j := 0; j < n; j++ {
				if c.Path[j] != words[j] {
					match = false
					break
				}
			}
			if match {
				return c, n
			}
		}
	}
	return nil, 0
}

// Extendable answers whether any row EXTENDS this word with a second one. It exists for
// the parser: `rent` and `rent ls` are both verbs, so a one-word match must be HELD until
// the next word says which one was meant. Binding the shorter row the moment it matched
// made every two-word verb under a one-word verb unreachable and turned its own second
// word into an argument the shorter row then refused (`login` / `login ls` had it too).
func Extendable(word string) bool {
	for i := range Commands {
		if len(Commands[i].Path) == 2 && Commands[i].Path[0] == word {
			return true
		}
	}
	return false
}

// Suggest returns near-miss command names for an unknown word.
func Suggest(word string) []string {
	var out []string
	for i := range Commands {
		head := Commands[i].Path[0]
		if strings.HasPrefix(head, word) || strings.HasPrefix(word, head) || editDistance(head, word) <= 2 {
			if !containsStr(out, head) {
				out = append(out, head)
			}
		}
	}
	sort.Strings(out)
	if len(out) > 3 {
		out = out[:3]
	}
	return out
}

func hasYes(c *Command) bool {
	for _, f := range c.Flags {
		if f.Name == "--yes" {
			return true
		}
	}
	return false
}

func containsStr(s []string, v string) bool {
	for _, x := range s {
		if x == v {
			return true
		}
	}
	return false
}

func editDistance(a, b string) int {
	prev := make([]int, len(b)+1)
	cur := make([]int, len(b)+1)
	for j := range prev {
		prev[j] = j
	}
	for i := 1; i <= len(a); i++ {
		cur[0] = i
		for j := 1; j <= len(b); j++ {
			cost := 1
			if a[i-1] == b[j-1] {
				cost = 0
			}
			cur[j] = min(prev[j]+1, min(cur[j-1]+1, prev[j-1]+cost))
		}
		copy(prev, cur)
	}
	return prev[len(b)]
}

// Capabilities is the feature-token set: foundation tokens plus the token of every
// IMPLEMENTED row. A planned row contributes nothing — tokens describe what this
// binary can actually do.
func Capabilities() []string {
	out := Tokens()
	for i := range Commands {
		c := &Commands[i]
		if c.Status == Implemented && c.Capability != "" {
			out = append(out, c.Capability)
		}
	}
	sort.Strings(out)
	return out
}

// SelfCheck refuses at startup unless the manifest and the handler registry agree.
// Advertised-with-no-handler and handler-not-advertised are both incoherence.
func SelfCheck(handlers []string) *exit.Error {
	var problems []string
	seenPath := map[string]bool{}
	claimed := map[string]bool{}
	registry := map[string]bool{}
	for _, h := range handlers {
		registry[h] = true
	}

	for i := range Commands {
		c := &Commands[i]
		name := c.Name()
		if seenPath[name] {
			problems = append(problems, fmt.Sprintf("duplicate command row %q", name))
		}
		seenPath[name] = true
		if c.Summary == "" {
			problems = append(problems, fmt.Sprintf("%q has no summary", name))
		}
		if !containsStr(Groups, c.Group) {
			problems = append(problems, fmt.Sprintf("%q has unknown group %q", name, c.Group))
		}
		for _, e := range c.Exits {
			if !e.Valid() {
				problems = append(problems, fmt.Sprintf("%q advertises exit %d, outside the shared matrix", name, int(e)))
			}
		}
		// A --yes row is either destructive (exit 7 without it) or plan-first
		// (a read without it) — never both, never neither, never silent.
		if hasYes(c) && c.Destructive == c.PlanFirst {
			problems = append(problems, fmt.Sprintf(
				"%q takes --yes but is neither Destructive nor PlanFirst (exactly one)", name))
		}
		if c.Destructive && !hasYes(c) {
			problems = append(problems, fmt.Sprintf("%q is destructive but advertises no --yes flag", name))
		}
		switch c.Status {
		case Implemented:
			if c.Handler == "" {
				problems = append(problems, fmt.Sprintf("%q is advertised as implemented but names no handler", name))
				continue
			}
			if !registry[c.Handler] {
				problems = append(problems, fmt.Sprintf("%q is advertised but handler %q is not registered", name, c.Handler))
			}
			claimed[c.Handler] = true
		case Planned:
			if c.Handler != "" {
				problems = append(problems, fmt.Sprintf("%q is planned but names handler %q", name, c.Handler))
			}
			if c.Issue == "" {
				problems = append(problems, fmt.Sprintf("%q is planned but names no owning issue", name))
			}
		default:
			problems = append(problems, fmt.Sprintf("%q has unknown status %q", name, c.Status))
		}
	}
	for _, h := range handlers {
		if !claimed[h] {
			problems = append(problems, fmt.Sprintf("handler %q is registered but no command advertises it", h))
		}
	}
	if len(problems) == 0 {
		return nil
	}
	sort.Strings(problems)
	return exit.Named(exit.Internal, "manifest_incoherent",
		"the command manifest and the handler registry disagree (%d problem(s)): %s",
		len(problems), strings.Join(problems, "; ")).
		WithRemedy("every implemented row must name a registered handler and every registered handler must be advertised").
		WithNext("cozy commands --full")
}
