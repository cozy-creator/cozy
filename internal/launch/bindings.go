package launch

import (
	"os"
	"path/filepath"
	"sort"
	"strings"

	"github.com/cozy-creator/cozy-creator-v2/internal/exit"
)

// The `[bindings]` table of an endpoint's `endpoint.toml`: CODE STATES CAPABILITY,
// BINDINGS STATE SELECTION (cozy-runtime.md §1.0). The author's code never names a repo
// or a release; this table does, keyed either by model CLASS (`[bindings."SdxlUnetModel"]`)
// or by the exact binding PATH (`[bindings."denoise.models.model"]`), path winning.
//
// This is a SECOND READER of the runtime's own grammar and it is deliberately narrow: the
// closed key set is `repo`, `release`, `lane` and nothing else, and a key outside it is a
// typed refusal rather than a silent ignore. The reason it exists at all is that no
// runtime verb emits the RESOLVED binding record — `describe` reports the surface,
// `fit` reports verdicts, and `run`'s local coordinator mints the record privately.
// Recorded as cl-010's seam for cr-016: the day the runtime can print its resolved
// binding record, this file deletes and the resolver reads that instead.

// Selection is one declared binding: which artifact a slot loads, and which lane.
type Selection struct {
	Key     string // the table key it came from: a class name or a binding path
	Repo    string
	Release string
	Lane    string
}

// Ref is the artifact ref this selection names: `org/repo[@release]`, the runtime's own
// spelling (internal/bindings.py `_ref`).
func (s Selection) Ref() string {
	if s.Release != "" {
		return s.Repo + "@" + s.Release
	}
	return s.Repo
}

// ReadBindings parses `<source>/endpoint.toml`'s [bindings] table. An endpoint with no
// table is not an error: a weightless endpoint declares no slot and needs no selection.
func ReadBindings(sourceDir string) (map[string]Selection, *exit.Error) {
	path := filepath.Join(sourceDir, "endpoint.toml")
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, exit.Named(exit.Structural, "endpoint_toml_absent",
			"this generation's source carries no endpoint.toml").
			WithRemedy("an endpoint declares its application object and its binding defaults there")
	}
	out := map[string]Selection{}
	key := ""
	current := Selection{}
	flush := func() *exit.Error {
		if key == "" {
			return nil
		}
		if current.Repo == "" {
			return exit.Named(exit.Validation, "binding_incomplete",
				"[bindings.%q] declares no repo=", key).
				WithRemedy("a binding names `repo` and optionally `release` and `lane`")
		}
		current.Key = key
		out[key] = current
		key, current = "", Selection{}
		return nil
	}
	for n, line := range strings.Split(string(data), "\n") {
		text := strings.TrimSpace(stripComment(line))
		if text == "" {
			continue
		}
		if strings.HasPrefix(text, "[") {
			if e := flush(); e != nil {
				return nil, e
			}
			name, ok := bindingHeader(text)
			if !ok {
				continue // some other table; endpoint.toml carries more than bindings
			}
			key = name
			continue
		}
		if key == "" {
			continue
		}
		field, value, ok := strings.Cut(text, "=")
		if !ok {
			return nil, tomlRefusal(path, n+1, text)
		}
		field = strings.TrimSpace(field)
		value = strings.Trim(strings.TrimSpace(value), `"`)
		switch field {
		case "repo":
			current.Repo = value
		case "release":
			current.Release = value
		case "lane":
			current.Lane = value
		default:
			// A key this reader does not know might CHANGE the selection, so ignoring it
			// would be the silent-wrong-artifact bug. Refuse and name it.
			return nil, exit.Named(exit.Validation, "binding_key_unknown",
				"[bindings.%q] declares %q, which this host's binding vocabulary does not name", key, field).
				WithRemedy("the closed set is repo, release, lane — a binding this build cannot read is never guessed")
		}
	}
	if e := flush(); e != nil {
		return nil, e
	}
	return out, nil
}

func tomlRefusal(path string, line int, text string) *exit.Error {
	return exit.Named(exit.Validation, "binding_malformed",
		"%s:%d is inside a [bindings] table and is not `key = value`: %s", path, line, text)
}

// bindingHeader recognises `[bindings."<key>"]` and `[bindings.<key>]`.
func bindingHeader(text string) (string, bool) {
	inner, ok := strings.CutPrefix(strings.TrimSuffix(text, "]"), "[bindings.")
	if !ok {
		return "", false
	}
	return strings.Trim(strings.TrimSpace(inner), `"`), true
}

func stripComment(line string) string {
	quoted := false
	for i, r := range line {
		switch r {
		case '"':
			quoted = !quoted
		case '#':
			if !quoted {
				return line[:i]
			}
		}
	}
	return line
}

// Select resolves one declared slot against the table, the runtime's ladder: the CLASS
// key first, the exact PATH key winning where both speak.
func Select(table map[string]Selection, slot Slot) (Selection, *exit.Error) {
	chosen, found := Selection{}, false
	for _, key := range []string{slot.Class, slot.Path} {
		if s, ok := table[key]; ok {
			chosen, found = s, true
		}
	}
	if !found {
		keys := make([]string, 0, len(table))
		for k := range table {
			keys = append(keys, k)
		}
		sort.Strings(keys)
		return Selection{}, exit.Named(exit.Validation, "unbound_slot",
			"%s (%s) has no binding: code states capability, bindings state selection", slot.Path, slot.Class).
			WithRemedy("endpoint.toml binds %s", strings.Join(keys, ", ")).
			WithNext("cozy describe <org/endpoint>")
	}
	return chosen, nil
}
