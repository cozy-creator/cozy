package packagepublish

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"maps"
	"os"
	"path/filepath"
	"reflect"
	"slices"
	"strings"

	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/config"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/hostruntime"
)

// CommittedInterfacePath is where a package tree may keep a copy of its PackageInterface.
// Publication uploads this host's fresh reading of the source; a committed copy is only
// compared, and a contradiction is reported, never refused.
const CommittedInterfacePath = "metadata/package-interface.json"

// describe stages the tree's PackageInterface. The reading is THIS host's Runtime parsing the
// source (hostruntime.Describe): package code is untrusted and nothing here imports it, so no
// environment is built for the question. Both an unpublished revision and a publication ship
// that reading. For a publication (a nonempty account), a committed copy that contradicts it —
// something the copy names that the source no longer declares, or declares differently — is
// returned as a notice; ordering and members only the fresh reading carries are not
// contradictions. The staged publication names the account in every org-relative lane.
func describe(ctx context.Context, tree, root, account string) (string, string, *exit.Error) {
	publish := account != ""
	env := config.Frozen().Tool()
	if _, problem := hostruntime.Path(env); problem != nil {
		return "", "", problem
	}
	raw, problem := hostruntime.Describe(ctx, env, tree)
	if problem != nil {
		return "", "", exit.Named(exit.Validation, "package_interface_refused",
			"cozy-runtime could not describe the package").WithRemedy("%s", problem.Message)
	}
	if len(raw) == 0 || len(raw) > 1<<20 || !json.Valid(raw) {
		return "", "", exit.Named(exit.Structural, "package_interface_invalid",
			"cozy-runtime returned an invalid package interface")
	}
	notice := ""
	if publish {
		if committed, err := os.ReadFile(filepath.Join(tree, CommittedInterfacePath)); err == nil {
			if contradiction := contradicts(committed, raw); contradiction != "" {
				notice = fmt.Sprintf("%s contradicts this tree's source (%s); publishing the source's interface. "+
					"Delete or regenerate the committed copy.", CommittedInterfacePath, contradiction)
				fmt.Fprintf(os.Stderr, "cozy: %s\n", notice)
			}
		}
	}
	if publish {
		var problem *exit.Error
		if raw, problem = QualifyInterface(raw, account); problem != nil {
			return "", "", problem
		}
	}
	path := filepath.Join(root, "package-interface.json")
	if err := os.WriteFile(path, raw, 0o600); err != nil {
		return "", "", exit.Internalf("cannot stage package interface: %s", err)
	}
	return path, notice, nil
}

// contradicts names the first thing the committed copy states that the fresh reading does
// not, or "" when the copy is consistent with it.
func contradicts(committed, tree []byte) string {
	var a, b any
	if json.Unmarshal(committed, &a) != nil {
		return "the committed file is not JSON"
	}
	if json.Unmarshal(tree, &b) != nil {
		return "the tree's reading is not JSON"
	}
	return contradiction("", a, b)
}

func contradiction(path string, committed, tree any) string {
	switch c := committed.(type) {
	case map[string]any:
		t, ok := tree.(map[string]any)
		if !ok {
			return fmt.Sprintf("%s: committed %s, source %s", path, spell(committed), spell(tree))
		}
		for _, key := range slices.Sorted(maps.Keys(c)) {
			child := key
			if path != "" {
				child = path + "." + key
			}
			tv, inTree := t[key]
			if !inTree {
				return child + ": absent from the source"
			}
			if found := contradiction(child, c[key], tv); found != "" {
				return found
			}
		}
	case []any:
		t, ok := tree.([]any)
		if !ok {
			return fmt.Sprintf("%s: committed %s, source %s", path, spell(committed), spell(tree))
		}
		named := map[string]any{}
		for _, item := range t {
			if object, ok := item.(map[string]any); ok {
				if name, ok := object["name"].(string); ok {
					named[name] = item
				}
			}
		}
		for i, item := range c {
			if object, ok := item.(map[string]any); ok {
				if name, ok := object["name"].(string); ok {
					match, found := named[name]
					if !found {
						return fmt.Sprintf("%s: %q is absent from the source", path, name)
					}
					if found := contradiction(fmt.Sprintf("%s[%s]", path, name), item, match); found != "" {
						return found
					}
					continue
				}
			}
			if !slices.ContainsFunc(t, func(candidate any) bool { return reflect.DeepEqual(candidate, item) }) {
				return fmt.Sprintf("%s[%d]: committed %s is absent from the source", path, i, spell(item))
			}
		}
	default:
		if !reflect.DeepEqual(committed, tree) {
			return fmt.Sprintf("%s: committed %s, source %s", path, spell(committed), spell(tree))
		}
	}
	return ""
}

func spell(value any) string {
	raw, _ := json.Marshal(value)
	return string(raw)
}

// QualifyInterface writes account into every org-relative default lane
// (`model@release/lane` becomes `account/model@release/lane`). Explicit orgs are kept, so the
// published interface names only absolute references.
func QualifyInterface(raw []byte, account string) ([]byte, *exit.Error) {
	decoder := json.NewDecoder(bytes.NewReader(raw))
	decoder.UseNumber()
	var document map[string]any
	if err := decoder.Decode(&document); err != nil {
		return nil, exit.Named(exit.Structural, "package_interface_invalid", "the package interface is not a JSON object")
	}
	changed := false
	for _, group := range []string{"entrypoints", "jobs"} {
		callables, _ := document[group].([]any)
		for _, callable := range callables {
			row, _ := callable.(map[string]any)
			models, _ := row["models"].([]any)
			for _, model := range models {
				slot, _ := model.(map[string]any)
				ladder, _ := slot["default_ladder"].([]any)
				for _, rung := range ladder {
					fields, _ := rung.(map[string]any)
					lane, _ := fields["lane"].(string)
					if name, _, found := strings.Cut(lane, "@"); found && !strings.Contains(name, "/") {
						fields["lane"], changed = account+"/"+lane, true
					}
				}
			}
		}
	}
	if !changed {
		return raw, nil
	}
	var out bytes.Buffer
	encoder := json.NewEncoder(&out)
	encoder.SetEscapeHTML(false)
	if err := encoder.Encode(document); err != nil {
		return nil, exit.Internalf("cannot encode the qualified package interface: %s", err)
	}
	normalized, err := canonical.NormalizeJCS(bytes.TrimSpace(out.Bytes()))
	if err != nil {
		return nil, exit.Internalf("cannot normalize the qualified package interface: %s", err)
	}
	return normalized, nil
}
