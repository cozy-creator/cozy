package packagepublish

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"reflect"
	"sort"

	"github.com/cozy-creator/cozy/internal/config"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/hostruntime"
)

// CommittedInterfacePath is where a publishable tree commits its own PackageInterface — the
// record Tensorhub accepts as truth (#713). Publication reads it and never re-derives it.
const CommittedInterfacePath = "metadata/package-interface.json"

// describe stages the tree's PackageInterface. The reading is THIS host's Runtime parsing the
// source (hostruntime.Describe): package code is untrusted and nothing here imports it, so no
// environment is built for the question. A private revision (Build) ships that reading. A
// publication (BuildForPublish) treats it as a pre-flight only — the committed file is what
// uploads — so a committed file that differs from the tree, or is absent, is refused naming
// the first difference, and an equal one is staged unchanged.
func describe(ctx context.Context, tree, root string, publish bool) (string, *exit.Error) {
	env := config.Frozen().Tool()
	if _, problem := hostruntime.Path(env); problem != nil {
		return "", problem
	}
	raw, problem := hostruntime.Describe(ctx, env, tree)
	if problem != nil {
		return "", exit.Named(exit.Validation, "package_interface_refused",
			"cozy-runtime could not describe the package").WithRemedy("%s", problem.Message)
	}
	if len(raw) == 0 || len(raw) > 1<<20 || !json.Valid(raw) {
		return "", exit.Named(exit.Structural, "package_interface_invalid",
			"cozy-runtime returned an invalid package interface")
	}
	if publish {
		committed, err := os.ReadFile(filepath.Join(tree, CommittedInterfacePath))
		if err != nil {
			return "", staleInterface("%s is absent", CommittedInterfacePath)
		}
		if difference := firstDifference(bytes.TrimSpace(committed), raw); difference != "" {
			return "", staleInterface("%s: %s", CommittedInterfacePath, difference)
		}
	}
	path := filepath.Join(root, "package-interface.json")
	if err := os.WriteFile(path, raw, 0o600); err != nil {
		return "", exit.Internalf("cannot stage package interface: %s", err)
	}
	return path, nil
}

func staleInterface(format string, args ...any) *exit.Error {
	return exit.Named(exit.Validation, "package_publish.interface_stale",
		"the committed PackageInterface is not this tree's — "+format, args...).
		WithRemedy("in the package tree run `cozy-runtime --json describe > %s` and commit it",
			CommittedInterfacePath)
}

// firstDifference names the first place the committed document and the tree's reading
// disagree — a path into the document and both values — or "" when they are equal.
func firstDifference(committed, tree []byte) string {
	if bytes.Equal(committed, tree) {
		return ""
	}
	var a, b any
	if json.Unmarshal(committed, &a) != nil {
		return "the committed file is not JSON"
	}
	if json.Unmarshal(tree, &b) != nil {
		return "the tree's reading is not JSON"
	}
	if difference := differ("", a, b); difference != "" {
		return difference
	}
	return "the committed file is not the canonical spelling of this tree's interface"
}

func differ(path string, committed, tree any) string {
	switch c := committed.(type) {
	case map[string]any:
		t, ok := tree.(map[string]any)
		if !ok {
			return fmt.Sprintf("%s: committed %s, tree %s", path, spell(committed), spell(tree))
		}
		keys := map[string]bool{}
		for key := range c {
			keys[key] = true
		}
		for key := range t {
			keys[key] = true
		}
		sorted := make([]string, 0, len(keys))
		for key := range keys {
			sorted = append(sorted, key)
		}
		sort.Strings(sorted)
		for _, key := range sorted {
			child := key
			if path != "" {
				child = path + "." + key
			}
			cv, inCommitted := c[key]
			tv, inTree := t[key]
			switch {
			case !inCommitted:
				return child + ": absent from the committed file"
			case !inTree:
				return child + ": absent from the tree"
			}
			if difference := differ(child, cv, tv); difference != "" {
				return difference
			}
		}
	case []any:
		t, ok := tree.([]any)
		if !ok {
			return fmt.Sprintf("%s: committed %s, tree %s", path, spell(committed), spell(tree))
		}
		for i := 0; i < len(c) && i < len(t); i++ {
			if difference := differ(fmt.Sprintf("%s[%d]", path, i), c[i], t[i]); difference != "" {
				return difference
			}
		}
		if len(c) != len(t) {
			return fmt.Sprintf("%s: committed %d entries, tree %d", path, len(c), len(t))
		}
	default:
		if !reflect.DeepEqual(committed, tree) {
			return fmt.Sprintf("%s: committed %s, tree %s", path, spell(committed), spell(tree))
		}
	}
	return ""
}

func spell(value any) string {
	raw, _ := json.Marshal(value)
	return string(raw)
}
