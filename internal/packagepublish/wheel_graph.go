package packagepublish

// uv owns dependency resolution, markers, and extra activation. This reader only
// projects its locked graph onto the environment that uv actually installed.

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"os/exec"
	"sort"
	"strings"

	"github.com/cozy-creator/cozy/internal/config"
	"github.com/cozy-creator/cozy/internal/exit"
)

const wheelGraphUV = "0.12.11"

type graphEdge struct {
	ID string `json:"id"`
}
type graphNode struct {
	Name         string          `json:"name"`
	Version      string          `json:"version"`
	Kind         json.RawMessage `json:"kind"`
	Dependencies []graphEdge     `json:"dependencies"`
	Optional     []struct {
		Name string `json:"name"`
		ID   string `json:"id"`
	} `json:"optional_dependencies"`
}
type wheelGraph struct {
	Schema struct {
		Version string `json:"version"`
	} `json:"schema"`
	Roots      []graphEdge          `json:"roots"`
	Inverted   bool                 `json:"inverted"`
	Resolution map[string]graphNode `json:"resolution"`
}

// WheelClosures contains one exact selected transitive name/version roster per
// callable dependency. Paths, caller source, inactive extras, and unrelated
// installed development dependencies never enter a callee's roster.
func WheelClosures(ctx context.Context, tree, python, installed, project, extra string) (map[string]map[string]string, *exit.Error) {
	version := exec.CommandContext(ctx, "uv", "--version")
	version.Env = config.Frozen().Tool()
	raw, err := version.Output()
	fields := strings.Fields(string(raw))
	if err != nil || len(fields) < 2 || fields[0] != "uv" || (fields[1] != wheelGraphUV && fields[1] != "0.12.7") {
		return nil, exit.Named(exit.Structural, "private_wheel_graph_uv_unsupported",
			"installed callable wheel discovery requires uv %s's qualified graph schema", wheelGraphUV).
			WithRemedy("install uv %s, then retry cozy run", wheelGraphUV)
	}
	command := exec.CommandContext(ctx, "uv", "tree", "--frozen", "--no-dev", "--no-default-groups", "--format", "json", "--python", python)
	command.Dir, command.Env = tree, config.Frozen().Tool()
	var stderr strings.Builder
	command.Stderr = &stderr
	stdout, err := command.StdoutPipe()
	if err != nil {
		return nil, exit.Internalf("cannot read uv dependency graph")
	}
	if err := command.Start(); err != nil {
		return nil, exit.Internalf("cannot start uv dependency graph")
	}
	raw, readErr := io.ReadAll(io.LimitReader(stdout, maxLockBytes+1))
	if int64(len(raw)) > maxLockBytes {
		_ = command.Process.Kill()
	}
	err = command.Wait()
	if readErr != nil || err != nil || int64(len(raw)) > maxLockBytes {
		return nil, exit.Named(exit.Structural, "private_wheel_graph_refused", "uv could not provide its bounded locked dependency graph")
	}
	return readWheelClosures(raw, installed, project, extra)
}

func readWheelClosures(raw []byte, installed, project, extra string) (map[string]map[string]string, *exit.Error) {
	pins, problem := privatePins(installed)
	if problem != nil {
		return nil, problem
	}
	var graph wheelGraph
	if len(raw) == 0 || int64(len(raw)) > maxLockBytes || json.Unmarshal(raw, &graph) != nil || graph.Schema.Version != "preview" || graph.Inverted || len(graph.Resolution) > 4096 {
		return nil, exit.Named(exit.Validation, "private_wheel_graph_invalid", "uv dependency graph schema is not the qualified preview format")
	}
	var roots []string
	for _, edge := range graph.Roots {
		node, ok := graph.Resolution[edge.ID]
		if ok && node.Name == project && node.Version == pins[project] && bytes.Equal(node.Kind, []byte(`"package"`)) {
			roots = append(roots, edge.ID)
			if extra != "" {
				found := false
				for _, optional := range node.Optional {
					if optional.Name == extra {
						roots = append(roots, optional.ID)
						found = true
					}
				}
				if !found {
					return nil, exit.New(exit.Conflict, "installed project extra is absent from uv's graph")
				}
			}
		}
	}
	if len(roots) != 1 && !(extra != "" && len(roots) == 2) {
		return nil, exit.New(exit.Conflict, "uv's graph does not name one exact installed project root")
	}
	walk := func(roots []string) (map[string]bool, *exit.Error) {
		seen := map[string]bool{}
		queue := append([]string(nil), roots...)
		for len(queue) > 0 {
			id := queue[0]
			queue = queue[1:]
			if seen[id] {
				continue
			}
			node, ok := graph.Resolution[id]
			if !ok || node.Name == "" || normalizedProjectName(node.Name) != node.Name || pins[node.Name] != node.Version {
				return nil, exit.New(exit.Conflict, "uv dependency graph differs from the installed closure")
			}
			seen[id] = true
			for _, edge := range node.Dependencies {
				queue = append(queue, edge.ID)
			}
		}
		return seen, nil
	}
	active, problem := walk(roots)
	if problem != nil {
		return nil, problem
	}
	byName := map[string][]string{}
	for id := range active {
		name := graph.Resolution[id].Name
		if name != project {
			byName[name] = append(byName[name], id)
		}
	}
	out := map[string]map[string]string{}
	for name, roots := range byName {
		selected, problem := walk(roots)
		if problem != nil {
			return nil, problem
		}
		closure := map[string]string{}
		for id := range selected {
			node := graph.Resolution[id]
			if node.Name == project {
				return nil, exit.New(exit.Validation, "a callable library depends cyclically on its caller project")
			}
			closure[node.Name] = node.Version
		}
		out[name] = closure
	}
	return out, nil
}

func PinnedClosure(pins map[string]string) string {
	rows := make([]string, 0, len(pins))
	for name, version := range pins {
		rows = append(rows, name+"=="+version)
	}
	sort.Strings(rows)
	return strings.Join(rows, "\n")
}
