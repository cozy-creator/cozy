package launch

import (
	"encoding/json"
	"os/exec"
	"path/filepath"
	"sort"
	"strings"

	"github.com/cozy-creator/cozy-creator-v2/internal/exit"
)

// The LOCAL ARTIFACT INDEX, read through the runtime's own `list` verb.
//
// cozy-runtime-cli.md §Local authority makes "install local snapshot refs" a CLI-layer
// power: the runtime CORE is handed exact store roots and snapshot digests and never
// resolves a ref, and `cozy_runtime.cli.artifacts` is where a ref becomes those facts.
// cozy-creator asks that owner rather than reading its files — the index's layout is the
// runtime's, and a second reader of it would be a second layout to keep in step.

// Artifact is one locally installed checkpoint, in the runtime's own vocabulary.
type Artifact struct {
	Ref            string            `json:"ref"`
	Store          string            `json:"store"`
	Config         string            `json:"config"`
	Snapshots      map[string]string `json:"snapshots"`
	Lane           string            `json:"lane"`
	Bytes          int64             `json:"bytes"`
	Tensors        int               `json:"tensors"`
	Variant        string            `json:"variant"`
	Custody        string            `json:"custody"`
	VRAMFloorBytes int64             `json:"vram_floor_bytes"`
}

// Components is the artifact's component set, ordered as the index orders it.
func (a Artifact) Components() []string {
	out := make([]string, 0, len(a.Snapshots))
	for name := range a.Snapshots {
		out = append(out, name)
	}
	sort.Strings(out)
	return out
}

// Runtime is one generation's own cozy-runtime binary, run against a named local root.
// Every question this host asks the runtime goes through here, so there is one place
// that knows how to invoke it and one place that renders its refusals.
type Runtime struct {
	Bin  string   // <generation>/venv/bin/cozy-runtime
	Dir  string   // the endpoint project root
	Home string   // COZY_HOME the runtime reads its artifact index out of
	Env  []string // the allowlisted child environment (config.Tool)
}

// Binary is the runtime a generation carries. An install already refused a generation
// whose venv provides none (cl-009's `runtime_missing`), so this is the same claim,
// re-asserted where it is used.
func Binary(generationDir string) string {
	return filepath.Join(generationDir, "venv", "bin", "cozy-runtime")
}

// json runs one verb and decodes its `--json` document.
func (r Runtime) call(out any, verb ...string) *exit.Error {
	args := append([]string{"--json", "--dir", r.Dir}, verb...)
	cmd := exec.Command(r.Bin, args...)
	cmd.Env = append(append([]string{}, r.Env...), "COZY_HOME="+r.Home)
	var stdout, stderr strings.Builder
	cmd.Stdout, cmd.Stderr = &stdout, &stderr
	err := cmd.Run()
	if cmd.ProcessState == nil {
		return exit.Named(exit.Structural, "runtime_missing",
			"cannot run %s: %s", r.Bin, err).
			WithRemedy("an endpoint's surface is answered by the runtime the release itself pinned")
	}
	if code := cmd.ProcessState.ExitCode(); code != 0 {
		return runtimeRefusal(code, strings.Join(verb, " "), stdout.String(), stderr.String())
	}
	if out == nil {
		return nil
	}
	if err := json.Unmarshal([]byte(stdout.String()), out); err != nil {
		return exit.Internalf("`cozy-runtime %s` answered a document this host cannot read: %s",
			strings.Join(verb, " "), err)
	}
	return nil
}

// runtimeRefusal renders the runtime's own words under its own exit code. The matrix is
// SHARED (cozy-runtime-cli.md), so a runtime exit is already a cozy exit — mapping it to
// something else here would be inventing a second vocabulary for one refusal.
func runtimeRefusal(code int, verb, stdout, stderr string) *exit.Error {
	said := strings.TrimSpace(stderr)
	if said == "" {
		said = strings.TrimSpace(stdout)
	}
	// The runtime's --json refusal is one document; render its message rather than the
	// raw JSON when it parses.
	var doc struct {
		Error struct {
			Name    string `json:"name"`
			Message string `json:"message"`
			Remedy  string `json:"remedy"`
		} `json:"error"`
	}
	name, remedy := "runtime_refused", ""
	if json.Unmarshal([]byte(stdout), &doc) == nil && doc.Error.Message != "" {
		said, remedy = doc.Error.Message, doc.Error.Remedy
		if doc.Error.Name != "" {
			name = doc.Error.Name
		}
	}
	c := exit.Code(code)
	if !c.Valid() {
		c = exit.Internal
	}
	e := exit.Named(c, name, "`cozy-runtime %s`: %s", verb, condense(said))
	if remedy != "" {
		e.WithRemedy("%s", remedy)
	}
	return e
}

func condense(s string) string {
	s = strings.Join(strings.Fields(s), " ")
	if len(s) > 400 {
		return s[:400] + "…"
	}
	return s
}

// Artifacts is the local artifact index, as the runtime reports it.
func (r Runtime) Artifacts() ([]Artifact, *exit.Error) {
	var rows []Artifact
	if e := r.call(&rows, "list"); e != nil {
		return nil, e
	}
	return rows, nil
}

// Find resolves one ref against the index. A miss is exit 4 naming the pull that fixes
// it — the same refusal the runtime's own local coordinator raises for the same cause.
func (r Runtime) Find(ref string) (Artifact, *exit.Error) {
	rows, e := r.Artifacts()
	if e != nil {
		return Artifact{}, e
	}
	for _, a := range rows {
		if a.Ref == ref {
			return a, nil
		}
	}
	// A bare `org/repo` matches any release of it, exactly as the index's own lookup does.
	if !strings.Contains(ref, "@") {
		for _, a := range rows {
			if strings.HasPrefix(a.Ref, ref+"@") {
				return a, nil
			}
		}
	}
	have := make([]string, 0, len(rows))
	for _, a := range rows {
		have = append(have, a.Ref)
	}
	sort.Strings(have)
	held := strings.Join(have, ", ")
	if held == "" {
		held = "nothing"
	}
	return Artifact{}, exit.New(exit.NotFound,
		"%s is not in this host's local artifact index", ref).
		WithRemedy("the index holds: %s", held).
		WithNext("cozy pull " + ref)
}

// Row is one `key: value` line of a runtime verb's own rendering.
type Row struct{ Key, Value string }

// Verdicts runs `fit` and hands back the runtime's own ROWS, verbatim. cozy-creator
// renders them; it derives no verdict of its own (cozy-creator.md: "the runtime derives").
//
// DELIBERATELY NOT `--json`, and the reason is a live defect rather than a preference:
// `cozy-runtime fit --json` faults on this build — `HostFacts` is a slotted dataclass and
// the document builder reads `facts.__dict__` — so `--json` is exit 1 `internal` for every
// invocation. The row rendering is the same verdict from the same walk. Recorded as
// cl-010's finding for cr-016; when the document works this reads it instead.
func (r Runtime) Verdicts(function string, payload []string) ([]Row, *exit.Error) {
	verb := []string{"fit"}
	if function != "" {
		verb = append(verb, function)
	}
	verb = append(verb, payload...)
	out, e := r.lines("fit", verb...)
	if e != nil {
		return nil, e
	}
	rows := []Row{}
	for _, line := range strings.Split(out, "\n") {
		key, value, ok := strings.Cut(strings.TrimRight(line, "\n"), ":")
		if !ok || strings.HasPrefix(line, " ") {
			continue
		}
		rows = append(rows, Row{strings.TrimSpace(key), strings.TrimSpace(value)})
	}
	return rows, nil
}

// lines runs one verb WITHOUT --json and returns its stdout.
func (r Runtime) lines(name string, verb ...string) (string, *exit.Error) {
	args := append([]string{"--dir", r.Dir}, verb...)
	cmd := exec.Command(r.Bin, args...)
	cmd.Env = append(append([]string{}, r.Env...), "COZY_HOME="+r.Home)
	var stdout, stderr strings.Builder
	cmd.Stdout, cmd.Stderr = &stdout, &stderr
	err := cmd.Run()
	if cmd.ProcessState == nil {
		return "", exit.Named(exit.Structural, "runtime_missing",
			"cannot run %s: %s", r.Bin, err)
	}
	if code := cmd.ProcessState.ExitCode(); code != 0 {
		return "", runtimeRefusal(code, name, stdout.String(), stderr.String())
	}
	return stdout.String(), nil
}

// HostFacts runs `doctor` — device/driver/CUDA tri-state, encoder capability, CAS and
// credential presence. Host health only; it derives no fit verdict.
func (r Runtime) HostFacts() (map[string]any, *exit.Error) {
	var doc map[string]any
	if e := r.call(&doc, "doctor"); e != nil {
		return nil, e
	}
	return doc, nil
}
