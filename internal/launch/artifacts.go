package launch

import (
	"encoding/json"
	"os/exec"
	"path/filepath"
	"sort"
	"strings"

	"github.com/cozy-creator/cozy-creator-v2/internal/exit"
	"github.com/cozy-creator/cozy-creator-v2/internal/home"
)

// The LOCAL ARTIFACT INDEX, read through the runtime's own `list` verb.
//
// cozy-runtime-cli.md §Local authority makes "install local snapshot refs" a CLI-layer
// power: the runtime CORE is handed exact store roots and snapshot digests and never
// resolves a ref, and `cozy_runtime.cli.artifacts` is where a ref becomes those facts.
// cozy-creator asks that owner rather than reading its files — the index's layout is the
// runtime's, and a second reader of it would be a second layout to keep in step.

// ModelArtifact is one locally installed checkpoint, in the runtime's own vocabulary.
type ModelArtifact struct {
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
func (a ModelArtifact) Components() []string {
	out := make([]string, 0, len(a.Snapshots))
	for name := range a.Snapshots {
		out = append(out, name)
	}
	sort.Strings(out)
	return out
}

// RuntimeCLI is one generation's own cozy-runtime binary, run against a named local root.
// Every question this host asks the runtime goes through here, so there is one place
// that knows how to invoke it and one place that renders its refusals.
type RuntimeCLI struct {
	Bin  string   // the generation venv's cozy-runtime (home.VenvTool spells the platform)
	Dir  string   // the endpoint project root
	Home string   // COZY_HOME the runtime reads its artifact index out of
	Env  []string // the allowlisted child environment (config.Tool)
}

// Binary is the runtime a generation carries. An install already refused a generation
// whose venv provides none (cl-009's `runtime_missing`), so this is the same claim,
// re-asserted where it is used.
func Binary(generationDir string) string {
	return home.VenvTool(filepath.Join(generationDir, "venv"), "cozy-runtime")
}

// json runs one verb and decodes its `--json` document.
func (r RuntimeCLI) call(out any, verb ...string) *exit.Error {
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
func (r RuntimeCLI) Artifacts() ([]ModelArtifact, *exit.Error) {
	var rows []ModelArtifact
	if e := r.call(&rows, "list"); e != nil {
		return nil, e
	}
	return rows, nil
}

// Find resolves one ref against the index. A miss is exit 4 naming the pull that fixes
// it — the same refusal the runtime's own local orchestrator raises for the same cause.
func (r RuntimeCLI) Find(ref string) (ModelArtifact, *exit.Error) {
	rows, e := r.Artifacts()
	if e != nil {
		return ModelArtifact{}, e
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
	return ModelArtifact{}, exit.New(exit.NotFound,
		"%s is not in this host's local artifact index", ref).
		WithRemedy("the index holds: %s", held).
		WithNext("cozy pull " + ref)
}

// Binding is one RESOLVED binding record, exactly as `cozy-runtime bindings` reports it.
// It is the runtime's own resolution of `endpoint.toml`'s selection grammar against the
// declared slots and the local artifact index. cozy-creator asks the owner rather than
// reading the table itself: cr-016 built this verb so that second reader could delete.
type Binding struct {
	Path       string            `json:"model_binding_path"`
	Param      string            `json:"model_parameter_name"`
	ModelClass string            `json:"model_class"`
	Ref        string            `json:"ref"`
	Lane       string            `json:"lane"`
	Source     string            `json:"source"`
	Components []string          `json:"components"`
	Store      string            `json:"store"`
	Snapshots  map[string]string `json:"snapshots"`
	Custody    string            `json:"custody"`
	Installed  bool              `json:"installed"`
}

// Bindings is what this project SELECTS, resolved by the one resolver that owns the
// grammar. It constructs nothing, touches no device and loads no weights.
func (r RuntimeCLI) Bindings() ([]Binding, *exit.Error) {
	var doc struct {
		Bindings []Binding `json:"bindings"`
	}
	if e := r.call(&doc, "bindings"); e != nil {
		return nil, e
	}
	return doc.Bindings, nil
}

// Verdict is one function's fit verdict, as the runtime's own document carries it.
type Verdict struct {
	Function string `json:"function"`
	Verdict  string `json:"verdict"`
	Rendered string `json:"rendered"`
	Exact    bool   `json:"exact"`
}

// Fit runs `fit --json` and hands back the runtime's own HOST FACTS and VERDICTS.
// cozy-creator renders them and derives no verdict of its own. cl-010 had to parse the
// row RENDERING because `fit --json` faulted on that build (a slotted dataclass read
// through `__dict__`); cr-016 fixed it, and reading the document is what deletes the
// parser.
func (r RuntimeCLI) Fit(function string, payload []string) (map[string]any, []Verdict, *exit.Error) {
	var doc struct {
		Host     map[string]any `json:"host"`
		Verdicts []Verdict      `json:"verdicts"`
	}
	verb := []string{"fit"}
	if function != "" {
		verb = append(verb, function)
	}
	verb = append(verb, payload...)
	if e := r.call(&doc, verb...); e != nil {
		return nil, nil, e
	}
	return doc.Host, doc.Verdicts, nil
}

// HostFacts runs `doctor` — device/driver/CUDA tri-state, encoder capability, CAS and
// credential presence. Host health only; it derives no fit verdict.
func (r RuntimeCLI) HostFacts() (map[string]any, *exit.Error) {
	var doc map[string]any
	if e := r.call(&doc, "doctor"); e != nil {
		return nil, e
	}
	return doc, nil
}
