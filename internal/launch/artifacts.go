package launch

import (
	"context"
	"encoding/json"
	"os/exec"
	"path/filepath"
	"strings"
	"time"

	"github.com/cozy-creator/cozy-creator/internal/exit"
	"github.com/cozy-creator/cozy-creator/internal/home"
)

// DefaultRuntimeQueryTimeout bounds metadata-only runtime questions used while selecting
// or starting a worker. These verbs read declarations; they do not construct an endpoint,
// inspect a device, or load weights. A runtime that cannot answer them promptly is stalled,
// and must not hold an orchestrator slot's in-memory starting fence forever.
const DefaultRuntimeQueryTimeout = 5 * time.Second

// THE LOCAL ARTIFACT INDEX IS NOT READ HERE ANY MORE (#567e).
//
// This file used to carry `ModelArtifact`, `Artifacts()` and `Find()` — a whole capability
// for turning a ref into store roots and snapshot digests by asking the runtime's `list`
// verb. Every caller of it was the binding-record writer, and #565b's rule says that writer
// is the wrong machine to ask: RESOLUTION IS THE RESOLVER'S OWN. A record now names the ref
// and the SERVING machine resolves it against its own index, so the capability has no
// honest caller left and is deleted rather than kept as an entry point nobody may use
// (#496e: the deletion unit is the capability, not the function).
//
// The refusal it used to raise moved with it, and improved: an artifact this host lacks is
// no longer exit 4 on the owner's box before a request is even sent, it is a typed
// BINDING_UNAVAILABLE from the pod that would have served it, naming that pod's own index.

// RuntimeCLI is one generation's own cozy-runtime binary, run against a named local root.
// Every question this host asks the runtime goes through here, so there is one place
// that knows how to invoke it and one place that renders its refusals.
type RuntimeCLI struct {
	Bin          string        // the generation venv's cozy-runtime (home.VenvTool spells the platform)
	Dir          string        // the endpoint project root
	Home         string        // COZY_HOME the runtime reads its artifact index out of
	Env          []string      // the allowlisted child environment (config.Tool)
	QueryTimeout time.Duration // metadata-query bound; zero selects DefaultRuntimeQueryTimeout
}

// Binary is the runtime a generation carries. An install already refused a generation
// whose venv provides none (cl-009's `runtime_missing`), so this is the same claim,
// re-asserted where it is used.
func Binary(generationDir string) string {
	return home.VenvTool(filepath.Join(generationDir, "venv"), "cozy-runtime")
}

// json runs one verb and decodes its `--json` document.
func (r RuntimeCLI) call(out any, verb ...string) *exit.Error {
	return r.callContext(context.Background(), out, verb...)
}

// query runs a metadata-only verb under the selection/start liveness bound. Fit and
// doctor deliberately continue through call: they inspect a real host and may perform
// work whose duration cannot honestly be represented by this metadata deadline.
func (r RuntimeCLI) query(out any, verb ...string) *exit.Error {
	timeout := r.QueryTimeout
	if timeout <= 0 {
		timeout = DefaultRuntimeQueryTimeout
	}
	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()
	return r.callContext(ctx, out, verb...)
}

func (r RuntimeCLI) callContext(ctx context.Context, out any, verb ...string) *exit.Error {
	args := append([]string{"--json", "--dir", r.Dir}, verb...)
	cmd := exec.CommandContext(ctx, r.Bin, args...)
	if _, bounded := ctx.Deadline(); bounded {
		// CommandContext kills the direct child at the deadline. WaitDelay also closes a
		// pipe retained by a misbehaving descendant, so the query itself remains bounded.
		cmd.WaitDelay = 250 * time.Millisecond
	}
	cmd.Env = append(append([]string{}, r.Env...), "COZY_HOME="+r.Home)
	var stdout, stderr strings.Builder
	cmd.Stdout, cmd.Stderr = &stdout, &stderr
	err := cmd.Run()
	if ctx.Err() == context.DeadlineExceeded {
		return exit.Named(exit.Deadline, "runtime_query_stalled",
			"`cozy-runtime %s` did not answer its metadata query within %s",
			strings.Join(verb, " "), r.queryTimeout()).
			WithRemedy("the release's runtime must answer bindings and job-describe metadata without importing or constructing the endpoint")
	}
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

func (r RuntimeCLI) queryTimeout() time.Duration {
	if r.QueryTimeout > 0 {
		return r.QueryTimeout
	}
	return DefaultRuntimeQueryTimeout
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

// WeightlessPlan is one exact ArtifactSubject for a canonical weightless binding plan.
// The runtime is the sole writer of those documents. Creator consumes their identities
// here before spawn and never reconstructs the private bytes.
type WeightlessPlan struct {
	Entrypoint string `json:"entrypoint"`
	SubjectID  string `json:"subject_id"`
	Kind       string `json:"kind"`
	Digest     string `json:"digest"`
	Length     uint64 `json:"length"`
}

// Bindings is what this project SELECTS, resolved by the one resolver that owns the
// grammar. It constructs nothing, touches no device and loads no weights.
func (r RuntimeCLI) Bindings() ([]Binding, []WeightlessPlan, *exit.Error) {
	var doc struct {
		Bindings        []Binding        `json:"bindings"`
		WeightlessPlans []WeightlessPlan `json:"weightless_plans"`
	}
	if e := r.query(&doc, "bindings"); e != nil {
		return nil, nil, e
	}
	return doc.Bindings, doc.WeightlessPlans, nil
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
