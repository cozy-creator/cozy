package launch

import (
	"context"
	"encoding/json"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"time"

	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/home"
	"github.com/cozy-creator/cozy/internal/records"
)

// InstallToolEnv makes one installed package's exact Runtime discoverable through the
// already-frozen child environment without reading ambient process state.
func InstallToolEnv(inst records.PackageInstall, env []string) []string {
	out := append([]string(nil), env...)
	prefix := filepath.Dir(Binary(inst))
	for index, value := range out {
		if strings.HasPrefix(value, "PATH=") {
			out[index] = "PATH=" + prefix + string(os.PathListSeparator) + strings.TrimPrefix(value, "PATH=")
			return out
		}
	}
	return append(out, "PATH="+prefix)
}

// DefaultRuntimeQueryTimeout bounds metadata-only runtime questions used while selecting
// or starting a worker. These verbs read declarations; they do not construct an package,
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

// RuntimeCLI is one install's own cozy-runtime binary, run against a named local root.
// Every question this host asks the runtime goes through here, so there is one place
// that knows how to invoke it and one place that renders its refusals.
type RuntimeCLI struct {
	Bin          string        // package Runtime for metadata; the control Runtime is selected separately
	Dir          string        // the package project root
	Descriptor   string        // exact published descriptor; empty for editable/source installs
	Home         string        // COZY_HOME the runtime reads its artifact index out of
	Env          []string      // the allowlisted child environment (config.Tool)
	QueryTimeout time.Duration // metadata-query bound; zero selects DefaultRuntimeQueryTimeout
}

// Binary is the runtime an install carries. The install transaction already refused a
// venv that provides none (cl-009's `runtime_missing`), so this is the same claim,
// re-asserted where it is used.
func Binary(inst records.PackageInstall) string {
	if inst.Runtime != "" {
		return inst.Runtime
	}
	return home.VenvTool(filepath.Join(inst.Dir, "venv"), "cozy-runtime")
}

// HostRuntime is the package-independent local worker control process. Package code runs
// through the separately selected venv interpreter.
func HostRuntime() (string, *exit.Error) {
	path, err := exec.LookPath("cozy-runtime")
	if err != nil {
		return "", exit.Named(exit.Structural, "host_runtime_missing",
			"this host has no cozy-runtime command on PATH").
			WithRemedy("install the Cozy Runtime tool that ships with this Cozy release")
	}
	return path, nil
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
	args := []string{"--json", "--dir", r.Dir}
	if r.Descriptor != "" {
		args = append(args, "--descriptor", r.Descriptor)
	}
	args = append(args, verb...)
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
			WithRemedy("the release's runtime must answer bindings and job-describe metadata without importing or constructing the package")
	}
	if cmd.ProcessState == nil {
		return exit.Named(exit.Structural, "runtime_missing",
			"cannot run %s: %s", r.Bin, err).
			WithRemedy("a package's surface is answered by the runtime the release itself pinned")
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
	if json.Unmarshal([]byte(stderr), &doc) == nil && doc.Error.Message != "" {
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
