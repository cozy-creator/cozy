package launch

import (
	"context"
	"encoding/json"
	"fmt"
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

// NO CLOCK BOUNDS A RUNTIME QUESTION (xs-007 row 10). `describe` and `bindings` used to run
// under a 5-second deadline whose expiry raised `runtime_query_stalled` and refused the job
// dispatch or the install outright. Five seconds is not a fact about a metadata verb: it is a
// guess about interpreter cold start, and a Python interpreter importing its entry module on
// a loaded host exceeds it routinely — so a healthy release was refused for being slow to
// start. The child's own exit is the answer, and it always was: a runtime that dies says so
// through its exit code, and one still working is still working.

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
	Bin              string   // package Runtime for metadata; the control Runtime is selected separately
	Dir              string   // the package project root
	PackageInterface string   // exact published package interface; empty for editable/source installs
	Home             string   // COZY_HOME the runtime reads its artifact index out of
	Env              []string // the allowlisted child environment (config.Tool)
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

// json runs one verb and decodes its `--json` document.
func (r RuntimeCLI) call(out any, verb ...string) *exit.Error {
	return r.callContext(context.Background(), out, verb...)
}

func (r RuntimeCLI) callContext(ctx context.Context, out any, verb ...string) *exit.Error {
	args := []string{"--json", "--dir", r.Dir}
	if r.PackageInterface != "" {
		args = append(args, "--package-interface", r.PackageInterface)
	}
	args = append(args, verb...)
	cmd := exec.CommandContext(ctx, r.Bin, args...)
	if ctx.Done() != nil {
		// CommandContext kills the direct child when the CALLER's context ends. WaitDelay
		// also closes a pipe a misbehaving descendant retained, so the cancellation the
		// caller asked for actually completes. Nothing here starts a clock of its own.
		cmd.WaitDelay = 250 * time.Millisecond
	}
	cmd.Env = append(append([]string{}, r.Env...), "COZY_HOME="+r.Home)
	var stdout, stderr strings.Builder
	cmd.Stdout, cmd.Stderr = &stdout, &stderr
	err := cmd.Run()
	if ctx.Err() != nil {
		return exit.New(exit.Canceled, "`cozy-runtime %s` was stopped: %s",
			strings.Join(verb, " "), ctx.Err())
	}
	if cmd.ProcessState == nil {
		return exit.Named(exit.Structural, "runtime_missing",
			"cannot run %s: %s", r.Bin, err).
			WithRemedy("a package's surface is answered by the runtime the release itself pinned")
	}
	if code := cmd.ProcessState.ExitCode(); code != 0 {
		return RuntimeExit(code, strings.Join(verb, " "), "runtime_query_failed",
			stdout.String(), stderr.String())
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

// RuntimeExit is the ONE reading of a cozy-runtime child that exited non-zero. The exit
// matrix is SHARED (cozy-runtime-cli.md), so a runtime exit is already a cozy exit and the
// `--json` refusal it wrote is already the answer: code, name, message and remedy pass
// through as themselves. Nothing here re-labels. The runtime is the only layer that
// evaluated the wheels, the GPU, the CAS or the models, so it alone names the verdict —
// preparation, wheel, GPU, CAS and network refusals keep their names, and "does not fit"
// is said by a Fit and nobody else (model-code-fit §3). A child that died without a
// document (a signal, a traceback) is `untyped`, with the tail of what it wrote as the
// detail, because the end of a traceback is where the exception is.
func RuntimeExit(code int, verb, untyped, stdout, stderr string) *exit.Error {
	var doc struct {
		Error struct {
			Name    string `json:"name"`
			Message string `json:"message"`
			Remedy  string `json:"remedy"`
		} `json:"error"`
	}
	c := exit.Code(code)
	if !c.Valid() {
		c = exit.Internal
	}
	if json.Unmarshal([]byte(stderr), &doc) == nil && doc.Error.Message != "" {
		name := doc.Error.Name
		if name == "" {
			name = untyped
		}
		e := exit.Named(c, name, "%s", doc.Error.Message)
		if doc.Error.Remedy != "" {
			e.WithRemedy("%s", doc.Error.Remedy)
		}
		return e
	}
	said := strings.TrimSpace(stderr)
	if said == "" {
		said = strings.TrimSpace(stdout)
	}
	what := fmt.Sprintf("exited %d", code)
	if code < 0 {
		what = "was killed"
	}
	return exit.Named(c, untyped, "`cozy-runtime %s` %s without a typed refusal: %s",
		verb, what, tail(said))
}

// tail keeps the END of what a child wrote; a traceback names its exception last.
func tail(s string) string {
	s = strings.Join(strings.Fields(s), " ")
	if r := []rune(s); len(r) > 400 {
		return "…" + string(r[len(r)-400:])
	}
	return s
}

func condense(s string) string {
	s = strings.Join(strings.Fields(s), " ")
	if len(s) > 400 {
		return s[:400] + "…"
	}
	return s
}
