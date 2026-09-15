package launch

import (
	"bytes"
	"context"
	"encoding/json"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"time"

	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/home"
	"github.com/cozy-creator/cozy/internal/hostruntime"
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

// RuntimeCLI invokes one cozy-runtime binary against a named local root. Every question this
// host asks the runtime goes through here, so there is one place that knows how to invoke it
// and one place that renders its refusals. A metadata question goes to THIS host's tool
// (hostruntime.Path, at or above hostruntime.Floor): package code is untrusted and a reading of it
// never imports it (cl-175). The install's own Runtime (Binary) is asked only for the
// interface wheel its executor consumes.
type RuntimeCLI struct {
	Bin               string   // the binary selected for the question: hostruntime.Path for metadata, Binary(install) for interface-wheel
	Dir               string   // the package project root
	EnvironmentPython string   // captured environment location for static source reads only
	PackageInterface  string   // exact installed interface; empty only for live editable source
	Home              string   // COZY_HOME the runtime reads its artifact index out of
	Env               []string // the allowlisted child environment (config.Tool)
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
	return r.callInputContext(ctx, nil, out, verb...)
}

func (r RuntimeCLI) callInputContext(ctx context.Context, input []byte, out any, verb ...string) *exit.Error {
	args := []string{"--json", "--dir", r.Dir}
	if r.PackageInterface != "" {
		args = append(args, "--package-interface", r.PackageInterface)
	}
	args = append(args, verb...)
	if len(verb) > 0 && verb[0] == "describe" && r.PackageInterface == "" && r.EnvironmentPython != "" {
		args = append(args, "--environment-python", r.EnvironmentPython)
	}
	cmd := exec.CommandContext(ctx, r.Bin, args...)
	if input != nil {
		cmd.Stdin = bytes.NewReader(input) //cozy:stdin-value exact owner-supplied metadata for a noninteractive Runtime capability
	}
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
		return hostruntime.RuntimeExit(code, strings.Join(verb, " "), "runtime_query_failed",
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

// CapturedBuiltinTool selects only the interpreter whose owned metadata was verified.
func CapturedBuiltinTool(root, python string, env []string) RuntimeCLI {
	return RuntimeCLI{Bin: filepath.Join(filepath.Dir(python), "cozy-runtime"), Dir: root, Home: root, Env: env}
}
