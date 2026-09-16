package producttest

import (
	"bytes"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/cozy-creator/cozy/internal/home"
	"github.com/cozy-creator/cozy/internal/localpackage"
	"github.com/cozy-creator/cozy/internal/records"
)

// This CPU proof drives the built CLI and real Runtime. It verifies that a
// standard optional implementation wheel reaches the executing package and the
// rented-worker capture unchanged. It does not qualify a GPU attention kernel.
func TestAttentionWheelCLIExecutesCapturedABA(t *testing.T) {
	root, err := os.MkdirTemp("", "cozy-attn-")
	must(t, err)
	t.Cleanup(func() {
		reapDaemonRoot(root)
		must(t, removeAllForce(root))
	})
	project := t.TempDir()
	must(t, os.Mkdir(filepath.Join(project, "vendor"), 0700))
	version := runtimeFixtureVersion(t, *privateChildRuntimeWheel)
	runtimeSource := ""
	if *privateChildRuntimeWheel != "" {
		path, err := filepath.Abs(*privateChildRuntimeWheel)
		must(t, err)
		runtimeSource = fmt.Sprintf("cozy-runtime = {path = %q}\n", path)
	}
	must(t, os.WriteFile(filepath.Join(project, "pyproject.toml"), []byte(fmt.Sprintf(`[project]
name = "cozy-weightless-package"
version = "1.0.0"
requires-python = ">=3.12,<3.13"
dependencies = ["cozy-runtime>=%s", "msgspec>=0.21"]
[project.entry-points."cozy.application"]
default = "weightless:app"
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
[tool.hatch.build.targets.wheel]
only-include = ["weightless.py"]
[tool.uv.sources]
%s`, version, runtimeSource)), 0600))
	must(t, os.WriteFile(filepath.Join(project, "package.toml"), []byte("[application]\nobject = 'weightless:app'\n"), 0600))
	must(t, os.WriteFile(filepath.Join(project, "weightless.py"), []byte(`import msgspec
from cozy_runtime.author import App
app = App()
class Request(msgspec.Struct):
    why: str
class EchoOutput(msgspec.Struct):
    text: str
@app.entrypoint
def echo(payload: Request) -> EchoOutput:
    return EchoOutput(payload.why)
`), 0600))
	library := t.TempDir()
	must(t, os.WriteFile(filepath.Join(library, "pyproject.toml"), []byte(`[project]
name = "attention-wheel-proof"
version = "0.1.0"
requires-python = ">=3.12,<3.13"
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
[tool.hatch.build.targets.wheel]
only-include = ["attention_wheel_proof.py"]
`), 0600))
	original, err := os.ReadFile(filepath.Join(project, "weightless.py"))
	must(t, err)
	body := strings.Replace(string(original), "return EchoOutput(payload.why)", `from attention_wheel_proof import candidate
    from importlib.metadata import version
    return EchoOutput(payload.why + "|" + candidate + "|" + version("attention-wheel-proof"))`, 1)
	must(t, os.WriteFile(filepath.Join(project, "weightless.py"), []byte(body), 0600))
	wheelName := "attention_wheel_proof-0.1.0-py3-none-any.whl"
	wheelPath := filepath.Join(project, "vendor", wheelName)
	metadataPath := filepath.Join(project, "pyproject.toml")
	metadata, err := os.ReadFile(metadataPath)
	must(t, err)
	text := strings.Replace(string(metadata), "dependencies = [", `dependencies = ["attention-wheel-proof>=0.1.0", `, 1)
	text += "attention-wheel-proof = {path = \"vendor/" + wheelName + "\"}\n"
	must(t, os.WriteFile(metadataPath, []byte(text), 0600))
	runUV := func(args ...string) {
		t.Helper()
		cmd := exec.Command("uv", args...)
		cmd.Env = childEnv(t, root)
		if out, err := cmd.CombinedOutput(); err != nil {
			t.Fatalf("uv: %v\n%s", err, out)
		}
	}
	var originalWheel []byte
	var revisions []localpackage.Revision
	for i, candidate := range []string{"A", "B", "A"} {
		if i == 2 {
			must(t, os.WriteFile(wheelPath, originalWheel, 0600))
		} else {
			must(t, os.WriteFile(filepath.Join(library, "attention_wheel_proof.py"), []byte(fmt.Sprintf("candidate = %q\n", candidate)), 0600))
			runUV("build", "--wheel", "--out-dir", filepath.Dir(wheelPath), library)
			// A valid large data member reproduces prebuilt native wheels above
			// the ordinary 64 MiB source-file bound without needing a GPU build.
			pad := exec.Command("python3", "-c", `import base64, csv, hashlib, io, pathlib, sys, zipfile
path = pathlib.Path(sys.argv[1])
with zipfile.ZipFile(path) as source:
    members = {name: source.read(name) for name in source.namelist() if not name.endswith("/RECORD")}
members["attention_wheel_payload.dat"] = bytes(65 << 20)
record = "attention_wheel_proof-0.1.0.dist-info/RECORD"
rows = io.StringIO()
writer = csv.writer(rows, lineterminator="\n")
with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as target:
    for name, data in members.items():
        target.writestr(name, data)
        digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()
        writer.writerow([name, "sha256=" + digest, len(data)])
    writer.writerow([record, "", ""])
    target.writestr(record, rows.getvalue())
`, wheelPath)
			if out, err := pad.CombinedOutput(); err != nil {
				t.Fatalf("large dependency wheel: %v\n%s", err, out)
			}
		}
		wheelBytes, err := os.ReadFile(wheelPath)
		must(t, err)
		if i == 0 {
			originalWheel = wheelBytes
		}
		runUV("lock", "--project", project, "--refresh-package", "attention-wheel-proof")
		for {
			code, out := runCozy(t, root, "package", "install", project, "--editable", "--no-model-download")
			if code != 0 && strings.Contains(out, "another Cozy writer holds") {
				// The editable watcher may be capturing these same edited bytes.
				// Retry explicit writer contention, never installation failures.
				time.Sleep(50 * time.Millisecond)
				continue
			}
			if code != 0 {
				t.Fatalf("install %s [%d]: %s", candidate, code, out)
			}
			break
		}
		key := fmt.Sprintf("attention-wheel-%d", i)
		code, out := runCozy(t, root, "run", localWeightlessRef+"/echo", "why=wheel-proof", "--idempotency-key="+key, "--await", "--json")
		if code != 0 || !strings.Contains(out, "wheel-proof|"+candidate+"|0.1.0") {
			t.Fatalf("executed wrong wheel %s [%d]: %s\n%s", candidate, code, out, productWorkerLogs(root))
		}
		// A CPU scalar has no attention site. An explicit pin must reach
		// Runtime and refuse there rather than silently succeeding unused.
		code, out = runCozy(t, root, "run", localWeightlessRef+"/echo", "why=wheel-proof",
			"kernel.attention=sdpa", "--idempotency-key="+key+"-pin", "--await", "--json")
		if code == 0 || !strings.Contains(out, "attention_kernel_unsupported") {
			t.Fatalf("unused attention pin did not reach Runtime refusal [%d]: %s", code, out)
		}
		store, problem := records.Open(filepath.Join(root, "creator.sqlite"))
		fatal(t, problem)
		request, problem := store.RequestByIdempotencyKey(key + "-pin")
		fatal(t, problem)
		store.Close()
		if request == nil || request.AttentionKernel != "sdpa" {
			t.Fatalf("executing request lost attention pin: %+v", request)
		}
		install := activeInstall(t, root, localWeightlessRef)
		revision, problem := localpackage.Stage(t.Context(), home.Layout{LocalPackages: filepath.Join(root, "local-packages")}, install)
		fatal(t, problem)
		found := false
		for _, file := range revision.Files {
			if file.Filename != wheelName {
				continue
			}
			captured, err := os.ReadFile(file.Path)
			must(t, err)
			if !bytes.Equal(captured, wheelBytes) {
				t.Fatal("rented-worker capture changed the executing candidate wheel")
			}
			found = true
		}
		if !found {
			t.Fatal("rented-worker capture omitted the candidate wheel")
		}
		revisions = append(revisions, revision)
		t.Logf("candidate=%s request=%s pin=%s revision=%s", candidate, request.ID, request.AttentionKernel, revision.Digest)
	}
	if revisions[0].Digest == revisions[1].Digest || revisions[0].Digest != revisions[2].Digest {
		t.Fatalf("same-version A/B/A bytes lost their capture identity: %s %s %s", revisions[0].Digest, revisions[1].Digest, revisions[2].Digest)
	}
}
