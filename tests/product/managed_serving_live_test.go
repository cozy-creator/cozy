package producttest

import (
	"database/sql"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"testing"

	"github.com/cozy-creator/cozy/internal/records"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
	"google.golang.org/protobuf/proto"
)

// A plain script produces a retained native model, then calls the original GPU
// serving function twice, observing capture without memoizing inference.
func TestOrdinaryScriptNativeModelServing(t *testing.T) { ordinaryScriptModelServing(t, false) }

func TestOrdinaryScriptMixedModelServing(t *testing.T) { ordinaryScriptModelServing(t, true) }

func ordinaryScriptModelServing(t *testing.T, mixed bool) {
	if *privateChildRuntimeWheel == "" {
		t.Skip("requires a Runtime candidate wheel and CUDA for actual serving")
	}
	wheel, err := filepath.Abs(*privateChildRuntimeWheel)
	must(t, err)
	version := strings.Split(filepath.Base(wheel), "-")[1]
	control := filepath.Join(t.TempDir(), "control")
	// Give the base the numerical dependencies present in real worker images.
	for _, args := range [][]string{{"venv", control, "--python", "3.12"},
		{"pip", "install", "--python", filepath.Join(control, "bin", "python"), wheel, "torch==2.13.0", "numpy>=1.26"}} {
		out, err := exec.Command("uv", args...).CombinedOutput()
		if err != nil {
			t.Fatalf("candidate environment: %v\n%s", err, out)
		}
	}
	if probe, err := exec.Command(filepath.Join(control, "bin", "python"), "-c", "import torch; print(torch.__version__); raise SystemExit(0 if torch.cuda.is_available() else 77)").CombinedOutput(); err != nil {
		if status, ok := err.(*exec.ExitError); ok && status.ExitCode() == 77 {
			t.Skipf("actual model serving needs CUDA; this peer is %s", probe)
		}
		t.Fatalf("CUDA peer probe: %v %s", err, probe)
	}
	root, err := os.MkdirTemp("", "cozy-native-serving-")
	must(t, err)
	path := filepath.Join(control, "bin")
	for _, item := range childEnv(t, root) {
		if strings.HasPrefix(item, "PATH=") {
			path += string(os.PathListSeparator) + strings.TrimPrefix(item, "PATH=")
		}
	}
	t.Cleanup(func() {
		compositionDown(t, root, path)
		if t.Failed() {
			t.Log("native serving evidence retained", root)
		} else {
			must(t, removeAllForce(root))
		}
	})
	defer tracePrivateChildWait(t, root)()
	project := t.TempDir()
	tools := filepath.Join(project, "model_tools")
	must(t, os.MkdirAll(tools, 0700))
	module, err := os.ReadFile(filepath.Join("testdata", "managed_serving", "model_tools.py"))
	must(t, err)
	if mixed {
		seedSource, err := os.ReadFile(filepath.Join("testdata", "local_serving_preparation", "seed.py"))
		must(t, err)
		seedSource = []byte(strings.ReplaceAll(string(seedSource), "(\"alpha\", 2.0)", "(\"alpha\", 3.0)"))
		seedFile := filepath.Join(project, "seed-base.py")
		must(t, os.WriteFile(seedFile, seedSource, 0600))
		seedBytes, err := exec.Command(filepath.Join(control, "bin", "python"), seedFile, filepath.Join(project, "catalog-store")).CombinedOutput()
		if err != nil {
			t.Fatalf("seed published base: %v\n%s", err, seedBytes)
		}
		var seed servingSeed
		must(t, json.Unmarshal(seedBytes, &seed))
		catalog := servingModelCatalog(t, seed)
		defer catalog.Close()
		must(t, os.WriteFile(filepath.Join(root, "config.yaml"), []byte("tensorhub_url: "+catalog.URL+"\ntensorhub_token: local-serving-fixture\ndaemon:\n  idle_shutdown_s: 0\n"), 0600))
		module = []byte(strings.NewReplacer(
			"import hashlib\n", "import hashlib\nimport math\n",
			"@app.entrypoint\ndef generate", "@app.entrypoint(defaults={\"adapter\":[{\"gpu\":\"*\",\"lane\":\"proof/ordered@1.0.0/bf16\"}]})\ndef generate",
			"model: OrderedModel, tel: Telemetry)", "model: OrderedModel, tel: Telemetry, adapter: OrderedModel)",
			"return model.measure(payload.seed, payload.steps, tel)", "primary = model.measure(payload.seed, payload.steps, tel)\n    base = adapter.measure(payload.seed, payload.steps, tel)\n    assert math.isclose(base.value, primary.value * (3.0 / 2.0) ** payload.steps, rel_tol=1e-6, abs_tol=1e-6)\n    assert primary.value != base.value\n    return msgspec.structs.replace(primary, value=primary.value + base.value)",
		).Replace(string(module)))
	}
	must(t, os.WriteFile(filepath.Join(tools, "model_tools.py"), module, 0600))
	metadata := fmt.Sprintf(`[project]
name="model-tools"
version="0.0.1"
requires-python=">=3.12,<3.13"
dependencies=["cozy-runtime==%s", "torch==2.13.0", "numpy>=1.26"]
[project.entry-points."cozy.application"]
default="model_tools:app"
[tool.uv.sources]
cozy-runtime={path=%s}
[build-system]
requires=["hatchling"]
build-backend="hatchling.build"
[tool.hatch.build.targets.wheel]
only-include=["model_tools.py"]
`, version, strconv.Quote(wheel))
	must(t, os.WriteFile(filepath.Join(tools, "pyproject.toml"), []byte(metadata), 0600))
	must(t, os.WriteFile(filepath.Join(tools, "package.toml"), []byte("[application]\nobject=\"model_tools:app\"\n"), 0600))
	body, err := os.ReadFile(filepath.Join("testdata", "managed_serving", "prepare.py"))
	must(t, err)
	body = []byte(strings.NewReplacer("__VERSION__", version, "__WHEEL__", strconv.Quote(wheel)).Replace(string(body)))
	if mixed {
		body = []byte(strings.ReplaceAll(string(body), "first = generate(seed=helper(), steps=2, model=model,\n        capture=ActivationCapture(components=(\"alpha\", \"zeta\"), steps=(0, 1)))", "first = generate(seed=helper(), steps=2, model=model)"))
		lines := []string{}
		for _, line := range strings.Split(string(body), "\n") {
			if !strings.Contains(line, "assert first.observation") {
				lines = append(lines, line)
			}
		}
		body = []byte(strings.Join(lines, "\n"))
	}

	script := filepath.Join(project, "prepare.py")
	must(t, os.WriteFile(script, body, 0600))
	for run := range 2 {
		args := []string{"run", script, "--json"}
		if run == 0 {
			args = append(args, "--await")
		} else {
			// The second edited script also supplies the cancellable call. This
			// keeps the same two independent producer scopes without capturing a third.
			waiting := append(append([]byte{}, body...), []byte("\n    await generate(model=model, wait_for_cancel=True)\n# Independent edited script; reuse compatible operations.\n")...)
			must(t, os.WriteFile(script, waiting, 0600))
		}
		code, out := runCozyPath(t, root, path, args...)
		if code != 0 {
			t.Fatalf("native serving script %d [exit %d]\n%s\n%s", run, code, out, productWorkerLogs(root))
		}
	}
	store, problem := records.Open(filepath.Join(root, "creator.sqlite"))
	fatal(t, problem)
	defer store.Close()
	parentRow, problem := store.RequestByReference("2")
	fatal(t, problem)
	if parentRow == nil {
		t.Fatal("edited script has no request")
	}
	parent := parentRow.ID
	db, err := sql.Open("sqlite", "file:"+filepath.Join(root, "tensorfs", ".cozy-workspace", "journal.sqlite3")+"?mode=ro")
	must(t, err)
	defer db.Close()
	var child string
	waitUntil(t, "Runtime accepted cancellable serving call", func() bool {
		var state string
		must(t, db.QueryRow(`SELECT state FROM executions WHERE request=?`, parent).Scan(&state))
		if state == "failed" || state == "canceled" || state == "succeeded" {
			t.Fatalf("script settled before cancellation: %s", state)
		}
		err := db.QueryRow(`SELECT c.child_request FROM execution_calls c JOIN executions e ON e.owner=c.owner AND e.request=c.child_request WHERE c.parent_request=? AND json_extract(CAST(c.intent AS TEXT),'$.export')='generate' AND json_extract(CAST(c.intent AS TEXT),'$.request.payload.wait_for_cancel')=1 AND e.state='running'`, parent).Scan(&child)
		if err != nil && err != sql.ErrNoRows {
			t.Fatal(err)
		}
		return err == nil
	})
	first, second := machineChildren(t, root, store, "1"), machineChildren(t, root, store, "2")
	if len(first) != 3 || len(second) != 4 {
		t.Fatalf("unexpected Runtime calls: first=%+v second=%+v", first, second)
	}
	if first[0].Executions != 1 || second[0].Executions != 0 || first[0].Computation == "" || first[0].Computation != second[0].Computation {
		t.Fatal("edited caller invalidated completed deterministic model production")
	}
	for _, calls := range [][]machineChildProof{first, second} {
		for _, call := range calls[1:3] {
			if call.State != "succeeded" || call.Executions != 1 {
				t.Fatalf("inference was skipped or failed: %+v", call)
			}
		}
	}
	var inputs int
	must(t, db.QueryRow(`SELECT count(*) FROM execution_model_holds h JOIN execution_calls c ON c.owner=h.owner AND c.child_request=h.recipient WHERE json_extract(CAST(c.intent AS TEXT),'$.export')='generate' AND h.path='input/model'`).Scan(&inputs))
	if inputs != 5 {
		t.Fatalf("expected independent per-call native custody, found %d", inputs)
	}
	var retained []byte
	must(t, db.QueryRow(`SELECT retention FROM execution_model_holds WHERE recipient=? AND path='input/model'`, child).Scan(&retained))
	var hold pb.DerivedRetentionRequest
	must(t, proto.Unmarshal(retained, &hold))
	if hold.RetentionId == "" {
		t.Fatal("serving input has no native retention identity")
	}
	var held string
	must(t, db.QueryRow(`SELECT state FROM holds WHERE id=?`, hold.RetentionId).Scan(&held))
	if held != "held" {
		t.Fatalf("running serving input has custody state %s", held)
	}
	if code, out := runCozyPath(t, root, path, "run", "cancel", parent, "--json"); code != 0 {
		t.Fatalf("cancel serving script [%d]: %s", code, out)
	}
	waitUntil(t, "Runtime serving cancellation releases recipient custody", func() bool {
		var remaining int
		must(t, db.QueryRow(`SELECT (SELECT count(*) FROM executions WHERE request IN (?,?) AND state<>'canceled') + (SELECT count(*) FROM holds WHERE id=? AND state<>'released')`, parent, child, hold.RetentionId).Scan(&remaining))
		return remaining == 0
	})
}
