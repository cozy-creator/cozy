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
	"time"
)

// A plain script produces a retained native model, then calls the original GPU
// serving function twice, observing capture without memoizing inference.
func TestOrdinaryScriptNativeModelServing(t *testing.T) { ordinaryScriptModelServing(t, false, false) }

func TestOrdinaryScriptMixedModelServing(t *testing.T) { ordinaryScriptModelServing(t, true, false) }

func TestEditedScriptsReuseImmutableDependencies(t *testing.T) {
	ordinaryScriptModelServing(t, false, true)
}

func ordinaryScriptModelServing(t *testing.T, mixed, dependencyReuse bool) {
	if *privateChildRuntimeWheel == "" {
		t.Skip("requires the exact wire44 Runtime candidate wheel")
	}
	wheel, err := filepath.Abs(*privateChildRuntimeWheel)
	must(t, err)
	version := strings.Split(filepath.Base(wheel), "-")[1]
	control := filepath.Join(t.TempDir(), "control")
	// Give the base the numerical dependencies present in real worker images.
	for _, args := range [][]string{{"venv", control, "--python", "3.12"},
		{"pip", "install", "--python", filepath.Join(control, "bin", "python"), wheel, "torch>=2.13,<3", "numpy>=1.26"}} {
		out, err := exec.Command("uv", args...).CombinedOutput()
		if err != nil {
			t.Fatalf("candidate environment: %v\n%s", err, out)
		}
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
			_ = os.RemoveAll(root)
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
dependencies=["cozy-runtime==%s", "torch>=2.13,<3", "numpy>=1.26"]
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
	if dependencyReuse {
		metadata = strings.Replace(metadata, `, "torch>=2.13,<3"`, "", 1)
		metadata += "\n[project.optional-dependencies]\ncu130=[\"torch>=2.13,<3\"]\n"
	}
	must(t, os.WriteFile(filepath.Join(tools, "pyproject.toml"), []byte(metadata), 0600))
	must(t, os.WriteFile(filepath.Join(tools, "package.toml"), []byte("[application]\nobject=\"model_tools:app\"\n"), 0600))
	body, err := os.ReadFile(filepath.Join("testdata", "managed_serving", "prepare.py"))
	must(t, err)
	body = []byte(strings.NewReplacer("__VERSION__", version, "__WHEEL__", strconv.Quote(wheel)).Replace(string(body)))
	if dependencyReuse {
		body = []byte(strings.ReplaceAll(string(body), "model-tools==0.0.1", "model-tools[cu130]==0.0.1"))
	}
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
	db, err := sql.Open("sqlite", filepath.Join(root, "creator.sqlite"))
	must(t, err)
	defer db.Close()
	var parent, child string
	must(t, db.QueryRow(`SELECT id FROM requests WHERE parent_request_id='' AND state<>'succeeded'`).Scan(&parent))
	for {
		var state string
		must(t, db.QueryRow(`SELECT state FROM requests WHERE id=?`, parent).Scan(&state))
		if state == "blocked" || state == "canceled" || state == "succeeded" {
			t.Fatalf("cancellable script settled before its serving call: %s", state)
		}
		err := db.QueryRow(`SELECT r.id FROM requests r JOIN attempts a ON a.request_id=r.id AND a.attempt=r.ordinal WHERE r.parent_request_id=? AND r.entrypoint='generate' AND json_extract(r.payload,'$.wait_for_cancel')=1 AND a.state='accepted'`, parent).Scan(&child)
		if err == nil {
			break
		}
		if err != sql.ErrNoRows {
			t.Fatal(err)
		}
		time.Sleep(100 * time.Millisecond)
	}
	var produced, reused, verified int
	must(t, db.QueryRow("SELECT count(*),sum(CASE WHEN reused_from<>'' THEN 1 ELSE 0 END) FROM requests WHERE parent_request_id<>'' AND entrypoint='produce' AND state='succeeded'").Scan(&produced, &reused))
	must(t, db.QueryRow("SELECT count(*) FROM requests WHERE parent_request_id<>'' AND entrypoint='generate' AND state='succeeded' AND reused_from=''").Scan(&verified))
	if produced != 2 || reused != 1 || verified != 4 {
		t.Fatalf("independent scripts did not reuse production and execute both recipient reads: produce=%d reused=%d verify=%d", produced, reused, verified)
	}
	installs, err := db.Query(`SELECT dir FROM installs WHERE package='local/model-tools'`)
	must(t, err)
	var staged []string
	var modelInstalls []string
	for installs.Next() {
		var install string
		must(t, installs.Scan(&install))
		modelInstalls = append(modelInstalls, install)
		paths, err := filepath.Glob(filepath.Join(install, "worker-environments", ".stage", "*", "wheels"))
		must(t, err)
		staged = append(staged, paths...)
	}
	must(t, installs.Err())
	must(t, installs.Close())
	if len(staged) != 2 {
		t.Fatalf("unchanged code was prepared more than once per worker: stages=%v", staged)
	}
	rows, err := db.Query(`SELECT count(DISTINCT p.body) FROM attempt_serving_placements p JOIN requests r ON r.id=p.request_id GROUP BY r.parent_request_id`)
	must(t, err)
	defer rows.Close()
	for rows.Next() {
		var placements int
		must(t, rows.Scan(&placements))
		if placements != 1 {
			t.Fatal("repeated inference rebuilt its unchanged prepared placement")
		}
	}
	must(t, rows.Err())
	var inputs int
	must(t, db.QueryRow(`SELECT count(DISTINCT h.request_id) FROM request_weights_retentions h JOIN requests r ON r.id=h.request_id WHERE r.entrypoint='generate' AND h.kind='input' AND h.slot='result/model'`).Scan(&inputs))
	if inputs != 5 {
		t.Fatal("prepared code reuse bypassed per-call model custody")
	}

	if code, out := runCozyPath(t, root, path, "run", "cancel", parent, "--json"); code != 0 {
		t.Fatalf("cancel accepted serving script [%d]: %s", code, out)
	}
	waitUntil(t, "accepted serving cancellation releases recipient custody", func() bool {
		var remaining int
		must(t, db.QueryRow(`SELECT (SELECT count(*) FROM requests WHERE id IN (?,?) AND state<>'canceled') + (SELECT count(*) FROM request_weights_retentions WHERE request_id=? AND state<>'released')`, parent, child, child).Scan(&remaining))
		return remaining == 0
	})
	var memoized int
	must(t, db.QueryRow(`SELECT count(*) FROM requests WHERE parent_request_id=? AND entrypoint='produce' AND reused_from<>'' AND ordinal=0`, parent).Scan(&memoized))
	if memoized != 1 {
		t.Fatal("canceled serving script repeated its completed model production")
	}
	if dependencyReuse {
		proveSharedDependencies(t, root, path, control, modelInstalls)
	}
}

func proveSharedDependencies(t *testing.T, root, path, control string, installs []string) {
	t.Helper()
	var generations []string
	for _, install := range installs {
		found, err := filepath.Glob(filepath.Join(install, "worker-environments", "contents", "*"))
		must(t, err)
		generations = append(generations, found...)
	}
	if len(generations) != 2 {
		t.Fatalf("edited script proof needs two retained generations: %v", generations)
	}
	for _, library := range []string{"libtorch_cpu.so", "libtorch_cuda.so"} {
		relative := filepath.Join("lib", "python3.12", "site-packages", "torch", "lib", library)
		first, err := os.Stat(filepath.Join(generations[0], relative))
		must(t, err)
		second, err := os.Stat(filepath.Join(generations[1], relative))
		must(t, err)
		sdk, err := os.Stat(filepath.Join(control, relative))
		must(t, err)
		if !os.SameFile(first, second) || os.SameFile(first, sdk) || first.Mode().Perm()&0222 != 0 {
			t.Fatalf("%s was copied again, shares mutable SDK storage, or is writable", library)
		}
		t.Logf("%s reused one immutable %d-byte inode across edited captures", library, first.Size())
	}
	// Stop fixture processes before modifying/removing their source environments.
	compositionDown(t, root, path)
	for _, install := range installs {
		must(t, os.RemoveAll(filepath.Join(install, "venv")))
	}
	must(t, os.RemoveAll(filepath.Join(root, "local-packages", "dependency-objects")))
	for _, generation := range generations {
		out, err := exec.Command(filepath.Join(generation, "bin", "python"), "-I", "-c",
			"import model_tools, torch; x=torch.tensor([2.,3.],device='cuda'); assert x.sum().item()==5").CombinedOutput()
		if err != nil {
			t.Fatalf("retained generation lost dependencies after source/cache removal: %v\n%s", err, out)
		}
	}
}
