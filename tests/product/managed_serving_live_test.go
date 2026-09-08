package producttest

import (
	"database/sql"
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
func TestOrdinaryScriptNativeModelServing(t *testing.T) {
	if *privateChildRuntimeWheel == "" {
		t.Skip("requires the exact wire44 Runtime candidate wheel")
	}
	wheel, err := filepath.Abs(*privateChildRuntimeWheel)
	must(t, err)
	version := strings.Split(filepath.Base(wheel), "-")[1]
	control := filepath.Join(t.TempDir(), "control")
	for _, args := range [][]string{{"venv", control, "--python", "3.12"},
		{"pip", "install", "--python", filepath.Join(control, "bin", "python"), wheel}} {
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
	must(t, os.WriteFile(filepath.Join(tools, "pyproject.toml"), []byte(metadata), 0600))
	must(t, os.WriteFile(filepath.Join(tools, "package.toml"), []byte("[application]\nobject=\"model_tools:app\"\n"), 0600))
	body, err := os.ReadFile(filepath.Join("testdata", "managed_serving", "prepare.py"))
	must(t, err)
	body = []byte(strings.NewReplacer("__VERSION__", version, "__WHEEL__", strconv.Quote(wheel)).Replace(string(body)))
	script := filepath.Join(project, "prepare.py")
	must(t, os.WriteFile(script, body, 0600))
	for run := range 2 {
		if run == 1 {
			must(t, os.WriteFile(script, append(body, []byte("\n# Independent edited script; reuse compatible operations.\n")...), 0600))
		}
		code, out := runCozyPath(t, root, path, "run", script, "--await", "--json")
		if code != 0 {
			t.Fatalf("native serving script %d [exit %d]\n%s\n%s", run, code, out, productWorkerLogs(root))
		}
	}
	db, err := sql.Open("sqlite", filepath.Join(root, "creator.sqlite"))
	must(t, err)
	defer db.Close()
	var produced, reused, verified int
	must(t, db.QueryRow("SELECT count(*),sum(CASE WHEN reused_from<>'' THEN 1 ELSE 0 END) FROM requests WHERE parent_request_id<>'' AND entrypoint='produce' AND state='succeeded'").Scan(&produced, &reused))
	must(t, db.QueryRow("SELECT count(*) FROM requests WHERE parent_request_id<>'' AND entrypoint='generate' AND state='succeeded' AND reused_from=''").Scan(&verified))
	if produced != 2 || reused != 1 || verified != 4 {
		t.Fatalf("independent scripts did not reuse production and execute both recipient reads: produce=%d reused=%d verify=%d", produced, reused, verified)
	}
	installs, err := db.Query(`SELECT dir FROM installs WHERE package='local/model-tools'`)
	must(t, err)
	var staged []string
	for installs.Next() {
		var install string
		must(t, installs.Scan(&install))
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
	if inputs != 4 {
		t.Fatal("prepared code reuse bypassed per-call model custody")
	}

	// Cancel an accepted serving call through the same CLI. Its already completed
	// native producer remains reusable, while this recipient's input hold closes.
	waiting := string(body[:strings.Index(string(body), "async def main(ctx):")]) + `async def main(ctx):
    model = await produce()
    await generate(model=model, wait_for_cancel=True)
`
	must(t, os.WriteFile(script, []byte(waiting), 0600))
	if code, out := runCozyPath(t, root, path, "run", script, "--json"); code != 0 {
		t.Fatalf("submit cancellable serving script [%d]: %s", code, out)
	}
	var parent, child string
	must(t, db.QueryRow(`SELECT id FROM requests WHERE parent_request_id='' AND state<>'succeeded'`).Scan(&parent))
	for {
		var state string
		must(t, db.QueryRow(`SELECT state FROM requests WHERE id=?`, parent).Scan(&state))
		if state == "blocked" || state == "canceled" || state == "succeeded" {
			t.Fatalf("cancellable script settled before its serving call: %s", state)
		}
		err := db.QueryRow(`SELECT r.id FROM requests r JOIN attempts a ON a.request_id=r.id AND a.ordinal=r.ordinal WHERE r.parent_request_id=? AND r.entrypoint='generate' AND a.state='accepted'`, parent).Scan(&child)
		if err == nil {
			break
		}
		if err != sql.ErrNoRows {
			t.Fatal(err)
		}
		time.Sleep(100 * time.Millisecond)
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
}
