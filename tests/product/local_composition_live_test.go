package producttest

import (
	"bytes"
	"database/sql"
	"encoding/json"
	"errors"
	"fmt"
	"io/fs"
	"net/http"
	"net/http/httptest"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/records"
)

// Ordinary single-file intake, real native provider/conversion, independent
// quantization/scoring executors and workspace reuse. The supplied PNGs qualify
// scoring composition only; they were not produced by the quantized model.
func TestOrdinaryScriptNativePreparationQuantizationAndScore(t *testing.T) {
	if *privateChildRuntimeWheel == "" || *privateChildEvalWheel == "" {
		t.Skip("requires the exact integrated Runtime and Eval candidate wheels")
	}
	runtimeWheel, err := filepath.Abs(*privateChildRuntimeWheel)
	must(t, err)
	evalWheel, err := filepath.Abs(*privateChildEvalWheel)
	must(t, err)
	runtimeVersion := strings.Split(filepath.Base(runtimeWheel), "-")[1]
	evalVersion := strings.Split(filepath.Base(evalWheel), "-")[1]
	fixture := filepath.Join("testdata", "local_composition")
	body, err := os.ReadFile(filepath.Join(fixture, "model.safetensors"))
	must(t, err)
	digest, _ := canonical.Spell(canonical.Digest(body))
	var bodyRequests atomic.Int64
	provider := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if strings.HasPrefix(r.URL.Path, "/api/models/") {
			_ = json.NewEncoder(w).Encode([]map[string]any{{"type": "file", "path": "provider/model.safetensors", "size": len(body), "lfs": map[string]any{"oid": strings.TrimPrefix(digest, "sha256:"), "size": len(body)}}})
			return
		}
		bodyRequests.Add(1)
		start, end := 0, len(body)-1
		if requested := r.Header.Get("Range"); requested != "" {
			value := strings.TrimPrefix(requested, "bytes=")
			first, last, ok := strings.Cut(value, "-")
			if !ok {
				http.Error(w, "range", 416)
				return
			}
			start, _ = strconv.Atoi(first)
			end, _ = strconv.Atoi(last)
			if start < 0 || end < start || end >= len(body) {
				http.Error(w, "range", 416)
				return
			}
			w.Header().Set("Content-Range", fmt.Sprintf("bytes %d-%d/%d", start, end, len(body)))
			w.Header().Set("Content-Length", strconv.Itoa(end-start+1))
			w.WriteHeader(206)
		} else {
			w.Header().Set("Content-Length", strconv.Itoa(len(body)))
		}
		_, _ = w.Write(body[start : end+1])
	}))
	defer provider.Close()
	control := filepath.Join(t.TempDir(), "control")
	uv := func(args ...string) {
		t.Helper()
		out, err := exec.Command("uv", args...).CombinedOutput()
		if err != nil {
			t.Fatalf("uv %v: %v\n%s", args, err, out)
		}
	}
	uv("venv", control, "--python", "3.12")
	uv("pip", "install", "--python", filepath.Join(control, "bin", "python"), runtimeWheel)
	root, err := os.MkdirTemp("", "cozy-composition-")
	must(t, err)
	registry, err := filepath.Abs(filepath.Join(fixture, "registry.json"))
	must(t, err)
	marker := filepath.Join(root, "quantization-checkpoint.json")
	configuration, _ := json.Marshal(map[string]string{"provider": provider.URL, "registry": registry, "interrupt_marker": marker})
	must(t, os.WriteFile(filepath.Join(control, "bin", "source_fixture.json"), configuration, 0600))
	launcher, err := os.ReadFile(filepath.Join(fixture, "runtime_fixture.py"))
	must(t, err)
	must(t, os.WriteFile(filepath.Join(control, "bin", "cozy-runtime"), append([]byte("#!"+filepath.Join(control, "bin", "python")+"\n"), launcher...), 0700)) //cozy:allow product fixture starts the actual Runtime with a local source provider
	path := filepath.Join(control, "bin")
	for _, item := range childEnv(t, root) {
		if strings.HasPrefix(item, "PATH=") {
			path += string(os.PathListSeparator) + strings.TrimPrefix(item, "PATH=")
		}
	}
	t.Cleanup(func() {
		compositionDown(t, root, path)
		if t.Failed() {
			t.Log("composition evidence retained", root)
		} else {
			_ = os.RemoveAll(root)
		}
	})
	defer tracePrivateChildWait(t, root)()
	project := t.TempDir()
	for _, name := range []string{"quantize_tools", "score_tools"} {
		from, to := filepath.Join(fixture, name), filepath.Join(project, name)
		must(t, filepath.WalkDir(from, func(source string, entry fs.DirEntry, err error) error {
			if err != nil {
				return err
			}
			relative, e := filepath.Rel(from, source)
			if e != nil {
				return e
			}
			destination := filepath.Join(to, relative)
			if entry.IsDir() {
				return os.MkdirAll(destination, 0700)
			}
			raw, e := os.ReadFile(source)
			if e != nil {
				return e
			}
			return os.WriteFile(destination, raw, 0600)
		}))
		if name == "quantize_tools" {
			module := filepath.Join(to, "quantize_tools.py")
			code, err := os.ReadFile(module)
			must(t, err)
			code = bytes.Replace(code, []byte(`INTERRUPT_MARKER = ""`), []byte("INTERRUPT_MARKER = "+strconv.Quote(marker)), 1)
			must(t, os.WriteFile(module, code, 0600))
		}
		deps := `"cozy-runtime==` + runtimeVersion + `","numpy>=1.26"`
		sources := `cozy-runtime={path=` + strconv.Quote(runtimeWheel) + `}`
		wheelFiles := `only-include=["quantize_tools.py"]`
		if name == "score_tools" {
			deps += `,"cozy-eval==` + evalVersion + `"`
			sources += "\ncozy-eval={path=" + strconv.Quote(evalWheel) + "}"
			wheelFiles = `packages=["src/score_tools"]`
		}
		metadata := fmt.Sprintf(`[project]
name=%q
version="0.0.1"
requires-python=">=3.12,<3.13"
dependencies=[%s]
[tool.uv.sources]
%s
[project.entry-points."cozy.application"]
default=%q
[build-system]
requires=["hatchling"]
build-backend="hatchling.build"
[tool.hatch.build.targets.wheel]
%s
`, strings.ReplaceAll(name, "_", "-"), deps, sources, name+":app", wheelFiles)
		must(t, os.WriteFile(filepath.Join(to, "pyproject.toml"), []byte(metadata), 0600))
		must(t, os.WriteFile(filepath.Join(to, "package.toml"), []byte("[application]\nobject="+strconv.Quote(name+":app")+"\n"), 0600))
	}
	script := filepath.Join(project, "prepare.py")
	clientScript, err := os.ReadFile(filepath.Join(fixture, "prepare.py"))
	must(t, err)
	code := strings.NewReplacer(
		"__RUNTIME_VERSION__", runtimeVersion,
		"__RUNTIME_WHEEL__", strconv.Quote(runtimeWheel),
		"__REVISION__", strings.Repeat("a", 40),
	).Replace(string(clientScript))
	must(t, os.WriteFile(script, []byte(code), 0600))
	command := exec.Command(cozyBin, "run", script, "--await", "--json")
	command.Env = childEnv(t, root)
	command.Env = append(command.Env, "PATH="+path)
	var firstOutput bytes.Buffer
	command.Stdout = &firstOutput
	command.Stderr = &firstOutput
	must(t, command.Start())
	done := make(chan error, 1)
	go func() { done <- command.Wait() }()
	var stopped struct {
		PID     int        `json:"pid"`
		Request string     `json:"request"`
		Parts   [][]string `json:"parts"`
	}

	for stopped.PID == 0 {
		select {
		case err := <-done:
			t.Fatalf("script ended before real checkpoint: %v\n%s", err, compositionTail(firstOutput.String()))
		case <-time.After(25 * time.Millisecond):
			raw, err := os.ReadFile(marker)
			if err == nil {
				_ = json.Unmarshal(raw, &stopped)
			}
		}
	}

	if len(stopped.Parts) != 2 {
		t.Fatalf("fault boundary is not one completed data/scale group: %+v", stopped.Parts)
	}
	stopped.Request, _, _ = strings.Cut(stopped.Request, "#")
	controlDB, err := sql.Open("sqlite", "file:"+filepath.Join(root, "creator.sqlite")+"?mode=ro")
	must(t, err)
	defer controlDB.Close()
	pause := exec.Command(cozyBin, "run", "pause", stopped.Request, "--json")
	pause.Env = command.Env
	var pauseOutput bytes.Buffer
	pause.Stdout = &pauseOutput
	pause.Stderr = &pauseOutput
	must(t, pause.Start())
	pauseDone := make(chan error, 1)
	go func() { pauseDone <- pause.Wait() }()
	pauseEnded := false
	for {
		var state string
		must(t, controlDB.QueryRow(`SELECT state FROM requests WHERE id=?`, stopped.Request).Scan(&state))
		if state == "pausing" || state == "paused" {
			break
		}
		select {
		case err := <-pauseDone:
			if err != nil {
				t.Fatalf("pause failed before fencing: %v %s", err, pauseOutput.String())
			}
			pauseEnded = true
			pauseDone = nil
		case <-time.After(25 * time.Millisecond):
		}
	}
	process, err := os.FindProcess(stopped.PID)
	must(t, err)
	if err = process.Kill(); err != nil && !errors.Is(err, os.ErrProcessDone) {
		t.Fatal(err)
	}
	if !pauseEnded {
		if err := <-pauseDone; err != nil {
			t.Fatalf("pause failed: %v %s", err, pauseOutput.String())
		}
	}

	if err := <-done; err == nil {
		t.Fatal("interrupted/rejected first script unexpectedly succeeded")
	}
	store, problem := records.Open(filepath.Join(root, "creator.sqlite"))
	fatal(t, problem)
	defer store.Close()
	first, problem := store.RequestByReference("1")
	fatal(t, problem)
	originalSource, problem := store.NativeCall(first.ID, 0)
	fatal(t, problem)
	originalModel, problem := store.NativeCall(first.ID, 1)
	fatal(t, problem)
	if originalSource == nil || originalModel == nil || originalSource.State != "succeeded" || originalModel.State != "succeeded" {
		t.Fatal("expensive source preparation did not survive", originalSource, originalModel)
	}
	if bodyRequests.Load() != 1 {
		t.Fatalf("source payload was fetched %d times", bodyRequests.Load())
	}
	journal, err := sql.Open("sqlite", "file:"+filepath.Join(root, "tensorfs", ".cozy-workspace", "journal.sqlite3")+"?mode=ro")
	must(t, err)
	defer journal.Close()
	journal.SetMaxOpenConns(1)
	_, err = journal.Exec("PRAGMA busy_timeout=5000")
	must(t, err)
	var interruptedTransaction string
	must(t, journal.QueryRow(`SELECT id FROM weights WHERE request=? AND length(checkpoint)>0 ORDER BY ordinal LIMIT 1`, stopped.Request).Scan(&interruptedTransaction))
	var memoize, schemaBytes int
	must(t, journal.QueryRow(`SELECT memoize,length(result_schema) FROM attempts WHERE request=? AND ordinal=1`, stopped.Request).Scan(&memoize, &schemaBytes))
	if memoize != 1 || schemaBytes == 0 {
		t.Fatalf("actual local invocable lost its memo declaration: memoize=%d schema_bytes=%d", memoize, schemaBytes)
	}
	originalQuant, problem := store.RequestRow(stopped.Request)
	fatal(t, problem)
	for originalQuant.State == "pausing" {
		time.Sleep(25 * time.Millisecond)
		originalQuant, problem = store.RequestRow(stopped.Request)
		fatal(t, problem)
	}
	if originalQuant.Ordinal != 1 || originalQuant.State != "paused" {
		t.Fatalf("fault did not retain its original paused attempt: id=%s state=%s ordinal=%d", originalQuant.ID, originalQuant.State, originalQuant.Ordinal)
	}

	db, err := sql.Open("sqlite", "file:"+filepath.Join(root, "creator.sqlite")+"?mode=ro")
	must(t, err)
	defer db.Close()
	latest := func() records.Request {
		t.Helper()
		var id string
		must(t, db.QueryRow(`SELECT id FROM requests WHERE parent_request_id='' ORDER BY rowid DESC LIMIT 1`).Scan(&id))
		request, problem := store.RequestRow(id)
		fatal(t, problem)
		return *request
	}
	code = strings.Replace(code, "report.ssim < 1.01", "report.ssim < 0.0", 1)
	must(t, os.WriteFile(script, []byte(code), 0600))
	run := func() {
		t.Helper()
		status, out := runCozyPath(t, root, path, "run", script, "--await", "--json")
		if status != 0 {
			t.Fatalf("fresh script failed [%d]: %s", status, compositionTail(out))
		}
	}
	run()
	resumed := latest()
	children, problem := store.Children(resumed.ID)
	fatal(t, problem)
	if len(children) != 2 || children[0].ParentCallIndex != 2 || children[1].ParentCallIndex != 3 || children[0].State != "succeeded" || children[1].State != "succeeded" {
		t.Fatalf("resumed composition children incomplete: %+v", children)
	}
	for index, original := range []*records.NativeCall{originalSource, originalModel} {
		reused, problem := store.NativeCall(resumed.ID, int64(index))
		fatal(t, problem)
		if reused == nil || !bytes.Equal(reused.Result, original.Result) || !bytes.Equal(reused.NativeReceipt, original.NativeReceipt) {
			t.Fatalf("fresh script lost compatible native result: %+v", reused)
		}
	}
	var adopted int
	must(t, journal.QueryRow(`SELECT COUNT(*) FROM weights WHERE adoption_source=?`, interruptedTransaction).Scan(&adopted))
	if adopted == 0 {
		t.Fatal("fresh quantization did not adopt the real interrupted tensor checkpoint")
	}
	if bodyRequests.Load() != 1 {
		t.Fatal("fresh script redownloaded completed source")
	}
	previous := children
	run()
	complete := latest()
	children, problem = store.Children(complete.ID)
	fatal(t, problem)
	if len(children) != 2 {
		t.Fatal("completed replay lost managed calls", children)
	}
	for index, child := range children {
		if child.Ordinal != 0 || child.ReusedFrom != previous[index].ID {
			t.Fatalf("completed operation ran again: %+v", child)
		}
	}
	if bodyRequests.Load() != 1 {
		t.Fatal("completed memo replay redownloaded source")
	}
	code = strings.Replace(code, "fp8-rowwise/1", "mxfp8/1", 1)
	must(t, os.WriteFile(script, []byte(code), 0600))
	run()
	changed := latest()
	children, problem = store.Children(changed.ID)
	fatal(t, problem)
	if len(children) != 2 || children[0].Ordinal != 1 || children[0].ReusedFrom != "" {
		t.Fatalf("changed encoding reused incompatible quantization: %+v", children)
	}
	if bodyRequests.Load() != 1 {
		t.Fatal("changed quantization discarded retained source")
	}

}

func compositionTail(value string) string {
	if len(value) > 8192 {
		return strings.ToValidUTF8(value[len(value)-8192:], "?")
	}
	return value
}

// A successful result is not proof that the daemon released its owned processes.
// Keep the home (including shutdown logs) if normal teardown leaves one behind.
func compositionDown(t *testing.T, root, path string) {
	t.Helper()
	processes := map[int]string{}
	db, err := sql.Open("sqlite3", filepath.Join(root, "creator.sqlite")+"?mode=ro&_busy_timeout=5000")
	if err == nil {
		var rows *sql.Rows
		rows, err = db.Query(`SELECT pid,birth FROM worker_processes WHERE pid>0`)
		if err == nil {
			for rows.Next() {
				var pid int
				var birth string
				if err = rows.Scan(&pid, &birth); err != nil {
					break
				}
				processes[pid] = birth
			}
			if err == nil {
				err = rows.Err()
			}
			_ = rows.Close()
		}
		_ = db.Close()
	}
	if err != nil {
		t.Errorf("read owned worker identities before teardown: %v", err)
	}
	code, out := runCozyPath(t, root, path, "down", "--all", "--json")
	if code != 0 {
		t.Errorf("normal composition teardown refused (%d): %s", code, compositionTail(out))
	}
	deadline := time.Now().Add(30 * time.Second)
	for len(processes) > 0 {
		for pid, birth := range processes {
			raw, err := os.ReadFile(filepath.Join("/proc", strconv.Itoa(pid), "stat"))
			if errors.Is(err, os.ErrNotExist) {
				delete(processes, pid)
				continue
			}
			if err != nil {
				t.Errorf("read owned worker %d during teardown: %v", pid, err)
				return
			}
			// Compare the kernel start time so a recycled PID is never mistaken
			// for a worker this test owns. Field 22 follows the parenthesized name.
			fields := strings.Fields(string(raw)[strings.LastIndex(string(raw), ")")+1:])
			if len(fields) < 20 || birth == "" {
				t.Errorf("owned worker %d has no verifiable process birth", pid)
				return
			}
			if fields[19] != birth {
				delete(processes, pid)
			}
		}
		if len(processes) == 0 {
			return
		}
		if time.Now().After(deadline) {
			t.Errorf("normal teardown left owned worker identities alive: %v; down: %s", processes, compositionTail(out))
			return
		}
		time.Sleep(25 * time.Millisecond)
	}
}
