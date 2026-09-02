package producttest

import (
	"encoding/json"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/cozy-creator/cozy/internal/records"
	"github.com/cozy-creator/cozy/internal/units"
)

const localWeightlessRef = "local/cozy-weightless-package"
const editableRuntimeFixtureSHA = "5f2aea3625ea31c82f82f01ec3510c128411b74c"

// TestProductPath is the one end-to-end product path: a local package installed from
// source, invoked as a user types it, answered with a typed result and real bytes on
// disk — then edited underneath a running system and re-invoked. Every step is the real
// binary against a real daemon on a real root; the only fixture is the weightless
// package, which has no weights and so needs no card.
func TestProductPath(t *testing.T) {
	root := filepath.Join(os.TempDir(), "cozy-product-test", "editable-refresh")
	must(t, os.RemoveAll(root))
	must(t, os.MkdirAll(root, 0o755))
	t.Cleanup(func() {
		_, _ = runCozy(t, root, "down", "--all")
		_ = os.RemoveAll(root)
	})

	project := weightlessProject(t)
	code, out := runCozy(t, root, "package", "install", project, "--editable")
	if code != 0 {
		t.Fatalf("local directory package install [exit %d]\n%s", code, out)
	}
	if !strings.Contains(out, "source bytes are pinned locally") {
		t.Fatalf("local install omitted its unqualified identity note\n%s", out)
	}
	if code, out := runCozy(t, root, "package", "list", "--full", "--json"); code != 0 ||
		!strings.Contains(out, localWeightlessRef) || !strings.Contains(out, `"source":"local `) {
		t.Fatalf("package list omitted the install [exit %d]\n%s", code, out)
	}
	// Disk is two byte columns: the package's own bytes and its shared dependencies —
	// integers for a program, binary units for a person.
	code, out = runCozy(t, root, "package", "list", "--json")
	var listed struct {
		Packages []struct {
			Package      string `json:"package"`
			Size         int64  `json:"size"`
			Dependencies int64  `json:"dependencies"`
		} `json:"packages"`
	}
	if code != 0 || json.Unmarshal([]byte(out), &listed) != nil || len(listed.Packages) != 1 ||
		listed.Packages[0].Package != localWeightlessRef || listed.Packages[0].Size <= 0 ||
		listed.Packages[0].Dependencies <= 0 {
		t.Fatalf("package list --json lacks integer size and dependencies [exit %d]\n%s", code, out)
	}
	code, out = runCozy(t, root, "package", "list")
	if code != 0 || !regexp.MustCompile(`VERSION +SIZE +DEPENDENCIES\n`).MatchString(out) ||
		!strings.Contains(out, units.Bytes(listed.Packages[0].Size)+"  "+units.Bytes(listed.Packages[0].Dependencies)) {
		t.Fatalf("package list does not show SIZE and DEPENDENCIES in units [exit %d]\n%s", code, out)
	}
	if code, out := runCozy(t, root, "run", localWeightlessRef); code != 0 ||
		!strings.Contains(out, "- tile") || !strings.Contains(out, "- refuse") {
		t.Fatalf("package-only run did not list functions [exit %d]\n%s", code, out)
	}
	if code, out := runCozy(t, root, "run", localWeightlessRef+"/v1.0.0/tile"); code != 2 ||
		!strings.Contains(out, localWeightlessRef+"/tile") || !strings.Contains(out, "installed release") {
		t.Fatalf("version-in-path remedy was not useful [exit %d]\n%s", code, out)
	}

	outputDir := filepath.Join(root, "human-run-output")
	code, stdout, stderr := runCozyStreams(t, root, "run", localWeightlessRef+"/tile",
		"size=32", "--out", outputDir, "--await")
	if code != 0 {
		t.Fatalf("human invocation failed [exit %d]\nstdout:\n%s\nstderr:\n%s", code, stdout, stderr)
	}
	for _, useful := range []string{"pixels:", "revision:", "size:", "warm:", "saved:"} {
		if !strings.Contains(stdout, useful) {
			t.Errorf("human result omitted %q\n%s", useful, stdout)
		}
	}
	for _, internal := range []string{"result:", "asset_ref", "blake2b:", "digest:"} {
		if strings.Contains(stdout, internal) {
			t.Errorf("human result exposed %q despite saving the output\n%s", internal, stdout)
		}
	}
	for _, raw := range []string{"progress value=", "stage value=", "metric value="} {
		if strings.Contains(stderr, raw) {
			t.Errorf("redirected default progress exposed %q\n%s", raw, stderr)
		}
	}
	files, err := os.ReadDir(outputDir)
	must(t, err)
	if len(files) != 1 || !requestOutputName(files[0].Name(), ".webp") {
		t.Fatalf("first invocation did not use one request-hash filename: %v", files)
	}
	if !strings.Contains(stderr, filepath.Join(outputDir, files[0].Name())) {
		t.Fatalf("invocation did not announce its output hash before execution\n%s", stderr)
	}
	firstOutput := files[0].Name()
	code, _, stderr = runCozyStreams(t, root, "run", localWeightlessRef+"/tile",
		"size=32", "--out", outputDir)
	if code != 0 {
		t.Fatalf("second human invocation failed [exit %d]\n%s", code, stderr)
	}
	files, err = os.ReadDir(outputDir)
	must(t, err)
	if len(files) != 2 || files[0].Name() == files[1].Name() ||
		!requestOutputName(files[0].Name(), ".webp") || !requestOutputName(files[1].Name(), ".webp") {
		t.Fatalf("independent invocations did not retain two request-hash outputs: %v", files)
	}
	if files[0].Name() != firstOutput && files[1].Name() != firstOutput {
		t.Fatalf("second invocation replaced the first output %q: %v", firstOutput, files)
	}
	fixedDir := filepath.Join(root, "fixed-seed-output")
	for attempt := 0; attempt < 2; attempt++ {
		code, _, stderr = runCozyStreams(t, root, "run", localWeightlessRef+"/tile",
			"size=32", "seed=7", "--out", fixedDir)
		if code != 0 {
			t.Fatalf("fixed-seed invocation %d failed [exit %d]\n%s", attempt+1, code, stderr)
		}
	}
	fixed, err := os.ReadDir(fixedDir)
	must(t, err)
	if len(fixed) != 1 || !requestOutputName(fixed[0].Name(), ".webp") {
		t.Fatalf("the same explicit payload did not resolve to one stable filename: %v", fixed)
	}

	type listedRun struct {
		Number    string `json:"number"`
		ID        string `json:"id"`
		Kind      string `json:"kind"`
		Target    string `json:"target"`
		Machine   string `json:"machine"`
		Status    string `json:"status"`
		Queued    string `json:"queued"`
		Execution string `json:"execution"`
	}
	listRuns := func() []listedRun {
		t.Helper()
		code, out := runCozy(t, root, "run", "list", "--json", "--full")
		var document struct {
			Invocations []listedRun `json:"invocations"`
		}
		if code != 0 {
			t.Fatalf("run list failed [exit %d]\n%s", code, out)
		}
		if err := json.Unmarshal([]byte(out), &document); err != nil {
			t.Fatalf("run list returned invalid JSON: %v\n%s", err, out)
		}
		return document.Invocations
	}
	runs := listRuns()
	if len(runs) < 4 {
		t.Fatalf("run list omitted recorded invocations: %+v", runs)
	}
	for index, row := range runs {
		number, err := strconv.ParseInt(row.Number, 10, 64)
		if err != nil || number < 1 || !strings.HasPrefix(row.ID, "req-") ||
			row.Kind != "invocation" || !strings.HasSuffix(row.Queued, "s") ||
			!strings.HasSuffix(row.Execution, "s") {
			t.Fatalf("run list row %d is not useful: %+v (%v)", index, row, err)
		}
		if row.Machine != "local" {
			t.Fatalf("a run attempted on this host's own worker is MACHINE %q, not local: %+v",
				row.Machine, row)
		}
		if index > 0 {
			prior, _ := strconv.ParseInt(runs[index-1].Number, 10, 64)
			if prior <= number {
				t.Fatalf("run list is not newest-first by local number: %+v", runs)
			}
		}
	}
	latest := runs[0]
	if code, out := runCozy(t, root, "run", "watch", latest.Number, "--json"); code != 0 ||
		!strings.Contains(out, `"status":"completed"`) ||
		!strings.Contains(out, `"machine":"local"`) {
		t.Fatalf("numeric run watch failed or lost its machine [exit %d]\n%s", code, out)
	}
	if code, out := runCozy(t, root, "run", "cancel", latest.Number, "--json"); code != 0 ||
		!strings.Contains(out, `"changed":false`) {
		t.Fatalf("numeric run cancel was not idempotent [exit %d]\n%s", code, out)
	}
	if code, out := runCozy(t, root, "run", "list", "--limit", "2"); code != 0 ||
		!strings.Contains(out, "NUMBER") || !strings.Contains(out, "MACHINE") ||
		!strings.Contains(out, "QUEUED") || !strings.Contains(out, "EXECUTION") ||
		!strings.Contains(out, "local") ||
		strings.Contains(out, "KIND") || strings.Index(out, runs[0].Number) > strings.Index(out, runs[1].Number) {
		t.Fatalf("human run list columns/order are not useful [exit %d]\n%s", code, out)
	}

	code, out = runCozy(t, root, "run", localWeightlessRef+"/tile_job",
		"size=8", "seed=11", "--await", "--json")
	if code != 1 || !strings.Contains(out, "editable_jobs_unsupported") {
		t.Fatalf("editable job red arm changed [exit %d]\n%s", code, out)
	}
	runs = listRuns()
	if len(runs) == 0 {
		t.Fatal("failed job is absent from run list")
	}
	// A request refused before any attempt waited and never ran: its queue clock is
	// closed at the terminal, and its execution time is exactly nothing.
	jobQueued, err := strconv.ParseFloat(strings.TrimSuffix(runs[0].Queued, "s"), 64)
	if runs[0].Kind != "job" || runs[0].Status != "failed" ||
		err != nil || jobQueued < 0 || jobQueued > 10 || runs[0].Execution != "0.0s" {
		t.Fatalf("pre-attempt job row lost its kind or its clocks: %+v (%v)", runs, err)
	}
	if runs[0].Machine != "" {
		t.Fatalf("a request that never landed anywhere claims MACHINE %q: %+v", runs[0].Machine, runs[0])
	}
	if code, out := runCozy(t, root, "run", "watch", runs[0].Number, "--json"); code == 0 ||
		!strings.Contains(out, "editable_jobs_unsupported") {
		t.Fatalf("numeric failed-job watch did not resolve the job [exit %d]\n%s", code, out)
	}

	detachedDir := filepath.Join(root, "detached-output")
	code, stdout, stderr = runCozyStreams(t, root, "--json", "run", localWeightlessRef+"/tile",
		"size=32", "seed=9", "delay_ms=4500", "--out", detachedDir)
	if code != 0 || !strings.Contains(stdout, `"status":"running"`) ||
		!strings.Contains(stdout, `"output":"`+detachedDir) ||
		!strings.Contains(stdout, `.webp"`) {
		t.Fatalf("default run did not detach with its durable WebP destination [exit %d]\nstdout:\n%s\nstderr:\n%s",
			code, stdout, stderr)
	}
	deadline := time.Now().Add(10 * time.Second)
	for {
		detached, err := os.ReadDir(detachedDir)
		if err == nil && len(detached) == 1 && requestOutputName(detached[0].Name(), ".webp") {
			break
		}
		if time.Now().After(deadline) {
			t.Fatalf("daemon did not publish the detached WebP: %v, %v", detached, err)
		}
		time.Sleep(25 * time.Millisecond)
	}

	// cl-089 first half: an unwritable --out refuses AT SUBMIT, typed, before any GPU
	// time — and before any request row exists to retry.
	unwritableDir := filepath.Join(root, "unwritable-output")
	must(t, os.MkdirAll(unwritableDir, 0o500))
	t.Cleanup(func() { _ = os.Chmod(unwritableDir, 0o755) })
	before := len(listRuns())
	code, out = runCozy(t, root, "run", localWeightlessRef+"/tile",
		"size=32", "--out", unwritableDir, "--await", "--json")
	if code != 1 || !strings.Contains(out, `"code":"output_destination_unwritable"`) ||
		!strings.Contains(out, unwritableDir) {
		t.Fatalf("mode-500 --out was not refused typed at submit [exit %d]\n%s", code, out)
	}
	if after := len(listRuns()); after != before {
		t.Fatalf("a refused --out submission recorded a request row: %d -> %d", before, after)
	}

	// cl-089 second half: the incident's exact shape — the destination dies AFTER the
	// preflight passed. The run's verdict stays completed; the export is reported as
	// owed, distinctly, never as run failure; and once the directory is restored the
	// same idempotency key re-triggers the durable export and the bytes land.
	incidentDir := filepath.Join(root, "incident-output")
	code, stdout, stderr = runCozyStreams(t, root, "--json", "run", localWeightlessRef+"/tile",
		"size=32", "seed=13", "delay_ms=4500", "--out", incidentDir,
		"--idempotency-key", "incident-cl089")
	if code != 0 {
		t.Fatalf("incident submit failed [exit %d]\nstdout:\n%s\nstderr:\n%s", code, stdout, stderr)
	}
	idMatch := regexp.MustCompile(`"run":"([^"]+)"`).FindStringSubmatch(stdout)
	if idMatch == nil {
		t.Fatalf("detached incident run printed no run reference\n%s", stdout)
	}
	must(t, os.Chmod(incidentDir, 0o500))
	t.Cleanup(func() { _ = os.Chmod(incidentDir, 0o755) })
	code, out = runCozy(t, root, "run", "watch", idMatch[1], "--json")
	if code != 0 ||
		!strings.Contains(out, `"status":"completed (export pending: output_export_io)"`) ||
		!strings.Contains(out, "daemon retries this durable export") {
		t.Fatalf("succeeded run with failed export did not report the distinct verdict [exit %d]\n%s", code, out)
	}
	must(t, os.Chmod(incidentDir, 0o755))
	code, out = runCozy(t, root, "--json", "run", localWeightlessRef+"/tile",
		"size=32", "seed=13", "delay_ms=4500", "--out", incidentDir,
		"--idempotency-key", "incident-cl089", "--await")
	if code != 0 || !strings.Contains(out, `"status":"completed"`) ||
		strings.Contains(out, "export pending") {
		t.Fatalf("restored destination did not publish on the same key [exit %d]\n%s", code, out)
	}
	incidentFiles, err := os.ReadDir(incidentDir)
	must(t, err)
	if len(incidentFiles) != 1 || !requestOutputName(incidentFiles[0].Name(), ".webp") {
		t.Fatalf("restored export did not land the WebP: %v", incidentFiles)
	}

	code, _, stderr = runCozyStreams(t, root, "run", localWeightlessRef+"/tile",
		"size=32", "seed=7", "--full", "--await")
	if code != 0 || !strings.Contains(stderr, "progress fraction=") {
		t.Fatalf("--full did not retain Runtime diagnostics [exit %d]\n%s", code, stderr)
	}

	for _, deleted := range []string{"--local", "--cloud", "--machine"} {
		if code, out := runCozy(t, root, "run", localWeightlessRef+"/tile",
			"size=32", "seed=7", deleted); code != 2 ||
			!strings.Contains(out, "unknown flag") {
			t.Fatalf("deleted %s did not refuse [exit %d]\n%s", deleted, code, out)
		}
	}
	code, out = runCozy(t, root, "run", localWeightlessRef+"/tile",
		"size=32", "seed=7", "--idempotency-key", "placement-proof", "--json", "--await")
	if code != 0 || !strings.Contains(out, `"revision":"first"`) ||
		!strings.Contains(out, `"digest":`) || !strings.Contains(out, `"result":`) {
		t.Fatalf("first editable invocation did not run source [exit %d]\n%s\n%s",
			code, out, productWorkerLogs(root))
	}
	store, problem := records.Open(filepath.Join(root, "records.db"))
	fatal(t, problem)
	requests, problem := store.RequestsOfKind("serving", "", 10)
	fatal(t, problem)
	store.Close()
	if len(requests) == 0 || requests[0].Rental || requests[0].Worker != "" {
		t.Fatalf("default local placement was not retained exactly: %+v", requests)
	}
	first := activePackageInstall(t, root)

	source := filepath.Join(project, "weightless.py")
	body, err := os.ReadFile(source)
	must(t, err)
	body = []byte(strings.Replace(string(body), `REVISION = "first"`, `REVISION = "second"`, 1))
	must(t, os.WriteFile(source, body, 0o644))
	code, out = runCozy(t, root, "run", localWeightlessRef+"/tile",
		"size=32", "seed=7", "--json", "--await")
	if code != 0 || !strings.Contains(out, `"revision":"second"`) {
		t.Fatalf("edited body was not live on the next invocation [exit %d]\n%s", code, out)
	}
	second := activePackageInstall(t, root)
	if second.ID == first.ID || second.SourceDigest == first.SourceDigest {
		t.Fatalf("body edit did not advance the editable install: %#v -> %#v", first, second)
	}

	pyproject := filepath.Join(project, "pyproject.toml")
	metadata, err := os.ReadFile(pyproject)
	must(t, err)
	must(t, os.WriteFile(pyproject, append(metadata, []byte("\n# editable metadata refresh\n")...), 0o644))
	lock := filepath.Join(project, "uv.lock")
	lockBytes, err := os.ReadFile(lock)
	must(t, err)
	must(t, os.WriteFile(lock, append(lockBytes, []byte("\n# editable lock refresh\n")...), 0o644))
	code, out = runCozy(t, root, "run", localWeightlessRef+"/tile",
		"size=32", "seed=7", "--json", "--await")
	if code != 0 || !strings.Contains(out, `"revision":"second"`) {
		t.Fatalf("metadata/lock refresh did not remain runnable [exit %d]\n%s", code, out)
	}
	third := activePackageInstall(t, root)
	if third.ID == second.ID || third.LockDigest == second.LockDigest {
		t.Fatalf("metadata/lock edit did not atomically refresh the environment: %#v -> %#v", second, third)
	}

	goodMetadata, err := os.ReadFile(pyproject)
	must(t, err)
	must(t, os.WriteFile(pyproject, []byte("[project\n"), 0o644))
	code, out = runCozy(t, root, "run", localWeightlessRef+"/tile",
		"size=32", "seed=7", "--json", "--await")
	if code != 1 || !strings.Contains(out, `"code":"editable_refresh_failed"`) {
		t.Fatalf("failed edit did not return the typed refresh refusal [exit %d]\n%s", code, out)
	}
	failed := activePackageInstall(t, root)
	if failed.ID != third.ID || failed.SourceDigest != third.SourceDigest {
		t.Fatalf("failed refresh displaced the last good install: %#v -> %#v", third, failed)
	}
	must(t, os.WriteFile(pyproject, goodMetadata, 0o644))
	code, out = runCozy(t, root, "run", localWeightlessRef+"/tile",
		"size=32", "seed=7", "--json", "--await")
	if code != 0 || !strings.Contains(out, `"revision":"second"`) {
		t.Fatalf("restored source did not reuse the last good install [exit %d]\n%s", code, out)
	}
}

func requestOutputName(name, extension string) bool {
	if !strings.HasSuffix(name, extension) {
		return false
	}
	digest := strings.TrimSuffix(name, extension)
	if len(digest) != 64 {
		return false
	}
	for _, char := range digest {
		if !strings.ContainsRune("0123456789abcdef", char) {
			return false
		}
	}
	return true
}

func activePackageInstall(t *testing.T, root string) records.PackageInstall {
	return activeInstall(t, root, localWeightlessRef)
}

func activeInstall(t *testing.T, root, packageRef string) records.PackageInstall {
	t.Helper()
	store, problem := records.Open(filepath.Join(root, "records.db"))
	fatal(t, problem)
	defer store.Close()
	_, install, problem := store.ActivePackage(packageRef)
	fatal(t, problem)
	if install == nil {
		t.Fatalf("editable package %s has no active install", packageRef)
	}
	return *install
}

func weightlessProject(t *testing.T) string {
	t.Helper()
	home, err := os.UserHomeDir()
	must(t, err)
	repo := filepath.Join(home, "cozy_v2", "cozy-runtime") //cozy:allow peer source; the fixture builds the exact install Runtime
	if _, err := os.Stat(filepath.Join(repo, "pyproject.toml")); err != nil {
		t.Skipf("no cozy-runtime peer at %s: %v", repo, err)
	}
	dir := t.TempDir()
	project := filepath.Join(dir, "source")
	build := exec.Command("/usr/bin/nice", "-n", "19", "python3",
		"tests/product/testdata/build-weightless.py", "--out", dir, "--source-out", project,
		"--runtime-sha", editableRuntimeFixtureSHA)
	build.Dir = "../.."
	build.Env = childEnv(t, repo, "RUNTIME_REPO="+repo)
	if out, err := build.CombinedOutput(); err != nil {
		t.Fatalf("building the weightless release: %v\n%s", err, out)
	}
	return project
}

func productWorkerLogs(root string) string {
	paths, _ := filepath.Glob(filepath.Join(root, "workers", "*", "worker.log"))
	var out strings.Builder
	for _, path := range paths {
		data, _ := os.ReadFile(path)
		out.WriteString("worker log " + filepath.Base(filepath.Dir(path)) + ":\n" + string(data) + "\n")
	}
	return out.String()
}
