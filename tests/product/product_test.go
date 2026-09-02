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
const editableRuntimeFixtureSHA = "22c159d9eb100c5981487f57b0aa2e25df01eb4a"

// TestProductPath is the one end-to-end product path: a local package installed from
// source, invoked as a user types it, answered with a typed result and real bytes on
// disk — then edited underneath a running system and re-invoked. Every step is the real
// binary against a real daemon on a real root; the only fixture is the weightless
// package, which has no weights and so needs no card.
func TestProductPath(t *testing.T) {
	// A per-process root: sessions on one box run this suite concurrently, and a shared
	// fixed path let one run's setup wipe another's mid-flight. Short on purpose — the
	// daemon's worker socket lives under it and unix socket paths are bounded.
	root, err := os.MkdirTemp(os.TempDir(), "cozy-product-")
	must(t, err)
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

	// THE DEFAULT LOCATION (cl-090): no --out, and the run still says where the file is —
	// the package's own store, `outputs/<org>-<package>/<content digest>.<ext>`.
	storeDir := filepath.Join(root, "outputs", "local-cozy-weightless-package")
	code, stdout, stderr := runCozyStreams(t, root, "run", localWeightlessRef+"/tile",
		"size=32", "--await")
	if code != 0 {
		t.Fatalf("human invocation failed [exit %d]\nstdout:\n%s\nstderr:\n%s", code, stdout, stderr)
	}
	for _, useful := range []string{"pixels:", "revision:", "size:", "warm:", "saved:", storeDir + "/"} {
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
	if !strings.Contains(stderr, "Saving outputs to "+storeDir) {
		t.Fatalf("invocation did not announce its real output directory before execution\n%s", stderr)
	}
	files, err := os.ReadDir(storeDir)
	must(t, err)
	if len(files) != 1 || !requestOutputName(files[0].Name(), ".webp") {
		t.Fatalf("default location does not hold one content-digest file: %v", files)
	}
	firstOutput := files[0].Name()
	if !strings.Contains(stdout, filepath.Join(storeDir, firstOutput)) {
		t.Fatalf("saved: does not name the store path\n%s", stdout)
	}
	assertOnlyResultFiles(t, filepath.Join(root, "outputs"))
	assertNoAttemptRoot(t, root)

	// An unseeded run draws fresh entropy, so its bytes — and its name — differ.
	code, _, stderr = runCozyStreams(t, root, "run", localWeightlessRef+"/tile", "size=32")
	if code != 0 {
		t.Fatalf("second human invocation failed [exit %d]\n%s", code, stderr)
	}
	files, err = os.ReadDir(storeDir)
	must(t, err)
	if len(files) != 2 || files[0].Name() == files[1].Name() ||
		!requestOutputName(files[0].Name(), ".webp") || !requestOutputName(files[1].Name(), ".webp") {
		t.Fatalf("distinct results did not retain two content-digest files: %v", files)
	}
	// The same bytes again land on the same name: one file, no second copy.
	var stable string
	for attempt := 0; attempt < 2; attempt++ {
		code, stdout, _ = runCozyStreams(t, root, "run", localWeightlessRef+"/tile", "size=32", "seed=7", "--await")
		if code != 0 || !strings.Contains(stdout, storeDir+"/") {
			t.Fatalf("seeded invocation %d did not report its store path [exit %d]\n%s", attempt+1, code, stdout)
		}
		if files, err = os.ReadDir(storeDir); err != nil || len(files) != 3 {
			t.Fatalf("seeded invocation %d did not land on one stable file: %v, %v", attempt+1, files, err)
		}
		reported := savedLine(stdout)
		if stable == "" {
			stable = reported
		} else if stable != reported {
			t.Fatalf("regenerating the same bytes reported a different path:\n%s\n%s", stable, reported)
		}
	}
	assertOnlyResultFiles(t, filepath.Join(root, "outputs"))

	// --out overrides the directory and keeps the naming.
	outputDir := filepath.Join(root, "human-run-output")
	code, stdout, stderr = runCozyStreams(t, root, "run", localWeightlessRef+"/tile",
		"size=32", "seed=7", "--out", outputDir, "--await")
	if code != 0 || !strings.Contains(stdout, outputDir+"/") {
		t.Fatalf("--out invocation did not save under the requested directory [exit %d]\n%s\n%s",
			code, stdout, stderr)
	}
	if outFiles, err := os.ReadDir(outputDir); err != nil || len(outFiles) != 1 ||
		!strings.Contains(stable, outFiles[0].Name()) {
		t.Fatalf("--out did not keep the content-digest name: %v, %v", outFiles, err)
	}
	if !strings.Contains(stderr, "Saving outputs to "+outputDir) {
		t.Fatalf("--out invocation did not announce its directory\n%s", stderr)
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
	assertOnlyResultFiles(t, outputDir)
	assertOnlyResultFiles(t, fixedDir)
	assertNoAttemptRoot(t, root)

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
		!strings.Contains(stdout, `"output":"`+detachedDir+`"`) {
		t.Fatalf("default run did not detach with its durable destination [exit %d]\nstdout:\n%s\nstderr:\n%s",
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

	// cl-089's incident shape — the destination dies AFTER the preflight passed — under
	// direct writes: the worker writes the result where it lives and there is no second
	// copy to publish later, so the run FAILS typed, naming the refused write, and a run
	// after the directory is restored lands the file.
	incidentDir := filepath.Join(root, "incident-output")
	code, stdout, stderr = runCozyStreams(t, root, "--json", "run", localWeightlessRef+"/tile",
		"size=32", "seed=13", "delay_ms=4500", "--out", incidentDir)
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
	if code != 1 || !strings.Contains(out, `"code":"failed"`) ||
		!strings.Contains(out, "output_spool_io") || !strings.Contains(out, "Permission denied") {
		t.Fatalf("a destination that died after submit did not fail the run typed [exit %d]\n%s", code, out)
	}
	must(t, os.Chmod(incidentDir, 0o755))
	code, out = runCozy(t, root, "--json", "run", localWeightlessRef+"/tile",
		"size=32", "seed=13", "--out", incidentDir, "--await")
	if code != 0 || !strings.Contains(out, `"status":"completed"`) {
		t.Fatalf("restored destination did not take the result [exit %d]\n%s", code, out)
	}
	incidentFiles, err := os.ReadDir(incidentDir)
	must(t, err)
	if len(incidentFiles) != 1 || !requestOutputName(incidentFiles[0].Name(), ".webp") {
		t.Fatalf("restored destination did not land the WebP: %v", incidentFiles)
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
		!strings.Contains(out, `"digest":`) || !strings.Contains(out, `"result":`) ||
		!strings.Contains(out, `"saved":[{`) || !strings.Contains(out, `"path":"`+storeDir+"/") {
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

// savedLine is the one `saved:` entry of a human run document.
func savedLine(stdout string) string {
	for _, line := range strings.Split(stdout, "\n") {
		if strings.HasPrefix(strings.TrimSpace(line), "- /") {
			return strings.TrimSpace(line)
		}
	}
	return ""
}

// assertOnlyResultFiles is the owner's rule for the user-facing store: nothing but
// result files — `<content digest>.<ext>`, regular files — ever lands under outputs/ or
// a --out directory. No payload, no manifest, no sidecar, no attempt directory.
func assertOnlyResultFiles(t *testing.T, dir string) {
	t.Helper()
	err := filepath.WalkDir(dir, func(path string, entry os.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if path == dir {
			return nil
		}
		rel, _ := filepath.Rel(dir, path)
		if entry.IsDir() {
			// The store is outputs/<org>-<package>/ and nothing deeper; --out is flat.
			if filepath.Base(dir) != "outputs" || strings.Contains(rel, string(filepath.Separator)) ||
				!strings.Contains(rel, "-") {
				t.Errorf("%s holds a directory %s; the store is one flat directory per package", dir, rel)
			}
			return nil
		}
		if !entry.Type().IsRegular() || !requestOutputName(entry.Name(), filepath.Ext(entry.Name())) ||
			filepath.Ext(entry.Name()) == "" {
			t.Errorf("%s holds %s, which is not a <content digest>.<ext> result file", dir, rel)
		}
		return nil
	})
	must(t, err)
}

// assertNoAttemptRoot is the owner's rule that a run stages nothing: the result is
// written where it lives, the payload rides the grant, and no `attempts/` exists.
func assertNoAttemptRoot(t *testing.T, root string) {
	t.Helper()
	if _, err := os.Stat(filepath.Join(root, "attempts")); !os.IsNotExist(err) {
		t.Fatalf("the local root holds an attempt working area: %v", err)
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
