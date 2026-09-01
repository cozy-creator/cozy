package producttest

import (
	"archive/zip"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"os"
	"os/exec"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
	"time"

	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/launch"
	"github.com/cozy-creator/cozy/internal/output"
	"github.com/cozy-creator/cozy/internal/packagepublish"
	"github.com/cozy-creator/cozy/internal/records"
	"github.com/cozy-creator/cozy/internal/wheel"
)

const weightlessRef = "cozy/cozy-weightless-package"
const localWeightlessRef = "local/cozy-weightless-package"
const editableRuntimeFixtureSHA = "ebdcb4a319dd1d6d2afa352d9dfe824c2bca2b57"
const editableTensorFSFixtureSHA = "0f49a4bf3fbe6fc8d41713b7ce9041c80161b7e6"

func TestHumanQueuePositionLabelKeepsMachineKey(t *testing.T) {
	record := output.Record{Fields: []output.Field{{K: "queue_position", V: "9/9"}}}
	var human bytes.Buffer
	if err := record.Emit(&human, output.Mode{Human: true}); err != nil {
		t.Fatal(err)
	}
	if got := human.String(); got != "queue position: 9/9\n" {
		t.Fatalf("human queue label = %q", got)
	}
	var jsonOut bytes.Buffer
	if err := record.Emit(&jsonOut, output.Mode{JSON: true}); err != nil {
		t.Fatal(err)
	}
	if got := jsonOut.String(); !strings.Contains(got, `"queue_position":"9/9"`) {
		t.Fatalf("machine queue key drifted: %s", got)
	}
}

func TestLiteralPayloadUsesOrdinaryScalarSyntax(t *testing.T) {
	entrypoint := &launch.Entrypoint{
		Name: "marco",
		Request: launch.Struct{Fields: []launch.Field{{
			Name: "message", Type: json.RawMessage(`{"literal":["marco"]}`), Wire: "required",
		}}},
	}
	for _, terms := range [][]string{{"message=marco"}, {"marco"}} {
		payload, problem := launch.ParsePayload(entrypoint, terms, "")
		if problem != nil || string(payload) != `{"message":"marco"}` {
			t.Fatalf("ParsePayload(%q) = %s, %v", terms, payload, problem)
		}
	}
	if _, problem := launch.ParsePayload(entrypoint, []string{"message=polo"}, ""); problem == nil || !strings.Contains(problem.Message, `must be one of: "marco"`) {
		t.Fatalf("wrong literal was not explained: %v", problem)
	}
}

func TestResultAssetSpecCarriesExactOutputMediaType(t *testing.T) {
	entrypoint := &launch.Entrypoint{Result: launch.Struct{Fields: []launch.Field{{
		Name: "image", Type: json.RawMessage(`{"asset":"image"}`),
		AssetBound: struct {
			MaxBytes   int64    `json:"max_bytes"`
			MediaTypes []string `json:"media_types"`
		}{MaxBytes: 64 << 20, MediaTypes: []string{"image/webp"}},
	}}}}
	spec, ok := launch.ResultAssetSpec(entrypoint, "image")
	if !ok || spec.Kind != "image" || spec.MaxBytes != 64<<20 ||
		len(spec.MediaTypes) != 1 || spec.MediaTypes[0] != "image/webp" {
		t.Fatalf("result asset spec lost the exact output contract: %#v, %v", spec, ok)
	}
}

func TestPackageHasOneActiveVersion(t *testing.T) {
	store, problem := records.Open(filepath.Join(t.TempDir(), "records.db"))
	fatal(t, problem)
	defer store.Close()
	install := func(id, version string, major int) records.PackageInstall {
		return records.PackageInstall{
			ID: id, Package: "cozy/example", Major: major, Version: version,
			SourceKind: "tensorhub", SourceRef: "cozy/example@" + version,
			SourceDigest: "sha256:" + strings.Repeat(id, 64/len(id)), Verified: true,
			Dir: filepath.Join(t.TempDir(), id),
		}
	}
	first := install("a", "1.0.0", 1)
	second := install("b", "2.0.0", 2)
	second.PlacementSetDigest = "sha256:" + strings.Repeat("b", 64)
	if _, problem = store.Activate(first); problem != nil {
		t.Fatal(problem)
	}
	if superseded, problem := store.Activate(second); problem != nil || superseded != first.ID {
		t.Fatalf("replacement = %q, %v", superseded, problem)
	}
	pins, problem := store.Pins("cozy/example")
	if problem != nil || len(pins) != 1 || pins[0].InstallID != second.ID {
		t.Fatalf("active pins = %+v, %v", pins, problem)
	}
	_, active, problem := store.ActivePackage("cozy/example")
	if problem != nil || active == nil || active.PlacementSetDigest != second.PlacementSetDigest {
		t.Fatalf("active selection = %+v, %v", active, problem)
	}
}

func TestModelProductionGrammar(t *testing.T) {
	root := t.TempDir()
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet || r.URL.Path != "/v1/accounts/current" ||
			r.Header.Get("Authorization") != "Bearer proof-token" {
			http.NotFound(w, r)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{"name":"acme"}`)
	}))
	defer server.Close()
	accountEnv := []string{"TENSORHUB_URL=" + server.URL, "TENSORHUB_TOKEN=proof-token"}
	code, help := runCozy(t, root, "model", "upload", "--help")
	if code != 0 || !strings.Contains(help, "<source>") ||
		strings.Contains(help, "--release") || !strings.Contains(help, "--producer") ||
		!strings.Contains(help, "--lane") || !strings.Contains(help, "--rental") ||
		!strings.Contains(help, "--dry-run") || !strings.Contains(help, "--detach") ||
		strings.Contains(help, "<manifest>") || strings.Contains(help, "--token-stdin") ||
		strings.Contains(help, "--cloud") || strings.Contains(help, "--remote") ||
		strings.Contains(help, "--machine") || strings.Contains(help, "--max-cost") {
		t.Fatalf("model upload grammar drifted [exit %d]\n%s", code, help)
	}
	code, out := runCozy(t, root, "model", "upload", "acme/model")
	if code != 2 || !strings.Contains(out, "<source>") {
		t.Fatalf("model upload accepted missing source [exit %d]\n%s", code, out)
	}
	local := filepath.Join(root, "source.safetensors")
	must(t, os.WriteFile(local, []byte("header-only-grammar-fixture"), 0o600))
	code, out = runCozyDir(t, root, ".", accountEnv, "--json", "model", "upload", "acme/model", local,
		"--dry-run")
	if code != 0 || !strings.Contains(out, `"kind":"model-upload"`) ||
		!strings.Contains(out, `"status":"planned"`) || !strings.Contains(out, `"id":"modelupload-`) {
		t.Fatalf("source-driven dry-run failed [exit %d]\n%s", code, out)
	}
	code, out = runCozyDir(t, root, ".", accountEnv, "model", "upload", "other/model", local,
		"--dry-run")
	if code != 2 || !strings.Contains(out, "logged in as Tensorhub account acme") ||
		!strings.Contains(out, "publish as acme/model") {
		t.Fatalf("cross-account model upload was not refused [exit %d]\n%s", code, out)
	}
	code, out = runCozy(t, root, "model", "upload", "acme/model", local,
		"--dry-run", "--detach")
	if code != 2 || !strings.Contains(out, "conflict") {
		t.Fatalf("dry-run plus detach was accepted [exit %d]\n%s", code, out)
	}
	code, help = runCozy(t, root, "model", "publish", "--help")
	if code != 0 || strings.Contains(help, "<source>") || !strings.Contains(help, "--release") ||
		!strings.Contains(help, "--lane") || !strings.Contains(help, "--remove-lane") ||
		strings.Contains(help, "--producer") || strings.Contains(help, "--rental") ||
		strings.Contains(help, "--dry-run") || strings.Contains(help, "--detach") {
		t.Fatalf("model publish grammar drifted [exit %d]\n%s", code, help)
	}
	code, out = runCozy(t, root, "model", "publish", "acme/model", "--release", "1.2.3")
	if code != 2 || !strings.Contains(out, "requires --lane") {
		t.Fatalf("model publish accepted no lane changes [exit %d]\n%s", code, out)
	}
	code, help = runCozy(t, root, "model", "yank", "--help")
	if code != 0 || !strings.Contains(help, "--release") {
		t.Fatalf("model yank grammar drifted [exit %d]\n%s", code, help)
	}
	code, help = runCozy(t, root, "model", "download", "--help")
	if code != 0 || strings.Contains(strings.ToLower(help), "snapshot") {
		t.Fatalf("model download retained snapshot vocabulary [exit %d]\n%s", code, help)
	}
}

func TestModelReleaseUpdateAndYankCLIContracts(t *testing.T) {
	checkpoint := "sha256:" + strings.Repeat("a", 64)
	requests := []string{}
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		requests = append(requests, r.Method+" "+r.URL.Path)
		w.Header().Set("Content-Type", "application/json")
		switch len(requests) {
		case 1, 4:
			if r.Method != http.MethodGet || r.URL.Path != "/v1/accounts/current" {
				t.Fatalf("account request = %s %s", r.Method, r.URL.Path)
			}
			_, _ = io.WriteString(w, `{"name":"acme"}`)
		case 2:
			if r.Method != http.MethodGet || r.URL.Path != "/v1/models/acme/model/releases/stable" {
				t.Fatalf("release read = %s %s", r.Method, r.URL.Path)
			}
			_, _ = io.WriteString(w, `{"release":"stable","revision":4,"yanked":false,"lanes":[{"lane":"broken","checkpoint_id":"`+checkpoint+`","contract":{"stamps":{},"structure":"sha256:`+strings.Repeat("b", 64)+`","encoding":{"set":["bf16"]}},"objects":1,"bytes":7}],"repository_sha256":"`+strings.Repeat("c", 64)+`","changed":false}`)
		case 3:
			if r.Method != http.MethodPost || r.URL.Path != "/v1/models/acme/model/releases/stable" ||
				r.Header.Get("X-Tensorhub-Reason") != "cozy model publish acme/model@stable" {
				t.Fatalf("release update = %s %s %#v", r.Method, r.URL.Path, r.Header)
			}
			var body struct {
				ExpectedRevision int64             `json:"expected_revision"`
				SetLanes         map[string]string `json:"set_lanes"`
				RemoveLanes      []string          `json:"remove_lanes"`
			}
			if err := json.NewDecoder(r.Body).Decode(&body); err != nil || body.ExpectedRevision != 4 ||
				body.SetLanes["fp8"] != checkpoint || !reflect.DeepEqual(body.RemoveLanes, []string{"broken"}) {
				t.Fatalf("release update body = %#v, %v", body, err)
			}
			_, _ = io.WriteString(w, `{"release":"stable","revision":5,"yanked":false,"lanes":[{"lane":"fp8","checkpoint_id":"`+checkpoint+`","contract":{"stamps":{},"structure":"sha256:`+strings.Repeat("b", 64)+`","encoding":{"set":["fp8"]}},"objects":1,"bytes":7}],"repository_sha256":"`+strings.Repeat("d", 64)+`","changed":true}`)
		case 5:
			if r.Method != http.MethodDelete || r.URL.Path != "/v1/models/acme/model/releases/stable" ||
				r.Header.Get("X-Tensorhub-Reason") != "cozy model yank acme/model@stable" {
				t.Fatalf("release yank = %s %s %#v", r.Method, r.URL.Path, r.Header)
			}
			_, _ = io.WriteString(w, `{"release":"stable","revision":6,"yanked":true,"lanes":[{"lane":"fp8","checkpoint_id":"`+checkpoint+`","contract":{"stamps":{},"structure":"sha256:`+strings.Repeat("b", 64)+`","encoding":{"set":["fp8"]}},"objects":1,"bytes":7}],"repository_sha256":"`+strings.Repeat("e", 64)+`","changed":true}`)
		default:
			t.Fatalf("unexpected request %s %s", r.Method, r.URL.Path)
		}
	}))
	defer server.Close()
	env := []string{"TENSORHUB_URL=" + server.URL, "TENSORHUB_TOKEN=proof-token"}
	root := t.TempDir()
	code, out := runCozyDir(t, root, ".", env, "--json", "model", "publish", "acme/model",
		"--release", "stable", "--lane", "fp8="+checkpoint, "--remove-lane", "broken")
	if code != 0 || !strings.Contains(out, `"revision":5`) || !strings.Contains(out, `"status":"published"`) {
		t.Fatalf("model release update [exit %d]\n%s", code, out)
	}
	code, out = runCozyDir(t, root, ".", env, "--json", "model", "yank", "acme/model",
		"--release", "stable")
	if code != 0 || !strings.Contains(out, `"revision":6`) || !strings.Contains(out, `"status":"yanked"`) {
		t.Fatalf("model release yank [exit %d]\n%s", code, out)
	}
}

func TestPackagePublishMetadataGrammar(t *testing.T) {
	root := t.TempDir()
	if code, help := runCozy(t, root, "run", "cozy/example/function", "--help"); code != 0 ||
		strings.Contains(help, "--version") || strings.Contains(help, "vN/function") ||
		!strings.Contains(help, "org/package[/function]") || !strings.Contains(help, "--await") ||
		!strings.Contains(help, "--model org/model@release") ||
		!strings.Contains(help, "--model slot=org/model@release") ||
		!strings.Contains(help, "JSON file") || !strings.Contains(help, "--in request.json") ||
		strings.Contains(help, "--detach") || strings.Contains(help, "--wait") ||
		strings.Contains(help, "--force-rental") {
		t.Fatalf("run retained versioned target grammar [exit %d]\n%s", code, help)
	}
	if code, out := runCozy(t, root, "run", "cozy/example/function", "--force-rental"); code != 2 || !strings.Contains(out, "rentals.max_hourly_spend_usd") ||
		strings.Contains(out, "unknown flag") {
		t.Fatalf("hidden force-rental flag did not enter the ordinary rental budget gate [exit %d]\n%s", code, out)
	}
	if code, out := runCozy(t, root, "run", "cozy/example/function", "--stream"); code != 2 ||
		!strings.Contains(out, "--stream requires --await") {
		t.Fatalf("run stream did not require an explicit await [exit %d]\n%s", code, out)
	}
	if code, out := runCozy(t, root, "run", "cozy/example/function", "--timeout", "1s"); code != 2 ||
		!strings.Contains(out, "--timeout requires --await") {
		t.Fatalf("run timeout did not require an explicit await [exit %d]\n%s", code, out)
	}
	if code, help := runCozy(t, root, "package", "publish", "--help"); code != 0 ||
		strings.Contains(help, "--release") || strings.Contains(help, "--dir") ||
		strings.Contains(help, "<package>") {
		t.Fatalf("package publish retained caller-authored identity [exit %d]\n%s", code, help)
	}
	if code, help := runCozy(t, root, "package", "install", "--help"); code != 0 ||
		!strings.Contains(help, "--version") || !strings.Contains(help, "--no-model-download") ||
		!strings.Contains(help, "explicit directory") || strings.Contains(help, "--skip-cozytensors") ||
		strings.Contains(help, "--profile") || strings.Contains(help, "--major") ||
		strings.Contains(help, "--from") || strings.Contains(help, "--dir") ||
		strings.Contains(help, "--digest") || strings.Contains(help, "--allow-unsigned") ||
		strings.Contains(help, "--force") {
		t.Fatalf("package install exposed internal source/destination flags [exit %d]\n%s", code, help)
	}
	if code, help := runCozy(t, root, "package", "--help"); code != 0 ||
		strings.Contains(help, "recover") {
		t.Fatalf("package repair command became routine package help clutter [exit %d]\n%s", code, help)
	}
	if code, out := runCozy(t, root, "package", "install", "cozy/example@1.2.3"); code != 2 ||
		!strings.Contains(out, "--version 1.2.3") {
		t.Fatalf("inline install version did not point to --version [exit %d]\n%s", code, out)
	}
	if code, help := runCozy(t, root, "package", "yank", "--help"); code != 0 ||
		!strings.Contains(help, "--version") || strings.Contains(help, "unyank") {
		t.Fatalf("package yank grammar changed [exit %d]\n%s", code, help)
	}
	if code, out := runCozy(t, root, "package", "yank", "cozy/example", "--version", "1.2"); code != 2 || !strings.Contains(out, "N.M.P") {
		t.Fatalf("package yank admitted a non-N.M.P release [exit %d]\n%s", code, out)
	}
	project := t.TempDir()
	must(t, os.WriteFile(filepath.Join(project, "pyproject.toml"), []byte(`[project]
name = "proof-package"
version = "1.0.0"
`), 0o644))
	must(t, os.WriteFile(filepath.Join(project, "package.toml"), []byte(
		"[application]\nobject = \"proof_package:app\"\n"), 0o644))
	must(t, os.WriteFile(filepath.Join(project, "uv.lock"), []byte("version = 1\n"), 0o644))
	pack, problem := packagepublish.PrepareFrom(project)
	if problem != nil || pack.Name != "proof-package" || pack.Release != "1.0.0" {
		t.Fatalf("project identity without publisher metadata = %+v, %v", pack, problem)
	}
	pack.Close()
}

func TestPackageYankUsesPermanentReleaseEndpoint(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodDelete || r.URL.Path != "/v1/packages/proof/example/releases/1.2.3" {
			http.NotFound(w, r)
			return
		}
		if r.Header.Get("X-Tensorhub-Reason") != "cozy package yank proof/example@1.2.3" {
			t.Errorf("package yank audit text = %q", r.Header.Get("X-Tensorhub-Reason"))
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w,
			`{"state":"yanked","release":"1.2.3","changed":true,"yanked_at":"2026-08-30T12:34:56Z"}`)
	}))
	defer server.Close()
	code, out := runCozyDir(t, t.TempDir(), ".", []string{
		"TENSORHUB_URL=" + server.URL, "TENSORHUB_TOKEN=proof-token",
	}, "package", "yank", "proof/example", "--version", "1.2.3")
	if code != 0 || !strings.Contains(out, "package: proof/example") ||
		!strings.Contains(out, "release: 1.2.3") || !strings.Contains(out, "status:  yanked") {
		t.Fatalf("package yank result changed [exit %d]\n%s", code, out)
	}
}

func TestPackageInstallExplicitDirectoryGrammar(t *testing.T) {
	root := t.TempDir()
	parent := t.TempDir()
	child := filepath.Join(parent, "child")
	must(t, os.Mkdir(child, 0o755))
	missing := filepath.Join(parent, "missing")
	for _, path := range []string{".", "..", "./missing", "../missing", missing} {
		code, out := runCozyDir(t, root, child, nil, "package", "install", path, "--editable")
		if code != 1 || (!strings.Contains(out, "package source has no package.toml") &&
			!strings.Contains(out, "is not a directory")) || strings.Contains(out, "org/package") {
			t.Fatalf("explicit directory %q entered registry resolution [exit %d]\n%s", path, code, out)
		}
	}
}

func TestLocalPackageIdentityBindsSourceBytesWithoutBuilding(t *testing.T) {
	project := t.TempDir()
	writePublishProject(t, project, "local-identity", "1.0.0", nil, "", true)
	first, problem := packagepublish.PrepareFrom(project)
	fatal(t, problem)
	defer first.Close()
	firstDigest, files, bytes, problem := first.SourceIdentity()
	fatal(t, problem)
	replayDigest, replayFiles, replayBytes, problem := first.SourceIdentity()
	fatal(t, problem)
	if firstDigest != replayDigest || files != replayFiles || bytes != replayBytes {
		t.Fatalf("unchanged local build identity drifted: %s/%d/%d vs %s/%d/%d",
			firstDigest, files, bytes, replayDigest, replayFiles, replayBytes)
	}
	must(t, os.WriteFile(filepath.Join(project, "local_identity", "__init__.py"),
		[]byte("VALUE = 2\n"), 0o644))
	second, problem := packagepublish.PrepareFrom(project)
	fatal(t, problem)
	defer second.Close()
	secondDigest, _, _, problem := second.SourceIdentity()
	fatal(t, problem)
	if secondDigest == firstDigest {
		t.Fatalf("changed source retained local source identity %s", firstDigest)
	}
}

func TestRunAutoInstallsAMissingLocalPackage(t *testing.T) {
	root := t.TempDir()
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost || r.URL.Path != "/v1/packages/proof/missing/download" {
			http.NotFound(w, r)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusConflict)
		_, _ = io.WriteString(w, `{"error":{"code":"package_download.release_absent","message":"package has no non-yanked release","remedy":"publish a package release first"}}`)
	}))
	defer server.Close()
	t.Cleanup(func() { _, _ = runCozy(t, root, "down", "--all") })

	code, out := runCozyDir(t, root, ".", []string{"TENSORHUB_URL=" + server.URL},
		"run", "proof/missing/generate")
	if code != 1 || !strings.Contains(out, "is not installed; installing it from Tensorhub") ||
		!strings.Contains(out, "package has no non-yanked release") || strings.Contains(out, "is not installed on this host") {
		t.Fatalf("missing package did not enter automatic registry installation [exit %d]\n%s", code, out)
	}
}

func TestRentalCommandsSeparateInventoryFromCatalog(t *testing.T) {
	root := t.TempDir()
	store, problem := records.Open(filepath.Join(root, "records.db"))
	fatal(t, problem)
	problem = store.RecordRental(records.Rental{
		ID: "rnt-proof", MachineName: "studio", SKU: "h200",
		AcceleratorModel: "NVIDIA H200 SXM", HourlyRateUSDMicros: 6_000_000,
		State: "ready", Hub: "https://tensorhub.test",
	})
	fatal(t, problem)
	problem = store.RecordRental(records.Rental{
		ID: "rnt-failed", MachineName: "failed-gpu", SKU: "small",
		AcceleratorModel: "GPU", HourlyRateUSDMicros: 250_000,
		State: "failed", Hub: "https://tensorhub.test",
	})
	fatal(t, problem)
	store.Close()

	if code, out := runCozy(t, root, "rental"); code != 0 ||
		!strings.Contains(out, "studio") || !strings.Contains(out, "h200") ||
		!strings.Contains(out, "rentals: 2 remote machines running · $6.25/hour of $0.00/hour") {
		t.Fatalf("bare rental did not show current machines [exit %d]\n%s", code, out)
	}
	if code, out := runCozy(t, root, "rental", "list"); code != 2 ||
		!strings.Contains(out, "unexpected argument list") {
		t.Fatalf("retired rental list did not refuse [exit %d]\n%s", code, out)
	}

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet || r.URL.Path != "/v1/rental-skus" {
			http.NotFound(w, r)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `[{"name":"h200","accelerator_model":"NVIDIA H200 SXM","compute_capability":"9.0","vram_gb":141,"minimum_ram_per_gpu_gb":64,"price_usd_micros_per_hour":3990000}]`)
	}))
	defer server.Close()
	code, out := runCozyDir(t, root, ".", []string{"TENSORHUB_URL=" + server.URL}, "rental", "new")
	if code != 0 || !strings.Contains(out, "h200") || !strings.Contains(out, "NVIDIA H200 SXM") ||
		!strings.Contains(out, "sm_90") || !strings.Contains(out, "141 GB") || !strings.Contains(out, "$3.99/hr") {
		t.Fatalf("rental new did not show the SKU catalog [exit %d]\n%s", code, out)
	}
	if code, out := runCozyDir(t, root, ".", []string{"TENSORHUB_URL=" + server.URL},
		"rental", "new", "--name", "studio"); code != 2 ||
		!strings.Contains(out, "options require a GPU SKU") {
		t.Fatalf("catalog view silently accepted rental options [exit %d]\n%s", code, out)
	}
}

func TestRentalSpendConfigIsNestedAndExact(t *testing.T) {
	valid := t.TempDir()
	must(t, os.WriteFile(filepath.Join(valid, "config.yaml"), []byte(
		"port: 0\nrentals:\n  max_hourly_spend_usd: 4.125001\n"), 0o600))
	t.Cleanup(func() { _, _ = runCozy(t, valid, "down", "--all") })
	if code, out := runCozy(t, valid, "rental"); code != 0 ||
		!strings.Contains(out, "rentals: 0 remote machines running · $0.00/hour of $4.125001/hour") {
		t.Fatalf("nested rental ceiling was not exact [exit %d]\n%s", code, out)
	}

	overPrecise := t.TempDir()
	must(t, os.WriteFile(filepath.Join(overPrecise, "config.yaml"), []byte(
		"rentals:\n  max_hourly_spend_usd: 4.0000001\n"), 0o600))
	if code, out := runCozy(t, overPrecise, "rental"); code != 2 ||
		!strings.Contains(out, "at most six decimal places") {
		t.Fatalf("over-precise rental ceiling did not refuse [exit %d]\n%s", code, out)
	}

	deleted := t.TempDir()
	must(t, os.WriteFile(filepath.Join(deleted, "config.yaml"), []byte(
		"cloud:\n  max_hourly_spend_usd: 4\n"), 0o600))
	if code, out := runCozy(t, deleted, "rental"); code != 2 ||
		!strings.Contains(out, `unknown key "cloud"`) {
		t.Fatalf("deleted cloud config key did not refuse [exit %d]\n%s", code, out)
	}

	posts := 0
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method == http.MethodGet && r.URL.Path == "/v1/rental-skus" {
			_ = json.NewEncoder(w).Encode([]map[string]any{{
				"name": "gpu", "accelerator_model": "GPU", "compute_capability": "9.0",
				"vram_gb": 80, "minimum_ram_per_gpu_gb": 64,
				"price_usd_micros_per_hour": 750_000,
			}})
			return
		}
		if r.Method == http.MethodPost && r.URL.Path == "/v1/rentals" {
			posts++
		}
		http.NotFound(w, r)
	}))
	defer server.Close()
	manual := t.TempDir()
	must(t, os.WriteFile(filepath.Join(manual, "config.yaml"), []byte(
		"tensorhub_url: "+server.URL+"\nrentals:\n  max_hourly_spend_usd: 0.50\n"), 0o600))
	if code, out := runCozy(t, manual, "rental", "new", "gpu"); code == 0 ||
		!strings.Contains(out, "would exceed") || posts != 0 {
		t.Fatalf("manual rental crossed the fleet cap [exit %d posts=%d]\n%s", code, posts, out)
	}
}

func TestPackagePublishRefusesSilentlyOmittedPrivateFiles(t *testing.T) {
	project := t.TempDir()
	writePublishProject(t, project, "private-package", "1.0.0", nil, "", true)
	must(t, os.WriteFile(filepath.Join(project, ".env"), []byte("TOKEN=secret\n"), 0o600))
	pack, problem := packagepublish.PrepareFrom(project)
	if pack != nil {
		pack.Close()
	}
	if problem == nil || problem.Name != "package_source_file_refused" {
		t.Fatalf("private source file was silently skipped: %v", problem)
	}
}

func TestPackagePublishCommittedReplayStaysCompact(t *testing.T) {
	project := t.TempDir()
	writePublishProject(t, project, "replay-package", "1.0.0", nil, "", true)

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		switch {
		case r.Method == http.MethodGet && r.URL.Path == "/v1/accounts/current":
			_, _ = io.WriteString(w, `{"name":"proof"}`)
		case r.Method == http.MethodGet && strings.Contains(r.URL.Path, "/releases/1.0.0"):
			_, _ = io.WriteString(w, `{"release":{"release":"1.0.0","release_digest":"sha256:44136fa355b3678a1146ad16f7e8649e94fb4fc21fe77e8310c060f61caaff8a","package_descriptor_digest":"sha256:44136fa355b3678a1146ad16f7e8649e94fb4fc21fe77e8310c060f61caaff8a","package_descriptor_length":2,"created_at":"2026-08-31T00:00:00Z"},"document":{},"package_descriptor":{}}`)
		default:
			http.Error(w, "unexpected method", http.StatusMethodNotAllowed)
		}
	}))
	defer server.Close()
	code, out := runCozyDir(t, t.TempDir(), project,
		[]string{"TENSORHUB_URL=" + server.URL, "TENSORHUB_TOKEN=proof-token"}, "package", "publish")
	if code != 0 || !strings.Contains(out, "status:  already published") ||
		!strings.Contains(out, "Checking proof/replay-package@1.0.0...") ||
		!strings.Contains(out, "Release already published; no build or upload needed.") ||
		strings.Contains(out, "Building package wheel") ||
		strings.Contains(out, "Committing exact package release") ||
		strings.Contains(out, "qualification:") || strings.Contains(out, "changed:") {
		t.Fatalf("committed replay emitted verbose state [exit %d]\n%s", code, out)
	}
}

func TestPackagePublishBuildsBoundedLocalDependencyClosure(t *testing.T) {
	workspace := t.TempDir()
	projects := filepath.Join(workspace, "projects")
	must(t, os.MkdirAll(projects, 0o755))

	a := filepath.Join(projects, "local-a")
	b := filepath.Join(projects, "local-b")
	c := filepath.Join(projects, "local-c")
	writePublishProject(t, a, "local-a", "1.0.0",
		[]string{"local-b>=2,<3", "local-b[images]>=2,<3", "cozy-runtime>=0.0.3", "typing-extensions>=4"},
		"local-b = { path = \"../local-b\" }\n", true)
	writePublishProject(t, b, "local-b", "2.1.0", nil,
		"local-c = { path = \"../local-c\", editable = true }\nabsent-local = { path = \"../absent-local\" }\n", false)
	appendProjectTOML(t, b, "\n[project.optional-dependencies]\nimages = [\"local-c==3.0.0\"]\nunused = [\"absent-local==1\"]\n")
	writePublishProject(t, c, "local-c", "3.0.0", nil, "", false)
	lockPublishProject(t, a)

	pack, problem := preparePublishPackage(a)
	if problem != nil {
		t.Fatalf("%s: %s", problem.Message, problem.Remedy)
	}
	defer pack.Close()
	wheels := map[string]string{}
	for _, dependency := range pack.DependencyWheels {
		identity, problem := wheel.InspectIdentity(dependency.Path)
		fatal(t, problem)
		wheels[identity.Distribution] = identity.Version
	}
	if len(pack.DependencyWheels) != 4 || wheels["local-b"] != "2.1.0" ||
		wheels["local-c"] != "3.0.0" || wheels["cozy-runtime"] != "0.0.11" || //cozy:allow distribution assertion, not executable access
		wheels["typing-extensions"] == "" { //cozy:allow distribution assertion, not executable access
		t.Fatalf("package dependency closure omitted an ordinary library: %+v", pack.DependencyWheels)
	}
	for _, dependency := range pack.DependencyWheels {
		if !strings.HasSuffix(dependency.Filename, ".whl") {
			t.Fatalf("dependency did not retain a wheel basename: %+v", dependency)
		}
		if info, err := os.Stat(dependency.Path); err != nil || !info.Mode().IsRegular() {
			t.Fatalf("dependency wheel is not a staged regular file: %+v err=%v", dependency, err)
		}
	}

	writePublishProject(t, a, "local-a", "1.0.0", []string{"local-b>=3"},
		"local-b = { path = \"../local-b\" }\n", true)
	if incompatible, problem := preparePublishPackage(a); problem == nil || problem.Name != "local_dependency_version_incompatible" {
		if incompatible != nil {
			incompatible.Close()
		}
		t.Fatalf("local wheel outside the parent PEP 440 requirement did not refuse: %v", problem)
	}

	writePublishProject(t, a, "local-a", "1.0.0",
		[]string{"local-b[images]>=2,<3"},
		"local-b = { path = \"../local-b\" }\n", true)
	writePublishProject(t, c, "local-c", "3.0.0", []string{"local-a==1.0.0"},
		"local-a = { path = \"../local-a\" }\n", false)
	if cycle, problem := preparePublishPackage(a); problem == nil || problem.Name != "local_dependency_cycle" {
		if cycle != nil {
			cycle.Close()
		}
		t.Fatalf("A->B->C->A did not refuse as a local dependency cycle: %v", problem)
	}

	conflictRoot := filepath.Join(t.TempDir(), "root")
	x1 := filepath.Join(filepath.Dir(conflictRoot), "x1")
	x2 := filepath.Join(filepath.Dir(conflictRoot), "x2")
	left := filepath.Join(filepath.Dir(conflictRoot), "left")
	right := filepath.Join(filepath.Dir(conflictRoot), "right")
	writePublishProject(t, conflictRoot, "conflict-root", "1.0.0",
		[]string{"left==1", "right==1"},
		"left = { path = \"../left\" }\nright = { path = \"../right\" }\n", true)
	writePublishProject(t, left, "left", "1", []string{"shared==1"},
		"shared = { path = \"../x1\" }\n", false)
	writePublishProject(t, right, "right", "1", []string{"shared==2"},
		"shared = { path = \"../x2\" }\n", false)
	writePublishProject(t, x1, "shared", "1", nil, "", false)
	writePublishProject(t, x2, "shared", "2", nil, "", false)
	if conflict, problem := preparePublishPackage(conflictRoot); problem == nil || problem.Name != "local_dependency_duplicate" {
		if conflict != nil {
			conflict.Close()
		}
		t.Fatalf("different sources for one normalized name did not refuse: %v", problem)
	}

	countRoot := filepath.Join(t.TempDir(), "root")
	writePublishProject(t, countRoot, "count-root", "1", []string{"count-01==1"},
		"count-01 = { path = \"../count-01\" }\n", true)
	countParent := filepath.Dir(countRoot)
	for i := 1; i <= packagepublish.MaxDependencyWheels+1; i++ {
		name := fmt.Sprintf("count-%02d", i)
		var dependencies []string
		var sources string
		if i <= packagepublish.MaxDependencyWheels {
			next := fmt.Sprintf("count-%02d", i+1)
			dependencies = []string{next + "==1"}
			sources = fmt.Sprintf("%s = { path = \"../%s\" }\n", next, next)
		}
		writePublishProject(t, filepath.Join(countParent, name), name, "1", dependencies, sources, false)
	}
	if counted, problem := preparePublishPackage(countRoot); problem == nil || problem.Name != "local_dependency_count_exceeded" {
		if counted != nil {
			counted.Close()
		}
		t.Fatalf("dependency closure above %d wheels was not refused: %v",
			packagepublish.MaxDependencyWheels, problem)
	}

	directRoot := filepath.Join(t.TempDir(), "direct")
	writePublishProject(t, directRoot, "direct-root", "1", []string{"foreign @ https://example.invalid/foreign.whl"}, "", true)
	if direct, problem := preparePublishPackage(directRoot); problem == nil || problem.Name != "project_dependency_direct_url_unsupported" {
		if direct != nil {
			direct.Close()
		}
		t.Fatalf("direct URL dependency did not refuse: %v", problem)
	}

	gitRoot := filepath.Join(t.TempDir(), "git")
	writePublishProject(t, gitRoot, "git-root", "1", []string{"foreign==1"},
		"foreign = { git = \"https://example.invalid/foreign.git\" }\n", true)
	if git, problem := preparePublishPackage(gitRoot); problem == nil || problem.Name != "project_dependency_source_unsupported" {
		if git != nil {
			git.Close()
		}
		t.Fatalf("VCS dependency source did not refuse: %v", problem)
	}
}

func TestPackagePublishPrunesRuntimeOwnedTensorFS(t *testing.T) {
	parent := t.TempDir()
	project := filepath.Join(parent, "modeled")
	runtimeFixture := filepath.Join(parent, "runtime-fixture")
	tensorFSFixture := filepath.Join(parent, "tensorfs")
	writeRuntimeFixture(t, runtimeFixture)
	writePublishProject(t, tensorFSFixture, "tensorfs", "0.0.6", nil, "", false)
	appendProjectTOML(t, runtimeFixture,
		"\n[project.optional-dependencies]\nmodel-execution = [\"tensorfs==0.0.6\"]\n")
	writePublishProject(t, project, "modeled", "1.0.0",
		[]string{"cozy-runtime[model-execution]==0.0.11"},
		fmt.Sprintf("cozy-runtime = { path = %q, editable = true }\n"+
			"tensorfs = { path = %q, editable = true }\n", runtimeFixture, tensorFSFixture), true)
	appendProjectTOML(t, project,
		"\n[dependency-groups]\ndev = [\"tensorfs==0.0.6\"]\n")
	lockPublishProject(t, project)

	pack, problem := preparePublishPackage(project)
	if problem != nil {
		t.Fatalf("%s: %s", problem.Message, problem.Remedy)
	}
	defer pack.Close()
	if len(pack.DependencyWheels) != 1 {
		t.Fatalf("modeled package emitted %d dependency wheels, want only Runtime", len(pack.DependencyWheels))
	}
	identity, problem := wheel.InspectIdentity(pack.DependencyWheels[0].Path)
	fatal(t, problem)
	if identity.Distribution != "cozy-runtime" { //cozy:allow distribution assertion, not executable access
		t.Fatalf("production overlay retained base-owned TensorFS: %+v", identity)
	}
}

func TestPackagePublishDownloadsLockedRegistryDependency(t *testing.T) {
	project := copyRegistryDependencyFixture(t)
	pack, problem := preparePublishPackage(project)
	if problem != nil {
		t.Fatalf("%s: %s", problem.Message, problem.Remedy)
	}
	defer pack.Close()
	if len(pack.DependencyWheels) != 2 {
		t.Fatalf("registry dependency wheel count = %d, want runtime plus humanize", len(pack.DependencyWheels))
	}
	var dependency wheel.Identity
	var dependencyPath string
	for _, candidate := range pack.DependencyWheels {
		identity, inspectProblem := wheel.InspectIdentity(candidate.Path)
		fatal(t, inspectProblem)
		if identity.Distribution == "humanize" {
			dependency, dependencyPath = identity, candidate.Path
		}
	}
	if dependency.Distribution != "humanize" || dependency.Version != "4.13.0" ||
		!strings.HasSuffix(dependency.Filename, "-py3-none-any.whl") {
		t.Fatalf("registry dependency identity = %+v", dependency)
	}
	metadata := projectWheelMetadata(t, pack.Wheel)
	for _, requirement := range []string{
		"Requires-Dist: numpy==2.5.2", "Requires-Dist: pillow==12.3.0", "Requires-Dist: torch>=2.13,<3",
	} {
		if !strings.Contains(metadata, requirement) {
			t.Fatalf("project wheel lost base compatibility requirement %q:\n%s", requirement, metadata)
		}
	}

	venv := filepath.Join(t.TempDir(), "venv")
	python, err := exec.LookPath("python3")
	must(t, err)
	command := exec.Command("uv", "venv", "--no-project", "--python", python, venv)
	if output, err := command.CombinedOutput(); err != nil {
		t.Fatalf("create offline-install proof venv: %v\n%s", err, output)
	}
	venvPython := filepath.Join(venv, "bin", "python")
	command = exec.Command("uv", "pip", "install", "--python", venvPython, "--offline", "--no-index",
		"--no-deps", pack.Wheel, dependencyPath)
	if output, err := command.CombinedOutput(); err != nil {
		t.Fatalf("offline exact-wheel install: %v\n%s", err, output)
	}
	command = exec.Command(venvPython, "-I", "-c",
		"from registry_dependency_proof import marco; assert marco('marco') == 'polo'")
	if output, err := command.CombinedOutput(); err != nil {
		t.Fatalf("offline-installed package callable: %v\n%s", err, output)
	}
}

func projectWheelMetadata(t *testing.T, path string) string {
	t.Helper()
	archive, err := zip.OpenReader(path)
	must(t, err)
	defer archive.Close()
	for _, member := range archive.File {
		if !strings.HasSuffix(member.Name, ".dist-info/METADATA") {
			continue
		}
		file, err := member.Open()
		must(t, err)
		raw, err := io.ReadAll(file)
		must(t, err)
		must(t, file.Close())
		return string(raw)
	}
	t.Fatal("project wheel has no METADATA")
	return ""
}

func TestPackagePublishRefusesUnsupportedRegistryDependencyLocks(t *testing.T) {
	cases := []struct {
		name, old, replacement, code string
	}{
		{
			name:        "changed hash",
			old:         "b810820b31891813b1673e8fec7f1ed3312061eab2f26e3fa192c393d11ed25f",
			replacement: "a810820b31891813b1673e8fec7f1ed3312061eab2f26e3fa192c393d11ed25f",
			code:        "registry_dependency_identity_mismatch",
		},
		{
			name:        "foreign origin",
			old:         "https://files.pythonhosted.org/packages/1e/c7/316e7ca04d26695ef0635dc81683d628350810eb8e9b2299fc08ba49f366/",
			replacement: "https://packages.example.invalid/",
			code:        "registry_dependency_origin_refused",
		},
		{
			name: "alternate index",
			old:  "name = \"humanize\"\nversion = \"4.13.0\"\nsource = { registry = \"https://pypi.org/simple\" }",
			replacement: "name = \"humanize\"\nversion = \"4.13.0\"\n" +
				"source = { registry = \"https://packages.example.invalid/simple\" }",
			code: "registry_dependency_index_refused",
		},
		{
			name: "native only", old: "humanize-4.13.0-py3-none-any.whl",
			replacement: "humanize-4.13.0-cp312-cp312-manylinux_2_28_x86_64.whl",
			code:        "registry_dependency_native_only",
		},
		{
			name:        "source only",
			old:         "wheels = [\n    { url = \"https://files.pythonhosted.org/packages/1e/c7/316e7ca04d26695ef0635dc81683d628350810eb8e9b2299fc08ba49f366/humanize-4.13.0-py3-none-any.whl\", hash = \"sha256:b810820b31891813b1673e8fec7f1ed3312061eab2f26e3fa192c393d11ed25f\", size = 128869, upload-time = \"2025-08-25T09:39:18.54Z\" },\n]\n",
			replacement: "", code: "registry_dependency_source_only",
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			project := copyRegistryDependencyFixture(t)
			mutateFixtureFile(t, filepath.Join(project, "uv.lock"), tc.old, tc.replacement)
			pack, problem := preparePublishPackage(project)
			if pack != nil {
				pack.Close()
			}
			if problem == nil || problem.Name != tc.code {
				t.Fatalf("registry dependency refusal = %v, want %s", problem, tc.code)
			}
		})
	}

	t.Run("lock drift", func(t *testing.T) {
		project := copyRegistryDependencyFixture(t)
		mutateFixtureFile(t, filepath.Join(project, "pyproject.toml"),
			"humanize==4.13.0", "humanize==4.12.0")
		pack, problem := preparePublishPackage(project)
		if pack != nil {
			pack.Close()
		}
		if problem == nil || problem.Name != "registry_dependency_lock_drift" {
			t.Fatalf("lock drift refusal = %v", problem)
		}
	})
}

func copyRegistryDependencyFixture(t *testing.T) string {
	t.Helper()
	target := t.TempDir()
	must(t, os.CopyFS(target, os.DirFS(filepath.Join("testdata", "registry-dependency"))))
	runtimeFixture := target + "-cozy-runtime"
	writeRuntimeFixture(t, runtimeFixture)
	project := filepath.Join(target, "pyproject.toml")
	raw, err := os.ReadFile(project)
	must(t, err)
	raw = []byte(strings.ReplaceAll(string(raw), "__COZY_RUNTIME_FIXTURE__", runtimeFixture))
	must(t, os.WriteFile(project, raw, 0o644))
	lockPublishProject(t, target)
	return target
}

func mutateFixtureFile(t *testing.T, path, old, replacement string) {
	t.Helper()
	raw, err := os.ReadFile(path)
	must(t, err)
	if strings.Count(string(raw), old) != 1 {
		t.Fatalf("fixture mutation target occurs %d times in %s", strings.Count(string(raw), old), path)
	}
	must(t, os.WriteFile(path, []byte(strings.Replace(string(raw), old, replacement, 1)), 0o644))
}

func preparePublishPackage(root string) (*packagepublish.Package, *exit.Error) {
	pack, problem := packagepublish.PrepareFrom(root)
	if problem != nil {
		return nil, problem
	}
	if problem := pack.Build(context.Background()); problem != nil {
		pack.Close()
		return pack, problem
	}
	return pack, nil
}

func lockPublishProject(t *testing.T, root string) {
	t.Helper()
	_ = os.Remove(filepath.Join(root, "uv.lock"))
	command := exec.Command("uv", "lock", "--directory", root)
	if output, err := command.CombinedOutput(); err != nil {
		t.Fatalf("lock publish fixture: %v\n%s", err, output)
	}
	if _, err := os.Stat(filepath.Join(root, "uv.lock")); err == nil {
		return
	}
	t.Fatal("uv lock produced no fixture-local lock file")
}

func writePublishProject(t *testing.T, root, name, version string, dependencies []string, sources string, publishable bool) {
	t.Helper()
	if dependencies == nil {
		dependencies = []string{}
	}
	if publishable {
		hasRuntime := false
		for _, dependency := range dependencies {
			hasRuntime = hasRuntime || strings.HasPrefix(dependency, "cozy-runtime") //cozy:allow distribution fixture, not executable access
		}
		if !hasRuntime {
			dependencies = append(dependencies, "cozy-runtime==0.0.11")
		}
		if !strings.Contains(sources, "cozy-runtime =") {
			runtimeFixture := root + "-cozy-runtime"
			writeRuntimeFixture(t, runtimeFixture)
			sources += fmt.Sprintf("cozy-runtime = { path = %q, editable = true }\n", runtimeFixture)
		}
	}
	must(t, os.MkdirAll(filepath.Join(root, strings.ReplaceAll(name, "-", "_")), 0o755))
	module := strings.ReplaceAll(name, "-", "_")
	body := "VALUE = 1\n"
	must(t, os.WriteFile(filepath.Join(root, module, "__init__.py"), []byte(body), 0o644))
	dependencyJSON, err := json.Marshal(dependencies)
	must(t, err)
	document := fmt.Sprintf(`[build-system]
requires = ["uv_build>=0.12.7,<0.13"]
build-backend = "uv_build"

[project]
name = %q
version = %q
dependencies = %s

[tool.uv.build-backend]
module-root = ""
`, name, version, dependencyJSON)
	if sources != "" {
		document += "\n[tool.uv.sources]\n" + sources
	}
	if publishable {
		must(t, os.WriteFile(filepath.Join(root, "package.toml"), []byte(
			"[application]\nobject = \""+module+":app\"\n"), 0o644))
		must(t, os.WriteFile(filepath.Join(root, "uv.lock"), []byte("version = 1\n"), 0o644))
	}
	must(t, os.WriteFile(filepath.Join(root, "pyproject.toml"), []byte(document), 0o644))
}

func writeRuntimeFixture(t *testing.T, root string) {
	t.Helper()
	if _, err := os.Stat(filepath.Join(root, "pyproject.toml")); err == nil {
		return
	}
	must(t, os.MkdirAll(filepath.Join(root, "cozy_runtime"), 0o755))
	must(t, os.WriteFile(filepath.Join(root, "pyproject.toml"), []byte(`[build-system]
requires = ["uv_build>=0.12.7,<0.13"]
build-backend = "uv_build"
[project]
name = "cozy-runtime" # //cozy:allow distribution fixture, not executable access
version = "0.0.11"
[project.scripts]
cozy-runtime = "cozy_runtime:main" # //cozy:allow isolated fake console script
[tool.uv.build-backend]
module-root = ""
`), 0o644))
	must(t, os.WriteFile(filepath.Join(root, "cozy_runtime", "__init__.py"), []byte(`import json, pathlib, sys, tomllib
def main():
    root = pathlib.Path(sys.argv[sys.argv.index("--dir") + 1])
    application = tomllib.loads((root / "package.toml").read_text())["application"]["object"]
    print(json.dumps({"application": application, "entrypoints": [{"name": "proof", "request": {"fields": []}, "result": {"fields": []}}], "format": "cozy.package.descriptor/1", "jobs": [], "model_productions": []}, separators=(",", ":"), sort_keys=True))
`), 0o644))
}

func appendProjectTOML(t *testing.T, root, document string) {
	t.Helper()
	file, err := os.OpenFile(filepath.Join(root, "pyproject.toml"), os.O_APPEND|os.O_WRONLY, 0)
	must(t, err)
	_, err = file.WriteString(document)
	must(t, err)
	must(t, file.Close())
}

func TestDaemonWebLifecycle(t *testing.T) {
	root := filepath.Join(os.TempDir(), "cozy-product-test", "daemon-web")
	must(t, os.RemoveAll(root))
	must(t, os.MkdirAll(root, 0o755))
	t.Cleanup(func() { _, _ = runCozy(t, root, "down", "--all") })

	code, help := runCozy(t, root)
	helpCode, explicitHelp := runCozy(t, root, "help")
	if helpCode != 0 || explicitHelp != help {
		t.Fatalf("cozy and cozy help rendered different root help [exit %d]\n%s", helpCode, explicitHelp)
	}
	for _, section := range []string{"Packages", "Models", "Authentication", "Runs", "Rentals", "Lifecycle"} {
		if code != 0 || strings.Count(help, "\n"+section+"\n") != 1 {
			t.Fatalf("bare cozy did not render one %q section [exit %d]\n%s", section, code, help)
		}
	}
	if strings.Contains(help, "\nResources\n") || strings.Contains(help, "\nWork\n") {
		t.Fatalf("bare cozy retained a combined command section\n%s", help)
	}
	if strings.Count(help, "run <org/package/function> [input]") != 1 || strings.Contains(help, "Primary command") {
		t.Fatalf("bare cozy did not state the primary run command in Runs\n%s", help)
	}
	for _, want := range []string{
		"Usage: cozy", "package install", "model download", "auth login", "run cancel",
		"rental new", "up", "down", "unload",
	} {
		if code != 0 || !strings.Contains(help, want) {
			t.Fatalf("bare cozy omitted %q [exit %d]\n%s", want, code, help)
		}
	}
	for _, retired := range []string{" exit ", "workflow", "video", "job submit"} {
		if strings.Contains(help, retired) {
			t.Fatalf("bare cozy retained %q\n%s", retired, help)
		}
	}
	if code, out := runCozy(t, root, "exit"); code != 2 || !strings.Contains(out, "unexpected argument exit") {
		t.Fatalf("retired exit command did not refuse [exit %d]\n%s", code, out)
	}

	env := childEnv(t, root)
	results := make(chan cozyResult, 2)
	for range 2 {
		go func() { results <- runCozyEnv(env, "up", "--json", "--full") }()
	}
	first, second := <-results, <-results
	if first.code != 0 || second.code != 0 {
		t.Fatalf("concurrent up did not converge: first=[%d] %s second=[%d] %s",
			first.code, first.output, second.code, second.output)
	}
	up := first.output
	var upDocument struct {
		URL     string `json:"url"`
		PID     int    `json:"pid"`
		Changed bool   `json:"changed"`
	}
	var secondDocument struct {
		URL     string `json:"url"`
		PID     int    `json:"pid"`
		Changed bool   `json:"changed"`
	}
	if err := json.Unmarshal([]byte(up), &upDocument); err != nil ||
		upDocument.URL == "" || upDocument.PID == 0 {
		t.Fatalf("up did not start one daemon: %v\n%s", err, up)
	}
	if err := json.Unmarshal([]byte(second.output), &secondDocument); err != nil ||
		upDocument.PID != secondDocument.PID || upDocument.URL != secondDocument.URL {
		t.Fatalf("concurrent up returned different daemon generations: %v\n%s\n%s",
			err, first.output, second.output)
	}
	if upDocument.Changed == secondDocument.Changed {
		t.Fatalf("concurrent up did not report one winner: first changed=%v second changed=%v\n%s\n%s",
			upDocument.Changed, secondDocument.Changed, first.output, second.output)
	}
	if strings.Contains(strings.ToLower(first.output+second.output), "lock") {
		t.Fatalf("up exposed its internal singleton mechanism\n%s\n%s", first.output, second.output)
	}
	url := upDocument.URL
	client := &http.Client{Timeout: 5 * time.Second}
	response, err := client.Get(url)
	if err != nil {
		t.Fatalf("up returned an unreachable web UI %q: %v", url, err)
	}
	page, readErr := io.ReadAll(response.Body)
	response.Body.Close()
	if readErr != nil || response.StatusCode != http.StatusOK ||
		!strings.Contains(string(page), "local generative workspace is running") {
		t.Fatalf("web stub is not ready at up return: status=%d err=%v\n%s",
			response.StatusCode, readErr, page)
	}
	if _, err := os.Stat(filepath.Join(root, "daemon.log")); !os.IsNotExist(err) {
		t.Fatalf("up created a persistent daemon log: %v", err)
	}
	if code, out := runCozy(t, root, "up", "--json", "--full"); code != 0 {
		t.Fatalf("repeated up failed [exit %d]\n%s", code, out)
	} else {
		var repeated struct {
			URL     string `json:"url"`
			PID     int    `json:"pid"`
			Changed bool   `json:"changed"`
		}
		if err := json.Unmarshal([]byte(out), &repeated); err != nil ||
			repeated.Changed || repeated.URL != url || repeated.PID != upDocument.PID {
			t.Fatalf("repeated up did not return the same healthy daemon with changed=false: %v\n%s",
				err, out)
		}
	}
	if code, out := runCozy(t, root, "down"); code != 0 ||
		!strings.Contains(out, "daemon:") || !strings.Contains(out, "stopped") {
		t.Fatalf("down [exit %d]\n%s", code, out)
	}
	if code, out := runCozy(t, root, "run", "list", "--json"); code != 0 ||
		!strings.Contains(out, `"invocations":[]`) {
		t.Fatalf("stateful command did not auto-start the daemon [exit %d]\n%s", code, out)
	}
	if code, out := runCozy(t, root, "unload"); code != 0 || !strings.Contains(out, "No workers found.") {
		t.Fatalf("unload [exit %d]\n%s", code, out)
	}
	if code, out := runCozy(t, root, "up"); code != 0 || !strings.Contains(out, "changed: false") {
		t.Fatalf("unload stopped the daemon [exit %d]\n%s", code, out)
	}
	if code, out := runCozy(t, root, "down"); code != 0 ||
		!strings.Contains(out, "daemon:") || !strings.Contains(out, "stopped") {
		t.Fatalf("final down [exit %d]\n%s", code, out)
	}
}

func TestUpReportsLocalGPUCompatibility(t *testing.T) {
	root := filepath.Join(os.TempDir(), "cozy-product-test", "up-gpu-compatibility")
	must(t, os.RemoveAll(root))
	toolDir := t.TempDir()
	nvidiaSMI := filepath.Join(toolDir, "nvidia-smi")
	must(t, os.WriteFile(nvidiaSMI, []byte(`#!/bin/sh
case "$1" in
  --query-gpu=*) printf '0,"NVIDIA, Test GPU",12288,24576,580.126.20,8.9\n' ;;
  *) printf '| NVIDIA-SMI 580.126.20 Driver Version: 580.126.20 CUDA Version: 13.0 |\n' ;;
esac
`), 0o755))
	env := childEnv(t, root, "PATH="+toolDir)
	t.Cleanup(func() { _ = runCozyEnv(env, "down", "--all") })

	result := runCozyEnv(env, "up", "--json", "--full")
	if result.code != 0 {
		t.Fatalf("up with NVIDIA GPU failed [exit %d]\n%s", result.code, result.output)
	}
	var document struct {
		GPUs       []string `json:"gpus"`
		GPUCount   int      `json:"gpu_count"`
		GPUDetails []struct {
			Model             string `json:"model"`
			VRAMFreeBytes     int64  `json:"vram_free_bytes"`
			VRAMTotalBytes    int64  `json:"vram_total_bytes"`
			DriverVersion     string `json:"driver_version"`
			DriverCUDAVersion string `json:"driver_cuda_version"`
			ComputeCapability string `json:"compute_capability"`
			SM                string `json:"sm"`
		} `json:"gpu_details"`
	}
	if err := json.Unmarshal([]byte(result.output), &document); err != nil ||
		document.GPUCount != 1 || len(document.GPUs) != 1 || len(document.GPUDetails) != 1 {
		t.Fatalf("up did not return one typed GPU: %v\n%s", err, result.output)
	}
	gpu := document.GPUDetails[0]
	if gpu.Model != "NVIDIA, Test GPU" || gpu.VRAMFreeBytes != 12<<30 ||
		gpu.VRAMTotalBytes != 24<<30 || gpu.DriverVersion != "580.126.20" ||
		gpu.DriverCUDAVersion != "13.0" || gpu.ComputeCapability != "8.9" || gpu.SM != "sm_89" {
		t.Fatalf("up GPU compatibility facts drifted: %+v", gpu)
	}
	for _, want := range []string{"NVIDIA, Test GPU", "12.0 / 24.0 GiB free", "driver 580.126.20", "driver CUDA 13.0", "sm_89"} {
		if !strings.Contains(document.GPUs[0], want) {
			t.Fatalf("up GPU summary omitted %q: %s", want, document.GPUs[0])
		}
	}
}

func TestUpDoesNotRequireLocalRuntime(t *testing.T) {
	root := filepath.Join(os.TempDir(), "cozy-product-test", "up-without-local-runtime")
	must(t, os.RemoveAll(root))
	env := childEnv(t, root, "PATH="+t.TempDir())
	t.Cleanup(func() { _ = runCozyEnv(env, "down", "--all") })

	result := runCozyEnv(env, "up", "--json", "--full")
	if result.code != 0 {
		t.Fatalf("remote-capable daemon required a local Runtime [exit %d]\n%s",
			result.code, result.output)
	}
}

func TestRentalCatalog(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet || r.URL.Path != "/v1/rental-skus" {
			http.NotFound(w, r)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `[{"name":"h200","accelerator_model":"NVIDIA H200","compute_capability":"9.0","vram_gb":141,"minimum_ram_per_gpu_gb":64,"price_usd_micros_per_hour":6000000},{"name":"rtx-4090","accelerator_model":"NVIDIA GeForce RTX 4090","compute_capability":"8.9","vram_gb":24,"minimum_ram_per_gpu_gb":32,"price_usd_micros_per_hour":1250000},{"name":"cpu","accelerator_model":"CPU","compute_capability":"","vram_gb":0,"minimum_ram_per_gpu_gb":0,"price_usd_micros_per_hour":70000}]`)
	}))
	defer server.Close()
	root := filepath.Join(os.TempDir(), "cozy-product-test", "rental-gpu-catalog")
	must(t, os.RemoveAll(root))
	env := childEnv(t, root, "TENSORHUB_URL="+server.URL)

	result := runCozyEnv(env, "rental", "new", "--json")
	if result.code != 0 {
		t.Fatalf("rental catalog failed [exit %d]\n%s", result.code, result.output)
	}
	var document struct {
		GPUs []map[string]string `json:"gpus"`
	}
	if err := json.Unmarshal([]byte(result.output), &document); err != nil || len(document.GPUs) != 3 {
		t.Fatalf("rental catalog was not a three-row machine list: %v\n%s", err, result.output)
	}
	if document.GPUs[0]["name"] != "h200" || document.GPUs[0]["model"] != "NVIDIA H200" ||
		document.GPUs[0]["compute"] != "sm_90" || document.GPUs[0]["vram"] != "141 GB" || document.GPUs[0]["price"] != "$6/hr" ||
		document.GPUs[1]["price"] != "$1.25/hr" || document.GPUs[2]["name"] != "cpu" ||
		document.GPUs[2]["model"] != "CPU" || document.GPUs[2]["price"] != "$0.07/hr" {
		t.Fatalf("rental catalog values drifted: %#v", document.GPUs)
	}
	if result := runCozyEnv(env, "rental", "new", "h200", "extra"); result.code != 2 || !strings.Contains(result.output, "unexpected argument") {
		t.Fatalf("generic rental accepted a second positional argument [exit %d]\n%s", result.code, result.output)
	}
}

func TestDefaultWebPortPreferenceAndFallback(t *testing.T) {
	held, err := net.Listen("tcp4", "127.0.0.1:8818") //cozy:allow product proof occupies the preferred loopback port to exercise fallback
	if err != nil {
		t.Skipf("localhost:8818 is already occupied outside this product test: %v", err)
	}

	fallbackRoot := filepath.Join(os.TempDir(), "cozy-product-test", "default-port-fallback")
	must(t, os.RemoveAll(fallbackRoot))
	code, out := runCozy(t, fallbackRoot, "up", "--json", "--full")
	if code != 0 {
		held.Close()
		t.Fatalf("up with occupied preferred port failed [exit %d]\n%s", code, out)
	}
	var fallback struct {
		URL string `json:"url"`
	}
	if err := json.Unmarshal([]byte(out), &fallback); err != nil || fallback.URL == "" ||
		strings.Contains(fallback.URL, ":8818") {
		held.Close()
		t.Fatalf("occupied 8818 did not select a fallback: %v\n%s", err, out)
	}
	if code, out := runCozy(t, fallbackRoot, "down"); code != 0 {
		held.Close()
		t.Fatalf("fallback daemon down [exit %d]\n%s", code, out)
	}
	must(t, held.Close())

	preferredRoot := filepath.Join(os.TempDir(), "cozy-product-test", "default-port-preferred")
	must(t, os.RemoveAll(preferredRoot))
	t.Cleanup(func() { _, _ = runCozy(t, preferredRoot, "down", "--all") })
	code, out = runCozy(t, preferredRoot, "up", "--json", "--full")
	if code != 0 {
		t.Fatalf("up on available preferred port failed [exit %d]\n%s", code, out)
	}
	var preferred struct {
		URL string `json:"url"`
	}
	if err := json.Unmarshal([]byte(out), &preferred); err != nil ||
		preferred.URL != "http://127.0.0.1:8818/" {
		t.Fatalf("available default did not bind 8818: %v\n%s", err, out)
	}
}

func TestDaemonStartupDiagnostic(t *testing.T) {
	root := filepath.Join(os.TempDir(), "cozy-product-test", "daemon-startup-diagnostic")
	must(t, os.RemoveAll(root))
	must(t, os.MkdirAll(root, 0o755))
	t.Cleanup(func() { _, _ = runCozy(t, root, "down", "--all") })

	held, err := net.Listen("tcp4", "127.0.0.1:0") //cozy:allow product proof occupies one loopback port to exercise the real startup refusal
	must(t, err)
	port := held.Addr().(*net.TCPAddr).Port
	must(t, os.WriteFile(filepath.Join(root, "config.yaml"),
		[]byte(fmt.Sprintf("port: %d\n", port)), 0o600))

	began := time.Now()
	code, out := runCozy(t, root, "up", "--json")
	if code != 1 {
		t.Fatalf("failed daemon startup exited %d, want operational 1\n%s", code, out)
	}
	if took := time.Since(began); took > 5*time.Second {
		t.Fatalf("failed daemon startup waited %s instead of relaying the exited child", took)
	}
	var document struct {
		Error struct {
			Code    string `json:"code"`
			Message string `json:"message"`
		} `json:"error"`
	}
	if err := json.Unmarshal([]byte(out), &document); err != nil ||
		document.Error.Code != "daemon_startup_failed" ||
		!strings.Contains(document.Error.Message, fmt.Sprintf("127.0.0.1:%d", port)) ||
		!strings.Contains(document.Error.Message, "held by another process") {
		t.Fatalf("startup failure did not relay its child diagnostic: %v\n%s", err, out)
	}
	if strings.Contains(strings.ToLower(out), "lock") {
		t.Fatalf("startup failure exposed its internal singleton mechanism\n%s", out)
	}
	if len(out) > maxDaemonDiagnosticOutput {
		t.Fatalf("startup diagnostic is unbounded: %d bytes", len(out))
	}
	if _, err := os.Stat(filepath.Join(root, "daemon.log")); !os.IsNotExist(err) {
		t.Fatalf("failed startup created a persistent daemon log: %v", err)
	}

	must(t, held.Close())
	if code, out := runCozy(t, root, "up", "--json"); code != 0 {
		t.Fatalf("root did not recover after the failed child [exit %d]\n%s", code, out)
	}
}

const maxDaemonDiagnosticOutput = 18 << 10

func TestDevelopmentInstallRefreshesBeforeInvocation(t *testing.T) {
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
		t.Fatalf("body edit did not advance the editable generation: %#v -> %#v", first, second)
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
		t.Fatalf("failed refresh displaced the last good generation: %#v -> %#v", third, failed)
	}
	must(t, os.WriteFile(pyproject, goodMetadata, 0o644))
	code, out = runCozy(t, root, "run", localWeightlessRef+"/tile",
		"size=32", "seed=7", "--json", "--await")
	if code != 0 || !strings.Contains(out, `"revision":"second"`) {
		t.Fatalf("restored source did not reuse the last good generation [exit %d]\n%s", code, out)
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

func TestModeledDevelopmentInstallRefreshesAndKeepsLastGoodSelection(t *testing.T) {
	root := filepath.Join(os.TempDir(), "cozy-product-test", "modeled-editable-refresh")
	must(t, os.RemoveAll(root))
	must(t, os.MkdirAll(root, 0o755))
	t.Cleanup(func() {
		_, _ = runCozy(t, root, "down", "--all")
		_ = os.RemoveAll(root)
	})

	project := modeledDevelopmentProject(t, root)
	code, out := runCozy(t, root, "package", "install", project, "--editable", "--json", "--full")
	if code != 0 || !strings.Contains(out, `"package":"local/modeled-development-package"`) {
		t.Fatalf("modeled editable install failed [exit %d]\n%s", code, out)
	}
	code, out = runCozy(t, root, "run", "local/modeled-development-package/render",
		"value=7", "--json", "--await")
	if code != 0 || !strings.Contains(out, `"value":8`) {
		t.Fatalf("modeled editable invocation failed [exit %d]\n%s\n%s",
			code, out, productWorkerLogs(root))
	}
	first := activeInstall(t, root, "local/modeled-development-package")

	source := filepath.Join(project, "src", "modeled_development_package", "__init__.py")
	body, err := os.ReadFile(source)
	must(t, err)
	body = []byte(strings.Replace(string(body), "return Result(payload.value + model.first())",
		"return Result(payload.value + model.first() + 1)", 1))
	must(t, os.WriteFile(source, body, 0o644))
	code, out = runCozy(t, root, "run", "local/modeled-development-package/render",
		"value=7", "--json", "--await")
	if code != 0 || !strings.Contains(out, `"value":9`) {
		t.Fatalf("modeled source edit was not live on the next invocation [exit %d]\n%s\n%s",
			code, out, productWorkerLogs(root))
	}
	second := activeInstall(t, root, "local/modeled-development-package")
	if second.ID == first.ID || second.SourceDigest == first.SourceDigest {
		t.Fatalf("modeled body edit did not advance generation: %#v -> %#v", first, second)
	}

	binding := filepath.Join(project, "package.toml")
	goodBinding, err := os.ReadFile(binding)
	must(t, err)
	brokenBinding := strings.Replace(string(goodBinding), `release = "1.0.0"`,
		`release = "9.9.9"`, 1)
	must(t, os.WriteFile(binding, []byte(brokenBinding), 0o644))
	code, out = runCozy(t, root, "run", "local/modeled-development-package/render",
		"value=7", "--json", "--await")
	if code != 1 || !strings.Contains(out, `"code":"editable_refresh_failed"`) {
		t.Fatalf("absent modeled release did not return typed last-good refusal [exit %d]\n%s",
			code, out)
	}
	failed := activeInstall(t, root, "local/modeled-development-package")
	if failed.ID != second.ID || failed.SourceDigest != second.SourceDigest {
		t.Fatalf("failed modeled refresh displaced last good generation: %#v -> %#v", second, failed)
	}
	must(t, os.WriteFile(binding, goodBinding, 0o644))
	code, out = runCozy(t, root, "run", "local/modeled-development-package/render",
		"value=7", "--json", "--await")
	if code != 0 || !strings.Contains(out, `"value":9`) {
		t.Fatalf("restored modeled selection did not reuse last good generation [exit %d]\n%s",
			code, out)
	}
}

func productWorkerLogs(root string) string {
	paths, _ := filepath.Glob(filepath.Join(root, "workers", "*", "worker.log"))
	var out strings.Builder
	for _, path := range paths {
		data, _ := os.ReadFile(path)
		fmt.Fprintf(&out, "worker log %s:\n%s\n", filepath.Base(filepath.Dir(path)), data)
	}
	return out.String()
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

func modeledDevelopmentProject(t *testing.T, root string) string {
	t.Helper()
	home, err := os.UserHomeDir()
	must(t, err)
	runtimeRepo := filepath.Join(home, "cozy_v2", "cozy-runtime") //cozy:allow peer source; exact test wheel
	tensorfsRepo := filepath.Join(home, "cozy_v2", "tensorfs")    //cozy:allow peer source; exact test wheel
	for _, repo := range []string{runtimeRepo, tensorfsRepo} {
		if _, err := os.Stat(filepath.Join(repo, "pyproject.toml")); err != nil {
			t.Skipf("no peer source at %s: %v", repo, err)
		}
	}
	dir := t.TempDir()
	project := filepath.Join(dir, "source")
	build := exec.Command("/usr/bin/nice", "-n", "19", "python3",
		"tests/product/testdata/build-modeled-editable.py",
		"--runtime-repo", runtimeRepo, "--runtime-sha", editableRuntimeFixtureSHA,
		"--tensorfs-repo", tensorfsRepo, "--tensorfs-sha", editableTensorFSFixtureSHA,
		"--out", dir, "--source-out", project, "--store", filepath.Join(root, "cas"))
	build.Dir = "../.."
	build.Env = childEnv(t, root)
	if out, err := build.CombinedOutput(); err != nil {
		t.Fatalf("building modeled editable fixture: %v\n%s", err, out)
	}
	return project
}

type cozyResult struct {
	code   int
	output string
}

func runCozyEnv(env []string, args ...string) cozyResult {
	cmd := exec.Command("/usr/bin/nice", append([]string{"-n", "19", cozyBin}, args...)...)
	cmd.Env = env
	data, _ := cmd.CombinedOutput()
	code := 0
	if cmd.ProcessState != nil {
		code = cmd.ProcessState.ExitCode()
	}
	return cozyResult{code: code, output: string(data)}
}

func weightlessProject(t *testing.T) string {
	t.Helper()
	home, err := os.UserHomeDir()
	must(t, err)
	repo := filepath.Join(home, "cozy_v2", "cozy-runtime") //cozy:allow peer source; the fixture builds the exact generation Runtime
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
