package producttest

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/cozy-creator/cozy/internal/api"
	"github.com/cozy-creator/cozy/internal/cli"
	localclient "github.com/cozy-creator/cozy/internal/client"
	"github.com/cozy-creator/cozy/internal/config"
	"github.com/cozy-creator/cozy/internal/daemon"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/home"
	"github.com/cozy-creator/cozy/internal/launch"
	"github.com/cozy-creator/cozy/internal/modelproduction"
	"github.com/cozy-creator/cozy/internal/records"
)

func TestModelProductionHumanProgressIsBoundedAndJSONStdoutStaysPure(t *testing.T) {
	var stderr bytes.Buffer
	progress := cli.NewModelProductionProgress(&stderr, true, 10)
	progress.Accepted("modelpub-proof", 4)
	progress.RentalSelecting("h200-sxm")
	progress.RentalReady("rental-1", "ready", "h200-sxm")
	progress.WorkerWaiting("rental-1")
	progress.WorkerReady("rental-1")
	progress.SourceStarting(27, 100<<20)
	for range 100 {
		progress.SourceProgress(10<<20, 100<<20)
	}
	progress.SourceProgress(20<<20, 100<<20)
	progress.SourcePrepared(2)
	progress.StepStarting(0, "assemble-full", "tensorhub/minimax-h3-tools/assemble", false)
	for range 100 {
		progress.StepRuntime(0, "assemble-full", "constructing", 0.2, true)
	}
	progress.StepRuntime(0, "assemble-full", "verifying", 0.1, true)
	progress.ArtifactAdopted(0, 0, 1, "assemble-full")
	progress.PublicationStarting("bf16-full")
	progress.PublicationPrepared("bf16-full")
	progress.StepCompleted(0, "assemble-full", "tensorhub/minimax-h3-tools/assemble")
	for index := 1; index < 10; index++ {
		name := fmt.Sprintf("step-%d", index+1)
		callable := fmt.Sprintf("tensorhub/minimax-h3-tools/function-%d", index+1)
		progress.StepStarting(index, name, callable, false)
		progress.StepCompleted(index, name, callable)
	}
	lanes := []string{"bf16-adaln-pruned", "bf16-full", "fp8-adaln-pruned", "mxfp8-adaln-pruned"}
	progress.ReleaseStarting("1.0.0", lanes)
	progress.ReleaseCut("1.0.0", len(lanes))
	progress.RentalReleaseStarting("rental-1")
	progress.RentalReleased("rental-1")

	human := stderr.String()
	for _, want := range []string{
		"Model production modelpub-proof accepted: 10 steps, 4 lanes.",
		"Rental rental-1: ready (h200-sxm).",
		"Worker: ready on rental rental-1.",
		"Source: preparing 27 files (100.0MiB).",
		"Source: downloaded 10.0MiB / 100.0MiB (10%).",
		"Source: prepared 2 model profiles.",
		"Step 1/10: starting assemble-full (tensorhub/minimax-h3-tools/assemble).",
		"Step 1/10: assemble-full — constructing (20%).",
		"Step 1/10: assemble-full — verifying (10%).",
		"Artifact: adopted output 1/1 for step 1/10 assemble-full.",
		"Publication: lane bf16-full prepared.",
		"Step 1/10: completed assemble-full (tensorhub/minimax-h3-tools/assemble).",
		"Release: 1.0.0 cut atomically with 4 lanes.",
		"Cleanup: rental rental-1 released; provider absence confirmed.",
	} {
		if !strings.Contains(human, want) {
			t.Errorf("human progress omits %q\n%s", want, human)
		}
	}
	if count := strings.Count(human, "Source: downloaded 10.0MiB"); count != 1 {
		t.Errorf("unchanged source polls emitted %d lines\n%s", count, human)
	}
	if count := strings.Count(human, "assemble-full — constructing (20%)"); count != 1 {
		t.Errorf("unchanged Runtime polls emitted %d lines\n%s", count, human)
	}
	if count := strings.Count(human, ": starting "); count != 10 {
		t.Errorf("ten-step production emitted %d start lines\n%s", count, human)
	}
	if count := strings.Count(human, ": completed "); count != 10 {
		t.Errorf("ten-step production emitted %d completion lines\n%s", count, human)
	}

	var disabled bytes.Buffer
	machineProgress := cli.NewModelProductionProgress(&disabled, false, 10)
	machineProgress.Accepted("modelpub-proof", 4)
	machineProgress.SourceProgress(50, 100)
	machineProgress.StepStarting(0, "assemble-full", "tensorhub/minimax-h3-tools/assemble", false)
	machineProgress.RentalReleased("rental-1")
	if disabled.Len() != 0 {
		t.Fatalf("JSON-mode progress wrote to its stream: %q", disabled.String())
	}
	var resumed bytes.Buffer
	resumeProgress := cli.NewModelProductionProgress(&resumed, true, 10)
	resumeProgress.SeedSourceProgress(40<<20, 100<<20)
	resumeProgress.Resume("modelpub-proof", "source preparation has transferred 40.0MiB / 100.0MiB")
	for range 100 {
		resumeProgress.SourceProgress(40<<20, 100<<20)
	}
	resumeProgress.SourceProgress(50<<20, 100<<20)
	if got := resumed.String(); !strings.Contains(got, "Resuming model production modelpub-proof") ||
		strings.Contains(got, "Source: downloaded 40.0MiB") ||
		strings.Count(got, "Source: downloaded") != 1 {
		t.Fatalf("resume repeated already-rendered progress:\n%s", got)
	}

	root := t.TempDir()
	source := filepath.Join(root, "source.safetensors")
	must(t, os.WriteFile(source, []byte("model-production-json-proof"), 0o600))
	code, stdout, jsonStderr := runCozyStreams(t, root, "--json", "model", "publish",
		"acme/proof", source, "--release", "1.0.0", "--dry-run")
	var document map[string]any
	if code != 0 || json.Unmarshal([]byte(stdout), &document) != nil || document["status"] != "planned" ||
		strings.TrimSpace(jsonStderr) != "" {
		t.Fatalf("JSON model production output [exit %d]\nstdout: %s\nstderr: %s", code, stdout, jsonStderr)
	}
}

func TestModelProductionCancellationReportsCleanupTruth(t *testing.T) {
	var confirmed, pending bytes.Buffer
	finished := cli.NewModelProductionProgress(&confirmed, true, 10)
	finished.Cancellation("modelpub-cancel")
	finished.RentalReleaseStarting("rental-cancel")
	finished.RentalReleased("rental-cancel")
	if got := confirmed.String(); !strings.Contains(got, "Cancellation: model production modelpub-cancel stopped") ||
		!strings.Contains(got, "provider absence confirmed") {
		t.Fatalf("confirmed cancellation cleanup changed:\n%s", got)
	}
	unconfirmed := cli.NewModelProductionProgress(&pending, true, 10)
	unconfirmed.Cancellation("modelpub-cancel")
	unconfirmed.RentalReleaseStarting("rental-cancel")
	unconfirmed.RentalReleaseUnconfirmed("rental-cancel")
	if got := pending.String(); !strings.Contains(got, "release is not yet confirmed") ||
		strings.Contains(got, "provider absence confirmed") {
		t.Fatalf("unconfirmed cancellation cleanup overstated reality:\n%s", got)
	}
}

func TestModelProductionSourceRequestHonorsCancellationContext(t *testing.T) {
	started, release := make(chan struct{}), make(chan struct{})
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost || !strings.HasPrefix(r.URL.Path, "/v1/local/model-productions/") {
			http.NotFound(w, r)
			return
		}
		close(started)
		select {
		case <-r.Context().Done():
		case <-release:
		}
	}))
	defer server.Close()
	defer close(release)
	root := t.TempDir()
	layout, problem := home.Open(root)
	fatal(t, problem)
	_, problem = api.Mint(layout)
	fatal(t, problem)
	client, problem := localclient.Open(config.Config{Home: root}, daemon.State{
		Addr: strings.TrimPrefix(server.URL, "http://"), Up: true,
	})
	fatal(t, problem)
	requestCtx, cancel := context.WithCancel(context.Background())
	result := make(chan *exit.Error, 1)
	go func() {
		_, callProblem := client.ModelProductionActionContext(requestCtx, "modelpub-cancel",
			api.ModelProductionAction{Action: "prepare_source", RentalID: "rental-1"})
		result <- callProblem
	}()
	select {
	case <-started:
	case <-time.After(2 * time.Second):
		t.Fatal("model-production source request did not start")
	}
	cancel()
	select {
	case callProblem := <-result:
		if callProblem == nil || callProblem.Code != exit.Canceled {
			t.Fatalf("canceled source request = %v", callProblem)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("canceled model-production source request did not return")
	}
}

func TestRemoteModelProductionResolvesHubMetadataWithoutLocalInstall(t *testing.T) {
	producerBytes := []byte(`{"application":"remote_producer:app","entrypoints":[],"format":"cozy.package.descriptor/1","jobs":[{"artifact_outputs":[{"max_bytes":4096,"mime_type":"application/vnd.cozy.model-manifest","output_id":"model"}],"models":[{"class":"ProofModel","component_use":{},"path":"assemble.models.source","stamps":{}}],"name":"assemble","publishes":false,"request":{"fields":[]},"resources":{"gpu_count":1,"placement":"single_node","requires":"sm90+,vram80g,ram64g"},"result":{"fields":[]}}],"model_productions":[{"name":"build","steps":[{"callable":"proof/remote-producer@v2/assemble","models":{"source":"source"},"name":"assemble","outputs":["model"],"resources":{"gpu_count":1,"placement":"single_node","requires":"sm90+,vram80g,ram64g"}},{"callable":"proof/remote-job@v3/derive","models":{"source":"assemble.model"},"name":"derive","outputs":["model"],"resources":{"gpu_count":1,"placement":"single_node","requires":"sm90+,vram80g,ram64g"}}],"outputs":[{"lane_key":"bf16","name":"bf16","required_contract":{"encodings":["sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"],"topology_digest":"sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"},"source":"derive.model"}],"sources":{"source":"proof/source/1"}}]}`)
	producer, problem := launch.DecodeDescriptor(producerBytes)
	fatal(t, problem)
	if _, invalid := launch.DecodeDescriptor(bytes.ReplaceAll(producerBytes,
		[]byte("remote-job@v3"), []byte("remote-job@v03"))); invalid == nil {
		t.Fatal("descriptor accepted a non-canonical production callable major")
	}
	jobBytes := []byte(`{"application":"remote_job:app","entrypoints":[],"format":"cozy.package.descriptor/1","jobs":[{"artifact_outputs":[{"max_bytes":4096,"mime_type":"application/vnd.cozy.model-manifest","output_id":"model"}],"models":[{"class":"ProofModel","component_use":{},"path":"derive.models.source","stamps":{}}],"name":"derive","publishes":false,"request":{"fields":[]},"resources":{"gpu_count":1,"placement":"single_node","requires":"sm90+,vram80g,ram64g"},"result":{"fields":[]}}],"model_productions":[]}`)
	job, problem := launch.DecodeDescriptor(jobBytes)
	fatal(t, problem)
	type fixture struct {
		release, digest string
		document        json.RawMessage
		descriptor      *launch.PackageDescriptor
	}
	newFixture := func(release string, descriptor *launch.PackageDescriptor) fixture {
		document := json.RawMessage(`{"format":"PackageRelease/1","release":"` + release + `"}`)
		return fixture{release: release, digest: "sha256:" + strings.Repeat(release[:1], 64), document: document,
			descriptor: descriptor}
	}
	producerRelease, jobRelease := newFixture("2.0.0", producer), newFixture("3.0.0", job)
	writeRelease := func(w http.ResponseWriter, selected fixture) {
		_ = json.NewEncoder(w).Encode(map[string]any{
			"release": map[string]any{
				"release": selected.release, "release_digest": selected.digest,
				"package_descriptor_digest": selected.descriptor.Digest,
				"package_descriptor_length": len(selected.descriptor.Raw),
				"created_at":                "2026-08-31T00:00:00Z",
			},
			"document":           selected.document,
			"package_descriptor": json.RawMessage(selected.descriptor.Raw),
		})
	}
	requests := []string{}
	producerCardReads := 0
	jobReleases := []map[string]any{
		{"release": "2.9.0", "cut_at": "2026-08-31T00:00:00Z"},
		{"release": "3.0.0", "cut_at": "2026-08-31T00:00:00Z"},
		{"release": "4.0.0", "cut_at": "2026-08-31T00:00:00Z"},
	}
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		requests = append(requests, r.Method+" "+r.URL.Path)
		w.Header().Set("Content-Type", "application/json")
		switch {
		case r.Method == http.MethodGet && r.URL.Path == "/v1/models/resolve":
			_ = json.NewEncoder(w).Encode(map[string]any{
				"model": "acme/input", "release": "1.0.0", "lane": "bf16",
				"manifest_id":   "sha256:" + strings.Repeat("d", 64),
				"header_digest": "sha256:" + strings.Repeat("e", 64),
				"objects":       1, "bytes": 4096,
			})
		case r.Method == http.MethodGet && r.URL.Path == "/v1/packages/proof/remote-producer":
			producerCardReads++
			producerReleases := []map[string]any{
				{"release": "1.0.0", "cut_at": "2026-08-31T00:00:00Z"},
				{"release": "2.0.0", "cut_at": "2026-08-31T00:00:00Z"},
				{"release": "3.0.0", "cut_at": "2026-08-31T00:00:00Z"},
			}
			if producerCardReads > 1 {
				producerReleases[1]["release"] = "2.0.1"
			}
			_ = json.NewEncoder(w).Encode(map[string]any{
				"package": map[string]any{"org": "proof", "name": "remote-producer",
					"created_at": "2026-08-31T00:00:00Z"},
				"releases": producerReleases,
			})
		case r.Method == http.MethodGet &&
			r.URL.Path == "/v1/packages/proof/remote-producer/releases/2.0.0":
			writeRelease(w, producerRelease)
		case r.Method == http.MethodGet && r.URL.Path == "/v1/packages/proof/remote-job":
			_ = json.NewEncoder(w).Encode(map[string]any{
				"package": map[string]any{"org": "proof", "name": "remote-job",
					"created_at": "2026-08-31T00:00:00Z"},
				"releases": jobReleases,
			})
		case r.Method == http.MethodGet &&
			r.URL.Path == "/v1/packages/proof/remote-job/releases/3.0.0":
			writeRelease(w, jobRelease)
		default:
			http.Error(w, "unexpected route", http.StatusNotFound)
		}
	}))
	defer server.Close()

	root := t.TempDir()
	must(t, os.WriteFile(filepath.Join(root, "config.yaml"), []byte(
		"tensorhub_url: "+server.URL+"\nrentals:\n  max_hourly_spend_usd: 10\n"), 0o600))
	code, out := runCozyDir(t, root, "", []string{"PATH=/usr/bin:/bin"},
		"--json", "--full", "model", "publish", "acme/output", "acme/input@1.0.0",
		"--release", "1.0.0", "--producer", "proof/remote-producer@v2/build",
		"--rental", "--dry-run")
	if code != 0 || !strings.Contains(out, `"status":"planned"`) ||
		!strings.Contains(out, `"producer":"proof/remote-producer@v2/build@2.0.0"`) {
		t.Fatalf("metadata-only remote production [exit %d]\n%s", code, out)
	}
	if got := strings.Join(requests, "\n"); strings.Contains(got, "/download") ||
		got != "GET /v1/models/resolve\nGET /v1/packages/proof/remote-producer\n"+
			"GET /v1/packages/proof/remote-producer/releases/2.0.0\n"+
			"GET /v1/packages/proof/remote-job\n"+
			"GET /v1/packages/proof/remote-job/releases/3.0.0" {
		t.Fatalf("remote production metadata routes =\n%s", got)
	}
	if producerCardReads != 1 {
		t.Fatalf("self-step re-resolved an advanced producer catalog %d times", producerCardReads)
	}
	jobReleases = []map[string]any{
		{"release": "2.9.0", "cut_at": "2026-08-31T00:00:00Z"},
		{"release": "4.0.0", "cut_at": "2026-08-31T00:00:00Z"},
	}
	requests = nil
	producerCardReads = 0
	code, out = runCozyDir(t, root, "", []string{"PATH=/usr/bin:/bin"},
		"model", "publish", "acme/output", "acme/input@1.0.0", "--release", "1.0.0",
		"--producer", "proof/remote-producer@v2/build", "--rental", "--dry-run")
	if code == 0 || !strings.Contains(out, "no active immutable release in v3") {
		t.Fatalf("missing production callable major [exit %d]\n%s", code, out)
	}
	requests = nil
	code, out = runCozyDir(t, root, "", []string{"PATH=/usr/bin:/bin"},
		"model", "publish", "acme/output", "acme/input@1.0.0", "--release", "1.0.0",
		"--producer", "proof/remote-producer@v2/build", "--dry-run")
	if code == 0 || !strings.Contains(out, "not installed") {
		t.Fatalf("local production stopped requiring a local install [exit %d]\n%s", code, out)
	}
	if got := strings.Join(requests, "\n"); got != "GET /v1/models/resolve" {
		t.Fatalf("local production unexpectedly resolved remote package metadata:\n%s", got)
	}
}

func TestDetachedModelProductionResumesInDaemonAndKeepsFrozenPackagePlan(t *testing.T) {
	producerBytes := []byte(`{"application":"remote_producer:app","entrypoints":[],"format":"cozy.package.descriptor/1","jobs":[],"model_productions":[{"name":"build","steps":[{"callable":"proof/remote-job@v3/derive","models":{"source":"source"},"name":"derive","outputs":["model"],"resources":{"gpu_count":1,"placement":"single_node","requires":"sm90+,vram80g,ram64g"}}],"outputs":[{"lane_key":"bf16","name":"bf16","required_contract":{"encodings":["sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"],"topology_digest":"sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"},"source":"derive.model"}],"sources":{"source":"proof/source/1"}}]}`)
	producer, problem := launch.DecodeDescriptor(producerBytes)
	fatal(t, problem)
	jobBytes := []byte(`{"application":"remote_job:app","entrypoints":[],"format":"cozy.package.descriptor/1","jobs":[{"artifact_outputs":[{"max_bytes":4096,"mime_type":"application/vnd.cozy.model-manifest","output_id":"model"}],"models":[{"class":"ProofModel","component_use":{},"path":"derive.models.source","stamps":{}}],"name":"derive","publishes":false,"request":{"fields":[]},"resources":{"gpu_count":1,"placement":"single_node","requires":"sm90+,vram80g,ram64g"},"result":{"fields":[]}}],"model_productions":[]}`)
	job, problem := launch.DecodeDescriptor(jobBytes)
	fatal(t, problem)

	var mu sync.Mutex
	producerRelease := "2.0.0"
	packageReads := 0
	sourceReads := 0
	writeRelease := func(w http.ResponseWriter, release string, descriptor *launch.PackageDescriptor) {
		digestCharacter := release[0:1]
		_ = json.NewEncoder(w).Encode(map[string]any{
			"release": map[string]any{
				"release": release, "release_digest": "sha256:" + strings.Repeat(digestCharacter, 64),
				"package_descriptor_digest": descriptor.Digest,
				"package_descriptor_length": len(descriptor.Raw),
				"created_at":                "2026-08-31T00:00:00Z",
			},
			"document":           json.RawMessage(`{"format":"PackageRelease/1","release":"` + release + `"}`),
			"package_descriptor": json.RawMessage(descriptor.Raw),
		})
	}
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		switch {
		case r.Method == http.MethodGet && r.URL.Path == "/v1/models/resolve":
			mu.Lock()
			sourceReads++
			mu.Unlock()
			_ = json.NewEncoder(w).Encode(map[string]any{
				"model": "acme/input", "release": "1.0.0", "lane": "bf16",
				"manifest_id":   "sha256:" + strings.Repeat("d", 64),
				"header_digest": "sha256:" + strings.Repeat("e", 64),
				"objects":       1, "bytes": 4096,
			})
		case r.Method == http.MethodGet && r.URL.Path == "/v1/packages/proof/remote-producer":
			mu.Lock()
			packageReads++
			release := producerRelease
			mu.Unlock()
			_ = json.NewEncoder(w).Encode(map[string]any{
				"package": map[string]any{"org": "proof", "name": "remote-producer",
					"created_at": "2026-08-31T00:00:00Z"},
				"releases": []map[string]any{{"release": release, "cut_at": "2026-08-31T00:00:00Z"}},
			})
		case r.Method == http.MethodGet && strings.HasPrefix(r.URL.Path,
			"/v1/packages/proof/remote-producer/releases/"):
			mu.Lock()
			packageReads++
			release := producerRelease
			mu.Unlock()
			writeRelease(w, release, producer)
		case r.Method == http.MethodGet && r.URL.Path == "/v1/packages/proof/remote-job":
			mu.Lock()
			packageReads++
			mu.Unlock()
			_ = json.NewEncoder(w).Encode(map[string]any{
				"package": map[string]any{"org": "proof", "name": "remote-job",
					"created_at": "2026-08-31T00:00:00Z"},
				"releases": []map[string]any{{"release": "3.0.0", "cut_at": "2026-08-31T00:00:00Z"}},
			})
		case r.Method == http.MethodGet && r.URL.Path == "/v1/packages/proof/remote-job/releases/3.0.0":
			mu.Lock()
			packageReads++
			mu.Unlock()
			writeRelease(w, "3.0.0", job)
		default:
			http.Error(w, "unexpected route "+r.Method+" "+r.URL.Path, http.StatusNotFound)
		}
	}))
	defer server.Close()

	root := t.TempDir()
	must(t, os.WriteFile(filepath.Join(root, "config.yaml"), []byte(
		"tensorhub_url: "+server.URL+"\nrentals:\n  max_hourly_spend_usd: 10\n"), 0o600))
	args := []string{"--json", "--full", "model", "publish", "acme/output", "acme/input@1.0.0",
		"--release", "1.0.0", "--producer", "proof/remote-producer/build", "--rental", "--detach"}
	code, out := runCozy(t, root, args...)
	var accepted api.ModelProductionState
	if err := json.Unmarshal([]byte(out), &accepted); code != 0 || err != nil ||
		!strings.HasPrefix(accepted.ID, "modelpub-") || accepted.Kind != "model-publication" {
		t.Fatalf("detached acceptance [exit %d, parse %v]\n%s", code, err, out)
	}
	deadline := time.Now().Add(5 * time.Second)
	for time.Now().Before(deadline) {
		mu.Lock()
		reads := sourceReads
		mu.Unlock()
		if reads >= 2 { // first acceptance resolution plus daemon-owned advancement refresh
			break
		}
		time.Sleep(25 * time.Millisecond)
	}
	layout, problem := home.Open(root)
	fatal(t, problem)
	store, problem := records.Open(layout.DB)
	fatal(t, problem)
	operation, problem := store.ModelProduction(accepted.ID)
	fatal(t, problem)
	if operation == nil {
		t.Fatal("detached model production was not durable")
	}
	plan, err := modelproduction.Parse(operation.Plan)
	must(t, err)
	store.Close()
	if plan.ProducerRelease != "2.0.0" || plan.ID() != accepted.ID {
		t.Fatalf("accepted plan = producer %s id %s", plan.ProducerRelease, plan.ID())
	}
	mu.Lock()
	initialPackageReads := packageReads
	producerRelease = "4.0.0"
	mu.Unlock()
	code, replayed := runCozy(t, root, args...)
	if code != 0 || !strings.Contains(replayed, `"id":"`+accepted.ID+`"`) {
		t.Fatalf("exact detached replay [exit %d]\n%s", code, replayed)
	}
	mu.Lock()
	readsAfterReplay := packageReads
	mu.Unlock()
	if readsAfterReplay != initialPackageReads {
		t.Fatalf("exact replay re-resolved package latest: reads %d -> %d", initialPackageReads, readsAfterReplay)
	}
	followArgs := append([]string(nil), args[:len(args)-1]...)
	follow := exec.Command(cozyBin, followArgs...)
	follow.Env = childEnv(t, root)
	var followStdout, followStderr bytes.Buffer
	follow.Stdout, follow.Stderr = &followStdout, &followStderr
	must(t, follow.Start())
	time.Sleep(250 * time.Millisecond)
	must(t, follow.Process.Signal(os.Interrupt))
	if err := follow.Wait(); err != nil || !strings.Contains(followStdout.String(), accepted.ID) ||
		strings.TrimSpace(followStderr.String()) != "" {
		t.Fatalf("follow detach [wait %v]\nstdout: %s\nstderr: %s",
			err, followStdout.String(), followStderr.String())
	}
	store, problem = records.Open(layout.DB)
	fatal(t, problem)
	stillRunning, problem := store.ModelProduction(accepted.ID)
	fatal(t, problem)
	store.Close()
	if stillRunning == nil || stillRunning.CancelRequested {
		t.Fatalf("follow interrupt canceled durable work: %+v", stillRunning)
	}
	if code, listed := runCozy(t, root, "--json", "run", "list"); code != 0 ||
		!strings.Contains(listed, accepted.ID) || !strings.Contains(listed, "model-publication") {
		t.Fatalf("run list omitted model production [exit %d]\n%s", code, listed)
	}

	mu.Lock()
	readsBeforeRestart := sourceReads
	mu.Unlock()
	terminateTestDaemon(t, root)
	if code, listed := runCozy(t, root, "--json", "run", "list"); code != 0 ||
		!strings.Contains(listed, accepted.ID) {
		t.Fatalf("restarted daemon lost model production [exit %d]\n%s", code, listed)
	}
	deadline = time.Now().Add(5 * time.Second)
	for time.Now().Before(deadline) {
		mu.Lock()
		reads := sourceReads
		mu.Unlock()
		if reads > readsBeforeRestart {
			break
		}
		time.Sleep(25 * time.Millisecond)
	}
	mu.Lock()
	readsAfterRestart := sourceReads
	mu.Unlock()
	if readsAfterRestart <= readsBeforeRestart {
		t.Fatal("restarted daemon did not resume accepted model production")
	}
	if code, canceled := runCozy(t, root, "--json", "run", "cancel", accepted.ID); code != 0 ||
		!strings.Contains(canceled, `"status":"canceled"`) {
		t.Fatalf("durable model production cancel [exit %d]\n%s", code, canceled)
	}
	terminateTestDaemon(t, root)
}

func TestModelProductionOperationSurvivesRestartAndReplaysExactly(t *testing.T) {
	instruction := modelproduction.Instruction{
		Destination: "tensorhub/minimax-h3", Release: "h3-2026-08-31",
		Source:   "hf://MiniMaxAI/MiniMax-H3@" + strings.Repeat("a", 40),
		Producer: "tensorhub/minimax-h3-tools/four-lane", Rental: true,
	}
	plan := modelproduction.Plan{
		Instruction: instruction,
		Destination: "tensorhub/minimax-h3", Release: "1.0.0",
		Source:          "hf://MiniMaxAI/MiniMax-H3@" + strings.Repeat("a", 40),
		SourceSelection: "sha256:" + strings.Repeat("b", 64),
		SourceFiles: []modelproduction.SourceFile{{
			Member: "transformer/model-00001-of-00002.safetensors",
			SHA256: strings.Repeat("c", 64), Length: 4096,
		}},
		Producer: "tensorhub/minimax-h3-tools/four-lane", ProducerRelease: "1.0.0",
		ProducerDigest:   "sha256:" + strings.Repeat("d", 64),
		DescriptorDigest: "sha256:" + strings.Repeat("e", 64),
		Jobs: []modelproduction.JobPin{{Step: "full", Callable: "tensorhub/minimax-h3-tools@v2/assemble",
			Release: "1.0.0", ReleaseDigest: "sha256:" + strings.Repeat("f", 64),
		}},
	}
	data, err := plan.Bytes()
	must(t, err)
	digest, err := plan.Digest()
	must(t, err)
	if bytes.Contains(data, []byte("https://")) || bytes.Contains(data, []byte("token")) ||
		bytes.Contains(data, []byte("upload_grant")) {
		t.Fatalf("restart plan contains transient authority: %s", data)
	}
	path := filepath.Join(t.TempDir(), "records.db")
	store, problem := records.Open(path)
	fatal(t, problem)
	instructionBytes, err := instruction.Bytes()
	must(t, err)
	instructionDigest, err := instruction.Digest()
	must(t, err)
	created, replay, problem := store.BeginModelProductionInstruction(
		instruction.ID(), instructionDigest, instructionBytes)
	fatal(t, problem)
	if replay || created.State != "resolving" || created.StepIndex != 0 ||
		!bytes.Equal(created.Plan, instructionBytes) {
		t.Fatalf("created operation = %+v replay=%t", created, replay)
	}
	created, replay, problem = store.AttachModelProductionPlan(instruction.ID(), instructionDigest,
		instructionBytes, data, digest)
	fatal(t, problem)
	if replay || created.State != "accepted" || !bytes.Equal(created.Plan, data) {
		t.Fatalf("accepted operation = %+v replay=%t", created, replay)
	}
	store.Close()

	store, problem = records.Open(path)
	fatal(t, problem)
	defer store.Close()
	replayed, replay, problem := store.BeginModelProductionInstruction(
		instruction.ID(), instructionDigest, instructionBytes)
	fatal(t, problem)
	if !replay || replayed.CreatedAt != created.CreatedAt || !bytes.Equal(replayed.Plan, data) {
		t.Fatalf("restarted replay = %+v replay=%t", replayed, replay)
	}
	fatal(t, store.AdvanceModelProduction(plan.ID(), "accepted", "source_preparing", 0, "rental-1"))
	fatal(t, store.AdvanceModelProduction(plan.ID(), "source_preparing", "source_prepared", 0, "rental-1"))
	fatal(t, store.AdvanceModelProduction(plan.ID(), "source_prepared", "step_running", 0, "rental-1"))
	fatal(t, store.AdvanceModelProduction(plan.ID(), "step_running", "step_running", 1, "rental-1"))
	current, problem := store.ModelProduction(plan.ID())
	fatal(t, problem)
	if current == nil || current.State != "step_running" || current.StepIndex != 1 ||
		current.RentalID != "rental-1" {
		t.Fatalf("advanced operation = %+v", current)
	}
	changedInstruction := instruction
	changedInstruction.Producer = "tensorhub/other/production"
	changedBytes, err := changedInstruction.Bytes()
	must(t, err)
	changedDigest, err := changedInstruction.Digest()
	must(t, err)
	if _, _, problem := store.BeginModelProductionInstruction(plan.ID(), changedDigest,
		changedBytes); problem == nil || problem.Name != "model_production.identity_conflict" {
		t.Fatalf("changed replay bytes = %v", problem)
	}
	if problem := store.AdvanceModelProduction(plan.ID(), "step_running", "completed", 1, "rental-1"); problem == nil || problem.Name != "model_production.transition_invalid" {
		t.Fatalf("skipped release/cleanup states = %v", problem)
	}
	fatal(t, store.CancelModelProduction(plan.ID(), "step_running", "user interrupted"))
	canceled, problem := store.ModelProduction(plan.ID())
	fatal(t, problem)
	if canceled == nil || canceled.State != "canceled" || !canceled.CancelRequested {
		t.Fatalf("canceled production = %+v", canceled)
	}
}

func TestModelProductionCancelCutOrderingAndUnattachedProviderAbsence(t *testing.T) {
	store, problem := records.Open(filepath.Join(t.TempDir(), "records.db"))
	fatal(t, problem)
	makeOperation := func(release string) (string, modelproduction.Plan) {
		instruction := modelproduction.Instruction{Destination: "acme/model", Release: release,
			Source:   "hf://acme/model@" + strings.Repeat("a", 40),
			Producer: "acme/tools/build", Rental: true}
		plan := modelproduction.Plan{Instruction: instruction, Destination: instruction.Destination,
			Release: release, Source: instruction.Source}
		instructionBytes, err := instruction.Bytes()
		must(t, err)
		instructionDigest, err := instruction.Digest()
		must(t, err)
		_, _, problem := store.BeginModelProductionInstruction(instruction.ID(),
			instructionDigest, instructionBytes)
		fatal(t, problem)
		planBytes, err := plan.Bytes()
		must(t, err)
		planDigest, err := plan.Digest()
		must(t, err)
		_, _, problem = store.AttachModelProductionPlan(instruction.ID(), instructionDigest,
			instructionBytes, planBytes, planDigest)
		fatal(t, problem)
		fatal(t, store.AdvanceModelProduction(instruction.ID(), "accepted", "source_preparing", 0, "rental-proof"))
		fatal(t, store.AdvanceModelProduction(instruction.ID(), "source_preparing", "source_prepared", 0, "rental-proof"))
		fatal(t, store.AdvanceModelProduction(instruction.ID(), "source_prepared", "step_running", 0, "rental-proof"))
		fatal(t, store.AdvanceModelProduction(instruction.ID(), "step_running", "outputs_preparing", 1, "rental-proof"))
		return instruction.ID(), plan
	}
	beforeID, _ := makeOperation("before-cut")
	fatal(t, store.RequestModelProductionCancel(beforeID))
	fatal(t, store.CancelModelProduction(beforeID, "outputs_preparing", "cancel won before cut"))
	before, problem := store.ModelProduction(beforeID)
	fatal(t, problem)
	if before == nil || before.State != "canceled" {
		t.Fatalf("cancel-before-cut state = %+v", before)
	}

	afterID, _ := makeOperation("after-cut")
	fatal(t, store.AdvanceModelProduction(afterID, "outputs_preparing", "release_cut", 1, "rental-proof"))
	fatal(t, store.RequestModelProductionCancel(afterID))
	fatal(t, store.AdvanceModelProduction(afterID, "release_cut", "cleanup_pending", 1, "rental-proof"))
	fatal(t, store.AdvanceModelProduction(afterID, "cleanup_pending", "completed", 1, "rental-proof"))
	after, problem := store.ModelProduction(afterID)
	fatal(t, problem)
	if after == nil || after.State != "completed" || !after.CancelRequested {
		t.Fatalf("commit-before-cancel state = %+v", after)
	}
	store.Close()

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method == http.MethodGet && r.URL.Path == "/v1/rentals/rental-proof" {
			http.NotFound(w, r)
			return
		}
		http.Error(w, "unexpected", http.StatusInternalServerError)
	}))
	defer server.Close()
	root := t.TempDir()
	layout, problem := home.Open(root)
	fatal(t, problem)
	store, problem = records.Open(layout.DB)
	fatal(t, problem)
	_, _, problem = store.BeginRentalOperation(records.RentalOperation{
		Key: "model-production-rental-proof", RequestDigest: "sha256:" + strings.Repeat("b", 64),
		RequestBody: []byte(`{}`), Hub: server.URL, Reason: "model production proof",
		HourlyRateUSDMicros: 1,
	}, 10)
	fatal(t, problem)
	fatal(t, store.AdvanceRentalOperation("model-production-rental-proof", "rental-proof", "acquiring"))
	store.Close()
	code, out := runCozyDir(t, root, "", []string{
		"TENSORHUB_URL=" + server.URL, "TENSORHUB_TOKEN=proof-token",
	}, "--json", "rental", "end", "rental-proof")
	if code != 0 || !strings.Contains(out, `"state":"ended"`) {
		t.Fatalf("provider absence recovery [exit %d]\n%s", code, out)
	}
	store, problem = records.Open(layout.DB)
	fatal(t, problem)
	defer store.Close()
	operation, problem := store.RentalOperation("model-production-rental-proof")
	fatal(t, problem)
	active, problem := store.ActiveRentalOperations()
	fatal(t, problem)
	if operation == nil || operation.State != "released" || len(active) != 0 {
		t.Fatalf("provider absence did not settle operation: operation=%+v active=%+v", operation, active)
	}
}

func TestModelProductionAmbiguousCutReplaysExactlyAfterDaemonRestart(t *testing.T) {
	var mu sync.Mutex
	cutBodies := []string{}
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch {
		case r.Method == http.MethodPost && r.URL.Path == "/v1/models/acme/output/releases/ambiguous":
			body, _ := io.ReadAll(r.Body)
			mu.Lock()
			cutBodies = append(cutBodies, string(body))
			attempt := len(cutBodies)
			mu.Unlock()
			w.Header().Set("Content-Type", "application/json")
			if attempt == 1 {
				w.WriteHeader(http.StatusServiceUnavailable)
				_, _ = io.WriteString(w, `{"error":{"code":"cut_unavailable","message":"cut response was lost"}}`)
				return
			}
			_ = json.NewEncoder(w).Encode(map[string]any{
				"operation": "modelpub-proof", "release": "ambiguous",
				"repository_sha256": "sha256:" + strings.Repeat("f", 64), "duplicate": true,
				"lanes": []map[string]any{{"lane": "bf16", "checkpoint_id": "sha256:" + strings.Repeat("d", 64)}},
			})
		case r.Method == http.MethodGet && r.URL.Path == "/v1/rentals/rental-cut-proof":
			http.NotFound(w, r)
		default:
			http.Error(w, "unexpected "+r.Method+" "+r.URL.Path, http.StatusNotFound)
		}
	}))
	defer server.Close()
	root := t.TempDir()
	must(t, os.WriteFile(filepath.Join(root, "config.yaml"), []byte(
		"tensorhub_url: "+server.URL+"\ntensorhub_token: proof-token\nrentals:\n  max_hourly_spend_usd: 10\n"), 0o600))
	layout, problem := home.Open(root)
	fatal(t, problem)
	store, problem := records.Open(layout.DB)
	fatal(t, problem)
	instruction := modelproduction.Instruction{Destination: "acme/output", Release: "ambiguous",
		Source:   "hf://acme/model@" + strings.Repeat("a", 40),
		Producer: "acme/tools/build", Rental: true}
	production := &launch.ModelProduction{Name: "build", Sources: map[string]string{"source": "proof/source/1"},
		Steps: []launch.ModelProductionStep{{Name: "derive", Callable: "acme/job@v1/derive",
			Models: map[string]string{"source": "source"}, Outputs: []string{"model"}}},
		Outputs: []launch.ModelProductionOutput{{Name: "bf16", Source: "derive.model", LaneKey: "bf16"}}}
	plan := modelproduction.Plan{Instruction: instruction, Destination: instruction.Destination,
		Release: instruction.Release, Source: instruction.Source, Producer: instruction.Producer,
		Production: production, Jobs: []modelproduction.JobPin{{Step: "derive", Callable: "acme/job@v1/derive"}}}
	instructionBytes, err := instruction.Bytes()
	must(t, err)
	instructionDigest, err := instruction.Digest()
	must(t, err)
	_, _, problem = store.BeginModelProductionInstruction(instruction.ID(), instructionDigest,
		instructionBytes)
	fatal(t, problem)
	planBytes, err := plan.Bytes()
	must(t, err)
	planDigest, err := plan.Digest()
	must(t, err)
	_, _, problem = store.AttachModelProductionPlan(instruction.ID(), instructionDigest,
		instructionBytes, planBytes, planDigest)
	fatal(t, problem)
	fatal(t, store.AdvanceModelProduction(instruction.ID(), "accepted", "source_preparing", 0, "rental-cut-proof"))
	fatal(t, store.AdvanceModelProduction(instruction.ID(), "source_preparing", "source_prepared", 0, "rental-cut-proof"))
	fatal(t, store.AdvanceModelProduction(instruction.ID(), "source_prepared", "step_running", 0, "rental-cut-proof"))
	fatal(t, store.AdvanceModelProduction(instruction.ID(), "step_running", "outputs_preparing", 1, "rental-cut-proof"))
	artifact := records.ModelProductionArtifact{OperationID: instruction.ID(), StepName: "derive",
		OutputSlot: "model", RequestID: "job-proof", Attempt: 1,
		InvocationDigest: "sha256:" + strings.Repeat("b", 64), TransactionID: "transaction-proof",
		WriterGeneration: 1, ReceiptDigest: "sha256:" + strings.Repeat("c", 64),
		Receipt: []byte(`{"format":"proof/1"}`), ManifestID: "sha256:" + strings.Repeat("d", 64),
		ManifestLength: 128, CheckpointEvidence: []byte(`{"format":"evidence/1"}`)}
	fatal(t, store.RecordModelProductionArtifact(artifact, nil))
	fatal(t, store.MarkModelProductionArtifactPublished(instruction.ID(), "derive", "model", "publish-proof"))
	store.Close()

	if code, out := runCozy(t, root, "--json", "run", "list"); code != 0 ||
		!strings.Contains(out, instruction.ID()) {
		t.Fatalf("daemon did not adopt prepared production [exit %d]\n%s", code, out)
	}
	deadline := time.Now().Add(8 * time.Second)
	for time.Now().Before(deadline) {
		store, problem = records.Open(layout.DB)
		fatal(t, problem)
		operation, readProblem := store.ModelProduction(instruction.ID())
		store.Close()
		fatal(t, readProblem)
		if operation != nil && operation.State == "completed" {
			break
		}
		time.Sleep(100 * time.Millisecond)
	}
	store, problem = records.Open(layout.DB)
	fatal(t, problem)
	operation, problem := store.ModelProduction(instruction.ID())
	store.Close()
	fatal(t, problem)
	mu.Lock()
	bodies := append([]string(nil), cutBodies...)
	mu.Unlock()
	if operation == nil || operation.State != "completed" || len(bodies) != 2 || bodies[0] != bodies[1] {
		t.Fatalf("ambiguous cut recovery = operation %+v bodies %#v", operation, bodies)
	}
	terminateTestDaemon(t, root)
}

func TestModelProductionJoinsStepArtifactAndTransferBeforeReplay(t *testing.T) {
	store, problem := records.Open(filepath.Join(t.TempDir(), "records.db"))
	fatal(t, problem)
	defer store.Close()
	plan := modelproduction.Plan{Destination: "acme/model", Release: "1.0.0",
		Source:          "hf://acme/model@" + strings.Repeat("a", 40),
		SourceSelection: "sha256:" + strings.Repeat("b", 64)}
	data, err := plan.Bytes()
	must(t, err)
	digest, err := plan.Digest()
	must(t, err)
	_, _, problem = store.BeginModelProduction(records.ModelProductionOperation{
		ID: plan.ID(), PlanDigest: digest, Plan: data,
	})
	fatal(t, problem)
	_, problem = store.BeginModelProductionStep(records.ModelProductionStep{
		OperationID: plan.ID(), StepIndex: 0, StepName: "derive", State: "pending",
	})
	fatal(t, problem)
	fatal(t, store.SetModelProductionStepRequest(plan.ID(), 0, "derive", "job-1", "submitted"))
	step, problem := store.ModelProductionStepByRequest("job-1")
	fatal(t, problem)
	if step == nil || step.OperationID != plan.ID() || step.StepName != "derive" {
		t.Fatalf("step request join = %+v", step)
	}
	receipt := []byte(`{"format":"proof/1"}`)
	evidence := []byte(`{"format":"evidence/1"}`)
	artifact := records.ModelProductionArtifact{OperationID: plan.ID(), StepName: "derive",
		OutputSlot: "model", RequestID: "job-1", Attempt: 1,
		InvocationDigest: "sha256:" + strings.Repeat("c", 64), TransactionID: "txn-1",
		WriterGeneration: 1, ReceiptDigest: "sha256:" + strings.Repeat("d", 64),
		Receipt: receipt, ManifestID: "sha256:" + strings.Repeat("e", 64),
		ManifestLength: 128, CheckpointEvidence: evidence}
	object := records.ModelProductionObject{OperationID: plan.ID(), StepName: "derive",
		OutputSlot: "model", ObjectID: "sha256:" + strings.Repeat("f", 64),
		Length: 256, SourceRef: "opaque-source"}
	fatal(t, store.RecordModelProductionArtifact(artifact, []records.ModelProductionObject{object}))
	changedObject := object
	changedObject.SourceRef = "other-source"
	if problem := store.RecordModelProductionArtifact(artifact,
		[]records.ModelProductionObject{changedObject}); problem == nil ||
		problem.Name != "model_production.artifact_conflict" {
		t.Fatalf("changed object inventory replay = %v", problem)
	}
	fatal(t, store.RecordModelProductionObjectStatus(records.ModelProductionObject{
		OperationID: plan.ID(), StepName: "derive", OutputSlot: "model",
		ObjectID: object.ObjectID, Length: object.Length, TransferOperationID: "pub-1",
		GrantRevision: 1, UpdateSequence: 1, State: "uploaded", TransferredBytes: object.Length,
	}))
	fatal(t, store.MarkModelProductionArtifactPublished(plan.ID(), "derive", "model", "publish-1"))
	artifacts, problem := store.ModelProductionArtifacts(plan.ID())
	fatal(t, problem)
	objects, problem := store.ModelProductionObjects(plan.ID(), "derive", "model")
	fatal(t, problem)
	if len(artifacts) != 1 || artifacts[0].PublicationID != "publish-1" ||
		artifacts[0].State != "prepared" || len(objects) != 1 || objects[0].State != "uploaded" ||
		objects[0].TransferOperationID != "pub-1" {
		t.Fatalf("durable artifact join = artifacts=%+v objects=%+v", artifacts, objects)
	}
	if problem := store.RecordModelProductionObjectStatus(records.ModelProductionObject{
		OperationID: plan.ID(), StepName: "derive", OutputSlot: "model",
		ObjectID: object.ObjectID, Length: object.Length, TransferOperationID: "other",
		GrantRevision: 1, UpdateSequence: 2, State: "uploaded", TransferredBytes: object.Length,
	}); problem == nil || problem.Name != "model_production.artifact_status_conflict" {
		t.Fatalf("changed transfer operation replay = %v", problem)
	}
}
