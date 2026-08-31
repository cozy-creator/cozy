package producttest

import (
	"bytes"
	"context"
	"crypto/ed25519"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/cozy-creator/cozy/internal/config"
	"github.com/cozy-creator/cozy/internal/daemon"
	"github.com/cozy-creator/cozy/internal/launch"
	"github.com/cozy-creator/cozy/internal/packagepublish"
	"github.com/cozy-creator/cozy/internal/records"
	"github.com/cozy-creator/cozy/internal/wheel"
)

func TestRentalRunAcquiresCheapestOnceAndReleasesFailedPreAttempt(t *testing.T) {
	const releaseDigest = "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
	const modelManifest = "sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
	var mu sync.Mutex
	var descriptor []byte
	var descriptorDigest string
	var activeManifestDigest string
	var packageConfig, publishedWheel []byte
	var publishedWheelFact wheel.Identity
	var publishedWheelDigest string
	var origin string
	var rentalRequest map[string]any
	var createReason, deleteReason string
	var authPublic ed25519.PublicKey
	activeRentalID := "pr-managed-e2e"
	posts, deletes, packageDownloads, loginBegins, loginFinishes := 0, 0, 0, 0, 0
	enrollmentChallenge := bytes.Repeat([]byte{0x11}, 32)
	loginChallenge := bytes.Repeat([]byte{0x22}, 32)
	authExpires := time.Now().UTC().Add(15 * time.Minute).Format(time.RFC3339Nano)
	const enrollmentToken = "managed-rental-enrollment-token"
	const daemonToken = "managed-rental-daemon-token"
	root := t.TempDir()
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		mu.Lock()
		defer mu.Unlock()
		if strings.HasPrefix(r.URL.Path, "/v1/rentals") &&
			r.Header.Get("Authorization") != "Bearer "+daemonToken {
			t.Errorf("%s %s carried authorization %q", r.Method, r.URL.Path,
				r.Header.Get("Authorization"))
		}
		switch {
		case r.Method == http.MethodPost && r.URL.Path == "/v1/auth/device-keys/enroll/begin":
			var body struct {
				PublicKey string `json:"public_key"`
			}
			if !decodeAuthBody(t, r, &body) {
				return
			}
			decoded, err := authBase64.DecodeString(body.PublicKey)
			if err != nil || len(decoded) != ed25519.PublicKeySize {
				t.Errorf("invalid enrollment public key: %v", err)
				return
			}
			authPublic = append(ed25519.PublicKey(nil), decoded...)
			writeAuthJSON(t, w, http.StatusAccepted, map[string]string{
				"enrollment_id": "managed-rental-enrollment",
				"challenge":     authBase64.EncodeToString(enrollmentChallenge),
				"expires_at":    authExpires,
			})
		case r.Method == http.MethodPost && r.URL.Path == "/v1/auth/device-keys/enroll/finish":
			var body map[string]string
			if !decodeAuthBody(t, r, &body) {
				return
			}
			if body["enrollment_id"] != "managed-rental-enrollment" || body["code"] != "123456" ||
				!verifyAuthSignature(authPublic, "authkit.device-key-enrollment/1",
					enrollmentChallenge, body["signature"]) {
				t.Error("invalid managed-rental enrollment proof")
				return
			}
			writeAuthToken(t, w, enrollmentToken, authExpires)
		case r.Method == http.MethodPost && r.URL.Path == "/v1/auth/device-keys/login/begin":
			loginBegins++
			writeAuthJSON(t, w, http.StatusAccepted, map[string]string{
				"challenge_id": "managed-rental-login",
				"challenge":    authBase64.EncodeToString(loginChallenge),
				"expires_at":   authExpires,
			})
		case r.Method == http.MethodPost && r.URL.Path == "/v1/auth/device-keys/login/finish":
			var body map[string]string
			if !decodeAuthBody(t, r, &body) {
				return
			}
			if body["challenge_id"] != "managed-rental-login" ||
				!verifyAuthSignature(authPublic, "authkit.device-key-login/1",
					loginChallenge, body["signature"]) {
				t.Error("invalid managed-rental login proof")
				return
			}
			loginFinishes++
			writeAuthToken(t, w, daemonToken, authExpires)
		case r.Method == http.MethodGet && r.URL.Path == "/v1/auth/me":
			if r.Header.Get("Authorization") != "Bearer "+enrollmentToken {
				t.Errorf("enrollment identity read carried %q", r.Header.Get("Authorization"))
			}
			writeAuthJSON(t, w, http.StatusOK, map[string]any{
				"id": "managed-rental-user", "email": "person@example.com",
				"email_verified": true, "entitlements": []string{}, "availability": []any{},
			})
		case r.Method == http.MethodGet && r.URL.Path == "/v1/models/cozy/tiny":
			_ = json.NewEncoder(w).Encode(map[string]any{
				"model": map[string]any{"org": "cozy", "name": "tiny"},
				"releases": []map[string]any{{"release": "1.0.0", "lanes": []map[string]any{{
					"lane": "bf16", "manifest_id": modelManifest,
				}}}},
			})
		case r.Method == http.MethodGet && r.URL.Path == "/v1/packages/cozy/cozy-weightless-package/releases/1.0.0":
			_ = json.NewEncoder(w).Encode(map[string]any{
				"release": map[string]any{
					"release": "1.0.0", "release_digest": releaseDigest,
					"package_descriptor_digest": descriptorDigest,
					"package_descriptor_length": len(descriptor),
					"created_at":                "2026-08-30T00:00:00Z",
				},
				"document": map[string]any{}, "package_descriptor": json.RawMessage(descriptor),
			})
		case r.Method == http.MethodGet && r.URL.Path == "/v1/packages/cozy/cozy-weightless-package":
			_ = json.NewEncoder(w).Encode(map[string]any{
				"package": map[string]any{"org": "cozy", "name": "cozy-weightless-package",
					"created_at": "2026-08-30T00:00:00Z"},
				"releases": []map[string]any{{"release": "1.0.0", "cut_at": "2026-08-30T00:00:00Z"}},
			})
		case r.Method == http.MethodPost && r.URL.Path ==
			"/v1/packages/cozy/cozy-weightless-package/download":
			packageDownloads++
			configSum := sha256.Sum256(packageConfig)
			_ = json.NewEncoder(w).Encode(map[string]any{
				"downloads": []map[string]any{{
					"digest": publishedWheelDigest, "distribution": publishedWheelFact.Distribution,
					"import_roots": []string{"weightless"}, "kind": "project_wheel",
					"length": publishedWheelFact.Length, "path": publishedWheelFact.Filename,
					"tags": []string{"py3-none-any"}, "url": origin + "/files/published-wheel",
					"version": publishedWheelFact.Version,
				}},
				"package_config": map[string]any{"canonical_bytes": packageConfig,
					"digest": "sha256:" + hex.EncodeToString(configSum[:]), "length": len(packageConfig)},
				"package_descriptor": map[string]any{"canonical_bytes": descriptor,
					"digest": descriptorDigest, "length": len(descriptor)},
				"release": "1.0.0", "release_digest": releaseDigest,
			})
		case r.Method == http.MethodGet && r.URL.Path == "/files/published-wheel":
			_, _ = w.Write(publishedWheel)
		case r.Method == http.MethodGet && r.URL.Path == "/v1/rental-skus":
			_ = json.NewEncoder(w).Encode([]map[string]any{
				{"name": "cpu", "accelerator_model": "CPU", "price_usd_micros_per_hour": 70_000},
				{"name": "h200", "accelerator_model": "H200", "compute_capability": "9.0", "vram_gb": 141, "minimum_ram_per_gpu_gb": 128, "price_usd_micros_per_hour": 6_000_000},
				{"name": "cheap", "accelerator_model": "GPU C", "compute_capability": "8.9", "vram_gb": 24, "minimum_ram_per_gpu_gb": 32, "price_usd_micros_per_hour": 300_000},
			})
		case r.Method == http.MethodGet && r.URL.Path == "/v1/base-worker-manifests":
			raw, err := os.ReadFile(filepath.Join(root, "active-base.json"))
			if err != nil {
				t.Error(err)
				http.Error(w, "local base absent", http.StatusInternalServerError)
				return
			}
			sum := sha256.Sum256(raw)
			digest := "sha256:" + hex.EncodeToString(sum[:])
			activeManifestDigest = digest
			_ = json.NewEncoder(w).Encode([]map[string]any{{
				"wheelhouse_manifest_digest": digest,
				"wheelhouse_manifest":        json.RawMessage(raw),
				"base_worker_image_digest":   digest,
				"compatibility_profile":      map[string]any{},
				"platform_target":            map[string]any{},
				"activated_at":               "2026-08-31T00:00:00Z",
			}})
		case r.Method == http.MethodPost && r.URL.Path == "/v1/rentals":
			posts++
			createReason = r.Header.Get("X-Tensorhub-Reason")
			if err := json.NewDecoder(r.Body).Decode(&rentalRequest); err != nil {
				t.Error(err)
			}
			accelerator, rate := "GPU C", 300_000
			if rentalRequest["sku"] == "cpu" {
				accelerator, rate = "CPU", 70_000
			}
			w.WriteHeader(http.StatusAccepted)
			_ = json.NewEncoder(w).Encode(map[string]any{
				"rental_id": activeRentalID, "state": "pending_acquisition",
				"requested_accelerator_model": accelerator, "hourly_rate_usd_micros": rate,
				"wheelhouse_manifest_digest": activeManifestDigest,
			})
		case r.Method == http.MethodDelete && r.URL.Path == "/v1/rentals/"+activeRentalID:
			deletes++
			deleteReason = r.Header.Get("X-Tensorhub-Reason")
			w.WriteHeader(http.StatusNoContent)
		case r.Method == http.MethodGet && r.URL.Path == "/v1/rentals/pr-idle-h200":
			_ = json.NewEncoder(w).Encode(map[string]any{
				"rental_id": "pr-idle-h200", "state": "ready",
				"requested_accelerator_model": "H200", "hourly_rate_usd_micros": 6_000_000,
				"wheelhouse_manifest_digest": activeManifestDigest,
			})
		case r.Method == http.MethodGet && r.URL.Path == "/v1/rentals/pr-idle-cpu-old":
			_ = json.NewEncoder(w).Encode(map[string]any{
				"rental_id": "pr-idle-cpu-old", "state": "ready",
				"requested_accelerator_model": "CPU", "hourly_rate_usd_micros": 70_000,
				"wheelhouse_manifest_digest": "sha256:" + strings.Repeat("f", 64),
			})
		case r.Method == http.MethodGet && r.URL.Path == "/v1/rentals/"+activeRentalID:
			state := "ready"
			if deletes > 0 {
				state = "released"
			}
			accelerator, rate := "GPU C", 300_000
			if rentalRequest["sku"] == "cpu" {
				accelerator, rate = "CPU", 70_000
			}
			answer := map[string]any{
				"rental_id": activeRentalID, "state": state,
				"requested_accelerator_model": accelerator, "hourly_rate_usd_micros": rate,
				"wheelhouse_manifest_digest": activeManifestDigest,
			}
			if state == "ready" {
				answer["worker_address"] = "127.0.0.1:9443"
				answer["media_address"] = "127.0.0.1:9444"
				answer["worker_id"] = "worker-e2e"
				answer["worker_boot_id"] = "boot-e2e"
				answer["cert_pem"] = "not-a-certificate"
				answer["creator_public_key"] = rentalRequest["creator_public_key"]
				answer["media_token_sha256"] = []any{rentalRequest["media_token_sha256"]}
			}
			_ = json.NewEncoder(w).Encode(answer)
		default:
			http.NotFound(w, r)
		}
	}))
	defer server.Close()
	origin = server.URL

	login := runAuthCozy(t, root, server.URL, "123456\n", "auth", "login",
		"person@example.com", "--json")
	if login.code != 0 {
		t.Fatalf("managed-rental machine enrollment [exit %d]\nstdout: %s\nstderr: %s",
			login.code, login.stdout, login.stderr)
	}
	must(t, os.WriteFile(filepath.Join(root, "config.yaml"), []byte(
		"tensorhub_url: "+server.URL+"\nport: 0\n"+
			"rentals:\n  max_hourly_spend_usd: 7.00\n"), 0o600))
	t.Cleanup(func() { _, _ = runCozy(t, root, "down", "--all") })
	project := weightlessProject(t)
	pack, problem := packagepublish.PrepareLocalFrom(project)
	fatal(t, problem)
	fatal(t, pack.Build(context.Background()))
	publishedWheel, _ = os.ReadFile(pack.Wheel)
	publishedWheelSum := sha256.Sum256(publishedWheel)
	publishedWheelDigest = "sha256:" + hex.EncodeToString(publishedWheelSum[:])
	publishedWheelFact, problem = wheel.InspectIdentity(pack.Wheel)
	fatal(t, problem)
	packageConfig, _ = os.ReadFile(filepath.Join(project, "package.toml"))
	pack.Close()
	if code, out := runCozy(t, root, "package", "install", project, "--editable"); code != 0 {
		t.Fatalf("fixture install [exit %d]\n%s", code, out)
	}
	local := activePackageInstall(t, root)
	fatal(t, launch.RefreshGenerationBase(local, root,
		filepath.Join(root, "active-base.json"), childEnv(t, root)))
	descriptor, _ = os.ReadFile(launch.DescriptorPath(local.Dir))
	decoded, problem := launch.DecodeDescriptor(descriptor)
	fatal(t, problem)
	descriptorDigest = decoded.Digest
	store, problem := records.Open(filepath.Join(root, "records.db"))
	fatal(t, problem)
	fatal(t, store.RecordRental(records.Rental{ID: "pr-idle-cpu-old",
		MachineName: "idle-cpu-old", SKU: "cpu", AcceleratorModel: "CPU",
		HourlyRateUSDMicros: 70_000, State: "ready", Hub: server.URL,
		WheelhouseManifestDigest: "sha256:" + strings.Repeat("f", 64)}))
	store.Close()
	daemonPath := filepath.Join(root, "cozy-daemon")
	must(t, os.Symlink(cozyBin, daemonPath))
	daemonCommand := exec.Command(daemonPath)
	daemonCommand.Env = launch.GenerationToolEnv(local, childEnv(t, root))
	var daemonOutput bytes.Buffer
	daemonCommand.Stdout, daemonCommand.Stderr = &daemonOutput, &daemonOutput
	must(t, daemonCommand.Start())
	t.Cleanup(func() {
		if daemonCommand.Process != nil {
			_ = daemonCommand.Process.Signal(os.Interrupt)
			_, _ = daemonCommand.Process.Wait()
		}
	})
	readyDeadline := time.Now().Add(5 * time.Second)
	for !daemon.Probe(config.Config{Home: root}).Up && time.Now().Before(readyDeadline) {
		time.Sleep(25 * time.Millisecond)
	}
	if !daemon.Probe(config.Config{Home: root}).Up {
		t.Fatalf("foreground daemon did not start:\n%s", daemonOutput.String())
	}

	code, out := runCozy(t, root, "run", localWeightlessRef+"/tile", "size=32", "seed=7",
		"--rental", "--idempotency-key", "managed-e2e", "--json")
	if code != 1 || !strings.Contains(out, `pinned media certificate`) {
		t.Fatalf("three-second observation missed the fast rental failure [exit %d]\n%s\ndaemon:\n%s",
			code, out, daemonOutput.String())
	}
	deadline := time.Now().Add(10 * time.Second)
	var request *records.Request
	var events []records.Event
	for time.Now().Before(deadline) {
		store, problem = records.Open(filepath.Join(root, "records.db"))
		fatal(t, problem)
		rows, rowsProblem := store.RequestsOfKind("serving", "", 20)
		fatal(t, rowsProblem)
		for i := range rows {
			if rows[i].IdemKey == "managed-e2e" {
				row := rows[i]
				request = &row
				events, _ = store.EventsAfter(row.ID, 0, 100)
			}
		}
		rentalRow, _ := store.RentalRow("pr-managed-e2e")
		store.Close()
		mu.Lock()
		done := posts == 1 && deletes == 1 && rentalRow == nil && request != nil && request.State == "failed"
		mu.Unlock()
		if done {
			break
		}
		time.Sleep(25 * time.Millisecond)
	}
	mu.Lock()
	gotPosts, gotDeletes, body := posts, deletes, rentalRequest
	gotCreateReason, gotDeleteReason := createReason, deleteReason
	mu.Unlock()
	if request == nil || request.Worker != "pr-managed-e2e" || request.InstallID != local.ID ||
		request.Release != "1.0.0" || request.PackageRevisionDigest != local.SourceDigest ||
		request.PrivatePackageDigest == "" || len(request.Models) != 0 ||
		len(request.AcceptableWheelhouseManifestDigests) != 1 ||
		request.AcceptableWheelhouseManifestDigests[0] != activeManifestDigest ||
		gotPosts != 1 || gotDeletes != 1 ||
		body["sku"] != "cpu" || fmt.Sprint(body["acceptable_wheelhouse_manifest_digests"]) !=
		"["+activeManifestDigest+"]" ||
		body["max_cost_usd_micros"] != nil || body["max_duration_seconds"] != nil {
		t.Fatalf("managed lifecycle request=%+v posts=%d deletes=%d body=%v", request, gotPosts, gotDeletes, body)
	}
	if gotCreateReason != "" || gotDeleteReason != "" ||
		strings.Contains(fmt.Sprint(body), localWeightlessRef) || strings.Contains(fmt.Sprint(body), "cozy/tiny") {
		t.Fatalf("managed rental leaked work identity in reason/body: create=%q delete=%q body=%v",
			gotCreateReason, gotDeleteReason, body)
	}
	wantLines := map[string]bool{
		"rentals: 1 remote machine running · $0.07/hour of $7.00/hour":  false,
		"rentals: 2 remote machines running · $0.14/hour of $7.00/hour": false,
	}
	for _, event := range events {
		if line, ok := event.Payload["line"].(string); ok {
			if _, wanted := wantLines[line]; wanted {
				wantLines[line] = true
			}
		}
	}
	for line, seen := range wantLines {
		if !seen {
			t.Fatalf("managed lifecycle omitted %q: %+v", line, events)
		}
	}
	code, out = runCozy(t, root, "run", localWeightlessRef+"/tile", "size=32", "seed=7",
		"--rental", "--idempotency-key", "managed-e2e", "--json")
	mu.Lock()
	gotPosts = posts
	mu.Unlock()
	if code != 1 || !strings.Contains(out, `pinned media certificate`) || gotPosts != 1 {
		t.Fatalf("exact replay purchased again [exit %d posts=%d]\n%s", code, gotPosts, out)
	}

	mu.Lock()
	posts, deletes, rentalRequest, createReason, deleteReason = 0, 0, nil, "", ""
	activeRentalID = "pr-private-job-replay"
	mu.Unlock()
	code, out = runCozy(t, root, "run", localWeightlessRef+"/tile_job", "size=32", "seed=7",
		"--rental", "--idempotency-key", "private-job-replay", "--json")
	if code != 1 || !strings.Contains(out, `pinned media certificate`) {
		t.Fatalf("private job submission [exit %d]\n%s\ndaemon:\n%s", code, out, daemonOutput.String())
	}
	deadline = time.Now().Add(10 * time.Second)
	request = nil
	for time.Now().Before(deadline) {
		store, problem = records.Open(filepath.Join(root, "records.db"))
		fatal(t, problem)
		rows, rowsProblem := store.RequestsOfKind("job", "", 20)
		fatal(t, rowsProblem)
		for i := range rows {
			if rows[i].IdemKey == "private-job-replay" {
				row := rows[i]
				request = &row
			}
		}
		rentalRow, _ := store.RentalRow("pr-private-job-replay")
		store.Close()
		mu.Lock()
		done := posts == 1 && deletes == 1 && rentalRow == nil && request != nil && request.State == "failed"
		mu.Unlock()
		if done {
			break
		}
		time.Sleep(25 * time.Millisecond)
	}
	if request == nil || request.PrivatePackageDigest == "" || request.InstallID == "" {
		t.Fatalf("private job did not freeze its exact install/revision: %+v", request)
	}
	must(t, os.RemoveAll(project))
	code, out = runCozy(t, root, "run", localWeightlessRef+"/tile_job", "size=32", "seed=7",
		"--rental", "--idempotency-key", "private-job-replay", "--json")
	mu.Lock()
	gotPosts = posts
	mu.Unlock()
	if code != 1 || !strings.Contains(out, `pinned media certificate`) || gotPosts != 1 {
		t.Fatalf("private job replay reread source or rented again [exit %d posts=%d]\n%s",
			code, gotPosts, out)
	}
	store, problem = records.Open(filepath.Join(root, "records.db"))
	fatal(t, problem)
	replayedRequest, problem := store.RequestByIdempotencyKey("private-job-replay")
	store.Close()
	fatal(t, problem)
	if replayedRequest == nil || replayedRequest.ID != request.ID ||
		replayedRequest.PrivatePackageDigest != request.PrivatePackageDigest {
		t.Fatalf("private job replay changed durable identity: %+v -> %+v", request, replayedRequest)
	}
	code, out = runCozy(t, root, "run", localWeightlessRef+"/tile_job", "size=64", "seed=7",
		"--rental", "--idempotency-key", "private-job-replay", "--json")
	mu.Lock()
	gotPosts = posts
	mu.Unlock()
	if code == 0 || !strings.Contains(out, "different body") || gotPosts != 1 {
		t.Fatalf("changed private job payload reused the key [exit %d posts=%d]\n%s",
			code, gotPosts, out)
	}

	// Preserve the published remote-job acquisition/release/replay acceptance beside the
	// new private-serving case; a new transport must not erase an existing product arm.
	var descriptorDocument map[string]any
	must(t, json.Unmarshal(descriptor, &descriptorDocument))
	entrypoints := descriptorDocument["entrypoints"].([]any)
	for _, entrypoint := range entrypoints {
		row := entrypoint.(map[string]any)
		if row["name"] == "tile" {
			encoded, _ := json.Marshal(row)
			var job map[string]any
			must(t, json.Unmarshal(encoded, &job))
			job["name"], job["publishes"] = "tile_job", false
			descriptorDocument["jobs"] = []any{job}
			row["models"] = []any{map[string]any{
				"class": "TinyModel", "path": "tile.models.model",
				"stamps": map[string]any{}, "component_use": map[string]any{"core": []any{"tile"}},
			}}
			descriptorDocument["entrypoints"] = []any{row}
		}
	}
	publishedDescriptor, _ := json.Marshal(descriptorDocument)
	publishedDecoded, problem := launch.DecodeDescriptor(publishedDescriptor)
	fatal(t, problem)
	jobDescriptorID := publishedDecoded.Jobs[0].DescriptorID
	if jobDescriptorID == "" {
		t.Fatal("remote job descriptor id was not derived")
	}
	mu.Lock()
	descriptor = publishedDescriptor
	descriptorDigest = publishedDecoded.Digest
	posts, deletes, rentalRequest, createReason, deleteReason = 0, 0, nil, "", ""
	activeRentalID = "pr-managed-job"
	mu.Unlock()
	store, problem = records.Open(filepath.Join(root, "records.db"))
	fatal(t, problem)
	fatal(t, store.RecordRental(records.Rental{
		ID: "pr-idle-h200", MachineName: "idle-h200", SKU: "h200", AcceleratorModel: "H200",
		HourlyRateUSDMicros: 6_000_000, State: "ready", Hub: server.URL,
		WheelhouseManifestDigest: activeManifestDigest,
	}))
	store.Close()
	code, out = runCozy(t, root, "run", weightlessRef+"/tile_job", "size=32", "seed=7",
		"--rental", "--idempotency-key", "managed-job", "--json")
	if code != 1 || !strings.Contains(out, `pinned media certificate`) {
		t.Fatalf("three-second observation missed the fast remote job failure [exit %d]\n%s\ndaemon:\n%s",
			code, out, daemonOutput.String())
	}
	deadline = time.Now().Add(10 * time.Second)
	request = nil
	for time.Now().Before(deadline) {
		store, problem = records.Open(filepath.Join(root, "records.db"))
		fatal(t, problem)
		rows, rowsProblem := store.RequestsOfKind("job", "", 20)
		fatal(t, rowsProblem)
		for i := range rows {
			if rows[i].IdemKey == "managed-job" {
				row := rows[i]
				request = &row
			}
		}
		rentalRow, _ := store.RentalRow("pr-managed-job")
		store.Close()
		mu.Lock()
		done := posts == 1 && deletes == 1 && rentalRow == nil && request != nil && request.State == "failed"
		mu.Unlock()
		if done {
			break
		}
		time.Sleep(25 * time.Millisecond)
	}
	mu.Lock()
	gotPosts, gotDeletes, body = posts, deletes, rentalRequest
	mu.Unlock()
	if request == nil || request.Worker != "pr-managed-job" || request.PlanID != jobDescriptorID || request.Kind != "job" ||
		request.Release != "1.0.0" || request.PackageRevisionDigest != releaseDigest ||
		request.JobGPUCount != 0 || len(request.Models) != 0 ||
		len(request.AcceptableWheelhouseManifestDigests) != 1 ||
		request.AcceptableWheelhouseManifestDigests[0] != activeManifestDigest ||
		gotPosts != 1 || gotDeletes != 1 ||
		body["sku"] != "cpu" ||
		strings.Contains(fmt.Sprint(body), weightlessRef) || strings.Contains(fmt.Sprint(body), "tile_job") {
		t.Fatalf("remote job lifecycle request=%+v posts=%d deletes=%d body=%v",
			request, gotPosts, gotDeletes, body)
	}
	code, out = runCozy(t, root, "run", weightlessRef+"/tile_job", "size=32", "seed=7",
		"--rental", "--idempotency-key", "managed-job", "--json")
	mu.Lock()
	gotPosts = posts
	mu.Unlock()
	if code != 1 || !strings.Contains(out, `pinned media certificate`) || gotPosts != 1 {
		t.Fatalf("remote job replay purchased again [exit %d posts=%d]\n%s", code, gotPosts, out)
	}
	if packageDownloads != 1 {
		t.Fatalf("published package preflight downloads=%d, want one exact resolution", packageDownloads)
	}
	if loginBegins != 1 || loginFinishes != 1 {
		t.Fatalf("daemon machine login exchanges = begin:%d finish:%d, want one each",
			loginBegins, loginFinishes)
	}
	credentials, err := filepath.Glob(filepath.Join(root, "auth", "*.json"))
	if err != nil || len(credentials) != 1 {
		t.Fatalf("machine credential files = %v, error = %v", credentials, err)
	}
	storedCredential, err := os.ReadFile(credentials[0])
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(string(storedCredential), enrollmentToken) ||
		strings.Contains(string(storedCredential), daemonToken) {
		t.Fatal("short AuthKit access token was persisted")
	}
	store, problem = records.Open(filepath.Join(root, "records.db"))
	fatal(t, problem)
	_, problem = store.ForgetRental("pr-idle-h200")
	fatal(t, problem)
	store.Close()
}
