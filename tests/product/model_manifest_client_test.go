package producttest

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/cozy-creator/cozy/internal/config"
	"github.com/cozy-creator/cozy/internal/hub"
	"github.com/cozy-creator/cozy/internal/secret"
	"github.com/cozy-creator/cozy/internal/tfs"
	"github.com/cozy-creator/cozy/internal/transfer"
)

const (
	manifestA = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
	topologyC = "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc"
)

func releaseEvidence() []byte {
	return []byte(`{"classification_digest":"sha256:` + topologyC + `"}`)
}

func TestLocalDownloadRequiresRealReleaseCoordinates(t *testing.T) {
	manifestID := "sha256:" + manifestA
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet || r.URL.Path != "/v1/models/resolve" {
			t.Fatalf("resolve request = %s %s", r.Method, r.URL.Path)
		}
		_, _ = io.WriteString(w, `{"model":"acme/model","manifest_id":"`+manifestID+`","header_digest":"sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb","objects":1,"bytes":7}`)
	}))
	defer server.Close()

	fetch := transfer.Fetch{Hub: hub.New(config.Config{HubURL: server.URL}, "test"), Spec: "acme/model@" + manifestID}
	_, problem := fetch.Resolve(context.Background())
	if problem == nil || problem.Name != "model.release_required" {
		t.Fatalf("digest-only local resolution = %v", problem)
	}
}

func TestLocalDownloadPreservesExactReleaseEvidence(t *testing.T) {
	evidence := releaseEvidence()
	encoded := base64.StdEncoding.EncodeToString(evidence)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = io.WriteString(w, `{"model":"acme/model","release":"1.0.0","lane":"bf16","manifest_id":"sha256:`+
			manifestA+`","header_digest":"sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb","objects":1,"bytes":7,"release_evidence_base64":"`+encoded+`"}`)
	}))
	defer server.Close()

	fetch := transfer.Fetch{Hub: hub.New(config.Config{HubURL: server.URL}, "test"), Spec: "acme/model@1.0.0"}
	resolved, problem := fetch.Resolve(context.Background())
	if problem != nil || !reflect.DeepEqual(resolved.ReleaseEvidence, evidence) {
		t.Fatalf("resolved release evidence = %q, %v", resolved.ReleaseEvidence, problem)
	}
}

func TestLocalDownloadRejectsInvalidReleaseEvidence(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = io.WriteString(w, `{"model":"acme/model","release":"1.0.0","lane":"bf16","manifest_id":"sha256:`+
			manifestA+`","header_digest":"sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb","objects":1,"bytes":7,"release_evidence_base64":"not-base64"}`)
	}))
	defer server.Close()

	fetch := transfer.Fetch{Hub: hub.New(config.Config{HubURL: server.URL}, "test"), Spec: "acme/model@1.0.0"}
	_, problem := fetch.Resolve(context.Background())
	if problem == nil || problem.Name != "hub.release_evidence_invalid" {
		t.Fatalf("invalid release evidence = %v", problem)
	}
}

func releaseRow(evidence, lane string) string {
	return `{"evidence":` + evidence + `,"lane":"` + lane + `","manifest_length":2,"manifest_sha256":"` +
		manifestA + `","name":"model","org":"acme","version":"1.0.0"}`
}

func tfsRows(t *testing.T, rows string) *tfs.Tool {
	t.Helper()
	dir := t.TempDir()
	rowsPath := filepath.Join(dir, "rows.jsonl")
	if err := os.WriteFile(rowsPath, []byte(rows), 0o600); err != nil {
		t.Fatal(err)
	}
	script := filepath.Join(dir, "tfs")
	body := "#!/bin/sh\n/usr/bin/cp '" + rowsPath + "' \"$5\"\n"
	if err := os.WriteFile(script, []byte(body), 0o700); err != nil {
		t.Fatal(err)
	}
	return &tfs.Tool{Bin: script, Root: dir}
}

func TestLocalPublishRequiresOneExactReleaseEvidence(t *testing.T) {
	exact := string(releaseEvidence())
	tool := tfsRows(t, releaseRow(exact, "a")+"\n"+releaseRow(exact, "b")+"\n")
	evidence, problem := tool.ReleaseEvidence("acme", "model", "sha256:"+manifestA,
		filepath.Join(t.TempDir(), "selected.jsonl"))
	if problem != nil || string(evidence) != exact {
		t.Fatalf("exact repeated evidence = %q, %v", evidence, problem)
	}

	absent := tfsRows(t, "")
	if _, problem := absent.ReleaseEvidence("acme", "model", "sha256:"+manifestA,
		filepath.Join(t.TempDir(), "absent.jsonl")); problem == nil || problem.Name != "model.release_evidence_absent" {
		t.Fatalf("unrooted manifest evidence = %v", problem)
	}

	other := `{"classification_digest":"sha256:dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd"}`
	ambiguous := tfsRows(t, releaseRow(exact, "a")+"\n"+releaseRow(other, "b")+"\n")
	if _, problem := ambiguous.ReleaseEvidence("acme", "model", "sha256:"+manifestA,
		filepath.Join(t.TempDir(), "ambiguous.jsonl")); problem == nil || problem.Name != "model.release_evidence_ambiguous" {
		t.Fatalf("ambiguous manifest evidence = %v", problem)
	}
}

func TestLocalReleaseCommitCarriesExactEvidence(t *testing.T) {
	dir := t.TempDir()
	empty, captured := filepath.Join(dir, "empty"), filepath.Join(dir, "mutation.json")
	if err := os.WriteFile(empty, nil, 0o600); err != nil {
		t.Fatal(err)
	}
	script := filepath.Join(dir, "tfs")
	body := "#!/bin/sh\n" +
		"if [ \"$2\" = list ]; then /usr/bin/cp '" + empty + "' \"$5\"; exit; fi\n" +
		"if [ \"$2\" = commit ]; then /usr/bin/cp \"$5\" '" + captured + "'; exit; fi\n" +
		"exit 1\n"
	if err := os.WriteFile(script, []byte(body), 0o700); err != nil {
		t.Fatal(err)
	}
	tool := &tfs.Tool{Bin: script, Root: dir}
	evidence := releaseEvidence()
	if problem := tool.CommitRelease("acme", "model", "1.0.0", "bf16",
		"sha256:"+manifestA, 2, evidence, t.TempDir()); problem != nil {
		t.Fatal(problem)
	}
	var mutation map[string]any
	if raw, err := os.ReadFile(captured); err != nil {
		t.Fatal(err)
	} else if err := json.Unmarshal(raw, &mutation); err != nil {
		t.Fatal(err)
	}
	if mutation["evidence_base64"] != base64.StdEncoding.EncodeToString(evidence) {
		t.Fatalf("commit mutation evidence = %#v", mutation["evidence_base64"])
	}
	if problem := tool.CommitRelease("acme", "model", "1.0.0", "bf16",
		"sha256:"+manifestA, 2, nil, t.TempDir()); problem == nil {
		t.Fatal("commit accepted missing release evidence")
	}
}

func TestManifestReadRoutes(t *testing.T) {
	manifestID := "sha256:" + manifestA
	manifest := []byte(`{"entries":[]}`)
	var calls int
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls++
		wantPath := "/v1/models/acme/model/manifests/" + manifestID
		if calls == 1 {
			if r.Method != http.MethodGet || r.URL.Path != wantPath {
				t.Fatalf("manifest request = %s %s", r.Method, r.URL.Path)
			}
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write(manifest)
			return
		}
		if r.Method != http.MethodPost || r.URL.Path != wantPath+"/reads" {
			t.Fatalf("reads request = %s %s", r.Method, r.URL.Path)
		}
		var body map[string][]string
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil ||
			!reflect.DeepEqual(body, map[string][]string{"object_ids": {"sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"}}) {
			t.Fatalf("reads body = %#v, %v", body, err)
		}
		_, _ = io.WriteString(w, `{"reads":[{"object_id":"sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb","length":7,"url":"https://objects.invalid/blob","expires_at":"2026-08-30T00:00:00Z"}]}`)
	}))
	defer server.Close()

	client := hub.New(config.Config{HubURL: server.URL}, "test")
	ref := hub.Ref{Org: "acme", Name: "model"}
	raw, problem := client.Manifest(context.Background(), ref, manifestID)
	if problem != nil || !reflect.DeepEqual(raw, manifest) {
		t.Fatalf("Manifest = %q, %v", raw, problem)
	}
	reads, problem := client.Reads(context.Background(), ref, manifestID,
		[]string{"sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"})
	if problem != nil || len(reads) != 1 || reads[0].Length != 7 {
		t.Fatalf("Reads = %#v, %v", reads, problem)
	}
}

func TestPublicationUsesReleaseLaneAndManifestOnlySeal(t *testing.T) {
	evidenceBase64 := base64.StdEncoding.EncodeToString(releaseEvidence())
	var calls int
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls++
		if r.Header.Get("Authorization") != "Bearer token" || r.Header.Get("X-Tensorhub-Reason") != "proof" {
			t.Fatalf("mutation headers = %#v", r.Header)
		}
		switch calls {
		case 1:
			if r.Method != http.MethodPut || r.URL.Path != "/v1/models/acme/model/publications/manifest-proof" {
				t.Fatalf("open request = %s %s", r.Method, r.URL.Path)
			}
			var body map[string]any
			if err := json.NewDecoder(r.Body).Decode(&body); err != nil || len(body) != 3 ||
				body["release"] != "1.0.0" || body["lane"] != "bf16" {
				t.Fatalf("open body = %#v, %v", body, err)
			}
			_, _ = io.WriteString(w, `{"created":true,"publication":{"operation":"manifest-proof","release":"1.0.0","lane":"bf16","state":"open","objects":[{"object_id":"sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb","length":7,"state":"claimed"}]}}`)
		case 2:
			if r.Method != http.MethodPost || r.URL.Path != "/v1/models/acme/model/publications/manifest-proof/seal" {
				t.Fatalf("seal request = %s %s", r.Method, r.URL.Path)
			}
			var body map[string]any
			if err := json.NewDecoder(r.Body).Decode(&body); err != nil || len(body) != 2 || body["manifest"] != "e30=" ||
				body["release_evidence_base64"] != evidenceBase64 {
				t.Fatalf("seal body = %#v, %v", body, err)
			}
			_, _ = io.WriteString(w, `{"publish_id":"manifest-proof","release":"1.0.0","lane":"bf16","manifest":{"sha256":"`+manifestA+`","length":2},"topology_digest":"sha256:`+topologyC+`","objects":1,"bytes":7,"release_evidence_base64":"`+evidenceBase64+`","duplicate":false}`)
		default:
			t.Fatalf("unexpected request %s %s", r.Method, r.URL.Path)
		}
	}))
	defer server.Close()

	client := hub.New(config.Config{HubURL: server.URL, HubToken: secret.New("token")}, "test")
	ref := hub.Ref{Org: "acme", Name: "model"}
	opened, problem := client.OpenPublication(context.Background(), ref, "manifest-proof",
		"1.0.0", "bf16", []hub.Object{{ID: "sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb", Length: 7}}, "proof")
	if problem != nil || !opened.Created || opened.Publication.Operation != "manifest-proof" {
		t.Fatalf("OpenPublication = %#v, %v", opened, problem)
	}
	done, problem := client.SealPublication(context.Background(), ref, "manifest-proof",
		hub.SealPublicationRequest{Manifest: "e30=", ReleaseEvidenceBase64: evidenceBase64}, "proof")
	if problem != nil || done.Manifest.Length != 2 || done.Release != "1.0.0" || done.Lane != "bf16" ||
		done.ReleaseEvidenceBase64 != evidenceBase64 {
		t.Fatalf("SealPublication = %#v, %v", done, problem)
	}
}

func TestModelPublicationBatchesGrantRequestsAndFinalizesOnce(t *testing.T) {
	const objectCount = 257
	manifestID := "sha256:" + manifestA
	manifest := []byte(`{}`)
	evidence := releaseEvidence()
	evidenceBase64 := base64.StdEncoding.EncodeToString(evidence)

	transfers := make([]hub.Transfer, 0, objectCount)
	var refs strings.Builder
	for i := range objectCount {
		objectID := "sha256:" + fmt.Sprintf("%064x", i+1)
		transfers = append(transfers, hub.Transfer{ObjectID: objectID, Length: 1, State: "claimed"})
		fmt.Fprintf(&refs, `{"sha256":"%s","length":1}`+"\n", strings.TrimPrefix(objectID, "sha256:"))
	}

	toolDir := t.TempDir()
	releasesPath := filepath.Join(toolDir, "releases.jsonl")
	refsPath := filepath.Join(toolDir, "refs.jsonl")
	manifestPath := filepath.Join(toolDir, "manifest.json")
	if err := os.WriteFile(releasesPath, []byte(releaseRow(string(evidence), "bf16")+"\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(refsPath, []byte(refs.String()), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(manifestPath, manifest, 0o600); err != nil {
		t.Fatal(err)
	}
	tfsPath := filepath.Join(toolDir, "tfs")
	script := "#!/bin/sh\n" +
		"if [ \"$1:$2\" = repo:list ]; then /usr/bin/cp '" + releasesPath + "' \"$5\"; exit; fi\n" +
		"if [ \"$1:$2\" = manifest:walk ]; then /usr/bin/cp '" + refsPath + "' \"$6\"; exit; fi\n" +
		"if [ \"$1:$2\" = manifest:get ]; then /usr/bin/cp '" + manifestPath + "' \"$6\"; exit; fi\n" +
		"if [ \"$1\" = get ]; then printf x > \"$5\"; exit; fi\n" +
		"exit 1\n"
	if err := os.WriteFile(tfsPath, []byte(script), 0o700); err != nil {
		t.Fatal(err)
	}
	tool := &tfs.Tool{Bin: tfsPath, Root: toolDir}

	var grantCalls atomic.Int32
	publicationPath := "/v1/models/acme/model/publications/batch-proof"
	var server *httptest.Server
	server = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch {
		case r.Method == http.MethodPut && r.URL.Path == publicationPath:
			_ = json.NewEncoder(w).Encode(hub.OpenPublicationResponse{Created: true,
				Publication: hub.Session{Operation: "batch-proof", Release: "1.0.0", Lane: "bf16", State: "open", Objects: transfers}})
		case r.Method == http.MethodPost && r.URL.Path == publicationPath+"/grants":
			var body struct {
				ObjectIDs []string `json:"object_ids"`
			}
			if err := json.NewDecoder(r.Body).Decode(&body); err != nil || len(body.ObjectIDs) == 0 || len(body.ObjectIDs) > 128 {
				t.Errorf("grant batch size = %d, %v", len(body.ObjectIDs), err)
			}
			grantCalls.Add(1)
			answer := hub.GrantResponse{Grants: make([]hub.Grant, 0, len(body.ObjectIDs))}
			for _, objectID := range body.ObjectIDs {
				answer.Grants = append(answer.Grants, hub.Grant{ObjectID: objectID, Length: 1,
					URL: server.URL + "/objects/" + objectID})
			}
			_ = json.NewEncoder(w).Encode(answer)
		case r.Method == http.MethodPut && strings.HasPrefix(r.URL.Path, "/objects/"):
			raw, err := io.ReadAll(r.Body)
			if err != nil || string(raw) != "x" {
				t.Errorf("object upload = %q, %v", raw, err)
			}
			w.WriteHeader(http.StatusOK)
		case r.Method == http.MethodPost && r.URL.Path == publicationPath+"/seal":
			_ = json.NewEncoder(w).Encode(hub.CompleteResponse{PublishID: "batch-proof",
				Release: "1.0.0", Lane: "bf16", Manifest: hub.ManifestRef{SHA256: manifestA, Length: int64(len(manifest))},
				TopologyDigest: "sha256:" + topologyC, Objects: objectCount, Bytes: objectCount,
				ReleaseEvidenceBase64: evidenceBase64})
		default:
			t.Errorf("unexpected request %s %s", r.Method, r.URL.Path)
			http.Error(w, "unexpected", http.StatusNotFound)
		}
	}))
	defer server.Close()

	publish := transfer.Publish{Tool: tool,
		Hub: hub.New(config.Config{HubURL: server.URL, HubToken: secret.New("token")}, "test"),
		Ref: hub.Ref{Org: "acme", Name: "model"}, ManifestID: manifestID,
		Release: "1.0.0", Lane: "bf16", Session: "batch-proof", Reason: "proof", Scratch: t.TempDir()}
	result, problem := publish.Run(context.Background())
	if problem != nil {
		t.Fatal(problem)
	}
	if grantCalls.Load() != 3 {
		t.Fatalf("control requests: grants=%d, want 3", grantCalls.Load())
	}
	if result.Uploaded != objectCount || result.Verified != objectCount || result.Moved != objectCount {
		t.Fatalf("publication result = %#v", result)
	}
}
