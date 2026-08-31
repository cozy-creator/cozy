package producttest

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/base64"
	"encoding/hex"
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
	namedLane = "bf16"
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

func TestPublicationUsesCurrentPrepareAndCutContract(t *testing.T) {
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
				body["release"] != "1.0.0" || body["lane"] != nil || body["lane_key"] != namedLane ||
				body["required_contract"] != nil {
				t.Fatalf("open body = %#v, %v", body, err)
			}
			_, _ = io.WriteString(w, `{"created":true,"publication":{"operation":"manifest-proof","release":"1.0.0","lane":"","lane_key":"`+namedLane+`","state":"open","objects":[{"object_id":"sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb","length":7,"state":"claimed"}]}}`)
		case 2:
			if r.Method != http.MethodPost || r.URL.Path != "/v1/models/acme/model/publications/manifest-proof/grants" {
				t.Fatalf("grant request = %s %s", r.Method, r.URL.Path)
			}
			_, _ = io.WriteString(w, `{"grants":[],"held":[{"transfer_id":"transfer-proof","object_id":"sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb","length":7,"state":"accepted","received_bytes":7,"verified_bytes":7,"last_progress_at":"2026-08-31T00:00:00Z","accepted_at":"2026-08-31T00:00:00Z"}]}`)
		case 3:
			if r.Method != http.MethodPost || r.URL.Path != "/v1/models/acme/model/publications/manifest-proof/finalize" {
				t.Fatalf("finalize request = %s %s", r.Method, r.URL.Path)
			}
			var body map[string]any
			if err := json.NewDecoder(r.Body).Decode(&body); err != nil || len(body) != 3 ||
				body["manifest_id"] != "sha256:"+manifestA || body["manifest_length"] != float64(2) ||
				body["release_evidence_base64"] != evidenceBase64 {
				t.Fatalf("finalize body = %#v, %v", body, err)
			}
			_, _ = io.WriteString(w, `{"publish_id":"manifest-proof","release":"1.0.0","lane":"`+namedLane+`","manifest":{"sha256":"`+manifestA+`","length":2},"contract":{"stamps":{},"structure":"sha256:`+topologyC+`","encoding":{"set":["bf16"]}},"topology_digest":"sha256:`+topologyC+`","objects":1,"bytes":7,"release_evidence_base64":"`+evidenceBase64+`","state":"verified/prepared","duplicate":false}`)
		case 4:
			if r.Method != http.MethodPost || r.URL.Path != "/v1/models/acme/model/releases/1.0.0" {
				t.Fatalf("cut request = %s %s", r.Method, r.URL.Path)
			}
			var body struct {
				Operation    string   `json:"operation"`
				Publications []string `json:"publications"`
			}
			if err := json.NewDecoder(r.Body).Decode(&body); err != nil || body.Operation != "cut-manifest-proof" ||
				!reflect.DeepEqual(body.Publications, []string{"manifest-proof"}) {
				t.Fatalf("cut body = %#v, %v", body, err)
			}
			_, _ = io.WriteString(w, `{"operation":"cut-manifest-proof","release":"1.0.0","repository_sha256":"`+manifestA+`","lanes":[{"lane":"`+namedLane+`","manifest":{"sha256":"`+manifestA+`","length":2},"contract":{"stamps":{},"structure":"sha256:`+topologyC+`","encoding":{"set":["bf16"]}},"objects":1,"bytes":7,"publication":"manifest-proof"}],"duplicate":false}`)
		default:
			t.Fatalf("unexpected request %s %s", r.Method, r.URL.Path)
		}
	}))
	defer server.Close()

	client := hub.New(config.Config{HubURL: server.URL, HubToken: secret.New("token")}, "test")
	ref := hub.Ref{Org: "acme", Name: "model"}
	opened, problem := client.OpenPublication(context.Background(), ref, "manifest-proof",
		"1.0.0", namedLane, []hub.Object{{ID: "sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb", Length: 7}}, "proof")
	if problem != nil || !opened.Created || opened.Publication.Operation != "manifest-proof" ||
		opened.Publication.Lane != "" || opened.Publication.LaneKey != namedLane {
		t.Fatalf("OpenPublication = %#v, %v", opened, problem)
	}
	granted, problem := client.GrantKnownTransfers(context.Background(), ref, "manifest-proof",
		[]string{"sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"}, "proof")
	if problem != nil || len(granted.Grants) != 0 || len(granted.Held) != 1 || granted.Held[0].VerifiedBytes != 7 {
		t.Fatalf("GrantKnownTransfers = %#v, %v", granted, problem)
	}
	prepared, problem := client.FinalizePublication(context.Background(), ref, "manifest-proof",
		hub.FinalizePublicationRequest{ManifestID: "sha256:" + manifestA, ManifestLength: 2,
			ReleaseEvidenceBase64: evidenceBase64}, "proof")
	if problem != nil || prepared.Manifest.Length != 2 || prepared.Release != "1.0.0" ||
		prepared.Lane != namedLane || prepared.ReleaseEvidenceBase64 != evidenceBase64 {
		t.Fatalf("FinalizePublication = %#v, %v", prepared, problem)
	}
	cut, problem := client.CutRelease(context.Background(), ref, "1.0.0", "cut-manifest-proof",
		[]string{"manifest-proof"}, "proof")
	if problem != nil || cut.Operation != "cut-manifest-proof" || len(cut.Lanes) != 1 ||
		cut.Lanes[0].Publication != "manifest-proof" || cut.Lanes[0].Lane != namedLane {
		t.Fatalf("CutRelease = %#v, %v", cut, problem)
	}
}

func TestPublicationResponsesRejectUnknownFields(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = io.WriteString(w, `{"created":true,"publication":{"operation":"strict-proof","release":"1.0.0","lane":"","lane_key":"bf16","state":"open","objects":[]},"unrecognized":true}`)
	}))
	defer server.Close()

	client := hub.New(config.Config{HubURL: server.URL, HubToken: secret.New("token")}, "test")
	_, problem := client.OpenPublication(context.Background(), hub.Ref{Org: "acme", Name: "model"},
		"strict-proof", "1.0.0", namedLane, []hub.Object{{ID: "sha256:" + manifestA, Length: 1}}, "proof")
	if problem == nil || problem.Name != "hub.unreadable_answer" {
		t.Fatalf("unknown publication response field = %v", problem)
	}
}

func TestNamedLanePublicationOperationIdentity(t *testing.T) {
	ref := hub.Ref{Org: "acme", Name: "model"}
	manifestID := "sha256:" + manifestA
	exact := transfer.PublicationOperationID(ref, "1.0.0", namedLane, manifestID)
	wantSum := sha256.Sum256([]byte("model-publication-named-lane/1\x00" + ref.String() +
		"\x00" + "1.0.0" + "\x00" + namedLane + "\x00" + manifestID))
	if want := "manifest-" + hex.EncodeToString(wantSum[:]); exact != want {
		t.Fatalf("named-lane operation = %s, want %s", exact, want)
	}
	if replay := transfer.PublicationOperationID(ref, "1.0.0", namedLane, manifestID); replay != exact {
		t.Fatalf("exact named-lane replay operation = %s, want %s", replay, exact)
	}
	if changed := transfer.PublicationOperationID(ref, "1.0.0", "fp16", manifestID); changed == exact {
		t.Fatalf("changed lane intent reused operation %s", exact)
	}
	legacySum := sha256.Sum256([]byte(ref.String() + "\x00" + "1.0.0" + "\x00" + namedLane + "\x00" + manifestID))
	if legacy := "manifest-" + hex.EncodeToString(legacySum[:]); legacy == exact {
		t.Fatalf("named-lane protocol reused legacy operation %s", legacy)
	}
}

func TestModelPublicationBatchesFinalizesCutsAndReplays(t *testing.T) {
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
	transfers = append(transfers, hub.Transfer{ObjectID: manifestID,
		Length: int64(len(manifest)), State: "claimed"})

	toolDir := t.TempDir()
	releasesPath := filepath.Join(toolDir, "releases.jsonl")
	refsPath := filepath.Join(toolDir, "refs.jsonl")
	manifestPath := filepath.Join(toolDir, "manifest.json")
	if err := os.WriteFile(releasesPath, []byte(releaseRow(string(evidence), namedLane)+"\n"), 0o600); err != nil {
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

	var openCalls, grantCalls, uploadCalls, finalizeCalls, cutCalls atomic.Int32
	publicationPath := "/v1/models/acme/model/publications/batch-proof"
	var server *httptest.Server
	server = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch {
		case r.Method == http.MethodPut && r.URL.Path == publicationPath:
			call := openCalls.Add(1)
			state, lane, objects := "open", "", transfers
			if call > 1 {
				state, lane, objects = "prepared", namedLane, append([]hub.Transfer(nil), transfers...)
				if cutCalls.Load() > 0 {
					state = "committed"
				}
				for i := range objects {
					objects[i].State = "accepted"
				}
			}
			_ = json.NewEncoder(w).Encode(hub.OpenPublicationResponse{Created: call == 1,
				Publication: hub.Session{Operation: "batch-proof", Release: "1.0.0", Lane: lane,
					LaneKey: namedLane, State: state, Objects: objects}})
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
				length := int64(1)
				if objectID == manifestID {
					length = int64(len(manifest))
				}
				answer.Grants = append(answer.Grants, hub.Grant{ObjectID: objectID, Length: length,
					URL: server.URL + "/objects/" + objectID})
			}
			_ = json.NewEncoder(w).Encode(answer)
		case r.Method == http.MethodPut && strings.HasPrefix(r.URL.Path, "/objects/"):
			uploadCalls.Add(1)
			raw, err := io.ReadAll(r.Body)
			if err != nil || string(raw) != "x" && !bytes.Equal(raw, manifest) {
				t.Errorf("object upload = %q, %v", raw, err)
			}
			w.WriteHeader(http.StatusOK)
		case r.Method == http.MethodPost && r.URL.Path == publicationPath+"/finalize":
			call := finalizeCalls.Add(1)
			_ = json.NewEncoder(w).Encode(hub.PreparedPublication{PublishID: "batch-proof",
				Release: "1.0.0", Lane: namedLane, Manifest: hub.ManifestRef{SHA256: manifestA, Length: int64(len(manifest))},
				Contract: hub.Contract{Stamps: map[string][]string{}, Structure: "sha256:" + topologyC,
					Encoding: hub.Encoding{Set: []string{"bf16"}}},
				TopologyDigest: "sha256:" + topologyC, Objects: objectCount, Bytes: objectCount,
				ReleaseEvidenceBase64: evidenceBase64, State: "verified/prepared", Duplicate: call > 1})
		case r.Method == http.MethodPost && r.URL.Path == "/v1/models/acme/model/releases/1.0.0":
			call := cutCalls.Add(1)
			_ = json.NewEncoder(w).Encode(hub.CutReleaseResponse{Operation: "cut-batch-proof", Release: "1.0.0",
				RepositorySHA256: manifestA, Lanes: []hub.CutLane{{Lane: namedLane,
					Manifest: hub.ManifestRef{SHA256: manifestA, Length: int64(len(manifest))},
					Contract: hub.Contract{Stamps: map[string][]string{}, Structure: "sha256:" + topologyC,
						Encoding: hub.Encoding{Set: []string{"bf16"}}},
					Objects: objectCount, Bytes: objectCount, Publication: "batch-proof"}}, Duplicate: call > 1})
		default:
			t.Errorf("unexpected request %s %s", r.Method, r.URL.Path)
			http.Error(w, "unexpected", http.StatusNotFound)
		}
	}))
	defer server.Close()

	publish := transfer.Publish{Tool: tool,
		Hub: hub.New(config.Config{HubURL: server.URL, HubToken: secret.New("token")}, "test"),
		Ref: hub.Ref{Org: "acme", Name: "model"}, ManifestID: manifestID,
		Release: "1.0.0", Lane: namedLane, Session: "batch-proof", Reason: "proof", Scratch: t.TempDir()}
	result, problem := publish.Run(context.Background())
	if problem != nil {
		t.Fatal(problem)
	}
	if result.Uploaded != objectCount+1 || result.Verified != objectCount ||
		result.Moved != int64(objectCount+len(manifest)) {
		t.Fatalf("publication result = %#v", result)
	}
	replayed, problem := publish.Run(context.Background())
	if problem != nil {
		t.Fatal(problem)
	}
	if !replayed.Dup || replayed.Uploaded != 0 || replayed.Grants != 0 || replayed.Moved != 0 ||
		replayed.Deduped != int64(objectCount+len(manifest)) || replayed.Verified != objectCount {
		t.Fatalf("replayed publication result = %#v", replayed)
	}
	publish.DryRun = true
	planned, problem := publish.Run(context.Background())
	if problem != nil || planned.Uploaded != 0 ||
		planned.Deduped != int64(objectCount+len(manifest)) || planned.Verified != 0 {
		t.Fatalf("committed dry-run result = %#v, %v", planned, problem)
	}
	if openCalls.Load() != 3 || grantCalls.Load() != 3 || uploadCalls.Load() != objectCount+1 ||
		finalizeCalls.Load() != 2 || cutCalls.Load() != 2 {
		t.Fatalf("publication calls: open=%d grants=%d uploads=%d finalize=%d cut=%d",
			openCalls.Load(), grantCalls.Load(), uploadCalls.Load(), finalizeCalls.Load(), cutCalls.Load())
	}
}
