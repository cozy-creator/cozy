package api

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"testing"
	"time"

	"github.com/cozy-creator/cozy-creator-v2/internal/exit"
	"github.com/cozy-creator/cozy-creator-v2/internal/home"
	"github.com/cozy-creator/cozy-creator-v2/internal/inputasset"
	"github.com/cozy-creator/cozy-creator-v2/internal/orchestrator"
	"github.com/cozy-creator/cozy-creator-v2/internal/records"
	"github.com/cozy-creator/cozy-creator-v2/internal/secret"
)

func TestSettledAssetReplayUsesDurableIdentityBeforeFiles(t *testing.T) {
	layout, e := home.Open(t.TempDir())
	if e != nil {
		t.Fatal(e)
	}
	store, e := records.Open(layout.DB)
	if e != nil {
		t.Fatal(e)
	}
	defer store.Close()
	owner, e := orchestrator.Open(orchestrator.Options{Layout: layout, Store: store})
	if e != nil {
		t.Fatal(e)
	}
	defer owner.Close(time.Second)
	cli := secret.New("cli-test-token")
	binding := &orchestrator.Binding{Entrypoint: "run", Outputs: []string{"video"},
		RuntimePlan: &orchestrator.BindingPlanSubject{SubjectID: workflowTestDigest,
			Digest: workflowTestDigest, Kind: "plan", Length: 1}}
	resolver := workflowResolver{placement: orchestrator.DesiredPlacement{
		Endpoint: "org/ep", ReleaseID: "org/ep@1.0.0", InstallID: "install-api",
		DescriptorDigest: workflowTestDigest, Bindings: []*orchestrator.Binding{binding}}}
	s := New(Options{Orchestrator: owner, Creds: Credentials{CLI: cli},
		Addr: "127.0.0.1:2699", Endpoints: resolver})
	handler, e := s.Handler()
	if e != nil {
		t.Fatal(e)
	}

	source := layout.Root + "/source.png"
	if err := os.WriteFile(source, []byte("asset bytes"), 0o600); err != nil {
		t.Fatal(err)
	}
	length, digest, mediaType, inspected := inputasset.Fingerprint(source, inputasset.MaxBytes)
	if inspected != nil {
		t.Fatal(inspected)
	}
	submission := Submission{
		Endpoint: "org/ep", Function: "run", PlanID: workflowTestDigest,
		Input:   json.RawMessage(`{"first_frame":"` + digest + `","prompt":"same"}`),
		Outputs: []string{"video"},
		LocalAssets: []records.AssetBinding{{
			FieldPath: "first_frame", LocalPath: source, Digest: digest,
			Length: length, MediaType: mediaType, Order: 0,
		}},
	}
	body, err := json.Marshal(submission)
	if err != nil {
		t.Fatal(err)
	}
	call := func(body []byte) *httptest.ResponseRecorder {
		t.Helper()
		req := httptest.NewRequest(http.MethodPost, "/v1/requests", bytes.NewReader(body))
		req.RemoteAddr = "127.0.0.1:12345"
		req.Host = "127.0.0.1:2699"
		req.Header.Set("Idempotency-Key", "asset-replay")
		Authorize(req, cli)
		response := httptest.NewRecorder()
		handler.ServeHTTP(response, req)
		return response
	}

	first := call(body)
	if first.Code != http.StatusAccepted {
		t.Fatalf("first submit = %d, body %s", first.Code, first.Body.String())
	}
	var original Handle
	if err := json.Unmarshal(first.Body.Bytes(), &original); err != nil {
		t.Fatal(err)
	}
	if original.RequestID == "" || original.Replay {
		t.Fatalf("first handle = %#v", original)
	}
	if e := store.SettleRequest(original.RequestID, "succeeded"); e != nil {
		t.Fatal(e)
	}
	if err := os.Remove(source); err != nil {
		t.Fatal(err)
	}
	if err := os.Remove(layout.InputAsset(digest)); err != nil {
		t.Fatal(err)
	}

	replayed := call(body)
	if replayed.Code != http.StatusOK {
		t.Fatalf("settled replay = %d, body %s", replayed.Code, replayed.Body.String())
	}
	var same Handle
	if err := json.Unmarshal(replayed.Body.Bytes(), &same); err != nil {
		t.Fatal(err)
	}
	if same.RequestID != original.RequestID || !same.Replay {
		t.Fatalf("replay handle = %#v, original = %#v", same, original)
	}

	changed := submission
	changed.Input = json.RawMessage(`{"first_frame":"` + digest + `","prompt":"changed"}`)
	changedBody, err := json.Marshal(changed)
	if err != nil {
		t.Fatal(err)
	}
	conflict := call(changedBody)
	if conflict.Code != http.StatusConflict {
		t.Fatalf("changed body after asset cleanup = %d, body %s", conflict.Code, conflict.Body.String())
	}
}

func TestRemoteSubmissionResolvesOnlyTheRentalsExactPlan(t *testing.T) {
	const planID = "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
	s := &Server{rentals: func(id string) (*orchestrator.DesiredPlacement, *exit.Error) {
		if id != "rnt-exact" {
			t.Fatalf("rental lookup = %q", id)
		}
		return &orchestrator.DesiredPlacement{Endpoint: "org/model", Bindings: []*orchestrator.Binding{{
			Entrypoint: "generate", Outputs: []string{"image"},
			RuntimePlan: &orchestrator.BindingPlanSubject{SubjectID: planID, Digest: planID,
				Kind: "plan", Length: 1, CanonicalBytes: []byte("x")},
		}}}, nil
	}}
	resolved, e := s.resolvePlan(Submission{
		Endpoint: "org/model", Function: "generate", Worker: "rnt-exact", Input: json.RawMessage(`{}`),
	})
	if e != nil {
		t.Fatal(e)
	}
	if resolved.PlanID != planID || len(resolved.Outputs) != 1 || resolved.Outputs[0] != "image" {
		t.Fatalf("remote resolution = %#v", resolved)
	}
	_, e = s.resolvePlan(Submission{Endpoint: "org/model", Function: "generate",
		Worker: "rnt-exact", PlanID: "sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
		Input: json.RawMessage(`{}`)})
	if e == nil || e.ErrName() != "rental.plan_mismatch" {
		t.Fatalf("caller-supplied remote plan = %v, want rental.plan_mismatch", e)
	}
}
