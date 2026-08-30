package producttest

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"reflect"
	"testing"

	"github.com/cozy-creator/cozy/internal/config"
	"github.com/cozy-creator/cozy/internal/hub"
	"github.com/cozy-creator/cozy/internal/secret"
	"github.com/cozy-creator/cozy/internal/transfer"
)

func TestLocalDownloadRequiresRealReleaseCoordinates(t *testing.T) {
	manifestID := "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
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

func TestManifestReadRoutes(t *testing.T) {
	manifestID := "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
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
			if err := json.NewDecoder(r.Body).Decode(&body); err != nil || len(body) != 1 || body["manifest"] != "e30=" {
				t.Fatalf("seal body = %#v, %v", body, err)
			}
			_, _ = io.WriteString(w, `{"publish_id":"manifest-proof","release":"1.0.0","lane":"bf16","manifest":{"sha256":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","length":2},"topology_digest":"sha256:cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc","objects":1,"bytes":7,"duplicate":false}`)
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
		hub.SealPublicationRequest{Manifest: "e30="}, "proof")
	if problem != nil || done.Manifest.Length != 2 || done.Release != "1.0.0" || done.Lane != "bf16" {
		t.Fatalf("SealPublication = %#v, %v", done, problem)
	}
}
