package producttest

// th-122 — the volume verbs' hub contract, against a stub hub. The client
// keeps no local volume state: everything asserted here rides the wire.

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/cozy-creator/cozy/internal/config"
	"github.com/cozy-creator/cozy/internal/hub"
)

func volumeClient(url string) *hub.Client {
	return hub.New(config.Config{HubURL: url}, "producttest").WithTokenSource(staleTokenSource{})
}

func TestVolumeUXDescribesAnOptionalDisposableCache(t *testing.T) {
	server, _ := stubVolumeHub(t)
	defer server.Close()
	root := t.TempDir()
	if err := os.WriteFile(filepath.Join(root, config.FileName), []byte(
		"tensorhub_url: "+server.URL+"\n"+
			"tensorhub_token: volume-ux-test\n"), 0o600); err != nil {
		t.Fatal(err)
	}

	code, help := runCozy(t, root, "help", "volume")
	if code != 0 {
		t.Fatalf("volume help [exit %d]:\n%s", code, help)
	}
	for _, want := range []string{
		"optional repo-object cache",
		"optional cache volume",
		"disposable cache volume",
	} {
		if !strings.Contains(help, want) {
			t.Fatalf("volume help omitted %q:\n%s", want, help)
		}
	}

	code, listed := runCozy(t, root, "volume")
	if code != 0 {
		t.Fatalf("volume list [exit %d]:\n%s", code, listed)
	}
	for _, want := range []string{
		"Optional cache storage per hour:",
		"Cache usage is not measured.",
		"model and dataset snapshot objects",
		"disposable copies",
		"never authoritative",
	} {
		if !strings.Contains(listed, want) {
			t.Fatalf("volume list omitted %q:\n%s", want, listed)
		}
	}

	code, machine := runCozy(t, root, "--json", "volume")
	if code != 0 || strings.Contains(machine, `"cached":`) || strings.Contains(machine, `"cached_objects":`) ||
		strings.Contains(machine, `"warm":`) || strings.Contains(machine, `"warm_objects":`) {
		t.Fatalf("machine volume list did not hardcut cached fields [exit %d]:\n%s", code, machine)
	}

	code, warmed := runCozy(t, root, "volume", "warm", "EU-RO-1")
	if code != 0 || !strings.Contains(warmed, "immutable model and dataset snapshot objects") ||
		!strings.Contains(warmed, "optional and disposable") {
		t.Fatalf("volume warm lost cache semantics [exit %d]:\n%s", code, warmed)
	}

	code, dropped := runCozy(t, root, "volume", "drop", "pvl-a")
	if code != 0 || !strings.Contains(dropped, "cached copies are gone") ||
		!strings.Contains(dropped, "fetches snapshots from another source") {
		t.Fatalf("volume drop lost cache semantics [exit %d]:\n%s", code, dropped)
	}

	all := strings.ToLower(help + listed + machine + warmed + dropped)
	for _, legacy := range []string{"persistent model store", "standing model store", "models warm onto"} {
		if strings.Contains(all, legacy) {
			t.Fatalf("volume UX retained legacy claim %q:\n%s", legacy, all)
		}
	}
}

type staleTokenSource = staleSource

func stubVolumeHub(t *testing.T) (*httptest.Server, *[]string) {
	t.Helper()
	var seen []string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		seen = append(seen, r.Method+" "+r.URL.Path)
		if r.Header.Get("Authorization") == "" {
			t.Error("every volume route is authenticated")
		}
		w.Header().Set("Content-Type", "application/json")
		switch r.Method + " " + r.URL.Path {
		case "GET /v1/volumes":
			_, _ = w.Write([]byte(`{"volumes":[{"id":"pvl-a","provider":"runpod",
				"datacenter":"EU-RO-1","state":"live","size_gb":500,
				"usd_micros_per_hour":48000,"created_at":"2026-09-02T00:00:00Z",
				"warm_bytes":5368709120,"warm_objects":12}]}`))
		case "POST /v1/volumes":
			var body struct{ Provider, Datacenter string }
			if err := json.NewDecoder(r.Body).Decode(&body); err != nil || body.Datacenter != "EU-RO-1" {
				t.Errorf("warm body: %+v %v", body, err)
			}
			if r.Header.Get("X-Tensorhub-Reason") == "" {
				t.Error("a warm is a mutation and must carry its reason")
			}
			_, _ = w.Write([]byte(`{"id":"pvl-a","provider":"runpod","datacenter":"EU-RO-1",
				"state":"live","size_gb":500,"usd_micros_per_hour":48000,
				"created_at":"2026-09-02T00:00:00Z","warm_bytes":0,"warm_objects":0}`))
		case "DELETE /v1/volumes/pvl-a":
			w.WriteHeader(http.StatusNoContent)
		default:
			t.Errorf("unexpected hub call %s %s", r.Method, r.URL.Path)
			w.WriteHeader(http.StatusNotFound)
		}
	}))
	return server, &seen
}

func TestVolumeVerbsSpeakTheHubContract(t *testing.T) {
	server, seen := stubVolumeHub(t)
	defer server.Close()
	c := volumeClient(server.URL)
	ctx := context.Background()

	volumes, e := c.Volumes(ctx)
	if e != nil || len(volumes) != 1 {
		t.Fatalf("list: %v %+v", e, volumes)
	}
	if v := volumes[0]; v.ID != "pvl-a" || v.Datacenter != "EU-RO-1" || v.SizeGB != 500 ||
		v.USDMicrosPerHour != 48000 {
		t.Fatalf("volume row: %+v", v)
	}

	warmed, e := c.WarmVolume(ctx, "", "EU-RO-1", "cozy volume warm EU-RO-1")
	if e != nil || warmed.ID != "pvl-a" || warmed.State != "live" {
		t.Fatalf("warm: %v %+v", e, warmed)
	}
	if _, e := c.WarmVolume(ctx, "", " ", "r"); e == nil {
		t.Fatal("an empty datacenter must refuse before the wire")
	}

	if e := c.DropVolume(ctx, "pvl-a", "cozy volume drop pvl-a"); e != nil {
		t.Fatalf("drop: %v", e)
	}
	if e := c.DropVolume(ctx, "EU-RO-1", "r"); e == nil {
		t.Fatal("drop takes a volume id; a bare datacenter resolves client-side first")
	}

	want := []string{"GET /v1/volumes", "POST /v1/volumes", "DELETE /v1/volumes/pvl-a"}
	if len(*seen) != len(want) {
		t.Fatalf("hub calls: %v", *seen)
	}
	for i, call := range want {
		if (*seen)[i] != call {
			t.Fatalf("hub call %d = %q, want %q", i, (*seen)[i], call)
		}
	}
}
