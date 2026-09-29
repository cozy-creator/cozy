package producttest

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"runtime"
	"strconv"
	"sync/atomic"
	"testing"

	"github.com/cozy-creator/cozy/internal/machines"
)

// releaseTransport routes the installer's public release requests to a real HTTPS
// fixture. URL selection, pagination, JSON decoding, download and digest checks remain
// the shipping path; no external GitHub release is read or changed by this test.
type releaseTransport struct {
	base      *url.URL
	transport http.RoundTripper
}

func (t releaseTransport) RoundTrip(request *http.Request) (*http.Response, error) {
	clone := request.Clone(request.Context())
	target := *request.URL
	target.Scheme, target.Host = t.base.Scheme, t.base.Host
	clone.URL = &target
	return t.transport.RoundTrip(clone)
}

func TestMachineAgentReleaseDiscoveryAndDigest(t *testing.T) {
	if *machineHostBinary == "" {
		t.Skip("requires -machine-host=<standalone cozy-machine>")
	}
	binary, err := os.ReadFile(*machineHostBinary)
	must(t, err)
	digest := sha256.Sum256(binary)
	var downloads atomic.Int32
	var pageTwo atomic.Int32
	var oldOnly atomic.Bool
	badDigest := false
	server := httptest.NewTLSServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		switch r.URL.Path {
		case "/repos/cozy-creator/cozy/releases":
			page, _ := strconv.Atoi(r.URL.Query().Get("page"))
			if page == 1 {
				releases := make([]map[string]any, 100)
				for i := range releases {
					releases[i] = map[string]any{"tag_name": fmt.Sprintf("v%d.0.0", i), "assets": []any{}}
				}
				_ = json.NewEncoder(w).Encode(releases)
			} else if page == 2 {
				pageTwo.Add(1)
				release := func(version string, extra map[string]any) map[string]any {
					value := map[string]any{"tag_name": "machine-v" + version, "assets": []map[string]string{{"name": "machine-agent.json", "browser_download_url": "https://github.com/manifest/" + version}}}
					for key, v := range extra {
						value[key] = v
					}
					return value
				}
				if oldOnly.Load() {
					_ = json.NewEncoder(w).Encode([]any{release("0.1.0", nil)})
					return
				}
				_ = json.NewEncoder(w).Encode([]any{release("1.9.0", nil), release("9.0.0", map[string]any{"draft": true}), release("8.0.0", map[string]any{"prerelease": true}), release("2.0.0", nil)})
			} else {
				t.Errorf("unexpected page %d", page)
				http.NotFound(w, r)
			}
		case "/manifest/2.0.0":
			hash := hex.EncodeToString(digest[:])
			if badDigest {
				hash = hex.EncodeToString(make([]byte, 32))
			}
			_ = json.NewEncoder(w).Encode(map[string]any{"version": "2.0.0", "future": true, "platforms": map[string]any{runtime.GOOS + "-" + runtime.GOARCH: map[string]string{"url": "https://github.com/artifact", "sha256": hash}}})
		case "/artifact":
			downloads.Add(1)
			_, _ = w.Write(binary)
		default:
			t.Errorf("installer read an unexpected release URL %s", r.URL.Path)
			http.NotFound(w, r)
		}
	}))
	defer server.Close()
	base, err := url.Parse(server.URL)
	must(t, err)
	previous := http.DefaultClient
	http.DefaultClient = &http.Client{Transport: releaseTransport{base, server.Client().Transport}}
	defer func() { http.DefaultClient = previous }()
	host := machines.NewHost(t.TempDir(), "", nil)
	path, problem := host.PublishedAgent(context.Background())
	fatal(t, problem)
	if machines.HostModule(path) != machines.AgentModule {
		t.Fatal("download did not select the independent machine agent")
	}
	_, problem = host.PublishedAgent(context.Background())
	fatal(t, problem)
	if pageTwo.Load() != 2 || downloads.Load() != 1 {
		t.Fatalf("pagination/cache not honored: page2=%d downloads=%d", pageTwo.Load(), downloads.Load())
	}
	badDigest = true
	if _, problem := host.PublishedAgent(context.Background()); problem == nil {
		t.Fatal("changed digest accepted incompatible artifact bytes")
	}
	oldOnly.Store(true)
	before := downloads.Load()
	if _, problem := host.PublishedAgent(context.Background()); problem == nil || downloads.Load() != before {
		t.Fatal("discovery downloaded an agent without delegated-principal protection")
	}
}
