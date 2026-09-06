package producttest

import (
	"bytes"
	"crypto/sha256"
	"encoding/base64"
	"encoding/hex"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
)

// The native Runtime streams actual bodies into this gate, which forwards the exact
// signed request to R2. Four real HTTP arrivals release the gate; no goroutine counter
// or canned transfer result can make the owner advance custody.
type checkpointHTTPGate struct {
	server                         *httptest.Server
	mu                             sync.Mutex
	targets                        map[string]string
	active, peak, arrived          int
	moved                          int64
	open                           chan struct{}
	verify                         func()
	getActive, getPeak, getArrived int
	getOpen                        chan struct{}
	putRelease, getRelease         sync.Once
	refuse                         bool
	refusedID                      string
}

func newCheckpointHTTPGate(t *testing.T, verify func()) *checkpointHTTPGate {
	t.Helper()
	g := &checkpointHTTPGate{targets: map[string]string{}, open: make(chan struct{}), getOpen: make(chan struct{}), refuse: true, verify: verify}
	g.server = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, err := io.ReadAll(io.LimitReader(r.Body, 64<<20+1))
		if err != nil || len(body) > 64<<20 {
			http.Error(w, "body bound", 400)
			return
		}
		id := strings.TrimPrefix(r.URL.Path, "/")
		if r.Method == http.MethodPut {
			sum := sha256.Sum256(body)
			if id != hex.EncodeToString(sum[:]) || len(r.Header.Values("If-None-Match")) != 1 ||
				r.Header.Get("If-None-Match") != "*" || r.Header.Get("X-Amz-Checksum-Sha256") != base64.StdEncoding.EncodeToString(sum[:]) {
				http.Error(w, "signed content mismatch", 400)
				return
			}
		}
		g.mu.Lock()
		target := g.targets[r.Method+id]
		g.active++
		g.peak = max(g.peak, g.active)
		g.arrived++
		if r.Method == http.MethodGet {
			g.getActive++
			g.getArrived++
			g.getPeak = max(g.getPeak, g.getActive)
			if g.getArrived == 5 {
				g.getRelease.Do(func() { close(g.getOpen) })
			}
		} else if g.arrived == 4 {
			g.verify()
			g.putRelease.Do(func() { close(g.open) })
		}
		waitGet := r.Method == http.MethodGet && g.getArrived > 1
		if r.Method == http.MethodPut && g.refusedID == "" {
			g.refusedID = id
		}
		refuse := g.refuse && r.Method == http.MethodPut && id == g.refusedID
		g.mu.Unlock()
		defer func() {
			g.mu.Lock()
			g.active--
			if r.Method == http.MethodGet {
				g.getActive--
			}
			g.moved += int64(len(body))
			g.mu.Unlock()
		}()
		if r.Method == http.MethodPut {
			select {
			case <-g.open:
			case <-r.Context().Done():
				return
			}
		}
		if waitGet {
			select {
			case <-g.getOpen:
			case <-r.Context().Done():
				return
			}
		}
		if refuse {
			http.Error(w, "fixture denied this signed PUT", 403)
			return
		}
		if target == "" {
			http.Error(w, "no exact grant", 403)
			return
		}
		request, err := http.NewRequestWithContext(r.Context(), r.Method, target, bytes.NewReader(body))
		if err != nil {
			http.Error(w, "request", 400)
			return
		}
		request.Header = r.Header.Clone()
		response, err := http.DefaultClient.Do(request)
		if err != nil {
			http.Error(w, "upstream", 502)
			return
		}
		defer response.Body.Close()
		for name, values := range response.Header {
			for _, value := range values {
				w.Header().Add(name, value)
			}
		}
		w.WriteHeader(response.StatusCode)
		_, _ = io.Copy(w, response.Body)
	}))
	t.Cleanup(func() {
		g.putRelease.Do(func() { close(g.open) })
		g.getRelease.Do(func() { close(g.getOpen) })
		g.server.Close()
	})
	return g
}

func (g *checkpointHTTPGate) route(method, objectID, target string) string {
	id := strings.TrimPrefix(objectID, "sha256:")
	g.mu.Lock()
	g.targets[method+id] = target
	g.mu.Unlock()
	return g.server.URL + "/" + id
}
