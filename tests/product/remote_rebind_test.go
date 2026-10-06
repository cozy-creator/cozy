package producttest

import (
	"encoding/json"
	"net/http"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/cozy-creator/cozy/internal/machines"
)

// A rebind made from another computer or the Hub reaches every machine without a request of
// its own: the account's bindings revision rides the rental listing this computer polls (its
// own machine) and each rental's authority poll (that rental's machine). Each machine reads
// the package's binding and its Model once more, then nothing again.
func TestARebindMadeElsewhereReachesEveryMachine(t *testing.T) {
	h, root, _, _ := parityMachines(t)
	resolved := seedProbe(t, h, root, machines.Local, "tessa")
	var mu sync.Mutex
	var seen []string
	count := func(hub string, handler http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			authority := r.Method == http.MethodGet && r.URL.Path == "/v1/worker/rental/authorized-keys"
			if !strings.HasPrefix(r.URL.Path, "/v1/rentals") && !authority {
				mu.Lock()
				seen = append(seen, hub+" "+r.Method+" "+r.URL.Path)
				mu.Unlock()
			}
			handler.ServeHTTP(w, r)
		})
	}
	publishParityRelease(t, h, root, probeProject(t, "proof/probe@1.0.0/bf16"))
	binding := `{"bindings":[{"slot":"touch.models.source","model":"proof/probe","release":"1.0.0","revision":1,"ladder":[{"gpu":"*","lane":"bf16"}]}]}`
	doors := h.worker.Config.Handler
	h.worker.Config.Handler = count("machine", http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch {
		case r.URL.Path == "/v1/packages/"+parityPublished+"/bindings":
			_, _ = w.Write([]byte(binding))
		case r.URL.Path == "/v1/models/resolve" && r.URL.Query().Get("ref") == "proof/probe@1.0.0" && r.URL.Query().Get("lane") == "bf16":
			_ = json.NewEncoder(w).Encode(resolved)
		default:
			doors.ServeHTTP(w, r)
		}
	}))
	h.server.Config.Handler = count("account", h.server.Config.Handler)
	run := func(venue string, args ...string) string {
		t.Helper()
		mu.Lock()
		seen = nil
		mu.Unlock()
		code, out := runCozy(t, root, append([]string{"run", parityPublished + "/touch", "value=1", "--await", "--json"}, args...)...)
		if code != 0 || !strings.Contains(out, `"value":2`) {
			t.Fatalf("touch on %s [exit %d]\n%s", venue, code, out)
		}
		mu.Lock()
		defer mu.Unlock()
		return strings.Join(seen, "\n")
	}
	read := func(calls string) bool {
		return strings.Contains(calls, "machine GET /v1/packages/"+parityPublished+"/bindings") && strings.Contains(calls, "machine GET /v1/models/resolve")
	}
	venues := map[string][]string{"local": nil, "tessa": {"--rental=tessa"}}
	for venue, args := range venues {
		run(venue, args...)
		if calls := run(venue, args...); calls != "" {
			t.Fatalf("the warm run on %s read: %q", venue, calls)
		}
	}

	// The owner rebinds elsewhere: only the Hub's revision moves.
	h.fakeRentalHub.mu.Lock()
	h.bindings = 7
	h.fakeRentalHub.mu.Unlock()
	for deadline := time.Now().Add(45 * time.Second); ; time.Sleep(200 * time.Millisecond) {
		h.fakeRentalHub.mu.Lock()
		listed := h.listedBindings
		h.fakeRentalHub.mu.Unlock()
		h.mu.Lock()
		polled := h.polledBindings
		h.mu.Unlock()
		if listed == 7 && polled == 7 {
			break
		}
		if time.Now().After(deadline) {
			t.Fatalf("the new revision was not carried: listing %d, authority poll %d", listed, polled)
		}
	}
	time.Sleep(time.Second) // the poll's answer is applied after it is served
	for venue, args := range venues {
		if calls := run(venue, args...); !read(calls) {
			t.Fatalf("the run on %s after a rebind elsewhere did not read the binding again: %q", venue, calls)
		}
		if calls := run(venue, args...); calls != "" {
			t.Fatalf("the warm run on %s after the rebind read: %q", venue, calls)
		}
	}
}
