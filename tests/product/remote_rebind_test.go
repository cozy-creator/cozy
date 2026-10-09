package producttest

import (
	"net/http"
	"slices"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/cozy-creator/cozy/internal/machines"
)

// A rebind made from another computer or the Hub reaches every machine without a request of
// its own: the account's bindings revision rides the rental listing this computer polls. The
// CLI reads the package's binding and its Model once more and hands every machine the named
// choice under the new catalog revision (th-245); each machine resolves it once more with the
// Hub's closure, then nothing is read again. No machine reads a binding or Model card.
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
	h.worker.Config.Handler = count("machine", h.worker.Config.Handler)
	account, probe := h.server.Config.Handler, probeModel(t, resolved)
	h.server.Config.Handler = count("account", http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch {
		case r.URL.Path == "/v1/packages/"+parityPublished+"/bindings":
			_, _ = w.Write([]byte(binding))
		case probe(w, r):
		default:
			account.ServeHTTP(w, r)
		}
	}))
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
		calls := strings.Join(seen, "\n")
		if strings.Contains(calls, "machine GET /v1/packages/"+parityPublished+"/bindings") || strings.Contains(calls, "machine GET /v1/models/") {
			t.Fatalf("a machine read the catalog at its Hub on %s: %q", venue, calls)
		}
		return calls
	}
	// resolves drops a machine's resolution of its model by name, which a new catalog revision asks for.
	resolves := func(calls string) string {
		return strings.Join(slices.DeleteFunc(strings.Split(calls, "\n"), func(call string) bool {
			return call == "machine POST /v1/tensorfs/closure"
		}), "\n")
	}
	read := func(calls string) bool {
		return strings.Contains(calls, "account GET /v1/packages/"+parityPublished+"/bindings") && strings.Contains(calls, "account GET /v1/models/proof/probe")
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
		if listed == 7 {
			break
		}
		if time.Now().After(deadline) {
			t.Fatalf("the new revision was not carried: listing %d", listed)
		}
	}
	time.Sleep(time.Second) // the listing's answer is applied after it is served
	first := true
	for venue, args := range venues {
		if calls := run(venue, args...); first && !read(calls) {
			t.Fatalf("the run on %s after a rebind elsewhere did not read the binding again: %q", venue, calls)
		} else if !strings.Contains(calls, "machine POST /v1/tensorfs/closure") {
			t.Fatalf("the machine %s did not resolve its model again under the new revision: %q", venue, calls)
		} else if !first && resolves(calls) != "" {
			t.Fatalf("the run on %s read again what the CLI resolved for another machine: %q", venue, calls)
		}
		first = false
		if calls := run(venue, args...); calls != "" {
			t.Fatalf("the warm run on %s after the rebind read: %q", venue, calls)
		}
	}
}
