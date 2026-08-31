package producttest

import (
	"encoding/json"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/cozy-creator/cozy/internal/records"
)

// TestLocalAPIDoor is the only thing standing between a loopback bind and every page the
// user's browser happens to load. A loopback bind is not a boundary: the Ollama CVE class
// is a DNS answer of 127.0.0.1 plus an attacker's Host header, and ambient cookie
// authority plus a CORS header would hand the whole surface over. Every arm is about the
// door, not the work, so it needs no model, no card, and no runtime peer.
func TestLocalAPIDoor(t *testing.T) {
	root := filepath.Join(os.TempDir(), "cozy-product-test", "api-door")
	must(t, os.RemoveAll(root))
	svc := startDaemonProcess(t, root)

	// A credential is required, and the refusal is the TYPED envelope naming its scheme.
	no := svc.call(t, "GET", "/v1/requests?limit=1", nil, "Authorization", "")
	if no.Status != http.StatusUnauthorized || no.code() != "unauthenticated" {
		t.Errorf("a request with NO credential: %s", no.brief())
	}
	if no.Header.Get("WWW-Authenticate") == "" {
		t.Error("the refusal does not name the scheme it wants")
	}
	wrong := svc.call(t, "GET", "/v1/requests?limit=1", nil, "Authorization", "Bearer "+strings.Repeat("0", 64))
	if wrong.Status != http.StatusUnauthorized || wrong.code() != "unauthenticated" {
		t.Errorf("a WRONG credential: %s", wrong.brief())
	}
	near := "Bearer " + svc.token[:len(svc.token)-1] + "0"
	if strings.HasSuffix(svc.token, "0") {
		near = "Bearer " + svc.token[:len(svc.token)-1] + "1"
	}
	if r := svc.call(t, "GET", "/v1/requests?limit=1", nil, "Authorization", near); r.Status != http.StatusUnauthorized {
		t.Errorf("a credential differing in ONE character: %s", r.brief())
	}

	// THE DNS-REBINDING KILL SWITCH. The request reaches 127.0.0.1 — because that is what
	// the attacker's DNS answered — and carries the attacker's hostname in Host.
	rebind := svc.call(t, "GET", "/v1/requests?limit=1", nil, "Host", "cozy.attacker.example")
	if rebind.Status != http.StatusForbidden || rebind.code() != "host_not_allowed" {
		t.Errorf("a rebinding-style foreign Host: %s", rebind.brief())
	}
	mutate := svc.call(t, "POST", "/v1/requests", map[string]any{"package": "x/y", "function": "f"},
		"Host", "cozy.attacker.example", "Idempotency-Key", "arm-rebind")
	if mutate.Status != http.StatusForbidden || mutate.code() != "host_not_allowed" {
		t.Errorf("a rebinding MUTATION: %s", mutate.brief())
	}

	// Origin, on a mutation. A cross-site fetch, form POST and EventSource all send one;
	// a same-origin GET does not, and a CLI never sends one at all.
	xs := svc.call(t, "POST", "/v1/requests", map[string]any{"package": "x/y", "function": "f"},
		"Origin", "https://evil.example", "Idempotency-Key", "arm-origin")
	if xs.Status != http.StatusForbidden || xs.code() != "origin_not_allowed" {
		t.Errorf("a CROSS-ORIGIN mutation: %s", xs.brief())
	}
	null := svc.call(t, "POST", "/v1/requests", map[string]any{"package": "x/y", "function": "f"},
		"Origin", "null", "Idempotency-Key", "arm-null")
	if null.Status != http.StatusForbidden {
		t.Errorf("a sandboxed-iframe `Origin: null` mutation: %s", null.brief())
	}
	same := svc.call(t, "GET", "/v1/requests?limit=1", nil, "Origin", "http://"+svc.addr)
	if same.Status != http.StatusOK {
		t.Errorf("a SAME-ORIGIN request: %s", same.brief())
	}

	// No CORS, no cookies, no sniffing.
	caps := svc.call(t, "GET", "/v1/requests?limit=1", nil)
	if got := caps.Header.Get("Access-Control-Allow-Origin"); got != "" { //cozy:allow the SUITE reads this header to prove its ABSENCE; the product sets none
		t.Errorf("a CORS header is served: %q", got)
	}
	if got := caps.Header.Get("Set-Cookie"); got != "" {
		t.Errorf("a Set-Cookie is served: %q", got)
	}
	if caps.Header.Get("X-Content-Type-Options") != "nosniff" {
		t.Error("no nosniff on the response")
	}
	if !strings.Contains(caps.Header.Get("Content-Security-Policy"), "default-src 'none'") {
		t.Errorf("no strict CSP: %q", caps.Header.Get("Content-Security-Policy"))
	}
	if caps.Header.Get("Referrer-Policy") != "no-referrer" {
		t.Errorf("a referrer leaves this origin: %q", caps.Header.Get("Referrer-Policy"))
	}
	// A cookie the server never reads: presenting one instead of a bearer authenticates
	// nothing, which is the property that makes cross-site ambient authority impossible.
	cookied := svc.call(t, "GET", "/v1/requests?limit=1", nil,
		"Authorization", "", "Cookie", "cozy="+svc.token)
	if cookied.Status != http.StatusUnauthorized {
		t.Errorf("a credential presented as a COOKIE authenticated something: %s", cookied.brief())
	}

	// The embedded launch page is intentionally public on the guarded localhost origin.
	// It contains no user state and cannot turn a cookie into ambient API authority.
	page := svc.call(t, "GET", "/", nil, "Authorization", "")
	if page.Status != http.StatusOK ||
		!strings.Contains(string(page.Body), "local generative workspace is running") {
		t.Errorf("the unauthenticated localhost web stub is unavailable: %s", page.brief())
	}
	if hostile := svc.call(t, "GET", "/", nil, "Authorization", "", "Host", "cozy.attacker.example"); hostile.Status != http.StatusForbidden {
		t.Errorf("the public web stub bypassed the Host guard: %s", hostile.brief())
	}

	// Browser-selected bytes cross one authenticated, bounded, content-addressed door.
	// The response exposes an opaque id, never a caller filesystem path.
	uploadBody := []byte("\x89PNG\r\n\x1a\ncozy-upload-arm")
	unauthenticatedUpload := svc.callBytes(t, "POST", "/v1/uploads", uploadBody, "image/png",
		"Authorization", "")
	if unauthenticatedUpload.Status != http.StatusUnauthorized {
		t.Errorf("an unauthenticated upload was admitted: %s", unauthenticatedUpload.brief())
	}
	first := svc.callBytes(t, "POST", "/v1/uploads", uploadBody, "image/png")
	second := svc.callBytes(t, "POST", "/v1/uploads", uploadBody, "image/png")
	if first.Status != http.StatusCreated || second.Status != http.StatusCreated {
		t.Fatalf("content-addressed upload failed: first=%s second=%s", first.brief(), second.brief())
	}
	var stored, replayed struct {
		ID     string `json:"upload_id"`
		Digest string `json:"digest"`
		Length int64  `json:"length"`
		URL    string `json:"url"`
	}
	must(t, json.Unmarshal(first.Body, &stored))
	must(t, json.Unmarshal(second.Body, &replayed))
	if stored.ID == "" || stored.ID != replayed.ID || stored.Digest != replayed.Digest ||
		stored.Length != int64(len(uploadBody)) || strings.Contains(string(first.Body), root) {
		t.Errorf("upload identity is not pathless and idempotent: %s / %s", first.brief(), second.brief())
	}
	fetched := svc.call(t, "GET", stored.URL, nil)
	if fetched.Status != http.StatusOK || string(fetched.Body) != string(uploadBody) ||
		fetched.Header.Get("X-Cozy-Digest") != stored.Digest {
		t.Errorf("opaque upload readback changed bytes or identity: %s", fetched.brief())
	}
	malformed := svc.call(t, "GET", "/v1/uploads/upl-../../../../etc/passwd", nil)
	if malformed.Status != http.StatusNotFound {
		t.Errorf("a path-shaped upload id reached the filesystem: %s", malformed.brief())
	}

	// The media plane takes opaque ids and nothing that could be a path.
	for _, attempt := range []string{
		"/v1/media/../../../../etc/passwd",
		"/v1/media/..%2f..%2f..%2fetc%2fpasswd",
		"/v1/media/%2e%2e%2f%2e%2e%2fetc%2fpasswd",
		"/v1/media/" + strings.ReplaceAll(root, "/", "%2F") + "%2Frecords.db",
	} {
		res := svc.call(t, "GET", attempt, nil)
		if res.Status != http.StatusNotFound && res.Status != http.StatusMovedPermanently {
			t.Errorf("a client-supplied PATH was served as media: %s -> %s", attempt, res.brief())
		}
	}

	// The typed envelope, everywhere — including before anything is recorded.
	if r := svc.call(t, "GET", "/v1/does-not-exist", nil); r.Status != http.StatusNotFound ||
		r.code() != "unknown_route" {
		t.Errorf("an unknown route answered without the typed envelope: %s", r.brief())
	}
	noKey := svc.call(t, "POST", "/v1/requests", map[string]any{"package": "x/y", "function": "f"},
		"Idempotency-Key", "")
	if noKey.Status != http.StatusBadRequest || noKey.code() != "idempotency_key_required" {
		t.Errorf("a submission with no Idempotency-Key: %s", noKey.brief())
	}
	negativeBudget := svc.call(t, "POST", "/v1/requests", map[string]any{
		"package": "x/y", "function": "f", "max_cost_usd_micros": -1,
	}, "Idempotency-Key", "negative-budget")
	if negativeBudget.Status != http.StatusBadRequest || negativeBudget.code() != "invalid_request" {
		t.Errorf("a negative rental budget reached the queue: %s", negativeBudget.brief())
	}
	mixedPlacement := svc.call(t, "POST", "/v1/requests", map[string]any{
		"package": "x/y", "function": "f", "worker": "rental-1", "max_cost_usd_micros": 1,
	}, "Idempotency-Key", "mixed-placement")
	if mixedPlacement.Status != http.StatusBadRequest || mixedPlacement.code() != "invalid_request" {
		t.Errorf("an explicit worker plus automatic budget reached the queue: %s", mixedPlacement.brief())
	}

	// The credential is handed over through an OS-protected file, never argv, and never
	// appears in anything the daemon writes.
	info, err := os.Stat(filepath.Join(root, "client.cred"))
	must(t, err)
	if info.Mode().Perm() != 0o600 {
		t.Errorf("the CLI credential file is %v, wanted 0600", info.Mode().Perm())
	}
	log, _ := os.ReadFile(filepath.Join(root, "daemon-test.log"))
	if strings.Contains(string(log), svc.token) {
		t.Error("the daemon's own log contains the credential")
	}
	if r := svc.call(t, "GET", "/v1/requests/req-nope", nil); strings.Contains(string(r.Body), svc.token) {
		t.Error("a rendered error contains the credential")
	}

	// Plain down is safe by default: one paid obligation refuses shutdown by exact id.
	store, problem := records.Open(filepath.Join(root, "records.db"))
	fatal(t, problem)
	fatal(t, store.RecordRental(records.Rental{
		ID:               "rental-down-arm",
		AcceleratorModel: "CPU", State: "ready", Hub: "https://hub.invalid",
	}))
	blocked := svc.call(t, "POST", "/v1/local/daemon/down", map[string]bool{"all": false})
	if blocked.Status != http.StatusConflict || blocked.code() != "active_work" ||
		!strings.Contains(string(blocked.Body), "rental-down-arm") {
		t.Errorf("plain down did not name and preserve the rental: %s", blocked.brief())
	}
	_, problem = store.ForgetRental("rental-down-arm")
	fatal(t, problem)
	store.Close()
}
