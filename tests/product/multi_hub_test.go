package producttest

import (
	"bytes"
	"crypto/ed25519"
	"crypto/rand"
	"crypto/sha256"
	"database/sql"
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/cozy-creator/cozy/internal/config"
	"github.com/cozy-creator/cozy/internal/daemon"
	"github.com/cozy-creator/cozy/internal/records"
)

// hubWitness is everything one stand-in Tensorhub was sent: which bearers reached it
// and which paths were asked, so a test can prove where each credential went.
type hubWitness struct {
	mu      sync.Mutex
	bearers map[string]int
	logins  map[string]int
	paths   []string
}

func (w *hubWitness) wrap(next http.Handler) http.Handler {
	return http.HandlerFunc(func(rw http.ResponseWriter, r *http.Request) {
		w.mu.Lock()
		if w.bearers == nil {
			w.bearers, w.logins = map[string]int{}, map[string]int{}
		}
		w.bearers[r.Header.Get("Authorization")]++
		w.paths = append(w.paths, r.Method+" "+r.URL.Path)
		if strings.HasSuffix(r.URL.Path, "/login/begin") {
			body, _ := io.ReadAll(r.Body)
			var login map[string]string
			_ = json.Unmarshal(body, &login)
			w.logins[login["device_key_id"]]++
			r.Body = io.NopCloser(bytes.NewReader(body))
		}
		w.mu.Unlock()
		next.ServeHTTP(rw, r)
	})
}

func (w *hubWitness) loggedIn(deviceID string) int {
	w.mu.Lock()
	defer w.mu.Unlock()
	return w.logins[deviceID]
}

func (w *hubWitness) saw(authorization string) int {
	w.mu.Lock()
	defer w.mu.Unlock()
	return w.bearers[authorization]
}

// machineKeyHub is a stand-in Tensorhub that knows one machine key: it mints its own
// bearer for that key's login, and answers the account and rental reads a model upload
// and a rental listing make only for that bearer.
func machineKeyHub(t *testing.T, witness *hubWitness, deviceID string, public ed25519.PublicKey, bearer, account string) *httptest.Server {
	t.Helper()
	mux := http.NewServeMux()
	authorized := func(r *http.Request) bool { return r.Header.Get("Authorization") == "Bearer "+bearer }
	mux.HandleFunc("GET /v1/accounts/current", func(w http.ResponseWriter, r *http.Request) {
		if !authorized(r) {
			http.Error(w, `{"error":{"code":"invalid_credentials","message":"authenticate"}}`, http.StatusUnauthorized)
			return
		}
		_ = json.NewEncoder(w).Encode(map[string]string{"name": account})
	})
	mux.HandleFunc("GET /v1/rentals", func(w http.ResponseWriter, r *http.Request) {
		if !authorized(r) {
			http.Error(w, `{"error":{"code":"invalid_credentials","message":"authenticate"}}`, http.StatusUnauthorized)
			return
		}
		_ = json.NewEncoder(w).Encode(map[string]any{"rentals": []any{}})
	})
	server := httptest.NewServer(witness.wrap(machineKeyLogin(deviceID, public, bearer, mux)))
	t.Cleanup(server.Close)
	return server
}

// machineKeyLogin answers the AuthKit machine-key login for one device, minting bearer,
// and hands every other request to next.
func machineKeyLogin(deviceID string, public ed25519.PublicKey, bearer string, next http.Handler) http.Handler {
	challenge := bytes.Repeat([]byte{7}, 32)
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/v1/auth/device-keys/login/begin":
			var body map[string]string
			if json.NewDecoder(r.Body).Decode(&body) != nil || body["device_key_id"] != deviceID {
				http.Error(w, `{"error":{"code":"auth.unknown_machine","message":"unknown machine"}}`, http.StatusUnauthorized)
				return
			}
			_ = json.NewEncoder(w).Encode(map[string]any{"challenge_id": "login-" + deviceID,
				"challenge": base64.RawURLEncoding.EncodeToString(challenge), "expires_at": time.Now().Add(time.Minute).UTC().Format(time.RFC3339)})
		case "/v1/auth/device-keys/login/finish":
			var body map[string]string
			_ = json.NewDecoder(r.Body).Decode(&body)
			signature, err := base64.RawURLEncoding.DecodeString(body["signature"])
			signed := append([]byte("authkit.device-key-login/1\x00"), challenge...)
			if err != nil || body["challenge_id"] != "login-"+deviceID || !ed25519.Verify(public, signed, signature) {
				http.Error(w, `{"error":{"code":"auth.invalid_signature","message":"invalid signature"}}`, http.StatusUnauthorized)
				return
			}
			_ = json.NewEncoder(w).Encode(map[string]any{"token_set": map[string]any{"access_token": bearer,
				"token_type": "Bearer", "expires_in": 3600}, "device_key": map[string]string{"id": deviceID}})
		default:
			next.ServeHTTP(w, r)
		}
	})
}

// plantMachineKey writes the per-origin machine credential `cozy auth login` would.
func plantMachineKey(t *testing.T, root, origin, deviceID string, private ed25519.PrivateKey) {
	t.Helper()
	directory := filepath.Join(root, "auth")
	must(t, os.MkdirAll(directory, 0o700))
	key := sha256.Sum256([]byte(origin))
	credential, err := json.Marshal(map[string]any{"version": 1, "hub": origin, "email": deviceID + "@example.test",
		"device_key_id": deviceID, "private_key": base64.RawURLEncoding.EncodeToString(private.Seed())})
	must(t, err)
	must(t, os.WriteFile(filepath.Join(directory, hex.EncodeToString(key[:])+".json"), credential, 0o600))
}

// One daemon serves two Tensorhubs at once. A rental bought from hub A and a model
// upload to hub B are both live under it; switching the current hub strands neither,
// `rental end` reaches A with A's credential, and B never receives A's credential.
func TestOneDaemonServesTwoHubs(t *testing.T) {
	root, hubA, stand := rentalEndRoot(t, "multi-hub")
	// Each hub knows only its own machine key and mints its own bearer for it; hub a's
	// bearer is the one its rental routes accept.
	publicA, privateA, err := ed25519.GenerateKey(rand.Reader)
	must(t, err)
	witnessA := &hubWitness{}
	stand.server.Config.Handler = witnessA.wrap(machineKeyLogin("machine-a", publicA,
		"rental-idle-test", stand.server.Config.Handler))
	plantMachineKey(t, root, hubA, "machine-a", privateA)
	publicB, privateB, err := ed25519.GenerateKey(rand.Reader)
	must(t, err)
	witnessB := &hubWitness{}
	hubB := machineKeyHub(t, witnessB, "machine-b", publicB, "token-b", "proof").URL
	plantMachineKey(t, root, hubB, "machine-b", privateB)

	// Hub a is current; b is named beside it. No static token: machine keys only.
	configPath := filepath.Join(root, config.FileName)
	raw, err := os.ReadFile(configPath)
	must(t, err)
	body := strings.Replace(string(raw), "tensorhub_url: "+hubA, "tensorhub_url: a", 1)
	body = strings.Replace(body, "tensorhub_token: rental-idle-test\n", "", 1) +
		"hubs:\n  a: " + hubA + "\n  b: " + hubB + "\n"
	must(t, os.WriteFile(configPath, []byte(body), 0o600))

	const rentalID = "pr-a1a1a1a1a1a1a1a1a1a1"
	stand.add(rentalID, "alpha")
	store, problem := records.Open(filepath.Join(root, "creator.sqlite"))
	fatal(t, problem)
	fatal(t, store.RecordRental(records.Rental{ID: rentalID, MachineName: "alpha", State: "ready",
		Hub: hubA, AcceleratorCount: 1, HourlyRateUSDMicros: 100_000}))
	store.Close()

	if code, out := runCozy(t, root, "up", "--json"); code != 0 {
		t.Fatalf("up: %d %s", code, out)
	}
	started := daemon.Probe(config.Config{Home: root})
	if !started.Up {
		t.Fatal("the daemon did not start")
	}

	// The upload to hub b is accepted by the same daemon that holds a's rental.
	source := filepath.Join(root, "source.safetensors")
	must(t, os.WriteFile(source, []byte("multi-hub upload source"), 0o600))
	code, out := runCozy(t, root, "model", "upload", source, "proof/multi-hub", "--tensorhub=b",
		"--idempotency-key", "multi-hub-upload", "--json")
	if code != 0 {
		t.Fatalf("upload to hub b was refused: %d %s", code, out)
	}
	store, problem = records.Open(filepath.Join(root, "creator.sqlite"))
	fatal(t, problem)
	upload, problem := store.RequestByIdempotencyKey("multi-hub-upload")
	fatal(t, problem)
	store.Close()
	if upload == nil || upload.Hub != hubB {
		t.Fatalf("the upload did not record hub b: %+v", upload)
	}

	type inventory struct {
		Hub       string           `json:"hub"`
		Running   int              `json:"machines_running"`
		Rentals   []map[string]any `json:"rentals"`
		OtherHubs []map[string]any `json:"other_hubs"`
	}
	list := func(args ...string) inventory {
		t.Helper()
		code, out := runCozy(t, root, append([]string{"rental", "list", "--json"}, args...)...)
		if code != 0 {
			t.Fatalf("rental list %v: %d %s", args, code, out)
		}
		var doc inventory
		must(t, json.Unmarshal([]byte(out), &doc))
		return doc
	}
	if current := list(); current.Hub != hubA || len(current.Rentals) != 1 || len(current.OtherHubs) != 0 {
		t.Fatalf("hub a's listing is wrong: %+v", current)
	}
	// Hub b's fleet is read with b's credential; a's live rental is named, not hidden.
	if other := list("--tensorhub=b"); other.Hub != hubB || len(other.Rentals) != 0 ||
		len(other.OtherHubs) != 1 || other.OtherHubs[0]["hub"] != hubA {
		t.Fatalf("hub b's listing is wrong: %+v", other)
	}

	if code, out := runCozy(t, root, "hub", "use", "b", "--json"); code != 0 {
		t.Fatalf("hub use b: %d %s", code, out)
	}
	code, out = runCozy(t, root, "rental", "list", "--no-watch")
	if code != 0 || !strings.Contains(out, "1 rental running on hub a") {
		t.Fatalf("switching hubs hid a's billing rental: %d %s", code, out)
	}
	if all := list("--all-hubs"); len(all.Rentals) != 1 || all.Rentals[0]["hub"] != hubA {
		t.Fatalf("--all-hubs lost a's rental: %+v", all)
	}
	code, out = runCozy(t, root, "run", "list", "--all-hubs", "--json")
	if code != 0 || !strings.Contains(out, fmt.Sprintf("%q", hubB)) {
		t.Fatalf("run list --all-hubs lost the upload's hub: %d %s", code, out)
	}

	// Ending a's rental while b is current goes to a, with a's credential.
	code, out = runCozy(t, root, "rental", "end", "alpha", "--json")
	if code != 0 || !strings.Contains(out, `"state":"ended"`) {
		t.Fatalf("rental end on a while b is current: %d %s", code, out)
	}
	if n := stand.releases(rentalID); n != 1 {
		t.Fatalf("hub a saw %d releases of its rental, want 1", n)
	}
	if witnessA.saw("Bearer rental-idle-test") == 0 || witnessB.saw("Bearer token-b") == 0 {
		t.Fatal("a hub was never addressed with its own credential")
	}
	if n := witnessB.saw("Bearer rental-idle-test"); n != 0 {
		t.Fatalf("hub b received hub a's credential %d times", n)
	}
	if n := witnessA.saw("Bearer token-b"); n != 0 {
		t.Fatalf("hub a received hub b's credential %d times", n)
	}
	if witnessA.loggedIn("machine-b") != 0 || witnessB.loggedIn("machine-a") != 0 {
		t.Fatal("a machine key was presented to a hub it was not issued for")
	}
	if witnessA.loggedIn("machine-a") == 0 || witnessB.loggedIn("machine-b") == 0 {
		t.Fatal("a hub's own machine key was never used")
	}
	if after := daemon.Probe(config.Config{Home: root}); after.PID != started.PID {
		t.Fatalf("a hub switch replaced the daemon: %d -> %d", started.PID, after.PID)
	}
}

// Hub names work like kubectl contexts, and --tensorhub accepts a name for one command.
func TestHubContextsSelectTheCurrentHub(t *testing.T) {
	root := t.TempDir()
	if code, out := runCozy(t, root, "hub", "list", "--json"); code != 0 ||
		!strings.Contains(out, `"url":"`+config.DefaultHubURL+`"`) {
		t.Fatalf("a fresh install does not list the default hub: %d %s", code, out)
	}
	if code, out := runCozy(t, root, "hub", "add", "local", "http://127.0.0.1:8819/", "--json"); code != 0 {
		t.Fatalf("hub add: %d %s", code, out)
	}
	if code, out := runCozy(t, root, "hub", "add", "Bad Name", "http://127.0.0.1:1", "--json"); code == 0 {
		t.Fatalf("an invalid hub name was accepted: %s", out)
	}
	if code, out := runCozy(t, root, "hub", "use", "local", "--json"); code != 0 {
		t.Fatalf("hub use: %d %s", code, out)
	}
	raw, err := os.ReadFile(filepath.Join(root, config.FileName))
	must(t, err)
	if !strings.Contains(string(raw), "tensorhub_url: local") || !strings.Contains(string(raw), "local: http://127.0.0.1:8819\n") {
		t.Fatalf("hub selection was not recorded in config.yaml:\n%s", raw)
	}
	code, out := runCozy(t, root, "hub", "list", "--json")
	var doc struct {
		Hubs []map[string]any `json:"hubs"`
	}
	must(t, json.Unmarshal([]byte(out), &doc))
	current := ""
	for _, row := range doc.Hubs {
		if row["current"] == true {
			current, _ = row["name"].(string)
		}
	}
	if code != 0 || current != "local" {
		t.Fatalf("hub list does not mark local current: %d %s", code, out)
	}
	if code, out := runCozy(t, root, "hub", "remove", "local", "--json"); code == 0 || !strings.Contains(out, "hub.current") {
		t.Fatalf("the current hub's name was removed: %d %s", code, out)
	}
	if code, out := runCozy(t, root, "--tensorhub=nosuch", "hub", "list", "--json"); code == 0 ||
		!strings.Contains(out, "config.tensorhub_url_invalid") {
		t.Fatalf("an unknown hub name was accepted: %d %s", code, out)
	}
	if code, out := runCozy(t, root, "hub", "use", config.DefaultHubName, "--json"); code != 0 {
		t.Fatalf("hub use %s: %d %s", config.DefaultHubName, code, out)
	}
	if code, out := runCozy(t, root, "hub", "remove", "local", "--json"); code != 0 {
		t.Fatalf("hub remove: %d %s", code, out)
	}
	// An operator token names no hub of its own, so switching would carry it along.
	path := filepath.Join(root, config.FileName)
	raw, err = os.ReadFile(path)
	must(t, err)
	must(t, os.WriteFile(path, append(raw, []byte("tensorhub_token: operator-token\n")...), 0o600))
	if code, out := runCozy(t, root, "hub", "use", "http://127.0.0.1:8819", "--json"); code == 0 ||
		!strings.Contains(out, "hub.static_token_bound") {
		t.Fatalf("hub use carried the operator token to another hub: %d %s", code, out)
	}
}

// configuredHub is the hub a test root's config.yaml selects.
func configuredHub(t *testing.T, root string) string {
	t.Helper()
	raw, err := os.ReadFile(filepath.Join(root, config.FileName))
	must(t, err)
	for _, line := range strings.Split(string(raw), "\n") {
		if value, ok := strings.CutPrefix(line, "tensorhub_url: "); ok {
			return strings.TrimRight(strings.TrimSpace(value), "/")
		}
	}
	t.Fatal("the fixture selects no hub")
	return ""
}

// Requests recorded before per-request hubs read as the hub they were created under:
// their rental's when they have one, else the daemon's configured hub.
func TestSchema48RequestsTakeTheirHub(t *testing.T) {
	path := filepath.Join(t.TempDir(), "creator.sqlite")
	store, problem := records.OpenForDaemon(path, "")
	fatal(t, problem)
	const rentalHub, defaultHub = "http://rental-hub.invalid", "http://default-hub.invalid"
	fatal(t, store.RecordRental(records.Rental{ID: "pr-b2b2b2b2b2b2b2b2b2b2", MachineName: "bravo", State: "ready",
		Hub: rentalHub, AcceleratorCount: 1, HourlyRateUSDMicros: 100_000}))
	for _, request := range []records.Request{
		{ID: "req-pinned", Worker: "pr-b2b2b2b2b2b2b2b2b2b2", Rental: true},
		{ID: "req-local"},
	} {
		request.IdemKey, request.BodyDigest = request.ID, "sha256:"+strings.Repeat("ab", 32)
		request.Package, request.Entrypoint, request.Payload = "proof/example", "generate", []byte("{}")
		_, _, problem := store.Submit(request)
		fatal(t, problem)
	}
	store.Close()
	db, err := sql.Open("sqlite", path)
	must(t, err)
	revertRecordsSchema(t, db, 48)
	_, err = db.Exec(`PRAGMA user_version=48`)
	must(t, err)
	must(t, db.Close())

	store, problem = records.OpenForDaemon(path, "")
	fatal(t, problem)
	defer store.Close()
	fatal(t, store.AssignRecordHubs(defaultHub+"/"))
	for id, want := range map[string]string{"req-pinned": rentalHub, "req-local": defaultHub} {
		row, problem := store.RequestRow(id)
		fatal(t, problem)
		if row == nil || row.Hub != want {
			t.Fatalf("%s migrated to hub %+v, want %s", id, row, want)
		}
	}
}
