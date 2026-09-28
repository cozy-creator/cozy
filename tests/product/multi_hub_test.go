package producttest

import (
	"bytes"
	"context"
	"crypto/ed25519"
	"crypto/rand"
	"crypto/sha256"
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/cozy-creator/cozy/internal/config"
	"github.com/cozy-creator/cozy/internal/daemon"
	"github.com/cozy-creator/cozy/internal/home"
	"github.com/cozy-creator/cozy/internal/host"
	"github.com/cozy-creator/cozy/internal/hub"
	"github.com/cozy-creator/cozy/internal/launch"
	"github.com/cozy-creator/cozy/internal/records"
	"github.com/cozy-creator/cozy/internal/secret"
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
	if code, out := runCozy(t, root, "hub", "add", "local", "http://127.0.0.1:1/", "--json"); code != 0 {
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
	if !strings.Contains(string(raw), "tensorhub_url: local") || !strings.Contains(string(raw), "local: http://127.0.0.1:1\n") {
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
	code, out = runCozy(t, root, "hub", "use", "http://127.0.0.1:1", "--json")
	if code == 0 || !strings.Contains(out, "hub.static_token_bound") ||
		!strings.Contains(out, "the tensorhub_token line in "+path) || !strings.Contains(out, "--tensorhub=http://127.0.0.1:1") {
		t.Fatalf("hub use did not say which token setting to remove: %d %s", code, out)
	}
	must(t, os.WriteFile(path, raw, 0o600))
	command := exec.Command(cozyBin, "hub", "use", "http://127.0.0.1:1", "--json")
	command.Env = childEnv(t, root, "TENSORHUB_TOKEN=operator-token")
	env, _ := command.CombinedOutput()
	if command.ProcessState.ExitCode() == 0 || !strings.Contains(string(env), "the TENSORHUB_TOKEN environment variable") ||
		strings.Contains(string(env), "tensorhub_token line") {
		t.Fatalf("hub use did not name the environment token: %s", env)
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

// An every-hub listing names a hub it cannot read, keeps that hub's rentals as this
// host's records marked unverified, and still lists every other hub in full. Run
// history is local and never depends on a hub.
func TestAllHubsListingIsolatesAnUnreadableHub(t *testing.T) {
	root, hubA, standA := rentalEndRoot(t, "all-hubs-isolation")
	standB := newFakeRentalHub(t, 0)
	hubB := standB.server.URL
	public, private, err := ed25519.GenerateKey(rand.Reader)
	must(t, err)
	standB.server.Config.Handler = machineKeyLogin("machine-b", public, "rental-idle-test", standB.server.Config.Handler)
	configPath := filepath.Join(root, config.FileName)
	raw, err := os.ReadFile(configPath)
	must(t, err)
	must(t, os.WriteFile(configPath, append(raw, []byte("hubs:\n  a: "+hubA+"\n  b: "+hubB+"\n")...), 0o600))

	const rentalA, rentalB = "pr-a3a3a3a3a3a3a3a3a3a3", "pr-b3b3b3b3b3b3b3b3b3b3"
	standA.add(rentalA, "alpha")
	standB.add(rentalB, "bravo")
	store, problem := records.Open(filepath.Join(root, "creator.sqlite"))
	fatal(t, problem)
	for id, row := range map[string]records.Rental{
		rentalA: {MachineName: "alpha", Hub: hubA}, rentalB: {MachineName: "bravo", Hub: hubB},
	} {
		row.ID, row.State, row.AcceleratorCount, row.HourlyRateUSDMicros = id, "ready", 1, 100_000
		fatal(t, store.RecordRental(row))
	}
	store.Close()
	if code, out := runCozy(t, root, "up", "--json"); code != 0 {
		t.Fatalf("up: %d %s", code, out)
	}

	type listing struct {
		Rentals    []map[string]any `json:"rentals"`
		Unreadable []struct {
			Hub  string `json:"hub"`
			Code string `json:"code"`
		} `json:"unreadable_hubs"`
	}
	list := func(wantCode int) listing {
		t.Helper()
		code, out := runCozy(t, root, "rental", "list", "--all-hubs", "--json")
		if code != wantCode {
			t.Fatalf("rental list --all-hubs exited %d, want %d: %s", code, wantCode, out)
		}
		var doc listing
		must(t, json.Unmarshal([]byte(out), &doc))
		return doc
	}
	unverified := func(doc listing) map[string]bool {
		out := map[string]bool{}
		for _, row := range doc.Rentals {
			out[row["rental_id"].(string)] = row["unverified"] == true
		}
		return out
	}
	// Hub b refuses: this machine holds no login for it.
	doc := list(1)
	if rows := unverified(doc); len(rows) != 2 || rows[rentalA] || !rows[rentalB] ||
		len(doc.Unreadable) != 1 || doc.Unreadable[0].Hub != hubB || doc.Unreadable[0].Code != "auth.machine_key_missing" {
		_, raw := runCozy(t, root, "rental", "list", "--all-hubs", "--json")
		t.Fatalf("a refusing hub hid or mixed up another: %+v\n%s", doc, raw)
	}
	if code, out := runCozy(t, root, "rental", "list", "--all-hubs", "--no-watch"); code != 1 ||
		!strings.Contains(out, "hub b unreachable: this machine is not logged in to "+hubB+" (next: cozy auth login <email> --tensorhub=b)") {
		t.Fatalf("a refusing hub's line does not say how to log in: %d %s", code, out)
	}
	plantMachineKey(t, root, hubB, "machine-b", private)
	if doc := list(0); len(doc.Unreadable) != 0 || len(unverified(doc)) != 2 || unverified(doc)[rentalB] {
		t.Fatalf("both hubs readable, listing still degraded: %+v", doc)
	}
	// Hub b stops answering.
	standB.close()
	if doc := list(1); len(doc.Unreadable) != 1 || doc.Unreadable[0].Hub != hubB || unverified(doc)[rentalA] || !unverified(doc)[rentalB] {
		t.Fatalf("an unreachable hub hid or mixed up another: %+v", doc)
	}
	code, out := runCozy(t, root, "rental", "list", "--all-hubs", "--no-watch")
	if code != 1 || !strings.Contains(out, "hub b unreachable: ") || !strings.Contains(out, "its rentals may still be billing") ||
		!strings.Contains(out, "alpha") || !strings.Contains(out, "bravo") {
		t.Fatalf("human listing did not name the unreadable hub beside the readable one: %d %s", code, out)
	}
	if code, out := runCozy(t, root, "run", "list", "--all-hubs", "--json"); code != 0 {
		t.Fatalf("run history depended on a hub: %d %s", code, out)
	}
}

// packageCardHub serves one package card per name and counts which packages it was asked for.
func packageCardHub(t *testing.T, asked *sync.Map) *httptest.Server {
	t.Helper()
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		name, ok := strings.CutPrefix(r.URL.Path, "/v1/packages/")
		if !ok || r.Method != http.MethodGet {
			http.NotFound(w, r)
			return
		}
		count, _ := asked.LoadOrStore(name, new(int))
		*count.(*int)++
		org, pkg, _ := strings.Cut(name, "/")
		_ = json.NewEncoder(w).Encode(map[string]any{
			"package":  map[string]any{"org": org, "name": pkg, "created_at": "2026-09-01T00:00:00Z", "latest_release": "1.0.0"},
			"releases": []any{map[string]any{"release": "1.0.0", "cut_at": "2026-09-01T00:00:00Z"}},
		})
	}))
	t.Cleanup(server.Close)
	return server
}

// Each published install is checked against the hub it came from, and a hub that cannot
// be read fails only its own packages.
func TestPackageUpdateAllUsesEachInstallsHub(t *testing.T) {
	root := t.TempDir()
	var askedA, askedB sync.Map
	hubA, hubB := packageCardHub(t, &askedA), packageCardHub(t, &askedB)
	must(t, os.WriteFile(filepath.Join(root, config.FileName),
		[]byte("tensorhub_url: a\nhubs:\n  a: "+hubA.URL+"\n  b: "+hubB.URL+"\n"), 0o600))
	layout, problem := home.Open(root)
	fatal(t, problem)
	store, problem := records.Open(layout.DB)
	fatal(t, problem)
	for id, install := range map[string]records.PackageInstall{
		"1111111111111111": {Package: "proof/alpha", Hub: hubA.URL},
		"2222222222222222": {Package: "proof/beta", Hub: hubB.URL},
	} {
		install.ID, install.Major, install.Version, install.Verified = id, 1, "1.0.0", true
		install.SourceKind, install.SourceRef, install.Dir = "tensorhub", install.Package+"@1.0.0", layout.InstallDir(id)
		_, problem := store.Activate(install)
		fatal(t, problem)
	}
	store.Close()
	type updated struct {
		Packages []map[string]any `json:"packages"`
	}
	update := func(wantCode int) map[string]map[string]any {
		t.Helper()
		code, out := runCozy(t, root, "package", "update-all", "--json", "--full")
		if code != wantCode {
			t.Fatalf("package update-all exited %d, want %d: %s", code, wantCode, out)
		}
		var doc updated
		must(t, json.Unmarshal([]byte(out), &doc))
		rows := map[string]map[string]any{}
		for _, row := range doc.Packages {
			rows[row["package"].(string)] = row
		}
		return rows
	}
	rows := update(0)
	if rows["proof/alpha"]["status"] != "current" || rows["proof/beta"]["status"] != "current" ||
		rows["proof/alpha"]["hub"] != "a" || rows["proof/beta"]["hub"] != "b" {
		t.Fatalf("installs were not checked against their own hubs: %+v", rows)
	}
	if _, crossed := askedA.Load("proof/beta"); crossed {
		t.Fatal("hub a was asked for hub b's package")
	}
	if _, crossed := askedB.Load("proof/alpha"); crossed {
		t.Fatal("hub b was asked for hub a's package")
	}
	hubB.Close()
	rows = update(1)
	if rows["proof/alpha"]["status"] != "current" || rows["proof/beta"]["status"] != "failed" {
		t.Fatalf("an unreachable hub failed more than its own packages: %+v", rows)
	}
	code, out := runCozy(t, root, "package", "list", "--json", "--full")
	if code != 0 || !strings.Contains(out, `"hub":"b"`) || !strings.Contains(out, `"hub":"a"`) {
		t.Fatalf("package list does not show each install's hub: %d %s", code, out)
	}
}

// A local run of a published install reads its model bindings from the hub it was
// installed from, not from whichever hub is current.
func TestLocalRunOfAnInstallUsesItsHub(t *testing.T) {
	root := t.TempDir()
	var askedA, askedB sync.Map
	hubA, hubB := packageCardHub(t, &askedA), packageCardHub(t, &askedB)
	must(t, os.WriteFile(filepath.Join(root, config.FileName),
		[]byte("tensorhub_url: b\nport: 0\nhubs:\n  a: "+hubA.URL+"\n  b: "+hubB.URL+"\n"), 0o600))
	layout, problem := home.Open(root)
	fatal(t, problem)
	store, problem := records.Open(layout.DB)
	fatal(t, problem)
	install := records.PackageInstall{ID: "5555555555555555", Package: "proof/alpha", Major: 1, Version: "1.0.0",
		Verified: true, SourceKind: "tensorhub", SourceRef: "proof/alpha@1.0.0", Dir: layout.InstallDir("5555555555555555"), Hub: hubA.URL}
	iface := []byte(`{"format":"cozy.package.interface/1","application":"alpha:app","entrypoints":[{"name":"generate",` +
		`"models":[{"class":"Model","path":"generate.models.model","component_use":{}}],` +
		`"request":{"fields":[]},"result":{"fields":[]}}],"jobs":[]}`)
	must(t, os.MkdirAll(filepath.Dir(launch.PackageInterfacePath(install.Dir)), 0o700))
	must(t, os.WriteFile(launch.PackageInterfacePath(install.Dir), iface, 0o444))
	_, problem = store.Activate(install)
	fatal(t, problem)
	store.Close()
	// This computer's machine is registered with the install's hub, as a signed-in first run
	// there leaves it: its Host reads that hub's catalog at the hub's worker doors.
	provisionMachine(t, root)
	if *machineHostBinary != "" {
		doors, ca := hubTLSServer(t, hubA.Config.Handler)
		t.Cleanup(doors.Close)
		registerMachine(t, root, hubA.URL, map[string]string{"TENSORHUB_ORIGIN": doors.URL, "TENSORHUB_PUBLIC_ORIGIN": doors.URL,
			"TENSORHUB_CA_DER_B64URL": base64.RawURLEncoding.EncodeToString(ca)})
	}
	// The machine resolves the call's Models; the release it runs is read from the hub the
	// install came from, never from the current one.
	code, out := runCozy(t, root, "run", "proof/alpha/generate", "--json")
	// A queued run reaches its machine after the command returns.
	for deadline := time.Now().Add(time.Minute); ; time.Sleep(50 * time.Millisecond) {
		if _, asked := askedA.Load("proof/alpha/releases/1.0.0"); asked {
			break
		}
		if time.Now().After(deadline) {
			t.Fatalf("the install's hub was not asked for its release: %d %s", code, out)
		}
	}
	crossed := false
	askedB.Range(func(key, _ any) bool {
		crossed = crossed || strings.HasPrefix(key.(string), "proof/alpha")
		return true
	})
	if crossed {
		t.Fatal("the current hub was asked for another hub's install")
	}
}

// Runs recorded before requests carried a hub stay listed: under their rental's hub, else
// the configured one, and every hub's listing holds them all.
func TestRunsWithoutAHubStayListed(t *testing.T) {
	root := t.TempDir()
	const hubA, hubB = "http://127.0.0.1:1", "http://127.0.0.1:2"
	must(t, os.WriteFile(filepath.Join(root, config.FileName), []byte("tensorhub_url: a\nport: 0\n"+
		"daemon:\n  idle_shutdown_s: 0\nhubs:\n  a: "+hubA+"\n  b: "+hubB+"\n"), 0o600))
	store, problem := records.Open(filepath.Join(root, "creator.sqlite"))
	fatal(t, problem)
	defer store.Close()
	fatal(t, store.RecordRental(records.Rental{ID: "pr-d5d5d5d5d5d5d5d5d5d5", MachineName: "delta", State: "ready",
		Hub: hubB, AcceleratorCount: 1, HourlyRateUSDMicros: 100_000}))
	startDaemonProcess(t, root)
	for id, worker := range map[string]string{"req-hubless-local": "", "req-hubless-rented": "pr-d5d5d5d5d5d5d5d5d5d5"} {
		_, _, problem := store.Submit(records.Request{ID: id, IdemKey: id, BodyDigest: "sha256:" + strings.Repeat("ab", 32),
			Package: "proof/example", Entrypoint: "generate", Payload: []byte("{}"), Worker: worker, Rental: worker != ""})
		fatal(t, problem)
	}
	runs := func(args ...string) map[string]string {
		t.Helper()
		code, out := runCozy(t, root, append([]string{"run", "list", "--json", "--full"}, args...)...)
		var doc struct {
			Runs []map[string]any `json:"invocations"`
		}
		if code != 0 || json.Unmarshal([]byte(out), &doc) != nil {
			t.Fatalf("run list %v: %d %s", args, code, out)
		}
		hubs := map[string]string{}
		for _, run := range doc.Runs {
			hub, _ := run["hub"].(string)
			hubs[run["id"].(string)] = hub
		}
		return hubs
	}
	if current := runs(); len(current) != 1 || current["req-hubless-local"] != hubA {
		t.Fatalf("the configured hub's listing lost its run: %+v", current)
	}
	if other := runs("--tensorhub=b"); len(other) != 1 || other["req-hubless-rented"] != hubB {
		t.Fatalf("the rental's hub's listing lost its run: %+v", other)
	}
	if all := runs("--all-hubs"); len(all) != 2 || all["req-hubless-local"] != hubA || all["req-hubless-rented"] != hubB {
		t.Fatalf("every hub's listing lost a run: %+v", all)
	}
}

// A package published only at a second hub runs on this computer's machine through
// `cozy run --tensorhub <second>`. The machine is registered with both hubs; it reads the
// release at the hub the command names and never asks the first hub for it.
func TestLocalRunReadsTheCommandsHub(t *testing.T) {
	if *machineHostBinary == "" {
		t.Skip("requires -machine-host: this computer's machine reads the release")
	}
	iface := json.RawMessage(`{"format":"cozy.package.interface/1","application":"beta:app","entrypoints":[{"name":"generate",` +
		`"models":[],"request":{"fields":[]},"result":{"fields":[]}}],"jobs":[]}`)
	var askedA, askedB, machineA, machineB sync.Map
	catalog := func(asked *sync.Map, publishes bool) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			asked.Store(r.URL.Path, true)
			if !publishes || !strings.HasPrefix(r.URL.Path, "/v1/packages/proof/beta") {
				http.NotFound(w, r)
				return
			}
			switch strings.TrimPrefix(r.URL.Path, "/v1/packages/proof/beta") {
			case "":
				_ = json.NewEncoder(w).Encode(map[string]any{
					"package":  map[string]any{"org": "proof", "name": "beta", "latest_release": "1.0.0"},
					"releases": []any{map[string]any{"release": "1.0.0", "cut_at": "2026-09-01T00:00:00Z"}}})
			case "/releases/1.0.0":
				_ = json.NewEncoder(w).Encode(map[string]any{"release": map[string]any{"release": "1.0.0"},
					"package_interface": iface, "requires_python": ">=3.12", "python_version": "3.12"})
			case "/releases/1.0.0/locked-requirements":
				_, _ = w.Write([]byte("msgspec==0.19.0\n"))
			case "/bindings":
				_ = json.NewEncoder(w).Encode(map[string]any{"bindings": []any{}})
			default:
				http.NotFound(w, r)
			}
		})
	}
	hubA, hubB := httptest.NewServer(catalog(&askedA, false)), httptest.NewServer(catalog(&askedB, true))
	t.Cleanup(hubA.Close)
	t.Cleanup(hubB.Close)
	root := t.TempDir()
	must(t, os.WriteFile(filepath.Join(root, config.FileName),
		[]byte("tensorhub_url: a\nport: 0\ntensorhub_token: two-hubs\nhubs:\n  a: "+hubA.URL+"\n  b: "+hubB.URL+"\n"), 0o600))
	provisionMachine(t, root)
	for _, hub := range []struct {
		origin  string
		handler http.Handler
	}{{hubA.URL, catalog(&machineA, false)}, {hubB.URL, catalog(&machineB, true)}} {
		doors, ca := hubTLSServer(t, hub.handler)
		t.Cleanup(doors.Close)
		registerMachineAt(t, root, hub.origin, map[string]string{"TENSORHUB_ORIGIN": doors.URL, "TENSORHUB_PUBLIC_ORIGIN": doors.URL,
			"TENSORHUB_CA_DER_B64URL": base64.RawURLEncoding.EncodeToString(ca)})
	}
	code, out := runCozy(t, root, "run", "proof/beta/generate", "--tensorhub", "b", "--json")
	if strings.Contains(out, "not a published release") || strings.Contains(out, "not a published package") {
		t.Fatalf("the machine read the release at another hub than the command's: %d %s", code, out)
	}
	// The machine, at the command's hub's doors, read the release's install facts.
	if _, asked := machineB.Load("/v1/packages/proof/beta/releases/1.0.0/locked-requirements"); !asked {
		t.Fatalf("the machine did not read the release at the command's hub: %d %s", code, out)
	}
	for _, other := range []*sync.Map{&askedA, &machineA} {
		other.Range(func(key, _ any) bool {
			if strings.HasPrefix(key.(string), "/v1/packages/proof/beta") {
				t.Fatalf("the other hub was asked %s for the command's package", key)
			}
			return true
		})
	}
	var registered map[string]json.RawMessage
	raw, err := os.ReadFile(filepath.Join(root, "machine", "registrations.json"))
	must(t, err)
	must(t, json.Unmarshal(raw, &registered))
	if len(registered) != 2 {
		t.Fatalf("the machine holds %d registrations, not one per hub: %s", len(registered), raw)
	}
}

// A newer hub may hand a registered machine settings this CLI does not know. Registration
// still succeeds: the hub's facts are kept and anything else is ignored, never refused.
// This computer's machine holds a registration at each hub it has run work of. A run of hub
// b after one of hub a reads its release at b's doors, and a model of hub b lands from b's
// doors, each with b's registration; hub a is asked nothing of either. A machine whose range
// reaches wire 67 serves both hubs where it is: the Host that served a serves b. An older
// one moves to b instead.
func TestLocalMachineServesEveryHubWhereItIs(t *testing.T) {
	if *machineHostBinary == "" {
		t.Skip("requires -machine-host: this computer's machine reads the releases")
	}
	iface := json.RawMessage(`{"format":"cozy.package.interface/1","application":"app:app","entrypoints":[{"name":"generate",` +
		`"models":[],"request":{"fields":[]},"result":{"fields":[]}}],"jobs":[]}`)
	type door struct {
		asked, workers, fetchedAs sync.Map
	}
	manifest := "sha256:" + strings.Repeat("a", 64)
	catalog := func(d *door, pkg string) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			d.asked.Store(r.URL.Path, true)
			if worker := r.Header.Get("X-Cozy-Worker-ID"); worker != "" {
				d.workers.Store(worker, true)
			}
			switch {
			case r.URL.Path == "/v1/tensorfs/closure":
				d.fetchedAs.Store(r.Header.Get("X-Cozy-Worker-ID"), true)
				// Bytes that are not the model: TensorFS fetches them and refuses what arrives.
				_ = json.NewEncoder(w).Encode(map[string]any{"complete": true, "lane": "", "model": "proof/model",
					"manifest":            map[string]any{"length": 128, "sha256": strings.TrimPrefix(manifest, "sha256:")},
					"objects":             []any{map[string]any{"length": 64, "sha256": strings.Repeat("b", 64)}},
					"presign_max_digests": 64, "release": "", "scope": "runtime"})
				return
			case r.URL.Path == "/v1/tensorfs/presign":
				var asked struct{ Digests []string }
				_ = json.NewDecoder(r.Body).Decode(&asked)
				urls := map[string]string{}
				for _, digest := range asked.Digests {
					urls[digest] = "https://" + r.Host + "/o/" + digest
				}
				now := time.Now().Unix()
				_ = json.NewEncoder(w).Encode(map[string]any{"expires_at_unix": now + 3600, "server_time_unix": now, "urls": urls})
				return
			case strings.HasPrefix(r.URL.Path, "/o/"):
				_, _ = w.Write(make([]byte, 64))
				return
			}
			switch strings.TrimPrefix(r.URL.Path, "/v1/packages/"+pkg) {
			case "":
				_ = json.NewEncoder(w).Encode(map[string]any{"releases": []any{map[string]any{"release": "1.0.0"}}})
			case "/releases/1.0.0":
				_ = json.NewEncoder(w).Encode(map[string]any{"release": map[string]any{"release": "1.0.0"},
					"package_interface": iface, "requires_python": ">=3.12", "python_version": "3.12"})
			case "/releases/1.0.0/locked-requirements":
				_, _ = w.Write([]byte("msgspec==0.19.0\n"))
			case "/bindings":
				_ = json.NewEncoder(w).Encode(map[string]any{"bindings": []any{}})
			default:
				http.NotFound(w, r)
			}
		})
	}
	// Each hub's own API, as the command reads it.
	var apiA, apiB sync.Map
	api := func(asked *sync.Map) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			asked.Store(r.URL.Path, true)
			if r.URL.Path != "/v1/models/resolve" {
				http.NotFound(w, r)
				return
			}
			_ = json.NewEncoder(w).Encode(map[string]any{"model": "proof/model", "manifest_id": manifest, "manifest_length": 128,
				"bytes": 4096, "components": []string{"transformer"}})
		})
	}
	hubA, hubB := httptest.NewServer(api(&apiA)), httptest.NewServer(api(&apiB))
	t.Cleanup(hubA.Close)
	t.Cleanup(hubB.Close)
	root := t.TempDir()
	must(t, os.WriteFile(filepath.Join(root, config.FileName),
		[]byte("tensorhub_url: a\nport: 0\ntensorhub_token: two-hubs\nhubs:\n  a: "+hubA.URL+"\n  b: "+hubB.URL+"\n"), 0o600))
	provisionMachine(t, root)
	var doorA, doorB door
	for _, hub := range []struct {
		origin  string
		handler http.Handler
	}{{hubA.URL, catalog(&doorA, "proof/alpha")}, {hubB.URL, catalog(&doorB, "proof/beta")}} {
		doors, ca := hubTLSServer(t, hub.handler)
		t.Cleanup(doors.Close)
		registerMachineAt(t, root, hub.origin, map[string]string{"TENSORHUB_ORIGIN": doors.URL, "TENSORHUB_PUBLIC_ORIGIN": doors.URL,
			"TENSORHUB_CA_DER_B64URL": base64.RawURLEncoding.EncodeToString(ca)})
	}
	var registered map[string]struct {
		ID string `json:"id"`
	}
	raw, err := os.ReadFile(filepath.Join(root, "machine", "registrations.json"))
	must(t, err)
	must(t, json.Unmarshal(raw, &registered))
	launched := func() (record struct {
		PID       int    `json:"pid"`
		WireMinor uint32 `json:"wire_minor"`
	}) {
		raw, err := os.ReadFile(filepath.Join(root, "machine", "host.json"))
		must(t, err)
		must(t, json.Unmarshal(raw, &record))
		return record
	}

	code, out := runCozy(t, root, "run", "proof/alpha/generate", "--json")
	if _, asked := doorA.asked.Load("/v1/packages/proof/alpha/releases/1.0.0/locked-requirements"); !asked {
		t.Fatalf("the machine did not read hub a's release: %d %s", code, out)
	}
	first := launched()
	if first.PID == 0 || first.WireMinor == 0 {
		t.Fatalf("the launched Host's record names no process or protocol range: %+v", first)
	}
	code, out = runCozy(t, root, "run", "proof/beta/generate", "--tensorhub", "b", "--json")
	if _, asked := doorB.asked.Load("/v1/packages/proof/beta/releases/1.0.0/locked-requirements"); !asked {
		t.Fatalf("the machine did not read the release at the run's hub: %d %s", code, out)
	}
	for _, other := range []*sync.Map{&apiA, &doorA.asked} {
		other.Range(func(path, _ any) bool {
			if strings.HasPrefix(path.(string), "/v1/packages/proof/beta") {
				t.Fatalf("hub a was asked %s for hub b's package", path)
			}
			return true
		})
	}
	second := launched()
	if (second.PID == first.PID) != (first.WireMinor >= 67) {
		t.Fatalf("a machine at wire %d moved (Host %d, then %d): it moves exactly when it cannot read a run's hub",
			first.WireMinor, first.PID, second.PID)
	}
	code, out = runCozy(t, root, "model", "download", "proof/model#"+manifest, "--tensorhub", "b", "--json")
	var accepted struct{ ID string }
	if code != 0 || json.Unmarshal([]byte(out), &accepted) != nil || accepted.ID == "" {
		t.Fatalf("the model download of hub b was not accepted [exit %d]\n%s", code, out)
	}
	store, problem := records.Open(filepath.Join(root, "creator.sqlite"))
	fatal(t, problem)
	defer store.Close()
	for deadline := time.Now().Add(3 * time.Minute); ; time.Sleep(250 * time.Millisecond) {
		row, problem := store.RentalInstall(accepted.ID)
		fatal(t, problem)
		if row != nil && (row.State == "succeeded" || row.State == "failed") {
			break
		}
		if time.Now().After(deadline) {
			t.Fatal("the machine never settled the model download")
		}
	}
	if _, as := doorB.fetchedAs.Load(registered[hubB.URL].ID); !as {
		t.Fatal("the machine did not fetch hub b's model at b's doors with its registration there")
	}
	if _, asked := doorA.asked.Load("/v1/tensorfs/closure"); asked {
		t.Fatal("hub a's doors were asked for hub b's model")
	}
	if _, asked := apiB.Load("/v1/models/resolve"); !asked {
		t.Fatal("the model of hub b was not resolved at hub b")
	}
	if third := launched(); third.PID != second.PID {
		t.Fatalf("the machine moved for a model of the hub it served (Host %d, then %d)", second.PID, third.PID)
	}
	if _, seen := doorB.workers.Load(registered[hubB.URL].ID); !seen {
		t.Fatalf("hub b's doors were not read as the machine's registration there: %d %s", code, out)
	}
	doorB.workers.Range(func(worker, _ any) bool {
		if worker != registered[hubB.URL].ID {
			t.Fatalf("hub b's doors were read as %s, not as the machine's registration there", worker)
		}
		return true
	})
}

func TestMachineRegistrationToleratesNewHubSettings(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost || r.URL.Path != "/v1/machines" {
			http.NotFound(w, r)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusCreated)
		_ = json.NewEncoder(w).Encode(map[string]any{"id": "om-" + randomToken(t)[:22], "worker_token": randomToken(t),
			"environment": map[string]string{"TENSORHUB_ORIGIN": "https://hub.example", "TENSORHUB_FUTURE_FACT": "1",
				"COZY_WEBRTC_INTERNAL_PORT": "8445", "SOME_NEW_SETTING": "x"}})
	}))
	t.Cleanup(server.Close)
	machine, problem := hub.New(config.Config{HubURL: server.URL, HubToken: secret.New("fixture")}, "").RegisterMachine(context.Background())
	fatal(t, problem)
	if machine.Environment["TENSORHUB_ORIGIN"] != "https://hub.example" || machine.Environment["TENSORHUB_FUTURE_FACT"] != "1" {
		t.Fatalf("the hub's facts were not kept: %v", machine.Environment)
	}
	if _, passed := machine.Environment["COZY_WEBRTC_INTERNAL_PORT"]; passed || strings.Join(machine.Ignored, ",") != "COZY_WEBRTC_INTERNAL_PORT,SOME_NEW_SETTING" {
		t.Fatalf("settings the machine does not read reached it or went unnamed: %v ignored %v", machine.Environment, machine.Ignored)
	}
}

// The launcher hands the machine every other registration in COZY_MACHINE_HUBS_JSON. One the
// machine cannot read is named and skipped; it never stops the machine from booting.
func TestMachineGrantReadsOtherHubsTolerantly(t *testing.T) {
	key := make([]byte, 32)
	token := base64.RawURLEncoding.EncodeToString(bytes.Repeat([]byte{1}, 32))
	hub := func(origin, id, token string) map[string]any {
		return map[string]any{"worker_id": id, "worker_token": token, "future": true,
			"environment": map[string]string{"TENSORHUB_ORIGIN": origin, "TENSORHUB_PUBLIC_ORIGIN": origin,
				"TENSORHUB_OBJECT_STORAGE_HOSTS": "objects.b.example"}}
	}
	others, err := json.Marshal([]any{hub("https://b.example", "om-b", token), hub("https://c.example", "om-c", "short"),
		hub("https://a.example", "om-a2", token), hub("http://d.example", "om-d", token), "not a registration"})
	must(t, err)
	base := []string{"COZY_MACHINE_ROOT=/", "COZY_WORKER_ID=om-a", "COZY_WORKER_AUTH_TOKEN=" + token,
		"COZY_WORKER_INTERNAL_PORT=8443", "TENSORHUB_ORIGIN=https://a.example",
		"COZY_AUTHORIZED_KEYS=" + base64.RawURLEncoding.EncodeToString(key)}
	grant, err := host.ReadGrant(append(base, "COZY_MACHINE_HUBS_JSON="+string(others)))
	must(t, err)
	if len(grant.Hubs) != 1 || grant.Hubs[0].Origin != "https://b.example" || grant.Hubs[0].WorkerID != "om-b" ||
		strings.Join(grant.Hubs[0].ObjectHosts, ",") != "objects.b.example" || len(grant.Skipped) != 4 {
		t.Fatalf("other hubs read as %+v, skipped %q", grant.Hubs, grant.Skipped)
	}
	grant, err = host.ReadGrant(append(base, "COZY_MACHINE_HUBS_JSON={"))
	must(t, err)
	if len(grant.Hubs) != 0 || len(grant.Skipped) != 1 {
		t.Fatalf("an unreadable list read as %+v, skipped %q", grant.Hubs, grant.Skipped)
	}
}
