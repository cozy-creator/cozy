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

	"github.com/cozy-creator/cozy/internal/cli"
	"github.com/cozy-creator/cozy/internal/config"
	"github.com/cozy-creator/cozy/internal/daemon"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/home"
	"github.com/cozy-creator/cozy/internal/launch"
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
	code, out = runCozy(t, root, "hub", "use", "http://127.0.0.1:8819", "--json")
	if code == 0 || !strings.Contains(out, "hub.static_token_bound") ||
		!strings.Contains(out, "the tensorhub_token line in "+path) || !strings.Contains(out, "--tensorhub=http://127.0.0.1:8819") {
		t.Fatalf("hub use did not say which token setting to remove: %d %s", code, out)
	}
	must(t, os.WriteFile(path, raw, 0o600))
	command := exec.Command(cozyBin, "hub", "use", "http://127.0.0.1:8819", "--json")
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

// The operator tools judge a rental and its request on the hub they belong to, not on
// the hub this process happens to be configured for.
func TestOperatorToolsUseTheRecordsHub(t *testing.T) {
	t.Run("development hold", func(t *testing.T) {
		f := developmentFixtureAt(t)
		address, stop := serveIdleHoldPeer(t, f.peer, f.cert, "127.0.0.1:0")
		defer stop()
		f.attach(t, address)
		elsewhere := f.cfg
		elsewhere.HubURL = "https://another-hub.invalid"
		eligible, problem := cli.InspectStoredDevelopmentHold(elsewhere, f.rentalID, f.peer.bootID)
		fatal(t, problem)
		if eligible.State != "eligible" {
			t.Fatalf("development hold refused a rental from another hub: %+v", eligible)
		}
		ctx, cancel := context.WithCancel(context.Background())
		defer cancel()
		updates := make(chan cli.DevelopmentHoldResult, 16)
		finished := make(chan *exit.Error, 1)
		go func() {
			finished <- cli.HoldStoredDevelopmentWorker(ctx, elsewhere, f.rentalID, f.peer.bootID, io.Discard,
				func(value cli.DevelopmentHoldResult) { updates <- value })
		}()
		awaitDevelopmentState(t, updates, "holding")
		cancel()
		select {
		case problem := <-finished:
			fatal(t, problem)
		case <-time.After(10 * time.Second):
			t.Fatal("hold did not stop")
		}
	})
	t.Run("source custody", func(t *testing.T) {
		root := t.TempDir()
		store, problem := records.Open(filepath.Join(root, "creator.sqlite"))
		fatal(t, problem)
		defer store.Close()
		const rentalHub, rentalID, boot = "http://rental-hub.invalid", "pr-c4c4c4c4c4c4c4c4c4c4", "custody-boot"
		fatal(t, store.RecordRental(records.Rental{ID: rentalID, MachineName: "charlie", State: "ready", Hub: rentalHub,
			AcceleratorCount: 1, HourlyRateUSDMicros: 100_000, ExpectedWorkerBootID: boot}))
		header := []byte(`{"weight_map":{"x":"shard.safetensors"}}`)
		submit := func(id, hub string) {
			t.Helper()
			intent := &records.ModelTransferIntent{Kind: "model-upload", Destination: "proof/model",
				Source: "hf://proof/source@" + strings.Repeat("4", 40), SourceSelection: "sha256:" + strings.Repeat("2", 64),
				SourceProfiles: map[string]string{"shared": "hf/minimax-h3/shared-bf16/1"},
				SourceFiles: []records.ModelTransferSourceFile{{Member: "model.safetensors.index.json",
					SHA256: strings.Repeat("1", 64), Length: int64(len(header)), Header: header}},
				Outputs: []records.ModelTransferOutput{{Name: "model"}}}
			_, _, problem := store.Submit(records.Request{ID: id, IdemKey: id, BodyDigest: "sha256:" + strings.Repeat("c", 64),
				Package: "proof/producer", Entrypoint: "produce", Kind: "job", Payload: []byte("{}"), Outputs: "[]",
				WeightsOutputs: "[]", Worker: rentalID, Rental: true, ModelTransfer: intent, Hub: hub})
			fatal(t, problem)
			fatal(t, store.BeginModelTransferMaterialization(id))
			fatal(t, store.ObserveModelSourceCheckpoints(id, intent.SourceSelection, boot, []records.ModelCheckpoint{{Slot: "shared",
				HeadID: "sha256:" + strings.Repeat("3", 64), HeadLength: 500, PlanDigest: "sha256:" + strings.Repeat("4", 64), Index: 1, Bytes: 64}}))
		}
		submit("job-custody-own-hub", rentalHub)
		elsewhere := config.Config{Home: root, HubURL: "https://another-hub.invalid"}
		result, problem := cli.InspectStoredSourceCustody(elsewhere, "job-custody-own-hub", rentalID, boot)
		fatal(t, problem)
		if len(result.Checkpoints) != 1 {
			t.Fatalf("source custody inspection lost its progress: %+v", result)
		}
		submit("job-custody-crossed", "http://request-hub.invalid")
		if _, problem := cli.InspectStoredSourceCustody(elsewhere, "job-custody-crossed", rentalID, boot); problem == nil ||
			!strings.Contains(problem.Message, "belongs to http://request-hub.invalid but its rental to "+rentalHub) {
			t.Fatalf("a request and rental on different hubs were accepted: %v", problem)
		}
	})
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
	// The machine resolves the call's Models; the release it runs is read from the hub the
	// install came from, never from the current one.
	code, out := runCozy(t, root, "run", "proof/alpha/generate", "--json")
	if _, asked := askedA.Load("proof/alpha/releases/1.0.0"); !asked {
		t.Fatalf("the install's hub was not asked for its release: %d %s", code, out)
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
