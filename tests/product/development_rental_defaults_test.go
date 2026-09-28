package producttest

import (
	"bytes"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/cozy-creator/cozy/internal/config"
	"github.com/cozy-creator/cozy/internal/hub"
)

const developmentFixtureKey = "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAINdamAGCsQq31Uv+08lkBzoO4XLz2qYjJa8CGmj3B1Ea fixture"

func configureDevelopmentRental(t *testing.T, root string) {
	t.Helper()
	path := filepath.Join(root, config.FileName)
	raw, err := os.ReadFile(path)
	must(t, err)
	raw = append(raw, []byte("rentals:\n  development: true\n  ssh_public_key: operator.pub\n")...)
	must(t, os.WriteFile(path, raw, 0600))
}

func TestDevelopmentDefaultsReachManualAndManagedAcquisitions(t *testing.T) {
	for _, managed := range []bool{false, true} {
		name := "manual"
		if managed {
			name = "managed"
		}
		t.Run(name, func(t *testing.T) {
			root, mu, posts, digest, _ := runModelCatalog(t)
			configureDevelopmentRental(t, root)
			keyPath := filepath.Join(root, "operator.pub")
			must(t, os.WriteFile(keyPath, []byte(developmentFixtureKey+"\n"), 0600))
			startDaemonProcess(t, root)
			args := []string{"rental", "new", "cpu", "--idempotency-key", "development-default", "--json"}
			if managed {
				args = []string{"run", "proof/quantize/quantize", "steps=7",
					"model.dits=proof/source#" + digest, "model.shared=proof/source#" + digest,
					"--upload-to", "proof/output", "--rental-only", "--idempotency-key", "development-default", "--json"}
			}
			code, out := runCozy(t, root, args...)
			if managed && code != 0 || !managed && !strings.Contains(out, "proof.no_paid_create") {
				t.Fatalf("development request did not reach isolated acquisition: %d %s", code, out)
			}
			waitUntil(t, "development acquisition", func() bool { mu.Lock(); defer mu.Unlock(); return len(*posts) > 0 })
			mu.Lock()
			body := append([]byte(nil), (*posts)[0]...)
			mu.Unlock()
			request, problem := hub.ParseRentalRequestBytes(body)
			fatal(t, problem)
			if request.Development == nil || request.Development.SSHPublicKey != developmentFixtureKey {
				t.Fatalf("configured owner key did not reach %s acquisition: %+v", name, request.Development)
			}
			if managed {
				return
			}
			// Reconciliation must use the recorded acquisition even when the file goes.
			must(t, os.Remove(keyPath))
			code, out = runCozy(t, root, args...)
			if code == 0 || !strings.Contains(out, "proof.no_paid_create") {
				t.Fatalf("replay consulted the deleted public key: %d %s", code, out)
			}
			mu.Lock()
			unchanged := len(*posts) == 2 && bytes.Equal(body, (*posts)[1])
			mu.Unlock()
			if !unchanged {
				t.Fatal("replay changed development acquisition bytes")
			}
			code, out = runCozy(t, root, append(args, "--development=false")...)
			if code == 0 || !strings.Contains(out, "rental.idempotency_conflict") {
				t.Fatalf("explicit false changed pinned development access: %d %s", code, out)
			}
			mu.Lock()
			defer mu.Unlock()
			if len(*posts) != 2 {
				t.Fatal("changed development mode reached acquisition")
			}
		})
	}
}

func TestDevelopmentDefaultsRefuseMissingOrPrivateKeyBeforeAcquisition(t *testing.T) {
	for _, private := range []bool{false, true} {
		root, mu, posts, _, _ := runModelCatalog(t)
		configureDevelopmentRental(t, root)
		if private {
			must(t, os.WriteFile(filepath.Join(root, "operator.pub"), []byte("-----BEGIN OPENSSH PRIVATE KEY-----\nprivate\n"), 0600))
		}
		startDaemonProcess(t, root)
		code, out := runCozy(t, root, "rental", "new", "cpu", "--json")
		if code == 0 || !strings.Contains(out, "key") || strings.Contains(out, "BEGIN OPENSSH") {
			t.Fatalf("invalid development key was not safely refused: %d %s", code, out)
		}
		mu.Lock()
		count := len(*posts)
		mu.Unlock()
		if count != 0 {
			t.Fatal("an invalid development key reached paid acquisition")
		}
	}
}

func TestDevelopmentFalseOverridesConfiguredDefault(t *testing.T) {
	root, mu, posts, _, _ := runModelCatalog(t)
	configureDevelopmentRental(t, root)
	// The configured key deliberately does not exist: disabling development
	// must neither read it nor require SSH credentials.
	startDaemonProcess(t, root)
	args := []string{"rental", "new", "cpu", "--idempotency-key", "ordinary-override", "--json"}
	code, out := runCozy(t, root, append(args, "--development=false")...)
	if code == 0 || !strings.Contains(out, "proof.no_paid_create") {
		t.Fatalf("explicit ordinary mode did not reach isolated acquisition: %d %s", code, out)
	}
	mu.Lock()
	body := append([]byte(nil), (*posts)[0]...)
	mu.Unlock()
	request, problem := hub.ParseRentalRequestBytes(body)
	fatal(t, problem)
	if request.Development != nil {
		t.Fatal("--development=false still authorized developer access")
	}
	// An omitted flag on replay retains the original ordinary request even
	// though the configured default still enables development.
	code, out = runCozy(t, root, args...)
	if code == 0 || !strings.Contains(out, "proof.no_paid_create") {
		t.Fatalf("ordinary replay consulted the configured development key: %d %s", code, out)
	}
	mu.Lock()
	defer mu.Unlock()
	if len(*posts) != 2 || !bytes.Equal(body, (*posts)[1]) {
		t.Fatal("replay changed ordinary acquisition bytes")
	}
}

func TestRetiredDevelopmentImageOptionIsUnknown(t *testing.T) {
	root := t.TempDir()
	code, out := runCozy(t, root, "rental", "new", "cpu", "--development-image", "sha256:"+strings.Repeat("a", 64), "--json")
	if code == 0 || !strings.Contains(out, "unknown flag --development-image") {
		t.Fatalf("retired image override was accepted: %d %s", code, out)
	}
	code, out = runCozy(t, root, "rental", "new", "--help")
	if code != 0 || strings.Contains(out, "development-image") || !strings.Contains(out, "--development") {
		t.Fatalf("rental help does not expose only the shared-image development mode: %d %s", code, out)
	}
}

func TestPrivateRentalDefaultCreatesAndReusesManagedSSHIdentity(t *testing.T) {
	root, mu, posts, _, _ := runModelCatalog(t)
	startDaemonProcess(t, root)
	args := []string{"rental", "new", "cpu", "--idempotency-key", "managed-ssh-default", "--json"}
	code, out := runCozy(t, root, args...)
	if code == 0 || !strings.Contains(out, "proof.no_paid_create") {
		t.Fatalf("private default did not reach isolated acquisition: %d %s", code, out)
	}
	keyPath := filepath.Join(root, "auth", "rental-ssh")
	private, err := os.ReadFile(keyPath)
	must(t, err)
	info, err := os.Stat(keyPath)
	must(t, err)
	if info.Mode().Perm() != 0600 || strings.Contains(out, string(private)) {
		t.Fatal("managed SSH identity is not private")
	}
	public, err := os.ReadFile(keyPath + ".pub")
	must(t, err)
	mu.Lock()
	body := append([]byte(nil), (*posts)[0]...)
	mu.Unlock()
	request, problem := hub.ParseRentalRequestBytes(body)
	fatal(t, problem)
	if request.Development == nil || request.Development.SSHPublicKey != strings.TrimSpace(string(public)) {
		t.Fatal("new private rental did not capture the managed public key")
	}
	code, out = runCozy(t, root, "rental", "new", "cpu", "--idempotency-key", "managed-ssh-second", "--json")
	if code == 0 || !strings.Contains(out, "proof.no_paid_create") {
		t.Fatalf("second private acquisition failed: %d %s", code, out)
	}
	after, err := os.ReadFile(keyPath)
	must(t, err)
	if !bytes.Equal(private, after) {
		t.Fatal("a new rental replaced the maintenance identity")
	}
}

func TestImmutableRentalOverrideDoesNotCreateManagedSSHIdentity(t *testing.T) {
	root, mu, posts, _, _ := runModelCatalog(t)
	startDaemonProcess(t, root)
	code, out := runCozy(t, root, "rental", "new", "cpu", "--development=false", "--json")
	if code == 0 || !strings.Contains(out, "proof.no_paid_create") {
		t.Fatalf("immutable override did not reach isolated acquisition: %d %s", code, out)
	}
	if _, err := os.Stat(filepath.Join(root, "auth", "rental-ssh")); !os.IsNotExist(err) {
		t.Fatal("immutable rental created maintenance credentials")
	}
	mu.Lock()
	defer mu.Unlock()
	request, problem := hub.ParseRentalRequestBytes((*posts)[0])
	fatal(t, problem)
	if request.Development != nil {
		t.Fatal("immutable rental carries developer access")
	}
}
