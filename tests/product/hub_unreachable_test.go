package producttest

import (
	"crypto/rand"
	"crypto/sha256"
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/cozy-creator/cozy/internal/config"
	"github.com/cozy-creator/cozy/internal/records"
)

// hubHome is a Creator home whose configured Tensorhub is hubURL, logged in by
// machine key and holding no operator token: every hub call goes through the
// device-key login first, the path `cozy rental list` failed on.
func hubHome(t *testing.T, name, hubURL string) string {
	t.Helper()
	base := scratchBase
	must(t, os.MkdirAll(base, 0o755))
	root, err := os.MkdirTemp(base, name+"-")
	must(t, err)
	t.Cleanup(func() { _ = os.RemoveAll(root) })
	must(t, os.WriteFile(filepath.Join(root, config.FileName), []byte(
		"tensorhub_url: "+hubURL+"\n"+
			""+
			"daemon:\n  idle_shutdown_s: 0\n"), 0o600))
	seed := make([]byte, 32)
	_, err = rand.Read(seed)
	must(t, err)
	credential, err := json.Marshal(map[string]any{"version": 1, "hub": hubURL, "email": "paul@example.com",
		"device_key_id": "dk-test", "private_key": base64.RawURLEncoding.EncodeToString(seed)})
	must(t, err)
	sum := sha256.Sum256([]byte(hubURL))
	must(t, os.MkdirAll(filepath.Join(root, "auth"), 0o700))
	must(t, os.WriteFile(filepath.Join(root, "auth", hex.EncodeToString(sum[:])+".json"), credential, 0o600))
	return root
}

func closedHub(t *testing.T) string {
	t.Helper()
	return fmt.Sprintf("http://127.0.0.1:%d", reservePort(t))
}

func errorOf(t *testing.T, out string) (code, message string) {
	t.Helper()
	var doc struct {
		Error struct {
			Code    string `json:"code"`
			Message string `json:"message"`
		} `json:"error"`
	}
	if err := json.Unmarshal([]byte(out), &doc); err != nil {
		t.Fatalf("not one JSON document: %v\n%s", err, out)
	}
	return doc.Error.Code, doc.Error.Message
}

// A hub with nothing listening is unreachable, not an authentication fault.
func TestUnreachableHubIsNotReportedAsAuthentication(t *testing.T) {
	hubURL := closedHub(t)
	root := hubHome(t, "hub-unreachable-auth", hubURL)

	code, out := runCozy(t, root, "--json", "auth")
	if code == 0 {
		t.Fatalf("`cozy auth` against a closed port exited 0\n%s", out)
	}
	name, message := errorOf(t, out)
	want := "Tensorhub unreachable at " + hubURL + " (connection refused)"
	if name != "hub.unreachable" || message != want {
		t.Fatalf("got %s %q, want hub.unreachable %q", name, message, want)
	}
	if strings.Contains(strings.ToLower(out), "authentication") {
		t.Fatalf("an unreachable hub still reads as an auth problem\n%s", out)
	}
}

// A hub that answers and refuses the machine key is still an auth failure.
func TestHubRefusalStillReportsAuthentication(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusUnauthorized)
		_, _ = w.Write([]byte(`{"error":{"code":"auth.device_key_revoked","message":"this machine key was revoked"}}`))
	}))
	defer server.Close()
	root := hubHome(t, "hub-refuses-auth", server.URL)

	code, out := runCozy(t, root, "--json", "auth")
	if code == 0 {
		t.Fatalf("a refused machine key exited 0\n%s", out)
	}
	if name, _ := errorOf(t, out); name != "auth.device_key_revoked" {
		t.Fatalf("a genuine auth refusal was not reported as one\n%s", out)
	}
}

// With the hub down, `cozy rental list` shows what this host recorded, says it
// may be stale, never reports totals it could not reconcile, and exits non-zero.
func TestRentalListFallsBackToLocalRecordsWhenHubUnreachable(t *testing.T) {
	hubURL := closedHub(t)
	root := hubHome(t, "hub-unreachable-rentals", hubURL)
	store, problem := records.Open(filepath.Join(root, "creator.sqlite"))
	fatal(t, problem)
	fatal(t, store.RecordRental(records.Rental{AcceleratorCount: 1,
		ID: "pr-unreachable0000000001", MachineName: "leonmitchelli", SKU: "h100",
		AcceleratorModel: "NVIDIA H100", HourlyRateUSDMicros: 2_990_000, State: "ready",
		Hub: hubURL, Address: "127.0.0.1:1", CertPath: filepath.Join(root, "pr-unreachable0000000001.pem"),
	}))
	store.Close()

	code, out := runCozy(t, root, "rental", "list", "--no-watch")
	if code == 0 {
		t.Fatalf("a board the hub could not confirm exited 0\n%s", out)
	}
	for _, want := range []string{
		"Tensorhub unreachable at " + hubURL + " (connection refused)",
		"last known rentals, which may be out of date",
		"leonmitchelli",
		"ready (unverified)",
	} {
		if !strings.Contains(out, want) {
			t.Fatalf("the degraded board lacks %q\n%s", want, out)
		}
	}
	if strings.Contains(out, "Remote machines running") || strings.Contains(out, "Current spend per hour") ||
		strings.Contains(strings.ToLower(out), "authentication") {
		t.Fatalf("the degraded board presents unreconciled totals or blames auth\n%s", out)
	}

	// The board names the Hub that did not answer even with an explicit source override.
	type hubProblem struct {
		Hub     string `json:"hub"`
		Code    string `json:"code"`
		Message string `json:"message"`
	}
	for _, withOverride := range []bool{false, true} {
		args := []string{"--json", "rental", "list"}
		if withOverride {
			args = append(args, "--tensorhub="+hubURL)
		}
		code, out = runCozy(t, root, args...)
		if code == 0 {
			t.Fatalf("the JSON board exited 0 with the hub unreachable\n%s", out)
		}
		var document struct {
			Rentals    []map[string]any `json:"rentals"`
			Live       *bool            `json:"live"`
			Running    *int             `json:"machines_running"`
			Unreadable []hubProblem     `json:"unreadable_hubs"`
		}
		must(t, json.Unmarshal([]byte(out), &document))
		reason := ""
		if len(document.Unreadable) == 1 && document.Unreadable[0].Hub == hubURL {
			reason = document.Unreadable[0].Code
		}
		if document.Live == nil || *document.Live || document.Running != nil || reason != "hub.unreachable" ||
			len(document.Rentals) != 1 || document.Rentals[0]["machine"] != "leonmitchelli" {
			t.Fatalf("the JSON board (source override %v) does not mark local records as unverified\n%s", withOverride, out)
		}
	}
}
