package producttest

import (
	"crypto/tls"
	"encoding/json"
	"encoding/pem"
	"net"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/cozy-creator/cozy/internal/config"
	"github.com/cozy-creator/cozy/internal/home"
	"github.com/cozy-creator/cozy/internal/records"
	"github.com/cozy-creator/cozy/internal/rental"
	v1 "github.com/cozy-creator/cozy/protocol/cozy/machine/v1"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials"
)

// idleClockMachine is a rental's machine as far as its idle clock goes: Status names the
// deadline it holds (0: none, as TensorD 0.5.2–0.5.5 after a job), and a keepalive restarts
// a held clock.
type idleClockMachine struct {
	v1.UnimplementedMachineServer
	deadline atomic.Int64
}

func (m *idleClockMachine) Status(request *v1.StatusRequest, stream grpc.ServerStreamingServer[v1.StatusFrame]) error {
	if request.Keepalive && m.deadline.Load() != 0 {
		m.deadline.Store(time.Now().Add(15 * time.Minute).UnixMilli())
	}
	return stream.Send(&v1.StatusFrame{WorkerId: "idle-worker", BootId: "idle-boot", Phase: "ready", IdleDeadlineUnixMs: m.deadline.Load()})
}

// The real CLI and daemon say the owner's rule, and `rental list`, `rental show` and
// `rental keepalive` show the deadline the rental's machine names, never one of their own.
func TestRentalListAndKeepaliveShowTheMachinesIdleDeadline(t *testing.T) {
	h := newFakeRentalHub(t, 0)
	root := t.TempDir()
	must(t, os.WriteFile(filepath.Join(root, config.FileName), []byte("tensorhub_url: "+h.server.URL+"\ntensorhub_token: rental-idle-test\n"), 0o600))
	layout, problem := home.Open(root)
	fatal(t, problem)
	store, problem := records.Open(layout.DB)
	fatal(t, problem)
	defer store.Close()
	must(t, os.MkdirAll(layout.Rentals, 0o700))
	owner, problem := rental.OwnerIdentityAt(layout.RentalCreatorIdentity("pr-idle"))
	fatal(t, problem)
	certServer := httptest.NewTLSServer(http.NotFoundHandler())
	cert := certServer.TLS.Certificates[0]
	certServer.Close()
	listener, err := net.Listen("tcp", "127.0.0.1:0") //cozy:allow pinned loopback machine fixture for the real daemon's Status reads
	must(t, err)
	machine := &idleClockMachine{}
	server := grpc.NewServer(grpc.Creds(credentials.NewTLS(&tls.Config{Certificates: []tls.Certificate{cert}})))
	v1.RegisterMachineServer(server, machine)
	go server.Serve(listener)
	defer server.Stop()
	certPEM := pem.EncodeToMemory(&pem.Block{Type: "CERTIFICATE", Bytes: cert.Certificate[0]})
	must(t, os.WriteFile(layout.RentalCert("pr-idle"), certPEM, 0o600))
	fatal(t, store.RecordRental(records.Rental{ID: "pr-idle", MachineName: "kirin", SKU: "cpu", State: "ready", Hub: h.server.URL,
		Address: listener.Addr().String(), CertPath: layout.RentalCert("pr-idle"), ExpectedWorkerID: "idle-worker", ExpectedWorkerBootID: "idle-boot",
		AcceleratorModel: "CPU", AcceleratorCount: 1, HourlyRateUSDMicros: 1, ReadyAt: time.Now().UTC().Format(time.RFC3339Nano)}))
	h.mu.Lock()
	h.rentals["pr-idle"] = map[string]any{"rental_id": "pr-idle", "name": "kirin", "state": "ready", "worker_address": listener.Addr().String(),
		"cert_pem": string(certPEM), "worker_id": "idle-worker", "worker_boot_id": "idle-boot", "creator_public_key": owner.PublicKey(),
		"hourly_rate_usd_micros": int64(1), "accelerator_count": 1, "requested_accelerator_model": "CPU"}
	h.mu.Unlock()
	startDaemonProcess(t, root)

	stamp := func(ms int64) string { return time.UnixMilli(ms).UTC().Format(time.RFC3339Nano) }
	listed := func() map[string]any {
		t.Helper()
		code, out := runCozy(t, root, "rental", "list", "--json", "--full")
		var document struct {
			Rentals []map[string]any `json:"rentals"`
		}
		if code != 0 || json.Unmarshal([]byte(out), &document) != nil || len(document.Rentals) != 1 {
			t.Fatalf("rental list [exit %d]: %s\n%s", code, out, tail(filepath.Join(root, "daemon.log")))
		}
		return document.Rentals[0]
	}
	keepalive := func(args ...string) map[string]any {
		t.Helper()
		code, out := runCozy(t, root, append([]string{"rental", "keepalive", "kirin", "--json"}, args...)...)
		var document map[string]any
		if code != 0 || json.Unmarshal([]byte(lastJSONLine(out)), &document) != nil {
			t.Fatalf("rental keepalive [exit %d]: %s", code, out)
		}
		return document
	}

	// The owner's rule, in the listing's trail and the keepalive help.
	if code, out := runCozy(t, root, "help", "rental", "keepalive"); code != 0 || !strings.Contains(out, "Reset a rental's 15-minute idle clock once.") ||
		strings.Contains(out, "unused") {
		t.Fatalf("keepalive help [exit %d]: %s", code, out)
	}

	// The listing shows the deadline the machine names now: the typed moment and, to a
	// person, how long until then.
	machine.deadline.Store(time.Now().Add(7 * time.Minute).UnixMilli())
	if row := listed(); row["release_due_at"] != stamp(machine.deadline.Load()) {
		t.Fatalf("listing lost the machine's deadline %s: %v", stamp(machine.deadline.Load()), row)
	}
	code, out := runCozy(t, root, "rental", "list")
	if code != 0 || !regexp.MustCompile(`QUEUED\s+ENDS`).MatchString(out) || strings.Contains(out, "IDLE") || !regexp.MustCompile(`kirin\s.*\sin [67]m(\d+s)?\s`).MatchString(out) ||
		!strings.Contains(out, "Rentals end themselves after 15 minutes idle (no queued or running job).") || strings.Contains(out, "nused") {
		t.Fatalf("the board does not show the machine's deadline or the owner's rule [exit %d]:\n%s", code, out)
	}
	code, out = runCozy(t, root, "rental", "show", "kirin", "--json")
	if code != 0 || strings.Count(out, `"release_due_at"`) != 1 || !strings.Contains(out, `"release_due_at":"`+stamp(machine.deadline.Load())+`"`) {
		t.Fatalf("rental show did not say the machine's deadline once [exit %d]: %s", code, out)
	}

	// A keepalive restarts the machine's clock and says the deadline it answered.
	before := machine.deadline.Load()
	acknowledged := keepalive()
	if after := machine.deadline.Load(); after <= before || acknowledged["release_due_at"] != stamp(after) {
		t.Fatalf("keepalive did not say the machine's new deadline %s: %v", stamp(after), acknowledged)
	}
	if row := listed(); row["release_due_at"] != acknowledged["release_due_at"] {
		t.Fatalf("listing after keepalive: %v", row)
	}
	if code, out := runCozy(t, root, "rental", "keepalive", "kirin"); code != 0 || !regexp.MustCompile(`ends:\s+in 1[45]m`).MatchString(out) {
		t.Fatalf("keepalive does not say when the rental ends [exit %d]:\n%s", code, out)
	}

	// A machine that names no deadline gets none, from either command.
	machine.deadline.Store(0)
	if row := listed(); row["release_due_at"] != nil {
		t.Fatalf("invented a deadline: %v", row)
	}
	if acknowledged := keepalive(); acknowledged["release_due_at"] != nil || acknowledged["rental"] != "pr-idle" {
		t.Fatalf("keepalive invented a deadline: %v", acknowledged)
	}
	if code, out := runCozy(t, root, "rental", "keepalive", "kirin"); code != 0 || strings.Contains(out, "ends") {
		t.Fatalf("human keepalive without a deadline [exit %d]:\n%s", code, out)
	}
}
