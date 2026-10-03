package producttest

import (
	"bytes"
	"context"
	"crypto/hmac"
	"crypto/sha256"
	"database/sql"
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
	"encoding/pem"
	"net"
	"net/http"
	"net/http/httptest"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/cozy-creator/cozy/internal/config"
	"github.com/cozy-creator/cozy/internal/home"
	"github.com/cozy-creator/cozy/internal/machines"
	"github.com/cozy-creator/cozy/internal/records"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
	"google.golang.org/protobuf/proto"
)

type frozenScopeMachine struct {
	*closureAdmissionMachine
	observed atomic.Value
}

// The ordinary Creator daemon and authenticated machine transport reconcile a kept
// receipt without source files or a fresh delegated account credential. The
// machine is a deterministic journal peer; no Runtime process is needed here.
func TestFrozenLocalReceiptDoesNotNeedSourceOrCurrentHubGrant(t *testing.T) {
	for _, rooted := range []bool{false, true} {
		for _, grant := range []string{"absent", "expired"} {
			t.Run(map[bool]string{false: "capture", true: "root"}[rooted]+"/"+grant, func(t *testing.T) {
				root := t.TempDir()
				layout, problem := home.Open(root)
				fatal(t, problem)
				var accountCalls atomic.Int32
				account := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
					accountCalls.Add(1)
					http.Error(w, "current login is unavailable", http.StatusUnauthorized)
				}))
				defer account.Close()
				settings := "tensorhub_url: " + account.URL + "\ndaemon:\n  idle_shutdown_s: 0\n"
				if grant == "expired" {
					settings += "tensorhub_token: expired-current-login\n"
				}
				must(t, os.WriteFile(filepath.Join(root, config.FileName), []byte(settings), 0600))
				must(t, os.MkdirAll(layout.Machine, 0700))
				host := machines.NewHost(layout.Machine, "", nil)
				owner, problem := host.Owner()
				fatal(t, problem)
				public, err := base64.RawURLEncoding.DecodeString(owner.PublicKey())
				must(t, err)
				machine := &frozenScopeMachine{closureAdmissionMachine: newClosureAdmissionMachine("supported")}
				pod := &fakePod{machine: machine, controlKey: public}
				key := make([]byte, 32)
				ready := make(chan struct{})
				var workerPort int
				var leaf []byte
				pod.mediaRequest = func(w http.ResponseWriter, r *http.Request) bool {
					if r.URL.Path != "/v1/bootstrap/receipt" {
						return false
					}
					<-ready
					payload, _ := json.Marshal(map[string]any{"pod_boot_id": podBootID, "machine_version": "fixture",
						"machine_capabilities": []string{machines.HubAccessCapability}, "worker_internal_port": workerPort,
						"tls_certificate_der_base64": base64.StdEncoding.EncodeToString(leaf)})
					mac := hmac.New(sha256.New, key)
					_, _ = mac.Write([]byte(machines.ReadinessReceiptDomain))
					_, _ = mac.Write(payload)
					_ = json.NewEncoder(w).Encode(map[string]any{"payload": payload, "hmac_sha256": hex.EncodeToString(mac.Sum(nil))})
					return true
				}
				connection, certPath := startFakePod(t, root, pod)
				certificate, err := os.ReadFile(certPath)
				must(t, err)
				block, _ := pem.Decode(certificate)
				if block == nil {
					t.Fatal("fixture machine certificate is absent")
				}
				leaf = block.Bytes
				_, port, err := net.SplitHostPort(connection.Addr)
				must(t, err)
				workerPort, err = strconv.Atoi(port)
				must(t, err)
				_, port, err = net.SplitHostPort(connection.Media.Addr)
				must(t, err)
				mediaPort, err := strconv.Atoi(port)
				must(t, err)
				close(ready)
				binary := filepath.Join(host.Root(), "usr/local/bin/cozy-machine")
				must(t, os.MkdirAll(filepath.Dir(binary), 0755))
				sleep, err := exec.LookPath("sleep")
				must(t, err)
				body, err := os.ReadFile(sleep)
				must(t, err)
				writeInstallFile(t, binary, body, 0755)
				process := exec.Command(binary, "3600")
				must(t, process.Start())
				t.Cleanup(func() { _ = process.Process.Kill(); _ = process.Wait() })
				record, err := json.Marshal(map[string]any{"pid": process.Process.Pid, "worker_id": podWorkerID,
					"worker_port": workerPort, "media_port": mediaPort, "receipt_key": base64.RawURLEncoding.EncodeToString(key)})
				must(t, err)
				writeInstallFile(t, filepath.Join(layout.Machine, "agent.json"), record, 0600)
				writeInstallFile(t, filepath.Join(layout.Machine, "installed.json"), []byte(`{"host":{"name":"cozy-machine"},"host_pinned":true}`), 0600)
				if grant == "expired" {
					cached, err := json.Marshal(map[string]any{account.URL: map[string]any{"origin": account.URL,
						"token": "expired-kept-grant", "expires_at": time.Now().Add(-time.Hour).Unix()}})
					must(t, err)
					writeInstallFile(t, filepath.Join(layout.Machine, "execution-access.json"), cached, 0600)
				}
				store, problem := records.Open(layout.DB)
				fatal(t, problem)
				defer store.Close()
				request, frozen := frozenSubmissionRecord(t, store, false, func(q *pb.MachineExecutionSubmit) {
					q.Hub = account.URL
					if rooted {
						q.ReleaseRoot = &pb.ReleaseRoot{Package: "local/example", InstallationId: "frozen-install", Entrypoint: "main", Job: true, Hub: account.URL}
						q.Hub = ""
						q.CaptureCanonicalBytes, q.CaptureDigest = nil, nil
						q.Offer.InvocationSpecCanonicalBytes, q.Offer.InvocationSpecDigest = nil, nil
					}
				})
				db, err := sql.Open("sqlite", layout.DB)
				must(t, err)
				_, err = db.Exec(`UPDATE machine_executions SET machine_id=? WHERE request_id=?`, machines.Local, request.ID)
				must(t, err)
				must(t, db.Close())
				var accepted pb.MachineExecutionSubmit
				must(t, proto.Unmarshal(frozen, &accepted))
				accepted.Claim = &pb.Claim{WorkerId: podWorkerID, WorkerBootId: podBootID}
				_, err = machine.terminalMachines.SubmitMachineExecution(t.Context(), &accepted)
				must(t, err) // the machine committed; the controller never got its receipt
				startDaemonProcess(t, root)
				waitFor(t, root, "local durable receipt without current source or grant", func() bool {
					link, _ := store.MachineExecution(request.ID)
					return link != nil && len(link.Receipt) > 0
				})
				link, problem := store.MachineExecution(request.ID)
				fatal(t, problem)
				if !bytes.Equal(link.Submission, frozen) || len(machine.submitted()) != 1 || accountCalls.Load() != 0 {
					t.Fatal("receipt recovery changed frozen scope, executed twice, or asked for a fresh account grant")
				}
			})
		}
	}
}

func (m *frozenScopeMachine) SubmitMachineExecution(ctx context.Context, q *pb.MachineExecutionSubmit) (*pb.MachineExecutionReceipt, error) {
	m.observed.Store(proto.Clone(q).(*pb.MachineExecutionSubmit))
	return m.closureAdmissionMachine.SubmitMachineExecution(ctx, q)
}

// Frozen authority survives removal of all local source/install metadata and a
// changed default account. The empty scope remains empty rather than borrowing it.
func TestFrozenSubmissionUsesStoredScopeWithoutItsLocalInstallation(t *testing.T) {
	for _, arm := range []struct {
		name, origin string
		rooted       bool
	}{{"offline", "", false}, {"scoped", "https://frozen.invalid", false}, {"root_offline", "", true}, {"root_scoped", "https://frozen.invalid", true}} {
		t.Run(arm.name, func(t *testing.T) {
			origin := arm.origin
			h := newLadderHub(t)
			h.bind(goodLadder())
			machine := &frozenScopeMachine{closureAdmissionMachine: newClosureAdmissionMachine("supported")}
			var request records.Request
			var frozen []byte
			root, layout := rentedLadderHome(t, h, &fakePod{machine: machine}, func(_ home.Layout, store *records.Store) {
				request, frozen = frozenSubmissionRecord(t, store, false, func(q *pb.MachineExecutionSubmit) {
					q.Hub = origin
					if arm.rooted {
						q.ReleaseRoot = &pb.ReleaseRoot{Package: "local/example", InstallationId: "frozen-install", Entrypoint: "main", Job: true, Hub: origin}
						q.Hub = ""
						q.CaptureCanonicalBytes, q.CaptureDigest = nil, nil
						q.Offer.InvocationSpecCanonicalBytes, q.Offer.InvocationSpecDigest = nil, nil
					}
				})
			})
			var borrowed atomic.Int32
			changed := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
				borrowed.Add(1)
				http.Error(w, "wrong account", http.StatusForbidden)
			}))
			defer changed.Close()
			cfg := filepath.Join(root, config.FileName)
			raw, err := os.ReadFile(cfg)
			must(t, err)
			must(t, os.WriteFile(cfg, []byte(strings.ReplaceAll(string(raw), h.server.URL, changed.URL)), 0600))
			// The frozen record deliberately has no local installation/source to resolve.
			startDaemonProcess(t, root)
			store, problem := records.Open(layout.DB)
			fatal(t, problem)
			defer store.Close()
			waitFor(t, root, "frozen replay accepted without source", func() bool {
				link, _ := store.MachineExecution(request.ID)
				return link != nil && len(link.Receipt) > 0
			})
			got, ok := machine.observed.Load().(*pb.MachineExecutionSubmit)
			gotOrigin := ""
			if ok {
				gotOrigin = got.Hub
				if got.ReleaseRoot != nil {
					gotOrigin = got.ReleaseRoot.Hub
				}
			}
			if !ok || gotOrigin != origin || (got.ReleaseRoot != nil) != arm.rooted {
				t.Fatalf("replay borrowed current scope: %+v", got)
			}
			link, problem := store.MachineExecution(request.ID)
			fatal(t, problem)
			if !bytes.Equal(link.Submission, frozen) {
				t.Fatal("replay rewrote immutable submission")
			}
			if borrowed.Load() != 0 {
				t.Fatal("replay contacted the changed default account")
			}
		})
	}
}

func TestInvalidFrozenSubmissionRefusesBeforeConnecting(t *testing.T) {
	for _, raw := range [][]byte{{0xff}, {}} {
		t.Run(map[bool]string{true: "malformed", false: "missing_offer"}[len(raw) > 0], func(t *testing.T) {
			h := newLadderHub(t)
			h.bind(goodLadder())
			machine := newClosureAdmissionMachine("supported")
			root, layout := rentedLadderHome(t, h, &fakePod{machine: machine}, func(layout home.Layout, store *records.Store) {
				request, _ := frozenSubmissionRecord(t, store, false)
				if len(raw) == 0 {
					var err error
					raw, err = proto.Marshal(&pb.MachineExecutionSubmit{SubmissionId: "invalid", ExpectedExecutionWorkspaceId: "rented-workspace"})
					must(t, err)
				}
				db, err := sql.Open("sqlite", layout.DB)
				must(t, err)
				defer db.Close()
				_, err = db.Exec(`UPDATE machine_executions SET submission=? WHERE request_id=?`, raw, request.ID)
				must(t, err)
			})
			startDaemonProcess(t, root)
			waitFor(t, root, "invalid frozen body refused", func() bool {
				_, out := runCozy(t, root, "run", "show", "frozen-request", "--json")
				return strings.Contains(out, "recorded machine submission")
			})
			if machine.sent.Load() != 0 || machine.probes.Load() != 0 {
				t.Fatal("invalid frozen body reached machine")
			}
			store, problem := records.Open(layout.DB)
			fatal(t, problem)
			defer store.Close()
			link, problem := store.MachineExecution("frozen-request")
			fatal(t, problem)
			if !bytes.Equal(link.Submission, raw) || len(link.Receipt) > 0 || link.SubmissionClosed {
				t.Fatal("invalid frozen record was rewritten or treated as nonaccepted")
			}
		})
	}
}
