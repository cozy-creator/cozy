package producttest

import (
	"bytes"
	"context"
	"database/sql"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/cozy-creator/cozy/internal/config"
	"github.com/cozy-creator/cozy/internal/home"
	"github.com/cozy-creator/cozy/internal/records"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
	"google.golang.org/protobuf/proto"
)

type frozenScopeMachine struct {
	*closureAdmissionMachine
	observed atomic.Value
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

// A lost local receipt must recover the machine's already accepted root even
// after editable source changes or disappears. It must never submit newer code.
func TestAcceptedFrozenRootReplaysAfterSourceChanges(t *testing.T) {
	for _, removed := range []bool{false, true} {
		t.Run(map[bool]string{false: "changed", true: "removed"}[removed], func(t *testing.T) {
			root := t.TempDir()
			project := parityProject(t)
			if code, out := runCozy(t, root, "package", "install", project, "--editable"); code != 0 {
				t.Fatalf("install: %s", out)
			}
			if code, out := runCozy(t, root, "run", parityPackage+"/add", "value=41", "--await", "--json", "--idempotency-key", "frozen-source"); code != 0 || !strings.Contains(out, `"value":42`) {
				t.Fatalf("initial execution: %s", out)
			}
			layout, problem := home.Open(root)
			fatal(t, problem)
			store, problem := records.Open(layout.DB)
			fatal(t, problem)
			defer store.Close()
			request, problem := store.RequestByIdempotencyKey("frozen-source")
			fatal(t, problem)
			before, problem := store.MachineExecution(request.ID)
			fatal(t, problem)
			if len(before.Receipt) == 0 {
				t.Fatal("initial execution was not accepted")
			}
			if code, out := runCozy(t, root, "down"); code != 0 {
				t.Fatalf("stop observer: %s", out)
			}
			if removed {
				must(t, os.RemoveAll(project))
			} else {
				path := filepath.Join(project, "machine_parity.py")
				raw, err := os.ReadFile(path)
				must(t, err)
				must(t, os.WriteFile(path, []byte(strings.ReplaceAll(string(raw), "payload.value + 1", "payload.value + 999")), 0600))
			}
			// Fault injection: the machine owns the accepted receipt, while the client's
			// last durable evidence is the exact frozen offer (ambiguous response loss).
			db, err := sql.Open("sqlite", layout.DB)
			must(t, err)
			_, err = db.Exec(`UPDATE machine_executions SET receipt=x'',observed_state=x'',outcome=x'',remote_cursor=0,collected=0 WHERE request_id=?`, request.ID)
			must(t, err)
			_, err = db.Exec(`UPDATE requests SET state='queued' WHERE id=?`, request.ID)
			must(t, err)
			db.Close()
			startDaemonProcess(t, root)
			waitFor(t, root, "original accepted root recovered", func() bool {
				link, _ := store.MachineExecution(request.ID)
				return link != nil && len(link.Receipt) > 0
			})
			after, problem := store.MachineExecution(request.ID)
			fatal(t, problem)
			if !bytes.Equal(before.Submission, after.Submission) || !bytes.Equal(before.Receipt, after.Receipt) {
				t.Fatal("changed source replaced frozen authority or accepted execution")
			}
			if code, out := runCozy(t, root, "run", "show", request.ID, "--json"); code != 0 || strings.Contains(out, `"value":1040`) {
				t.Fatalf("replay executed changed source: %s", out)
			}
		})
	}
}
