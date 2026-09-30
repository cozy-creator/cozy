package producttest

import (
	"flag"
	"net/http"
	"net/http/httptest"
	"os"
	"os/exec"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/cozy-creator/cozy/internal/api"
	localclient "github.com/cozy-creator/cozy/internal/client"
	"github.com/cozy-creator/cozy/internal/config"
	"github.com/cozy-creator/cozy/internal/daemon"
	"github.com/cozy-creator/cozy/internal/home"
	"github.com/cozy-creator/cozy/internal/records"
)

var previousDaemonBinary = flag.String("previous-daemon-binary", "", "official Cozy 0.1.21 binary for mixed client/daemon qualification")

func TestModelOverrideClientRequiresOnlyItsOperationCapability(t *testing.T) {
	for _, response := range []string{`{}`, `{"model_overrides":false}`, `{"model_overrides":true,"future_hint":"ignored"}`} {
		t.Run(response, func(t *testing.T) {
			root := t.TempDir()
			layout, problem := home.Open(root)
			fatal(t, problem)
			must(t, os.WriteFile(layout.Daemon, nil, 0600))
			_, problem = api.Mint(layout)
			fatal(t, problem)
			var reads, submitted atomic.Int32
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				switch r.URL.Path {
				case "/v1/capabilities":
					reads.Add(1)
					_, _ = w.Write([]byte(response))
				case "/v1/requests":
					submitted.Add(1)
					_, _ = w.Write([]byte(`{"request_id":"accepted","status":"queued"}`))
				default:
					t.Errorf("unexpected request %s", r.URL.Path)
				}
			}))
			defer server.Close()
			client, problem := localclient.Open(config.Config{Home: root}, daemon.State{Addr: strings.TrimPrefix(server.URL, "http://")})
			fatal(t, problem)
			_, problem = client.Submit(api.Submission{Package: "proof/parent", Function: "compose", Models: []records.ModelRef{{
				Choice: true, Package: "proof/parent", Slot: "child.models.model", Model: "proof/base",
			}}}, "child")
			supported := strings.Contains(response, `"model_overrides":true`)
			if supported {
				fatal(t, problem)
			} else if problem == nil || problem.ErrName() != "daemon.model_overrides_unavailable" || submitted.Load() != 0 {
				t.Fatalf("absent capability submitted the child override: %v", problem)
			}
			_, problem = client.Submit(api.Submission{Package: "proof/parent", Function: "compose", Models: []records.ModelRef{{
				Choice: true, Package: "proof/parent", Slot: "compose.models.model", Model: "proof/base",
			}}}, "base")
			fatal(t, problem)
			want := int32(1)
			if supported {
				want++
			}
			if reads.Load() != 1 || submitted.Load() != want {
				t.Fatalf("ordinary base override required new capability: %d reads, %d submissions", reads.Load(), submitted.Load())
			}
		})
	}
}

func TestNewModelOverrideClientPreservesAnOfficialOldDaemon(t *testing.T) {
	if *previousDaemonBinary == "" {
		t.Skip("requires -previous-daemon-binary pointing to the official 0.1.21 artifact")
	}
	version, err := exec.Command(*previousDaemonBinary, "-v").CombinedOutput()
	must(t, err)
	if strings.TrimSpace(string(version)) != "0.1.21" {
		t.Fatalf("expected official 0.1.21 daemon fixture, got %s", version)
	}
	h := newLadderHub(t)
	h.bind(goodLadder())
	machine := newReleaseMachine()
	machine.changed = make(chan struct{})
	root, layout := rentedLadderHome(t, h, &fakePod{machine: machine}, nil)
	owner := startDaemonBinary(t, *previousDaemonBinary, root)
	store, problem := records.Open(layout.DB)
	fatal(t, problem)
	defer store.Close()
	base := []string{"run", ladderPackage + "/long_form", "model.source=" + sourceJobModel,
		"--source-profile", "source=fixture/diffusers/1", "--rental=tessa", "--json"}
	code, out := runCozy(t, root, append(base, "--idempotency-key", "old-owner-active")...)
	if code != 0 {
		t.Fatalf("ordinary job could not use the old daemon [exit %d]: %s", code, out)
	}
	var active *records.Request
	waitFor(t, root, "old daemon's ordinary job acceptance", func() bool {
		active, _ = store.RequestByIdempotencyKey("old-owner-active")
		return active != nil && machine.accepted(active.ID)
	})
	machine.begin(active.ID)
	for _, args := range [][]string{
		{"--lora", "generate.models.model:fl2va_dit=proof/style@1.0.0,0.5"},
		{"model.generate.models.model=proof/base@1.0.0/bf16"},
	} {
		argv := append(append([]string(nil), base...), args...)
		code, out := runCozy(t, root, argv...)
		if code == 0 || !strings.Contains(out, "daemon.model_overrides_unavailable") || !strings.Contains(out, "restart the daemon") {
			t.Fatalf("new selection did not ask for an operation-specific daemon update [exit %d]: %s", code, out)
		}
	}
	if machine.submits.Load() != 1 {
		t.Fatalf("a downgraded request reached the machine: %d submissions", machine.submits.Load())
	}
	if state := daemon.Probe(config.Config{Home: root}); !state.Up || state.PID != owner.cmd.Process.Pid {
		t.Fatal("operation preflight replaced the old daemon")
	}
	retained, problem := store.RequestRow(active.ID)
	fatal(t, problem)
	if records.Settled(retained.State) || !machine.accepted(active.ID) {
		t.Fatal("operation refusal changed the ordinary job already in flight")
	}
}
