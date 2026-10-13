package producttest

import (
	"bytes"
	"context"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/alecthomas/kong"
	"github.com/cozy-creator/cozy/internal/api"
	"github.com/cozy-creator/cozy/internal/cli"
	localapi "github.com/cozy-creator/cozy/internal/client"
	"github.com/cozy-creator/cozy/internal/config"
	"github.com/cozy-creator/cozy/internal/daemon"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/machines"
	"github.com/cozy-creator/cozy/internal/orchestrator"
	"github.com/cozy-creator/cozy/internal/output"
	"github.com/cozy-creator/cozy/internal/records"
)

func TestQueuedInstallCancellationCompetesWithDispatch(t *testing.T) {
	_, store := rentalInstallStore(t)
	prior, problem := store.BeginRentalInstall(machines.Local, records.RentalInstallSelection{Package: "proof/interrupted"})
	rentalInstallCheck(t, problem)
	_, problem = store.StartRentalInstall(prior.ID, "")
	rentalInstallCheck(t, problem)
	if problem := store.SettleRentalInstall(prior.ID, "queued", nil, nil); problem == nil {
		t.Fatal("retry lost its evidence of prior dispatch")
	}
	rentalInstallCheck(t, store.SettleRentalInstall(prior.ID, "queued", nil, exit.Unavailablef("controller disconnected; native work may continue")))
	if _, problem := store.CancelQueuedRentalInstall(machines.Local, prior.ID); problem == nil || problem.ErrName() != "install.not_queued" {
		t.Fatalf("previously dispatched work was cancelable after requeue: %v", problem)
	}
	for n := 0; n < 12; n++ {
		row, problem := store.BeginRentalInstall(machines.Local, records.RentalInstallSelection{Package: "proof/queued", Release: fmt.Sprint(n)})
		rentalInstallCheck(t, problem)
		if _, problem := store.CancelQueuedRentalInstall("another-machine", row.ID); problem == nil {
			t.Fatal("another machine canceled this installation")
		}
		start := make(chan struct{})
		canceled, claimed := make(chan *exit.Error, 1), make(chan *exit.Error, 1)
		go func() {
			<-start
			_, problem := store.CancelQueuedRentalInstall(machines.Local, row.ID)
			canceled <- problem
		}()
		go func() { <-start; _, problem := store.StartRentalInstall(row.ID, ""); claimed <- problem }()
		close(start)
		cancelProblem, claimProblem := <-canceled, <-claimed
		if (cancelProblem == nil) == (claimProblem == nil) {
			t.Fatalf("exactly one of cancel and dispatch must win: cancel=%v dispatch=%v", cancelProblem, claimProblem)
		}
		got, problem := store.RentalInstall(row.ID)
		rentalInstallCheck(t, problem)
		if cancelProblem == nil {
			if got.State != "failed" || got.ErrorCode != "install.canceled" {
				t.Fatalf("cancellation did not persist its explicit terminal reason: %+v", got)
			}
			_, problem := store.CancelQueuedRentalInstall(machines.Local, row.ID)
			rentalInstallCheck(t, problem)
		} else if got.State != "installing" || cancelProblem.ErrName() != "install.not_queued" {
			t.Fatalf("dispatch winner was changed: %+v %v", got, cancelProblem)
		}
	}
}

func TestMachineInstallControlsUseExistingQueueAPI(t *testing.T) {
	layout, store := rentalInstallStore(t)
	q := machines.NewInstalls(store, prepared(func(context.Context, records.RentalInstall, func(machines.InstallProgress)) *exit.Error {
		t.Error("observation/cancellation dispatched new work")
		return nil
	}), io.Discard)
	row, problem := q.Accept(machines.Local, records.RentalInstallSelection{Destination: "paul/checkpoint", Models: []records.ModelRef{{Source: "hf://org/model@commit/weights.safetensors"}}})
	rentalInstallCheck(t, problem)
	owner, problem := orchestrator.Open(orchestrator.Options{Store: store, Layout: layout})
	rentalInstallCheck(t, problem)
	defer owner.Close(time.Second)
	peer := httptest.NewUnstartedServer(nil)
	defer peer.Close()
	address := peer.Listener.Addr().String()
	held, problem := daemon.Hold(layout, address, "")
	rentalInstallCheck(t, problem)
	defer held.Release()
	credentials, problem := api.Mint(layout)
	rentalInstallCheck(t, problem)
	cfg := config.Config{Home: layout.Root, HubURL: "https://unused.example"}
	var ready atomic.Int32
	server := api.New(api.Options{Orchestrator: owner, Cfg: cfg, Creds: credentials, Addr: address,
		RentalInstallStatus: q.Status, RentalInstallCancel: q.Cancel,
		MachineReady: func(machine string) {
			if machine != machines.Local {
				t.Errorf("readiness named %q", machine)
			}
			ready.Add(1)
			q.Wake()
		},
	})
	handler, problem := server.Handler()
	rentalInstallCheck(t, problem)
	peer.Config.Handler = http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/" {
			w.WriteHeader(http.StatusOK) // The fixture has no frontend; all API routes remain real.
			return
		}
		handler.ServeHTTP(w, r)
	})
	peer.Start()
	client, problem := localapi.Open(cfg, daemon.Probe(cfg))
	rentalInstallCheck(t, problem)
	rentalInstallCheck(t, client.MachineReady(machines.Local))
	if ready.Load() != 1 {
		t.Fatal("ready notification did not reach the queue")
	}
	command := func(verb string, extra ...string) string {
		t.Helper()
		var out, diagnostic bytes.Buffer
		var grammar cli.CLI
		parser, err := kong.New(&grammar, kong.Writers(&out, &diagnostic))
		if err != nil {
			t.Fatal(err)
		}
		parsed, err := parser.Parse(append([]string{"machine", "installs", verb, row.ID}, extra...))
		if err != nil {
			t.Fatal(err)
		}
		if err := parsed.Run(&cli.Runtime{Cfg: cfg, Out: &out, Err: &diagnostic, Mode: output.Mode{JSON: true, Full: true}}); err != nil {
			t.Fatalf("%s: %v: %s", verb, err, &diagnostic)
		}
		return out.String()
	}
	if out := command("show"); !strings.Contains(out, `"status":"queued"`) || !strings.Contains(out, "paul/checkpoint") {
		t.Fatalf("missing existing operation identity/selection: %s", out)
	}
	for n := 0; n < 2; n++ {
		if out := command("cancel"); !strings.Contains(out, `"status":"canceled"`) || !strings.Contains(out, "install.canceled") {
			t.Fatalf("cancel was not explicit and idempotent: %s", out)
		}
	}
	if out := command("show", "--await"); !strings.Contains(out, `"status":"canceled"`) {
		t.Fatalf("await changed terminal cancellation: %s", out)
	}
	row, problem = q.Accept(machines.Local, records.RentalInstallSelection{Package: "proof/other"})
	rentalInstallCheck(t, problem)
	go func() {
		time.Sleep(20 * time.Millisecond)
		store.SettleRentalInstall(row.ID, "succeeded", nil, nil)
	}()
	if out := command("show", "--await"); !strings.Contains(out, `"status":"succeeded"`) {
		t.Fatalf("await did not observe the existing operation: %s", out)
	}
}
