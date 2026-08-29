package live

import (
	"bytes"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/cozy-creator/cozy-creator/internal/api"
	"github.com/cozy-creator/cozy-creator/internal/config"
	"github.com/cozy-creator/cozy-creator/internal/home"
	"github.com/cozy-creator/cozy-creator/internal/hub"
	"github.com/cozy-creator/cozy-creator/internal/media"
	"github.com/cozy-creator/cozy-creator/internal/mediawire"
	"github.com/cozy-creator/cozy-creator/internal/orchestrator"
	"github.com/cozy-creator/cozy-creator/internal/records"
	"github.com/cozy-creator/cozy-creator/internal/secret"
	pb "github.com/cozy-creator/cozy-creator/protocol/cozy/worker/v1"
)

// TestDownDoesNotResurrectReleasedRental proves the localhost API decision that drives
// `cozy down --all`: once the durable rental authority records provider absence, an
// attached worker cannot turn that released rental back into a paid obligation.
func TestDownDoesNotResurrectReleasedRental(t *testing.T) {
	root := t.TempDir()
	certPath, keyPath := podTLS(t, root)
	tokenText := "released-rental-owner"
	mediaAddr := serveMediaPeer(t, certPath, keyPath, tokenText, mediawire.ContractRev)
	controlAddr, stopControl := serveWorkerPeer(t, certPath, keyPath, heldWorkerPeer{})
	defer stopControl()

	layout, problem := home.Open(root)
	fatal(t, problem)
	store, problem := records.Open(layout.DB)
	fatal(t, problem)
	defer store.Close()
	owner, problem := orchestrator.Open(orchestrator.Options{
		Layout: layout, Store: store, Log: io.Discard, MaxOutputMiB: 1,
	})
	fatal(t, problem)
	defer owner.Close(time.Second)

	rentalID := "rental-already-released"
	_, _, problem = owner.EnsureWorker(orchestrator.WorkerLaunchSpec{
		Placement: orchestrator.DesiredPlacement{
			Package: "cozy/marco", PackageReleaseID: "release-1",
		},
		Connection: &orchestrator.WorkerConnection{
			RentalID: rentalID, Addr: controlAddr, Token: secret.New(tokenText),
			CACert: certPath, Media: &media.Spec{
				Addr: mediaAddr, Token: secret.New(tokenText), CACert: certPath,
			},
		},
	})
	fatal(t, problem)
	if workers := owner.Workers(); len(workers) != 1 || workers[0].RentalID != rentalID {
		t.Fatalf("attached rental workers = %#v", workers)
	}
	fatal(t, store.RecordRental(records.Rental{
		ID: rentalID, PackageRef: "cozy/marco", AcceleratorModel: "CPU",
		State: hub.RentalReleased, Hub: "http://tensorhub.invalid",
	}))
	forgotten, problem := store.ForgetRental(rentalID)
	fatal(t, problem)
	if !forgotten {
		t.Fatal("released rental row was not removed")
	}

	cliToken := secret.New("localhost-down-client")
	server := httptest.NewUnstartedServer(nil)
	shutdown := make(chan struct{}, 1)
	local := api.New(api.Options{
		Orchestrator: owner, Cfg: config.Config{},
		Creds: api.Credentials{CLI: cliToken}, Addr: server.Listener.Addr().String(),
		Log: io.Discard, Web: http.NotFoundHandler(), Shutdown: func() { shutdown <- struct{}{} },
	})
	handler, problem := local.Handler()
	fatal(t, problem)
	server.Config.Handler = handler
	server.Start()
	defer server.Close()

	body, err := json.Marshal(map[string]bool{"all": true})
	must(t, err)
	request, err := http.NewRequest(http.MethodPost, server.URL+"/v1/local/daemon/down",
		bytes.NewReader(body))
	must(t, err)
	request.Header.Set("Content-Type", "application/json")
	api.Authorize(request, cliToken)
	response, err := server.Client().Do(request)
	must(t, err)
	defer response.Body.Close()
	var result api.DownResult
	must(t, json.NewDecoder(response.Body).Decode(&result))
	if response.StatusCode != http.StatusAccepted || !result.ShuttingDown || len(result.Rentals) != 0 {
		t.Fatalf("down after confirmed release = HTTP %d %#v", response.StatusCode, result)
	}
	select {
	case <-shutdown:
	case <-time.After(time.Second):
		t.Fatal("accepted down did not invoke cooperative shutdown")
	}
}

type heldWorkerPeer struct {
	pb.UnimplementedWorkerControlServer
}

func (heldWorkerPeer) Control(stream pb.WorkerControl_ControlServer) error {
	if _, err := stream.Recv(); err != nil {
		return err
	}
	<-stream.Context().Done()
	return nil
}

func (heldWorkerPeer) WatchProgress(_ *pb.ProgressOpen, stream pb.WorkerControl_WatchProgressServer) error {
	<-stream.Context().Done()
	return nil
}
