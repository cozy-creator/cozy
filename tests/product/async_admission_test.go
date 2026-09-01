package producttest

import (
	"context"
	"encoding/json"
	"net/http/httptest"
	"sync/atomic"
	"testing"
	"time"

	"github.com/cozy-creator/cozy/internal/api"
	localclient "github.com/cozy-creator/cozy/internal/client"
	"github.com/cozy-creator/cozy/internal/daemon"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/launch"
	"github.com/cozy-creator/cozy/internal/orchestrator"
	"github.com/cozy-creator/cozy/internal/privatepackage"
	"github.com/cozy-creator/cozy/internal/records"
)

type admissionResolver struct{ job launch.Entrypoint }

func (r admissionResolver) RefreshEditable(string) (string, bool, bool, *exit.Error) {
	return "", false, false, nil
}
func (r admissionResolver) PreparePrivate(context.Context, string) (privatepackage.Revision, *exit.Error) {
	return privatepackage.Revision{}, exit.Unavailablef("not used")
}
func (r admissionResolver) PrivateRevision(string, string) (privatepackage.Revision, *exit.Error) {
	return privatepackage.Revision{}, exit.Unavailablef("not used")
}
func (r admissionResolver) ResolvePlacement(string) (orchestrator.DesiredPlacement, *exit.Error) {
	return orchestrator.DesiredPlacement{}, exit.Unavailablef("not used")
}
func (r admissionResolver) ResolveInstall(string, []orchestrator.ModelRef) (orchestrator.WorkerLaunchSpec, *exit.Error) {
	return orchestrator.WorkerLaunchSpec{}, exit.Unavailablef("not used")
}
func (r admissionResolver) ResolveRemoteRelease(string, string, string, string,
	[]orchestrator.ModelRef,
) (orchestrator.LogicalPackage, *launch.Entrypoint, *exit.Error) {
	return orchestrator.LogicalPackage{}, nil, exit.Unavailablef("not used")
}
func (r admissionResolver) ResolveRemoteJob(pkg, release, releaseDigest, function string,
	_ []orchestrator.ModelRef, _ bool,
) (orchestrator.LogicalJob, *launch.Entrypoint, *exit.Error) {
	return orchestrator.LogicalJob{Package: pkg, Release: release, ReleaseDigest: releaseDigest,
		Function: function, DescriptorID: digest("d"), Requires: "ram8g"}, &r.job, nil
}
func (r admissionResolver) Entrypoint(string, string) (*launch.Entrypoint, *exit.Error) {
	return nil, exit.Unavailablef("not used")
}
func (r admissionResolver) Jobs(string) ([]launch.JobFacts, *exit.Error) { return nil, nil }
func (r admissionResolver) JobsInstall(string) ([]launch.JobFacts, *exit.Error) {
	return nil, nil
}

func TestSubmissionReturnsBeforeRentalAndLaterFailureIsWatchable(t *testing.T) {
	started := make(chan struct{}, 2)
	release := make(chan struct{})
	var acquisitions atomic.Int64
	owner := hostOwnerConfigured(t, "async-admission", nil, func(options *orchestrator.Options) {
		options.RentalFleet = func() (string, *exit.Error) { return "rentals: proof", nil }
		options.AcquireManagedRental = func(records.Request) (string, string, *exit.Error) {
			acquisitions.Add(1)
			started <- struct{}{}
			<-release
			return "", "", exit.Named(exit.Unavailable, "rental.proof_unavailable",
				"the proof provider refused capacity")
		}
	})
	owner.cfg.RentalsMaxHourlySpendUSDMicros = 10_000_000
	creds, problem := api.Mint(owner.l)
	fatal(t, problem)
	server := httptest.NewUnstartedServer(nil)
	localAPI := api.New(api.Options{Orchestrator: owner.c, Cfg: owner.cfg, Creds: creds,
		Addr: server.Listener.Addr().String(), Packages: admissionResolver{job: launch.Entrypoint{
			Name: "work", Kind: "job", DescriptorID: digest("d"), Request: launch.Struct{Fields: []launch.Field{{
				Name: "value", Type: json.RawMessage(`"int"`), Wire: "required",
			}}},
		}}})
	handler, problem := localAPI.Handler()
	fatal(t, problem)
	server.Config.Handler = handler
	server.Start()
	defer server.Close()
	client, problem := localclient.Open(owner.cfg, daemon.State{Up: true,
		Addr: server.Listener.Addr().String()})
	fatal(t, problem)

	submission := api.JobSubmission{Package: "proof/job", Function: "work",
		Input: json.RawMessage(`{"value":1}`), Release: "1.0.0", ReleaseDigest: digest("e"),
		Rental: true, RentalRequired: true}
	before := time.Now()
	handle, problem := client.SubmitJob(submission, "async-admission-one")
	fatal(t, problem)
	if time.Since(before) > time.Second || handle.JobID == "" || handle.Attempt != 0 ||
		handle.Status != "queued" {
		t.Fatalf("submission waited for activation or returned an inconsistent handle: %+v in %s",
			handle, time.Since(before))
	}
	select {
	case <-started:
	case <-time.After(time.Second):
		t.Fatal("background rental activation did not start")
	}

	// A pending provider call cannot convoy a second mutation or `down`.
	second := submission
	second.Input = json.RawMessage(`{"value":2}`)
	if another, problem := client.SubmitJob(second, "async-admission-two"); problem != nil ||
		another.JobID == "" {
		t.Fatalf("second submission convoyed behind provider activation: %+v %v", another, problem)
	}
	downStarted := time.Now()
	if _, problem := client.Down(false); problem == nil || time.Since(downStarted) > time.Second {
		t.Fatalf("down did not promptly report active work: %v in %s", problem, time.Since(downStarted))
	}

	// Invalid payloads are refused before a row or provider call exists.
	invalid := submission
	invalid.Input = json.RawMessage(`{"unknown":1}`)
	if _, problem := client.SubmitJob(invalid, "async-admission-invalid"); problem == nil ||
		problem.Code != exit.Validation {
		t.Fatalf("invalid payload was admitted: %v", problem)
	}
	if row, problem := owner.store.RequestByIdempotencyKey("async-admission-invalid"); problem != nil || row != nil {
		t.Fatalf("invalid payload created a durable row: %+v %v", row, problem)
	}

	close(release)
	for _, key := range []string{"async-admission-one", "async-admission-two"} {
		row, problem := owner.store.RequestByIdempotencyKey(key)
		fatal(t, problem)
		if row == nil {
			t.Fatalf("accepted request %s disappeared", key)
		}
		waitRequest(t, owner, row.ID, "failed", 3*time.Second)
		events, problem := owner.store.EventsAfter(row.ID, 0, 100)
		fatal(t, problem)
		found := false
		for _, event := range events {
			found = found || event.Type == "request.failed" &&
				event.Payload["error_type"] == "rental.proof_unavailable"
		}
		if !found {
			t.Fatalf("accepted activation failure is absent from watch events: %+v", events)
		}
	}
	if acquisitions.Load() != 2 {
		t.Fatalf("provider calls = %d, want exactly the two accepted requests", acquisitions.Load())
	}
}
