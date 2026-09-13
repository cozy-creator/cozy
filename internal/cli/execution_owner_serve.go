package cli

import (
	"context"
	"crypto/ed25519"
	"encoding/json"
	"net/http"
	"os"
	"os/signal"
	"path/filepath"
	"strings"
	"syscall"

	"github.com/cozy-creator/cozy/internal/api"
	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/daemon"
	"github.com/cozy-creator/cozy/internal/executionowner"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/home"
	"github.com/cozy-creator/cozy/internal/launch"
	"github.com/cozy-creator/cozy/internal/media"
	"github.com/cozy-creator/cozy/internal/orchestrator"
	"github.com/cozy-creator/cozy/internal/records"
	"github.com/cozy-creator/cozy/internal/secret"
	cozyweb "github.com/cozy-creator/cozy/web"
)

// This is bootstrap wiring for the existing coordinator and API. There is no
// second request scheduler or interpreter, and no rental/provider authority.
func serveExecutionOwner(runtime *Runtime, bootstrap *executionowner.Bootstrap) *exit.Error {
	authority := bootstrap.Authority
	layout, problem := home.Open(authority.CreatorHome)
	if problem != nil {
		return problem
	}
	v4, v6, address, problem := api.Listeners(0)
	if problem != nil {
		return problem
	}
	defer v4.Close()
	if v6 != nil {
		defer v6.Close()
	}
	held, problem := daemon.Hold(layout, address, filepath.Join(layout.Root, "worker.sock"))
	if problem != nil {
		return problem
	}
	defer held.Release()
	if problem := bootstrap.PinGeneration(); problem != nil {
		return problem
	}
	store, problem := records.OpenForDaemon(layout.DB, filepath.Join(layout.Root, "triage"))
	if problem != nil {
		return problem
	}
	defer store.Close()
	// Only Host-declared local roots and the frozen process tool environment
	// participate. External source/publication grants are a separate capability.
	cfg := runtime.Cfg
	cfg.Home, cfg.TensorFSRoot, cfg.TensorFSRootSource, cfg.Yield = layout.Root, authority.TensorFSRoot, "execution-owner", "never"
	cfg.HubURL, cfg.HubToken, cfg.HuggingFaceToken, cfg.CivitaiToken, cfg.Bootstrap = "", secret.Value{}, secret.Value{}, secret.Value{}, secret.Value{}
	cfg.RentalsIdleRelease, cfg.DaemonIdleShutdown, cfg.MaintenanceGCCron = 0, 0, ""
	resolver := NewResolver(store, cfg, nil)
	grant := bootstrap.Grant.Grant
	grantBytes, err := canonical.Bytes(grant)
	if err != nil {
		return exit.New(exit.Validation, "execution grant cannot be identified")
	}
	grantDigest, _ := canonical.Spell(canonical.Digest(grantBytes))
	const worker = "private-execution-worker"
	connection := &orchestrator.WorkerConnection{RentalID: worker, Addr: authority.WorkerAddress,
		CACert: authority.WorkerTLSCertificatePath, WorkerID: grant.WorkerId, WorkerBootID: grant.WorkerBootId,
		Media: &media.Spec{Addr: strings.TrimPrefix(authority.MediaAddress, "https://"), CACert: authority.WorkerTLSCertificatePath, Token: secret.New(authority.MediaBearer)}}
	owner, problem := orchestrator.Open(orchestrator.Options{Cfg: cfg, Layout: layout, Store: store, Yield: "never", Log: runtime.Err,
		Packages:         resolver,
		ObserveRental:    bootstrap.ObserveWorker,
		PrivateExecution: &orchestrator.PrivateExecutionOwner{Authorization: bootstrap.Grant, PrivateKey: ed25519.PrivateKey(authority.ExecutionPrivateKey)},
		Rentals: func(id string) (*orchestrator.RemoteTarget, *exit.Error) {
			if id != worker {
				return nil, exit.New(exit.Credential, "private execution cannot resolve another machine")
			}
			return &orchestrator.RemoteTarget{Connection: connection}, nil
		}})
	if problem != nil {
		return problem
	}
	defer owner.Close(orchestrator.StopGrace)
	if _, _, problem := owner.Reconcile(); problem != nil {
		return problem
	}
	creds, problem := api.Mint(layout)
	if problem != nil {
		return problem
	}
	stop := make(chan os.Signal, 1)
	server := api.New(api.Options{Orchestrator: owner, Cfg: cfg, Creds: creds, Addr: address, Log: runtime.Err,
		Web:      cozyweb.Handler(),
		Packages: resolver, ExecutionGrantDigest: grantDigest,
		ExecutionCapture: func(ctx context.Context, key string, raw []byte) (orchestrator.Submission, *exit.Error) {
			return resolveExecutionCapture(ctx, bootstrap, layout, store, resolver, worker, grantDigest, key, raw)
		}, Shutdown: func() {
			select {
			case stop <- syscall.SIGTERM:
			default:
			}
		}})
	handler, problem := server.Handler()
	if problem != nil {
		return problem
	}
	httpServer := daemon.NewHTTPServer(handler)
	defer func() {
		ctx, cancel := context.WithTimeout(context.Background(), orchestrator.StopGrace)
		defer cancel()
		_ = httpServer.Shutdown(ctx)
	}()
	go func() { _ = httpServer.Serve(v4) }()
	if v6 != nil {
		go func() { _ = httpServer.Serve(v6) }()
	}
	go func() { _ = owner.Serve() }()
	row, fresh, problem := server.AdmitExecutionCapture(context.Background(), bootstrap.Capsule.Capsule.Root.IdempotencyKey, bootstrap.Capsule.Raw)
	if problem != nil {
		return problem
	}
	requestID := row.ID
	if fresh {
		go func() { _, _ = owner.ActivateRecordedRequest(row) }()
	}
	if problem := bootstrap.RecordAdmission(requestID); problem != nil {
		return problem
	}
	if err := json.NewEncoder(runtime.Out).Encode(map[string]any{"request_id": requestID, "capsule_digest": bootstrap.Capsule.Digest, "admitted": true}); err != nil {
		return exit.New(exit.Unavailable, "execution admission receipt could not be written")
	}
	signal.Notify(stop, syscall.SIGINT, syscall.SIGTERM)
	defer signal.Stop(stop)
	<-stop
	ctx, cancel := context.WithTimeout(context.Background(), orchestrator.StopGrace)
	defer cancel()
	if err := httpServer.Shutdown(ctx); err != nil && err != http.ErrServerClosed {
		return exit.New(exit.Unavailable, "execution API response drain did not finish")
	}
	return nil
}

func resolveExecutionCapture(ctx context.Context, bootstrap *executionowner.Bootstrap, layout home.Layout, store *records.Store, resolver *Resolver, worker, grantDigest, key string, raw []byte) (orchestrator.Submission, *exit.Error) {
	if ctx.Err() != nil {
		return orchestrator.Submission{}, exit.New(exit.Canceled, "capture import was interrupted")
	}
	captured, problem := bootstrap.ForSubmission(raw)
	if problem != nil {
		return orchestrator.Submission{}, problem
	}
	input := captured.Capsule.Capsule.Root
	if input.IdempotencyKey != key {
		return orchestrator.Submission{}, exit.New(exit.Conflict, "capture key differs from its admission header")
	}
	root, problem := captured.Import(layout, store)
	if problem != nil {
		return orchestrator.Submission{}, problem
	}
	entry, _, problem := resolver.Entrypoint(root.ID, input.Entrypoint)
	if problem != nil {
		return orchestrator.Submission{}, problem
	}
	if problem := launch.ValidatePayload(root.Package, entry, input.Input); problem != nil {
		return orchestrator.Submission{}, problem
	}
	if len(entry.Models) != 0 || entry.Assets != nil {
		return orchestrator.Submission{}, exit.New(exit.Validation, "execution admission requires a captured script without external asset arguments")
	}
	jobs, problem := resolver.JobsInstall(root.ID)
	if problem != nil {
		return orchestrator.Submission{}, problem
	}
	var facts *launch.JobFacts
	for index := range jobs {
		if jobs[index].Name == entry.Name {
			facts = &jobs[index]
			break
		}
	}
	if facts == nil {
		return orchestrator.Submission{}, exit.New(exit.Validation, "execution callable is not a captured job")
	}
	return orchestrator.Submission{
		RequestID: input.RequestID, ExecutionGrantDigest: grantDigest,
		IdemKey: input.IdempotencyKey, BodyDigest: captured.Capsule.Digest,
		Package: root.Package, Entrypoint: entry.Name, Release: root.Version, InstallID: root.ID,
		PlanID: facts.DescriptorID, LocalPackageDigest: input.Revision,
		Payload: input.Input, Kind: "job", RetainWork: true, ReleaseImplicitWork: !facts.RetainsArtifacts,
		ChildArtifacts: facts.RetainsArtifacts, NeedsAccelerator: facts.NeedsAccelerator,
		Outputs: facts.Outputs, WeightsOutputs: facts.WeightsOutputs,
		Worker: worker, RequestedRental: worker, Rental: true, RentalRequired: true,
	}, nil
}
