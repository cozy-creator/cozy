package producttest

import (
	"bufio"
	"context"
	"crypto/ed25519"
	"crypto/x509"
	"encoding/base64"
	"encoding/json"
	"encoding/pem"
	"flag"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/cozy-creator/cozy/internal/accountauth"
	"github.com/cozy-creator/cozy/internal/cli"
	"github.com/cozy-creator/cozy/internal/config"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/hub"
	"github.com/cozy-creator/cozy/internal/orchestrator"
	"github.com/cozy-creator/cozy/internal/records"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
	"google.golang.org/protobuf/proto"
)

var composedBridge = flag.String("derived-composed-bridge", "", "actual Go Host bridge test executable")
var composedLauncher = flag.String("derived-composed-launcher", "", "isolated scheduled Runtime launcher")
var composedObjectHost = flag.String("derived-composed-object-host", "", "explicit R2 host for the real tiny source")

type measuredCheckpointOwner struct {
	orchestrator.ModelTransferOwner
	uploaded   atomic.Int64
	downloaded atomic.Int64
}

func (o *measuredCheckpointOwner) RestoreWeightsCheckpoint(ctx context.Context, request string, host orchestrator.CheckpointHost, subject *pb.WeightsCheckpointSubject) (*pb.CheckpointRef, *exit.Error) {
	transfer := host.Transfer
	host.Transfer = func(ctx context.Context, call *pb.CheckpointTransferRequest) (*pb.CheckpointTransferStatus, *exit.Error) {
		answer, problem := transfer(ctx, call)
		if problem == nil && answer != nil && call.GetDownloadUrl() != "" {
			o.downloaded.Add(int64(answer.TransferredBytes))
		}
		return answer, problem
	}
	return o.ModelTransferOwner.RestoreWeightsCheckpoint(ctx, request, host, subject)
}

func (o *measuredCheckpointOwner) SyncCheckpoints(ctx context.Context, request string, host orchestrator.CheckpointHost) *exit.Error {
	transfer := host.Transfer
	host.Transfer = func(ctx context.Context, call *pb.CheckpointTransferRequest) (*pb.CheckpointTransferStatus, *exit.Error) {
		answer, problem := transfer(ctx, call)
		if problem == nil && answer != nil && call.GetUploadGrant() != nil {
			o.uploaded.Add(int64(answer.TransferredBytes))
		}
		return answer, problem
	}
	return o.ModelTransferOwner.SyncCheckpoints(ctx, request, host)
}

func TestComposedScheduledDerivedCheckpointCustody(t *testing.T) {
	if *composedBridge == "" || *composedLauncher == "" || *composedObjectHost == "" || *publicationHub == "" || *publicationHome == "" || *publicationModel == "" {
		t.Skip("requires explicit actual Host/Runtime and task-owned live publication")
	}
	for _, fault := range []string{"runtime_exit", "executor_exit"} {
		t.Run(fault, func(t *testing.T) { composedScheduledDerivedCheckpointCustody(t, fault) })
	}
}

func composedScheduledDerivedCheckpointCustody(t *testing.T, fault string) {
	root := t.TempDir()
	// Reuse the established input/output file peer; its unused stand-in control
	// address is replaced by the real Go Host before Creator connects.
	connection, _ := startFakePod(t, root, &fakePod{})
	mediaDir := filepath.Join(root, "pod-media")
	must(t, os.MkdirAll(mediaDir, 0755))
	command := exec.Command(*composedBridge, "-test.run=^TestComposedDerivedWorkerBridge$", "-test.v", "-composed-worker-launcher="+*composedLauncher, "-composed-media-directory="+mediaDir, "-composed-source-host="+*composedObjectHost)
	input, err := command.StdinPipe()
	must(t, err)
	output, err := command.StdoutPipe()
	must(t, err)
	command.Stderr = io.Discard
	must(t, command.Start())
	defer func() {
		input.Close()
		if command.ProcessState == nil {
			command.Process.Kill()
			command.Wait()
		}
	}()
	scanner := bufio.NewScanner(output)
	scanner.Buffer(make([]byte, 4096), 1<<20)
	var bridge struct {
		Address, Certificate, IdentityPath, WorkerID, WorkerBootID, Descriptor, Build, Application, Locked, Inventory, Encoding, SignalPath, SourceManifest, SourceBindingPath string
		SourceBytes                                                                                                                                                            int64
	}
	// Field names are the exact native fixture projections, not new identities.
	var raw map[string]json.RawMessage
	for scanner.Scan() {
		if strings.HasPrefix(scanner.Text(), "BRIDGE ") {
			must(t, json.Unmarshal([]byte(strings.TrimPrefix(scanner.Text(), "BRIDGE ")), &raw))
			break
		}
	}
	if raw == nil {
		t.Fatal("real scheduled bridge did not boot")
	}
	for name, dest := range map[string]*string{"address": &bridge.Address, "certificate": &bridge.Certificate, "identity_path": &bridge.IdentityPath, "worker_id": &bridge.WorkerID, "worker_boot_id": &bridge.WorkerBootID, "descriptor": &bridge.Descriptor, "build": &bridge.Build, "application": &bridge.Application, "locked": &bridge.Locked, "inventory": &bridge.Inventory, "encoding": &bridge.Encoding, "signal_path": &bridge.SignalPath, "source_manifest": &bridge.SourceManifest, "source_binding_path": &bridge.SourceBindingPath} {
		must(t, json.Unmarshal(raw[name], dest))
	}
	must(t, json.Unmarshal(raw["source_bytes"], &bridge.SourceBytes))
	certificate := filepath.Join(root, "real-host.pem")
	must(t, os.WriteFile(certificate, []byte(bridge.Certificate), 0600))
	key, err := os.ReadFile(bridge.IdentityPath)
	must(t, err)
	block, _ := pem.Decode(key)
	if block == nil {
		t.Fatal("real Host creator key missing")
	}
	parsed, err := x509.ParsePKCS8PrivateKey(block.Bytes)
	must(t, err)
	signer, ok := parsed.(ed25519.PrivateKey)
	if !ok {
		t.Fatal("real Host key type")
	}
	connection.Addr, connection.CACert, connection.WorkerID, connection.WorkerBootID = bridge.Address, certificate, bridge.WorkerID, bridge.WorkerBootID
	locked, err := base64.StdEncoding.DecodeString(bridge.Locked)
	must(t, err)
	inventoryRaw, err := base64.StdEncoding.DecodeString(bridge.Inventory)
	must(t, err)
	inventory := new(pb.ImageInventory)
	must(t, proto.Unmarshal(inventoryRaw, inventory))
	auth := accountauth.New(config.Config{HubURL: *publicationHub, Home: *publicationHome})
	var transferOwner *measuredCheckpointOwner
	configure := func(options *orchestrator.Options) {
		rentalWiring(connection, signer)(options)
		options.Cfg.HubURL = *publicationHub
		transferOwner = &measuredCheckpointOwner{ModelTransferOwner: cli.NewModelTransferOwner(options.Cfg, options.Store, options.Log, auth)}
		options.ModelTransfers = transferOwner
		options.ReleaseManagedRental = func(string) (string, *exit.Error) { return "", nil }
		options.RentalPrepareFacts = func(context.Context, *orchestrator.WorkerConnection, *pb.DownloadPackageRef) (orchestrator.PrepareFacts, *exit.Error) {
			return orchestrator.PrepareFacts{Application: bridge.Application, ModelSlotPaths: []string{"produce.models.source"}, LockedRequirements: locked, ImageInventory: inventory}, nil
		}
	}
	o := hostOwner(t, "derived-composed-"+records.NewID("proof"), configure)
	// The isolated rental adapter has no provider. Persist its test rental just
	// like the existing rental product fixtures so normal startup can reattach it.
	fatal(t, o.store.RecordRental(records.Rental{AcceleratorCount: 1, ID: podRental, MachineName: "derived-proof", SKU: "cpu",
		AcceleratorModel: "CPU", HourlyRateUSDMicros: 1, State: "ready", Hub: *publicationHub,
		Address: bridge.Address, CertPath: certificate}))
	scaleMarker := records.NewID("scale")
	payload, err := json.Marshal(map[string]string{"encoding": bridge.Encoding, "signal_path": bridge.SignalPath, "scale_marker": scaleMarker})
	must(t, err)
	request, _, problem := o.c.Submit(orchestrator.Submission{IdemKey: "composed-derived", Package: "proof/derived-recovery-job", Entrypoint: "produce", PlanID: bridge.Descriptor, Release: "0.1.0", Kind: "job", Org: "paul", Payload: payload, Outputs: []string{"model"}, WeightsOutputs: []orchestrator.WeightsOutput{{OutputID: "model", MimeType: orchestrator.WeightsManifestMime, MaxBytes: 8 << 20}}, Worker: podRental, Rental: true, RentalRequired: true, ProducerParams: []string{"source"}, Models: []orchestrator.ModelRef{{Package: "proof/derived-recovery-job", Slot: "source", BindingPath: bridge.SourceBindingPath, Model: "paul/creator-output-b93c35ccecf8c183", Release: "0.0.0-recovery-source-config.20260906", Lane: "native-source", Manifest: bridge.SourceManifest, ManifestLength: 161, Bytes: bridge.SourceBytes}}, ModelTransfer: &records.ModelTransferIntent{Kind: "model-upload", Destination: *publicationModel, Outputs: []records.ModelTransferOutput{{Name: "model"}}}})
	fatal(t, problem)
	checkpointsReleased := false
	ownerClosed := false
	defer func() {
		if checkpointsReleased {
			return
		}
		cleanupOwner := transferOwner.ModelTransferOwner
		if ownerClosed {
			store, problem := records.Open(o.l.DB)
			if problem != nil {
				t.Errorf("cleanup reopen: %s", problem.ErrName())
				return
			}
			defer store.Close()
			cfg := o.cfg
			cfg.HubURL = *publicationHub
			cleanupOwner = cli.NewModelTransferOwner(cfg, store, nil, auth)
		}
		if problem := cleanupOwner.ReleaseCheckpoints(context.Background(), request); problem != nil {
			t.Errorf("checkpoint cleanup: %s", problem.ErrName())
		}
	}()
	for deadline := time.Now().Add(45 * time.Second); ; {
		rows, problem := o.store.ModelWeightsProgress(request)
		fatal(t, problem)
		if len(rows) == 1 && rows[0].Acknowledged != nil {
			t.Logf("SAME scheduled UID65533 artifact -> actual Go Host -> actual Creator uploader -> R2 custody acknowledged: bytes=%d", rows[0].Acknowledged.Bytes)
			break
		}
		if time.Now().After(deadline) {
			t.Fatalf("scheduled checkpoint not acknowledged: %s\n%v", tail(filepath.Join(o.root, "orchestrator.log")), o.c.Events())
		}
		time.Sleep(100 * time.Millisecond)
	}
	before := transferOwner.uploaded.Load()
	if fault == "executor_exit" {
		// Close the real record owner before the executor fails. The Host's
		// production detached stream must retain the actual Runtime outcome.
		o.close()
		ownerClosed = true
	}
	must(t, json.NewEncoder(input).Encode(map[string]string{"action": "restart_worker", "fault": fault}))
	restarted := false
	for scanner.Scan() {
		if strings.HasPrefix(scanner.Text(), "FAULT ") || strings.Contains(scanner.Text(), ".go:") {
			t.Log(scanner.Text())
		}
		if strings.HasPrefix(scanner.Text(), "RESULT ") {
			var result struct{ Restarted, ColdStore bool }
			var raw map[string]bool
			must(t, json.Unmarshal([]byte(strings.TrimPrefix(scanner.Text(), "RESULT ")), &raw))
			result.Restarted, result.ColdStore = raw["restarted"], raw["cold_store"]
			if !result.Restarted || !result.ColdStore {
				t.Fatal("worker replacement was not cold")
			}
			restarted = true
			break
		}
	}
	if !restarted {
		t.Fatal("worker replacement did not return")
	}
	if fault == "executor_exit" {
		store, problem := records.Open(o.l.DB)
		fatal(t, problem)
		log, err := os.OpenFile(filepath.Join(o.root, "orchestrator.log"), os.O_WRONLY|os.O_APPEND, 0600)
		must(t, err)
		options := orchestrator.Options{Cfg: o.cfg, Layout: o.l, Store: store, Yield: "smart", Log: log, MaxOutputMiB: 8}
		configure(&options)
		controller, problem := orchestrator.Open(options)
		fatal(t, problem)
		o = &owner{root: o.root, cfg: o.cfg, l: o.l, store: store, c: controller}
		ownerClosed = false
		o.closer = func() { controller.Close(20 * time.Second); store.Close(); log.Close() }
		t.Cleanup(o.close)
		_, _, problem = controller.Reconcile()
		fatal(t, problem)
		go func() { _ = controller.Serve() }()
		before = 0
	}
	drained := make(chan struct{})
	results := make(chan string, 1)
	go func() {
		defer close(drained)
		for scanner.Scan() {
			if strings.HasPrefix(scanner.Text(), "UPLOAD ") {
				t.Log(scanner.Text())
			} else if strings.HasPrefix(scanner.Text(), "RESULT ") {
				results <- strings.TrimPrefix(scanner.Text(), "RESULT ")
			}
		}
	}()
	for deadline := time.Now().Add(60 * time.Second); ; {
		row, problem := o.store.RequestRow(request)
		fatal(t, problem)
		if row != nil && row.State == "succeeded" {
			if row.Ordinal != 2 {
				t.Fatal("recovery must requeue exactly once")
			}
			moved := transferOwner.uploaded.Load() - before
			if moved >= 4<<20 {
				t.Fatal("replacement reuploaded the complete data role")
			}
			if transferOwner.downloaded.Load() < 4<<20 {
				t.Fatal("cold replacement did not fetch the retained data role through the real checkpoint transport")
			}
			t.Logf("scheduled process crash/cold Store/owner rejoin/native role reuse COMPLETE: attempts=%d replacementCheckpointUpload=%d restored=%d", row.Ordinal, moved, transferOwner.downloaded.Load())
			break
		}
		if row != nil && (row.State == "failed" || row.State == "refused" || row.State == "canceled") {
			t.Fatalf("replacement request ended %s: %s", row.State, tail(filepath.Join(o.root, "orchestrator.log")))
		}
		if time.Now().After(deadline) {
			t.Fatalf("replacement did not finish: %s", tail(filepath.Join(o.root, "orchestrator.log")))
		}
		time.Sleep(100 * time.Millisecond)
	}
	outputs, problem := o.store.AllModelTransferWeights(request, 2)
	fatal(t, problem)
	attempts, problem := o.store.Attempts(request)
	fatal(t, problem)
	if len(attempts) != 2 || attempts[0].State != "closed" || attempts[1].State != "closed" ||
		attempts[0].InvocationDigest != attempts[1].InvocationDigest {
		t.Fatal("recovery changed immutable work or failed to close exactly two attempts")
	}
	if len(outputs) != 1 || outputs[0].FinalID == "" {
		t.Fatal("normal finalizer did not retain one exact checkpoint")
	}
	ref, problem := hub.ParseRef(*publicationModel)
	fatal(t, problem)
	client := hub.New(config.Config{HubURL: *publicationHub, Home: *publicationHome}, "cozy-derived-proof").WithTokenSource(auth)
	defer func() {
		if problem := client.RemoveCheckpoint(context.Background(), ref, outputs[0].ManifestID, "remove unique composed-recovery output"); problem != nil {
			t.Errorf("output checkpoint cleanup: %s", problem.ErrName())
		}
	}()
	must(t, json.NewEncoder(input).Encode(map[string]string{"action": "verify", "manifest": outputs[0].ManifestID, "transaction": outputs[0].TransactionID, "scale_marker": scaleMarker}))
	select {
	case raw := <-results:
		var verified struct {
			Verified         bool
			PayloadBytes     int64          `json:"payload_bytes"`
			Uploads          map[string]int `json:"uploads"`
			PriorRevision    uint64         `json:"prior_desired_revision"`
			RetainedRevision uint64         `json:"retained_desired_revision"`
		}
		must(t, json.Unmarshal([]byte(raw), &verified))
		if !verified.Verified || verified.PayloadBytes != (4<<20)+4096+2048 {
			t.Fatalf("native post-GC exact payload verification failed: %s", raw)
		}
		for id, count := range verified.Uploads {
			if count != 1 {
				t.Fatalf("final output mover issued %d PUTs for %s", count, id)
			}
			found := false
			for _, object := range outputs[0].Objects {
				if object.ObjectID == id {
					found = true
					if object.Length >= 4<<20 {
						t.Fatal("final mover reuploaded the retained data role")
					}
				}
			}
			if !found {
				t.Fatal("final mover uploaded outside its exact adopted inventory")
			}
		}
		if len(verified.Uploads) < 2 {
			t.Fatal("fresh header/Manifest PUT path was not exercised")
		}
		if verified.PriorRevision == 0 || verified.RetainedRevision <= verified.PriorRevision {
			t.Fatal("replacement did not advance the Host's existing revision")
		}
		t.Logf("existing Host desired revision advanced %d -> %d", verified.PriorRevision, verified.RetainedRevision)
		t.Logf("existing final-output mover issued one PUT for each of %d fresh objects", len(verified.Uploads))
		t.Logf("native ADOPT + GC + current Store lease verified all %d payload bytes", verified.PayloadBytes)
	case <-time.After(15 * time.Second):
		t.Fatal("native post-GC verification did not return")
	}
	fatal(t, transferOwner.ReleaseCheckpoints(context.Background(), request))
	checkpointsReleased = true
	o.close()
	must(t, json.NewEncoder(input).Encode(map[string]string{"action": "stop"}))
	<-drained
	must(t, command.Wait())
}
