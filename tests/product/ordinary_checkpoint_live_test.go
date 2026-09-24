package producttest

import (
	"bufio"
	"bytes"
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
	"regexp"
	"strings"
	"testing"
	"time"

	"github.com/cozy-creator/cozy/internal/accountauth"
	"github.com/cozy-creator/cozy/internal/api"
	"github.com/cozy-creator/cozy/internal/cli"
	"github.com/cozy-creator/cozy/internal/config"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/hub"
	"github.com/cozy-creator/cozy/internal/orchestrator"
	"github.com/cozy-creator/cozy/internal/records"
	"github.com/cozy-creator/cozy/internal/rental"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
	"google.golang.org/protobuf/proto"
)

var ordinaryBridge = flag.String("ordinary-job-bridge", "", "actual Go Host bridge for the ordinary CLI checkpoint proof")
var ordinaryLauncher = flag.String("ordinary-job-launcher", "", "isolated scheduled Runtime launcher")
var ordinaryObjectHost = flag.String("ordinary-job-object-host", "", "explicit object-storage hostname for the retained tiny source")

func TestOrdinaryRunPublishedCheckpointJobThroughRealRuntimeAndHub(t *testing.T) {
	if *ordinaryBridge == "" || *ordinaryLauncher == "" || *ordinaryObjectHost == "" || *publicationHub == "" || *publicationHome == "" || *publicationModel == "" {
		t.Skip("requires explicit actual Host/Runtime and live task-owned source/publication")
	}
	ordinaryScheduledCheckpoint(t)
}

func ordinaryScheduledCheckpoint(t *testing.T) {
	root := t.TempDir()
	// Reuse the established input/output file peer; its unused stand-in control
	// address is replaced by the real Go Host before Creator connects.
	connection, _ := startFakePod(t, root, &fakePod{})
	mediaDir := filepath.Join(root, "pod-media")
	must(t, os.MkdirAll(mediaDir, 0755))
	command := exec.Command(*ordinaryBridge, "-test.run=^TestOrdinaryCheckpointWorkerBridge$", "-test.v", "-composed-worker-launcher="+*ordinaryLauncher, "-composed-media-directory="+mediaDir, "-composed-source-host="+*ordinaryObjectHost)
	input, err := command.StdinPipe()
	must(t, err)
	output, err := command.StdoutPipe()
	must(t, err)
	command.Stderr = io.Discard
	must(t, command.Start())
	defer func() {
		input.Close()
		if command.ProcessState == nil {
			done := make(chan error, 1)
			go func() { done <- command.Wait() }()
			select {
			case <-done:
			case <-time.After(15 * time.Second):
				command.Process.Kill()
				<-done
			}
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
	for name, dest := range map[string]*string{"address": &bridge.Address, "certificate": &bridge.Certificate, "identity_path": &bridge.IdentityPath, "worker_id": &bridge.WorkerID, "worker_boot_id": &bridge.WorkerBootID, "descriptor": &bridge.Descriptor, "build": &bridge.Build, "application": &bridge.Application, "locked": &bridge.Locked, "inventory": &bridge.Inventory, "encoding": &bridge.Encoding, "source_manifest": &bridge.SourceManifest, "source_binding_path": &bridge.SourceBindingPath} {
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

	var resolver *cli.Resolver
	configure := func(options *orchestrator.Options) {
		rentalWiring(connection, signer)(options)
		options.Cfg.HubURL = *publicationHub
		options.Cfg.RentalsMaxHourlySpendUSDMicros = 20_000_000
		resolver = cli.NewResolver(options.Store, options.Cfg, nil)
		options.Packages = resolver
		options.ModelTransfers = cli.NewModelTransferOwner(options.Cfg, options.Store, options.Log, auth)
		options.ObserveRental = rental.ObserveWorker(options.Store)
		options.RentalFleet = func() (string, *exit.Error) { return "isolated existing CPU fixture", nil }
		options.AcquireManagedRental = func(request records.Request) (orchestrator.PlacementDecision, string, *exit.Error) {
			if len(request.Models) != 1 || request.Models[0].Manifest != bridge.SourceManifest {
				return orchestrator.PlacementDecision{}, "", exit.Internalf("fixture received another model selection")
			}
			return orchestrator.PlacementDecision{RentalID: podRental}, "isolated existing CPU fixture; no paid acquisition", nil
		}
		options.ReleaseManagedRental = func(string) (string, *exit.Error) { return "", nil }
		options.RentalPrepareFacts = func(_ context.Context, _ *orchestrator.WorkerConnection, request *pb.DownloadPackageRef) (orchestrator.PrepareFacts, *exit.Error) {
			if request.Package != "paul/ordinary-checkpoint-job-proof" || request.Release != "0.1.0" {
				return orchestrator.PrepareFacts{}, exit.Internalf("fixture received another package")
			}
			return orchestrator.PrepareFacts{Application: bridge.Application, ModelSlotPaths: []string{bridge.SourceBindingPath}, LockedRequirements: locked, ImageInventory: inventory}, nil
		}
	}
	o := hostOwner(t, "ordinary-checkpoint-"+records.NewID("proof"), configure)
	o.cfg.HubURL = *publicationHub
	o.cfg.RentalsMaxHourlySpendUSDMicros = 20_000_000
	fatal(t, o.store.RecordRental(records.Rental{AcceleratorCount: 1, ID: podRental, MachineName: "ordinary-proof", SKU: "cpu", AcceleratorModel: "CPU", HourlyRateUSDMicros: 1, State: "ready", Hub: *publicationHub, Address: bridge.Address, CertPath: certificate, ExpectedWorkerID: bridge.WorkerID, ExpectedWorkerBootID: bridge.WorkerBootID}))
	// Copy only the enrolled credential for this exact Hub into the isolated CLI
	// home; neither private key nor bearer enters the worker or the test output.
	authDir := filepath.Join(o.root, "auth")
	must(t, os.MkdirAll(authDir, 0700))
	entries, err := os.ReadDir(filepath.Join(*publicationHome, "auth"))
	must(t, err)
	copied := 0
	for _, entry := range entries {
		if entry.IsDir() {
			continue
		}
		raw, err := os.ReadFile(filepath.Join(*publicationHome, "auth", entry.Name()))
		must(t, err)
		var credential struct{ Hub string }
		if json.Unmarshal(raw, &credential) == nil && credential.Hub == *publicationHub {
			must(t, os.WriteFile(filepath.Join(authDir, entry.Name()), raw, 0600))
			copied++
		}
	}
	if copied != 1 {
		t.Fatal("expected one enrolled credential for the selected Hub")
	}
	must(t, os.WriteFile(filepath.Join(o.root, config.FileName), []byte("tensorhub_url: "+*publicationHub+"\nrentals:\n  max_hourly_spend_usd: 20\ndaemon:\n  idle_shutdown_s: 0\n"), 0600))
	closeAPI := publicationControlAPI(t, o, func(options *api.Options) { options.Packages = resolver; options.Rentals = rental.Known(o.store) })
	defer closeAPI()
	results := make(chan string, 1)
	go func() {
		for scanner.Scan() {
			if strings.HasPrefix(scanner.Text(), "UPLOAD ") {
				t.Log(scanner.Text())
			}
			if strings.HasPrefix(scanner.Text(), "RESULT ") {
				results <- strings.TrimPrefix(scanner.Text(), "RESULT ")
			} else if strings.Contains(scanner.Text(), ".go:") || strings.HasPrefix(scanner.Text(), "FAIL") {
				t.Log(regexp.MustCompile(`https?://[^\s]+`).ReplaceAllString(scanner.Text(), "<url>"))
			}
		}
	}()
	marker := records.NewID("marker")
	args := []string{"run", "paul/ordinary-checkpoint-job-proof/produce", "model.source=paul/creator-output-b93c35ccecf8c183@0.0.0-recovery-source-config.20260906/native-source", "encoding=" + bridge.Encoding, "marker=" + marker, "--publish-to", *publicationModel, "--rental-only", "--json", "--full", "--idempotency-key", "ordinary-checkpoint"}
	code, cliOutput := runCozy(t, o.root, args...)
	if code != 0 {
		t.Fatalf("normal CLI submission failed: %d %s", code, cliOutput)
	}
	request, problem := o.store.RequestByIdempotencyKey("ordinary-checkpoint")
	fatal(t, problem)
	if request == nil || request.Kind != "job" || request.ModelTransfer == nil || request.ModelTransfer.HasAcquisition() || len(request.Models) != 1 || request.Models[0].BindingPath != bridge.SourceBindingPath {
		t.Fatal("ordinary CLI lost the exact job/input/publication declaration")
	}
	waitUntil(t, "actual scheduled native job and publication complete", func() bool {
		row, problem := o.store.RequestRow(request.ID)
		fatal(t, problem)
		if row.State == "failed" || row.State == "refused" {
			t.Fatalf("ordinary native job failed: %v", o.c.Events())
		}
		return row.State == "succeeded"
	})
	attempts, problem := o.store.Attempts(request.ID)
	fatal(t, problem)
	outputs, problem := o.store.AllModelTransferWeights(request.ID, 1)
	fatal(t, problem)
	if len(attempts) != 1 || attempts[0].State != "closed" || len(attempts[0].InvocationCanonical) == 0 || len(outputs) != 1 || outputs[0].FinalID == "" || outputs[0].InvocationDigest != attempts[0].InvocationDigest {
		t.Fatal("ordinary invocation/receipt/checkpoint did not share exact identity")
	}
	// Exact idempotent CLI replay preserves the completed invocation and its receipts.
	code, cliOutput = runCozy(t, o.root, args...)
	if code != 0 {
		t.Fatalf("completed CLI replay failed: %d %s", code, cliOutput)
	}
	after, problem := o.store.Attempts(request.ID)
	fatal(t, problem)
	if len(after) != 1 || !bytes.Equal(after[0].InvocationCanonical, attempts[0].InvocationCanonical) {
		t.Fatal("CLI replay reran or rebuilt the invocation")
	}
	ref, problem := hub.ParseRef(*publicationModel)
	fatal(t, problem)
	client := hub.New(config.Config{HubURL: *publicationHub, Home: *publicationHome}, "cozy-ordinary-job-proof").WithTokenSource(auth)
	publication, problem := client.FinalizePublication(context.Background(), ref, outputs[0].FinalID,
		hub.FinalizePublicationRequest{ManifestID: outputs[0].ManifestID, ManifestLength: outputs[0].ManifestLength}, "read back the ordinary scheduled job checkpoint")
	fatal(t, problem)
	if publication.State != "checkpointed" || publication.CheckpointID != outputs[0].ManifestID {
		t.Fatal("Hub checkpoint readback differs")
	}
	must(t, json.NewEncoder(input).Encode(map[string]string{"action": "verify", "manifest": outputs[0].ManifestID, "transaction": outputs[0].TransactionID, "marker": marker}))
	select {
	case raw := <-results:
		var verified struct {
			Verified              bool
			PayloadBytes          int64  `json:"payload_bytes"`
			HostAttemptsRemaining int    `json:"host_attempts_remaining"`
			HostInvocation        string `json:"host_invocation"`
			ObjectUploads         int    `json:"object_uploads"`
		}
		must(t, json.Unmarshal([]byte(raw), &verified))
		hostInvocation, err := base64.StdEncoding.DecodeString(verified.HostInvocation)
		must(t, err)
		if !bytes.Equal(hostInvocation, attempts[0].InvocationCanonical) {
			t.Fatalf("Host did not receive the exact retained Creator InvocationSpec bytes (Host %d bytes, Creator %d)", len(hostInvocation), len(attempts[0].InvocationCanonical))
		}
		if !verified.Verified || verified.PayloadBytes != (4<<20)+4096+2048 || verified.HostAttemptsRemaining != 0 || verified.ObjectUploads < 2 {
			t.Fatal("real Host/native payload proof failed")
		}
	case <-time.After(20 * time.Second):
		t.Fatal("native checkpoint readback did not return")
	}
	t.Logf("normal CLI -> exact retained InvocationSpec -> scheduled UID65533 -> actual Go Host/native receipt -> Hub checkpoint: request=%s invocation=%s checkpoint=%s", request.ID, attempts[0].InvocationDigest, outputs[0].ManifestID)
	must(t, json.NewEncoder(input).Encode(map[string]string{"action": "stop"}))
}
