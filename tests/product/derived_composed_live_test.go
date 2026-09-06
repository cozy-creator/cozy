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
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/cozy-creator/cozy/internal/accountauth"
	"github.com/cozy-creator/cozy/internal/cli"
	"github.com/cozy-creator/cozy/internal/config"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/orchestrator"
	"github.com/cozy-creator/cozy/internal/records"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
	"google.golang.org/protobuf/proto"
)

var composedBridge = flag.String("derived-composed-bridge", "", "actual Go Host bridge test executable")
var composedLauncher = flag.String("derived-composed-launcher", "", "isolated scheduled Runtime launcher")
var composedObjectHost = flag.String("derived-composed-object-host", "", "explicit R2 host for the real tiny source")

func TestComposedScheduledDerivedCheckpointCustody(t *testing.T) {
	if *composedBridge == "" || *composedLauncher == "" || *composedObjectHost == "" || *publicationHub == "" || *publicationHome == "" || *publicationModel == "" {
		t.Skip("requires explicit actual Host/Runtime and task-owned live publication")
	}
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
		Address, Certificate, IdentityPath, WorkerID, WorkerBootID, Descriptor, Build, Application, Locked, Inventory, Encoding, SignalPath, SourceManifest string
		SourceBytes                                                                                                                                         int64
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
	for name, dest := range map[string]*string{"address": &bridge.Address, "certificate": &bridge.Certificate, "identity_path": &bridge.IdentityPath, "worker_id": &bridge.WorkerID, "worker_boot_id": &bridge.WorkerBootID, "descriptor": &bridge.Descriptor, "build": &bridge.Build, "application": &bridge.Application, "locked": &bridge.Locked, "inventory": &bridge.Inventory, "encoding": &bridge.Encoding, "signal_path": &bridge.SignalPath, "source_manifest": &bridge.SourceManifest} {
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
	configure := func(options *orchestrator.Options) {
		rentalWiring(connection, signer)(options)
		options.Cfg.HubURL = *publicationHub
		options.ModelTransfers = cli.NewModelTransferOwner(options.Cfg, options.Store, options.Log, auth)
		options.ReleaseManagedRental = func(string) (string, *exit.Error) { return "", nil }
		options.RentalPrepareFacts = func(context.Context, *orchestrator.WorkerConnection, *pb.DownloadPackageRef) (orchestrator.PrepareFacts, *exit.Error) {
			return orchestrator.PrepareFacts{Application: bridge.Application, ModelSlotPaths: []string{"produce.models.source"}, LockedRequirements: locked, ImageInventory: inventory}, nil
		}
	}
	owner := hostOwner(t, "derived-composed-"+records.NewID("proof"), configure)
	payload, err := json.Marshal(map[string]string{"encoding": bridge.Encoding, "signal_path": bridge.SignalPath})
	must(t, err)
	request, _, problem := owner.c.Submit(orchestrator.Submission{IdemKey: "composed-derived", Package: "proof/derived-recovery-job", Entrypoint: "produce", PlanID: bridge.Descriptor, Release: "0.1.0", Kind: "job", Org: "paul", Payload: payload, Outputs: []string{"model"}, WeightsOutputs: []orchestrator.WeightsOutput{{OutputID: "model", MimeType: orchestrator.WeightsManifestMime, MaxBytes: 8 << 20}}, Worker: podRental, Rental: true, RentalRequired: true, ProducerParams: []string{"source"}, Models: []orchestrator.ModelRef{{Package: "proof/derived-recovery-job", Slot: "source", Model: "paul/creator-output-b93c35ccecf8c183", Release: "0.0.0-recovery-source.20260906", Lane: "native-source", Manifest: bridge.SourceManifest, ManifestLength: 161, Bytes: bridge.SourceBytes}}, ModelTransfer: &records.ModelTransferIntent{Kind: "model-upload", Destination: *publicationModel, Outputs: []records.ModelTransferOutput{{Name: "model"}}}})
	fatal(t, problem)
	for deadline := time.Now().Add(45 * time.Second); ; {
		rows, problem := owner.store.ModelWeightsProgress(request)
		fatal(t, problem)
		if len(rows) == 1 && rows[0].Acknowledged != nil {
			t.Logf("SAME scheduled UID65533 artifact -> actual Go Host -> actual Creator uploader -> R2 custody acknowledged: bytes=%d", rows[0].Acknowledged.Bytes)
			break
		}
		if time.Now().After(deadline) {
			t.Fatalf("scheduled checkpoint not acknowledged: %s\n%v", tail(filepath.Join(owner.root, "orchestrator.log")), owner.c.Events())
		}
		time.Sleep(100 * time.Millisecond)
	}
	// The next arm will kill/rejoin this same artifact after durable custody.
	must(t, json.NewEncoder(input).Encode(map[string]string{"action": "stop"}))
	for scanner.Scan() {
		fmt.Fprintln(io.Discard, scanner.Text())
	}
	must(t, command.Wait())
}
