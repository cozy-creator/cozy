package producttest

import (
	"bufio"
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
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
	"github.com/cozy-creator/cozy/internal/daemon"
	"github.com/cozy-creator/cozy/internal/home"
	"github.com/cozy-creator/cozy/internal/records"
	"github.com/cozy-creator/cozy/internal/rental"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
	"google.golang.org/protobuf/proto"
)

func proveOperatorSourceCustody(t *testing.T, ctx context.Context, root string, store *records.Store, auth *accountauth.Manager,
	request *pb.PrepareModelSourceRequest, address string, observed records.ModelSourceCheckpoint) {
	t.Helper()
	raw, err := proto.Marshal(request)
	must(t, err)
	input, err := json.Marshal(map[string]string{"Address": address, "Request": base64.StdEncoding.EncodeToString(raw), "Root": root})
	must(t, err)
	setup := filepath.Join(root, "bridge.json")
	must(t, os.WriteFile(setup, input, 0600))
	command := exec.CommandContext(ctx, *sourceCustodyBridge, "-test.run=^TestOperatorSourceCustodyBridge$", "-source-custody-bridge="+setup)
	stdin, err := command.StdinPipe()
	must(t, err)
	stdout, err := command.StdoutPipe()
	must(t, err)
	var diagnostic bytes.Buffer
	command.Stderr = &diagnostic
	must(t, command.Start())
	defer func() {
		_ = stdin.Close()
		if err := command.Wait(); err != nil {
			t.Errorf("Go Host bridge failed: %v %s", err, diagnostic.String())
		}
	}()
	var bridge struct{ Address, Certificate, IdentityPath, WorkerID, WorkerBootID string }
	// Decode snake-case fields explicitly; the only private-key output is a path in this fixture root.
	scanner := bufio.NewScanner(stdout)
	scanner.Buffer(make([]byte, 4096), pb.MaxInlineControlBytes)
	ready := false
	for scanner.Scan() {
		line := scanner.Text()
		if !strings.HasPrefix(line, "BRIDGE ") {
			continue
		}
		var value map[string]string
		must(t, json.Unmarshal([]byte(strings.TrimPrefix(line, "BRIDGE ")), &value))
		bridge.Address = value["address"]
		bridge.Certificate = value["certificate"]
		bridge.IdentityPath = value["identity_path"]
		bridge.WorkerID = value["worker_id"]
		bridge.WorkerBootID = value["worker_boot_id"]
		ready = true
		break
	}
	if !ready {
		t.Fatalf("Go Host did not become ready: %s", diagnostic.String())
	}
	layout, problem := home.Open(root)
	fatal(t, problem)
	const rentalID = "pr-11111111111111111111"
	must(t, os.MkdirAll(layout.Rentals, 0700))
	key, err := os.ReadFile(bridge.IdentityPath)
	must(t, err)
	must(t, os.WriteFile(layout.RentalCreatorIdentity(rentalID), key, 0600))
	identity, problem := rental.CreatorIdentityFor(layout, rentalID)
	fatal(t, problem)
	token, problem := rental.PendingMediaToken(layout, "source-custody-proof")
	fatal(t, problem)
	fatal(t, rental.Attach(layout, store, records.Rental{ID: rentalID, MachineName: "proof", SKU: "cpu", AcceleratorModel: "CPU", HourlyRateUSDMicros: 1, Address: bridge.Address, State: "ready", Hub: *publicationHub, ExpectedWorkerID: bridge.WorkerID, ExpectedWorkerBootID: bridge.WorkerBootID}, bridge.Certificate, token, identity))
	cfg := config.Config{Home: root, HubURL: *publicationHub}
	before, problem := cli.InspectStoredSourceCustody(cfg, request.OperationId, rentalID, bridge.WorkerBootID)
	fatal(t, problem)
	if before.SourcePrepared || before.ControlClaimed {
		t.Fatal("partial native checkpoint was presented as final or claimed before control")
	}
	held, problem := daemon.Hold(layout, "", "")
	fatal(t, problem)
	_, problem = cli.SyncStoredSourceCustody(ctx, cfg, request.OperationId, rentalID, bridge.WorkerBootID, io.Discard, auth, nil)
	if problem == nil {
		t.Fatal("custody helper bypassed a live daemon lock")
	}
	held.Release()
	owner := cli.NewModelTransferOwner(cfg, store, io.Discard, auth)
	defer func() {
		if problem := owner.ReleaseSourceCheckpoints(context.Background(), request.OperationId); problem != nil {
			t.Errorf("exact fixture hold cleanup: %s", problem.ErrName())
		}
	}()
	var log bytes.Buffer
	result, problem := cli.SyncStoredSourceCustody(ctx, cfg, request.OperationId, rentalID, bridge.WorkerBootID, &log, auth, func(held context.Context, _ *cli.SourceCustodyResult) {
		if held.Err() != nil || !daemon.Probe(cfg).Up {
			t.Fatal("acknowledged custody dropped its live control/root hold")
		}
		second, problem := daemon.Hold(layout, "", "")
		if problem == nil {
			second.Release()
			t.Fatal("second daemon entered while acknowledged custody was retained")
		}
	})
	fatal(t, problem)
	if !result.ControlClaimed || result.SnapshotAcknowledged || result.SourcePrepared || len(result.Checkpoints) != 1 || result.Checkpoints[0].Acknowledged == nil || *result.Checkpoints[0].Acknowledged != observed {
		t.Fatalf("operator custody did not preserve honest exact facts: %+v", result)
	}
	if daemon.Probe(cfg).Up {
		t.Fatal("custody helper retained the daemon lock after completion")
	}
	attempts, problem := store.Attempts(request.OperationId)
	fatal(t, problem)
	if len(attempts) != 0 {
		t.Fatal("custody helper created an attempt")
	}
	fmt.Fprintf(stdin, "check\n")
	for scanner.Scan() {
		if strings.HasPrefix(scanner.Text(), "CHECK ") {
			t.Log(scanner.Text())
		}
	}
	t.Logf("actual GoHost/native/R2 custody acknowledged %s; source_prepared=false, control held, snapshot admission closed, zero attempts", observed.HeadID)
}

// Inspection remains read-only while the live daemon owns the root, but a custody
// writer must acquire that same lock before it may touch an operation.
func TestSourceCustodyCannotRunBesideTheDaemon(t *testing.T) {
	root := t.TempDir()
	cfg := config.Config{Home: root, HubURL: "http://127.0.0.1:1"}
	layout, problem := home.Open(root)
	fatal(t, problem)
	held, problem := daemon.Hold(layout, "127.0.0.1:1", "")
	fatal(t, problem)
	defer held.Release()
	_, problem = cli.SyncStoredSourceCustody(context.Background(), cfg, "existing-job", "existing-rental", "existing-boot", io.Discard, nil, nil)
	if problem == nil || !strings.Contains(problem.Message, "another Cozy daemon") {
		t.Fatal("custody writer did not respect the existing owner lock")
	}
	held.Release()
	second, problem := daemon.Hold(layout, "", "")
	fatal(t, problem)
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	command := exec.CommandContext(ctx, cozyBin, "up", "--json", "--full")
	command.Env = append(os.Environ(), "COZY_HOME="+root, "TENSORHUB_URL=http://127.0.0.1:1")
	output, err := command.CombinedOutput()
	if ctx.Err() != nil || err == nil || !bytes.Contains(output, []byte("daemon.operator_owned")) {
		t.Fatalf("cozy up did not refuse the operator's live root: %v %s", err, output)
	}
	second.Release()
}
