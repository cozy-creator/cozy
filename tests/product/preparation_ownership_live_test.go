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

	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/orchestrator"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
	"google.golang.org/protobuf/proto"
)

var preparationOwnershipBridge = flag.String("preparation-ownership-bridge", "", "actual task-built Host test binary")
var preparationOwnershipPython = flag.String("preparation-ownership-python", "", "actual isolated Runtime/native interpreter")
var preparationOwnershipFixture = flag.String("preparation-ownership-fixture", "", "actual Runtime ownership fixture script")

var preparationOwnershipUV = flag.String("preparation-ownership-uv", "", "uv for the tiny fixture wheel")

// The production Creator orchestrator drives the production Go Host Plane and
// actual Runtime Worker over their real TLS services. The fixture injects native
// GC between prepare and desired, retaining an independent native output root.
func TestActualPreparationOwnershipReacquiresAfterNativeGC(t *testing.T) {
	proveActualPreparationOwnership(t, false, false)
}

func TestActualJobOwnershipReacquiresAfterNativeGC(t *testing.T) {
	for _, repeated := range []bool{false, true} {
		t.Run(map[bool]string{false: "recovers", true: "bounded-repeat-absence"}[repeated], func(t *testing.T) {
			proveActualPreparationOwnership(t, true, repeated)
		})
	}
}

func proveActualPreparationOwnership(t *testing.T, job, repeated bool) {
	if *preparationOwnershipBridge == "" || *preparationOwnershipPython == "" || *preparationOwnershipUV == "" {
		t.Skip("requires explicit task-built actual peers")
	}
	command := exec.Command(*preparationOwnershipBridge, "-test.run=^TestPreparationOwnershipBridge$", "-test.v", "-preparation-bridge-python="+*preparationOwnershipPython, "-preparation-bridge-uv="+*preparationOwnershipUV, "-preparation-bridge-fixture="+*preparationOwnershipFixture, fmt.Sprintf("-preparation-bridge-job=%t", job), fmt.Sprintf("-preparation-bridge-repeat-eviction=%t", repeated))
	input, err := command.StdinPipe()
	must(t, err)
	output, err := command.StdoutPipe()
	must(t, err)
	command.Stderr = os.Stderr
	must(t, command.Start())
	defer func() {
		fmt.Fprintln(input, "stop")
		input.Close()
		if command.ProcessState == nil {
			done := make(chan error, 1)
			go func() {
				_, _ = io.Copy(os.Stderr, output)
				done <- command.Wait()
			}()
			select {
			case err := <-done:
				if err != nil {
					t.Errorf("ownership bridge shutdown: %v", err)
				}
			case <-time.After(20 * time.Second):
				command.Process.Kill()
				<-done
				t.Error("ownership bridge did not stop its owned Runtime")
			}
		}
	}()
	scanner := bufio.NewScanner(output)
	scanner.Buffer(make([]byte, 4096), 1<<20)
	read := func(prefix string) map[string]json.RawMessage {
		t.Helper()
		for scanner.Scan() {
			line := scanner.Text()
			if !strings.HasPrefix(line, prefix) {
				t.Log(line)
				continue
			}
			var raw map[string]json.RawMessage
			must(t, json.Unmarshal([]byte(strings.TrimPrefix(line, prefix)), &raw))
			return raw
		}
		t.Fatal("actual bridge ended before " + prefix)
		return nil
	}
	raw := read("BRIDGE ")
	var address, certificateText, identityPath, workerID, bootID, model, lockedText, inventoryText string
	for field, destination := range map[string]*string{"address": &address, "certificate": &certificateText, "identity_path": &identityPath, "worker_id": &workerID, "worker_boot_id": &bootID, "model": &model, "locked": &lockedText, "inventory": &inventoryText} {
		must(t, json.Unmarshal(raw[field], destination))
	}
	root := t.TempDir()
	certificate := filepath.Join(root, "host.pem")
	must(t, os.WriteFile(certificate, []byte(certificateText), 0600))
	key, err := os.ReadFile(identityPath)
	must(t, err)
	block, _ := pem.Decode(key)
	if block == nil {
		t.Fatal("no fixture identity")
	}
	parsed, err := x509.ParsePKCS8PrivateKey(block.Bytes)
	must(t, err)
	signer, ok := parsed.(ed25519.PrivateKey)
	if !ok {
		t.Fatal("wrong fixture key")
	}
	connection, _ := startFakePod(t, root, &fakePod{})
	connection.Addr, connection.CACert, connection.WorkerID, connection.WorkerBootID = address, certificate, workerID, bootID
	locked, err := base64.StdEncoding.DecodeString(lockedText)
	must(t, err)
	inventoryBytes, err := base64.StdEncoding.DecodeString(inventoryText)
	must(t, err)
	inventory := new(pb.ImageInventory)
	must(t, proto.Unmarshal(inventoryBytes, inventory))
	binding := "generate.models.model"
	if job {
		binding = "inspect.models.model"
	}
	o := hostOwner(t, fmt.Sprintf("native-gc-preparation-job%t-repeat%t", job, repeated), rentalWiring(connection, signer), func(options *orchestrator.Options) {
		options.RentalPrepareFacts = func(_ context.Context, _ *orchestrator.WorkerConnection, _ *pb.DownloadPackageRef) (orchestrator.PrepareFacts, *exit.Error) {
			return orchestrator.PrepareFacts{Application: "preparation_proof:app", ModelSlotPaths: []string{binding}, LockedRequirements: locked, ImageInventory: inventory}, nil
		}
	})
	// Close the owner before waiting for the Host's graceful gRPC shutdown;
	// otherwise each side waits for the other's still-open control stream.
	defer o.close()
	var instance, requestID string
	if job {
		var descriptor string
		var length, modelBytes int64
		must(t, json.Unmarshal(raw["descriptor"], &descriptor))
		must(t, json.Unmarshal(raw["model_length"], &length))
		must(t, json.Unmarshal(raw["model_bytes"], &modelBytes))
		var problem *exit.Error
		requestID, _, problem = o.c.Submit(orchestrator.Submission{
			IdemKey: "actual-job-native-reensure", Package: "cozy/preparation-proof", Release: "1.0.0", Entrypoint: "inspect", PlanID: descriptor,
			Kind: "job", Org: "cozy", Payload: []byte(`{"prompt":"37"}`), Worker: podRental, Rental: true, RentalRequired: true, ProducerParams: []string{"model"},
			Models: []orchestrator.ModelRef{{Package: "cozy/preparation-proof", Slot: "model", BindingPath: binding, Model: "proof/input", Manifest: model, ManifestLength: length, Bytes: modelBytes}},
		})
		fatal(t, problem)
	} else {
		var problem *exit.Error
		instance, _, _, problem = o.c.EnsureRental(podRental)
		fatal(t, problem)
		fatal(t, o.c.ConvergePackageSet(instance, []*pb.DownloadPackageRef{{Package: "cozy/preparation-proof", Release: "1.0.0"}}, []*pb.DownloadModelRef{{Package: "cozy/preparation-proof", Slot: binding, Model: "proof/input", Manifest: model}}))
	}
	deadline := time.NewTimer(3 * time.Minute)
	defer deadline.Stop()
	tick := time.NewTicker(50 * time.Millisecond)
	defer tick.Stop()
	for {
		if job {
			row, problem := o.store.RequestRow(requestID)
			fatal(t, problem)
			if row != nil && (row.State == "succeeded" || row.State == "failed") {
				break
			}
		} else {
			facts := o.c.Worker(instance)
			if facts != nil && facts.AcceptedRevision == 2 && facts.DesiredRevision == 2 && facts.Materialization == "STAGED" {
				break
			}
		}
		select {
		case <-deadline.C:
			log, _ := os.ReadFile(filepath.Join(o.root, "orchestrator.log"))
			t.Fatalf("actual Runtime did not reacquire after native GC:\n%s", log)
		case <-tick.C:
		}
	}
	if job {
		rows, problem := o.store.Attempts(requestID)
		fatal(t, problem)
		want := 2
		if repeated {
			want = orchestrator.MaxRequeues + 1
		}
		if len(rows) != want {
			for _, row := range rows {
				t.Logf("attempt %d: %s/%s %s", row.Attempt, row.TerminalStatus, row.TerminalCause, row.SafeMessage)
			}
			t.Fatalf("expected %d real attempts under one request, got %d", want, len(rows))
		}
		for index, row := range rows {
			if row.RequestID != requestID || row.Attempt != int64(index+1) {
				t.Fatal("retry replaced the logical request or its ordinal")
			}
			if repeated || index == 0 {
				if row.TerminalStatus != "REFUSED" || row.TerminalCause != "PLACEMENT_NOT_DISPATCHABLE" || !strings.HasPrefix(row.SafeMessage, "model_materialization_required:") {
					t.Fatalf("unexpected real pre-execution refusal: %s/%s %s", row.TerminalStatus, row.TerminalCause, row.SafeMessage)
				}
			} else {
				var body struct {
					ExecutionStarted bool `json:"execution_started"`
					Result           struct {
						Inline []byte `json:"inline_result"`
					} `json:"result"`
				}
				must(t, json.Unmarshal(row.TerminalBody, &body))
				var result struct {
					Value, Elements int
					Checkpoint      string
					RequestID       string `json:"request_id"`
				}
				must(t, json.Unmarshal(body.Result.Inline, &result))
				if row.TerminalStatus != "SUCCEEDED" || !body.ExecutionStarted || result.Value != 37 || result.Elements != 1 || result.Checkpoint != model || result.RequestID != requestID {
					t.Fatalf("actual model-reader job result disagrees: %+v status=%s", result, row.TerminalStatus)
				}
			}
		}
	}
	fmt.Fprintln(input, "verify")
	verified := read("RESULT ")
	var retained, downloaded bool
	must(t, json.Unmarshal(verified["retained_ok"], &retained))
	must(t, json.Unmarshal(verified["downloaded_ok"], &downloaded))
	var ensures []struct {
		Moved int64 `json:"moved"`
		Total int64 `json:"total"`
	}
	must(t, json.Unmarshal(verified["ensures"], &ensures))
	wantEnsures := 2
	if repeated {
		wantEnsures = orchestrator.MaxRequeues + 1
	}
	if !retained || downloaded == repeated || len(ensures) != wantEnsures || ensures[1].Moved <= 0 {
		t.Fatalf("incomplete real reacquisition: %v %v %+v", retained, downloaded, ensures)
	}
	t.Logf("real Host/Runtime/Creator recovered %d B after native GC; independent output retained", ensures[1].Moved)
}
