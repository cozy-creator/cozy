package producttest

import (
	"bufio"
	"bytes"
	"context"
	"crypto/ed25519"
	"crypto/rand"
	"encoding/base64"
	"encoding/json"
	"flag"
	"fmt"
	"os/exec"
	"path/filepath"
	"regexp"
	"strings"
	"sync"
	"testing"
	"time"

	"google.golang.org/protobuf/proto"

	"github.com/cozy-creator/cozy/internal/accountauth"
	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/cli"
	"github.com/cozy-creator/cozy/internal/config"
	"github.com/cozy-creator/cozy/internal/hub"
	"github.com/cozy-creator/cozy/internal/modeltransfer"
	"github.com/cozy-creator/cozy/internal/orchestrator"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
)

var publicationHub = flag.String("publication-hub", "", "live Tensorhub for the explicit tiny publication proof")
var publicationHome = flag.String("publication-home", "", "existing enrolled home used only for the live proof's account credential")
var publicationPython = flag.String("publication-python", "", "public Runtime 0.2.23/TensorFS 0.3.9 interpreter for the live proof")
var publicationModel = flag.String("publication-model", "", "existing task-owned fixture model reused by the live proof; no new repository is created")

// This explicitly armed test runs Creator's real publication owner and mover,
// Runtime's native receipt/upload/finalization, and a real Tensorhub. The small
// claimed TLS peer supplies transport; Tensorhub's own suite covers its Go host ledger.
func TestOutputPublicationThroughNativeRuntimeAndHub(t *testing.T) {
	if *publicationHub == "" || *publicationHome == "" || *publicationPython == "" || *publicationModel == "" {
		t.Skip("requires -publication-hub, -publication-home, -publication-python and -publication-model; creates and removes one tiny checkpoint")
	}
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	authCfg := config.Config{HubURL: *publicationHub, Home: *publicationHome}
	auth := accountauth.New(authCfg)
	client := hub.New(authCfg, "cozy-product-proof").WithTokenSource(auth)
	account, problem := client.CurrentAccount(ctx)
	fatal(t, problem)
	var nonce [8]byte
	_, err := rand.Read(nonce[:])
	must(t, err)
	destination := *publicationModel
	fixture, problem := hub.ParseRef(destination)
	fatal(t, problem)
	if fixture.Org != account.Name {
		t.Fatal("the fixture model must belong to the selected account")
	}
	_, problem = client.ModelCard(ctx, fixture)
	fatal(t, problem)

	command := exec.CommandContext(ctx, *publicationPython, "testdata/output-publication.py", t.TempDir())
	input, err := command.StdinPipe()
	must(t, err)
	output, err := command.StdoutPipe()
	must(t, err)
	var stderr bytes.Buffer
	command.Stderr = &stderr
	must(t, command.Start())
	var waitOnce sync.Once
	wait := func() { waitOnce.Do(func() { _ = command.Wait() }) }
	defer func() {
		_ = input.Close()
		cancel()
		wait()
	}()
	public, private, err := ed25519.GenerateKey(rand.Reader)
	must(t, err)
	pod := &fakePod{controlKey: public, serve: true, jobReady: true}
	errors := make(chan error, 1)
	finished := make(chan struct{})
	var readOnce sync.Once
	var manifest *pb.Ref
	var transfers, sourceRequests int
	pod.onFrame = func(frame *pb.RecordOwnerFrame, send func(*pb.WorkerFrame) error) (bool, error) {
		switch frame.Msg.(type) {
		case *pb.RecordOwnerFrame_ModelSourceFileRequest, *pb.RecordOwnerFrame_ModelSourcePrepareRequest:
			pod.mu.Lock()
			sourceRequests++
			pod.mu.Unlock()
			return true, fmt.Errorf("output-only publication requested source acquisition")
		case *pb.RecordOwnerFrame_Claim, *pb.RecordOwnerFrame_AttemptOffer, *pb.RecordOwnerFrame_WeightsTransferRequest,
			*pb.RecordOwnerFrame_WeightsFinalizeRequest, *pb.RecordOwnerFrame_OutcomeAck:
		default:
			return false, nil
		}
		if offer := frame.GetAttemptOffer(); offer != nil {
			pod.mu.Lock()
			pod.offers = append(pod.offers, offer)
			pod.mu.Unlock()
		}
		readOnce.Do(func() {
			go func() {
				defer close(finished)
				defer wait()
				scanner := bufio.NewScanner(output)
				scanner.Buffer(make([]byte, 4096), 2*pb.MaxInlineControlBytes)
				for scanner.Scan() {
					var line struct{ Frame string }
					if err := json.Unmarshal(scanner.Bytes(), &line); err != nil {
						errors <- fmt.Errorf("native peer emitted malformed JSON")
						return
					}
					body, err := base64.StdEncoding.DecodeString(line.Frame)
					answer := new(pb.WorkerFrame)
					if err != nil || proto.Unmarshal(body, answer) != nil {
						errors <- fmt.Errorf("native peer emitted malformed protocol bytes")
						return
					}
					pod.mu.Lock()
					if receipt := answer.GetWeightsReceipt(); receipt != nil {
						manifest = proto.Clone(receipt.Manifest).(*pb.Ref)
					}
					if status := answer.GetWeightsTransferStatus(); status != nil {
						transfers++
					}
					pod.mu.Unlock()
					if err := send(answer); err != nil {
						errors <- fmt.Errorf("native peer's claimed stream closed")
						return
					}
				}
				if err := scanner.Err(); err != nil {
					errors <- fmt.Errorf("native peer output was unreadable")
				}
			}()
		})
		body, err := proto.Marshal(frame)
		if err == nil {
			_, err = fmt.Fprintln(input, base64.StdEncoding.EncodeToString(body))
		}
		return frame.GetClaim() == nil, err
	}
	connection, _ := startFakePod(t, t.TempDir(), pod)
	o := hostOwner(t, fmt.Sprintf("output-publication-live-%x", nonce), rentalWiring(connection, private),
		func(options *orchestrator.Options) {
			options.Cfg.HubURL = *publicationHub
			options.ModelTransfers = cli.NewModelTransferOwner(options.Cfg, options.Store, options.Log, auth)
		})
	sub := outputPublicationSubmission()
	sub.Org, sub.ModelTransfer.Destination = account.Name, destination
	fatal(t, modeltransfer.ValidateSubmission(sub))
	requestID, _, problem := o.c.Submit(sub)
	fatal(t, problem)
	for deadline := time.Now().Add(90 * time.Second); ; {
		select {
		case err := <-errors:
			t.Fatal(err)
		default:
		}
		request, problem := o.store.RequestRow(requestID)
		fatal(t, problem)
		if request.State == "succeeded" {
			break
		}
		if request.State == "failed" || time.Now().After(deadline) {
			t.Fatalf("native publication did not settle (%s): %v", request.State, o.c.Events())
		}
		select {
		case <-finished:
			// URLs never reach test output even if a dependency included one in its error.
			text := regexp.MustCompile(`https?://[^\s]+`).ReplaceAllString(stderr.String(), "<url>")
			if text != "" {
				t.Fatalf("native peer exited before settlement: %s", text)
			}
		default:
		}
		time.Sleep(20 * time.Millisecond)
	}
	pod.mu.Lock()
	root, moved, acquired := manifest, transfers, sourceRequests
	pod.mu.Unlock()
	if root == nil || moved == 0 || acquired != 0 {
		t.Fatalf("incomplete publication path: manifest=%v transfers=%d source requests=%d", root, moved, acquired)
	}
	ref, problem := hub.ParseRef(destination)
	fatal(t, problem)
	digest, err := canonical.Spell(root.Digest)
	must(t, err)
	defer func() {
		if problem := client.RemoveCheckpoint(ctx, ref, digest, "remove the tiny product-proof checkpoint"); problem != nil {
			t.Errorf("task checkpoint cleanup failed: %s", problem.Message)
		}
	}()
	transfer, problem := o.store.ModelTransferOf(requestID)
	fatal(t, problem)
	if transfer.State != "completed" || transfer.Checkpoints["model"] != digest {
		t.Fatalf("the existing finalizer did not retain the exact output: %+v", transfer)
	}
	rows, problem := o.store.AllModelTransferWeights(requestID, 1)
	fatal(t, problem)
	if len(rows) != 1 || rows[0].FinalID == "" {
		t.Fatal("publication lost its durable native receipt or final checkpoint identity")
	}
	// Output publication retains an owner-only checkpoint; it does not create
	// a public release. Read back the exact existing publication operation.
	objects := make([]hub.Object, 0, len(rows[0].Objects))
	operation := ""
	for _, object := range rows[0].Objects {
		objects = append(objects, hub.Object{ID: object.ObjectID, Length: object.Length})
		if operation == "" {
			operation = object.OperationID
		}
		if object.OperationID != operation {
			t.Fatal("output objects belong to different publication operations")
		}
	}
	opened, problem := client.OpenPublication(ctx, ref, operation, objects, "read back the tiny product proof")
	fatal(t, problem)
	if opened.Created || opened.Publication.State != "checkpointed" || len(opened.Publication.Objects) != len(objects) {
		t.Fatal("Hub did not retain the exact completed publication")
	}
	for _, object := range opened.Publication.Objects {
		if object.State != "accepted" {
			t.Fatal("Hub did not verify every declared object")
		}
	}
	checkpoint, problem := client.FinalizePublication(ctx, ref, operation,
		hub.FinalizePublicationRequest{ManifestID: digest, ManifestLength: int64(root.Length)},
		"read back the tiny product proof")
	fatal(t, problem)
	if checkpoint.PublishID != rows[0].FinalID || checkpoint.CheckpointID != digest ||
		checkpoint.Manifest.Length != int64(root.Length) || checkpoint.State != "checkpointed" {
		t.Fatal("Hub readback differs from the native committed manifest")
	}
	// The actual catalog resolver still enforces exact input coverage when no
	// acquisition was deferred; adding a destination must not disable that check.
	resolver := cli.NewResolver(o.store, authCfg, nil)
	if _, _, problem := resolver.ResolveRemoteJob("paul/minimax-h3-tools", "2.2.4", "four-lane", nil, false); problem == nil ||
		!strings.Contains(problem.ErrName(), "job_model_selection_incomplete") {
		t.Fatalf("generic model selection was not enforced: %v", problem)
	}
	t.Logf("native receipt, upload, Hub readback and finalization passed: %s %s; owner records %s",
		destination, digest, filepath.Join(o.root, "creator.sqlite"))
}
