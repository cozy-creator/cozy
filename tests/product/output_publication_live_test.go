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
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"google.golang.org/protobuf/proto"

	"github.com/cozy-creator/cozy/internal/accountauth"
	"github.com/cozy-creator/cozy/internal/api"
	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/cli"
	"github.com/cozy-creator/cozy/internal/config"
	"github.com/cozy-creator/cozy/internal/daemon"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/hub"
	"github.com/cozy-creator/cozy/internal/modeltransfer"
	"github.com/cozy-creator/cozy/internal/orchestrator"
	"github.com/cozy-creator/cozy/internal/records"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
	cozyweb "github.com/cozy-creator/cozy/web"
)

var publicationHub = flag.String("publication-hub", "", "live Tensorhub for the explicit tiny publication proof")
var publicationHome = flag.String("publication-home", "", "existing enrolled home used only for the live proof's account credential")
var publicationPython = flag.String("publication-python", "", "public Runtime 0.2.24/TensorFS 0.3.10 interpreter for the live proof")
var publicationCancel = flag.Bool("publication-cancel", false, "prove explicit cancellation of a retained publication after owner restart")
var publicationRetry = flag.Bool("publication-retry", false, "prove failed bound upload, owner restart and explicit same-receipt retry")
var publicationHostBridge = flag.String("publication-host-bridge", "", "compiled real Go Host bridge for the full native publication proof")
var publicationModel = flag.String("publication-model", "", "existing task-owned fixture model reused by the live proof; no new repository is created")

// This explicitly armed test runs Creator's real publication owner and mover,
// Runtime's native receipt/upload/finalization, and a real Tensorhub. The small
// claimed TLS peer supplies transport; Tensorhub's own suite covers its Go host ledger.
func TestOutputPublicationThroughNativeRuntimeAndHub(t *testing.T) {
	if *publicationHub == "" || *publicationHome == "" || *publicationPython == "" || *publicationModel == "" {
		t.Skip("requires -publication-hub, -publication-home, -publication-python and -publication-model; creates and removes one tiny checkpoint")
	}
	if *publicationRetry && *publicationCancel {
		t.Fatal("select retry or cancel")
	}
	lifecycle := *publicationRetry || *publicationCancel
	if lifecycle && *publicationHostBridge == "" {
		t.Fatal("publication retry proof requires the actual Host bridge")
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
	if *publicationHostBridge != "" {
		helper, err := filepath.Abs("testdata/output-publication.py")
		must(t, err)
		command = exec.CommandContext(ctx, *publicationHostBridge,
			"-test.run=^TestWeightsOwnerBridge$", "-weights-owner-python="+*publicationPython,
			"-weights-owner-helper="+helper)
		if lifecycle {
			command.Args = append(command.Args, "-weights-owner-fail-upload-once")
		}
	}
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
	var acknowledgments, releases atomic.Int64
	var restarting, historyComplete atomic.Bool
	var nativeWrite sync.Mutex
	var sendMu sync.Mutex
	var currentSend func(*pb.WorkerFrame) error
	legacyObject := ""
	legacySent := false
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
		nativeWrite.Lock()
		defer nativeWrite.Unlock()
		sendMu.Lock()
		currentSend = send
		sendMu.Unlock()
		if frame.GetClaim() != nil {
			restarting.Store(false)
		}
		if frame.GetOutcomeAck() != nil {
			acknowledgments.Add(1)
		}
		if transfer := frame.GetWeightsTransferRequest(); transfer != nil && lifecycle &&
			transfer.GetUploadGrant() != nil && !historyComplete.Load() {
			if legacyObject == "" {
				legacyObject = transfer.GetUploadGrant().ObjectId
			}
			if transfer.GetUploadGrant().ObjectId == legacyObject {
				if legacySent {
					return true, nil
				}
				legacySent = true
				// Establish a real historical correlation before its controlled failure.
				transfer.OperationId = "prior-bound-object-transfer"
			}
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
					if *publicationHostBridge != "" && scanner.Text() == "PASS" {
						continue
					}
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
					sendMu.Lock()
					target := currentSend
					sendMu.Unlock()
					if err := target(answer); err != nil {
						if lifecycle && restarting.Load() {
							continue
						}
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
	configure := func(options *orchestrator.Options) {
		rentalWiring(connection, private)(options)
		options.Cfg.HubURL = *publicationHub
		options.ModelTransfers = cli.NewModelTransferOwner(options.Cfg, options.Store, options.Log, auth)
		options.ReleaseManagedRental = func(string) (string, *exit.Error) { releases.Add(1); return "", nil }
	}
	o := hostOwner(t, fmt.Sprintf("output-publication-live-%x", nonce), configure)
	sub := outputPublicationSubmission()
	sub.Org, sub.ModelTransfer.Destination = account.Name, destination
	fatal(t, modeltransfer.ValidateSubmission(sub))
	requestID, _, problem := o.c.Submit(sub)
	fatal(t, problem)
	retried := false
	for deadline := time.Now().Add(90 * time.Second); ; {
		select {
		case err := <-errors:
			t.Fatal(err)
		default:
		}
		request, problem := o.store.RequestRow(requestID)
		fatal(t, problem)
		if lifecycle && !retried {
			transfer, problem := o.store.ModelTransferOf(requestID)
			fatal(t, problem)
			if transfer.State == "failed" {
				if request.State != "finalizing" || acknowledgments.Load() != 0 || releases.Load() != 0 {
					t.Fatal("failed publication acknowledged or recycled successful computation")
				}
				prior, problem := o.store.AllModelTransferWeights(requestID, 1)
				fatal(t, problem)
				complete := true
				for _, object := range prior[0].Objects {
					if object.State == "pending" || object.State == "accepted" || object.State == "uploading" {
						complete = false
					}
				}
				if !complete {
					time.Sleep(20 * time.Millisecond)
					continue
				}
				historyComplete.Store(true)
				restarting.Store(true)
				o.close()
				store, problem := records.Open(o.l.DB)
				fatal(t, problem)
				log, err := os.OpenFile(filepath.Join(o.root, "restarted.log"), os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0600)
				must(t, err)
				options := orchestrator.Options{Cfg: o.cfg, Layout: o.l, Store: store, Yield: "smart", Log: log, MaxOutputMiB: 8}
				configure(&options)
				controller, problem := orchestrator.Open(options)
				fatal(t, problem)
				o = &owner{root: o.root, cfg: o.cfg, l: o.l, store: store, c: controller}
				o.closer = func() { controller.Close(20 * time.Second); store.Close(); log.Close() }
				t.Cleanup(o.close)
				go func() { _ = controller.Serve() }()
				fatal(t, controller.ResumeModelTransfers())
				current, problem := store.RequestRow(requestID)
				fatal(t, problem)
				if current.State != "finalizing" || releases.Load() != 0 {
					t.Fatal("restart discarded blocked output custody")
				}
				// The replacement peer's canonical empty snapshot compacts this old
				// attempt. Its unbanked successful publication must remain actionable.
				for until := time.Now().Add(5 * time.Second); ; {
					attempt, problem := store.AttemptRow(requestID, 1)
					fatal(t, problem)
					if attempt.State == "closed" {
						break
					}
					if time.Now().After(until) {
						t.Fatal("replacement snapshot did not close the retained attempt")
					}
					time.Sleep(10 * time.Millisecond)
				}
				stopAPI := publicationControlAPI(t, o)
				defer stopAPI()
				verb := "retry-publication"
				if *publicationCancel {
					verb = "cancel"
				}
				code, output := runCozy(t, o.root, "run", verb, requestID, "--json")
				if code != 0 {
					t.Fatalf("public CLI %s failed [%d]: %s", verb, code, output)
				}
				after, problem := store.AllModelTransferWeights(requestID, 1)
				fatal(t, problem)
				if len(prior) != len(after) || !bytes.Equal(prior[0].Receipt, after[0].Receipt) {
					t.Fatal("retry changed native receipt")
				}
				retried = true
			}
		}
		if request.State == "succeeded" || (*publicationCancel && request.State == "canceled") {
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
	root, moved, acquired, offers := manifest, transfers, sourceRequests, len(pod.offers)
	pod.mu.Unlock()
	if lifecycle && (!retried || acknowledgments.Load() != 1 || releases.Load() != 1) {
		t.Fatal("retry did not settle exactly one original attempt")
	}
	if root == nil || moved < 2 || acquired != 0 || offers != 1 {
		t.Fatalf("incomplete publication path: manifest=%v transfers=%d source requests=%d", root, moved, acquired)
	}
	if *publicationCancel {
		transfer, problem := o.store.ModelTransferOf(requestID)
		fatal(t, problem)
		if transfer.State != "canceled" {
			t.Fatal("cancellation cleanup did not finish")
		}
		cleanupCfg := o.cfg
		cleanupCfg.HubURL = *publicationHub
		fatal(t, cli.NewModelTransferOwner(cleanupCfg, o.store, nil, auth).AbandonModelTransferPublications(ctx, requestID))
		t.Log("real failed publication retained through owner restart, then explicitly canceled and cleaned")
		return
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
	operation := rows[0].FinalID
	for _, object := range rows[0].Objects {
		objects = append(objects, hub.Object{ID: object.ObjectID, Length: object.Length})
		expectedOperation := "weights-" + strings.TrimPrefix(object.ObjectID, "sha256:")
		if *publicationRetry && object.ObjectID == legacyObject {
			expectedOperation = "prior-bound-object-transfer"
			if object.GrantRevision < 2 {
				t.Fatal("bound object was not regranted")
			}
		}
		if object.OperationID != expectedOperation || object.OperationID == operation {
			t.Fatal("per-object transfer correlation was conflated with the Hub publication")
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

// Expose the existing owner through the ordinary authenticated API/daemon record so
// the built CLI exercises the actual public recovery control, including closed attempts.
func publicationControlAPI(t *testing.T, o *owner) func() {
	t.Helper()
	v4, v6, addr, problem := api.Listeners(0)
	fatal(t, problem)
	if v6 != nil {
		_ = v6.Close()
	}
	held, problem := daemon.Hold(o.l, addr, "")
	fatal(t, problem)
	creds, problem := api.Mint(o.l)
	fatal(t, problem)
	handler, problem := api.New(api.Options{Orchestrator: o.c, Cfg: o.cfg, Creds: creds, Addr: addr, Web: cozyweb.Handler()}).Handler()
	fatal(t, problem)
	server := &http.Server{Handler: handler, ReadHeaderTimeout: 5 * time.Second}
	go func() { _ = server.Serve(v4) }()
	return func() { _ = server.Close(); held.Release() }
}
