package producttest

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"net"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/protobuf/proto"

	"github.com/cozy-creator/cozy/internal/accountauth"
	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/cli"
	"github.com/cozy-creator/cozy/internal/config"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/hub"
	"github.com/cozy-creator/cozy/internal/orchestrator"
	"github.com/cozy-creator/cozy/internal/records"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
)

var sourceCheckpointConcurrent = flag.Bool("source-checkpoint-concurrency", false, "exercise four actual native PUT/GET bodies through a controlled R2 proxy")

var sourceCheckpointHelper = flag.String("source-checkpoint-helper", "testdata/public_source.py", "native partial-H3 source fixture and actual RuntimePreparation gRPC service")

// The source owner is the production CLI implementation, including link traversal,
// Hub grants, verification and durable acknowledgment. Its callbacks cross actual
// public Runtime gRPC handlers into TensorFS and R2; no callback invents a byte verdict.
func TestSourceCheckpointThroughPublicRuntimeAndHub(t *testing.T) {
	if *sourceCheckpointHelper == "" || *publicationHub == "" || *publicationHome == "" ||
		*publicationPython == "" || *publicationModel == "" {
		t.Skip("requires source-checkpoint-helper and publication hub/home/python/existing-model flags")
	}
	ctx, cancel := context.WithTimeout(context.Background(), 90*time.Second)
	defer cancel()
	root := t.TempDir()
	authCfg := config.Config{HubURL: *publicationHub, Home: *publicationHome}
	auth := accountauth.New(authCfg)
	client := hub.New(authCfg, "cozy-source-proof").WithTokenSource(auth)
	account, problem := client.CurrentAccount(ctx)
	fatal(t, problem)
	fixture, problem := hub.ParseRef(*publicationModel)
	fatal(t, problem)
	if fixture.Org != account.Name {
		t.Fatal("source checkpoint fixture must belong to the selected account")
	}
	_, problem = client.ModelCard(ctx, fixture)
	fatal(t, problem)

	command := exec.CommandContext(ctx, *publicationPython, *sourceCheckpointHelper, filepath.Join(root, "native"))
	if *sourceCheckpointConcurrent {
		command.Args = append(command.Args, "--transfer-proof")
	}
	command.Env = []string{"PATH=/usr/local/bin:/usr/bin:/bin", "PYTHONNOUSERSITE=1"}
	input, err := command.StdinPipe()
	must(t, err)
	output, err := command.StdoutPipe()
	must(t, err)
	command.Stderr = io.Discard // dependency errors must not print capabilities
	must(t, command.Start())
	defer func() { _ = input.Close(); cancel(); _ = command.Wait() }()
	decoder := json.NewDecoder(io.LimitReader(output, 2*pb.MaxInlineControlBytes))
	var initial struct {
		Request  string
		Software struct{ Runtime, Tensorfs, Commit string }
	}
	must(t, decoder.Decode(&initial))
	if initial.Software.Runtime != "0.2.23" || initial.Software.Tensorfs != "0.3.9" {
		t.Fatal("source proof did not select the expected public Runtime/TensorFS pair")
	}
	raw, err := base64.StdEncoding.DecodeString(initial.Request)
	must(t, err)
	preparedRequest := new(pb.PrepareModelSourceRequest)
	must(t, proto.Unmarshal(raw, preparedRequest))
	selection, err := canonical.Spell(preparedRequest.SourceSelectionDigest)
	must(t, err)
	intent := &records.ModelTransferIntent{Kind: "model-upload", Destination: fixture.String(),
		Source: "hf://cozy-proof/tiny-h3@" + strings.Repeat("0", 40), SourceSelection: selection,
		SourceProfiles: map[string]string{}, Outputs: []records.ModelTransferOutput{{Name: "model"}}}
	for _, profile := range preparedRequest.Profiles {
		intent.SourceProfiles[profile.Slot] = profile.Profile
	}
	for _, file := range preparedRequest.Files {
		header, err := os.ReadFile(file.HeaderPath)
		must(t, err)
		intent.SourceFiles = append(intent.SourceFiles, records.ModelTransferSourceFile{
			Member: file.Member, SHA256: strings.TrimPrefix(file.ObjectId, "sha256:"),
			Length: int64(file.Length), Header: header})
	}
	store, problem := records.Open(filepath.Join(root, "creator.sqlite"))
	fatal(t, problem)
	defer store.Close()
	requestID := records.NewID("job")
	_, fresh, problem := store.Submit(records.Request{ID: requestID, IdemKey: requestID,
		BodyDigest: "sha256:" + strings.Repeat("c", 64), Package: "proof/source-checkpoint",
		Entrypoint: "prepare", State: "queued", Kind: "job", Payload: []byte("{}"),
		Outputs: "[]", WeightsOutputs: "[]", ModelTransfer: intent})
	fatal(t, problem)
	if !fresh {
		t.Fatal("source proof request was not newly recorded")
	}
	fatal(t, store.BeginModelTransferMaterialization(requestID))
	const boot = "source-proof-boot"
	for _, file := range preparedRequest.Files {
		fatal(t, store.RecordModelTransferSourceStatus(records.ModelTransferSourceStatus{
			RequestID: requestID, Member: file.Member, ObjectID: file.ObjectId, WorkerBootID: boot,
			Length: int64(file.Length), CapabilityRevision: 1, State: "accepted"}))
	}
	_, err = fmt.Fprintf(input, "{\"operation_id\":%q}\n", requestID)
	must(t, err)
	var started struct {
		Address, Prepared string
		RestoreAddress    string `json:"restore_address"`
	}
	must(t, decoder.Decode(&started))
	hostname, _, err := net.SplitHostPort(started.Address)
	must(t, err)
	if hostname != "127.0.0.1" {
		t.Fatal("native proof did not bind its private loopback service")
	}
	raw, err = base64.StdEncoding.DecodeString(started.Prepared)
	must(t, err)
	prepared := new(pb.PrepareModelSourceResult)
	must(t, proto.Unmarshal(raw, prepared))
	if prepared.Outcome != pb.ModelSourcePrepareOutcome_MODEL_SOURCE_PREPARE_OUTCOME_INCOMPLETE || len(prepared.Checkpoints) != 1 {
		t.Fatal("native preparation did not return one actual partial checkpoint")
	}
	checkpoint := prepared.Checkpoints[0]
	head, err := canonical.Spell(checkpoint.Head.Digest)
	must(t, err)
	plan, err := canonical.Spell(checkpoint.PlanDigest)
	must(t, err)
	observed := records.ModelSourceCheckpoint{Slot: checkpoint.Slot, HeadID: head,
		HeadLength: int64(checkpoint.Head.Length), PlanDigest: plan, Index: int64(checkpoint.Index), Bytes: int64(checkpoint.Bytes)}
	fatal(t, store.ObserveModelSourceCheckpoints(requestID, selection, boot, []records.ModelSourceCheckpoint{observed}))
	connection, err := grpc.NewClient(started.Address, grpc.WithTransportCredentials(insecure.NewCredentials()))
	must(t, err)
	defer connection.Close()
	runtime := pb.NewRuntimePreparationClient(connection)
	var pages, transfers atomic.Int64
	var gate *checkpointHTTPGate
	if *sourceCheckpointConcurrent {
		gate = newCheckpointHTTPGate(t, func() {
			progress, problem := store.ModelSourceProgress(requestID)
			if problem != nil || len(progress) != 1 || progress[0].Acknowledged != nil {
				t.Error("source custody was acknowledged before native HTTP bodies completed")
			}
		})
	}
	host := orchestrator.SourceCheckpointHost{BootID: boot,
		Page: func(ctx context.Context, request *pb.SourceCheckpointPageRequest) (*pb.SourceCheckpointPageResult, *exit.Error) {
			request.RecordOwnerEpoch, request.WorkerBootId = 1, boot
			answer, err := runtime.SourceCheckpointPage(ctx, request)
			if err != nil {
				return nil, exit.Unavailablef("native checkpoint page RPC failed")
			}
			if answer.SafeCode != "" {
				return nil, exit.Named(exit.Failed, answer.SafeCode, "native checkpoint page refused")
			}
			pages.Add(1)
			return answer, nil
		},
		Transfer: func(ctx context.Context, request *pb.SourceCheckpointTransferRequest) (*pb.SourceCheckpointTransferStatus, *exit.Error) {
			request.RecordOwnerEpoch, request.WorkerBootId = 1, boot
			if gate != nil {
				if grant := request.GetUploadGrant(); grant != nil {
					grant.Url = gate.route(http.MethodPut, grant.ObjectId, grant.Url)
				} else {
					id, _ := canonical.Spell(request.Object.Ref.Digest)
					request.Decision = &pb.SourceCheckpointTransferRequest_DownloadUrl{DownloadUrl: gate.route(http.MethodGet, id, request.GetDownloadUrl())}
				}
			}
			answer, err := runtime.SourceCheckpointTransfer(ctx, request)
			if err != nil {
				return nil, exit.Unavailablef("native checkpoint transfer RPC failed")
			}
			if answer.State != pb.WeightsTransferState_WEIGHTS_TRANSFER_STATE_UPLOADED &&
				answer.State != pb.WeightsTransferState_WEIGHTS_TRANSFER_STATE_ALREADY_PRESENT && answer.State != pb.WeightsTransferState_WEIGHTS_TRANSFER_STATE_HELD {
				return nil, exit.Named(exit.Unavailable, answer.SafeCode, "native checkpoint upload refused (%s)", answer.SafeCode)
			}
			if answer.TransferId != request.TransferId || !bytes.Equal(answer.Head.Digest, request.Head.Digest) {
				return nil, exit.New(exit.Conflict, "native checkpoint transfer identity changed")
			}
			transfers.Add(1)
			return answer, nil
		}}
	owner := cli.NewModelTransferOwner(config.Config{HubURL: *publicationHub, Home: root}, store, io.Discard, auth)
	defer func() {
		cleanup, stop := context.WithTimeout(context.Background(), 30*time.Second)
		defer stop()
		if problem := owner.ReleaseSourceCheckpoints(cleanup, requestID); problem != nil {
			t.Errorf("exact source publication cleanup failed: %s", problem.ErrName())
		}
	}()
	if gate != nil {
		denied := owner.SyncSourceCheckpoints(ctx, requestID, host)
		if denied == nil || denied.Code != exit.Unavailable {
			t.Fatalf("native denied PUT lost its typed failure: %v", denied)
		}
		failed, problem := store.ModelSourceProgress(requestID)
		fatal(t, problem)
		if len(failed) != 1 || failed[0].Acknowledged != nil {
			t.Fatal("failed Link advanced its custody acknowledgment")
		}
		held, problem := store.SourcePublications(requestID)
		fatal(t, problem)
		if len(held) == 0 {
			t.Fatal("failed Link dropped its recovery holds")
		}
		gate.mu.Lock()
		gate.refuse = false
		gate.mu.Unlock()
	}
	fatal(t, owner.SyncSourceCheckpoints(ctx, requestID, host))
	progress, problem := store.ModelSourceProgress(requestID)
	fatal(t, problem)
	if pages.Load() == 0 || transfers.Load() == 0 || len(progress) != 1 || progress[0].Acknowledged == nil ||
		*progress[0].Acknowledged != observed {
		t.Fatal("native source upload did not reach exact durable owner acknowledgment")
	}
	if gate != nil {
		gate.mu.Lock()
		peak, moved := gate.peak, gate.moved
		gate.mu.Unlock()
		if peak != 4 || moved == 0 {
			t.Fatalf("native PUT concurrency/bytes: peak=%d moved=%d", peak, moved)
		}
		restored, err := grpc.NewClient(started.RestoreAddress, grpc.WithTransportCredentials(insecure.NewCredentials()))
		must(t, err)
		defer restored.Close()
		runtime = pb.NewRuntimePreparationClient(restored)
		fatal(t, owner.RestoreSourceCheckpoints(ctx, requestID, host))
		gate.mu.Lock()
		getPeak := gate.getPeak
		gate.mu.Unlock()
		if getPeak != 4 {
			t.Fatalf("native GET concurrency: peak=%d", getPeak)
		}
		t.Logf("actual native network overlap: PUT peak=%d, GET peak=%d, uploaded body bytes=%d; empty-Store head-first restore passed", peak, getPeak, moved)
	}

	held, problem := store.SourcePublications(requestID)
	fatal(t, problem)
	if len(held) == 0 {
		t.Fatal("source acknowledgment has no retained Hub publication")
	}
	t.Logf("public Runtime %s/TensorFS %s source checkpoint: %d native pages, %d successful native transfers, %d verified Hub holds, exact head %s acknowledged",
		initial.Software.Runtime, initial.Software.Tensorfs, pages.Load(), transfers.Load(), len(held), head)
}
