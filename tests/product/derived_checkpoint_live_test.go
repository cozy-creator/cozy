package producttest

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"io"
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
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/protobuf/proto"
)

// Real shared owner uploader -> actual Runtime checkpoint RPC -> native signed PUT/GET
// -> actual Hub custody, followed by process death and an empty replacement Store.
func TestDerivedCheckpointCustodyRestoresOnlyCompletedRoles(t *testing.T) {
	if *publicationHub == "" || *publicationHome == "" || *publicationPython == "" || *publicationModel == "" {
		t.Skip("requires explicit existing task model/hub/home/native Python")
	}
	ctx, cancel := context.WithTimeout(t.Context(), 120*time.Second)
	defer cancel()
	root := t.TempDir()
	authCfg := config.Config{HubURL: *publicationHub, Home: *publicationHome}
	auth := accountauth.New(authCfg)
	client := hub.New(authCfg, "derived-checkpoint-proof").WithTokenSource(auth)
	account, problem := client.CurrentAccount(ctx)
	fatal(t, problem)
	ref, problem := hub.ParseRef(*publicationModel)
	fatal(t, problem)
	if ref.Org != account.Name {
		t.Fatal("checkpoint fixture is not owned by current account")
	}
	_, problem = client.ModelCard(ctx, ref)
	fatal(t, problem)
	db := filepath.Join(root, "creator.sqlite")
	store, problem := records.Open(db)
	fatal(t, problem)
	defer func() { store.Close() }()
	request := records.NewID("job")
	digest := "sha256:" + strings.Repeat("a", 64)
	outputs := `[{"output_id":"model","mime_type":"application/vnd.cozy.model-manifest","max_bytes":8388608}]`
	_, _, problem = store.Submit(records.Request{ID: request, IdemKey: request, BodyDigest: digest, Package: "test/derive", Entrypoint: "produce", State: "queued", Kind: "job", Payload: []byte("{}"), Outputs: "[]", WeightsOutputs: outputs,
		ModelTransfer: &records.ModelTransferIntent{Kind: "model-upload", Destination: ref.String(), Outputs: []records.ModelTransferOutput{{Name: "model"}}}})
	fatal(t, problem)
	const instance, session = "derived-proof-worker", "derived-proof-session"
	fatal(t, store.SpawnWorker(records.WorkerProcess{InstanceID: instance, Package: "test/derive", WorkerID: "remote", Devices: []string{"cpu"}}))
	dispatch := func() int64 {
		ordinal, problem := store.Dispatch(records.Attempt{RequestID: request, InstanceID: instance, SessionID: session, InvocationDigest: digest, InvocationCanonical: []byte("{}"), WeightsOutputs: outputs})
		fatal(t, problem)
		fatal(t, store.OfferDispatch(request, ordinal, session))
		fatal(t, store.Accepted(request, ordinal, session))
		return ordinal
	}
	ordinal := dispatch()
	owner := cli.NewModelTransferOwner(config.Config{HubURL: *publicationHub, Home: root}, store, io.Discard, auth)
	defer func() {
		if problem := owner.ReleaseCheckpoints(context.Background(), request); problem != nil {
			t.Errorf("task checkpoint publication cleanup refused: %s", problem.ErrName())
		}
	}()
	start := func(phase string) (*exec.Cmd, io.WriteCloser, *json.Decoder, *pb.WeightsTransactionStatus, orchestrator.CheckpointHost, *atomic.Int64) {
		command := exec.CommandContext(ctx, *publicationPython, "testdata/derived-checkpoint.py", filepath.Join(root, phase), request, phase)
		command.Env = []string{"PATH=/usr/local/bin:/usr/bin:/bin", "PYTHONNOUSERSITE=1"}
		input, err := command.StdinPipe()
		must(t, err)
		output, err := command.StdoutPipe()
		must(t, err)
		command.Stderr = io.Discard
		must(t, command.Start())
		t.Cleanup(func() {
			input.Close()
			if command.ProcessState == nil {
				command.Process.Kill()
				command.Wait()
			}
		})
		decoder := json.NewDecoder(output)
		var initial struct{ Address, Status string }
		must(t, decoder.Decode(&initial))
		raw, err := base64.StdEncoding.DecodeString(initial.Status)
		must(t, err)
		status := new(pb.WeightsTransactionStatus)
		must(t, proto.Unmarshal(raw, status))
		conn, err := grpc.NewClient(initial.Address, grpc.WithTransportCredentials(insecure.NewCredentials()))
		must(t, err)
		t.Cleanup(func() { conn.Close() })
		runtime := pb.NewRuntimePreparationClient(conn)
		transferred := new(atomic.Int64)
		host := orchestrator.CheckpointHost{BootID: phase,
			Page: func(ctx context.Context, request *pb.CheckpointPageRequest) (*pb.CheckpointPageResult, *exit.Error) {
				answer, err := runtime.CheckpointPage(ctx, request)
				if err != nil {
					return nil, exit.Unavailablef("native checkpoint page transport failed")
				}
				if answer.SafeCode != "" {
					return nil, exit.Named(exit.Failed, answer.SafeCode, "native checkpoint page refused")
				}
				return answer, nil
			},
			Transfer: func(ctx context.Context, request *pb.CheckpointTransferRequest) (*pb.CheckpointTransferStatus, *exit.Error) {
				answer, err := runtime.CheckpointTransfer(ctx, request)
				if err != nil {
					return nil, exit.Unavailablef("native checkpoint transfer transport failed")
				}
				if answer.State != pb.WeightsTransferState_WEIGHTS_TRANSFER_STATE_UPLOADED && answer.State != pb.WeightsTransferState_WEIGHTS_TRANSFER_STATE_ALREADY_PRESENT && answer.State != pb.WeightsTransferState_WEIGHTS_TRANSFER_STATE_HELD {
					return nil, exit.Named(exit.Failed, answer.SafeCode, "native checkpoint transfer refused")
				}
				if request.GetUploadGrant() != nil {
					transferred.Add(int64(answer.TransferredBytes))
				}
				return answer, nil
			}}
		return command, input, decoder, status, host, transferred
	}
	initial, _, _, status, host, _ := start("initial")
	fatal(t, store.ObserveModelWeightsCheckpoint(instance, host.BootID, status))
	rows, problem := store.ModelWeightsProgress(request)
	fatal(t, problem)
	if len(rows) != 1 || rows[0].Acknowledged != nil {
		t.Fatal("local role claimed remote custody")
	}
	fatal(t, owner.SyncCheckpoints(ctx, request, host))
	rows, problem = store.ModelWeightsProgress(request)
	fatal(t, problem)
	if rows[0].Acknowledged == nil || *rows[0].Acknowledged != rows[0].Observed {
		t.Fatal("complete native Link was not durably acknowledged")
	}
	banked := *rows[0].Acknowledged
	// Abrupt process death occurs inside the native scale-role reader, after data custody.
	must(t, initial.Process.Kill())
	if err := initial.Wait(); err == nil {
		t.Fatal("initial process did not die")
	}
	store.Close()
	store, problem = records.Open(db)
	fatal(t, problem)
	owner = cli.NewModelTransferOwner(config.Config{HubURL: *publicationHub, Home: root}, store, io.Discard, auth)
	_, problem = store.AcceptTerminal(records.Terminal{RequestID: request, Attempt: ordinal, SessionID: session, InvocationDigest: digest, TerminalID: "process-lost", TerminalDigest: "sha256:" + strings.Repeat("e", 64), Status: "FAILED", Cause: "worker_lost", RequestState: "requeue_pending"})
	fatal(t, problem)
	fatal(t, store.Closed(request, ordinal))
	_, _, _, problem = store.BeginRequeue(request, 2, false)
	fatal(t, problem)
	ordinal = dispatch()
	replacement, input, decoder, status, host, moved := start("replacement")
	fatal(t, store.ObserveModelWeightsCheckpoint(instance, host.BootID, status))
	rows, problem = store.ModelWeightsProgress(request)
	fatal(t, problem)
	if rows[0].Observed.HeadID != "" || rows[0].Acknowledged == nil || *rows[0].Acknowledged != banked {
		t.Fatal("replacement lost banked role or claimed unfinished work")
	}
	checkpoint, problem := owner.RestoreWeightsCheckpoint(ctx, request, host, rows[0].Subject)
	fatal(t, problem)
	raw, err := proto.Marshal(checkpoint)
	must(t, err)
	must(t, json.NewEncoder(input).Encode(map[string]string{"action": "resume", "checkpoint": base64.StdEncoding.EncodeToString(raw)}))
	var resumed struct {
		Status string
		Reused int64 `json:"reused_bytes"`
		New    int64 `json:"new_role_bytes"`
	}
	must(t, decoder.Decode(&resumed))
	raw, err = base64.StdEncoding.DecodeString(resumed.Status)
	must(t, err)
	must(t, proto.Unmarshal(raw, status))
	if resumed.Reused != 4<<20 || resumed.New != 4096 {
		t.Fatal("native role completion was not reused exactly")
	}
	fatal(t, store.ObserveModelWeightsCheckpoint(instance, host.BootID, status))
	fatal(t, owner.SyncCheckpoints(ctx, request, host))
	firstMoved := moved.Load()
	if firstMoved >= 4<<20 {
		t.Fatal("completed data payload was reuploaded")
	}
	fatal(t, owner.SyncCheckpoints(ctx, request, host))
	if moved.Load() != firstMoved {
		t.Fatal("warm custody replay moved bytes")
	}
	must(t, json.NewEncoder(input).Encode(map[string]string{"action": "commit"}))
	var committed struct {
		Committed bool
		Verified  int64  `json:"verified_bytes"`
		Error     string `json:"proof_error"`
		Line      int
	}
	must(t, decoder.Decode(&committed))
	if committed.Error != "" {
		t.Fatalf("native fixture failed: %s line%d", committed.Error, committed.Line)
	}
	must(t, replacement.Wait())
	if !committed.Committed || committed.Verified != (4<<20)+4096 {
		t.Fatal("restored roles did not commit and survive native ADOPT/GC")
	}
	t.Logf("real partial role/process death/owner reopen/emptyStore restore/commit/ADOPT/GC PASS; reused=%d newrole=%d replacementUpload=%d warmUpload=0", resumed.Reused, resumed.New, firstMoved)
}
