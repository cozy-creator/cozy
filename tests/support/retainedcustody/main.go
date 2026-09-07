// Explicit operator recovery for receipts retained by an unresponsive Runtime.
package main

import (
	"bufio"
	"bytes"
	"context"
	"encoding/hex"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"os/exec"
	"os/signal"
	"syscall"
	"time"

	"github.com/cozy-creator/cozy/internal/accountauth"
	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/cli"
	"github.com/cozy-creator/cozy/internal/config"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/home"
	"github.com/cozy-creator/cozy/internal/orchestrator"
	"github.com/cozy-creator/cozy/internal/records"
	"github.com/cozy-creator/cozy/internal/rental"
	"github.com/cozy-creator/cozy/internal/workertls"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials"
)

func main() {
	defer func() {
		if r := recover(); r != nil {
			if message, ok := r.(operatorFailure); ok {
				fmt.Fprintln(os.Stderr, string(message))
				os.Exit(1)
			}
			panic(r)
		}
	}()
	run()
}

func run() {
	requestID := flag.String("request", "", "exact retained producer request")
	rentalID := flag.String("rental", "", "exact original rental")
	bootID := flag.String("boot", "", "exact original worker boot")
	reader := flag.String("reader", "", "explicit serial reader executable; remaining arguments go to it")
	proofObject := flag.String("proof-object", "", "bank only one exact object through the complete native/Hub/Host custody path")
	execute := flag.Bool("execute", false, "publish retained bytes through existing Hub and PodHost APIs")
	flag.Parse()
	if *requestID == "" || *rentalID == "" || *bootID == "" {
		fail("exact request, rental and boot are required")
	}
	ctx, stop := signal.NotifyContext(context.Background(), syscall.SIGINT, syscall.SIGTERM)
	defer stop()
	cfg, problem := config.Load()
	check(problem)
	layout, problem := home.Open(cfg.Home)
	check(problem)
	st, problem := records.Open(layout.DB)
	check(problem)
	defer st.Close()
	req, problem := st.RequestRow(*requestID)
	check(problem)
	if req == nil || !req.IsJob() || req.ModelTransfer == nil || req.ModelTransfer.Kind != "model-upload" || req.Worker != *rentalID {
		fail("request does not bind this retained upload")
	}
	transfer, problem := st.ModelTransferOf(req.ID)
	check(problem)
	if transfer == nil || transfer.State != "failed" {
		fail("operator recovery requires a blocked publication")
	}
	attempt, problem := st.AttemptRow(req.ID, req.Ordinal)
	check(problem)
	if attempt == nil || attempt.TerminalStatus != "SUCCEEDED" || attempt.State != "terminal" {
		fail("producer has no retained successful outcome")
	}
	for _, pair := range []struct {
		raw    []byte
		digest string
	}{{attempt.InvocationCanonical, attempt.InvocationDigest}, {attempt.TerminalBody, attempt.TerminalDigest}} {
		digest, err := canonical.Raw(pair.digest)
		if err != nil || !bytes.Equal(digest, canonical.Digest(pair.raw)) {
			fail("retained producer identity does not verify")
		}
	}
	rows, problem := st.AllModelTransferWeights(req.ID, req.Ordinal)
	check(problem)
	if len(rows) == 0 || len(rows) != len(req.ModelTransfer.Outputs) {
		fail("producer receipt roster is incomplete")
	}
	for _, row := range rows {
		digest, err := canonical.Raw(row.ReceiptDigest)
		if err != nil || !bytes.Equal(digest, canonical.Digest(row.Receipt)) || row.InvocationDigest != attempt.InvocationDigest {
			fail("retained output receipt changed")
		}
	}
	target, problem := rental.Resolver(layout, st)(*rentalID)
	check(problem)
	remote := target.Connection
	if remote == nil || remote.WorkerBootID != *bootID {
		fail("rental boot changed")
	}
	emit(map[string]any{"preflight": true, "request": req.ID, "rental": *rentalID, "boot": *bootID, "receipts": len(rows), "execute": *execute})
	if !*execute {
		return
	}
	if *reader == "" {
		fail("an explicit serial reader is required")
	}
	auth := accountauth.New(cfg)
	pin, err := workertls.LoadPin(remote.CACert)
	if err != nil {
		fail("rental TLS pin is unreadable")
	}
	conn, err := grpc.NewClient(remote.Addr, grpc.WithTransportCredentials(credentials.NewTLS(pin.TLSConfig())))
	if err != nil {
		fail("cannot dial pinned PodHost")
	}
	defer conn.Close()
	proof, problem := rental.ClaimProof(layout)(remote, 1)
	check(problem)
	claim := &pb.Claim{RecordOwnerEpoch: 1, RecordOwnerId: "cozy-local-client", WorkerId: remote.WorkerID, WorkerBootId: *bootID, WireMinor: pb.WireMinor, Proof: proof}
	command := exec.CommandContext(ctx, *reader, flag.Args()...)
	command.Env = cfg.Tool()
	input, err := command.StdinPipe()
	if err != nil {
		fail("reader stdin unavailable")
	}
	output, err := command.StdoutPipe()
	if err != nil {
		fail("reader stdout unavailable")
	}
	command.Stderr = io.Discard
	if err = command.Start(); err != nil {
		fail("reader cannot start")
	}
	defer func() {
		input.Close()
		if command.ProcessState == nil {
			command.Process.Kill()
			command.Wait()
		}
	}()
	answers := bufio.NewScanner(output)
	answers.Buffer(make([]byte, 4096), 64*1024)
	mover := &custodyMover{store: st, host: pb.NewPodHostClient(conn), claim: claim, input: json.NewEncoder(input), answers: answers, proofObject: *proofObject}
	owner := cli.NewModelTransferOwner(cfg, st, os.Stderr, auth)
	problem = owner.Finalize(ctx, req.ID, mover.move)
	if problem != nil && problem.ErrName() == "operator.proof_complete" {
		emit(map[string]any{"joined_custody_proof": true, "object": *proofObject})
		return
	}
	finalized, readProblem := st.AllModelTransferWeights(req.ID, req.Ordinal)
	check(readProblem)
	all := true
	for _, row := range finalized {
		all = all && row.FinalID != ""
		emit(map[string]any{"output": row.OutputSlot, "manifest": row.ManifestID, "final_id": row.FinalID})
	}
	if all {
		emit(map[string]any{"custody_complete": true, "request_lifecycle_recovery_pending": problem != nil})
		return
	}
	check(problem)
	fail("not all output checkpoints have verified custody")
}

type custodyMover struct {
	store       *records.Store
	host        pb.PodHostClient
	claim       *pb.Claim
	input       *json.Encoder
	answers     *bufio.Scanner
	proofObject string
}

type readerResult struct {
	OK          bool   `json:"ok"`
	Manifest    string `json:"manifest_digest"`
	ObjectID    string `json:"object_id"`
	Length      int64  `json:"length"`
	Transferred int64  `json:"transferred_bytes"`
	HTTPStatus  int    `json:"http_status"`
	Code        string `json:"code"`
	ErrorType   string `json:"error_type"`
}

func (m *custodyMover) move(ctx context.Context, weights records.ModelTransferWeights, mint orchestrator.WeightsGrantMinter) *exit.Error {
	objects, problem := m.store.ModelTransferObjects(weights.RequestID, weights.Attempt, weights.OutputSlot)
	if problem != nil {
		return problem
	}
	ids := make([]string, len(objects))
	for i, o := range objects {
		ids[i] = o.ObjectID
	}
	window := orchestrator.NewWeightsGrantWindow(mint)
	sent := int64(0)
	for i, object := range objects {
		if m.proofObject != "" && object.ObjectID != m.proofObject {
			continue
		}
		if object.State == "held" || object.State == "uploaded" || object.State == "already_present" {
			continue
		}
		decision, problem := window.Spendable(ctx, object.ObjectID, ids[i:], time.Now())
		if problem != nil {
			return problem
		}
		if decision.ObjectID != object.ObjectID || decision.Length != object.Length {
			return exit.New(exit.Conflict, "Hub grant changed retained object identity")
		}
		if !decision.Held {
			doc := map[string]any{"manifest": map[string]any{"digest": weights.ManifestID, "length": weights.ManifestLength}, "object": map[string]any{"digest": object.ObjectID, "length": object.Length}, "grant": map[string]any{"url": decision.URL, "required_headers": decision.Headers}}
			if m.input.Encode(doc) != nil {
				return exit.Unavailablef("serial native reader input closed")
			}
			if !m.answers.Scan() {
				return exit.Unavailablef("serial native reader returned no bounded result")
			}
			var result readerResult
			if json.Unmarshal(m.answers.Bytes(), &result) != nil || !result.OK || result.Manifest != weights.ManifestID || result.ObjectID != object.ObjectID || result.Length != object.Length || result.Transferred < 0 || result.Transferred > object.Length || !(result.HTTPStatus >= 200 && result.HTTPStatus < 300 || result.HTTPStatus == 412) {
				emit(map[string]any{"reader_ok": result.OK, "reader_code": result.Code, "reader_error_type": result.ErrorType})
				return exit.Named(exit.Conflict, "operator.reader_refused", "serial native reader refused or changed the exact object result")
			}
			sent += result.Transferred
			verified, problem := mint(ctx, []string{object.ObjectID})
			if problem != nil {
				return problem
			}
			if len(verified.Decisions) != 1 || !verified.Decisions[0].Held || verified.Decisions[0].ObjectID != object.ObjectID || verified.Decisions[0].Length != object.Length {
				return exit.Unavailablef("Hub has not verified the uploaded exact object")
			}
		}
		if problem = m.hold(ctx, weights, object); problem != nil {
			return problem
		}
		emit(map[string]any{"output": weights.OutputSlot, "object": object.ObjectID, "held": true, "objects_done": i + 1, "objects": len(objects), "reader_bytes_sent": sent})
		if m.proofObject != "" {
			return exit.Named(exit.Canceled, "operator.proof_complete", "one exact object has verified Hub and Host custody")
		}
	}
	return nil
}

func (m *custodyMover) hold(ctx context.Context, weights records.ModelTransferWeights, object records.ModelTransferObject) *exit.Error {
	invocation, err := canonical.Raw(weights.InvocationDigest)
	if err != nil {
		return exit.New(exit.Structural, "invalid retained invocation digest")
	}
	receipt, err := canonical.Raw(weights.ReceiptDigest)
	if err != nil {
		return exit.New(exit.Structural, "invalid retained receipt digest")
	}
	digest, err := canonical.Raw(object.ObjectID)
	if err != nil {
		return exit.New(exit.Structural, "invalid retained object digest")
	}
	operation := object.OperationID
	if operation == "" {
		operation = "weights-" + hex.EncodeToString(digest)
	}
	request := &pb.WeightsTransferRequest{RecordOwnerEpoch: m.claim.RecordOwnerEpoch, WorkerBootId: m.claim.WorkerBootId, RequestId: weights.RequestID, AttemptOrdinal: uint64(weights.Attempt), InvocationSpecDigest: invocation, OutputSlot: weights.OutputSlot, WeightsTransactionId: weights.TransactionID, WeightsReceiptDigest: receipt, OperationId: operation, GrantRevision: uint64(object.GrantRevision + 1), Decision: &pb.WeightsTransferRequest_Held{Held: &pb.WeightsObjectRef{ObjectId: object.ObjectID, Length: uint64(object.Length)}}}
	callCtx, cancel := context.WithTimeout(ctx, 30*time.Second)
	defer cancel()
	stream, err := m.host.WeightsTransfer(callCtx, &pb.WeightsTransferCall{Claim: m.claim, Request: request})
	if err != nil {
		return exit.Unavailablef("pinned PodHost custody call failed")
	}
	for {
		status, err := stream.Recv()
		if err != nil {
			return exit.Unavailablef("pinned PodHost returned no held custody")
		}
		if status.RecordOwnerEpoch != request.RecordOwnerEpoch || status.ControlStreamEpoch != 0 || status.WorkerBootId != request.WorkerBootId || status.RequestId != request.RequestId || status.AttemptOrdinal != request.AttemptOrdinal || !bytes.Equal(status.InvocationSpecDigest, invocation) || status.OutputSlot != weights.OutputSlot || status.WeightsTransactionId != weights.TransactionID || status.ObjectId != object.ObjectID || status.OperationId != operation || status.GrantRevision != request.GrantRevision || status.Length != uint64(object.Length) {
			return exit.Named(exit.Conflict, "operator.host_binding_changed", "PodHost custody response changed its request binding")
		}
		if status.State == pb.WeightsTransferState_WEIGHTS_TRANSFER_STATE_ACCEPTED {
			continue
		}
		if status.State != pb.WeightsTransferState_WEIGHTS_TRANSFER_STATE_HELD || status.TransferredBytes > uint64(object.Length) || status.ChecksumSha256 != object.ObjectID || status.UpdateSequence == 0 {
			return exit.Named(exit.Conflict, "operator.host_custody_refused", "PodHost did not confirm exact held custody")
		}
		return m.store.RecordModelTransferObjectStatus(records.ModelTransferObject{RequestID: weights.RequestID, Attempt: weights.Attempt, OutputSlot: weights.OutputSlot, ObjectID: object.ObjectID, Length: object.Length, OperationID: operation, GrantRevision: int64(status.GrantRevision), UpdateSequence: int64(status.UpdateSequence), State: "held", Transferred: int64(status.TransferredBytes)})
	}
}
func emit(value any) { _ = json.NewEncoder(os.Stdout).Encode(value) }
func check(problem *exit.Error) {
	if problem != nil {
		fail(problem.ErrName())
	}
}

type operatorFailure string

func fail(message string) { panic(operatorFailure(message)) }
