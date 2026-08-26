package main

import (
	"context"
	"crypto/rand"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"

	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/grpc/metadata"

	"github.com/cozy-creator/cozy-creator-v2/internal/canonical"
	"github.com/cozy-creator/cozy-creator-v2/internal/config"
	"github.com/cozy-creator/cozy-creator-v2/internal/home"
	"github.com/cozy-creator/cozy-creator-v2/internal/secret"
	pb "github.com/cozy-creator/cozy-creator-v2/protocol/cozy/worker/v1"
)

// fakeWorker is the ADVERSARY: a second, independent implementation of the worker side
// of `cozy.worker.v1`, in Go, that dials the real coordinator socket exactly as the real
// supervisor does. It exists so the coordinator's refusal arms have someone to refuse —
// a real peer sending real bytes, never a plant inside the coordinator.
//
// It also proves something the SDXL run cannot: the coordinator interoperates with a
// worker it did not co-develop against, over the committed contract alone.
func fakeWorker() int {
	// `--socket`, because that is what the coordinator now speaks: `cozy-runtime serve`'s
	// public launch grammar. The adversary reads the same flags the real supervisor does.
	socket := flag("socket", "")
	// --fake-instance wins over the --instance-id StartWorker appends, so an arm can
	// claim an instance this coordinator never spawned.
	instance := flag("fake-instance", flag("instance-id", ""))
	releaseID := flag("release-id", "")
	arm := flag("arm", "idle")
	session := flag("session", "fake-"+randomHex(8))

	say := func(format string, args ...any) {
		fmt.Printf("[fake %s] %s\n", arm, fmt.Sprintf(format, args...))
		os.Stdout.Sync()
	}

	// The coordinator hands `--socket` a unix path where the OS has one, and a loopback
	// `host:port` on Windows (#449) — the adversary speaks both, exactly as the real
	// supervisor must.
	target := "unix://" + socket
	if len(socket) > 5 && socket[:5] == "unix:" {
		target = "unix://" + socket[5:]
	} else if strings.Contains(socket, ":") && !strings.ContainsAny(socket, `/\`) {
		target = socket
	}
	conn, err := grpc.NewClient(target, grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		say("cannot dial %s: %v", socket, err)
		return 1
	}
	defer conn.Close()
	// On the loopback transport the launcher hands a PER-SPAWN bootstrap credential
	// through the environment; Register must echo it as metadata (#449). The adversary
	// reads it the way every process reads its environment — through the ONE env reader —
	// and forwards whatever it was handed; an arm that wants the refusal simply is not
	// handed one.
	ctx := context.Background()
	if cfg, e := config.Load(); e == nil && cfg.Bootstrap.Present() {
		k, v := secret.GRPCMetadataPair("cozy-bootstrap", cfg.Bootstrap)
		ctx = metadata.AppendToOutgoingContext(ctx, k, v)
	}
	stream, err := pb.NewWorkerClient(conn).Control(ctx)
	if err != nil {
		say("cannot open Control: %v", err)
		return 1
	}

	if arm == "badrelease" {
		releaseID = "cozy/not-the-pinned-release@v0"
	}
	send := func(m *pb.WorkerMessage) {
		if err := stream.Send(m); err != nil {
			say("send failed: %v", err)
		}
	}
	send(&pb.WorkerMessage{Msg: &pb.WorkerMessage_Register{Register: &pb.Register{
		SessionId: session, ExecutorIncarnation: 1, WireMinor: pb.WireMinor,
		WorkerId: "local", InstanceId: instance, ReleaseId: releaseID,
		Resources: &pb.WorkerResources{GpuCount: 0, Platform: "fake"},
	}}})
	say("Register sent: session=%s instance=%s release=%s", session, instance, releaseID)

	var planIDs []string
	var dropAck *pb.AttemptTerminal
	for {
		msg, err := stream.Recv()
		if err != nil {
			say("stream closed: %v", err)
			return 0
		}
		switch m := msg.Msg.(type) {
		case *pb.CoordinatorMessage_RegisterAck:
			if !m.RegisterAck.Accepted {
				say("REGISTER REFUSED: %s",
					pb.RegisterRejection_name[int32(m.RegisterAck.Rejection)])
				return 0
			}
			say("RegisterAck accepted, wire_minor=%d", m.RegisterAck.WireMinor)
		case *pb.CoordinatorMessage_Directive:
			planIDs = m.Directive.GetServing().GetEntrypointBindingPlanIds()
			say("Directive revision=%d plans=%d", m.Directive.Revision, len(planIDs))
			send(&pb.WorkerMessage{Msg: &pb.WorkerMessage_Report{Report: &pb.Report{
				SessionId: session, ExecutorIncarnation: 1,
				AppliedRevision: m.Directive.Revision,
				IntakeState:     pb.IntakeState_INTAKE_STATE_READY,
				ReadinessEpoch:  1, AppliedWireMinor: pb.WireMinor,
				Capacity: &pb.Report_ServingCapacity{ServingCapacity: &pb.ServingCapacity{
					ReadyEntrypointBindingPlanIds: planIDs, FreeVramBytes: 1 << 30,
				}},
			}}})
			say("Report READY for %d plan(s)", len(planIDs))
			if arm == "steal" {
				stealTerminal(send, session, say)
				time.Sleep(2 * time.Second)
				return 0
			}
		case *pb.CoordinatorMessage_StartAttempt:
			start := m.StartAttempt
			say("StartAttempt %s#%d spec=%s", start.RequestId, start.Attempt,
				hex.EncodeToString(start.ExecSpecDigest)[:16])
			send(&pb.WorkerMessage{Msg: &pb.WorkerMessage_AttemptAccepted{
				AttemptAccepted: &pb.AttemptAccepted{
					SessionId: session, ExecutorIncarnation: 1,
					RequestId: start.RequestId, Attempt: start.Attempt,
					ExecSpecDigest:          start.ExecSpecDigest,
					PlanDigest:              canonical.Digest([]byte("fake-plan")),
					ModelConstructionDigest: canonical.Digest([]byte("fake-construction")),
					Plan:                    &pb.AttemptPlanSummary{Delivery: "native", Placement: "all_resident"},
				}}})
			if arm == "badterminal" {
				badTerminals(send, session, start, say)
			}
			if arm == "dropack" {
				dropAck = droppedAckTerminal(send, session, start, say)
			}
		case *pb.CoordinatorMessage_TerminalAck:
			say("TerminalAck for %s#%d digest=%s", m.TerminalAck.RequestId,
				m.TerminalAck.Attempt, hex.EncodeToString(m.TerminalAck.TerminalDigest)[:16])
			if arm == "badterminal" {
				time.Sleep(500 * time.Millisecond)
				return 0
			}
			if arm == "dropack" && dropAck != nil {
				// THE DROP. A worker whose ack never arrived is a worker that keeps
				// replaying its journaled terminal — byte for byte, because the document
				// is what it journaled and not something it re-derives. Ignoring the ack
				// here is what a lost one looks like from the coordinator's side.
				say("ARM: the TerminalAck is DROPPED, and the journaled terminal is replayed")
				send(&pb.WorkerMessage{Msg: &pb.WorkerMessage_AttemptTerminal{
					AttemptTerminal: dropAck}})
				dropAck = nil
				go func() { time.Sleep(3 * time.Second); os.Exit(0) }()
			}
		case *pb.CoordinatorMessage_CancelAttempt:
			say("CancelAttempt %s#%d", m.CancelAttempt.RequestId, m.CancelAttempt.Attempt)
		}
	}
}

// terminalFor builds one journaled TerminalBody and its envelope, the way a worker does:
// the document is canonicalized once, the digest is over exactly those bytes, and the
// envelope's routing copies are copies of the document's own fields.
func terminalFor(session string, requestID string, attempt uint64, spec []byte,
	status pb.TerminalStatus, message string) (*pb.AttemptTerminal, []byte) {
	body := &pb.TerminalBody{
		RequestId: requestID, Attempt: attempt, ExecSpecDigest: spec, Status: status,
		SafeMessage: message,
		Cause: &pb.TerminalCause{
			Code:   pb.CauseCode_CAUSE_CODE_AUTHOR_EXCEPTION,
			Origin: pb.CauseOrigin_CAUSE_ORIGIN_AUTHOR,
			Detail: "a fake worker's terminal",
		},
	}
	if status == pb.TerminalStatus_TERMINAL_STATUS_SUCCEEDED {
		body.Cause = &pb.TerminalCause{Origin: pb.CauseOrigin_CAUSE_ORIGIN_RUNTIME}
	}
	data, digest, err := canonical.Identity(body)
	if err != nil {
		panic(err)
	}
	return &pb.AttemptTerminal{
		SessionId: session, ExecutorIncarnation: 1, RequestId: requestID, Attempt: attempt,
		ExecSpecDigest: spec, TerminalId: "trm-" + randomHex(8), TerminalDigest: digest,
		TerminalCanonical: data,
	}, data
}

// badTerminals is the refusal matrix, sent in order. Only the LAST one is admissible,
// and the ack that follows it is the coordinator saying so.
func badTerminals(send func(*pb.WorkerMessage), session string, start *pb.StartAttempt,
	say func(string, ...any)) {
	emit := func(t *pb.AttemptTerminal) {
		send(&pb.WorkerMessage{Msg: &pb.WorkerMessage_AttemptTerminal{AttemptTerminal: t}})
		time.Sleep(400 * time.Millisecond)
	}

	// 1. A terminal_digest that does not hash the resident bytes. The receiver
	//    RECOMPUTES; a digest never bypasses the lower check.
	t, _ := terminalFor(session, start.RequestId, start.Attempt, start.ExecSpecDigest,
		pb.TerminalStatus_TERMINAL_STATUS_SUCCEEDED, "planted digest")
	t.TerminalDigest = canonical.Digest([]byte("not the body"))
	say("ARM 1: terminal_digest planted")
	emit(t)

	// 2. The envelope's routing copies disagree with the document. The DOCUMENT is
	//    authoritative, so divergence refuses rather than picking a winner.
	t, _ = terminalFor(session, start.RequestId, start.Attempt+7, start.ExecSpecDigest,
		pb.TerminalStatus_TERMINAL_STATUS_SUCCEEDED, "divergent envelope")
	t.Attempt = start.Attempt // the envelope says N, the document says N+7
	say("ARM 2: envelope/document divergence")
	emit(t)

	// 3. A key the closed document has no slot for, written BY HAND because the schema
	//    cannot express it — which is the point of the arm.
	t, data := terminalFor(session, start.RequestId, start.Attempt, start.ExecSpecDigest,
		pb.TerminalStatus_TERMINAL_STATUS_SUCCEEDED, "planted key")
	doc, err := canonical.Read(data, &pb.TerminalBody{})
	if err == nil {
		raw := map[string]canonical.Value(doc)
		raw["service_class"] = "priority"
		planted, werr := canonical.Write(raw)
		if werr == nil {
			t.TerminalCanonical = planted
			t.TerminalDigest = canonical.Digest(planted)
		}
	}
	say("ARM 3: a planted key in the terminal document")
	emit(t)

	// 4. An admissible terminal for an attempt this session does hold.
	t, _ = terminalFor(session, start.RequestId, start.Attempt, start.ExecSpecDigest,
		pb.TerminalStatus_TERMINAL_STATUS_FAILED, "the fake worker has no GPU")
	say("ARM 4: an admissible terminal")
	emit(t)
	// 5. The exact same terminal again: a replay must re-ack and apply nothing twice.
	say("ARM 5: the same terminal replayed")
	emit(t)
}

// droppedAckTerminal writes ONE real output under the attempt's granted directory and
// sends a SUCCEEDED terminal that declares it. It returns the identical envelope, which
// the caller replays when the ack arrives — the coordinator half of a lost TerminalAck.
//
// The output matters: "applied exactly once" is only interesting if applying it twice
// would publish the bytes twice, so the terminal has to carry bytes.
func droppedAckTerminal(send func(*pb.WorkerMessage), session string, start *pb.StartAttempt,
	say func(string, ...any)) *pb.AttemptTerminal {
	// The grant is the coordinator's own attempt directory, derived from the root this
	// worker was launched against — the same namespace the real runtime writes under.
	// The root is a FLAG, not the environment: this repository reads the environment
	// once, at the product entrypoint, and a driver is not it.
	layout, e := home.Open(flag("cozy-home", ""))
	if e != nil {
		say("no layout: %s", e.Message)
		return nil
	}
	dir := layout.AttemptDir(start.RequestId, start.Attempt)
	if err := os.MkdirAll(dir, 0o755); err != nil {
		say("cannot write under the grant: %v", err)
		return nil
	}
	// A minimal, real PNG: an output is bytes on disk, not a claim in a document.
	body, err := hex.DecodeString(onePixelPNG)
	if err != nil {
		say("bad fixture: %v", err)
		return nil
	}
	if err := os.WriteFile(filepath.Join(dir, "image"), body, 0o644); err != nil {
		say("cannot write the output: %v", err)
		return nil
	}
	sum := sha256.Sum256(body)
	t, _ := terminalFor(session, start.RequestId, start.Attempt, start.ExecSpecDigest,
		pb.TerminalStatus_TERMINAL_STATUS_SUCCEEDED, "one output, and an ack that will be lost")
	doc, err := canonical.Read(t.TerminalCanonical, &pb.TerminalBody{})
	if err != nil {
		say("cannot read back the terminal: %v", err)
		return nil
	}
	raw := map[string]canonical.Value(doc)
	raw["output_manifest"] = map[string]canonical.Value{
		"manifest_id": "man-" + start.RequestId,
		"outputs": []canonical.Value{map[string]canonical.Value{
			"output_id": "image", "digest": "sha256:" + hex.EncodeToString(sum[:]),
			"length": int64(len(body)), "mime_type": "image/png",
		}},
	}
	written, err := canonical.Write(raw)
	if err != nil {
		say("cannot canonicalize the manifest: %v", err)
		return nil
	}
	t.TerminalCanonical, t.TerminalDigest = written, canonical.Digest(written)
	say("terminal with ONE %d B output under %s", len(body), dir)
	send(&pb.WorkerMessage{Msg: &pb.WorkerMessage_AttemptTerminal{AttemptTerminal: t}})
	return t
}

// onePixelPNG is a 1x1 PNG, hex-encoded: the smallest thing that is really an image.
const onePixelPNG = "89504e470d0a1a0a0000000d4948445200000001000000010806000000" +
	"1f15c4890000000d49444154789c6360000002000100ffff03000006000557bfabd40000000049454e44ae426082"

// stealTerminal is a SECOND session trying to write another session's attempt row.
func stealTerminal(send func(*pb.WorkerMessage), session string, say func(string, ...any)) {
	requestID := flag("request", "")
	var attempt uint64
	fmt.Sscanf(flag("attempt", "1"), "%d", &attempt)
	spec, _ := hex.DecodeString(flag("spec", ""))
	t, _ := terminalFor(session, requestID, attempt, spec,
		pb.TerminalStatus_TERMINAL_STATUS_SUCCEEDED, "a terminal from a session that does not own it")
	say("ARM: session %s claims %s#%d, which it was never assigned", session, requestID, attempt)
	send(&pb.WorkerMessage{Msg: &pb.WorkerMessage_AttemptTerminal{AttemptTerminal: t}})
}

// randomHex uses crypto/rand, which speaks for every OS — the /dev/urandom spelling
// silently produced all-zero session ids on Windows, colliding every fake worker on one
// session (found by the windows-proc run: the "second writer" and "worker B" arms were
// really SESSION_COLLISION refusals).
func randomHex(n int) string {
	b := make([]byte, n)
	if _, err := rand.Read(b); err != nil {
		panic("no randomness for a fake session id: " + err.Error())
	}
	return hex.EncodeToString(b)
}
