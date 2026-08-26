package main

import (
	"crypto/rand"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"net"
	"os"
	"path/filepath"
	"strings"
	"time"

	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials"

	"github.com/cozy-creator/cozy-creator-v2/internal/canonical"
	"github.com/cozy-creator/cozy-creator-v2/internal/config"
	"github.com/cozy-creator/cozy-creator-v2/internal/home"
	pb "github.com/cozy-creator/cozy-creator-v2/protocol/cozy/worker/v1"
)

// fakeWorker is the ADVERSARY: a second, independent implementation of the WORKER side of
// `cozy.worker.v1`, in Go, that HOSTS `WorkerControl` exactly as the real worker does
// (#436 — the owner dials). It exists so the coordinator's refusal arms have someone real
// to refuse — a real peer serving real bytes, never a plant inside the coordinator.
//
// It also proves something the SDXL run cannot: the coordinator interoperates with a
// worker it did not co-develop against, over the committed contract alone.
func fakeWorker() int {
	// The serve grammar the coordinator speaks: `--socket` is the LISTEN grant, `--out`
	// the root whose `control.addr` publishes the bound address (the discovery contract).
	listen := flag("socket", "")
	out := flag("out", "")
	// --fake-instance wins over the --instance-id StartWorker appends, so an arm can
	// report an instance identity this coordinator never spawned.
	instance := flag("fake-instance", flag("instance-id", ""))
	releaseID := flag("release-id", "")
	arm := flag("arm", "idle")
	bootID := flag("session", "boot-fake-"+randomHex(8))

	say := func(format string, args ...any) {
		fmt.Printf("[fake %s] %s\n", arm, fmt.Sprintf(format, args...))
		os.Stdout.Sync()
	}
	if arm == "badrelease" {
		releaseID = "cozy/not-the-pinned-release@v0"
	}

	// Bind exactly as the real worker does: a unix path, or host:port loopback.
	network, address := "unix", listen
	if strings.Contains(listen, ":") && !strings.ContainsAny(listen, `/\`) {
		network, address = "tcp", listen
	}
	if network == "unix" {
		_ = os.MkdirAll(filepath.Dir(address), 0o755)
	}
	_ = os.Remove(address)
	ln, err := net.Listen(network, address)
	if err != nil {
		say("cannot bind %s: %v", listen, err)
		return 1
	}
	bound := address
	if network == "tcp" {
		bound = ln.Addr().String()
	}
	if out != "" {
		if err := os.MkdirAll(out, 0o755); err == nil {
			staged := filepath.Join(out, "control.addr.staging")
			_ = os.WriteFile(staged, []byte(bound+"\n"), 0o644)
			_ = os.Rename(staged, filepath.Join(out, "control.addr"))
		}
	}
	say("hosting WorkerControl at %s (boot %s)", bound, bootID)

	// The per-spawn bootstrap credential the launcher handed through the environment;
	// the owner must present it as Claim.proof (#463's flip). Verification is the secret
	// package's constant-time Equal — Reveal never happens here. The `badcred` arm
	// refuses EVERY proof, so the real owner's correct one is refused — which is the arm.
	verify := func(string) bool { return true }
	if cfg, e := config.Load(); e == nil && cfg.Bootstrap.Present() {
		bootstrap := cfg.Bootstrap
		verify = bootstrap.Equal
	}
	if arm == "badcred" {
		verify = func(string) bool { return false }
	}

	// The POD's spelling of the same listener (#445): the same `WorkerControl`, behind TLS
	// with the certificate the rental pins. `cozy-runtime serve --tls-cert/--tls-key` is
	// the real worker's flag pair and this is the adversary's, so the owner's dial is
	// exercised against a peer it did not co-develop with.
	var opts []grpc.ServerOption
	if cert, key := flag("tls-cert", ""), flag("tls-key", ""); cert != "" && key != "" {
		creds, err := credentials.NewServerTLSFromFile(cert, key)
		if err != nil {
			say("cannot host TLS from %s/%s: %v", cert, key, err)
			return 1
		}
		opts = append(opts, grpc.Creds(creds))
		say("hosting behind TLS with the certificate the owner pins")
	}
	server := grpc.NewServer(opts...)
	pb.RegisterWorkerControlServer(server, &fakeControl{
		say: say, arm: arm, bootID: bootID, instance: instance, releaseID: releaseID,
		root: flag("cozy-home", ""), verify: verify,
	})
	if err := server.Serve(ln); err != nil {
		say("serve ended: %v", err)
	}
	return 0
}

type fakeControl struct {
	pb.UnimplementedWorkerControlServer
	say        func(string, ...any)
	arm        string
	bootID     string
	instance   string
	releaseID  string
	root       string // this worker's OWN filesystem root; nothing outside it is writable
	verify     func(string) bool
	generation uint64
}

func (f *fakeControl) WatchProgress(open *pb.ProgressOpen, stream pb.WorkerControl_WatchProgressServer) error {
	<-stream.Context().Done()
	return nil
}

func (f *fakeControl) Control(stream pb.WorkerControl_ControlServer) error {
	frame, err := stream.Recv()
	if err != nil {
		return nil
	}
	claim := frame.GetClaim()
	if claim == nil {
		f.say("first frame was not a Claim; closing")
		return nil
	}
	f.generation++
	generation := f.generation
	env := func(set func(epoch uint64, gen uint64, boot string)) {
		set(claim.OwnerEpoch, generation, f.bootID)
	}
	send := func(m *pb.WorkerFrame) {
		if err := stream.Send(m); err != nil {
			f.say("send failed: %v", err)
		}
	}
	if !f.verify(string(claim.Proof)) {
		ack := &pb.ClaimAck{Accepted: false,
			Rejection: pb.ClaimRejection_CLAIM_REJECTION_UNAUTHENTICATED,
			WireMinor: pb.WireMinor}
		env(func(e, g uint64, b string) { ack.OwnerEpoch, ack.ControlGeneration, ack.WorkerBootId = e, g, b })
		send(&pb.WorkerFrame{Msg: &pb.WorkerFrame_ClaimAck{ClaimAck: ack}})
		f.say("Claim REFUSED: the presented proof is not the provisioned credential")
		return nil
	}
	ack := &pb.ClaimAck{
		Accepted: true, WireMinor: pb.WireMinor, WorkerId: "local",
		InstanceId: f.instance, ReleaseId: f.releaseID,
		Resources: &pb.WorkerResources{Platform: "fake", Backend: ""},
	}
	env(func(e, g uint64, b string) { ack.OwnerEpoch, ack.ControlGeneration, ack.WorkerBootId = e, g, b })
	send(&pb.WorkerFrame{Msg: &pb.WorkerFrame_ClaimAck{ClaimAck: ack}})
	f.say("ClaimAck sent: boot=%s instance=%s release=%s", f.bootID, f.instance, f.releaseID)

	snapshotID := "snp-fake-" + randomHex(6)
	begin := &pb.SnapshotBegin{SnapshotId: snapshotID}
	env(func(e, g uint64, b string) { begin.OwnerEpoch, begin.ControlGeneration, begin.WorkerBootId = e, g, b })
	send(&pb.WorkerFrame{Msg: &pb.WorkerFrame_SnapshotBegin{SnapshotBegin: begin}})
	end := &pb.SnapshotEnd{SnapshotId: snapshotID, EntryCount: 0}
	env(func(e, g uint64, b string) { end.OwnerEpoch, end.ControlGeneration, end.WorkerBootId = e, g, b })
	send(&pb.WorkerFrame{Msg: &pb.WorkerFrame_SnapshotEnd{SnapshotEnd: end}})

	report := func(revision uint64, deploymentID string, planIDs []string) {
		r := &pb.Report{
			AppliedRevision:  revision,
			IntakeState:      pb.IntakeState_INTAKE_STATE_READY,
			AppliedWireMinor: pb.WireMinor,
		}
		if deploymentID != "" {
			r.Deployments = []*pb.DeploymentStatus{{
				DeploymentId:   deploymentID,
				IntakeState:    pb.IntakeState_INTAKE_STATE_READY,
				ReadinessEpoch: 1, ExecutorGeneration: 1, AttemptCredits: 2,
				ReadyEntrypointBindingPlanIds: planIDs,
			}}
		}
		env(func(e, g uint64, b string) { r.OwnerEpoch, r.ControlGeneration, r.WorkerBootId = e, g, b })
		send(&pb.WorkerFrame{Msg: &pb.WorkerFrame_Report{Report: r}})
	}

	terminal := func(t *pb.AttemptTerminal) {
		env(func(e, g uint64, b string) { t.OwnerEpoch, t.ControlGeneration, t.WorkerBootId = e, g, b })
		send(&pb.WorkerFrame{Msg: &pb.WorkerFrame_AttemptTerminal{AttemptTerminal: t}})
	}

	var dropAck *pb.AttemptTerminal
	for {
		frame, err := stream.Recv()
		if err != nil {
			f.say("stream closed: %v", err)
			return nil
		}
		switch m := frame.Msg.(type) {
		case *pb.OwnerFrame_SnapshotAck:
			f.say("SnapshotAck for %s; dispatch open", m.SnapshotAck.SnapshotId)
			if f.arm == "steal" {
				f.stealTerminal(terminal)
				time.Sleep(2 * time.Second)
				return nil
			}
		case *pb.OwnerFrame_Directive:
			d := m.Directive
			deploymentID, planIDs := "", []string(nil)
			if ds := d.GetDeploymentSet(); ds != nil && len(ds.GetSet().GetDeployments()) > 0 {
				dep := ds.GetSet().GetDeployments()[0]
				deploymentID, planIDs = dep.DeploymentId, dep.EntrypointBindingPlanIds
			}
			f.say("Directive revision=%d deployment=%s plans=%d", d.Revision, deploymentID, len(planIDs))
			report(d.Revision, deploymentID, planIDs)
			f.say("Report READY for %d plan(s)", len(planIDs))
		case *pb.OwnerFrame_StartAttempt:
			start := m.StartAttempt
			f.say("StartAttempt %s#%d deployment=%s spec=%s", start.RequestId, start.Attempt,
				start.DeploymentId, hex.EncodeToString(start.InvocationDigest)[:16])
			accepted := &pb.AttemptAccepted{
				RequestId: start.RequestId, Attempt: start.Attempt,
				InvocationDigest:        start.InvocationDigest,
				PlanDigest:              canonical.Digest([]byte("fake-plan")),
				ModelConstructionDigest: canonical.Digest([]byte("fake-construction")),
				Plan:                    &pb.AttemptPlanSummary{Delivery: "native", Placement: "all_resident"},
				DeploymentId:            start.DeploymentId, ExecutorGeneration: 1,
			}
			env(func(e, g uint64, b string) {
				accepted.OwnerEpoch, accepted.ControlGeneration, accepted.WorkerBootId = e, g, b
			})
			send(&pb.WorkerFrame{Msg: &pb.WorkerFrame_AttemptAccepted{AttemptAccepted: accepted}})
			if f.arm == "badterminal" {
				f.badTerminals(terminal, start)
			}
			if f.arm == "dropack" {
				dropAck = f.terminalWithOutput(terminal, start)
			}
			if f.arm == "remote" || f.arm == "remotelie" {
				f.remoteTerminal(terminal, start)
			}
		case *pb.OwnerFrame_TerminalAck:
			f.say("TerminalAck for %s#%d digest=%s", m.TerminalAck.RequestId,
				m.TerminalAck.Attempt, hex.EncodeToString(m.TerminalAck.TerminalDigest)[:16])
			if f.arm == "badterminal" {
				time.Sleep(500 * time.Millisecond)
				return nil
			}
			if f.arm == "dropack" && dropAck != nil {
				// THE DROP. A worker whose ack never arrived keeps replaying its journaled
				// terminal — byte for byte. Ignoring the ack here is what a lost one looks
				// like from the coordinator's side.
				f.say("ARM: the TerminalAck is DROPPED, and the journaled terminal is replayed")
				terminal(dropAck)
				dropAck = nil
				go func() { time.Sleep(3 * time.Second); os.Exit(0) }()
			}
		case *pb.OwnerFrame_CancelAttempt:
			cancel := m.CancelAttempt
			f.say("CancelAttempt %s#%d", cancel.RequestId, cancel.Attempt)
			if f.arm == "remote" || f.arm == "remotelie" {
				// A cancel is an ASK and the attempt's own journaled terminal is what
				// settles it. A worker that never answers leaves the owner watching
				// forever — a worker defect, not a protocol one — so the pod side answers
				// here the way a real supervisor does.
				t, _ := terminalFor(cancel.RequestId, cancel.Attempt, cancel.InvocationDigest,
					pb.TerminalStatus_TERMINAL_STATUS_CANCELED, "canceled by the owner")
				terminal(t)
			}
		}
	}
}

// terminalFor builds one journaled TerminalBody and its envelope, the way a worker does:
// the document is canonicalized once, the digest is over exactly those bytes, and the
// envelope's routing copies are copies of the document's own fields.
func terminalFor(requestID string, attempt uint64, spec []byte,
	status pb.TerminalStatus, message string) (*pb.AttemptTerminal, []byte) {
	spelled, _ := canonical.Spell(spec)
	body := &pb.TerminalBody{
		RequestId: requestID, Attempt: attempt, InvocationDigest: spelled, Status: status,
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
		RequestId: requestID, Attempt: attempt,
		InvocationDigest: spec, TerminalId: "trm-" + randomHex(8), TerminalDigest: digest,
		TerminalCanonical: data,
	}, data
}

// badTerminals is the refusal matrix, sent in order. Only the LAST one is admissible,
// and the ack that follows it is the coordinator saying so.
func (f *fakeControl) badTerminals(emit func(*pb.AttemptTerminal), start *pb.StartAttempt) {
	send := func(t *pb.AttemptTerminal) {
		emit(t)
		time.Sleep(400 * time.Millisecond)
	}

	// 1. A terminal_digest that does not hash the resident bytes. The receiver
	//    RECOMPUTES; a digest never bypasses the lower check.
	t, _ := terminalFor(start.RequestId, start.Attempt, start.InvocationDigest,
		pb.TerminalStatus_TERMINAL_STATUS_SUCCEEDED, "planted digest")
	t.TerminalDigest = canonical.Digest([]byte("not the body"))
	f.say("ARM 1: terminal_digest planted")
	send(t)

	// 2. The envelope's routing copies disagree with the document. The DOCUMENT is
	//    authoritative, so divergence refuses rather than picking a winner.
	t, _ = terminalFor(start.RequestId, start.Attempt+7, start.InvocationDigest,
		pb.TerminalStatus_TERMINAL_STATUS_SUCCEEDED, "divergent envelope")
	t.Attempt = start.Attempt // the envelope says N, the document says N+7
	f.say("ARM 2: envelope/document divergence")
	send(t)

	// 3. A key the closed document has no slot for, written BY HAND because the schema
	//    cannot express it — which is the point of the arm.
	t, data := terminalFor(start.RequestId, start.Attempt, start.InvocationDigest,
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
	f.say("ARM 3: a planted key in the terminal document")
	send(t)

	// 4. An admissible terminal for an attempt this worker does hold.
	t, _ = terminalFor(start.RequestId, start.Attempt, start.InvocationDigest,
		pb.TerminalStatus_TERMINAL_STATUS_FAILED, "the fake worker has no GPU")
	f.say("ARM 4: an admissible terminal")
	send(t)
	// 5. The exact same terminal again: a replay must re-ack and apply nothing twice.
	f.say("ARM 5: the same terminal replayed")
	send(t)
}

// remoteTerminal is the POD's side of one attempt, and the arm that makes the byte
// boundary REAL. A worker writes where the GRANT says and nowhere else — so a pod whose
// grant names a directory on the OWNER's filesystem has been handed a destination it
// cannot reach, and says so by name instead of finding somewhere else to put the bytes.
//
//	remote     refuses the foreign destination — what a real pod does with a file:// URL
//	           rooted on the client's disk
//	remotelie  does the work, writes the bytes on ITS OWN disk, and declares them anyway:
//	           the manifest is true about what exists on the pod and false about what the
//	           owner can show, which is exactly what an UNMIRRORED output is
func (f *fakeControl) remoteTerminal(emit func(*pb.AttemptTerminal), start *pb.StartAttempt) {
	granted := ""
	for _, o := range start.Grant.GetOutputs() {
		if o.OutputId == "image" {
			granted = strings.TrimPrefix(o.Url, "file://")
		}
	}
	if granted == "" {
		f.say("the grant names no `image` destination; nothing to write")
		return
	}
	if f.arm == "remote" && !underRoot(granted, f.root) {
		f.say("ARM: the granted destination %s is not on this machine (my root is %s)", granted, f.root)
		t, _ := terminalFor(start.RequestId, start.Attempt, start.InvocationDigest,
			pb.TerminalStatus_TERMINAL_STATUS_FAILED,
			"the granted output destination is on the owner's filesystem, not this worker's")
		emit(t)
		return
	}
	// remotelie: the bytes go where this worker CAN write, which is not where the owner
	// granted. Nothing here lies about the digest — the manifest describes real bytes on
	// a real disk. What is false is only that the OWNER holds them.
	f.terminalWithOutput(emit, start)
}

// underRoot answers whether a granted path is on this worker's own filesystem root. It
// is the whole boundary check, and it is a string test on purpose: the interesting case
// is a path that does not exist HERE at all, which no stat can tell apart from a typo.
func underRoot(path, root string) bool {
	if root == "" {
		return false
	}
	rel, err := filepath.Rel(root, path)
	return err == nil && !strings.HasPrefix(rel, "..")
}

// terminalWithOutput writes ONE real output under the attempt's granted directory and
// sends a SUCCEEDED terminal that declares it. It returns the identical envelope, which
// the `dropack` arm replays when the ack arrives — the coordinator half of a lost
// TerminalAck — and which the `remote` arm sends once and is done with.
func (f *fakeControl) terminalWithOutput(emit func(*pb.AttemptTerminal),
	start *pb.StartAttempt) *pb.AttemptTerminal {
	layout, e := home.Open(flag("cozy-home", ""))
	if e != nil {
		f.say("no layout: %s", e.Message)
		return nil
	}
	dir := layout.AttemptDir(start.RequestId, start.Attempt)
	if err := os.MkdirAll(dir, 0o755); err != nil {
		f.say("cannot write under the grant: %v", err)
		return nil
	}
	// A minimal, real PNG: an output is bytes on disk, not a claim in a document.
	body, err := hex.DecodeString(onePixelPNG)
	if err != nil {
		f.say("bad fixture: %v", err)
		return nil
	}
	if err := os.WriteFile(filepath.Join(dir, "image"), body, 0o644); err != nil {
		f.say("cannot write the output: %v", err)
		return nil
	}
	sum := sha256.Sum256(body)
	t, _ := terminalFor(start.RequestId, start.Attempt, start.InvocationDigest,
		pb.TerminalStatus_TERMINAL_STATUS_SUCCEEDED, "one output, and an ack that will be lost")
	doc, err := canonical.Read(t.TerminalCanonical, &pb.TerminalBody{})
	if err != nil {
		f.say("cannot read back the terminal: %v", err)
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
		f.say("cannot canonicalize the manifest: %v", err)
		return nil
	}
	t.TerminalCanonical, t.TerminalDigest = written, canonical.Digest(written)
	f.say("terminal with ONE %d B output under %s", len(body), dir)
	emit(t)
	return t
}

// onePixelPNG is a 1x1 PNG, hex-encoded: the smallest thing that is really an image.
const onePixelPNG = "89504e470d0a1a0a0000000d4948445200000001000000010806000000" +
	"1f15c4890000000d49444154789c6360000002000100ffff03000006000557bfabd40000000049454e44ae426082"

// stealTerminal is a worker trying to write an attempt row it was never assigned.
func (f *fakeControl) stealTerminal(emit func(*pb.AttemptTerminal)) {
	requestID := flag("request", "")
	var attempt uint64
	fmt.Sscanf(flag("attempt", "1"), "%d", &attempt)
	spec, _ := hex.DecodeString(flag("spec", ""))
	t, _ := terminalFor(requestID, attempt, spec,
		pb.TerminalStatus_TERMINAL_STATUS_SUCCEEDED, "a terminal from a worker that does not own it")
	f.say("ARM: boot %s claims %s#%d, which it was never assigned", f.bootID, requestID, attempt)
	emit(t)
}

// randomHex uses crypto/rand, which speaks for every OS — the /dev/urandom spelling
// silently produced all-zero session ids on Windows, colliding every fake worker on one
// session (found by the windows-proc run: the "second writer" and "worker B" arms were
// really SESSION_COLLISION refusals).
func randomHex(n int) string {
	b := make([]byte, n)
	if _, err := rand.Read(b); err != nil {
		panic("no randomness for a fake boot id: " + err.Error())
	}
	return hex.EncodeToString(b)
}
