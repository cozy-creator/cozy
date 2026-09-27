package producttest

import (
	"context"
	"crypto/ecdsa"
	"crypto/ed25519"
	"crypto/elliptic"
	"crypto/rand"
	"crypto/tls"
	"crypto/x509"
	"crypto/x509/pkix"
	"encoding/base64"
	"encoding/json"
	"encoding/pem"
	"errors"
	"fmt"
	"math/big"
	"net"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"

	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/credentials"
	"google.golang.org/grpc/status"

	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/home"
	"github.com/cozy-creator/cozy/internal/media"
	"github.com/cozy-creator/cozy/internal/mediawire"
	"github.com/cozy-creator/cozy/internal/orchestrator"
	"github.com/cozy-creator/cozy/internal/rental"
	"github.com/cozy-creator/cozy/internal/secret"
	"github.com/cozy-creator/cozy/internal/workertls"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
)

// TestRentalDownloadSetCarriesNoCredential is xs-007 row 5, settled by deletion.
//
// Creator used to sign a DownloadDelegation the rented pod presented as its hub
// credential. That credential had a lifetime — the hub refused one outliving an hour —
// and it, not the transfer, decided whether a materialization succeeded: 100 GB at
// 50 MB/s takes 33 minutes, so a large download failed as `worker.desired_state_refused`,
// Structural and never requeued, with nothing in that failure having observed a stalled
// download. The answer used to be re-signing mid-transfer. The owner deleted the whole
// authorization plane instead (ruling 2026-09-03): packages and repos are public, the
// hub's closure and presign routes take no credential, and the document Creator authors
// is DESIRED STATE alone.
//
// So a download can no longer become unauthorized by running long, and only OBSERVED lack
// of progress ends the work. The pod is the party that can see bytes, and it refuses a
// refreshed plan that landed none; that verdict is what fails the request.
//
// Every arm below runs the real orchestrator against a real second implementation of the
// worker protocol over pinned TLS, with the real Creator claim signer and the real
// download-set author. Nothing is stubbed inside the code under test.
func TestRentalDownloadSetCarriesNoCredential(t *testing.T) {
	t.Run("no credential to lapse", func(t *testing.T) {
		// RED ARM for the deletion. The pod holds the set far longer than the old
		// credential would have lived, then prepares. There is nothing to expire, so it
		// succeeds on the FIRST presentation: no re-issue, no refusal — and the document
		// it was handed carries no expiry, no rental/worker/boot binding, and no
		// signature.
		pod := &standInPod{holdFor: 2 * time.Second}
		o, instance := attachStandInRental(t, "download-set-no-credential", pod)

		fatal(t, o.c.ConvergePackageSet(instance, delegatedPackages(), nil))

		seen := pod.await(t, 1, 30*time.Second)
		if !seen[0].accepted {
			t.Fatalf("a credential-free download set was refused:\n%s", pod.report())
		}
		if seen[0].credentialFields != "" {
			t.Fatalf("the download set still carries credential field(s) %s:\n%s",
				seen[0].credentialFields, pod.report())
		}
		if seen[0].signature {
			t.Fatalf("the download set still carries an Ed25519 signature:\n%s", pod.report())
		}
		if n := countEvents(o, "REFUSED before it was applied"); n != 0 {
			t.Fatalf("a slow download was refused %d time(s):\n%s", n, pod.report())
		}
		pod.stayAt(t, 1, 2*time.Second)
	})

	t.Run("refused when the pod reports no bytes landed", func(t *testing.T) {
		// The pod's own progress verdict: a refreshed plan expired having landed no new
		// byte. That is the observation the credential clock was standing in for, and it
		// is still permanent — nothing is re-signed for a store that is not serving.
		pod := &standInPod{refusal: noBytesLandedRefusal}
		o, instance := attachStandInRental(t, "download-set-no-bytes", pod)

		fatal(t, o.c.ConvergePackageSet(instance, delegatedPackages(), nil))

		line, ok := waitEvent(o, "REFUSED before it was applied", 30*time.Second)
		if !ok {
			t.Fatalf("a download that landed no bytes was not refused:\n%s", pod.report())
		}
		if !strings.Contains(line, "no new bytes landed") {
			t.Fatalf("the refusal does not name the pod's progress verdict: %s", line)
		}
		pod.stayAt(t, 1, 3*time.Second)
	})

	t.Run("a credential refusal is now an ordinary permanent one", func(t *testing.T) {
		// The anti-spin fence, now unconditional. The owner used to answer this exact
		// text by re-issuing under a fresh credential. No credential exists to refresh,
		// so the refusal stands and the owner stops — one presentation, no loop.
		pod := &standInPod{refusal: retiredCredentialRefusal}
		o, instance := attachStandInRental(t, "download-set-credential-refusal", pod)

		fatal(t, o.c.ConvergePackageSet(instance, delegatedPackages(), nil))

		if _, ok := waitEvent(o, "REFUSED before it was applied", 30*time.Second); !ok {
			t.Fatalf("a credential-shaped refusal was not refused:\n%s", pod.report())
		}
		if n := countEvents(o, "re-issuing the package set under a fresh one"); n != 0 {
			t.Fatalf("the deleted re-sign path ran %d time(s):\n%s", n, pod.report())
		}
		pod.stayAt(t, 1, 3*time.Second)
	})
}

// retiredCredentialRefusal is the text a pod used to send when tensorhub refused to mint
// a plan for an aged-out delegation. Nothing produces it any more; it is kept here as the
// exact input that once bought a re-sign, to prove it now buys nothing.
const retiredCredentialRefusal = "refresh expired download plan: " +
	"worker_downloads.delegation_unauthorized: delegation is expired or exceeds the " +
	"one-hour lifetime; send a fresh delegation signed by this rental's Creator key (HTTP 401)"

// noBytesLandedRefusal is the pod's OWN progress verdict from `poddownloads.materialize`:
// a refreshed plan that expired without one new byte on disk.
const noBytesLandedRefusal = "refreshed download plan 1 expired with no new bytes landed " +
	"(0 of 107374182400): download grant expired"

func delegatedPackages() []*pb.DownloadPackageRef {
	return []*pb.DownloadPackageRef{{
		Package: "acme/diffusion", Release: "0.4.2",
	}}
}

// ---------------------------------------------------------------- the stand-in pod

// presentedDownloadSet is what the pod was handed: whether it prepared, and the two
// facts the deletion is about — any surviving credential field, and any signature.
type presentedDownloadSet struct {
	document         []byte
	accepted         bool
	credentialFields string
	signature        bool
}

// standInPod is a second implementation of the pod side standing where a rented pod
// stands: WorkerControl and PodHost on one pinned TLS listener (proto-025). It claims,
// mints its own canonical snapshot, and answers PodHost.PreparePackageSet by reading the
// desired download set the way pod-supervisor does. A refusal is the prepare stream's
// terminal REFUSED event carrying the hub's code and words, which is what pod-supervisor
// answers; the owner then sends the prepared placement_set itself on WorkerControl.
type standInPod struct {
	controlDefaults
	pb.UnimplementedPodHostServer
	workerID, bootID, instance string
	creatorPublicKey           ed25519.PublicKey
	leafDigest                 []byte // the pinned leaf the owner's ClaimProof names

	// holdFor keeps the first download set in hand this long before preparing, the way a
	// transfer that is still landing bytes does. Nothing expires while it waits.
	holdFor time.Duration
	// refusal, when set, is answered to every package_set.
	refusal string
	// unimplemented, once closed, makes this pod answer its control stream the way a pod
	// that DELETED a lane answers one: codes.Unimplemented over the whole stream. A nil
	// channel never fires, so an arm that does not set it is untouched.
	unimplemented chan struct{}
	// prepareStatus, when set, ends every prepare stream with that gRPC status instead of
	// any prepare event — the way a host answers a desire it cannot serve at all.
	prepareStatus *status.Status
	// prepareCalls counts PreparePackageSet calls. It is the owner's re-issue rate seen
	// from the other end, and the only honest measure of a desired-state retry loop.
	prepareCalls uint64
	// onSession, when set, is called once per accepted control stream with a sender bound
	// to that stream and the stream's own envelope. It is how an arm drives a lane this
	// pod does not otherwise speak, on the real wire and behind the real fence.
	onSession func(send func(*pb.WorkerFrame) error, ownerEpoch, controlEpoch uint64, bootID string)

	mu     sync.Mutex
	sendMu sync.Mutex
	seen   []presentedDownloadSet
	epoch  uint64
}

func (p *standInPod) ProtocolInfo(context.Context, *pb.ProtocolInfoRequest) (*pb.ProtocolInfoResult, error) {
	return &pb.ProtocolInfoResult{WireMinor: pb.WireMinor, MinimumWireMinor: pb.MinCompatibleWireMinor, SupportsRentalKeepalive: true}, nil
}

// claims counts the control streams this pod has accepted a Claim on. It is the owner's
// redial rate seen from the other end, and the only honest measure of a reconnect loop.
func (p *standInPod) claims() uint64 {
	p.mu.Lock()
	defer p.mu.Unlock()
	return p.epoch
}

// prepares counts the prepare streams this pod has been asked to open.
func (p *standInPod) prepares() uint64 {
	p.mu.Lock()
	defer p.mu.Unlock()
	return p.prepareCalls
}

func (p *standInPod) WatchProgress(_ *pb.ProgressOpen, stream pb.WorkerControl_WatchProgressServer) error {
	<-stream.Context().Done()
	return nil
}

func (p *standInPod) Control(stream pb.WorkerControl_ControlServer) error {
	frame, err := stream.Recv()
	if err != nil || frame.GetClaim() == nil {
		return nil
	}
	claim := frame.GetClaim()
	p.mu.Lock()
	p.epoch++
	epoch := p.epoch
	p.mu.Unlock()

	ack := &pb.ClaimAck{
		Accepted: true, WireMinor: pb.WireMinor, WorkerId: p.workerID,
		WorkerInstanceId: p.instance, RecordOwnerEpoch: claim.RecordOwnerEpoch,
		ControlStreamEpoch: epoch, WorkerBootId: p.bootID,
		Resources: &pb.WorkerResources{Platform: "stand-in", Backend: "cpu", DeviceName: "CPU"},
	}
	if err := stream.Send(&pb.WorkerFrame{Msg: &pb.WorkerFrame_ClaimAck{ClaimAck: ack}}); err != nil {
		return nil
	}

	emptySet, emptySetDigest, err := canonical.Identity(&pb.PlacementSet{})
	if err != nil {
		return err
	}
	bodyBytes, bodyDigest, err := canonical.Identity(&pb.WorkerSnapshotBody{
		WorkerPhase:                pb.WorkerPhase_WORKER_PHASE_ONLINE,
		AdmissionState:             pb.AdmissionState_ADMISSION_STATE_CLOSED,
		AdmissionEpoch:             1,
		AcceptedPlacementSetDigest: emptySetDigest,
	})
	if err != nil {
		return err
	}
	snapshot := &pb.WorkerSnapshot{
		SnapshotId: "snp-standin", SnapshotDigest: bodyDigest,
		SnapshotCanonicalBytes: bodyBytes, AcceptedPlacementSetCanonicalBytes: emptySet,
		RecordOwnerEpoch: claim.RecordOwnerEpoch, ControlStreamEpoch: epoch, WorkerBootId: p.bootID,
	}
	send := func(m *pb.WorkerFrame) error {
		p.sendMu.Lock()
		defer p.sendMu.Unlock()
		return stream.Send(m)
	}
	if err := send(&pb.WorkerFrame{Msg: &pb.WorkerFrame_Snapshot{Snapshot: snapshot}}); err != nil {
		return nil
	}
	if p.onSession != nil {
		go p.onSession(send, claim.RecordOwnerEpoch, epoch, p.bootID)
	}

	frames := make(chan *pb.RecordOwnerFrame)
	go func() {
		defer close(frames)
		for {
			received, err := stream.Recv()
			if err != nil {
				return
			}
			select {
			case frames <- received:
			case <-stream.Context().Done():
				return
			}
		}
	}()

	for {
		var frame *pb.RecordOwnerFrame
		select {
		case <-p.unimplemented:
			// What a pod that deleted a lane actually answers. Returning a status from
			// the bidi handler tears down the whole session, which is why a status the
			// owner does not treat as terminal becomes a reconnect loop.
			return status.Error(codes.Unimplemented, "this lane is deleted")
		case received, open := <-frames:
			if !open {
				return nil
			}
			frame = received
		}
		desired := frame.GetDesiredState()
		if desired == nil {
			continue
		}
		if desired.GetPlacementSet() == nil {
			// A host mode on the control stream is exactly what proto-025 retired.
			return status.Error(codes.FailedPrecondition, "host modes are PodHost calls")
		}
		observed := &pb.ObservedWorkerState{
			AcceptedDesiredStateRevision: desired.Revision, ConvergedRevision: desired.Revision,
			AcceptedPlacementSetDigest: emptySetDigest,
			WorkerPhase:                pb.WorkerPhase_WORKER_PHASE_ONLINE,
			AppliedWireMinor:           pb.WireMinor,
			AdmissionState:             pb.AdmissionState_ADMISSION_STATE_OPEN,
			AdmissionEpoch:             1, AvailableAttemptSlots: 1,
			RecordOwnerEpoch: claim.RecordOwnerEpoch, ControlStreamEpoch: epoch,
			WorkerBootId: p.bootID,
		}
		if err := send(&pb.WorkerFrame{
			Msg: &pb.WorkerFrame_ObservedState{ObservedState: observed}}); err != nil {
			return nil
		}
	}
}

// PreparePackageSet is the host lane, answered as prepare events. A refusal's code and
// words ride the terminal REFUSED event exactly as pod-supervisor forwards them.
func (p *standInPod) PreparePackageSet(call *pb.PreparePackageSetCall,
	stream grpc.ServerStreamingServer[pb.PrepareEvent]) error {
	if call.GetClaim() == nil || call.Claim.ControlStreamEpoch != 0 ||
		!ed25519.Verify(p.creatorPublicKey, p.claimProof(call.Claim), call.Claim.Proof) {
		return status.Error(codes.Unauthenticated, "the host call carries no valid ClaimProof")
	}
	p.mu.Lock()
	p.prepareCalls++
	p.mu.Unlock()
	if p.prepareStatus != nil {
		return p.prepareStatus.Err()
	}
	set := call.GetPackageSet()
	if set == nil {
		return status.Error(codes.InvalidArgument, "no package set")
	}
	// MINOR 31 (xs-019): the pod host's facts fence, exactly as workerhost spells it.
	if call.Application == "" || len(call.LockedRequirements) == 0 ||
		len(call.LockedRequirements) > pb.MaxLockedRequirementsBytes ||
		len(call.ModelSlotPaths) > pb.MaxModelSlotPaths {
		return status.Error(codes.InvalidArgument,
			"PreparePackageSet requires the release facts: application, bounded model_slot_paths, and the locked requirements export")
	}
	if err := stream.Send(&pb.PrepareEvent{Stage: pb.PrepareStage_PREPARE_STAGE_RESOLVED}); err != nil {
		return err
	}
	if err := p.applyPackageSet(set.DownloadDelegation, set.DownloadDelegationSignature); err != nil { //nolint:staticcheck // wire field renamed by proto-033
		code, detail, _ := strings.Cut(err.Error(), ": ")
		return stream.Send(&pb.PrepareEvent{Stage: pb.PrepareStage_PREPARE_STAGE_REFUSED,
			SafeCode: code, SafeDetail: detail})
	}
	emptySet, emptySetDigest, err := canonical.Identity(&pb.PlacementSet{})
	if err != nil {
		return err
	}
	return stream.Send(&pb.PrepareEvent{Stage: pb.PrepareStage_PREPARE_STAGE_PREPARED,
		PlacementSet: &pb.DesiredPlacementSet{PlacementSetDigest: emptySetDigest,
			PlacementSetCanonicalBytes: emptySet}})
}

// claimProof rebuilds the ClaimProof/1 bytes the owner signed for this pod, from the pod's
// own facts and the pinned leaf, exactly as pod-supervisor verifies a host call.
func (p *standInPod) claimProof(claim *pb.Claim) []byte {
	body, err := canonical.Bytes(&pb.ClaimProof{RecordOwnerEpoch: claim.RecordOwnerEpoch,
		WorkerId: p.workerID, WorkerBootId: p.bootID, WorkerTlsCertificateDigest: p.leafDigest})
	if err != nil {
		return nil
	}
	return body
}

// applyPackageSet is the pod's download edge: read the desired download set, take as long
// over it as the arm says a transfer would, and record what the document actually carried.
// There is no credential to check — the hub's closure and presign routes take none — so
// only an arm's own refusal can fail a preparation.
func (p *standInPod) applyPackageSet(desired, signature []byte) error {
	document, err := canonical.Read(desired, &pb.DownloadDelegation{})
	if err != nil {
		return fmt.Errorf("download set is not canonical: %w", err)
	}
	// The credential fields the deletion removed. A document carrying any of them means
	// some part of the authorization plane grew back.
	surviving := []string(nil)
	for _, field := range []string{"expires_at_unix", "rental_id", "worker_boot_id",
		"worker_id", "worker_tls_certificate_digest"} {
		if _, present := document[field]; present {
			surviving = append(surviving, field)
		}
	}
	presented := presentedDownloadSet{document: append([]byte(nil), desired...), credentialFields: strings.Join(surviving, ","),
		signature: len(signature) != 0}

	p.mu.Lock()
	first := len(p.seen) == 0
	p.mu.Unlock()
	if p.holdFor > 0 && first {
		time.Sleep(p.holdFor)
	}
	var refusal error
	if p.refusal != "" {
		refusal = errors.New(p.refusal)
	}
	presented.accepted = refusal == nil

	p.mu.Lock()
	p.seen = append(p.seen, presented)
	p.mu.Unlock()
	return refusal
}

func (p *standInPod) resolved() []presentedDownloadSet {
	p.mu.Lock()
	defer p.mu.Unlock()
	return append([]presentedDownloadSet(nil), p.seen...)
}

// await waits for the pod to have been presented `count` download sets.
func (p *standInPod) await(t *testing.T, count int, timeout time.Duration) []presentedDownloadSet {
	t.Helper()
	deadline := time.Now().Add(timeout)
	for time.Now().Before(deadline) {
		if seen := p.resolved(); len(seen) >= count {
			return seen
		}
		time.Sleep(25 * time.Millisecond)
	}
	t.Fatalf("the pod was presented %d download set(s), wanted %d:\n%s",
		len(p.resolved()), count, p.report())
	return nil
}

// stayAt proves the owner stopped: nothing re-issues a set the pod already answered.
func (p *standInPod) stayAt(t *testing.T, count int, settle time.Duration) {
	t.Helper()
	time.Sleep(settle)
	if seen := p.resolved(); len(seen) != count {
		t.Fatalf("the package set was re-issued: %d download set(s), wanted %d:\n%s",
			len(seen), count, p.report())
	}
}

func (p *standInPod) report() string {
	out := ""
	for i, row := range p.resolved() {
		fields := row.credentialFields
		if fields == "" {
			fields = "(none)"
		}
		out += fmt.Sprintf("  download set %d: accepted=%v credential fields=%s signature=%v\n",
			i+1, row.accepted, fields, row.signature)
	}
	if out == "" {
		return "  (the pod was presented no download set)"
	}
	return out
}

// ---------------------------------------------------------------- the rented machine

// attachStandInRental brings up one rental: a real Creator identity, a real pinned TLS
// leaf, the real claim-proof signer, and the real download-set author. There is no
// lifetime to configure — the credential that had one is deleted. Everything is the
// product's own code, including the exact bytes the pod is handed.
func attachStandInRental(t *testing.T, name string, pod *standInPod) (*owner, string) {
	t.Helper()
	rentalID := "rental-" + name
	pod.workerID = "wrk-" + name
	pod.bootID = "boot-" + name
	pod.instance = "ins-standin-" + name

	// The owner's root is made by hostOwner, so everything on disk is minted after it.
	var layout home.Layout
	var connection *orchestrator.WorkerConnection
	o := hostOwner(t, name, func(options *orchestrator.Options) {
		options.Rentals = func(id string) (*orchestrator.RemoteTarget, *exit.Error) {
			if id != rentalID || connection == nil {
				return nil, exit.New(exit.NotFound, "no rental %s", id)
			}
			return &orchestrator.RemoteTarget{Connection: connection}, nil
		}
		options.ObserveRental = func(orchestrator.RentalObservation) *exit.Error { return nil }
		options.RentalClaimProof = func(c *orchestrator.WorkerConnection, epoch uint64) ([]byte, *exit.Error) {
			return rental.ClaimProof(layout)(c, epoch)
		}
		options.RentalPackageSet = rental.PackageSetSource()
		options.RentalPrepareFacts = func(_ context.Context, _ *orchestrator.WorkerConnection,
			ref *pb.DownloadPackageRef) (orchestrator.PrepareFacts, *exit.Error) {
			return testPrepareFacts(ref.Package, ref.Release), nil
		}
	})
	layout = o.l

	// The rental's Creator identity is minted before the paid POST in production; here it
	// is minted and adopted directly so the signers hold the exact key they would hold.
	identity := adoptCreatorIdentity(t, layout, rentalID)
	key, err := base64.RawURLEncoding.DecodeString(identity.PublicKey())
	must(t, err)
	pod.creatorPublicKey = ed25519.PublicKey(key)

	certPath := standInCertificate(t, o.root)
	pin, err := workertls.LoadPin(certPath)
	must(t, err)
	pod.leafDigest = pin.Digest()
	connection = &orchestrator.WorkerConnection{
		RentalID: rentalID, WorkerID: pod.workerID, WorkerBootID: pod.bootID,
		CACert: certPath, Addr: servePod(t, pod, certPath),
		Media: &media.Spec{Addr: standInMedia(t), Token: secret.New("stand-in")},
	}

	instance, _, _, e := o.c.EnsureRental(rentalID)
	fatal(t, e)
	return o, instance
}

func adoptCreatorIdentity(t *testing.T, l home.Layout, rentalID string) rental.CreatorIdentity {
	t.Helper()
	identity, e := rental.PendingCreatorIdentity(l, "op-"+rentalID)
	fatal(t, e)
	must(t, os.Rename(l.PendingRentalCreatorIdentity("op-"+rentalID),
		l.RentalCreatorIdentity(rentalID)))
	return identity
}

// servePod hosts the pod's WorkerControl on the exact certificate the rental pins.
func servePod(t *testing.T, pod *standInPod, certPath string) string {
	t.Helper()
	certificate, err := tls.LoadX509KeyPair(certPath, certPath+".key")
	must(t, err)
	listener, err := net.Listen("tcp", "127.0.0.1:0") //cozy:allow the stand-in pod's own control listener
	must(t, err)
	server := grpc.NewServer(grpc.Creds(credentials.NewServerTLSFromCert(&certificate)))
	pb.RegisterWorkerControlServer(server, pod)
	pb.RegisterPodHostServer(server, pod)
	go func() { _ = server.Serve(listener) }()
	t.Cleanup(server.Stop)
	return listener.Addr().String()
}

// standInMedia is the pod's byte plane, which a connected worker must have: the owner
// dials it and reads its contract before anything else happens.
func standInMedia(t *testing.T) string {
	t.Helper()
	revision := mediawire.ContractRev
	health, err := json.Marshal(mediawire.Health{Service: mediawire.Service, ContractRev: &revision})
	must(t, err)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write(health)
	}))
	t.Cleanup(server.Close)
	return server.Listener.Addr().String()
}

// standInCertificate mints the self-signed leaf the rental pins, as a pod's own readiness
// certificate is pinned: the owner admits these exact bytes and nothing else.
func standInCertificate(t *testing.T, root string) string {
	t.Helper()
	key, err := ecdsa.GenerateKey(elliptic.P256(), rand.Reader)
	must(t, err)
	template := &x509.Certificate{
		SerialNumber:          big.NewInt(1),
		Subject:               pkix.Name{CommonName: "stand-in-pod"},
		NotBefore:             time.Now().Add(-time.Hour),
		NotAfter:              time.Now().Add(24 * time.Hour),
		DNSNames:              []string{"stand-in-pod", workertls.ServerName},
		IPAddresses:           []net.IP{net.ParseIP("127.0.0.1")},
		KeyUsage:              x509.KeyUsageDigitalSignature | x509.KeyUsageCertSign,
		ExtKeyUsage:           []x509.ExtKeyUsage{x509.ExtKeyUsageServerAuth},
		IsCA:                  true,
		BasicConstraintsValid: true,
	}
	der, err := x509.CreateCertificate(rand.Reader, template, template, &key.PublicKey, key)
	must(t, err)
	certPath := filepath.Join(root, "pod.pem")
	must(t, os.WriteFile(certPath,
		pem.EncodeToMemory(&pem.Block{Type: "CERTIFICATE", Bytes: der}), 0o600))
	keyDER, err := x509.MarshalECPrivateKey(key)
	must(t, err)
	must(t, os.WriteFile(certPath+".key",
		pem.EncodeToMemory(&pem.Block{Type: "EC PRIVATE KEY", Bytes: keyDER}), 0o600))
	return certPath
}

func TestUnversionedCheckpointReachesRentalDownloadSet(t *testing.T) {
	pod := &standInPod{}
	o, instance := attachStandInRental(t, "checkpoint-no-release", pod)
	packages := delegatedPackages()
	digest := "sha256:" + strings.Repeat("4", 64)
	models := []*pb.DownloadModelRef{{Package: packages[0].Package, Slot: "prepare.models.source", Model: "proof/checkpoint", Manifest: digest}}
	fatal(t, o.c.ConvergePackageSet(instance, packages, models))
	seen := pod.await(t, 1, 10*time.Second)
	document, err := canonical.Read(seen[0].document, &pb.DownloadDelegation{})
	must(t, err)
	rows := document.List("models")
	if !seen[0].accepted || len(rows) != 1 || rows[0].Str("manifest") != digest || rows[0].Str("release") != "" || rows[0].Str("lane") != "" {
		t.Fatalf("checkpoint gained a synthetic release or was lost: %s", seen[0].document)
	}
	// Half a release label beside an exact manifest is that manifest as a checkpoint.
	for _, pair := range [][2]string{{"release", ""}, {"", "lane"}} {
		models[0].Release, models[0].Lane = pair[0], pair[1]
		raw, problem := rental.DownloadSet(append(packages, packages...), models)
		fatal(t, problem)
		document, err := canonical.Read(raw, &pb.DownloadDelegation{})
		must(t, err)
		rows := document.List("models")
		if len(document.List("packages")) != len(packages) || len(rows) != 1 || rows[0].Str("manifest") != digest ||
			rows[0].Str("release") != "" || rows[0].Str("lane") != "" {
			t.Fatalf("partial label %v was not fetched as the exact checkpoint: %s", pair, raw)
		}
		if models[0].Release != pair[0] || models[0].Lane != pair[1] {
			t.Fatal("normalization mutated the caller's selection")
		}
	}
}
