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

// TestRentalDownloadDelegationOutlivesItsExpiry is xs-007 row 5 as behaviour.
//
// Creator signs the download delegation a rented pod resolves its packages and models
// with. That credential has a lifetime — the hub refuses one that outlives an hour — and it
// used to be the thing that decided whether a materialization succeeded: 30 minutes, while
// 100 GB at 50 MB/s takes 33, so a large download failed as `worker.desired_state_refused`,
// Structural and never requeued. Nothing in that failure observed a stalled download.
//
// The rule now: a lapsed credential is answered by minting another one, and only OBSERVED
// lack of progress ends the work. The pod is the party that can see bytes, and it already
// refuses a refreshed plan that landed none; that verdict is what fails the request.
//
// Every arm below runs the real orchestrator against a real second implementation of the
// worker protocol over pinned TLS, with the real Creator signer and the hub's real
// delegation rule. Nothing is stubbed inside the code under test.
func TestRentalDownloadDelegationOutlivesItsExpiry(t *testing.T) {
	t.Run("re-signed while the pod is still working", func(t *testing.T) {
		// The credential ages out mid-download: the pod holds the first delegation past
		// its own expiry (a transfer in flight) and only then asks the hub for the next
		// plan, which the hub refuses because the delegation is spent.
		pod := &standInPod{holdFirstPastExpiry: true}
		o, instance := attachStandInRental(t, "delegation-resigned", 2*time.Second, pod)

		fatal(t, o.c.ConvergePackageSet(instance, delegatedPackages(), nil))

		if _, ok := waitEvent(o, "re-issuing the package set under a fresh one", 30*time.Second); !ok {
			t.Fatalf("the lapsed delegation was not re-signed:\n%s", pod.report())
		}
		seen := pod.await(t, 2, 30*time.Second)
		if !seen[1].accepted {
			t.Fatalf("the re-signed delegation was refused as well:\n%s", pod.report())
		}
		if !seen[1].expiry.After(seen[0].expiry) {
			t.Fatalf("the second delegation is not a fresh credential (%s then %s)",
				seen[0].expiry.UTC(), seen[1].expiry.UTC())
		}
		if !seen[1].signed {
			t.Fatalf("the re-issued delegation is not signed by this rental's Creator key")
		}
		if n := countEvents(o, "REFUSED before it was applied"); n != 0 {
			t.Fatalf("a still-working download was refused %d time(s):\n%s", n, pod.report())
		}
	})

	t.Run("refused when the pod reports no bytes landed", func(t *testing.T) {
		// The pod's own progress verdict: a refreshed plan expired having landed no new
		// byte. That is the observation the delegation clock was standing in for, and it
		// is still permanent — nothing is re-signed for a store that is not serving.
		pod := &standInPod{refusal: noBytesLandedRefusal}
		o, instance := attachStandInRental(t, "delegation-no-bytes", time.Minute, pod)

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

	t.Run("refused when a live delegation is reported lapsed", func(t *testing.T) {
		// The anti-spin fence. A pod reporting the credential expired while this owner's
		// copy is still live means the two ends disagree about the time; re-signing would
		// produce the same answer forever, so the refusal stands.
		pod := &standInPod{refusal: hubDelegationLapsedRefusal}
		o, instance := attachStandInRental(t, "delegation-skew", time.Minute, pod)

		fatal(t, o.c.ConvergePackageSet(instance, delegatedPackages(), nil))

		if _, ok := waitEvent(o, "REFUSED before it was applied", 30*time.Second); !ok {
			t.Fatalf("a lapse this owner's own clock contradicts was not refused:\n%s", pod.report())
		}
		pod.stayAt(t, 1, 3*time.Second)
	})
}

// hubDelegationLapsedRefusal is what a pod says when tensorhub refuses to mint a plan for
// an aged-out delegation: `worker_downloads.delegation_unauthorized` from
// `internal/workerdownloads/service.go`, rendered by `poddownloads.resolveRefusal` and
// wrapped by `materialize`, then forwarded verbatim as pod-supervisor's FailedPrecondition.
const hubDelegationLapsedRefusal = "refresh expired download plan: " +
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

type resolvedDelegation struct {
	expiry   time.Time
	accepted bool
	signed   bool
}

// standInPod is a second implementation of the pod side standing where a rented pod
// stands: WorkerControl and PodHost on one pinned TLS listener (proto-025). It claims,
// mints its own canonical snapshot, and answers PodHost.PreparePackageSet by resolving the
// delegation exactly as tensorhub's resolve route would. A refusal is the prepare stream's
// terminal REFUSED event carrying the hub's code and words, which is what pod-supervisor
// answers; the owner then sends the prepared placement_set itself on WorkerControl.
type standInPod struct {
	pb.UnimplementedWorkerControlServer
	pb.UnimplementedPodHostServer
	workerID, bootID, instance string
	creatorPublicKey           ed25519.PublicKey
	leafDigest                 []byte // the pinned leaf the owner's ClaimProof names

	// holdFirstPastExpiry keeps the first delegation in hand until it has aged out, the
	// way a transfer that is still landing bytes does.
	holdFirstPastExpiry bool
	// refusal, when set, is answered to every package_set without consulting the clock.
	refusal string

	mu    sync.Mutex
	seen  []resolvedDelegation
	epoch uint64
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
	if err := stream.Send(&pb.WorkerFrame{Msg: &pb.WorkerFrame_Snapshot{Snapshot: snapshot}}); err != nil {
		return nil
	}

	for {
		frame, err := stream.Recv()
		if err != nil {
			return nil
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
		if err := stream.Send(&pb.WorkerFrame{
			Msg: &pb.WorkerFrame_ObservedState{ObservedState: observed}}); err != nil {
			return nil
		}
	}
}

// PreparePackageSet is the host lane: the same delegation resolve, answered as prepare
// events. The hub's refusal code and words ride the terminal REFUSED event exactly as
// pod-supervisor forwards them, so the owner's lapse rule reads one text either way.
func (p *standInPod) PreparePackageSet(call *pb.PreparePackageSetCall,
	stream grpc.ServerStreamingServer[pb.PrepareEvent]) error {
	if call.GetClaim() == nil || call.Claim.ControlStreamEpoch != 0 ||
		!ed25519.Verify(p.creatorPublicKey, p.claimProof(call.Claim), call.Claim.Proof) {
		return status.Error(codes.Unauthenticated, "the host call carries no valid ClaimProof")
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
	if err := p.applyPackageSet(set.DownloadDelegation, set.DownloadDelegationSignature); err != nil {
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

// applyPackageSet is the pod's download edge: read the delegation, take as long over it as
// the arm says a transfer would, then present it to the hub's rule.
func (p *standInPod) applyPackageSet(delegation, signature []byte) error {
	document, err := canonical.Read(delegation, &pb.DownloadDelegation{})
	if err != nil {
		return fmt.Errorf("delegation is not canonical: %w", err)
	}
	expiry := time.Unix(document.Int("expires_at_unix"), 0)
	signed := ed25519.Verify(p.creatorPublicKey, delegation, signature)

	p.mu.Lock()
	first := len(p.seen) == 0
	p.mu.Unlock()

	if p.holdFirstPastExpiry && first {
		time.Sleep(time.Until(expiry) + 100*time.Millisecond)
	}
	resolveErr := errors.New(p.refusal)
	if p.refusal == "" {
		resolveErr = standInHubResolve(expiry, signed, time.Now())
	}

	p.mu.Lock()
	p.seen = append(p.seen, resolvedDelegation{expiry: expiry, accepted: resolveErr == nil, signed: signed})
	p.mu.Unlock()
	return resolveErr
}

// standInHubResolve is tensorhub's admission rule for a delegation and nothing else
// (`internal/workerdownloads/delegation.go` verifyDelegation): the Creator signature must
// verify, the credential must not have aged out, and it must not outlive the hub's
// one-hour ceiling. A refusal is rendered in the hub's own words.
func standInHubResolve(expiry time.Time, signed bool, now time.Time) error {
	if !signed {
		return errors.New("worker_downloads.delegation_unauthorized: delegation signature is invalid " +
			"(HTTP 401)")
	}
	if !expiry.After(now) || expiry.After(now.Add(time.Hour)) {
		return errors.New(hubDelegationLapsedRefusal)
	}
	return nil
}

func (p *standInPod) resolved() []resolvedDelegation {
	p.mu.Lock()
	defer p.mu.Unlock()
	return append([]resolvedDelegation(nil), p.seen...)
}

// await waits for the pod to have been presented `count` delegations.
func (p *standInPod) await(t *testing.T, count int, timeout time.Duration) []resolvedDelegation {
	t.Helper()
	deadline := time.Now().Add(timeout)
	for time.Now().Before(deadline) {
		if seen := p.resolved(); len(seen) >= count {
			return seen
		}
		time.Sleep(25 * time.Millisecond)
	}
	t.Fatalf("the pod was presented %d delegation(s), wanted %d:\n%s",
		len(p.resolved()), count, p.report())
	return nil
}

// stayAt proves the owner stopped: a permanent refusal must not become a re-sign loop.
func (p *standInPod) stayAt(t *testing.T, count int, settle time.Duration) {
	t.Helper()
	time.Sleep(settle)
	if seen := p.resolved(); len(seen) != count {
		t.Fatalf("a refused package set was re-signed: %d delegation(s), wanted %d:\n%s",
			len(seen), count, p.report())
	}
}

func (p *standInPod) report() string {
	out := ""
	for i, row := range p.resolved() {
		out += fmt.Sprintf("  delegation %d: expires %s accepted=%v signed=%v\n",
			i+1, row.expiry.UTC().Format(time.RFC3339), row.accepted, row.signed)
	}
	if out == "" {
		return "  (the pod was presented no delegation)"
	}
	return out
}

// ---------------------------------------------------------------- the rented machine

// attachStandInRental brings up one rental: a real Creator identity, a real pinned TLS
// leaf, the real claim-proof and delegation signers, and this test's own delegation
// lifetime. The lifetime is the security bound the product keeps — shortening it is what
// lets a lapse be observed in seconds instead of fifty minutes. Everything else is the
// product's own code, including the bytes that get signed.
func attachStandInRental(t *testing.T, name string, lifetime time.Duration,
	pod *standInPod) (*owner, string) {
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
		options.RentalPackageSet = func(c *orchestrator.WorkerConnection,
			packages []*pb.DownloadPackageRef, models []*pb.DownloadModelRef) ([]byte, []byte, *exit.Error) {
			return rental.SignDownloadDelegation(layout, c, packages, models,
				time.Now().Add(lifetime))
		}
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
