package main

import (
	"bytes"
	"crypto/sha256"
	"crypto/tls"
	"crypto/x509"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"syscall"
	"time"

	"github.com/cozy-creator/cozy-creator/internal/app"
	"github.com/cozy-creator/cozy-creator/internal/config"
	"github.com/cozy-creator/cozy-creator/internal/home"
	"github.com/cozy-creator/cozy-creator/internal/media"
	"github.com/cozy-creator/cozy-creator/internal/mediawire"
	"github.com/cozy-creator/cozy-creator/internal/records"
	"github.com/cozy-creator/cozy-creator/internal/workertls"
)

// cl-015's section: RENT A POD, ATTACH ITS WORKER, ROUTE TO IT, AND SEE WHERE THE BYTES
// ACTUALLY ARE.
//
// Every arm drives the product binary as a user types it, against two real peers this
// product did not co-develop with: `podHub` speaking the hub's rental routes, and
// `fakeworker` hosting `WorkerControl` behind TLS with the certificate the rental pins.
// No card is billed — what is proved is the OWNER side, which is the half that has to be
// right before a real pod is worth renting.
//
// THE POD HAS ITS OWN FILESYSTEM. That is not a detail: an earlier version of this
// section handed the "pod" the LocalService's own root, so client-local `file://` grants
// resolved by accident and the arm over the boundary was green about nothing. With the
// roots split, the boundary is where it really is, and the two arms that matter — a pod
// refusing a destination it cannot reach, and an owner refusing to ack an output it does
// not hold — are observations rather than assertions.
//
// AND NOW THERE IS A WAY ACROSS IT (#506a/#506b). Every pod this hub provisions runs a
// SECOND process beside its worker: the real `cozy-media` binary, holding its own keys,
// coupled to the worker by the filesystem alone. So the leg below is the whole product
// path — the binding-plan RECORDS are delivered to the pod and re-hashed there, the
// payload is UPLOADED to the pod, the worker reads it off the pod's own disk, writes into
// the pod directory the owner reserved, and the owner FETCHES the bytes back and verifies
// them before it acks. Nothing in it is stubbed, and the arms say where each byte was.

// The whole section runs on the WEIGHTLESS fixture: `tile` (a scalar-input, image-output
// entrypoint) is what every pinned run and the local unpinned run invoke, and
// `video_transport` carries the image->MP4 asset leg. No model, no weights, no GPU — the
// rental proof is about the OWNER side, and the shared box runs no inference.
const (
	transportEndpointRef       = "cozy/weightless"
	rentalEndpointRef          = transportEndpointRef + "/v1/tile"
	transportRentalEndpointRef = transportEndpointRef + "/v1/video_transport"
	localTileRef               = rentalEndpointRef
	rentalEntrypoint           = "tile"
	tilePayload                = "size=8"
)

// accelerator is the provider-neutral model every rental in this section asks for and
// every stand-in pod reports (`--accelerator`, default NVIDIA H200).
var accelerator = "NVIDIA H200"

// otherAccelerator is any accelerator that is not the one under test, for the arm that
// changes a body under a reused idempotency key.
func otherAccelerator() string {
	if accelerator == "NVIDIA H100" {
		return "NVIDIA H200"
	}
	return "NVIDIA H100"
}

func sectionRent() {
	root := flag("home", filepath.Join(os.TempDir(), "cozy-live", "cl015-rent"))
	accelerator = flag("accelerator", accelerator)
	must("clearing the root", os.RemoveAll(root))
	must("creating the root", os.MkdirAll(root, 0o755))

	head("the weightless endpoint contract, installed the way a user installs it")
	installTransportEndpoint(root)
	release, plans := endpointIdentityFor(root, transportEndpointRef)
	transportRelease, transportPlans := release, plans
	// THE HUB'S CONTROL TRUTH: every stand-in hub below authors Tensorhub's exact
	// acquisition-attempt snapshot from the release this host installed.
	defaultPodControl = localPodControl(root, rentalEndpointRef)
	check("the stand-in hub's control closure names the release this host serves",
		defaultPodControl.release == release && len(defaultPodControl.plans) == len(plans),
		defaultPodControl.release)
	port := freePort(2990)
	svc := startService(root, port, false)
	defer svc.stop()
	// The honest pod: it reads the granted input off its own disk, writes where the grant
	// says, and refuses a grant it cannot reach. It DECLARES the release it serves, which
	// is what the owner's pin is checked against.
	hub := newPodHub(podHubSpec{Dir: filepath.Join(root, "hub"), Arm: "remote", Release: release})
	defer hub.close()
	fmt.Printf("  the stand-in hub is at %s · pod filesystem %s\n", hub.url(), hub.pods)
	fmt.Printf("  this host serves %s · %d entrypoint(s) · plan %s\n",
		release, len(plans), shortID(planFor(plans, rentalEntrypoint)))
	fmt.Printf("  transport-only contract %s · plan %s (no H3 inference claim)\n",
		transportRelease, shortID(planFor(transportPlans, "video_transport")))

	head("the credential arms come FIRST: a rental is a first-party write")
	code, out := cozyRunEnv(root, []string{"TENSORHUB_URL=" + hub.url()},
		"rent", rentalEndpointRef, "--accelerator", accelerator, "--reason", "cl-015 live")
	check("no configured token -> 5, answered BEFORE the dial",
		code == 5 && strings.Contains(out, "no admin token is configured"), firstLine(out))
	code, out = cozyRunEnv(root,
		[]string{"TENSORHUB_URL=" + hub.url(), "TENSORHUB_TOKEN=" + randomHex(16)},
		"rent", rentalEndpointRef, "--accelerator", accelerator, "--reason", "cl-015 live")
	check("a FOREIGN token -> 5 carrying the hub's own typed refusal",
		code == 5 && strings.Contains(out, "rental.unauthenticated"), firstLine(out))

	head("paid POST response loss: the durable operation retries once and never buys twice")
	lost := newPodHub(podHubSpec{Dir: filepath.Join(root, "hub-response-loss"), Arm: "remote",
		Release: release, LoseFirstRentResponse: true})
	defer lost.close()
	operationKey := "cl-019-response-loss"
	code, out = cozyRunEnv(root, lost.env(), "rent", rentalEndpointRef, "--accelerator", accelerator,
		"--idempotency-key", operationKey, "--reason", "cl-019 response-loss arm")
	check("the first caller sees a transport failure after the hub committed",
		code == 9 && lost.created() == 1, fmt.Sprintf("exit %d · creates %d", code, lost.created()))
	pending, _ := filepath.Glob(filepath.Join(root, "rentals", "pending-*.token"))
	var pendingToken []byte
	if len(pending) == 1 {
		pendingToken, _ = os.ReadFile(pending[0])
	}
	check("the original renter token survived under the pending operation",
		len(pending) == 1 && len(strings.TrimSpace(string(pendingToken))) == 64,
		fmt.Sprintf("%d pending token(s)", len(pending)))
	code, out = cozyRunEnv(root, lost.env(), "rent", rentalEndpointRef, "--accelerator", accelerator,
		"--idempotency-key", operationKey, "--reason", "retry prose cannot rewrite the ask")
	lostRental := field(out, "rental")
	finalToken, _ := os.ReadFile(filepath.Join(root, "rentals", lostRental+".token"))
	check("the retry attaches the SAME rental and token with one provider create",
		code == 0 && strings.HasPrefix(lostRental, "rnt-") && lost.created() == 1 &&
			bytes.Equal(pendingToken, finalToken),
		fmt.Sprintf("exit %d · rental %s · creates %d", code, lostRental, lost.created()))
	code, out = cozyRunEnv(root, lost.env(), "rent", rentalEndpointRef, "--accelerator", otherAccelerator(),
		"--idempotency-key", operationKey,
		"--reason", "changed body")
	check("the same key with a changed body conflicts before another create",
		code == 13 && lost.created() == 1, fmt.Sprintf("exit %d · creates %d", code, lost.created()))

	head("the ask names its hardware and its reason, or it is a usage refusal")
	code, out = cozyRunEnv(root, hub.env(), "rent", rentalEndpointRef, "--reason", "cl-015 live")
	check("no --accelerator -> 2", code == 2 && strings.Contains(out, "--accelerator is required"), firstLine(out))
	code, out = cozyRunEnv(root, hub.env(), "rent", rentalEndpointRef, "--accelerator", accelerator)
	check("no --reason -> 2: the hub records why before it spends",
		code == 2 && strings.Contains(out, "--reason is required"), firstLine(out))

	head("cozy rent — provision, poll to ready, PIN the triple")
	t0 := time.Now()
	rentalA, out := rentOne(root, hub, "cl-015 target A")
	rentMS := elapsedMS(t0)
	fmt.Println(indent(out))
	check("cozy rent -> 0 with a rental id", strings.HasPrefix(rentalA, "rnt-"), rentalA)
	check("it reports the rental state, direct address, and requested accelerator",
		strings.Contains(out, "state:") && strings.Contains(out, "127.0.0.1:") &&
			field(out, "accelerator") == accelerator, field(out, "address")+" "+field(out, "accelerator"))
	code, control, controlOut := cozyJSON(root, "rent", "show", rentalA)
	selectedRoots, _ := control["model_root_digests"].([]any)
	selectedPlans, _ := control["binding_plan_digests"].([]any)
	// A weightless release selects NO model root and one plan per visible entrypoint.
	check("rent show projects exact selected execution/root/plan identities, never selection inputs",
		code == 0 && strings.HasPrefix(fmt.Sprint(control["endpoint_execution_digest"]), "sha256:") &&
			len(selectedRoots) == 0 && len(selectedPlans) == len(plans),
		fmt.Sprintf("%d root(s) · %d plan(s) · %s", len(selectedRoots), len(selectedPlans), firstLine(controlOut)))
	code, probe, probeOut := cozyJSON(root, "rent", "probe", rentalA)
	check("rent probe claims the worker without invoking a model and persists actual GPU identity",
		code == 0 && probe["state"] == "ready" && probe["accelerator"] == accelerator &&
			probe["observed_accelerator"] == accelerator &&
			probe["observed_accelerator_count"] == float64(1) &&
			probe["observed_backend"] == "cuda" && fmt.Sprint(probe["observed_worker_instance"]) != "" &&
			fmt.Sprint(probe["observed_worker_boot_id"]) != "", firstLine(probeOut))
	code, observedControl, controlOut := cozyJSON(root, "rent", "show", rentalA)
	check("rent show retains the exact worker readback after the probe",
		code == 0 && observedControl["observed_accelerator"] == probe["observed_accelerator"] &&
			observedControl["observed_worker_instance"] == probe["observed_worker_instance"] &&
			observedControl["endpoint_execution_digest"] == control["endpoint_execution_digest"],
		firstLine(controlOut))
	fmt.Printf("  bench ask -> ready -> pinned: %d ms\n", rentMS)

	head("the owner token is MINTED HERE, and the hub is never told it (#495e)")
	// The token exists on THIS host and nowhere else. Reading it out of the 0600 file the
	// client wrote is the only way anything in this driver can know it — the hub cannot be
	// asked, because the hub has no such field.
	tokenFile := filepath.Join(root, "rentals", rentalA+".token")
	info, err := os.Stat(tokenFile)
	check("the token landed in a 0600 file, not a record and not argv",
		err == nil && info != nil && info.Mode().Perm() == 0o600,
		tokenFile+" "+modeOf(info))
	stored, err := os.ReadFile(tokenFile)
	minted := strings.TrimSpace(string(stored))
	check("and it is 32 bytes of this host's own entropy, hex-spelled",
		err == nil && len(minted) == 64, fmt.Sprintf("%d hex chars", len(minted)))
	check("it appears NOWHERE in what cozy printed",
		minted != "" && !strings.Contains(out, minted), "searched the whole rendering")
	check("what IS printed is its digest, comparable against the pod's own",
		strings.Contains(field(out, "owner_token"), "sha256:"), field(out, "owner_token"))

	// THE INVERSION, as an observation over the peer's own state. The hub holds a hash of
	// the minted token and holds nothing that could produce it.
	hashes := hub.hashesOf(rentalA)
	wantSum := sha256.Sum256([]byte(minted))
	wantHash := hex.EncodeToString(wantSum[:]) // the hub's spelling: bare hex
	check("the hub holds exactly one credential fact for this pod, and it is a HASH",
		len(hashes) == 1 && hashes[0] == wantHash, strings.Join(hashes, " "))
	check("and that hash is not the token: nothing the hub holds could be presented as proof",
		hashes != nil && hashes[0] != minted, "a hash is not a preimage")
	podSet, err := os.ReadFile(filepath.Join(root, "hub", rentalA+"-run", "pod.tokens"))
	check("the pod was provisioned with that same hash, and the token is absent from the pod",
		err == nil && strings.Contains(string(podSet), "sha256:"+wantHash) &&
			!strings.Contains(string(podSet), minted),
		strings.TrimSpace(string(podSet)))

	pemBytes, err := os.ReadFile(filepath.Join(root, "rentals", rentalA+".pem"))
	check("the certificate to PIN is beside it, and it is a certificate",
		err == nil && strings.HasPrefix(string(pemBytes), "-----BEGIN CERTIFICATE-----"), "")

	head("cozy rent ls — the listing renders the digest and never the value")
	code, out = cozyRun(root, "rent", "ls", "--full")
	fmt.Println(indent(out))
	check("cozy rent ls names the rental this host holds",
		code == 0 && strings.Contains(out, rentalA), firstLine(out))
	check("with the owner token digested, and the raw value absent",
		strings.Contains(out, "sha256:") && !strings.Contains(out, minted), "")

	head("RED: a run pinned to a rental this host does not hold")
	_, before, _ := cozyJSON(root, "status")
	code, out = cozyRun(root, "run", rentalEndpointRef, tilePayload, "--worker", "rnt-nope")
	check("cozy run --worker <unknown> -> 4, the server's own envelope",
		code == 4 && strings.Contains(out, "no rental rnt-nope"), firstLine(out))
	_, after, _ := cozyJSON(root, "status")
	check("and NO request was recorded: an unplaceable pin is refused before the row",
		before["requests"] == after["requests"],
		fmt.Sprint(before["requests"])+" -> "+fmt.Sprint(after["requests"]))

	head("THE LEG: attach over TLS, claim with the provisioned token, dispatch to the pod")
	outDir := filepath.Join(root, "out")
	t0 = time.Now()
	code, out = cozyRun(root, "run", rentalEndpointRef, tilePayload,
		"--out", outDir, "--stream", "--worker", rentalA)
	legMS := elapsedMS(t0)
	instanceA := instanceFor(root, rentalA)
	podLog := podWorkerLog(hub, rentalA)
	check("the pod hosted WorkerControl behind TLS",
		strings.Contains(podLog, "hosting behind TLS"), "")
	check("and it ACCEPTED the claim — the owner presented the provisioned owner token",
		strings.Contains(podLog, "ClaimAck sent"), "")
	check("the orchestrator ATTACHED rather than spawned: the slot is the RENTAL's",
		strings.HasSuffix(instanceEndpoint(root, instanceA), "@"+rentalA),
		instanceEndpoint(root, instanceA)+" "+instanceA)
	check("the attempt was DISPATCHED to that pod's own instance",
		instanceA != "" && dispatchedInstance(out) == instanceA,
		dispatchedInstance(out)+" == "+instanceA)
	slots := rentalWorkers(svc, rentalA)
	check("probe then run left exactly ONE worker slot for the rental (#probe/run double-slot)",
		len(slots) == 1 && slots[0] == instanceA, strings.Join(slots, ", "))
	fmt.Printf("  bench attach + claim + dispatch over TLS: %d ms\n", legMS)

	head("#506a: the BINDING RECORD reached the pod, and the pod re-derived its own id")
	planID := planFor(plans, rentalEntrypoint)
	delivered := podPlanRecord(hub, rentalA, planID)
	check("the plan is on the POD's filesystem, under the id the directive names",
		delivered != nil, filepath.Join(hub.pods, rentalA, "binding-plans",
			strings.TrimPrefix(planID, "sha256:")+".json"))
	check("the media server ACCEPTED it only because it hashes to that id — it recomputes",
		delivered != nil && fmt.Sprint(delivered["entrypoint"]) == rentalEntrypoint &&
			digestOfFile(filepath.Join(hub.pods, rentalA, "binding-plans",
				strings.TrimPrefix(planID, "sha256:")+".json")) == planID,
		shortID(planID))
	tampered := map[string]any{}
	for k, v := range delivered {
		tampered[k] = v
	}
	tampered["entrypoint"] = "not-the-entrypoint-that-was-digested"
	tamperedBody, _ := json.Marshal(tampered)
	status, said := mediaCall(root, rentalA, http.MethodPut,
		"/v1/plans/"+strings.TrimPrefix(planID, "sha256:"), minted, tamperedBody)
	check("RED: a plan whose identity was edited is refused under the id it arrived as",
		status == 400 && strings.Contains(said, "media.binding_plan_"),
		itoa(status)+" "+firstLine(said))
	still := podPlanRecord(hub, rentalA, planID)
	check("and the pod still holds the ORIGINAL plan: a refusal wrote nothing",
		still != nil && fmt.Sprint(still["entrypoint"]) == rentalEntrypoint,
		fmt.Sprint(still["entrypoint"]))

	head("THE BYTE PLANE: the payload crossed, and the output came home verified")
	podFiles := podMediaFiles(hub, rentalA)
	check("the pod READ it from there — it says what it found",
		strings.Contains(podLog, "granted input read from the pod's own disk"),
		lineWith(podLog, "granted input read"))
	check("the orchestrator FETCHED those bytes home and verified them before acking",
		strings.Contains(serviceLog(root), "mirrored "),
		lineWith(serviceLog(root), "mirrored "))
	check("the run succeeded end to end over a real byte boundary",
		code == 0, firstLine(out)+" [exit "+itoa(code)+"]")
	// `--out` names outputs by manifest ORDER and MIME (`output-01.png`), never by the
	// endpoint's field name.
	local := filepath.Join(outDir, "output-01.png")
	check("and `--out` wrote a file the client actually holds",
		bytesAt(local) > 0, local+" "+itoa(bytesAt(local))+" B")
	attemptSlot := media.Slot(dispatchedRequest(out), 1)
	remoteAttemptBytes := false
	for _, name := range podFiles {
		remoteAttemptBytes = remoteAttemptBytes || strings.Contains(name, attemptSlot)
	}
	check("the verified local output remains while the acked pod attempt was deleted",
		digestOfFile(local) != "" && !remoteAttemptBytes,
		shortID(digestOfFile(local))+" · remote "+strings.Join(podFiles, ", "))

	head("IMAGE -> MP4 TRANSPORT: the real CLI flag, exact grant, mirror, ack, cleanup")
	// This is deliberately a transport contract, not a synthetic H3 claim. The installed
	// weightless descriptor declares the same interesting shape (text + first-frame image
	// -> video asset); the fake pod proves those bytes and identities cross the boundary.
	videoHub := newPodHub(podHubSpec{Dir: filepath.Join(root, "hub-video-transport"),
		Arm: "remote", Release: transportRelease})
	defer videoHub.close()
	videoRental, _ := rentEndpoint(root, videoHub, transportRentalEndpointRef,
		"cl-019 image-to-mp4 transport proof")
	firstFrame := filepath.Join(root, "first-frame.png")
	frameBytes, err := hex.DecodeString(onePixelPNG)
	must("decoding the first-frame fixture", err)
	must("writing the first-frame fixture", os.WriteFile(firstFrame, frameBytes, 0o600))
	frameDigest := digestOfFile(firstFrame)
	videoOut := filepath.Join(root, "out-video")
	code, out = cozyRun(root, "run", transportRentalEndpointRef,
		"prompt=transport boundary only",
		"--asset", "first_frame="+firstFrame,
		"--out", videoOut, "--stream", "--worker", videoRental)
	videoRequest := dispatchedRequest(out)
	videoLog := podWorkerLog(videoHub, videoRental)
	check("the real `cozy run --asset first_frame=<image>` invocation succeeded",
		code == 0 && videoRequest != "", firstLine(out)+" [exit "+itoa(code)+"]")
	check("the fake pod consumed first_frame and checked length, digest, and image/png binding",
		strings.Contains(videoLog, "validated granted input first_frame") &&
			strings.Contains(videoLog, frameDigest) && strings.Contains(videoLog, "image/png"),
		lineWith(videoLog, "validated granted input first_frame"))

	expectedVideo, err := hex.DecodeString(transportMP4)
	must("decoding the MP4 transport fixture", err)
	expectedVideoDigest := digestOfBytes(expectedVideo)
	localVideo := filepath.Join(videoOut, "output-01.mp4")
	localVideoBytes, _ := os.ReadFile(localVideo)
	life := svc.call("GET", "/v1/requests/"+videoRequest, nil).json()
	videoRef := lifecycleOutput(life, "video")
	check("the mirrored record preserves the worker's exact MP4 digest, length, and MIME",
		fmt.Sprint(videoRef["digest"]) == expectedVideoDigest &&
			int(number(videoRef["length"])) == len(expectedVideo) &&
			fmt.Sprint(videoRef["mime_type"]) == "video/mp4",
		fmt.Sprintf("%v · %v B · %v", videoRef["digest"], videoRef["length"], videoRef["mime_type"]))
	check("the CLI published the exact MP4-shaped bytes at the declared .mp4 path",
		bytes.Equal(localVideoBytes, expectedVideo) && isMP4(localVideoBytes),
		localVideo+" "+itoa(len(localVideoBytes))+" B "+shortID(digestOfFile(localVideo)))
	staging, _ := filepath.Glob(filepath.Join(root, ".out-video.staging-*"))
	check("verified bytes were atomically renamed with no client staging residue",
		len(staging) == 0 && digestOfFile(localVideo) == expectedVideoDigest,
		fmt.Sprintf("%d staging files · %s", len(staging), shortID(digestOfFile(localVideo))))
	check("the orchestrator logged the exact-length remote mirror before settlement",
		strings.Contains(serviceLog(root), "/video: "+itoa(len(expectedVideo))+" B from"),
		lineWith(serviceLog(root), "/video: "))

	videoSlot := media.Slot(videoRequest, 1)
	// The run already exited 0, so the ack and the cleanup are in flight between two live
	// processes: wait while either side is still making progress, not for a clock.
	podProgress := func() string {
		return itoa(len(podWorkerLog(videoHub, videoRental))) + "/" +
			strings.Join(podMediaFiles(videoHub, videoRental), ",") + "/" + itoa(len(serviceLog(root)))
	}
	acked := waitProgressing(func() bool {
		return strings.Contains(podWorkerLog(videoHub, videoRental),
			"AttemptOutcomeAck for "+videoRequest+"#1")
	}, podProgress)
	remoteClean := waitProgressing(func() bool {
		for _, name := range podMediaFiles(videoHub, videoRental) {
			if strings.Contains(name, videoSlot) {
				return false
			}
		}
		return true
	}, podProgress)
	check("the fake pod received the exact terminal ack", acked,
		lineWith(podWorkerLog(videoHub, videoRental), "AttemptOutcomeAck for "+videoRequest))
	check("only after that ack, the pod's payload, image grant, and output slot were removed",
		remoteClean, strings.Join(podMediaFiles(videoHub, videoRental), ", "))
	layout, layoutErr := home.Open(root)
	check("terminal cleanup dropped the staged host copy but preserved the user's source image",
		layoutErr == nil && !exists(layout.InputAsset(frameDigest)) && digestOfFile(firstFrame) == frameDigest,
		firstFrame+" "+shortID(frameDigest))

	head("THE MEDIA SERVER'S OWN DOOR: bearer, names, and the quota")
	// The pod's byte plane is dialled DIRECTLY here, with the certificate this host pinned,
	// because these arms need a caller that can present the wrong credential and name the
	// wrong thing — which the product's own client structurally cannot.
	status, said = mediaCall(root, rentalA, http.MethodGet, "/v1/health", "", nil)
	check("RED: no bearer -> 401 media.unauthenticated, before anything is served",
		status == 401 && strings.Contains(said, "media.unauthenticated"),
		itoa(status)+" "+firstLine(said))
	status, said = mediaCall(root, rentalA, http.MethodGet, "/v1/health", randomHex(32), nil)
	check("RED: a FOREIGN bearer -> 401 — the pod checks a hash it was provisioned with",
		status == 401 && strings.Contains(said, "media.unauthenticated"),
		itoa(status)+" "+firstLine(said))
	status, said = mediaCall(root, rentalA, http.MethodGet, "/v1/health", minted, nil)
	var served mediawire.Health
	_ = json.Unmarshal([]byte(said), &served)
	check("the rental's OWN token is admitted, and the plane names itself AND its revision",
		status == 200 && served.Service == mediawire.Service &&
			served.ContractRev != nil && *served.ContractRev == mediawire.ContractRev,
		itoa(status)+" "+firstLine(said))
	status, said = mediaCall(root, rentalA, http.MethodGet,
		"/v1/outputs/"+media.Slot("req-nope", 1)+"/..%2f..%2fpod-worker.log", minted, nil)
	check("RED: a traversal in an output name is not a name this server can hold",
		status == 400 && strings.Contains(said, "media.bad_name"),
		itoa(status)+" "+firstLine(said))
	status, said = mediaCall(root, rentalA, http.MethodPut, "/v1/inputs/oversize", minted,
		make([]byte, 8<<20))
	check("RED: a body past the pod's media quota refuses; the subtree is separately bounded",
		(status == 507 || status == 413) &&
			(strings.Contains(said, "media.quota_exhausted") || strings.Contains(said, "media.over_bound")),
		itoa(status)+" "+firstLine(said))
	check("and the worker's journal is untouched by it: the quota is not the pod's disk",
		podWorkerLog(hub, rentalA) != "", "the pod's worker log is still readable")

	head("FILESYSTEM HANDOFF ONLY: the byte plane outlives the worker beside it")
	// cl-014's coupling claim, observed instead of asserted. The two processes in this pod
	// share an outputs directory and a token-hash file and NOTHING else — no RPC in either
	// direction — so killing the worker must not take down the byte plane. This attempt's
	// bytes were already mirrored, acked, and deleted; the local mirror is the durable copy.
	// The static half of the same claim is the `media` fence family: the media server's
	// source contains no outbound call and cannot import the protocol at all.
	deadPID := hub.killPodWorker(rentalA)
	check("the pod's WORKER process is killed, its media server is not",
		deadPID > 0 && !alivePID(deadPID), "pid "+itoa(deadPID))
	status, said = mediaCall(root, rentalA, http.MethodGet, "/v1/health", minted, nil)
	check("the media service stays healthy without a request-path call to the worker",
		status == 200 && strings.Contains(said, "cozy-media"),
		itoa(status)+" "+firstLine(said))
	status, said = mediaCall(root, rentalA, http.MethodGet,
		"/v1/outputs/"+attemptSlot+"/image", minted, nil)
	check("the acked attempt stays deleted instead of becoming a remote output cache",
		status == 404, itoa(status)+" "+firstLine(said))

	head("RED: a pod with a control leg and NO byte plane is refused, never worked around")
	starved := newPodHub(podHubSpec{Dir: filepath.Join(root, "hub-nomedia"), Arm: "remote",
		Release: release, NoMedia: true})
	defer starved.close()
	code, out = cozyRunEnv(root, starved.env(), "rent", rentalEndpointRef,
		"--accelerator", accelerator, "--reason", "cl-019 no-media-plane arm")
	check("the rental refuses rather than pinning an unusable pod",
		code != 0 && (strings.Contains(out, "media") || strings.Contains(out, "byte plane")),
		firstLine(out)+" [exit "+itoa(code)+"]")
	rentalF := ""
	for _, id := range heldRentals(root) {
		if starved.holds(id) {
			rentalF = id
			break
		}
	}
	check("and no attempt was dispatched to it: nothing crossed a boundary that is not there",
		rentalF != "" && !strings.Contains(podWorkerLog(starved, rentalF), "AttemptOffer"),
		"the pod's worker saw no AttemptOffer")

	head("cl-031 RED: a byte plane at ANOTHER CONTRACT REVISION is refused before a byte moves")
	// The defect this closes: `cozy-media` is compiled into the pod image from a pinned
	// cozy-creator commit while this client floats with master, and until now the only
	// version signal between the two was the literal `/v1/` in a URL. The pods below run a
	// STAND-IN media plane because a pod at another revision is an older build of this
	// tree, which this worktree cannot produce; everything else about them is real.
	otherRev := mediawire.ContractRev + 1
	skewed := newPodHub(podHubSpec{Dir: filepath.Join(root, "hub-mediarev"), Arm: "remote",
		Release: release, MediaSkew: &mediawire.Contract{Service: mediawire.Service,
			ContractRev: &otherRev, MaxReceiptBytes: mediawire.MaxReceiptBytes}})
	defer skewed.close()
	rentalSkew, _ := rentOne(root, skewed, "cl-031 media revision skew arm")
	code, out = cozyRun(root, "run", rentalEndpointRef, tilePayload,
		"--timeout", "20s", "--worker", rentalSkew)
	check("a plane at rev N+1 is refused by name, and the refusal names BOTH revisions",
		code != 0 && strings.Contains(out, "media_contract_mismatch") &&
			strings.Contains(out, "rev "+itoa(otherRev)) &&
			strings.Contains(out, "rev "+itoa(mediawire.ContractRev)),
		lineWith(out, "media_contract_mismatch")+" [exit "+itoa(code)+"]")
	check("and NOT ONE byte was offered to it: the plane was asked its revision, nothing else",
		len(skewed.mediaTouched()) == 0,
		"stand-in served: "+strings.Join(append([]string{"GET /v1/health"}, skewed.mediaTouched()...), ", "))

	head("cl-031 RED: a plane that declares NO revision is refused — absence IS the skew case")
	// Fail-closed, and this is the arm that says why: a pod too old to declare a revision
	// is precisely the pod the check exists for. Indulging it would leave the whole window
	// open and call it compatibility.
	mute := newPodHub(podHubSpec{Dir: filepath.Join(root, "hub-mediamute"), Arm: "remote",
		Release: release, MediaSkew: &mediawire.Contract{Service: mediawire.Service,
			MaxReceiptBytes: mediawire.MaxReceiptBytes}})
	defer mute.close()
	rentalMute, _ := rentOne(root, mute, "cl-031 media revision absent arm")
	code, out = cozyRun(root, "run", rentalEndpointRef, tilePayload,
		"--timeout", "20s", "--worker", rentalMute)
	check("a plane declaring no revision is refused, not indulged as an older peer",
		code != 0 && strings.Contains(out, "media_contract_mismatch") &&
			strings.Contains(out, "declares NO media contract revision"),
		lineWith(out, "media_contract_mismatch")+" [exit "+itoa(code)+"]")
	check("and it too was never handed a byte",
		len(mute.mediaTouched()) == 0,
		"stand-in served: "+strings.Join(append([]string{"GET /v1/health"}, mute.mediaTouched()...), ", "))

	head("#505: the RELEASE PIN is verified, not merely carried")
	lying := newPodHub(podHubSpec{Dir: filepath.Join(root, "hub-badrelease"), Arm: "remote",
		Release: transportEndpointRef + "@not-this-one"})
	defer lying.close()
	rentalG, _ := rentOne(root, lying, "cl-019 release-mismatch arm")
	code, out = cozyRun(root, "run", rentalEndpointRef, tilePayload,
		"--timeout", "20s", "--worker", rentalG)
	check("a pod serving a DIFFERENT release is refused, and the client is told which",
		code != 0 && strings.Contains(out, "not-this-one"),
		firstLine(out)+" [exit "+itoa(code)+"]")
	check("the orchestrator named the fence: this host pinned %s",
		strings.Contains(serviceLog(root), "release_mismatch"),
		lineWith(serviceLog(root), "release_mismatch"))
	silent := newPodHub(podHubSpec{Dir: filepath.Join(root, "hub-norelease"), Arm: "remote"})
	defer silent.close()
	rentalH, _ := rentOne(root, silent, "cl-019 release-undeclared arm")
	code, out = cozyRun(root, "run", rentalEndpointRef, tilePayload,
		"--timeout", "20s", "--worker", rentalH)
	check("a pod that will not SAY what it serves is refused too — carried is not verified",
		code != 0 && strings.Contains(serviceLog(root), "release_undeclared"),
		lineWith(serviceLog(root), "release_undeclared"))

	head("#563e: the INSTANCE identity of a pod is the POD's, and it must be declared")
	// A rented pod names its own worker: this host never spawned it and could not have
	// chosen the name. What it is held to is saying WHICH worker it is — silence there
	// is a terminal that belongs to nobody — and to not changing the answer later.
	nameless := newPodHub(podHubSpec{Dir: filepath.Join(root, "hub-noinstance"), Arm: "remote",
		Release: release, NoInstance: true})
	defer nameless.close()
	rentalI, _ := rentOne(root, nameless, "cl-019 instance-undeclared arm")
	code, out = cozyRun(root, "run", rentalEndpointRef, tilePayload,
		"--timeout", "20s", "--worker", rentalI)
	check("a pod that will not say WHICH worker it is is refused — carried is not verified",
		code != 0 && strings.Contains(serviceLog(root), "worker_instance_undeclared"),
		lineWith(serviceLog(root), "worker_instance_undeclared"))
	check("and the pod's own instance name is ACCEPTED when it gives one: the slot does not "+
		"demand a machine it never spawned answer to the slot's name",
		strings.Contains(podWorkerLog(hub, rentalA), "instance=ins-"+rentalA),
		lineWith(podWorkerLog(hub, rentalA), "ClaimAck sent"))

	head("EXACT ROUTING: a second attached target advertising the SAME plan")
	rentalB, _ := rentOne(root, hub, "cl-015 target B")
	code, out = cozyRun(root, "run", rentalEndpointRef, tilePayload,
		"--stream", "--worker", rentalB)
	instanceB := instanceFor(root, rentalB)
	check("B is its own instance, not A's", instanceB != "" && instanceB != instanceA,
		instanceA+" vs "+instanceB)
	check("a run pinned to B lands on B — the pin ROUTES, it does not merely resolve",
		dispatchedInstance(out) == instanceB, dispatchedInstance(out))
	check("and it never lands on A", dispatchedInstance(out) != instanceA, "")
	_ = code

	head("an UNPINNED request never consumes rented capacity")
	// The unpinned request is a weightless CPU tile: the service spawns a REAL local
	// worker for it with no model and no GPU (the shared box runs no inference), and the
	// run settles on its own rather than on a client deadline.
	code, out = cozyRun(root, "run", localTileRef, tilePayload, "--stream")
	landed := dispatchedInstance(out)
	check("it was NOT dispatched to either rented pod",
		landed != instanceA && landed != instanceB,
		"landed on "+nothingOr(landed)+" (A "+instanceA+", B "+instanceB+")")
	check("it landed on a LOCALLY spawned worker, and settled",
		code == 0 && landed != "" && !strings.Contains(instanceEndpoint(root, landed), "@rnt-"),
		instanceEndpoint(root, landed)+" [exit "+itoa(code)+"]")
	code, out = cozyRun(root, "stop", "--all")
	check("cozy stop --all clears whatever that started", code == 0, firstLine(out))

	head("RED: a FOREIGN owner token cannot drive a pod")
	rentalC, _ := rentOne(root, hub, "cl-015 foreign-token arm")
	must("planting a foreign owner token", os.WriteFile(
		filepath.Join(root, "rentals", rentalC+".token"), []byte(randomHex(32)+"\n"), 0o600))
	code, out = cozyRun(root, "run", rentalEndpointRef, tilePayload, "--worker", rentalC)
	check("the run does NOT succeed: the pod refused a claim it did not provision",
		code != 0, firstLine(out)+" [exit "+itoa(code)+"]")
	check("the POD's media server refused it first — the byte plane is dialled before the claim",
		strings.Contains(serviceLog(root), "not one this pod was provisioned with"),
		lineWith(serviceLog(root), "not one this pod was provisioned with"))
	check("so the pod's worker was never claimed at all",
		!strings.Contains(podWorkerLog(hub, rentalC), "ClaimAck sent"),
		"no ClaimAck in the pod's worker log")

	head("a widened token file is refused AT THE DIAL, which is the only place it is read")
	// A FRESH rental, because the check lives where the credential is read: the HTTP layer
	// asks only whether this host holds the pin, and an already-attached worker is never
	// re-resolved. That is the point of the split — the token has exactly one reader.
	rentalE, _ := rentOne(root, hub, "cl-015 widened-token arm")
	must("widening the token mode", os.Chmod(filepath.Join(root, "rentals", rentalE+".token"), 0o644))
	code, out = cozyRun(root, "run", rentalEndpointRef, tilePayload, "--worker", rentalE)
	check("mode 0644 refuses by name, and nothing was dialled",
		code != 0 && strings.Contains(out, "0600"), firstLine(out)+" [exit "+itoa(code)+"]")
	check("the pod was never claimed: no worker for it exists",
		instanceFor(root, rentalE) == "", instanceFor(root, rentalE))

	head("THE MIRROR: an output the owner does not hold is NEVER acked")
	// This pod does the work and writes the bytes on its own disk — a true manifest about
	// a real file the owner cannot see. It is what every real pod does today.
	liar := newPodHub(podHubSpec{Dir: filepath.Join(root, "hub-unmirrored"),
		Arm: "remotelie", Release: release})
	defer liar.close()
	rentalD, _ := rentOne(root, liar, "cl-015 unmirrored-output arm")
	mirrorDir := filepath.Join(root, "out-mirror")
	code, out = cozyRun(root, "run", rentalEndpointRef, tilePayload,
		"--out", mirrorDir, "--timeout", "12s", "--worker", rentalD)
	check("the pod DID write bytes — on its own filesystem",
		podWroteOutput(liar, rentalD), filepath.Join(liar.pods, rentalD))
	check("the owner did not ack it: the request never settled and the deadline answered",
		code == 10, firstLine(out)+" [exit "+itoa(code)+"]")
	check("the orchestrator said WHY: the pod's media plane has no such output to hand over",
		strings.Contains(serviceLog(root), "no output"),
		lineWith(serviceLog(root), "no output"))
	check("no output became visible and no file was written",
		!exists(filepath.Join(mirrorDir, "output-01.png")) && !exists(mirrorDir), mirrorDir)

	head("#506a: TWO MACHINES, ONE RELEASE, ONE PLAN ID")
	// A SECOND independent install of the byte-identical release archive on its own root:
	// the shape of a pod that installed what this host installed. The plan the
	// orchestrator names on the wire has to be the same plan.
	rootB := filepath.Join(root, "machine-b")
	must("creating the second root", os.MkdirAll(rootB, 0o755))
	installTransportEndpoint(rootB)
	releaseB, plansB := endpointIdentityFor(rootB, transportEndpointRef)
	check("they serve the SAME endpoint release", releaseB == release, releaseB)
	check("and they compute the SAME entrypoint_binding_plan_id — #506a's whole point",
		planFor(plans, rentalEntrypoint) != "" &&
			planFor(plans, rentalEntrypoint) == planFor(plansB, rentalEntrypoint),
		shortID(planFor(plans, rentalEntrypoint))+" == "+shortID(planFor(plansB, rentalEntrypoint)))
	same := 0
	for name, id := range plans {
		if plansB[name] == id {
			same++
		}
	}
	check("for EVERY entrypoint the release declares, not just the one under test",
		same == len(plans) && same > 0, itoa(same)+"/"+itoa(len(plans))+" plan ids equal")

	head("cozy rent release — plan first, then the pod is gone")
	code, out = cozyRunEnv(root, hub.env(), "rent", "release", rentalC)
	check("without --yes it prints the plan and exits 0",
		code == 0 && strings.Contains(out, "DESTROYS"), firstLine(out))
	check("and it changed nothing: the hub still holds the pod", !hub.released(rentalC), "")
	code, out = cozyRunEnv(root, hub.env(), "rent", "release", rentalC, "--yes")
	check("with --yes it exits 0 and the hub SAW the delete",
		code == 0 && hub.released(rentalC), firstLine(out))
	check("and the hub reports the rental ABSENT afterwards: the post-DELETE poll terminates",
		hub.status(rentalC) == 404, itoa(hub.status(rentalC)))
	_, err = os.Stat(filepath.Join(root, "rentals", rentalC+".token"))
	check("the owner token is gone from this host", os.IsNotExist(err), "")
	code, out = cozyRunEnv(root, hub.env(), "rent", "release", rentalC, "--yes")
	check("release is IDEMPOTENT: a second --yes for the same id exits 0",
		code == 0, firstLine(out)+" [exit "+itoa(code)+"]")
	code, out = cozyRun(root, "run", rentalEndpointRef, tilePayload, "--worker", rentalC)
	check("a run pinned to the released rental is 4 again", code == 4, firstLine(out))

	head("RED: releasing something this host does not hold")
	code, out = cozyRunEnv(root, hub.env(), "rent", "release", "rnt-nope", "--yes")
	check("cozy rent release <unknown> is idempotent: the hub reports it absent -> 0",
		code == 0 && strings.Contains(out, "already released"), firstLine(out))

	head("RED: a hub whose pods never come up")
	broken := startPodHub(filepath.Join(root, "hub-broken"), "remote", true)
	defer broken.close()
	code, out = cozyRunEnv(root, broken.env(), "rent", rentalEndpointRef, "--accelerator", accelerator,
		"--reason", "cl-015 failed-provision arm")
	check("a rental that fails to provision -> 11, carrying the hub's own detail",
		code == 11 && strings.Contains(out, "no eligible capacity"), firstLine(out))
	check("and nothing was pinned for it: there is no triple to pin",
		!strings.Contains(out, "owner_token"), "")

	head("teardown: every rental this host holds is released and ZERO is proved")
	// Several stand-in hubs are live at once, so the pass asks each one whether it still
	// holds the pod. A rental nobody owns would be a pod nothing can destroy, which is the
	// exact failure this pass exists to make impossible.
	hubs := []*podHub{hub, videoHub, lost, liar, starved, skewed, mute, lying, silent,
		nameless, broken}
	for _, id := range heldRentals(root) {
		on := hub
		for _, candidate := range hubs {
			if candidate.holds(id) {
				on = candidate
				break
			}
		}
		code, out = cozyRunEnv(root, on.env(), "rent", "release", id, "--yes")
		check("released "+id, code == 0, firstLine(out))
	}
	check("this host holds ZERO rentals — the teardown is proved, not assumed",
		zeroRentals(root), "")
	pending, _ = filepath.Glob(filepath.Join(root, "rentals", "pending-*.token"))
	check("and no rejected or released operation retains a pending token", len(pending) == 0,
		fmt.Sprintf("%d pending token(s)", len(pending)))
}

// rentalWorkers is every live worker slot the service holds for one rental, from
// `GET /v1/local/workers`. Probe followed by run must leave exactly one.
func rentalWorkers(svc *liveService, rentalID string) []string {
	doc := svc.call("GET", "/v1/local/workers", nil).json()
	rows, _ := doc["workers"].([]any)
	out := []string{}
	for _, raw := range rows {
		row, _ := raw.(map[string]any)
		if exited, _ := row["exited"].(bool); exited {
			continue
		}
		if strings.HasSuffix(fmt.Sprint(row["endpoint"]), "@"+rentalID) {
			out = append(out, fmt.Sprint(row["instance_id"]))
		}
	}
	sort.Strings(out)
	return out
}

// waitProgressing waits for cond while `progress` keeps changing: it gives up only when
// the peers have stopped moving for a settle window, never on a flat deadline.
func waitProgressing(cond func() bool, progress func() string) bool {
	const settle = 2 * time.Second
	last, since := progress(), time.Now()
	for {
		if cond() {
			return true
		}
		if now := progress(); now != last {
			last, since = now, time.Now()
		} else if time.Since(since) > settle {
			return false
		}
		time.Sleep(20 * time.Millisecond)
	}
}

// rentOne rents a pod and returns its id with the whole rendering, so an arm can search
// what was printed as well as act on the id.
func rentOne(root string, h *podHub, reason string) (string, string) {
	return rentEndpoint(root, h, rentalEndpointRef, reason)
}

func rentEndpoint(root string, h *podHub, endpoint, reason string) (string, string) {
	code, out := cozyRunEnv(root, h.env(), "rent", endpoint, "--accelerator", accelerator, "--reason", reason)
	if code != 0 {
		fmt.Println(indent(out))
		return "", out
	}
	return field(out, "rental"), out
}

// dispatchedInstance reads the instance the orchestrator actually placed the attempt on,
// out of the request's own `request.dispatched` frame. It is the routing observation:
// which worker took the work, from the stream the client watched.
func dispatchedInstance(stream string) string {
	for _, line := range strings.Split(stream, "\n") {
		if !strings.Contains(line, `"request.dispatched"`) {
			continue
		}
		if _, rest, ok := strings.Cut(line, `"instance_id":"`); ok {
			if id, _, ok := strings.Cut(rest, `"`); ok {
				return id
			}
		}
	}
	return ""
}

// instanceFor is the live instance id of one rental's attached worker, read from the
// worker listing where the rental-scoped slot name appears.
func instanceFor(root, rentalID string) string {
	for _, row := range liveWorkers(root) {
		fields := strings.Fields(row)
		if len(fields) >= 2 && strings.HasSuffix(fields[0], "@"+rentalID) {
			return fields[1]
		}
	}
	return ""
}

// instanceEndpoint is the slot name one live instance is running under.
func instanceEndpoint(root, instance string) string {
	if instance == "" {
		return ""
	}
	for _, row := range liveWorkers(root) {
		fields := strings.Fields(row)
		if len(fields) >= 2 && fields[1] == instance {
			return fields[0]
		}
	}
	return ""
}

func liveWorkers(root string) []string {
	_, doc, _ := cozyJSON(root, "status")
	live, _ := doc["live_workers"].([]any)
	out := make([]string, 0, len(live))
	for _, w := range live {
		out = append(out, fmt.Sprint(w))
	}
	return out
}

func heldRentals(root string) []string {
	_, doc, _ := cozyJSON(root, "rent", "ls")
	rows, _ := doc["rows"].([]any)
	out := []string{}
	for _, row := range rows {
		fields, _ := row.(map[string]any)
		if id := fmt.Sprint(fields["rental"]); id != "" && id != "<nil>" {
			out = append(out, id)
		}
	}
	return out
}

func zeroRentals(root string) bool {
	_, doc, _ := cozyJSON(root, "rent", "ls")
	count, ok := doc["count"].(float64)
	return ok && count == 0
}

// podWorkerLog is what the pod's own worker wrote. An arm about a claim being accepted or
// a destination being refused is answered by the PEER, not by this side's inference.
func podWorkerLog(h *podHub, rentalID string) string {
	data, err := os.ReadFile(filepath.Join(h.dir, rentalID+"-run", "pod-worker.log"))
	if err != nil {
		return ""
	}
	return string(data)
}

// podWroteOutput answers whether any bytes landed on the POD's filesystem — the other
// half of "the owner does not hold them".
func podWroteOutput(h *podHub, rentalID string) bool {
	found := false
	_ = filepath.Walk(filepath.Join(h.pods, rentalID), func(path string, info os.FileInfo, err error) error {
		if err == nil && info != nil && !info.IsDir() && filepath.Base(path) == "image" {
			found = true
		}
		return nil
	})
	return found
}

func serviceLog(root string) string {
	data, err := os.ReadFile(filepath.Join(root, "driver-service.log"))
	if err != nil {
		return ""
	}
	return string(data)
}

// nothingOr names an empty placement, so an arm's detail line says "nothing" rather
// than trailing off into whitespace.
func nothingOr(s string) string {
	if s == "" {
		return "(nothing)"
	}
	return s
}

func modeOf(info os.FileInfo) string {
	if info == nil {
		return "absent"
	}
	return fmt.Sprintf("mode %#o", info.Mode().Perm())
}

// endpointIdentity asks a host's own resolver what it installed: the endpoint release id
// it serves under, the plan id of every entrypoint, and the record each id was taken over.
//
// It runs in a CHILD PROCESS, and that is the arm's whole point rather than a workaround.
// The environment is read once per process (`internal/config`, the env fence), so one
// process has exactly one COZY_HOME — which is also true of a real machine. Two roots
// therefore means two processes, and the cross-machine plan-id arm is comparing what two
// independent runs of the product's own resolver said, not what one run said twice.
func endpointIdentityFor(root, endpoint string) (string, map[string]string) {
	release, ids, _ := endpointFactsFor(root, endpoint)
	return release, ids
}

func endpointFactsFor(root, endpoint string) (string, map[string]string, map[string]map[string]any) {
	cmd := niceCmd(selfBinary(), "planids", "--endpoint", endpoint)
	cmd.Env = childEnv(root)
	data, err := cmd.CombinedOutput()
	if err != nil {
		fmt.Println(string(data))
		must("resolving "+endpoint+" on "+root, err)
	}
	var doc struct {
		Release string                    `json:"release"`
		Plans   map[string]string         `json:"plans"`
		Records map[string]map[string]any `json:"records"`
	}
	must("reading the resolver's answer", json.Unmarshal(data, &doc))
	return doc.Release, doc.Plans, doc.Records
}

// sectionPlanIDs is that child: it resolves ONE endpoint through the product's own
// resolver against the COZY_HOME it was given, and prints the identity as a document.
func sectionPlanIDs() {
	cfg, e := config.Load()
	must("config", errOf(e))
	l, e := home.Open(cfg.Home)
	must("layout", errOf(e))
	st, e := records.Open(l.DB)
	must("records", errOf(e))
	defer st.Close()
	spec, e := app.NewResolver(st, cfg).Resolve(flag("endpoint", transportEndpointRef))
	must("resolving the endpoint", errOf(e))
	doc := map[string]any{"release": spec.Placement.ReleaseID}
	plans, recs := map[string]string{}, map[string]map[string]any{}
	for _, b := range spec.Placement.Bindings {
		id, e := b.PlanID()
		must("plan id", errOf(e))
		plans[b.Entrypoint], recs[b.Entrypoint] = id, b.Record
	}
	doc["plans"], doc["records"] = plans, recs
	data, err := json.Marshal(doc)
	must("rendering the identity", err)
	fmt.Println(string(data))
}

// selfBinary is this driver, so a section can run another section as its own child.
func selfBinary() string {
	self, err := os.Executable()
	must("locating this binary", err)
	return self
}

func shortID(id string) string {
	bare := strings.TrimPrefix(id, "sha256:")
	if len(bare) > 12 {
		return "sha256:" + bare[:12]
	}
	return id
}

// podPlanRecord reads back the binding record the OWNER delivered to one pod, from the
// pod's own filesystem. An arm about delivery is answered by what is on the far side.
func podPlanRecord(h *podHub, rentalID, planID string) map[string]any {
	path := filepath.Join(h.pods, rentalID, "binding-plans",
		strings.TrimPrefix(planID, "sha256:")+".json")
	data, err := os.ReadFile(path)
	if err != nil {
		return nil
	}
	var doc map[string]any
	if json.Unmarshal(data, &doc) != nil {
		return nil
	}
	return doc
}

// podMediaFiles lists what landed in one pod's media subtree, relative to it. It is how an
// arm sees that the payload really crossed rather than inferring it from a green terminal.
func podMediaFiles(h *podHub, rentalID string) []string {
	root := h.mediaRootOf(rentalID)
	out := []string{}
	if root == "" {
		return out
	}
	_ = filepath.Walk(root, func(path string, info os.FileInfo, err error) error {
		if err == nil && info != nil && !info.IsDir() {
			rel, _ := filepath.Rel(root, path)
			out = append(out, rel)
		}
		return nil
	})
	sort.Strings(out)
	return out
}

// mediaCall drives one pod's media server DIRECTLY, over TLS with the certificate this
// host pinned. The refusal arms need a caller that can present the wrong credential and
// name the wrong thing, which the product's own client structurally cannot.
func mediaCall(root, rentalID, method, path, bearer string, body []byte) (int, string) {
	pem, err := os.ReadFile(filepath.Join(root, "rentals", rentalID+".pem"))
	if err != nil {
		return 0, "no pinned certificate: " + err.Error()
	}
	pool := x509.NewCertPool()
	if !pool.AppendCertsFromPEM(pem) {
		return 0, "the pinned PEM holds no certificate"
	}
	addr := rentalMedia(root, rentalID)
	if addr == "" {
		return 0, "this rental pins no media address"
	}
	request, err := http.NewRequest(method, "https://"+addr+path, bytes.NewReader(body))
	if err != nil {
		return 0, err.Error()
	}
	request.Header.Set("Authorization", "Bearer "+bearer)
	client := &http.Client{Transport: &http.Transport{
		TLSClientConfig: &tls.Config{
			RootCAs: pool, MinVersion: tls.VersionTLS12, ServerName: workertls.ServerName,
		},
	}}
	response, err := client.Do(request)
	if err != nil {
		return 0, err.Error()
	}
	defer response.Body.Close()
	data, _ := io.ReadAll(response.Body)
	return response.StatusCode, string(data)
}

// rentalMedia is the media address `cozy rent ls` says this host holds for one rental.
func rentalMedia(root, rentalID string) string {
	_, doc, _ := cozyJSON(root, "rent", "ls", "--full")
	rows, _ := doc["rows"].([]any)
	for _, row := range rows {
		fields, _ := row.(map[string]any)
		if fmt.Sprint(fields["rental"]) == rentalID {
			return fmt.Sprint(fields["media"])
		}
	}
	return ""
}

// bytesAt is how many bytes are at a path — 0 for a file that is not there, because the
// arm's question is "does the client hold this" and both answers to that are the same one.
func bytesAt(path string) int {
	info, err := os.Stat(path)
	if err != nil || info.IsDir() {
		return 0
	}
	return int(info.Size())
}

// digestOfFile hashes what is at a path, so an arm can compare the bytes on the client
// against the bytes on the pod rather than trusting either side's claim about them.
func digestOfFile(path string) string {
	data, err := os.ReadFile(path)
	if err != nil {
		return ""
	}
	sum := sha256.Sum256(data)
	return "sha256:" + hex.EncodeToString(sum[:])
}

func digestOfBytes(data []byte) string {
	sum := sha256.Sum256(data)
	return "sha256:" + hex.EncodeToString(sum[:])
}

func lifecycleOutput(life map[string]any, outputID string) map[string]any {
	rows, _ := life["outputs"].([]any)
	for _, raw := range rows {
		row, _ := raw.(map[string]any)
		if fmt.Sprint(row["output_id"]) == outputID {
			return row
		}
	}
	return map[string]any{}
}

func number(value any) float64 {
	n, _ := value.(float64)
	return n
}

func isMP4(data []byte) bool {
	return len(data) >= 12 && bytes.Equal(data[4:8], []byte("ftyp"))
}

// lineWith is the first line of a log that mentions a phrase — the DETAIL an arm prints,
// so a reader sees the peer's own words rather than "true".
func lineWith(log, phrase string) string {
	for _, line := range strings.Split(log, "\n") {
		if strings.Contains(line, phrase) {
			return strings.TrimSpace(line)
		}
	}
	return "(no line said " + phrase + ")"
}

// planFor looks one entrypoint up by the FUNCTION name a user types. The
// binding's own key is the descriptor's entrypoint name, and the release under test
// registers several — matching on the suffix keeps the arms readable without the driver
// re-deriving the descriptor's naming rule.
func planFor(plans map[string]string, function string) string {
	for name, id := range plans {
		if name == function || strings.HasSuffix(name, "/"+function) {
			return id
		}
	}
	return ""
}

// alivePID answers whether a pid is still a live process — the kernel's own answer, which
// is what an arm about a killed process needs rather than this driver's memory of it.
func alivePID(pid int) bool {
	proc, err := os.FindProcess(pid)
	if err != nil {
		return false
	}
	return proc.Signal(syscall.Signal(0)) == nil
}

// dispatchedRequest reads the request id out of the stream a `--stream` run printed, so an
// arm can name the media slot the orchestrator's own grant pointed at.
func dispatchedRequest(stream string) string {
	if _, rest, ok := strings.Cut(stream, `"request_id":"`); ok {
		if id, _, ok := strings.Cut(rest, `"`); ok {
			return id
		}
	}
	return ""
}
