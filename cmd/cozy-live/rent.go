package main

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"
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

func sectionRent() {
	root := flag("home", filepath.Join(os.TempDir(), "cozy-live", "cl015-rent"))
	must("clearing the root", os.RemoveAll(root))
	must("creating the root", os.MkdirAll(root, 0o755))

	head("a real endpoint, installed the way a user installs one")
	installEndpoint(root)
	port := freePort(2990)
	svc := startService(root, port, false)
	defer svc.stop()
	// The honest pod: it writes where the grant says, and refuses a grant it cannot reach.
	hub := startPodHub(filepath.Join(root, "hub"), "remote", false)
	defer hub.close()
	fmt.Printf("  the stand-in hub is at %s · pod filesystem %s\n", hub.url(), hub.pods)

	head("the credential arms come FIRST: a rental is a first-party write")
	code, out := cozyRunEnv(root, []string{"TENSORHUB_URL=" + hub.url()},
		"rent", "h3", "--card", "H200", "--reason", "cl-015 live")
	check("no configured token -> 5, answered BEFORE the dial",
		code == 5 && strings.Contains(out, "no admin token is configured"), firstLine(out))
	code, out = cozyRunEnv(root,
		[]string{"TENSORHUB_URL=" + hub.url(), "TENSORHUB_TOKEN=" + randomHex(16)},
		"rent", "h3", "--card", "H200", "--reason", "cl-015 live")
	check("a FOREIGN token -> 5 carrying the hub's own typed refusal",
		code == 5 && strings.Contains(out, "rental.unauthenticated"), firstLine(out))

	head("the ask names its hardware and its reason, or it is a usage refusal")
	code, out = cozyRunEnv(root, hub.env(), "rent", "h3", "--reason", "cl-015 live")
	check("no --card -> 2", code == 2 && strings.Contains(out, "--card is required"), firstLine(out))
	code, out = cozyRunEnv(root, hub.env(), "rent", "h3", "--card", "H200")
	check("no --reason -> 2: the hub records why before it spends",
		code == 2 && strings.Contains(out, "--reason is required"), firstLine(out))

	head("cozy rent — provision, poll to ready, PIN the triple")
	t0 := time.Now()
	rentalA, out := rentOne(root, hub, "cl-015 target A")
	rentMS := elapsedMS(t0)
	fmt.Println(indent(out))
	check("cozy rent -> 0 with a rental id", strings.HasPrefix(rentalA, "rnt-"), rentalA)
	check("it reports the pod's own state, address and pod id",
		strings.Contains(out, "state:") && strings.Contains(out, "127.0.0.1:") &&
			strings.Contains(out, "pod-"), field(out, "address")+" "+field(out, "pod"))
	fmt.Printf("  bench ask -> ready -> pinned: %d ms\n", rentMS)

	head("the owner token is a SECRET, and every arm here is about it not leaking")
	planted := hub.tokenOf(rentalA)
	check("the driver knows the value the hub issued", len(planted) == 64, "64 hex chars")
	check("and it appears NOWHERE in what cozy printed",
		!strings.Contains(out, planted), "searched the whole rendering")
	check("what IS printed is its digest, comparable against the pod's own",
		strings.Contains(field(out, "owner_token"), "sha256:"), field(out, "owner_token"))
	tokenFile := filepath.Join(root, "rentals", rentalA+".token")
	info, err := os.Stat(tokenFile)
	check("the token landed in a 0600 file, not a record and not argv",
		err == nil && info != nil && info.Mode().Perm() == 0o600,
		tokenFile+" "+modeOf(info))
	stored, err := os.ReadFile(tokenFile)
	check("and the bytes on disk are exactly the token the hub issued",
		err == nil && strings.TrimSpace(string(stored)) == planted, "")
	pemBytes, err := os.ReadFile(filepath.Join(root, "rentals", rentalA+".pem"))
	check("the certificate to PIN is beside it, and it is a certificate",
		err == nil && strings.HasPrefix(string(pemBytes), "-----BEGIN CERTIFICATE-----"), "")

	head("cozy rent ls — the listing renders the digest and never the value")
	code, out = cozyRun(root, "rent", "ls", "--full")
	fmt.Println(indent(out))
	check("cozy rent ls names the rental this host holds",
		code == 0 && strings.Contains(out, rentalA), firstLine(out))
	check("with the owner token digested, and the raw value absent",
		strings.Contains(out, "sha256:") && !strings.Contains(out, planted), "")

	head("RED: a run pinned to a rental this host does not hold")
	_, before, _ := cozyJSON(root, "status")
	code, out = cozyRun(root, "run", endpointRef+"/v1/denoise", "steps=2", "--worker", "rnt-nope")
	check("cozy run --worker <unknown> -> 4, the server's own envelope",
		code == 4 && strings.Contains(out, "no rental rnt-nope"), firstLine(out))
	_, after, _ := cozyJSON(root, "status")
	check("and NO request was recorded: an unplaceable pin is refused before the row",
		before["requests"] == after["requests"],
		fmt.Sprint(before["requests"])+" -> "+fmt.Sprint(after["requests"]))

	head("THE LEG: attach over TLS, claim with the provisioned token, dispatch to the pod")
	outDir := filepath.Join(root, "out")
	t0 = time.Now()
	code, out = cozyRun(root, "run", endpointRef+"/v1/denoise", "steps=2",
		"--out", outDir, "--stream", "--worker", rentalA)
	legMS := elapsedMS(t0)
	instanceA := instanceFor(root, rentalA)
	podLog := podWorkerLog(hub, rentalA)
	check("the pod hosted WorkerControl behind TLS",
		strings.Contains(podLog, "hosting behind TLS"), "")
	check("and it ACCEPTED the claim — the owner presented the provisioned owner token",
		strings.Contains(podLog, "ClaimAck sent"), "")
	check("the coordinator ATTACHED rather than spawned: the slot is the RENTAL's",
		strings.HasSuffix(instanceEndpoint(root, instanceA), "@"+rentalA),
		instanceEndpoint(root, instanceA)+" "+instanceA)
	check("the attempt was DISPATCHED to that pod's own instance",
		instanceA != "" && dispatchedInstance(out) == instanceA,
		dispatchedInstance(out)+" == "+instanceA)
	fmt.Printf("  bench attach + claim + dispatch over TLS: %d ms\n", legMS)

	head("THE BYTE BOUNDARY IS REAL: the pod refuses a destination on the owner's disk")
	check("the pod said so by name",
		strings.Contains(podLog, "not on this machine"), "")
	check("and the client's answer is the POD's own words, not a local guess",
		code == 11 && strings.Contains(out, "owner's filesystem"),
		firstLine(out)+" [exit "+itoa(code)+"]")
	check("nothing was written into the client's output directory",
		!exists(filepath.Join(outDir, "image.png")), outDir)

	head("EXACT ROUTING: a second attached target advertising the SAME plan")
	rentalB, _ := rentOne(root, hub, "cl-015 target B")
	code, out = cozyRun(root, "run", endpointRef+"/v1/denoise", "steps=2",
		"--stream", "--worker", rentalB)
	instanceB := instanceFor(root, rentalB)
	check("B is its own instance, not A's", instanceB != "" && instanceB != instanceA,
		instanceA+" vs "+instanceB)
	check("a run pinned to B lands on B — the pin ROUTES, it does not merely resolve",
		dispatchedInstance(out) == instanceB, dispatchedInstance(out))
	check("and it never lands on A", dispatchedInstance(out) != instanceA, "")
	_ = code

	head("an UNPINNED request never consumes rented capacity")
	code, out = cozyRun(root, "run", endpointRef+"/v1/denoise", "steps=2",
		"--stream", "--timeout", "6s")
	landed := dispatchedInstance(out)
	check("it was NOT dispatched to either rented pod",
		landed != instanceA && landed != instanceB,
		"landed on "+nothingOr(landed)+" (A "+instanceA+", B "+instanceB+")")
	check("it looked for LOCAL capacity instead", code != 0 || landed != "", firstLine(out))
	code, out = cozyRun(root, "stop", "--all")
	check("cozy stop --all clears whatever that started", code == 0, firstLine(out))

	head("RED: a FOREIGN owner token cannot drive a pod")
	rentalC, _ := rentOne(root, hub, "cl-015 foreign-token arm")
	must("planting a foreign owner token", os.WriteFile(
		filepath.Join(root, "rentals", rentalC+".token"), []byte(randomHex(32)+"\n"), 0o600))
	code, out = cozyRun(root, "run", endpointRef+"/v1/denoise", "steps=2", "--worker", rentalC)
	check("the run does NOT succeed: the pod refused a claim it did not provision",
		code != 0, firstLine(out)+" [exit "+itoa(code)+"]")
	check("and the pod says so in its own words",
		strings.Contains(podWorkerLog(hub, rentalC), "Claim REFUSED"), "")

	head("a widened token file is refused AT THE DIAL, which is the only place it is read")
	// A FRESH rental, because the check lives where the credential is read: the HTTP layer
	// asks only whether this host holds the pin, and an already-attached worker is never
	// re-resolved. That is the point of the split — the token has exactly one reader.
	rentalE, _ := rentOne(root, hub, "cl-015 widened-token arm")
	must("widening the token mode", os.Chmod(filepath.Join(root, "rentals", rentalE+".token"), 0o644))
	code, out = cozyRun(root, "run", endpointRef+"/v1/denoise", "steps=2", "--worker", rentalE)
	check("mode 0644 refuses by name, and nothing was dialled",
		code != 0 && strings.Contains(out, "0600"), firstLine(out)+" [exit "+itoa(code)+"]")
	check("the pod was never claimed: no worker for it exists",
		instanceFor(root, rentalE) == "", instanceFor(root, rentalE))

	head("THE MIRROR: an output the owner does not hold is NEVER acked")
	// This pod does the work and writes the bytes on its own disk — a true manifest about
	// a real file the owner cannot see. It is what every real pod does today.
	liar := startPodHub(filepath.Join(root, "hub-unmirrored"), "remotelie", false)
	defer liar.close()
	rentalD, _ := rentOne(root, liar, "cl-015 unmirrored-output arm")
	mirrorDir := filepath.Join(root, "out-mirror")
	code, out = cozyRun(root, "run", endpointRef+"/v1/denoise", "steps=2",
		"--out", mirrorDir, "--timeout", "12s", "--worker", rentalD)
	check("the pod DID write bytes — on its own filesystem",
		podWroteOutput(liar, rentalD), filepath.Join(liar.pods, rentalD))
	check("the owner did not ack it: the request never settled and the deadline answered",
		code == 10, firstLine(out)+" [exit "+itoa(code)+"]")
	check("the coordinator said WHY: the declared output is not readable here",
		strings.Contains(serviceLog(root), "cannot be read here"), "")
	check("no output became visible and no file was written",
		!exists(filepath.Join(mirrorDir, "image.png")), mirrorDir)

	head("cozy rent release — plan first, then the pod is gone")
	code, out = cozyRunEnv(root, hub.env(), "rent", "release", rentalC)
	check("without --yes it prints the plan and exits 0",
		code == 0 && strings.Contains(out, "DESTROYS"), firstLine(out))
	check("and it changed nothing: the hub still holds the pod", !hub.released(rentalC), "")
	code, out = cozyRunEnv(root, hub.env(), "rent", "release", rentalC, "--yes")
	check("with --yes it exits 0 and the hub SAW the delete",
		code == 0 && hub.released(rentalC), firstLine(out))
	_, err = os.Stat(filepath.Join(root, "rentals", rentalC+".token"))
	check("the owner token is gone from this host", os.IsNotExist(err), "")
	code, out = cozyRun(root, "run", endpointRef+"/v1/denoise", "steps=2", "--worker", rentalC)
	check("a run pinned to the released rental is 4 again", code == 4, firstLine(out))

	head("RED: releasing something this host does not hold")
	code, out = cozyRunEnv(root, hub.env(), "rent", "release", "rnt-nope", "--yes")
	check("cozy rent release <unknown> -> 4, before any dial",
		code == 4 && strings.Contains(out, "no rental rnt-nope"), firstLine(out))

	head("RED: a hub whose pods never come up")
	broken := startPodHub(filepath.Join(root, "hub-broken"), "remote", true)
	defer broken.close()
	code, out = cozyRunEnv(root, broken.env(), "rent", "h3", "--card", "H200",
		"--reason", "cl-015 failed-provision arm")
	check("a rental that fails to provision -> 11, carrying the hub's own detail",
		code == 11 && strings.Contains(out, "no capacity"), firstLine(out))
	check("and nothing was pinned for it: there is no triple to pin",
		!strings.Contains(out, "owner_token"), "")

	head("teardown: every rental this host holds is released and ZERO is proved")
	for _, id := range heldRentals(root) {
		on := hub
		switch {
		case id == rentalD:
			on = liar
		case !hub.holds(id) && broken.holds(id):
			on = broken
		}
		code, out = cozyRunEnv(root, on.env(), "rent", "release", id, "--yes")
		check("released "+id, code == 0, firstLine(out))
	}
	check("this host holds ZERO rentals — the teardown is proved, not assumed",
		zeroRentals(root), "")
}

// rentOne rents a pod and returns its id with the whole rendering, so an arm can search
// what was printed as well as act on the id.
func rentOne(root string, h *podHub, reason string) (string, string) {
	code, out := cozyRunEnv(root, h.env(), "rent", "h3", "--card", "H200", "--reason", reason)
	if code != 0 {
		fmt.Println(indent(out))
		return "", out
	}
	return field(out, "rental"), out
}

// dispatchedInstance reads the instance the coordinator actually placed the attempt on,
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
