package main

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"syscall"
	"time"

	"github.com/cozy-creator/cozy-creator-v2/internal/canonical"
	"github.com/cozy-creator/cozy-creator-v2/internal/media"
	"github.com/cozy-creator/cozy-creator-v2/internal/orchestrator"
	"github.com/cozy-creator/cozy-creator-v2/internal/secret"
)

// sectionTLSPin is cl-026's certificate-substitution arm, guarding cl-019's design: the
// fixed SAN (`cozy-worker`) carries no per-pod identity, so per-pod identity IS the
// per-rental PINNED certificate. If a future regression ever made rentals share a CA,
// every pod's certificate would silently become valid for every pod — this arm is what
// would catch it: rental A's certificate presented where rental B's pin is checked MUST
// refuse, on both of the pod's listeners.
func sectionTLSPin() {
	lv := hostCoordinator("tlspin", true)
	defer lv.close()
	dir := filepath.Join(lv.root, "tls-arm")
	podRoot, runRoot := filepath.Join(dir, "pod"), filepath.Join(dir, "run")
	for _, p := range []string{dir, podRoot, runRoot} {
		must("creating "+p, os.MkdirAll(p, 0o755))
	}

	head("two rentals, two certificates — the pin is per-rental, never a shared CA")
	certA, keyA := filepath.Join(dir, "rental-a.pem"), filepath.Join(dir, "rental-a.key")
	certB, keyB := filepath.Join(dir, "rental-b.pem"), filepath.Join(dir, "rental-b.key")
	_, err := writeSelfSigned(certA, keyA)
	must("minting rental A's certificate", err)
	_, err = writeSelfSigned(certB, keyB)
	must("minting rental B's certificate", err)

	// The renter-minted token, provisioned to the pod as a hash set — both pod processes
	// verify it; neither holds it.
	token := secret.New("tls-arm-" + randomHex(24))
	tokens := filepath.Join(dir, "pod.tokens")
	must("provisioning the pod token hashes",
		os.WriteFile(tokens, []byte(secret.HashLine(token)+"\n"), 0o600))

	// RENTAL A'S POD: a fakeworker hosting WorkerControl behind certificate A, and the
	// REAL cozy-media beside it holding the same keys — the same two processes the
	// stand-in hub provisions.
	self, err := os.Executable()
	must("locating this binary", err)
	worker := niceCmd(self, "fakeworker", "--arm", "idle",
		"--socket", "127.0.0.1:0", "--out", runRoot,
		"--tls-cert", certA, "--tls-key", keyA,
		"--tokens", tokens, "--instance-id", "ins-tlspin-pod",
		"--release-id", release, "--cozy-home", podRoot)
	worker.Env = childEnv(podRoot)
	workerLog, err := os.Create(filepath.Join(runRoot, "pod-worker.log"))
	must("the pod worker log", err)
	worker.Stdout, worker.Stderr = workerLog, workerLog
	setProcessGroup(worker)
	must("starting the pod worker", worker.Start())
	defer func() { _ = killGroup(worker.Process.Pid, syscall.SIGKILL); _ = worker.Wait() }()
	controlAddr := awaitAddr(filepath.Join(runRoot, "control.addr"), worker)
	check("the pod worker hosts WorkerControl behind TLS", controlAddr != "", controlAddr)

	mediaCmd := niceCmd(mediaBinary(),
		"--listen", "127.0.0.1:0", "--root", filepath.Join(podRoot, "media"),
		"--plans", filepath.Join(podRoot, "binding-plans"),
		"--tokens", tokens, "--out", runRoot,
		"--tls-cert", certA, "--tls-key", keyA,
		"--quota", "134217728", "--max-body", "4194304")
	mediaLog, err := os.Create(filepath.Join(runRoot, "pod-media.log"))
	must("the pod media log", err)
	mediaCmd.Stdout, mediaCmd.Stderr = mediaLog, mediaLog
	setProcessGroup(mediaCmd)
	must("starting the pod media server", mediaCmd.Start())
	defer func() { _ = killGroup(mediaCmd.Process.Pid, syscall.SIGKILL); _ = mediaCmd.Wait() }()
	mediaAddr := awaitAddr(filepath.Join(runRoot, "media.addr"), mediaCmd)
	check("the pod media server is behind the same certificate", mediaAddr != "", mediaAddr)

	// The delivered plan must be one exact canonical EntrypointBindingPlan — the pod's
	// media server re-derives its identity from the bytes — so the arm mints one and
	// carries it the way a rental carries Tensorhub's (BindingPlanSubject with bytes).
	planBytes, err := canonical.Write(map[string]canonical.Value{
		"format":     "cozy.endpoint.EntrypointBindingPlan/1",
		"entrypoint": "fake",
		"descriptor": map[string]canonical.Value{"functions": []canonical.Value{"fake"}},
		"bindings":   []canonical.Value{},
	})
	must("minting the arm's binding plan", err)
	planID, err := canonical.Spell(canonical.Digest(planBytes))
	must("spelling the arm's plan id", err)
	placement := orchestrator.DesiredPlacement{
		Endpoint: "fake/tlspin@rnt-a", ReleaseID: release,
		Bindings: []*orchestrator.Binding{{
			Entrypoint: "fake",
			RuntimePlan: &orchestrator.BindingPlanSubject{
				SubjectID: planID, Kind: "plan", Digest: planID,
				Length: uint64(len(planBytes)), CanonicalBytes: planBytes,
			},
		}},
	}
	specWith := func(controlPin, mediaPin string) orchestrator.WorkerLaunchSpec {
		return orchestrator.WorkerLaunchSpec{
			Placement: placement,
			Connection: &orchestrator.WorkerConnection{
				Addr: controlAddr, Token: token, CACert: controlPin,
				Media: &media.Spec{Addr: mediaAddr, Token: token, CACert: mediaPin},
			},
		}
	}

	head("RED: rental A's certificate where rental B's pin is checked — the BYTE plane")
	_, _, e := lv.c.EnsureWorker(specWith(certB, certB))
	check("the media dial refuses the foreign certificate", e != nil &&
		strings.Contains(strings.ToLower(e.Message+e.Remedy), "certificate"), briefly(e))

	head("RED: the same substitution on the CONTROL leg")
	// The byte plane gets the CORRECT pin so the refusal observed is the control leg's own.
	instance, _, e := lv.c.EnsureWorker(specWith(certB, certA))
	check("the worker attaches (the dial verdict is the stream's)", e == nil, briefly(e))
	line, ok := waitEvent(lv, "authentication handshake failed", 30*time.Second)
	check("the control dial refuses the foreign certificate at the handshake", ok, trimLog(line))
	facts := lv.c.Worker(instance)
	boot := "(no worker row)"
	if facts != nil {
		boot = fmt.Sprintf("boot %q", facts.BootID)
	}
	check("and NOTHING was claimed under the wrong pin", facts != nil && facts.BootID == "", boot)
	lv.c.ShutdownWorker(instance, 5*time.Second)

	head("the positive control: the SAME pod admits under its OWN pinned certificate")
	instance, _, e = lv.c.EnsureWorker(specWith(certA, certA))
	check("the worker attaches under rental A's own pin", e == nil, briefly(e))
	line, ok = waitEvent(lv, "ClaimAck boot=", 30*time.Second)
	check("the claim lands — the earlier refusals were the PIN, not the peer", ok, trimLog(line))

	head("teardown")
	lv.c.Close(10 * time.Second)
	rows, _ := lv.store.LiveWorkers()
	check("every worker row is closed", len(rows) == 0, fmt.Sprintf("%d live row(s)", len(rows)))
}
