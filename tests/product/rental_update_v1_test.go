package producttest

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"encoding/pem"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/cozy-creator/cozy/internal/machinev1"
	"github.com/cozy-creator/cozy/internal/workertls"
)

// `cozy rental update` with local wheels sends them to the rental's machine with Write and
// runs Run kind: update there: the machine restarts its service on the candidate in place
// (same boot), the daemon records the outcome, and a daemon that crashed mid-update attaches
// to the same update rather than starting another.
func TestRentalUpdateRunsOnTheMachineAcrossADaemonCrash(t *testing.T) {
	root, store, launch, identity, daemon := statusRental(t)
	pin, err := workertls.ParsePin(pem.EncodeToMemory(&pem.Block{Type: "CERTIFICATE", Bytes: launch.Leaf}))
	must(t, err)
	public, err := base64.RawURLEncoding.DecodeString(identity.PublicKey())
	must(t, err)
	machine, err := machinev1.Dial(launch.Addr, pin.TLSConfig(), launch.WorkerID, machinev1.Signer{Public: public, Sign: identity.Sign})
	must(t, err)
	defer machine.Close()
	runtime := func() string {
		t.Helper()
		ctx, cancel := context.WithTimeout(context.Background(), time.Minute)
		defer cancel()
		for {
			frame, err := machine.Status(ctx)
			if err == nil && frame.Phase == "ready" {
				if frame.BootId != launch.BootID {
					t.Fatalf("the update changed the machine's boot: %s", frame.BootId)
				}
				return frame.Runtime
			}
			if ctx.Err() != nil {
				t.Fatalf("the machine did not answer ready: %v", err)
			}
			time.Sleep(200 * time.Millisecond)
		}
	}
	version := func(wheel string) string {
		name := strings.TrimPrefix(filepath.Base(wheel), "cozy_runtime-")
		return name[:strings.Index(name, "-")]
	}
	update := func(candidate string) map[string]any {
		t.Helper()
		code, out := runCozy(t, root, "rental", "update", "tessa", "--runtime-wheel", candidate, "--tensorfs-wheel", *machineTensorFSWheel, "--json")
		var shown map[string]any
		if code != 0 || json.Unmarshal([]byte(lastJSONLine(out)), &shown) != nil {
			t.Fatalf("rental update [exit %d]: %s\n%s", code, out, tail(filepath.Join(root, "daemon.log")))
		}
		return shown
	}

	first := localBuild(t, *machineRuntimeWheel, "v1update")
	shown := update(first)
	if shown["runtime"] != version(first) || shown["status"] != "ready" || runtime() != version(first) {
		t.Fatalf("the update did not leave the machine on %s: %v", version(first), shown)
	}

	// The daemon crashes while the machine updates; the next daemon finishes the same update.
	second := localBuild(t, *machineRuntimeWheel, "v1crash")
	command := exec.Command("/usr/bin/nice", "-n", "19", cozyBin, "rental", "update", "tessa", "--runtime-wheel", second, "--tensorfs-wheel", *machineTensorFSWheel, "--json")
	command.Env = childEnv(t, root)
	must(t, command.Start())
	t.Cleanup(func() { _ = command.Process.Kill(); _ = command.Wait() })
	eventually(t, root, "the daemon sends the update", func() bool {
		row, problem := store.RuntimeUpdate(parityRental)
		return problem == nil && row != nil && row.State == "updating"
	})
	sent, problem := store.RuntimeUpdate(parityRental)
	fatal(t, problem)
	daemon = crashAndRestartTransactionDaemon(t, daemon)
	_ = daemon
	eventually(t, root, "the resumed update settles", func() bool {
		row, problem := store.RuntimeUpdate(parityRental)
		return problem == nil && row != nil && !row.Active()
	})
	settled, problem := store.RuntimeUpdate(parityRental)
	fatal(t, problem)
	if settled.ID != sent.ID || settled.State != "succeeded" || runtime() != version(second) {
		t.Fatalf("the resumed update did not settle the sent one: sent %s, settled %+v, machine on %s", sent.ID, settled, runtime())
	}
}
