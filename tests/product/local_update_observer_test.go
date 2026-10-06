package producttest

import (
	"context"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/cozy-creator/cozy/internal/config"
	"github.com/cozy-creator/cozy/internal/machines"
	"github.com/cozy-creator/cozy/internal/machinev1"
	pb "github.com/cozy-creator/cozy/protocol/cozy/machine/v1"
)

// A machine commits its update even when the controller stops observing it. A later CLI
// launch must use that committed software without requiring another install operation.
func TestLocalMachineStartsAfterItsUpdateObserverDisconnects(t *testing.T) {
	if *machineRuntimeWheel == "" || *machineUpdateWheel == "" || *machineTensorFSWheel == "" {
		t.Skip("requires original and distinct update Runtime wheels plus TensorFS")
	}
	hub := newMachineHub(t)
	root, err := os.MkdirTemp("", "czobs")
	must(t, err)
	must(t, os.WriteFile(filepath.Join(root, config.FileName), []byte("tensorhub_url: "+hub.server.URL+"\ntensorhub_token: local-update-test\n"), 0o600))
	t.Cleanup(func() {
		_, _ = runCozy(t, root, "machine", "stop")
		_, _ = runCozy(t, root, "down")
		if t.Failed() {
			t.Logf("evidence retained at %s\n%s", root, tail(filepath.Join(root, "machine", "host.log")))
		} else {
			_ = removeAllForce(root)
		}
	})
	for _, args := range [][]string{
		{"machine", "install", "--runtime-wheel", *machineRuntimeWheel, "--tensorfs-wheel", *machineTensorFSWheel},
		{"machine", "start"},
	} {
		if code, out := runCozy(t, root, args...); code != 0 {
			t.Fatalf("%v [%d]: %s", args, code, out)
		}
	}
	host := machines.NewHost(filepath.Join(root, "machine"), "", nil)
	resolver := &machines.Resolver{Host: host}
	machine, problem := resolver.DialV1(machines.AttachOnly(t.Context()), machines.Local, "local update observer")
	fatal(t, problem)
	defer machine.Close()
	ctx, cancel := context.WithTimeout(t.Context(), time.Minute)
	defer cancel()
	wheel := *machineUpdateWheel
	file, err := os.Open(wheel)
	must(t, err)
	defer file.Close()
	info, err := file.Stat()
	must(t, err)
	member := &machinev1.Member{Wheel: filepath.Base(wheel), Digest: "sha256:" + fileSHA(t, wheel), Length: uint64(info.Size())}
	must(t, machine.Write(ctx, member.Digest, member.Length, file))
	cohort := machinev1.Cohort{Agent: "bundled", Runtime: member}
	observe, disconnect := context.WithCancel(ctx)
	defer disconnect()
	accepted := false
	_, err = machine.Update(observe, "observer-disconnected", cohort, func(event *pb.RunEvent) {
		if event.GetState() != nil {
			accepted = true
			disconnect()
		}
	})
	if !accepted || err == nil || observe.Err() == nil {
		t.Fatalf("the accepted update observer did not disconnect: accepted=%v err=%v", accepted, err)
	}
	// Reattaching observes the same durable operation. Neither observer is Host.Install,
	// so no client has copied the new service into the local launcher's executable path.
	outcome, err := machine.Update(ctx, "observer-disconnected", cohort, nil)
	must(t, err)
	if outcome.Status != "succeeded" {
		t.Fatalf("the detached update did not commit: %v", outcome)
	}
	for range 2 {
		if code, out := runCozy(t, root, "machine", "stop"); code != 0 {
			t.Fatalf("machine stop [%d]: %s", code, out)
		}
		if code, out := cozyWithin(t, root, time.Minute, "machine", "start"); code != 0 {
			t.Fatalf("the committed update cannot restart after observer loss [%d]: %s", code, out)
		}
	}
	if fileSHA(t, filepath.Join(host.Root(), "usr/local/bin/cozy-machine")) != fileSHA(t, filepath.Join(host.Root(), "var/lib/cozy/rust-machine/agent/current")) {
		t.Fatal("the next launch did not retain the committed machine executable")
	}
}
