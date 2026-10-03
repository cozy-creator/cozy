package producttest

import (
	"context"
	"encoding/base64"
	"encoding/pem"
	"os"
	"os/exec"
	"strings"
	"testing"
	"time"

	"github.com/cozy-creator/cozy/internal/home"
	"github.com/cozy-creator/cozy/internal/machines"
	"github.com/cozy-creator/cozy/internal/machinev1"
	"github.com/cozy-creator/cozy/internal/workertls"
	pb "github.com/cozy-creator/cozy/protocol/cozy/machine/v1"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

// The machine a rental runs, on cozy.machine.v1: anyone reaching it reads its identity and
// sealed receipt; its owner reads the whole machine, and one keepalive moves the idle deadline
// while merely watching never does. An update naming a wheel this owner never wrote is refused.
func TestMachineV1StatusKeepaliveAndUpdate(t *testing.T) {
	if *machineHostBinary == "" {
		t.Skip("requires -machine-host: a machine serving cozy.machine.v1")
	}
	uv, err := exec.LookPath("uv")
	if err != nil {
		t.Skip("uv lays out the machine root")
	}
	h := newMachineHub(t)
	root, err := os.MkdirTemp("", "czs")
	must(t, err)
	t.Cleanup(func() {
		if !t.Failed() {
			_ = removeAllForce(root)
		}
	})
	layout, problem := home.Open(root)
	fatal(t, problem)
	source := machines.Source{Host: *machineHostBinary, RuntimeWheel: *machineRuntimeWheel, TensorFSWheel: *machineTensorFSWheel}
	launch, identity, _, _ := providerHost(t, h, layout, source, uv)
	pin, err := workertls.ParsePin(pem.EncodeToMemory(&pem.Block{Type: "CERTIFICATE", Bytes: launch.Leaf}))
	must(t, err)
	public, err := base64.RawURLEncoding.DecodeString(identity.PublicKey())
	must(t, err)
	client, err := machinev1.Dial(launch.Addr, pin.TLSConfig(), launch.WorkerID, machinev1.Signer{Public: public, Sign: identity.Sign})
	must(t, err)
	defer client.Close()
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Minute)
	defer cancel()

	anyone, err := machinev1.Identity(ctx, launch.Addr, pin.TLSConfig())
	must(t, err)
	if anyone.WorkerId != launch.WorkerID || anyone.BootId != launch.BootID || len(anyone.Receipt) == 0 {
		t.Fatalf("the identity frame does not name this machine's boot and receipt: %v", anyone)
	}
	if anyone.IdleDeadlineUnixMs != 0 || len(anyone.Runs) != 0 || len(anyone.Hubs) != 0 {
		t.Fatal("a caller without a capability read more than identity")
	}

	watched := make(chan *pb.StatusFrame, 8)
	watch, stop := context.WithCancel(ctx)
	defer stop()
	go func() {
		_ = client.Watch(watch, func(frame *pb.StatusFrame) error { watched <- frame; return nil })
	}()
	first := <-watched
	if first.Phase != "ready" || first.IdleDeadlineUnixMs <= 0 || string(first.Receipt) != string(anyone.Receipt) {
		t.Fatalf("the owner's picture is not a ready rental with a deadline: %v", first)
	}
	if len(first.Hubs) != 1 || first.Hubs[0].Origin != h.worker.URL {
		t.Fatalf("the picture does not name the rental's Hub: %v", first.Hubs)
	}
	select {
	case frame := <-watched:
		t.Fatalf("an unchanged machine sent another frame: %v", frame)
	case <-time.After(2500 * time.Millisecond):
	}
	renewed, err := client.Keepalive(ctx)
	must(t, err)
	if renewed.IdleDeadlineUnixMs <= first.IdleDeadlineUnixMs {
		t.Fatalf("keepalive did not move the deadline: %d -> %d", first.IdleDeadlineUnixMs, renewed.IdleDeadlineUnixMs)
	}
	select {
	case frame := <-watched:
		if frame.IdleDeadlineUnixMs != renewed.IdleDeadlineUnixMs {
			t.Fatalf("the watcher saw another deadline: %d", frame.IdleDeadlineUnixMs)
		}
	case <-time.After(10 * time.Second):
		t.Fatal("the watcher was not told of the new deadline")
	}

	unwritten := &machinev1.Member{Wheel: "cozy_runtime-9.9.9-py3-none-any.whl", Digest: "sha256:" + strings.Repeat("0", 64), Length: 1}
	_, err = client.Update(ctx, "update-unwritten", machinev1.Cohort{Runtime: unwritten}, nil)
	if status.Code(err) != codes.FailedPrecondition {
		t.Fatalf("an update of a wheel the owner never wrote was not refused: %v", err)
	}
}
