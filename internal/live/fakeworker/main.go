// cozy-fakeworker is a SECOND, INDEPENDENT implementation of the worker side of
// `cozy.worker.v1`, in Go, that hosts `WorkerControl` exactly as cozy-runtime does
// (#436 — the record owner dials). It exists so the orchestrator's refusal arms have a
// real peer to refuse: real protocol bytes, real canonical documents, a real process.
// Nothing here is a mock, and nothing in the orchestrator knows it exists.
//
// It is spawned by `internal/live`'s suite as an ordinary worker: the orchestrator
// appends its own launch grammar (--socket/--out/--instance-id/--release-id/--devices/
// --grace), and `--arm` picks which of the adversary behaviours this process plays.
package main

import (
	"flag"
	"fmt"
	"net"
	"os"
	"path/filepath"
	"strings"

	"google.golang.org/grpc"

	"github.com/cozy-creator/cozy-creator/internal/config"
	pb "github.com/cozy-creator/cozy-creator/protocol/cozy/worker/v1"
)

var (
	socket = flag.String("socket", "", "the listen grant: a unix path or host:port")
	out    = flag.String("out", "", "the run root whose control.addr publishes the bound address")
	// --fake-instance wins over the --instance-id the orchestrator appends, so an arm can
	// report an instance identity this owner never spawned.
	fakeInstance = flag.String("fake-instance", "", "report an instance identity nobody spawned")
	instanceID   = flag.String("instance-id", "", "")
	releaseID    = flag.String("release-id", "", "")
	arm          = flag.String("arm", "idle", "idle|badcred|badrelease|steal|badterminal|dropack|output|missing-output")
	session      = flag.String("session", "", "a fixed worker_boot_id (the collision arm)")
	cozyHome     = flag.String("cozy-home", "", "this worker's own root")
	stealRequest = flag.String("request", "", "the steal arm's victim request")
	stealAttempt = flag.Uint64("attempt", 1, "the steal arm's victim ordinal")
	stealSpec    = flag.String("spec", "", "the steal arm's victim invocation digest, hex")
	_            = flag.String("devices", "", "")
	_            = flag.String("grace", "", "")
	_            = flag.Bool("no-warm", false, "")
)

func main() { os.Exit(run()) }

func run() int {
	flag.Parse()
	instance := *fakeInstance
	if instance == "" {
		instance = *instanceID
	}
	release := *releaseID
	if *arm == "badrelease" {
		release = "cozy/not-the-pinned-release@v0"
	}
	boot := *session
	if boot == "" {
		boot = "boot-fake-" + randomHex(8)
	}
	say := func(format string, args ...any) {
		fmt.Printf("[fake %s] %s\n", *arm, fmt.Sprintf(format, args...))
		_ = os.Stdout.Sync()
	}

	// Bind exactly as the real worker does: a unix path, or host:port loopback.
	network, address := "unix", *socket
	if strings.Contains(*socket, ":") && !strings.ContainsAny(*socket, `/\`) {
		network, address = "tcp", *socket
	}
	if network == "unix" {
		_ = os.MkdirAll(filepath.Dir(address), 0o755)
	}
	_ = os.Remove(address)
	ln, err := net.Listen(network, address) //cozy:allow the ADVERSARY binds exactly as the real worker does (#436: the worker hosts, the owner dials); the product binds through internal/api
	if err != nil {
		say("cannot bind %s: %v", *socket, err)
		return 1
	}
	bound := address
	if network == "tcp" {
		bound = ln.Addr().String()
	}
	// The discovery contract: publish the bound address atomically for the owner to dial.
	if *out != "" && os.MkdirAll(*out, 0o755) == nil {
		staged := filepath.Join(*out, "control.addr.staging")
		if os.WriteFile(staged, []byte(bound+"\n"), 0o644) == nil {
			_ = os.Rename(staged, filepath.Join(*out, "control.addr"))
		}
	}
	say("hosting WorkerControl at %s (boot %s)", bound, boot)

	// THE CLAIM CREDENTIAL. The launcher mints a per-spawn bootstrap credential and hands
	// it over through the one channel only this child inherits; both ends compare it in
	// constant time and neither reads a raw value out of the Value that holds it. The
	// `badcred` arm refuses EVERY proof, so the owner's correct one is refused too.
	verify := func(string) bool { return true }
	if cfg, e := config.Load(); e == nil && cfg.Bootstrap.Present() {
		verify = cfg.Bootstrap.Equal
	}
	if *arm == "badcred" {
		verify = func(string) bool { return false }
	}

	server := grpc.NewServer()
	pb.RegisterWorkerControlServer(server, &fakeControl{
		say: say, arm: *arm, bootID: boot, instance: instance, releaseID: release,
		root: *cozyHome, verify: verify,
	})
	if err := server.Serve(ln); err != nil {
		say("serve ended: %v", err)
	}
	return 0
}
