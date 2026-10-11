package producttest

import (
	"encoding/json"
	"fmt"
	"net"
	"os"
	"os/exec"
	"path/filepath"
	"sync"
	"testing"
	"time"
)

const outputRangesPackage = `"""One entrypoint that returns an output of the size asked for."""
import os
from typing import Annotated

import msgspec

from cozy_runtime.author import App, AssetBound, Context, FileAsset, Outputs

app = App()


class Request(msgspec.Struct):
    mib: int


class Response(msgspec.Struct):
    blob: Annotated[FileAsset, AssetBound(max_bytes=256 << 20, media_types=("application/octet-stream",))]


@app.entrypoint
def blob(payload: Request, ctx: Context, out: Outputs) -> Response:
    return Response(out.save_bytes(os.urandom(payload.mib << 20), media_type="application/octet-stream"))
`

const outputRangesProject = `[build-system]
requires = ["uv_build>=0.12.7,<0.13"]
build-backend = "uv_build"

[project]
name = "cozy-output-ranges"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = ["cozy-runtime>=0.18.67", "msgspec>=0.19,<1"]

[project.entry-points."cozy.application"]
default = "output_ranges:app"

[tool.uv.build-backend]
module-name = "output_ranges"
module-root = ""
`

// farLink stands between the CLI and a machine as a long path does: every byte arrives `delay`
// late each way, and one connection carries at most `rate` bytes a second from the machine,
// as one TCP connection's window bounds what it carries however wide the link.
type farLink struct {
	listener net.Listener
	delay    time.Duration
	rate     int64
	size     int64

	mu      sync.Mutex
	moved   []int64   // bytes from the machine, per connection
	total   int64     // bytes from the machine in all
	began   time.Time // when total passed its first MiB
	crossed time.Time // when total reached size
}

func newFarLink(t *testing.T, machine string, delay time.Duration, rate, size int64) *farLink {
	listener, err := net.Listen("tcp", "127.0.0.1:0")
	must(t, err)
	link := &farLink{listener: listener, delay: delay, rate: rate, size: size}
	t.Cleanup(func() { listener.Close() })
	go func() {
		for {
			near, err := listener.Accept()
			if err != nil {
				return
			}
			far, err := net.Dial("tcp", machine)
			if err != nil {
				near.Close()
				continue
			}
			link.mu.Lock()
			connection := len(link.moved)
			link.moved = append(link.moved, 0)
			link.mu.Unlock()
			go link.carry(far, near, -1)
			go link.carry(near, far, connection)
		}
	}()
	return link
}

// carry moves bytes to `to` late and, from the machine (connection >= 0), no faster than rate.
// It holds what a socket buffer would, so a sender that outruns the link waits.
func (l *farLink) carry(to, from net.Conn, connection int) {
	type held struct {
		data []byte
		due  time.Time
	}
	line := make(chan held, 32)
	go func() {
		defer close(line)
		for {
			buffer := make([]byte, 32<<10)
			n, err := from.Read(buffer)
			if n > 0 {
				line <- held{buffer[:n], time.Now().Add(l.delay)}
			}
			if err != nil {
				return
			}
		}
	}()
	free := time.Now()
	for part := range line {
		if connection >= 0 {
			if free.Before(part.due) {
				free = part.due
			}
			part.due, free = free, free.Add(time.Duration(len(part.data))*time.Second/time.Duration(l.rate))
		}
		time.Sleep(time.Until(part.due))
		if _, err := to.Write(part.data); err != nil {
			break
		}
		if connection >= 0 {
			l.mu.Lock()
			l.moved[connection] += int64(len(part.data))
			if l.total += int64(len(part.data)); l.total >= 1<<20 && l.began.IsZero() {
				l.began = time.Now()
			}
			if l.total >= l.size && l.crossed.IsZero() {
				l.crossed = time.Now()
			}
			l.mu.Unlock()
		}
	}
	to.Close()
	from.Close()
	for range line {
	}
}

// A big output comes back over a far link as ranges side by side on several connections: its
// bytes cross several times faster than one connection can carry them, and it lands whole.
func TestABigOutputCrossesAFarLinkOnSeveralConnections(t *testing.T) {
	if *machineHostBinary == "" {
		t.Skip("requires -machine-host=<cozy-machine>")
	}
	root, err := os.MkdirTemp(os.TempDir(), "czr")
	must(t, err)
	provisionMachine(t, root)
	t.Cleanup(func() {
		_, _ = runCozy(t, root, "machine", "stop")
		_, _ = runCozy(t, root, "down")
		if t.Failed() {
			t.Logf("evidence retained at %s\nmachine log tail:\n%s", root, tail(filepath.Join(root, "machine", "host.log")))
		} else {
			_ = removeAllForce(root)
		}
	})
	project := filepath.Join(t.TempDir(), "output_ranges")
	must(t, os.MkdirAll(filepath.Join(project, "output_ranges"), 0o755))
	must(t, os.WriteFile(filepath.Join(project, "pyproject.toml"), []byte(outputRangesProject), 0o644))
	must(t, os.WriteFile(filepath.Join(project, "package.toml"), []byte("[application]\nobject = \"output_ranges:app\"\n"), 0o644))
	must(t, os.WriteFile(filepath.Join(project, "output_ranges", "__init__.py"), []byte(outputRangesPackage), 0o644))
	if out, err := exec.Command("uv", "lock", "--directory", project).CombinedOutput(); err != nil {
		t.Fatalf("uv lock: %v\n%s", err, out)
	}
	if code, out := runCozy(t, root, "package", "install", project); code != 0 {
		t.Fatalf("package install [exit %d]\n%s", code, out)
	}
	if code, out := runCozy(t, root, "machine", "start"); code != 0 {
		t.Fatalf("machine start [exit %d]\n%s", code, out)
	}
	var agent struct {
		WorkerID   string `json:"worker_id"`
		WorkerPort int    `json:"worker_port"`
	}
	raw, err := os.ReadFile(filepath.Join(root, "machine", "agent.json"))
	must(t, err)
	must(t, json.Unmarshal(raw, &agent))
	leaf, err := os.ReadFile(filepath.Join(root, "machine", "leaf.pem"))
	must(t, err)

	const size, rate = 64 << 20, 4 << 20
	link := newFarLink(t, fmt.Sprintf("127.0.0.1:%d", agent.WorkerPort), 40*time.Millisecond, rate, size)
	endpoint := filepath.Join(root, "endpoint.json")
	document, _ := json.Marshal(map[string]string{"format": "cozy.machine.endpoint/1", "address": link.listener.Addr().String(),
		"worker_id": agent.WorkerID, "worker_boot_id": "boot", "tls_certificate_pem": string(leaf), "execution_workspace_id": "workspace"})
	must(t, os.WriteFile(endpoint, document, 0o600))
	in := filepath.Join(root, "req.json")
	must(t, os.WriteFile(in, []byte(`{"mib":64}`), 0o600))
	out := filepath.Join(root, "got")
	if code, said, logged := runCozyStreamsWith(t, root, nil, "run", "local/cozy-output-ranges/blob", "--machine-endpoint-file", endpoint,
		"--input", in, "--out", out, "--await", "--json"); code != 0 {
		t.Fatalf("run [exit %d]\n%s\n%s", code, said, logged)
	}
	files, _ := filepath.Glob(filepath.Join(out, "*"))
	if len(files) != 1 {
		t.Fatalf("the run left %v", files)
	}
	if info, err := os.Stat(files[0]); err != nil || info.Size() != size {
		t.Fatalf("the output is not its %d bytes: %v", size, err)
	}
	link.mu.Lock()
	defer link.mu.Unlock()
	lanes := 0
	for _, moved := range link.moved {
		if moved >= 1<<20 {
			lanes++
		}
	}
	crossed := float64(size) / link.crossed.Sub(link.began).Seconds()
	t.Logf("%d MiB crossed at %.1f MiB/s on %d connections (one carries %d MiB/s); the link moved %d MiB",
		size>>20, crossed/(1<<20), lanes, rate>>20, link.total>>20)
	if lanes < 4 || crossed < 3*rate {
		t.Fatalf("the output crossed at %.1f MiB/s on %d connections; one connection carries %d MiB/s", crossed/(1<<20), lanes, rate>>20)
	}
}
