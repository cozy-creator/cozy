//go:build linux

package producttest

import (
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"syscall"
	"testing"
	"time"

	"github.com/cozy-creator/cozy/internal/records"
	"golang.org/x/sys/unix"
)

const strayTerminalText = "ZZZ123abcDEFzzz"

type inputPTY struct {
	master   *os.File
	command  *exec.Cmd
	output   *renderBuffer
	original unix.Termios
	readDone chan struct{}
}

func liveInputPTY(t *testing.T, root string, args ...string) *inputPTY {
	t.Helper()
	p := &inputPTY{output: &renderBuffer{}, readDone: make(chan struct{})}
	p.master, p.command = startPTYSetup(t, root, 18, 90, func(master *os.File) {
		state, err := unix.IoctlGetTermios(int(master.Fd()), unix.TCGETS)
		must(t, err)
		// Restoration must preserve non-default settings, not merely apply stty sane.
		state.Lflag ^= unix.ECHOK
		state.Oflag &^= unix.ONLCR
		must(t, unix.IoctlSetTermios(int(master.Fd()), unix.TCSETS, state))
		p.original = *state
	}, args...)
	go func() { _, _ = io.Copy(p.output, p.master); close(p.readDone) }()
	return p
}

func (p *inputPTY) typeWhileLive(t *testing.T) {
	t.Helper()
	waitUntil(t, "terminal input is owned without echo", func() bool {
		state, err := unix.IoctlGetTermios(int(p.master.Fd()), unix.TCGETS)
		return err == nil && state.Lflag&(unix.ECHO|unix.ICANON) == 0
	})
	_, err := p.master.Write([]byte(strayTerminalText))
	must(t, err)
	time.Sleep(100 * time.Millisecond)
	if strings.Contains(p.output.String(), strayTerminalText) {
		t.Fatalf("typed characters were echoed over live output: %q", p.output.String())
	}
}

func (p *inputPTY) finish(t *testing.T, wantCode int) string {
	t.Helper()
	_ = p.command.Wait()
	<-p.readDone
	state, err := unix.IoctlGetTermios(int(p.master.Fd()), unix.TCGETS)
	must(t, err)
	if *state != p.original {
		t.Fatalf("terminal settings changed after exit: before=%+v after=%+v", p.original, *state)
	}
	text := p.output.String()
	if p.command.ProcessState.ExitCode() != wantCode || strings.Contains(text, strayTerminalText) {
		t.Fatalf("terminal exit/echo: code=%d wanted=%d output=%q", p.command.ProcessState.ExitCode(), wantCode, text)
	}
	return text
}

func TestLiveTablesAndWatchOwnTerminalInput(t *testing.T) {
	root, _, _ := rentalEndRoot(t, "live-input")
	startDaemonProcess(t, root)
	store, problem := records.Open(filepath.Join(root, "creator.sqlite"))
	fatal(t, problem)
	defer store.Close()
	for _, kind := range []string{"serving", "job"} {
		id := "req-input-" + kind
		if kind == "job" {
			id = "job-input-job"
		}
		_, _, problem := store.Submit(records.Request{ID: id, IdemKey: id, Kind: kind,
			BodyDigest: "sha256:" + strings.Repeat("ab", 32), Package: "fixture/input", Entrypoint: "generate", Payload: []byte("{}")})
		fatal(t, problem)
		fatal(t, store.AppendEvent(id, "run.created", 0, map[string]any{"package": "fixture/input"}))
	}
	for _, test := range []struct {
		name string
		args []string
		key  string
	}{
		{"run table", []string{"run", "list"}, "q"},
		{"rental table", []string{"rental", "list"}, "\x1b"},
		{"serving watch", []string{"run", "watch", "req-input-serving"}, "\x03"},
		{"job watch", []string{"run", "watch", "job-input-job"}, "Q"},
		{"signal", []string{"run", "watch", "req-input-serving"}, ""},
	} {
		t.Run(test.name, func(t *testing.T) {
			p := liveInputPTY(t, root, test.args...)
			p.typeWhileLive(t)
			if test.key == "" {
				must(t, p.command.Process.Signal(syscall.SIGTERM))
			} else {
				_, err := p.master.Write([]byte(test.key))
				must(t, err)
			}
			p.finish(t, 0)
		})
	}
	for _, id := range []string{"req-input-serving", "job-input-job"} {
		row, problem := store.RequestRow(id)
		fatal(t, problem)
		if row == nil || row.State != "submitted" {
			t.Fatalf("watcher changed durable work %s: %+v", id, row)
		}
		events, problem := store.EventsAfter(id, 0, 256)
		fatal(t, problem)
		for _, event := range events {
			if event.Type == "request.cancel_requested" || event.Type == "run.canceled" {
				t.Fatalf("watcher signal created a cancellation event for %s: %+v", id, event)
			}
		}
	}
	t.Run("json does not borrow input", func(t *testing.T) {
		p := liveInputPTY(t, root, "run", "watch", "req-input-serving", "--json")
		waitUntil(t, "JSON watcher attached", func() bool { return strings.Contains(p.output.String(), "run.created") })
		state, err := unix.IoctlGetTermios(int(p.master.Fd()), unix.TCGETS)
		must(t, err)
		if *state != p.original {
			t.Fatal("JSON watcher changed terminal mode")
		}
		must(t, p.command.Process.Signal(syscall.SIGTERM))
		p.finish(t, 0)
	})
}

func TestAwaitInputDetachesWithoutCancellingAcceptedWork(t *testing.T) {
	root, err := os.MkdirTemp("", "cozy-input-")
	must(t, err)
	t.Cleanup(func() { _, _ = runCozy(t, root, "down"); _ = os.RemoveAll(root) })
	if code, out := runCozy(t, root, "package", "install", weightlessProject(t), "--editable"); code != 0 {
		t.Fatalf("weightless install: %d %s", code, out)
	}
	if code, out := runCozy(t, root, "run", localWeightlessRef+"/tile", "size=8", "--await", "--json"); code != 0 {
		t.Fatalf("weightless warmup: %d %s\n%s", code, out, productWorkerLogs(root))
	}
	before := invocationIDSet(t, root)
	p := liveInputPTY(t, root, "run", localWeightlessRef+"/tile", "size=8", "delay_ms=4000", "--await")
	p.typeWhileLive(t)
	run := awaitNewInvocation(t, root, before, p.output, p.output)
	awaitInvocationStatus(t, root, run.ID, "in_progress")
	_, err = p.master.Write([]byte("q"))
	must(t, err)
	if text := p.finish(t, 0); !strings.Contains(text, "detached") {
		t.Fatalf("q did not detach the watcher: %s", text)
	}
	awaitInvocationStatus(t, root, run.ID, "completed")
	// Normal completion restores the same deliberately non-default termios flags.
	p = liveInputPTY(t, root, "run", localWeightlessRef+"/tile", "size=8", "delay_ms=600", "--await")
	p.typeWhileLive(t)
	p.finish(t, 0)
}

func TestLiveRentalAcquisitionRestoresInput(t *testing.T) {
	for _, result := range []string{"detach", "error"} {
		t.Run(result, func(t *testing.T) {
			root, _, stand := rentalEndRoot(t, "rent-input")
			stand.setSKUs(map[string]any{"name": "h100-nvl", "accelerator_model": "NVIDIA H100 NVL", "accelerator_count": 1,
				"base_worker_profile": "proof", "compute_capability": "9.0", "vram_gb": 94, "minimum_ram_per_gpu_gb": 64, "price_usd_micros_per_hour": 3190000})
			const id = "pr-input-watch"
			stand.mu.Lock()
			stand.rent = func(request map[string]any) map[string]any {
				return map[string]any{"rental_id": id, "name": request["name"], "state": "pending_acquisition",
					"requested_accelerator_model": "NVIDIA H100 NVL", "accelerator_count": 1, "hourly_rate_usd_micros": 3190000}
			}
			stand.mu.Unlock()
			p := liveInputPTY(t, root, "rental", "new", "h100-nvl")
			p.typeWhileLive(t)
			waitUntil(t, "accepted rental displayed", func() bool { return strings.Contains(p.output.String(), "acquiring on") })
			want := 0
			if result == "detach" {
				_, err := p.master.Write([]byte("q"))
				must(t, err)
			} else {
				stand.setState(id, "failed", "intentional fixture failure")
				want = 1
			}
			p.finish(t, want)
			if result == "detach" && stand.releases(id) != 0 {
				t.Fatal("quitting acquisition released a paid rental")
			}
		})
	}
}
