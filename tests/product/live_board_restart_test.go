package producttest

import (
	"bytes"
	"fmt"
	"net"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"sync"
	"syscall"
	"testing"

	localapi "github.com/cozy-creator/cozy/internal/client"
	"github.com/cozy-creator/cozy/internal/config"
	"github.com/cozy-creator/cozy/internal/daemon"
)

type screenBuffer struct {
	mu  sync.Mutex
	buf bytes.Buffer
}

func (s *screenBuffer) Write(p []byte) (int, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.buf.Write(p)
}

func (s *screenBuffer) String() string {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.buf.String()
}

// An open `cozy run list` board keeps drawing across a daemon restart. A restart too quick
// for a client to see the daemon gone still mints a new credential, which it adopts.
func TestRunListBoardFollowsAQuickDaemonRestart(t *testing.T) {
	root := t.TempDir()
	// The owner's port: each launch binds it again, so a restart answers where the last did.
	free, err := net.Listen("tcp", "127.0.0.1:0")
	must(t, err)
	port := free.Addr().(*net.TCPAddr).Port
	must(t, free.Close())
	must(t, os.WriteFile(filepath.Join(root, config.FileName), []byte(fmt.Sprintf("port: %d\n", port)), 0o600))
	first := startDaemonProcess(t, root)
	board := exec.Command("script", "-qfec", "stty cols 160 rows 40; exec "+cozyBin+" run list", "/dev/null")
	board.Env = childEnv(t, root)
	screen := &screenBuffer{}
	board.Stdout, board.Stderr = screen, screen
	keys, err := board.StdinPipe()
	must(t, err)
	setProcessGroup(board)
	must(t, board.Start())
	t.Cleanup(func() { _ = killGroup(board.Process.Pid) })
	exited := make(chan error, 1)
	go func() { exited <- board.Wait() }()
	frames := func() int { return strings.Count(screen.String(), "\x1b[H\x1b[J") }
	landed(t, "the board's first frames", func() bool { return frames() >= 2 })

	must(t, syscall.Kill(-board.Process.Pid, syscall.SIGSTOP))
	if code, out := runCozy(t, root, "down", "--json"); code != 0 {
		t.Fatalf("stopping the daemon [%d]: %s", code, out)
	}
	if second := startDaemonProcess(t, root); second.token == first.token {
		t.Fatal("the restarted daemon kept its predecessor's credential")
	}
	before := frames()
	must(t, syscall.Kill(-board.Process.Pid, syscall.SIGCONT))
	landed(t, "frames drawn from the restarted daemon", func() bool {
		select {
		case err := <-exited:
			t.Fatalf("the board exited across the restart (%v): %q", err, screen.String())
		default:
		}
		return frames() >= before+3
	})
	_, err = keys.Write([]byte("q"))
	must(t, err)
	if err := <-exited; err != nil {
		t.Fatalf("the board did not quit cleanly (%v): %q", err, screen.String())
	}
	if strings.Contains(screen.String(), "client credential") {
		t.Fatalf("the board showed a credential refusal: %q", screen.String())
	}

	cfg, e := config.Load()
	fatal(t, e)
	// The process config is frozen by whichever test loaded it first.
	cfg.Home, cfg.Port = root, port
	following, e := localapi.Open(cfg, daemon.Probe(cfg))
	fatal(t, e)
	following = following.Following(func() {})
	if code, out := runCozy(t, root, "down", "--json"); code != 0 {
		t.Fatalf("stopping the daemon [%d]: %s", code, out)
	}
	startDaemonProcess(t, root)
	// Its first exchange reaches the new daemon directly: no failed connection precedes it.
	http.DefaultTransport.(*http.Transport).CloseIdleConnections()
	if _, e := following.RequestPage(t.Context(), "", "", 1, 0); e != nil {
		t.Fatalf("a following client refused by the restarted daemon: %s", e.Message)
	}
}
