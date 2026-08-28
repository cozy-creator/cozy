package main

import (
	"bufio"
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"syscall"
	"time"

	"github.com/cozy-creator/cozy-creator-v2/internal/config"
)

// The cl-006 driver's own HTTP client, and a SEPARATE `cozy up` process to point it at.
//
// The separate process is the point, not an accident of style. cl-001 could not arm its
// orchestrator-kill crash branch because the orchestrator ran inside the verification
// driver — killing it killed the observer. Here the driver is a CLIENT: it starts a real
// `cozy up`, drives the real HTTP API, and can `kill -9` the service and keep watching
// from the outside, which is exactly the seat a real client occupies.

// service is one detached `cozy up` this driver owns.
type liveService struct {
	root    string
	port    int
	addr    string
	token   string
	cmd     *exec.Cmd
	log     string
	http    *http.Client
	stopped bool
}

// startService launches a REAL `cozy up` in its own process group and waits for the API
// to answer. `fresh` wipes the root; a restart after a kill must NOT, because the whole
// point is that the authority and the worker journal survived.
func startService(root string, port int, fresh bool) *liveService {
	if fresh {
		must("clearing the service root", os.RemoveAll(root))
	}
	must("creating the service root", os.MkdirAll(root, 0o755))

	binary := flag("cozy", defaultCozy())
	abs, err := filepath.Abs(binary)
	must("resolving the cozy binary", err)
	if _, err := os.Stat(abs); err != nil {
		must("the cozy binary", fmt.Errorf("%s: %w (build it: go build -o %s ./cmd/cozy)",
			abs, err, filepath.Base(defaultCozy())))
	}

	logPath := filepath.Join(root, "driver-service.log")
	logFile, err := os.Create(logPath)
	must("service log", err)
	defer logFile.Close()

	// niceCmd is the DRIVER's resource discipline on a shared box, imposed on the launch
	// rather than baked into the product (and absent on Windows, which has no nice(1)).
	cmd := niceCmd(abs, "up", "--port", fmt.Sprint(port))
	// The child's environment comes through the PRODUCT's own allowlist, not through
	// os.Environ(): the driver has no business inventing a second child-env mechanism,
	// and the env fence says there is only one reader.
	must("setting COZY_HOME for the one env reader", os.Setenv("COZY_HOME", root))
	cfg, e := config.Load()
	must("config", errOf(e))
	cmd.Env = cfg.Child("COZY_HOME=" + root)
	cmd.Stdout, cmd.Stderr = logFile, logFile
	setProcessGroup(cmd)
	must("starting cozy up", cmd.Start())

	s := &liveService{
		root: root, port: port, addr: fmt.Sprintf("127.0.0.1:%d", port), cmd: cmd,
		log: logPath, http: &http.Client{Timeout: 5 * time.Minute},
	}
	onExit(s.stop)
	deadline := time.Now().Add(30 * time.Second)
	for time.Now().Before(deadline) {
		if s.alive() {
			data, err := os.ReadFile(filepath.Join(root, "client.cred"))
			if err == nil {
				s.token = strings.TrimSpace(string(data))
				return s
			}
		}
		time.Sleep(50 * time.Millisecond)
	}
	fmt.Println(tail(logPath, 30))
	must("cozy up", fmt.Errorf("the LocalService did not answer on %s in 30s", s.addr))
	return nil
}

// alive is an HTTP fact, not a lock fact: the driver is a CLIENT, and a client knows the
// service is up when the service answers.
func (s *liveService) alive() bool {
	res, err := (&http.Client{Timeout: time.Second}).Get("http://" + s.addr + "/healthz")
	if err != nil {
		return false
	}
	defer res.Body.Close()
	_, _ = io.Copy(io.Discard, res.Body)
	return res.StatusCode == http.StatusOK
}

// kill9 is the RECORD-OWNER-KILL arm's own instrument: SIGKILL to the owner's process
// group, so no owner shutdown path runs. Endpoint workers have separate process groups;
// restart must recognize and kill those exact orphans before replaying their journals.
func (s *liveService) kill9() {
	_ = killGroup(s.cmd.Process.Pid, syscall.SIGKILL)
	deadline := time.Now().Add(10 * time.Second)
	for time.Now().Before(deadline) && s.alive() {
		time.Sleep(50 * time.Millisecond)
	}
	go func() { _ = s.cmd.Wait() }()
}

func (s *liveService) stop() {
	if s.stopped || s.cmd == nil || s.cmd.Process == nil {
		return
	}
	s.stopped = true
	_ = killGroup(s.cmd.Process.Pid, syscall.SIGTERM)
	deadline := time.Now().Add(45 * time.Second)
	for time.Now().Before(deadline) && s.alive() {
		time.Sleep(100 * time.Millisecond)
	}
	_ = killGroup(s.cmd.Process.Pid, syscall.SIGKILL)
	go func() { _ = s.cmd.Wait() }()
}

// ---------------------------------------------------------------------- requesting

type reply struct {
	Status  int
	Header  http.Header
	Body    []byte
	Elapsed time.Duration
}

func (r reply) json() map[string]any {
	var out map[string]any
	_ = json.Unmarshal(r.Body, &out)
	return out
}

func (r reply) code() string {
	if e, ok := r.json()["error"].(map[string]any); ok {
		if code, ok := e["code"].(string); ok {
			return code
		}
	}
	return ""
}

func (r reply) brief() string {
	body := strings.TrimSpace(string(r.Body))
	if len(body) > 140 {
		body = body[:140] + "…"
	}
	return fmt.Sprintf("%d %s", r.Status, body)
}

// call issues one authenticated request. Headers may be overridden per call, which is
// how the refusal matrix presents a foreign Host or a foreign Origin.
func (s *liveService) call(method, path string, body any, headers ...string) reply {
	var reader io.Reader
	if body != nil {
		data, err := json.Marshal(body)
		must("rendering a request body", err)
		reader = bytes.NewReader(data)
	}
	req, err := http.NewRequest(method, "http://"+s.addr+path, reader)
	must("building a request", err)
	if body != nil {
		req.Header.Set("Content-Type", "application/json")
	}
	req.Header.Set("Authorization", "Bearer "+s.token)
	for i := 0; i+1 < len(headers); i += 2 {
		if strings.EqualFold(headers[i], "Host") {
			req.Host = headers[i+1]
			continue
		}
		if headers[i+1] == "" {
			req.Header.Del(headers[i])
			continue
		}
		req.Header.Set(headers[i], headers[i+1])
	}
	began := time.Now()
	res, err := s.http.Do(req)
	if err != nil {
		return reply{Status: 0, Body: []byte(err.Error()), Elapsed: time.Since(began)}
	}
	defer res.Body.Close()
	data, _ := io.ReadAll(res.Body)
	return reply{Status: res.StatusCode, Header: res.Header, Body: data, Elapsed: time.Since(began)}
}

// callAddr dials a SPECIFIC address with a specific Host header — the rebinding arm's
// instrument, and the only way to reach the IPv6 listener explicitly.
func (s *liveService) callAddr(addr, method, path, host string) reply {
	req, err := http.NewRequest(method, "http://"+addr+path, nil)
	must("building a request", err)
	req.Header.Set("Authorization", "Bearer "+s.token)
	if host != "" {
		req.Host = host
	}
	res, err := (&http.Client{Timeout: 10 * time.Second}).Do(req)
	if err != nil {
		return reply{Status: 0, Body: []byte(err.Error())}
	}
	defer res.Body.Close()
	data, _ := io.ReadAll(res.Body)
	return reply{Status: res.StatusCode, Header: res.Header, Body: data}
}

// ---------------------------------------------------------------------- the SSE client
//
// A REAL SSE client, modelled on cozy.art's `startSSE` (services/sse.ts): cursor resume,
// `id:` tracking, terminal-stop, and a reconnect that is never a verdict. The header form
// is the reference client's too — it uses EventSourcePolyfill precisely so the credential
// rides an Authorization header instead of a query parameter.

type sseEvent struct {
	Type      string         `json:"type"`
	RequestID string         `json:"request_id"`
	Attempt   uint64         `json:"attempt"`
	EventID   int64          `json:"event_id"`
	At        string         `json:"at"`
	Payload   map[string]any `json:"payload"`
	received  time.Time
}

type sseStream struct {
	events   []sseEvent
	cursor   int64
	stopped  bool // a terminal closed it
	opened   time.Time
	firstAt  time.Time
	retryMS  int
	closeErr error
}

// openSSE consumes one PER-REQUEST stream to its end: terminal-stop, or the deadline.
func (s *liveService) openSSE(path string, deadline time.Duration, headers ...string) *sseStream {
	return s.readSSE(path, deadline, true, headers...)
}

// openMultiplexed consumes the multiplexed stream, which NEVER terminal-stops: a terminal
// there ends one request, not the connection. Applying the per-request stop rule to it is
// precisely the client bug this split exists to make impossible.
func (s *liveService) openMultiplexed(path string, deadline time.Duration) *sseStream {
	return s.readSSE(path, deadline, false)
}

// openUntilLive reads a per-request stream only until the attempt is provably RUNNING —
// the first live progress frame — and returns with the cursor it holds. The crash arm
// needs this: consuming to the terminal would mean the attempt finished, and there would
// be nothing left to crash INTO.
func (s *liveService) openUntilLive(path string, deadline time.Duration) *sseStream {
	return s.readUntil(path, deadline, func(e sseEvent) bool {
		return e.Type == "request.progress" || terminalType(e.Type)
	})
}

func terminalType(t string) bool {
	return t == "request.completed" || t == "request.failed" || t == "request.canceled"
}

func (s *liveService) readSSE(path string, deadline time.Duration, stopOnTerminal bool, headers ...string) *sseStream {
	stop := func(sseEvent) bool { return false }
	if stopOnTerminal {
		stop = func(e sseEvent) bool { return terminalType(e.Type) }
	}
	return s.readUntil(path, deadline, stop, headers...)
}

func (s *liveService) readUntil(path string, deadline time.Duration, stop func(sseEvent) bool, headers ...string) *sseStream {
	st := &sseStream{opened: time.Now()}
	req, err := http.NewRequest("GET", "http://"+s.addr+path, nil)
	must("building the SSE request", err)
	req.Header.Set("Authorization", "Bearer "+s.token)
	req.Header.Set("Accept", "text/event-stream")
	for i := 0; i+1 < len(headers); i += 2 {
		if strings.EqualFold(headers[i], "Host") {
			req.Host = headers[i+1]
			continue
		}
		req.Header.Set(headers[i], headers[i+1])
	}
	client := &http.Client{Timeout: deadline}
	res, err := client.Do(req)
	if err != nil {
		st.closeErr = err
		return st
	}
	defer res.Body.Close()
	if res.StatusCode != http.StatusOK {
		data, _ := io.ReadAll(res.Body)
		st.closeErr = fmt.Errorf("%d %s", res.StatusCode, strings.TrimSpace(string(data)))
		return st
	}
	reader := bufio.NewReader(res.Body)
	var id, data string
	for {
		line, err := reader.ReadString('\n')
		if err != nil {
			st.closeErr = err
			return st
		}
		line = strings.TrimRight(line, "\r\n")
		switch {
		case strings.HasPrefix(line, "retry: "):
			fmt.Sscanf(line, "retry: %d", &st.retryMS)
		case strings.HasPrefix(line, "id: "):
			id = strings.TrimPrefix(line, "id: ")
		case strings.HasPrefix(line, "data: "):
			data = strings.TrimPrefix(line, "data: ")
		case line == "" && data != "":
			var ev sseEvent
			if json.Unmarshal([]byte(data), &ev) == nil {
				ev.received = time.Now()
				if st.firstAt.IsZero() {
					st.firstAt = ev.received
				}
				st.events = append(st.events, ev)
				if id != "" {
					fmt.Sscanf(id, "%d", &st.cursor)
				}
				// TERMINAL-STOP, the reference client's `shouldStop`: the stream ends
				// and is never reconnected. `stop` generalizes it so an arm that must
				// observe a RUNNING attempt can leave before it finishes.
				if stop(ev) {
					st.stopped = terminalType(ev.Type)
					return st
				}
			}
			id, data = "", ""
		}
	}
}

func (st *sseStream) types() []string {
	out := make([]string, 0, len(st.events))
	for _, e := range st.events {
		out = append(out, e.Type)
	}
	return out
}

func (st *sseStream) count(kind string) int {
	n := 0
	for _, e := range st.events {
		if e.Type == kind {
			n++
		}
	}
	return n
}

func (st *sseStream) find(kind string) *sseEvent {
	for i := range st.events {
		if st.events[i].Type == kind {
			return &st.events[i]
		}
	}
	return nil
}

func freePort(preferred int) int {
	for port := preferred; port < preferred+50; port++ {
		ln, err := net.Listen("tcp", fmt.Sprintf("127.0.0.1:%d", port)) //cozy:allow the DRIVER probes for a free port; the product binds through internal/api
		if err == nil {
			ln.Close()
			return port
		}
	}
	must("a free port", fmt.Errorf("nothing free near %d", preferred))
	return 0
}
