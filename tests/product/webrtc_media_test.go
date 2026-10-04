package producttest

import (
	"bytes"
	"context"
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net"
	"os"
	"os/exec"
	"path/filepath"
	"slices"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/cozy-creator/cozy/internal/capability"
	"github.com/cozy-creator/cozy/tests/product/webrtctest"
	"github.com/pion/stun/v4"
)

// mediaScriptPackage is a package whose job publishes, a step at a time, the bytes a test hands
// it: an append extends the output's parts (its revision extends the last one), a replace
// starts them again, an end returns. wheel is the Runtime it runs on.
func mediaScriptPackage(t *testing.T, wheel string) string {
	t.Helper()
	project := t.TempDir()
	must(t, os.WriteFile(filepath.Join(project, "pyproject.toml"), fmt.Appendf(nil, `[project]
name="media-script"
version="0.0.1"
requires-python=">=3.12,<3.13"
dependencies=["cozy-runtime>=%s"]
[tool.uv.sources]
cozy-runtime={path=%q}
[project.entry-points."cozy.application"]
default="script:app"
[build-system]
requires=["hatchling"]
build-backend="hatchling.build"
[tool.hatch.build.targets.wheel]
only-include=["script.py"]
`, runtimeFixtureVersion(t, wheel), wheel), 0o600))
	must(t, os.WriteFile(filepath.Join(project, "package.toml"), []byte("[application]\nobject=\"script:app\"\n"), 0o600))
	must(t, os.WriteFile(filepath.Join(project, "script.py"), []byte(`import json, shutil, time
from pathlib import Path
from typing import Annotated
import msgspec
from cozy_runtime.author import App, AssetBound, Context, FileAsset, Outputs
from cozy_runtime.author._executor_requests import PublishPart
app = App()
class Script(msgspec.Struct):
    gate: str
class Shown(msgspec.Struct):
    video: Annotated[FileAsset, AssetBound(media_types=("video/mp4",))]
@app.job(emits_media=True)
def script(ctx: Context, payload: Script, out: Outputs) -> Shown:
    """step-<k>.json: {op: append|replace|end, file, duration_us}. done-<k> once published."""
    gate, parts = Path(payload.gate), out._attempt.spool / "parts"
    parts.mkdir(exist_ok=True)
    held, shown, k = [], None, 0
    while True:
        k += 1
        step = gate / f"step-{k}.json"
        while not step.exists():
            ctx.raise_if_cancelled()
            time.sleep(0.02)
        step = json.loads(step.read_text())
        if step["op"] == "end":
            return Shown(video=shown)
        shutil.copyfile(step["file"], parts / f"{k}")
        held = (held if step["op"] == "append" else []) + [(f"{k}", step["duration_us"])]
        shown = out.save_bytes(b"".join((parts / name).read_bytes() for name, _ in held), media_type="video/mp4")
        # As a joined video's revisions are: its parts, each with the media time it adds.
        out._attempt.parts[shown.ref] = tuple(PublishPart(local=f"parts/{name}", duration_us=us) for name, us in held)
        out.publish("video", shown)
        (gate / f"done-{k}").touch()
`), 0o600))
	if out, err := exec.Command("/usr/bin/nice", "-n", "19", "uv", "lock", "--project", project).CombinedOutput(); err != nil {
		t.Fatalf("uv lock: %v %s", err, out)
	}
	return project
}

// mediaMachine is a test home's Rust machine, granted a WebRTC port, running scripted runs of
// mediaScriptPackage. A test names its runs by number; a link and a cap name each by its id,
// as the machine does.
type mediaMachine struct {
	*cozy1Machine
	runs map[uint64]*scriptedRun
}

type scriptedRun struct {
	id, gate string
	steps    int
}

func newMediaMachine(t *testing.T) *mediaMachine {
	if *machineHostBinary == "" || *privateScriptRuntimeWheel == "" {
		t.Skip("requires -machine-host=<cozy-machine> and -script-runtime-wheel")
	}
	wheel, err := filepath.Abs(*privateScriptRuntimeWheel)
	must(t, err)
	root, port := cozy1Root(t)
	if code, out := runCozy(t, root, "package", "install", mediaScriptPackage(t, wheel), "--editable"); code != 0 {
		t.Fatalf("package install [exit %d]\n%s", code, out)
	}
	if code, out := runCozy(t, root, "machine", "start"); code != 0 {
		t.Fatalf("machine start [exit %d]\n%s", code, out)
	}
	return &mediaMachine{cozy1Machine: newCozy1Machine(t, root, port), runs: map[uint64]*scriptedRun{}}
}

// Begin starts scripted run n, once, and waits for its machine to accept it.
func (m *mediaMachine) Begin(n uint64) *scriptedRun {
	m.t.Helper()
	if run := m.runs[n]; run != nil {
		return run
	}
	run := &scriptedRun{gate: m.t.TempDir()}
	code, out := runCozy(m.t, m.root, "run", "local/media-script/script", "gate="+run.gate, "--json")
	var started struct {
		Run string `json:"run"`
	}
	if code != 0 || json.Unmarshal([]byte(lastJSONLine(out)), &started) != nil || started.Run == "" {
		m.t.Fatalf("scripted run %d did not start [exit %d]\n%s", n, code, out)
	}
	landed(m.t, fmt.Sprintf("the machine to accept scripted run %d", n), func() bool {
		var state struct {
			ID     string `json:"request_id"`
			Status string `json:"status"`
		}
		_, shown := runCozy(m.t, m.root, "run", "show", started.Run, "--json")
		_ = json.Unmarshal([]byte(lastJSONLine(shown)), &state)
		run.id = state.ID
		return state.Status == "in_progress"
	})
	m.runs[n] = run
	return run
}

// step hands run n its next step and waits until the run has taken it.
func (m *mediaMachine) step(n uint64, op string, data []byte, duration uint64) {
	m.t.Helper()
	run := m.Begin(n)
	run.steps++
	k := run.steps
	step := map[string]any{"op": op, "duration_us": duration}
	if data != nil {
		step["file"] = filepath.Join(run.gate, fmt.Sprintf("data-%d", k))
		must(m.t, os.WriteFile(step["file"].(string), data, 0o600))
	}
	raw, _ := json.Marshal(step)
	staged := filepath.Join(run.gate, fmt.Sprintf(".step-%d", k))
	must(m.t, os.WriteFile(staged, raw, 0o600))
	must(m.t, os.Rename(staged, filepath.Join(run.gate, fmt.Sprintf("step-%d.json", k))))
	if op != "end" {
		landed(m.t, fmt.Sprintf("scripted run %d to publish step %d", n, k), func() bool {
			_, err := os.Stat(filepath.Join(run.gate, fmt.Sprintf("done-%d", k)))
			return err == nil
		})
	}
}

// Append publishes a revision of run n's video that extends the last one by data.
func (m *mediaMachine) Append(n uint64, data []byte, duration uint64) {
	m.step(n, "append", data, duration)
}

// Replace publishes a revision of run n's video that is data alone.
func (m *mediaMachine) Replace(n uint64, data []byte, duration uint64) {
	m.step(n, "replace", data, duration)
}

// End ends run n: completed returns its result, canceled cancels it. It waits for the run to settle.
func (m *mediaMachine) End(n uint64, status string) {
	m.t.Helper()
	run := m.Begin(n)
	if status == "canceled" {
		if code, out := runCozy(m.t, m.root, "run", "cancel", run.id, "--json"); code != 0 {
			m.t.Fatalf("cancel scripted run %d [exit %d]\n%s", n, code, out)
		}
	} else {
		m.step(n, "end", nil, 0)
	}
	landed(m.t, fmt.Sprintf("scripted run %d to end %s", n, status), func() bool {
		var state struct {
			Status string `json:"status"`
		}
		_, shown := runCozy(m.t, m.root, "run", "show", run.id, "--json")
		_ = json.Unmarshal([]byte(lastJSONLine(shown)), &state)
		return state.Status == status || status == "completed" && state.Status == "succeeded"
	})
}

// mint signs g with the owner key; a run number names that scripted run.
func (m *mediaMachine) mint(g capability.Grant) string {
	if n, err := strconv.ParseUint(g.Run, 10, 64); err == nil {
		g.Run = m.Begin(n).id
	}
	return m.cozy1Machine.mint(g)
}

// Every test runs the real Rust machine and a real pion ICE-TCP client over real TCP; outputs
// are what a real run published.

type mediaHarness struct {
	t   *testing.T
	ctx context.Context
	m   *mediaMachine
}

func newMediaHarness(t *testing.T) *mediaHarness {
	m := newMediaMachine(t)
	m.Begin(7)
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
	t.Cleanup(cancel)
	return &mediaHarness{t: t, ctx: ctx, m: m}
}

// run is scripted run n's id, which every request names.
func (h *mediaHarness) run(n uint64) string { return h.m.Begin(n).id }

// grant mints a capability for run 7 of this machine, valid for an hour unless edit says otherwise.
func (h *mediaHarness) grant(edit func(*capability.Grant)) string {
	g := capability.Grant{Run: "7", Expires: time.Now().Add(time.Hour).Unix()}
	if edit != nil {
		edit(&g)
	}
	return h.m.mint(g)
}

func (h *mediaHarness) dial() *webrtctest.Client {
	c, err := webrtctest.Dial(h.ctx, h.m.Addr, h.m.Fingerprint, webrtctest.Options{})
	if err != nil {
		h.t.Fatal(err)
	}
	h.t.Cleanup(func() { c.Close() })
	return c
}

// open is a welcomed session with credit.
func (h *mediaHarness) open(credit uint64) *webrtctest.Client {
	c := h.dial()
	h.send(c, map[string]any{"t": "hello", "v": 1, "cap": h.grant(nil)})
	h.expect(c, "welcome")
	h.send(c, map[string]any{"t": "credit", "bytes": credit})
	return c
}

func (h *mediaHarness) send(c *webrtctest.Client, record any) {
	if err := c.Send(record); err != nil {
		h.t.Fatal(err)
	}
}

func (h *mediaHarness) expect(c *webrtctest.Client, t string) webrtctest.Message {
	h.t.Helper()
	m, _ := c.Recv(h.ctx)
	if m.T != t {
		h.t.Fatalf("expected %s, got %s %s", t, m.T, m.Raw)
	}
	return m
}

// body reads one stream's binary messages from its open to its end, checking every offset.
func (h *mediaHarness) body(c *webrtctest.Client) ([]byte, webrtctest.Message) {
	h.t.Helper()
	open := h.expect(c, "open")
	var got []byte
	for {
		m, _ := c.Recv(h.ctx)
		switch m.T {
		case "data":
			if m.Stream != open.Stream || m.Offset != open.Offset+uint64(len(got)) {
				h.t.Fatalf("data at %d of stream %d; expected %d of %d", m.Offset, m.Stream, open.Offset+uint64(len(got)), open.Stream)
			}
			got = append(got, m.Data...)
		case "end":
			if uint64(len(got)) != open.Length {
				h.t.Fatalf("%d bytes before end; open announced %d", len(got), open.Length)
			}
			return got, m
		default:
			h.t.Fatalf("unexpected %s", m.Raw)
		}
	}
}

func mediaSegment(n int, fill byte) []byte { return bytes.Repeat([]byte{fill}, n) }

func TestWebRTCGetServesARangeOfTheCurrentBytes(t *testing.T) {
	h := newMediaHarness(t)
	seg := [][]byte{mediaSegment(100_000, 1), mediaSegment(150_000, 2), mediaSegment(70_000, 3)}
	h.m.Append(7, seg[0], 1_000_000)
	h.m.Append(7, seg[1], 1_000_000)
	h.m.Append(7, seg[2], 1_000_000)
	h.m.End(7, "completed")
	c := h.open(1 << 20)

	h.send(c, map[string]any{"t": "get", "id": 1, "run": h.run(7), "output": "video", "offset": 100_000, "length": 150_000})
	got, end := h.body(c)
	if !bytes.Equal(got, seg[1]) {
		t.Fatal("the middle mediaSegment differs")
	}
	whole := bytes.Join(seg, nil)
	if end.Length != uint64(len(whole)) || end.SHA256 != digestOf(whole) {
		t.Fatalf("end %s", end.Raw)
	}

	h.send(c, map[string]any{"t": "get", "id": 2, "run": h.run(7), "output": "video", "offset": 0, "etag": "r2"})
	if m := h.expect(c, "error"); m.Code != "changed" {
		t.Fatalf("a stale etag got %s", m.Raw)
	}
	h.send(c, map[string]any{"t": "get", "id": 3, "run": h.run(7), "output": "video", "offset": 0, "etag": "r3"})
	if got, _ := h.body(c); !bytes.Equal(got, whole) {
		t.Fatal("the whole output differs")
	}
}

// mediaFollower is a client's copy of a followed output and its cursor (seq, len(got)).
type mediaFollower struct {
	got     []byte
	seq     uint64 // the last entry whose bytes it holds in full
	entries []webrtctest.Message
	resets  int
	end     *webrtctest.Message
}

// until applies messages until done says so, checking there is never a gap or a duplicate.
func (h *mediaHarness) until(c *webrtctest.Client, f *mediaFollower, done func() bool) {
	h.t.Helper()
	for !done() {
		m, ok := c.Recv(h.ctx)
		if !ok {
			h.t.Fatalf("the session ended: %s", m.T)
		}
		switch m.T {
		case "entry":
			f.entries = append(f.entries, m)
		case "reset": // entries are eager: those from the reset on are the replacement's
			f.entries = slices.DeleteFunc(f.entries, func(e webrtctest.Message) bool { return e.Seq < m.Seq })
			f.got, f.seq = nil, 0
			f.resets++
		case "open", "data":
			if m.Offset != uint64(len(f.got)) {
				h.t.Fatalf("%s at %d while holding %d", m.T, m.Offset, len(f.got))
			}
			f.got = append(f.got, m.Data...)
		case "end":
			f.end = &m
		default:
			h.t.Fatalf("unexpected %s", m.Raw)
		}
		for _, e := range f.entries {
			if e.Length <= uint64(len(f.got)) {
				f.seq = e.Seq
			}
		}
	}
}

func (h *mediaHarness) follow(c *webrtctest.Client, after, offset uint64) {
	h.send(c, map[string]any{"t": "follow", "id": "v", "run": h.run(7), "output": "video", "after": after, "offset": offset})
}

func TestWebRTCFollowStreamsEachAppendAsItLands(t *testing.T) {
	h := newMediaHarness(t)
	c := h.open(64 << 20)
	h.follow(c, 0, 0)
	var f mediaFollower
	var whole []byte
	for k := range 3 {
		seg := mediaSegment(200_000+k*50_000, byte(k+1))
		whole = append(whole, seg...)
		h.m.Append(7, seg, 1_000_000)
		h.until(c, &f, func() bool {
			return len(f.got) == len(whole) && len(f.entries) == k+1 && f.seq == f.entries[k].Seq
		})
	}
	h.m.End(7, "completed")
	h.until(c, &f, func() bool { return f.end != nil })
	if !bytes.Equal(f.got, whole) || f.end.Status != "completed" || f.end.SHA256 != digestOf(whole) || f.end.Length != uint64(len(whole)) {
		t.Fatalf("end %s after %d bytes", f.end.Raw, len(f.got))
	}
	if len(f.entries) != 3 || *f.entries[2].AppendedFrom != uint64(len(whole)-300_000) || f.entries[2].Rev != 3 ||
		f.entries[2].DurationUS != 3_000_000 {
		t.Fatalf("entries %+v", f.entries)
	}
}

func TestWebRTCFollowResumesAfterTheConnectionDies(t *testing.T) {
	h := newMediaHarness(t)
	r := newMediaRelay(t, h.m.Addr, 0)
	seg := [][]byte{mediaSegment(300_000, 1), mediaSegment(400_000, 2), mediaSegment(250_000, 3)}
	c, err := webrtctest.Dial(h.ctx, r.addr(), h.m.Fingerprint, webrtctest.Options{})
	if err != nil {
		t.Fatal(err)
	}
	defer c.Close()
	h.send(c, map[string]any{"t": "hello", "v": 1, "cap": h.grant(nil)})
	h.expect(c, "welcome")
	held := uint64(len(seg[0]) + len(seg[1])/2) // credit ends mid-segment
	h.send(c, map[string]any{"t": "credit", "bytes": held})
	h.follow(c, 0, 0)
	var f mediaFollower
	h.m.Append(7, seg[0], 1_000_000)
	h.m.Append(7, seg[1], 1_000_000)
	h.until(c, &f, func() bool { return uint64(len(f.got)) == held })
	r.kill()

	// A browser reconnects TCP with the dropped session's ufrag: closed unanswered, so its
	// PeerConnection fails and it starts afresh. Once the machine has seen the drop.
	for !ufragRefused(t, h, c.Ufrag) {
		time.Sleep(10 * time.Millisecond)
	}

	h.m.Append(7, seg[2], 1_000_000)
	h.m.End(7, "completed")
	resumed := h.open(64 << 20)
	if len(f.entries) == 0 || f.seq != f.entries[0].Seq {
		t.Fatalf("the cursor is at entry %d", f.seq)
	}
	h.follow(resumed, f.seq, uint64(len(f.got)))
	h.until(resumed, &f, func() bool { return f.end != nil })
	whole := bytes.Join(seg, nil)
	if !bytes.Equal(f.got, whole) || f.end.SHA256 != digestOf(whole) || f.resets != 0 {
		t.Fatalf("resumed to %d bytes, %d resets; end %s", len(f.got), f.resets, f.end.Raw)
	}
}

func TestWebRTCFollowResetsWhenTheOutputIsReplaced(t *testing.T) {
	h := newMediaHarness(t)
	c := h.open(64 << 20)
	h.follow(c, 0, 0)
	var f mediaFollower
	h.m.Append(7, mediaSegment(100_000, 1), 1_000_000)
	h.until(c, &f, func() bool { return len(f.got) == 100_000 })
	cursor := f
	replaced := mediaSegment(80_000, 9)
	h.m.Replace(7, replaced, 1_000_000)
	h.until(c, &f, func() bool { return f.resets == 1 && bytes.Equal(f.got, replaced) })

	// A client that held the first revision resumes into the replacement: reset, then all of it.
	late := h.open(64 << 20)
	h.follow(late, cursor.seq, uint64(len(cursor.got)))
	h.until(late, &cursor, func() bool { return cursor.resets == 1 && bytes.Equal(cursor.got, replaced) })

	h.m.End(7, "canceled")
	h.until(c, &f, func() bool { return f.end != nil })
	if f.end.Status != "canceled" || f.end.Length != uint64(len(replaced)) || f.end.SHA256 != digestOf(replaced) {
		t.Fatalf("end %s", f.end.Raw)
	}
}

// A resumed client chooses a new ICE credential after the previous session ended.
func ufragRefused(t *testing.T, h *mediaHarness, ufrag string) bool {
	conn, err := net.Dial("tcp", h.m.Addr.String())
	must(t, err)
	defer conn.Close()
	message, err := stun.Build(stun.TransactionID, stun.BindingRequest, stun.NewUsername(ufrag+":browser"), stun.NewShortTermIntegrity(ufrag), stun.Fingerprint)
	must(t, err)
	_, err = conn.Write(append(binary.BigEndian.AppendUint16(nil, uint16(len(message.Raw))), message.Raw...))
	must(t, err)
	conn.SetReadDeadline(time.Now().Add(10 * time.Second))
	n, err := conn.Read(make([]byte, 1))
	if n == 0 && !errors.Is(err, io.EOF) && (err == nil || !strings.Contains(err.Error(), "connection reset")) {
		t.Fatalf("reused credential neither answered nor closed: %v", err)
	}
	return n == 0
}
