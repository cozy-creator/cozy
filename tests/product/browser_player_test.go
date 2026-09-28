package producttest

import (
	"bytes"
	"crypto/ed25519"
	"crypto/rand"
	"flag"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/cozy-creator/cozy/internal/capability"
	"github.com/cozy-creator/cozy/tests/product/webrtctest"
	"github.com/playwright-community/playwright-go"
)

var playerBrowsers = flag.String("player-browsers", "", "browsers that prove web/player, comma-separated: chrome (the installed Google Chrome), firefox, webkit (Playwright's builds)")

// The player page plays a run's output straight from a machine's WebRTC listener in real
// browsers: live as the output grows, resumed after a reload and after a lost connection,
// seeked by get once finished, and with one clear message for each way a link can fail. The
// listener is the machine's own, serving real files; the film is real H.264 and AAC.
func TestPlayerPagePlaysAGrowingOutput(t *testing.T) {
	if *playerBrowsers == "" {
		t.Skip("requires -player-browsers and Playwright (go run github.com/playwright-community/playwright-go/cmd/playwright install firefox webkit)")
	}
	film := playerFilm(t)
	ip := playerLANAddress(t)
	pages := httptest.NewServer(http.FileServer(http.Dir(filepath.Join("..", "..", "web", "player"))))
	defer pages.Close()
	pw, err := playwright.Run(&playwright.RunOptions{SkipInstallBrowsers: true})
	must(t, err)
	defer pw.Stop()
	for _, name := range strings.Split(*playerBrowsers, ",") {
		t.Run(name, func(t *testing.T) {
			kind, options := pw.Chromium, playwright.BrowserTypeLaunchOptions{}
			switch name {
			case "chrome": // Playwright's own Chromium has no H.264 or AAC
				options.Channel = playwright.String("chrome")
			case "firefox":
				kind = pw.Firefox
			case "webkit":
				kind = pw.WebKit
			default:
				t.Fatalf("unknown browser %q", name)
			}
			browser, err := kind.Launch(options)
			must(t, err)
			defer browser.Close()
			m := newPlayerMachine(t, ip)
			open := func(link string) playwright.Page {
				page, err := browser.NewPage()
				must(t, err)
				t.Cleanup(func() { page.Close() })
				must(t, page.AddInitScript(playwright.Script{Content: playwright.String(playerRecorder)}))
				_, err = page.Goto(pages.URL + "/index.html#" + link)
				must(t, err)
				return page
			}

			t.Run("live", func(t *testing.T) {
				m.machine.Append(7, "video", -1, film[0], 500_000)
				page := open(m.link(nil, "video"))
				playerWait(t, page, "segment 1 plays", `playing(0.5, 10)`)
				_, err := page.Reload() // a reload mid-run follows from the start again
				must(t, err)
				playerWait(t, page, "segment 1 plays after a reload", `playing(0.5, 10)`)
				m.machine.Append(7, "video", -1, film[1], 500_000)
				playerWait(t, page, "segment 2 plays", `playing(1.0, 22)`)
				// The connection is lost mid-run. The browser's PeerConnection fails (the machine
				// refuses the dead session's ufrag), and the page follows again from its cursor.
				cursor := fmt.Sprint(playerEval(t, page, `[player().seq, player().pos]`))
				m.relay.cut()
				playerWait(t, page, "the PeerConnection fails and the page reconnects", `pcStates.includes("failed") && player().stats.connects === 2`)
				m.machine.Append(7, "video", -1, film[2], 500_000)
				m.machine.End(7, "completed")
				playerWait(t, page, "the finished film ends", `video().ended && decoded(1.5, 36)`)
				resumed := fmt.Sprint(playerEval(t, page, `(f => [f.after, f.offset])(sent.filter(m => m.t === "follow").at(-1))`))
				received := fmt.Sprint(playerEval(t, page, `player().stats.bytes`))
				if resumed != cursor || received != fmt.Sprint(len(bytes.Join(film, nil))) {
					t.Fatalf("held (seq, offset) %s, resumed at %s, received %s bytes of %d: a gap or a duplicate", cursor, resumed, received, len(bytes.Join(film, nil)))
				}
			})

			t.Run("seek", func(t *testing.T) {
				if name == "webkit" {
					// Recorded 2026-09-28, Playwright WebKit 26.0 (GStreamer MSE, not Safari's): the get lands
					// the segment, buffered [1.08, 1.58], but the seek never leaves readyState 1.
					t.Skip("WebKitGTK's MSE never completes a seek into a fetched segment; Safari is proven on a rental")
				}
				for _, segment := range film {
					m.machine.Append(8, "video", -1, segment, 500_000)
				}
				m.machine.End(8, "completed")
				// A small window and lookahead: the film's end is not held when the seek lands.
				page := open(m.link(func(g *capability.Grant) { g.Run = "8" }, "video"))
				result := playerEval(t, page, `(async () => {
					const {play, parseLink} = await import("./cozy-webrtc.js");
					player().close();
					const v = video();
					v.autoplay = false;
					window.cozyPlayer = play(v, parseLink(location.hash), {window: 65536, ahead: 0.3});
					await until(() => player().entries.length === 3 && v.readyState >= 1);
					v.currentTime = 1.25;
					v.play().catch(() => {});
					await until(() => !v.seeking && v.currentTime > 1.3);
					return {gets: player().stats.gets, skipped: !ranges().some(([s, e]) => s < 1 && e > 0.5), buffered: ranges()};
				})()`)
				// The last segment came by get, and the one before it was never fetched.
				if r := result.(map[string]any); fmt.Sprint(r["gets"]) != "1" || r["skipped"] != true {
					t.Fatalf("the seek was not served by one get: %v", r)
				}
				playerWait(t, page, "the seeked film ends", `video().ended`)
			})

			for _, arm := range []struct{ name, link, says string }{
				{"wrong fingerprint", m.linkWith(func(q map[string]string) { q["f"] = strings.Repeat("ab", 32) }), "did not prove its identity"},
				{"expired", m.link(func(g *capability.Grant) { g.Expires = time.Now().Add(-time.Minute).Unix() }, "video"), "expired"},
				{"another output", m.link(func(g *capability.Grant) { g.Outputs = []string{"video"} }, "references"), "does not grant this output"},
			} {
				t.Run(arm.name, func(t *testing.T) {
					page := open(arm.link)
					playerWait(t, page, "the page says "+arm.says, fmt.Sprintf(`document.getElementById("status").textContent.includes(%q)`, arm.says))
				})
			}
		})
	}
}

// playerMachine is a machine's real WebRTC listener serving files, reached through a relay
// the test can cut, and the key that signs its links.
type playerMachine struct {
	t       *testing.T
	machine *webrtctest.Machine
	server  webrtctest.Server
	relay   *playerRelay
	key     ed25519.PrivateKey
}

func newPlayerMachine(t *testing.T, ip string) *playerMachine {
	public, key, err := ed25519.GenerateKey(rand.Reader)
	must(t, err)
	m := webrtctest.NewMachine(t.TempDir(), public)
	server := webrtctest.Serve(t, ip, m)
	return &playerMachine{t: t, machine: m, server: server, relay: newPlayerRelay(t, ip, server.Addr.String()), key: key}
}

// link is the fragment `cozy run play` prints: run 7's output, unless edit says otherwise.
func (m *playerMachine) link(edit func(*capability.Grant), output string) string {
	g := capability.Grant{Machine: m.server.Machine, Run: "7", Expires: time.Now().Add(time.Hour).Unix()}
	if edit != nil {
		edit(&g)
	}
	token, err := capability.Mint(m.key, g)
	must(m.t, err)
	pin := strings.ToLower(strings.ReplaceAll(strings.TrimPrefix(m.server.Fingerprint, "sha-256 "), ":", ""))
	return fmt.Sprintf("v=1&a=%s&f=%s&c=%s&r=%s&o=%s", m.relay.addr, pin, token, g.Run, output)
}

func (m *playerMachine) linkWith(edit func(map[string]string)) string {
	q := map[string]string{}
	for _, pair := range strings.Split(m.link(nil, "video"), "&") {
		k, v, _ := strings.Cut(pair, "=")
		q[k] = v
	}
	edit(q)
	return fmt.Sprintf("v=1&a=%s&f=%s&c=%s&r=%s&o=%s", q["a"], q["f"], q["c"], q["r"], q["o"])
}

// playerRelay forwards TCP to the listener; cut drops every connection, as a lost network does.
type playerRelay struct {
	addr  string
	mu    sync.Mutex
	conns []net.Conn
}

func newPlayerRelay(t *testing.T, ip, target string) *playerRelay {
	ln, err := net.Listen("tcp4", net.JoinHostPort(ip, "0")) //cozy:allow a test path the browser reaches on a non-loopback address
	must(t, err)
	t.Cleanup(func() { ln.Close() })
	r := &playerRelay{addr: ln.Addr().String()}
	go func() {
		for {
			in, err := ln.Accept()
			if err != nil {
				return
			}
			out, err := net.Dial("tcp", target)
			if err != nil {
				in.Close()
				continue
			}
			r.mu.Lock()
			r.conns = append(r.conns, in, out)
			r.mu.Unlock()
			go func() { io.Copy(out, in); out.Close() }()
			go func() { io.Copy(in, out); in.Close() }()
		}
	}()
	return r
}

func (r *playerRelay) cut() {
	r.mu.Lock()
	defer r.mu.Unlock()
	for _, c := range r.conns {
		c.Close()
	}
	r.conns = nil
}

// playerFilm is a real 1.5 s film, H.264 High and AAC, fragmented as the Runtime joins H3:
// the init with the first 12-frame fragment, then one fragment per segment.
func playerFilm(t *testing.T) [][]byte {
	path := filepath.Join(t.TempDir(), "film.mp4")
	if out, err := exec.Command("/usr/bin/nice", "-n", "19", "ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc2=size=640x360:rate=24",
		"-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000", "-t", "1.5", "-vf", "noise=alls=60:allf=t",
		"-c:v", "libx264", "-profile:v", "high", "-pix_fmt", "yuv420p", "-g", "12", "-sc_threshold", "0", "-c:a", "aac", "-ac", "2",
		"-movflags", "+frag_keyframe+empty_moov+default_base_moof+skip_trailer", path).CombinedOutput(); err != nil {
		t.Fatalf("ffmpeg: %v %s", err, out)
	}
	film, err := os.ReadFile(path)
	must(t, err)
	header, pieces := fmp4Fragments(film)
	if len(pieces) != 3 {
		t.Fatalf("the film has %d fragments, not 3", len(pieces))
	}
	return [][]byte{append(header, pieces[0]...), pieces[1], pieces[2]}
}

// playerLANAddress is this computer's first non-loopback IPv4 address: Chrome gathers no candidate
// on loopback, so it pairs with nothing there.
func playerLANAddress(t *testing.T) string {
	addrs, err := net.InterfaceAddrs()
	must(t, err)
	for _, a := range addrs {
		if ip, ok := a.(*net.IPNet); ok && !ip.IP.IsLoopback() && ip.IP.To4() != nil {
			return ip.IP.String()
		}
	}
	t.Skip("no non-loopback IPv4 address for the browser to reach")
	return ""
}

// playerRecorder runs before the page: it keeps every PeerConnection state and every message
// the page sends, for the test to read.
const playerRecorder = `window.pcStates = []; window.sent = [];
const PC = RTCPeerConnection;
window.RTCPeerConnection = function (...args) {
  const pc = new PC(...args);
  pc.addEventListener("connectionstatechange", () => pcStates.push(pc.connectionState));
  return pc;
};
RTCPeerConnection.prototype = PC.prototype;
const send = RTCDataChannel.prototype.send;
RTCDataChannel.prototype.send = function (m) { if (typeof m === "string") sent.push(JSON.parse(m)); return send.call(this, m); };`

// The page helpers every wait and evaluation may use.
const playerHelpers = `const video = () => document.getElementById("video"), player = () => window.cozyPlayer;
const frames = () => video().getVideoPlaybackQuality().totalVideoFrames;
const ranges = () => Array.from({length: video().buffered.length}, (_, i) => [video().buffered.start(i), video().buffered.end(i)]);
// Headless WebKit on Linux counts about half the frames it shows, so there the playhead is the evidence.
const decoded = (end, n) => navigator.vendor.startsWith("Apple") ? video().currentTime >= end - 0.1 : frames() >= n;
const playing = (end, n) => ranges().some(([, e]) => e >= end - 0.05) && decoded(end, n) && !player().error;
const until = f => new Promise((resolve, reject) => { const start = Date.now(), t = setInterval(() => {
  if (f()) { clearInterval(t); resolve(); } else if (Date.now() - start > 60000) { clearInterval(t); reject(new Error("timed out: " + f + " " + JSON.stringify({rs: video().readyState,
    buffered: ranges(), t: video().currentTime, entries: player().entries.length, pos: player().pos, stats: player().stats, error: player().error?.message}))); } }, 50); });`

func playerWait(t *testing.T, page playwright.Page, what, condition string) {
	t.Helper()
	if _, err := page.WaitForFunction("() => { "+playerHelpers+" return "+condition+"; }", nil,
		playwright.PageWaitForFunctionOptions{Polling: 100.0, Timeout: playwright.Float(60000)}); err != nil {
		state, _ := page.Evaluate("() => { " + playerHelpers + ` return {status: document.getElementById("status").textContent, t: video().currentTime,
			frames: frames(), buffered: ranges(), ended: video().ended, rs: video().readyState, paused: video().paused, pc: window.pcStates, stats: player() && player().stats, error: player()?.error?.message}; }`)
		t.Fatalf("waiting for %s: %v; the page: %v", what, err, state)
	}
}

func playerEval(t *testing.T, page playwright.Page, expression string) any {
	t.Helper()
	result, err := page.Evaluate("async () => { " + playerHelpers + " return await " + expression + "; }")
	must(t, err)
	return result
}
