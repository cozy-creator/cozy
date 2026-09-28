package producttest

import (
	"crypto/ed25519"
	"crypto/rand"
	"encoding/binary"
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
	"github.com/cozy-creator/cozy/internal/host/webrtc/webrtctest"
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
	film := filmSegments(t)
	ip := lanAddress(t)
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
				_, err = page.Goto(pages.URL + "/index.html#" + link)
				must(t, err)
				return page
			}

			t.Run("live", func(t *testing.T) {
				m.machine.Append("7", "video", -1, []byte(film[0]), 500_000, false)
				page := open(m.link(nil, "video"))
				waitPage(t, page, "segment 1 plays", `playing(0.5, 10)`)
				_, err := page.Reload() // a reload mid-run follows from the start again
				must(t, err)
				waitPage(t, page, "segment 1 plays after a reload", `playing(0.5, 10)`)
				m.machine.Append("7", "video", -1, []byte(film[1]), 500_000, false)
				waitPage(t, page, "segment 2 plays", `playing(1.0, 22)`)
				m.relay.cut() // the connection is lost mid-run: the page resumes at its cursor
				m.machine.Append("7", "video", -1, []byte(film[2]), 500_000, true)
				m.machine.End("7", "completed")
				waitPage(t, page, "the finished film ends", `video().ended && frames() >= 36 && player().stats.connects === 2`)
				if got := evaluate(t, page, `player().stats.bytes`); int(got.(float64)) != len(strings.Join(film, "")) {
					t.Fatalf("the page received %v bytes for a %d-byte film: a resume repeated or skipped bytes", got, len(strings.Join(film, "")))
				}
			})

			t.Run("seek", func(t *testing.T) {
				for k, segment := range film {
					m.machine.Append("8", "video", -1, []byte(segment), 500_000, k == 2)
				}
				m.machine.End("8", "completed")
				// A small window and lookahead: the film's end is not held when the seek lands.
				page := open(m.link(func(g *capability.Grant) { g.Run = "8" }, "video"))
				result := evaluate(t, page, `(async () => {
					const {play, parseLink} = await import("./cozy-webrtc.js");
					player().close();
					const v = video();
					v.autoplay = false;
					window.cozyPlayer = play(v, parseLink(location.hash), {window: 65536, ahead: 0.1});
					await until(() => player().entries.length === 3 && v.readyState >= 2);
					v.currentTime = 1.25;
					await until(() => !v.seeking && v.readyState >= 2);
					return {gets: player().stats.gets, t: v.currentTime, buffered: ranges()};
				})()`)
				if r := result.(map[string]any); r["gets"] != 1.0 {
					t.Fatalf("the seek was not served by one get: %v", r)
				}
				evaluate(t, page, `video().play()`)
				waitPage(t, page, "the seeked film ends", `video().ended`)
			})

			for _, arm := range []struct{ name, link, says string }{
				{"wrong fingerprint", m.linkWith(func(q map[string]string) { q["f"] = strings.Repeat("ab", 32) }), "did not prove its identity"},
				{"expired", m.link(func(g *capability.Grant) { g.Expires = time.Now().Add(-time.Minute).Unix() }, "video"), "expired"},
				{"another output", m.link(func(g *capability.Grant) { g.Outputs = []string{"video"} }, "references"), "does not grant this output"},
			} {
				t.Run(arm.name, func(t *testing.T) {
					page := open(arm.link)
					waitPage(t, page, "the page says "+arm.says, fmt.Sprintf(`document.getElementById("status").textContent.includes(%q)`, arm.says))
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
	relay   *relay
	key     ed25519.PrivateKey
}

func newPlayerMachine(t *testing.T, ip string) *playerMachine {
	public, key, err := ed25519.GenerateKey(rand.Reader)
	must(t, err)
	m := webrtctest.NewMachine(t.TempDir(), public)
	server := webrtctest.Serve(t, ip, m)
	return &playerMachine{t: t, machine: m, server: server, relay: newRelay(t, ip, server.Addr.String()), key: key}
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

// relay forwards TCP to the listener; cut drops every connection, as a lost network does.
type relay struct {
	addr  string
	mu    sync.Mutex
	conns []net.Conn
}

func newRelay(t *testing.T, ip, target string) *relay {
	ln, err := net.Listen("tcp4", net.JoinHostPort(ip, "0"))
	must(t, err)
	t.Cleanup(func() { ln.Close() })
	r := &relay{addr: ln.Addr().String()}
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

func (r *relay) cut() {
	r.mu.Lock()
	defer r.mu.Unlock()
	for _, c := range r.conns {
		c.Close()
	}
	r.conns = nil
}

// filmSegments is a real 1.5 s film, H.264 High and AAC, fragmented as the Runtime joins H3:
// the init with the first 12-frame fragment, then one fragment per segment.
func filmSegments(t *testing.T) []string {
	path := filepath.Join(t.TempDir(), "film.mp4")
	if out, err := exec.Command("/usr/bin/nice", "-n", "19", "ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc2=size=640x360:rate=24",
		"-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000", "-t", "1.5", "-vf", "noise=alls=60:allf=t",
		"-c:v", "libx264", "-profile:v", "high", "-pix_fmt", "yuv420p", "-g", "12", "-sc_threshold", "0", "-c:a", "aac", "-ac", "2",
		"-movflags", "+frag_keyframe+empty_moov+default_base_moof+skip_trailer", path).CombinedOutput(); err != nil {
		t.Fatalf("ffmpeg: %v %s", err, out)
	}
	data, err := os.ReadFile(path)
	must(t, err)
	var cuts []int
	for p := 0; p+8 <= len(data); p += int(binary.BigEndian.Uint32(data[p:])) {
		if string(data[p+4:p+8]) == "moof" {
			cuts = append(cuts, p)
		}
	}
	if len(cuts) != 3 {
		t.Fatalf("the film has %d fragments, not 3", len(cuts))
	}
	return []string{string(data[:cuts[1]]), string(data[cuts[1]:cuts[2]]), string(data[cuts[2]:])}
}

// lanAddress is this computer's first non-loopback IPv4 address: Chrome gathers no candidate
// on loopback, so it pairs with nothing there.
func lanAddress(t *testing.T) string {
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

// The page helpers every wait and evaluation may use.
const pageHelpers = `const video = () => document.getElementById("video"), player = () => window.cozyPlayer;
const frames = () => video().getVideoPlaybackQuality().totalVideoFrames;
const ranges = () => Array.from({length: video().buffered.length}, (_, i) => [video().buffered.start(i), video().buffered.end(i)]);
const playing = (end, n) => ranges().some(([, e]) => e >= end - 0.05) && frames() >= n && !player().error;
const until = f => new Promise((resolve, reject) => { const start = Date.now(), t = setInterval(() => {
  if (f()) { clearInterval(t); resolve(); } else if (Date.now() - start > 60000) { clearInterval(t); reject(new Error("timed out: " + f)); } }, 50); });`

func waitPage(t *testing.T, page playwright.Page, what, condition string) {
	t.Helper()
	if _, err := page.WaitForFunction("() => { "+pageHelpers+" return "+condition+"; }", nil,
		playwright.PageWaitForFunctionOptions{Polling: 100.0, Timeout: playwright.Float(60000)}); err != nil {
		state, _ := page.Evaluate("() => { " + pageHelpers + ` return {status: document.getElementById("status").textContent, t: video().currentTime,
			frames: frames(), buffered: ranges(), ended: video().ended, stats: player() && player().stats, error: player()?.error?.message}; }`)
		t.Fatalf("waiting for %s: %v; the page: %v", what, err, state)
	}
}

func evaluate(t *testing.T, page playwright.Page, expression string) any {
	t.Helper()
	result, err := page.Evaluate("async () => { " + pageHelpers + " return await " + expression + "; }")
	must(t, err)
	return result
}
