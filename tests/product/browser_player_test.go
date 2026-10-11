package producttest

import (
	"bytes"
	"crypto/ed25519"
	"encoding/binary"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/cozy-creator/cozy/internal/capability"
	localapi "github.com/cozy-creator/cozy/internal/client"
	"github.com/cozy-creator/cozy/internal/config"
	"github.com/cozy-creator/cozy/internal/output"
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
	completed, largeLive := playerMP4Variants(t, film)
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
				options.Args = []string{"--disable-gpu"}
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
			open := func(t *testing.T, link string) playwright.Page {
				t.Helper()
				page, err := browser.NewPage()
				must(t, err)
				t.Cleanup(func() { page.Close() })
				must(t, page.AddInitScript(playwright.Script{Content: playwright.String(playerRecorder)}))
				_, err = page.Goto(pages.URL + "/index.html#" + link)
				must(t, err)
				return page
			}

			t.Run("live", func(t *testing.T) {
				m.machine.Append(7, film[0], 500_000)
				page := open(t, m.link(nil, "video"))
				playerWait(t, page, "segment 1 plays", `playing(0.5, 10)`)
				_, err := page.Reload() // a reload mid-run follows from the start again
				must(t, err)
				playerWait(t, page, "segment 1 plays after a reload", `playing(0.5, 10)`)
				m.machine.Append(7, film[1], 500_000)
				playerWait(t, page, "segment 2 plays", `playing(1.0, 22)`)
				// The connection is lost mid-run. The browser's PeerConnection fails (the machine
				// refuses the dead session's ufrag), and the page follows again from its cursor.
				cursor := fmt.Sprint(playerEval(t, page, `[player().seq, player().pos]`))
				m.relay.cut()
				playerWait(t, page, "the PeerConnection fails and the page reconnects", `pcStates.includes("failed") && player().stats.connects === 2`)
				m.machine.Append(7, film[2], 500_000)
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
					m.machine.Append(8, segment, 500_000)
				}
				m.machine.End(8, "completed")
				// A small window and lookahead: the film's end is not held when the seek lands.
				page := open(t, m.link(func(g *capability.Grant) { g.Run = "8" }, "video"))
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
					// The midpoint avoids counting AAC padding at the first fragment's edge as the second fragment.
					return {gets: player().stats.gets, skipped: !ranges().some(([s, e]) => s <= 0.75 && e > 0.75), buffered: ranges()};
				})()`)
				// The last segment came by get, and the one before it was never fetched.
				if r := result.(map[string]any); fmt.Sprint(r["gets"]) != "1" || r["skipped"] != true {
					t.Fatalf("the seek was not served by one get: %v", r)
				}
				playerWait(t, page, "the seeked film ends", `video().ended`)
			})

			for i, movie := range completed {
				t.Run([]string{"completed end moov", "completed faststart", "completed large header"}[i], func(t *testing.T) {
					run := uint64(20 + i)
					m.machine.Replace(run, movie, 1_500_000)
					m.machine.End(run, "completed")
					page := open(t, m.link(func(g *capability.Grant) { g.Run = fmt.Sprint(run) }, "video"))
					if i == 1 {
						playerEval(t, page, `(() => { globalThis.MediaSource = undefined; globalThis.ManagedMediaSource = undefined; return true; })()`)
					}
					result := playerEval(t, page, `(async () => {
						const {play, parseLink} = await import("./cozy-webrtc.js");
						player().close(); const v = video(); v.autoplay = false;
						window.cozyPlayer = play(v, parseLink(location.hash), {window: 65536});
						await until(() => player().finished && v.readyState >= 1);
						for (const time of [1.2, 0.2, 0.8]) {
							v.currentTime = time;
							await until(() => !v.seeking && Math.abs(v.currentTime - time) < 0.05 && v.readyState >= 2);
						}
						return {type: player().stats.type, bytes: player().stats.bytes, gets: player().stats.gets,
							credited: credit() >= player().stats.bytes,
							boundedCredit: credit() <= player().stats.bytes + 65536, error: player().error?.message};
					})()`)
					r := result.(map[string]any)
					if r["type"] != "video/mp4" || fmt.Sprint(r["bytes"]) != fmt.Sprint(len(movie)) ||
						fmt.Sprint(r["gets"]) != "0" || r["credited"] != true || r["boundedCredit"] != true || r["error"] != nil {
						t.Fatalf("completed MP4 did not finish through its small credit window and seek natively: %v", r)
					}
				})
			}

			t.Run("large live header", func(t *testing.T) {
				m.machine.Begin(24)
				page := open(t, m.link(func(g *capability.Grant) { g.Run = "24" }, "video"))
				playerEval(t, page, `(async () => {
					const {play, parseLink} = await import("./cozy-webrtc.js");
					player().close(); window.cozyPlayer = play(video(), parseLink(location.hash), {window: 65536});
					video().play().catch(() => {});
					return true;
				})()`)
				m.machine.Append(24, largeLive[0], 500_000)
				playerWait(t, page, "large-header live preview plays", `playing(0.5, 10)`)
				for _, fragment := range largeLive[1:] {
					m.machine.Append(24, fragment, 500_000)
				}
				m.machine.End(24, "completed")
				playerWait(t, page, "large-header fragmented film ends", `video().ended && decoded(1.5, 36)`)
				result := playerEval(t, page, `({mse: player().stats.type.includes("codecs="), bytes: player().stats.bytes,
					boundedCredit: credit() <= player().stats.bytes + 65536})`)
				r := result.(map[string]any)
				if r["mse"] != true || r["boundedCredit"] != true || fmt.Sprint(r["bytes"]) != fmt.Sprint(len(bytes.Join(largeLive, nil))) {
					t.Fatalf("large live header stalled, changed playback mode or returned credits twice: %v", r)
				}
			})

			t.Run("completed bad header", func(t *testing.T) {
				m.machine.Replace(25, []byte("unrecognized MP4 header"), 1_500_000)
				m.machine.End(25, "completed")
				page := open(t, m.link(func(g *capability.Grant) { g.Run = "25" }, "video"))
				playerWait(t, page, "the completed invalid file reports a media error", `player().error?.code === "media" && player().error.message.includes("MP4 header")`)
			})

			t.Run("live becomes completed file", func(t *testing.T) {
				m.machine.Append(23, film[0], 500_000)
				page := open(t, m.link(func(g *capability.Grant) { g.Run = "23" }, "video"))
				playerWait(t, page, "the live preview plays", `playing(0.5, 10)`)
				old := playerEval(t, page, `(async () => {
					video().pause(); video().currentTime = 0.25;
					await until(() => !video().seeking);
					return video().src;
				})()`)
				m.machine.Replace(23, completed[0], 1_500_000)
				m.machine.End(23, "completed")
				playerWait(t, page, "the finalized file replaces the live stream", `player().finished && player().stats.resets === 1 && player().stats.type === "video/mp4" && video().readyState >= 2`)
				result := playerEval(t, page, `({positionHeld: Math.abs(video().currentTime - 0.25) < 0.05, paused: video().paused, revoked: window.revokedURLs, error: player().error?.message})`)
				r := result.(map[string]any)
				if r["paused"] != true || r["positionHeld"] != true ||
					!strings.Contains(fmt.Sprint(r["revoked"]), fmt.Sprint(old)) || r["error"] != nil {
					t.Fatalf("finalization lost the playhead/pause or retained the old URL: %v", r)
				}
				closed := playerEval(t, page, `(() => { const url = video().src; player().close(); return {revoked: revokedURLs.includes(url), src: video().getAttribute("src")}; })()`)
				if c := closed.(map[string]any); c["revoked"] != true || c["src"] != nil {
					t.Fatalf("closing the file player retained its object URL: %v", c)
				}
			})

			t.Run("indexed revisions while running", func(t *testing.T) {
				var movies [][]byte
				for count := 1; count <= 2; count++ {
					dir := t.TempDir()
					source, target := filepath.Join(dir, "fragments.mp4"), filepath.Join(dir, "indexed.mp4")
					must(t, os.WriteFile(source, bytes.Join(film[:count], nil), 0600))
					if out, err := exec.Command("ffmpeg", "-v", "error", "-i", source, "-map", "0", "-c", "copy", target).CombinedOutput(); err != nil {
						t.Fatalf("index partial browser fixture: %v %s", err, out)
					}
					movie, err := os.ReadFile(target)
					must(t, err)
					movies = append(movies, movie)
				}
				movies = append(movies, completed[0])
				for i, movie := range movies {
					movies[i] = playerLargeMdat(t, movie, (i+1)*2<<20)
				}
				m.machine.Replace(26, movies[0], 500_000)
				page := open(t, m.link(func(g *capability.Grant) { g.Run = "26" }, "video"))
				playerEval(t, page, `(async () => {
					const {play, parseLink} = await import("./cozy-webrtc.js");
					player().close(); video().autoplay = false;
					window.cozyPlayer = play(video(), parseLink(location.hash), {window: 65536});
					return true;
				})()`)
				playerWait(t, page, "indexed partial plays before run completion", `!player().finished && player().stats.type === "video/mp4" && video().readyState >= 2`)
				old := playerEval(t, page, `(async () => {
					video().pause(); video().currentTime = 0.2;
					await until(() => !video().seeking);
					return video().src;
				})()`)
				m.machine.Replace(26, movies[1], 1_000_000)
				playerWait(t, page, "a longer indexed partial replaces the first", `!player().finished && player().stats.resets === 1 && video().readyState >= 2 && video().duration > 0.9`)
				result := playerEval(t, page, `(async () => {
					const retained = {paused: video().paused, position: video().currentTime, revoked: revokedURLs};
					for (const time of [0.7, 0.1]) {
						video().currentTime = time;
						await until(() => !video().seeking && Math.abs(video().currentTime-time) < 0.05 && video().readyState >= 2);
					}
					video().playbackRate = 0.1;
					await video().play();
					return retained;
				})()`)
				if r := result.(map[string]any); r["paused"] != true ||
					!strings.Contains(fmt.Sprint(r["revoked"]), fmt.Sprint(old)) ||
					fmt.Sprint(r["position"]) != "0.2" {
					t.Fatalf("indexed partial replacement lost paused position or old URL: %v", r)
				}
				m.machine.Replace(26, movies[2], 1_500_000)
				playerWait(t, page, "playing state survives another indexed replacement", `!player().finished && player().stats.resets === 2 && video().readyState >= 2 && video().duration > 1.4 && !video().paused`)
				m.machine.End(26, "completed")
				playerWait(t, page, "the indexed revision settles without another reset", `player().finished && player().stats.resets === 2 && !player().error`)
				playerEval(t, page, `(async () => {
					video().pause(); video().currentTime = 1.2;
					await until(() => !video().seeking && video().readyState >= 2);
					return true;
				})()`)
			})

			for _, arm := range []struct{ name, link, says string }{
				{"wrong fingerprint", m.linkWith(func(q map[string]string) { q["f"] = strings.Repeat("ab", 32) }), "did not prove its identity"},
				{"expired", m.link(func(g *capability.Grant) { g.Expires = time.Now().Add(-time.Minute).Unix() }, "video"), "expired"},
				{"another output", m.link(func(g *capability.Grant) { g.Outputs = []string{"video"} }, "references"), "does not grant this output"},
			} {
				t.Run(arm.name, func(t *testing.T) {
					page := open(t, arm.link)
					playerWait(t, page, "the page says "+arm.says, fmt.Sprintf(`document.getElementById("status").textContent.includes(%q)`, arm.says))
				})
			}
		})
	}
}

// Keep real sample bytes and their offsets unchanged, but move the trailing moov
// beyond many credit windows. Unreferenced trailing bytes inside mdat are legal.
func playerLargeMdat(t *testing.T, movie []byte, padding int) []byte {
	t.Helper()
	for at := 0; at+8 <= len(movie); {
		size := int(binary.BigEndian.Uint32(movie[at:]))
		if size < 8 || at+size > len(movie) {
			t.Fatal("invalid indexed fixture box")
		}
		if kind := string(movie[at+4 : at+8]); kind == "moov" {
			t.Fatal("fixture must put moov after mdat")
		} else if kind == "mdat" {
			out := make([]byte, len(movie)+padding)
			copy(out, movie[:at+size])
			copy(out[at+size+padding:], movie[at+size:])
			binary.BigEndian.PutUint32(out[at:], uint32(size+padding))
			return out
		}
		at += size
	}
	t.Fatal("indexed fixture has no mdat")
	return nil
}

// playerMachine is the Rust machine's WebRTC listener, reached on this computer's LAN address
// through a relay the test can cut.
type playerMachine struct {
	t       *testing.T
	machine *mediaMachine
	relay   *playerRelay
}

func newPlayerMachine(t *testing.T, ip string) *playerMachine {
	m := newMediaMachine(t)
	return &playerMachine{t: t, machine: m, relay: newPlayerRelay(t, ip, m.Addr.String())}
}

// link is the fragment `cozy run play` prints: scripted run 7's output, unless edit says otherwise.
func (m *playerMachine) link(edit func(*capability.Grant), output string) string {
	g := capability.Grant{Run: "7", Expires: time.Now().Add(time.Hour).Unix()}
	if edit != nil {
		edit(&g)
	}
	if n, err := strconv.ParseUint(g.Run, 10, 64); err == nil {
		g.Run = m.machine.Begin(n).id
	}
	token := m.machine.mint(g)
	pin := strings.ToLower(strings.ReplaceAll(strings.TrimPrefix(m.machine.Fingerprint, "sha-256 "), ":", ""))
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

func playerMP4Variants(t *testing.T, fragments [][]byte) ([][]byte, [][]byte) {
	t.Helper()
	directory := t.TempDir()
	source := filepath.Join(directory, "fragments.mp4")
	must(t, os.WriteFile(source, bytes.Join(fragments, nil), 0600))
	metadata := filepath.Join(directory, "metadata.txt")
	must(t, os.WriteFile(metadata, []byte(";FFMETADATA1\ncomment="+strings.Repeat("header metadata ", 8192)+"\n"), 0600))
	var movies [][]byte
	var live [][]byte
	for variant := range 4 {
		target := filepath.Join(directory, fmt.Sprintf("variant-%d.mp4", variant))
		args := []string{"-v", "error", "-nostdin", "-i", source}
		if variant >= 2 {
			args = append(args, "-f", "ffmetadata", "-i", metadata, "-map_metadata", "1")
		}
		args = append(args, "-map", "0", "-c", "copy")
		if variant == 1 || variant == 2 {
			args = append(args, "-movflags", "+faststart")
		} else if variant == 3 {
			args = append(args, "-movflags", "+frag_keyframe+empty_moov+default_base_moof+skip_trailer")
		}
		if out, err := exec.Command("ffmpeg", append(args, target)...).CombinedOutput(); err != nil {
			t.Fatalf("finalize browser fixture: %v %s", err, out)
		}
		movie, err := os.ReadFile(target)
		must(t, err)
		if len(movie) <= 65536 {
			t.Fatal("completed MP4 must exceed the browser test's credit window")
		}
		if variant >= 2 {
			moov := 0
			for at := 0; at+8 <= len(movie); {
				size := int(binary.BigEndian.Uint32(movie[at:]))
				if size < 8 || at+size > len(movie) {
					t.Fatal("invalid MP4 fixture box")
				}
				if string(movie[at+4:at+8]) == "moov" {
					moov = size
				}
				at += size
			}
			if moov <= 65536 {
				t.Fatal("metadata must make the valid moov exceed the credit window")
			}
		}
		if variant == 3 {
			header, pieces := fmp4Fragments(movie)
			live = append([][]byte{append(header, pieces[0]...)}, pieces[1:]...)
		} else {
			movies = append(movies, movie)
		}
	}
	return movies, live
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
window.revokedURLs = [];
const revoke = URL.revokeObjectURL.bind(URL);
URL.revokeObjectURL = function (url) { revokedURLs.push(url); revoke(url); };
const PC = RTCPeerConnection;
window.RTCPeerConnection = function (...args) {
  const pc = new PC(...args);
  pc.addEventListener("connectionstatechange", () => pcStates.push(pc.connectionState));
  return pc;
};
RTCPeerConnection.prototype = PC.prototype;
const send = RTCDataChannel.prototype.send;
const channels = new WeakMap(); let nextChannel = 0;
RTCDataChannel.prototype.send = function (m) {
  if (!channels.has(this)) channels.set(this, ++nextChannel);
  if (typeof m === "string") sent.push({...JSON.parse(m), channel: channels.get(this)});
  return send.call(this, m);
};`

// The page helpers every wait and evaluation may use.
const playerHelpers = `const video = () => document.getElementById("video"), player = () => window.cozyPlayer;
const frames = () => video().getVideoPlaybackQuality().totalVideoFrames;
const ranges = () => Array.from({length: video().buffered.length}, (_, i) => [video().buffered.start(i), video().buffered.end(i)]);
const credit = () => { const channel = sent.filter(m => m.t === "follow").at(-1)?.channel;
  return sent.filter(m => m.t === "credit" && m.channel === channel).at(-1)?.bytes ?? 0; };
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

// `cozy run play` prints a link from the run's acceptance, the Hub's rental view and this home's
// The ordinary CLI prints a direct link using the actual pinned machine and owner key.
// It does not depend on Hub availability or a personal daemon upgrade. The browser follows
// that exact printed link to the completed synthetic CPU film.
func TestRunPlayPrintsALinkThatPlays(t *testing.T) {
	if *playerBrowsers == "" {
		t.Skip("requires -player-browsers=chrome")
	}
	m := newMediaMachine(t)
	pages := httptest.NewServer(http.FileServer(http.Dir(filepath.Join("..", "..", "web", "player"))))
	defer pages.Close()
	cfg := filepath.Join(m.root, config.FileName)
	raw, err := os.ReadFile(cfg)
	must(t, err)
	must(t, os.WriteFile(cfg, append(raw, []byte("tensorhub_url: http://127.0.0.1:1\nplayer_url: "+pages.URL+"/index.html\n")...), 0o600))
	for _, segment := range playerFilm(t) {
		m.Append(7, segment, 500_000)
	}
	m.End(7, "completed")
	run := m.Begin(7).id
	code, out := runCozy(t, m.root, "run", "play", run, "--json")
	var printed struct{ Link string }
	if code != 0 || json.Unmarshal([]byte(lastJSONLine(out)), &printed) != nil {
		t.Fatalf("cozy run play: [%d] %s", code, out)
	}
	parsed, err := url.Parse(printed.Link)
	must(t, err)
	link, err := url.ParseQuery(parsed.Fragment)
	must(t, err)
	grant, err := capability.Verify(link.Get("c"), m.Machine, []ed25519.PublicKey{m.public}, time.Now(), "")
	must(t, err)
	if parsed.Scheme+"://"+parsed.Host+parsed.Path != pages.URL+"/index.html" || link.Get("r") != run || grant.Run != run || grant.Machine != m.Machine || len(grant.Outputs) != 1 || grant.Outputs[0] != "video" {
		t.Fatalf("the link names run %q; its capability grants %+v", link.Get("r"), grant)
	}
	pw, err := playwright.Run(&playwright.RunOptions{SkipInstallBrowsers: true})
	must(t, err)
	defer pw.Stop()
	browser, err := pw.Chromium.Launch(playwright.BrowserTypeLaunchOptions{Channel: playwright.String("chrome"), Args: []string{"--disable-gpu"}})
	must(t, err)
	defer browser.Close()
	page, err := browser.NewPage()
	must(t, err)
	must(t, page.AddInitScript(playwright.Script{Content: playwright.String(playerRecorder)}))
	_, err = page.Goto(printed.Link)
	must(t, err)
	playerWait(t, page, "the printed link plays the film to its end", `video().ended && decoded(1.5, 36)`)
}

// This computer's machine plays its own run's growing film (the owner's local-playback ruling):
// with no port configured it takes one and keeps it, `cozy run play` prints the link while the
// run is running, and Chrome plays each segment the Runtime encodes as it lands, then the whole
// film, every byte `cozy run --out` saved.
func TestRunPlayPlaysALocalMachinesGrowingFilm(t *testing.T) {
	if *playerBrowsers == "" || *machineHostBinary == "" || *privateScriptRuntimeWheel == "" {
		t.Skip("requires -player-browsers=chrome, -machine-host and -script-runtime-wheel")
	}
	wheel, err := filepath.Abs(*privateScriptRuntimeWheel)
	must(t, err)
	root, err := os.MkdirTemp(os.TempDir(), "czp")
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
	pages := httptest.NewServer(http.FileServer(http.Dir(filepath.Join("..", "..", "web", "player"))))
	defer pages.Close()
	must(t, os.WriteFile(filepath.Join(root, config.FileName), []byte("tensorhub_url: http://127.0.0.1:1\nplayer_url: "+pages.URL+"/index.html\n"), 0o600))
	if code, out := runCozy(t, root, "package", "install", outputLogProof(t, wheel)); code != 0 {
		t.Fatalf("package install [exit %d]\n%s", code, out)
	}
	gate, out := t.TempDir(), filepath.Join(root, "film")
	run, command, ran := cozy1Run(t, root, "local/output-log-proof/film", "gate="+gate, "--await", "--json", "--out", out)
	must(t, os.WriteFile(filepath.Join(gate, "go-1"), nil, 0o600))
	code, printed := runCozy(t, root, "run", "play", run, "--json")
	var play struct{ Link string }
	if code != 0 || json.Unmarshal([]byte(lastJSONLine(printed)), &play) != nil {
		t.Fatalf("cozy run play: [%d] %s", code, printed)
	}
	parsed, err := url.Parse(play.Link)
	must(t, err)
	link, err := url.ParseQuery(parsed.Fragment)
	must(t, err)
	port, _ := os.ReadFile(filepath.Join(root, "machine", "root", "var", "lib", "cozy", "machine", "webrtc-port"))
	if _, at, _ := net.SplitHostPort(link.Get("a")); len(port) == 0 || at != strings.TrimSpace(string(port)) || link.Get("r") != run {
		t.Fatalf("the link plays run %q at %q; the machine keeps port %q", link.Get("r"), link.Get("a"), port)
	}
	pw, err := playwright.Run(&playwright.RunOptions{SkipInstallBrowsers: true})
	must(t, err)
	defer pw.Stop()
	browser, err := pw.Chromium.Launch(playwright.BrowserTypeLaunchOptions{Channel: playwright.String("chrome"), Args: []string{"--disable-gpu"}})
	must(t, err)
	defer browser.Close()
	page, err := browser.NewPage()
	must(t, err)
	must(t, page.AddInitScript(playwright.Script{Content: playwright.String(playerRecorder)}))
	_, err = page.Goto(play.Link)
	must(t, err)
	// Each segment is shown as it lands, whether its revision extends the video or replaces it.
	shown := func(seconds float64) string {
		return fmt.Sprintf(`!player().error && video().currentTime >= %v && ranges().some(([, e]) => e >= %v)`, seconds-0.1, seconds-0.05)
	}
	playerWait(t, page, "segment 1 plays while the run runs", shown(0.5))
	// The film runs out at its live edge, says so, and plays on as the next segment lands.
	playerWait(t, page, "segment 1 runs out at the live edge", `video().ended && document.getElementById("status").textContent.includes("Waiting")`)
	must(t, os.WriteFile(filepath.Join(gate, "go-2"), nil, 0o600))
	playerWait(t, page, "segment 2 plays as it lands", shown(1.0))
	must(t, os.WriteFile(filepath.Join(gate, "go-3"), nil, 0o600))
	playerWait(t, page, "the finished film plays to its end", "player().finished && "+shown(1.5))
	if err := command.Wait(); err != nil {
		t.Fatalf("the job exited %v:\n%s", err, ran.String())
	}
	saved := savedVideo(t, out, ran)
	if held := fmt.Sprint(playerEval(t, page, `[player().pos, player().final.sha256]`)); held != fmt.Sprint([]any{len(saved), digestOf(saved)}) {
		t.Fatalf("the page holds %s; --out saved %d bytes, %s", held, len(saved), digestOf(saved))
	}
}

// A rented run's growing video says, once, how to watch it in a browser; a local run's does not.
func TestAwaitNamesThePlayCommandForARentedVideo(t *testing.T) {
	added := func(output string) localapi.Event {
		return localapi.Event{Type: "output_item.added", RequestID: "job-film", Payload: map[string]any{"output_index": 0,
			"item": map[string]any{"id": "12/" + output, "type": "video", "name": output, "media_type": "video/mp4",
				"status": "in_progress", "path": "/out/12-" + output + ".mp4"}}}
	}
	delta := func(output string, rev float64) localapi.Event {
		return localapi.Event{Type: "output_item.delta", RequestID: "job-film", Payload: map[string]any{"item_id": "12/" + output,
			"output_index": 0, "rev": rev, "length": 1000 * rev}}
	}
	for _, rented := range []bool{true, false} {
		p, buf := progressSink(output.Mode{Human: true}, false)
		if rented {
			p.On(localapi.Event{Type: "request.rentals", RequestID: "job-film", Payload: map[string]any{"line": "renting H100 on jaguarman"}})
		}
		p.On(added("video"))
		p.On(delta("video", 1))
		p.On(delta("video", 2))
		p.On(added("preview"))
		p.On(delta("preview", 1))
		p.Done()
		got := buf.String()
		plays := strings.Count(got, "play in a browser: cozy run play job-film")
		if rented && (plays != 2 || !strings.Contains(got, "cozy run play job-film --output preview")) || !rented && plays != 0 {
			t.Fatalf("rented=%t:\n%s", rented, got)
		}
	}
}
